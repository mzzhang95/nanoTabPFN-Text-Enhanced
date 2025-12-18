from typing import Optional

import torch
from torch import nn

from tfmplayground.attn_v2 import PFNAttentionConfig, PFNMultiHeadAttentionV2

def _attention_probs_and_head_output(  # noqa: PLR0913
    *,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    kv: torch.Tensor | None,
    qkv: torch.Tensor | None,
    dropout_p: float | None,
    softmax_scale: float | None,
    training: bool,
    attn_weight_external: torch.Tensor | None,
    external_gate: torch.Tensor | float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (attention_probs, head_output) with optional external blending.

    - attention_probs: [B, Lq, Lk, H]
    - head_output: [B, Lq, H, d_v]
    """
    assert (k is None) == (v is None)
    assert sum([qkv is None, kv is None, k is None and v is None]) == 2
    assert (qkv is None) != (q is None)

    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    assert q is not None
    assert k is not None
    assert v is not None

    # Apply external (text-enhanced) attention only when both tensors are provided.
    # If either is missing, fall back to regular attention.
    if attn_weight_external is None or external_gate is None:
        attn_weight_external = None
        external_gate = None

    batch_size, seqlen_q, nhead, d_k = q.shape
    _, seqlen_kv, nhead_kv, d_v = v.shape
    share_kv_across_n_heads = nhead // nhead_kv

    if dropout_p is None:
        dropout_p = 0.0

    # Broadcast kv heads (GQA/MQA) to full heads for manual attention.
    k = PFNMultiHeadAttentionV2.broadcast_kv_across_heads(k, share_kv_across_n_heads)
    v = PFNMultiHeadAttentionV2.broadcast_kv_across_heads(v, share_kv_across_n_heads)

    logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
    if softmax_scale is None:
        logits *= torch.sqrt(torch.tensor(1.0 / d_k, device=logits.device, dtype=logits.dtype))
    else:
        logits *= softmax_scale

    ps = torch.softmax(logits, dim=2)  # [B, Lq, Lk, H]
    ps = torch.dropout(ps, dropout_p, train=training)

    if attn_weight_external is not None and external_gate is not None:
        gate = external_gate
        if not isinstance(gate, torch.Tensor):
            gate = torch.tensor(gate, device=ps.device, dtype=ps.dtype)
        # Accept per-head gate shapes:
        # - [H] -> [1,1,1,H]
        # - [1,H,1,1] -> [1,1,1,H]
        if gate.dim() == 1 and gate.numel() == nhead:
            gate = gate.view(1, 1, 1, nhead)
        elif gate.dim() == 4 and gate.shape[1] == nhead and gate.shape[2] == 1 and gate.shape[3] == 1:
            gate = gate.permute(0, 2, 3, 1)  # [1,1,1,H]

        ext = attn_weight_external
        # if ext.dim() == 2:
        #     ext = ext.unsqueeze(0)
        # if ext.shape[0] == 1 and batch_size > 1:
        #     ext = ext.expand(batch_size, -1, -1)
        # if ext.shape[:3] != (batch_size, seqlen_q, seqlen_kv):
        #     raise ValueError(
        #         f"attn_weight_external must have shape [B, Lq, Lk] = "
        #         f"[{batch_size}, {seqlen_q}, {seqlen_kv}], got {tuple(ext.shape)}"
        #     )
        ext = ext.unsqueeze(-1).expand(-1, -1, -1, nhead)  # [B, Lq, Lk, H]
        ps = gate * ps + (1.0 - gate) * ext

    head_output = torch.einsum("b q k h, b k h d -> b q h d", ps, v).reshape(
        batch_size, seqlen_q, nhead, d_v
    )
    return ps, head_output


class PFNMultiheadAttentionV2Wrapper(nn.Module):
    """Torch-like MultiheadAttention wrapper backed by PFNMultiHeadAttentionV2."""

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        *,
        config: PFNAttentionConfig | None = None,
        softmax_scale: float | None = None,
        reuse_first_head_kv: bool = False,
        # When False, ignores attn_weight_external even if provided.
        text_enhanced: bool = True,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("PFNMultiHeadAttentionV2 does not support bias terms.")
        if add_bias_kv or add_zero_attn:
            raise ValueError("add_bias_kv/add_zero_attn are not supported by PFN attention.")
        if kdim is not None and kdim != embed_dim:
            raise ValueError("kdim must equal embed_dim for PFNMultiHeadAttentionV2.")
        if vdim is not None and vdim != embed_dim:
            raise ValueError("vdim must equal embed_dim for PFNMultiHeadAttentionV2.")

        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.reuse_first_head_kv = reuse_first_head_kv
        self.text_enhanced = text_enhanced
        self.config = config or PFNAttentionConfig(emsize=embed_dim, nhead=num_heads)
        self.core = PFNMultiHeadAttentionV2(
            d_k=self.head_dim,
            d_v=self.head_dim,
            device=device,
            dtype=dtype,
            config=self.config,
            dropout_p=dropout if dropout > 0 else None,
            softmax_scale=softmax_scale,
        )

    def _maybe_transpose(self, tensor: torch.Tensor, batch_first: bool) -> torch.Tensor:
        return tensor if batch_first else tensor.transpose(0, 1)

    def _forward_attention(  # noqa: PLR0913
        self,
        query_b: torch.Tensor,
        key_b: torch.Tensor,
        *,
        attn_weight_external: torch.Tensor | None,
        external_gate: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, kv, qkv = self.core.compute_qkv(
            query_b,
            key_b,
            None,
            None,
            None,
            cache_kv=False,
            use_cached_kv=False,
            reuse_first_head_kv=self.reuse_first_head_kv,
        )
        probs_bqkh, head_output = _attention_probs_and_head_output(
            q=q,
            k=k,
            v=v,
            kv=kv,
            qkv=qkv,
            dropout_p=self.core.dropout_p,
            softmax_scale=self.core.softmax_scale,
            training=self.training,
            attn_weight_external=attn_weight_external,
            external_gate=external_gate,
        )
        attn_output = torch.einsum("b q h d, h d s -> b q s", head_output, self.core.w_out)
        return attn_output, probs_bqkh

    def forward(  # noqa: PLR0913
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        *,
        attn_weight_external: torch.Tensor | None = None,
        external_gate: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise NotImplementedError("Masks/causal attention are not implemented for PFNMultiHeadAttentionV2.")

        # PFN attention uses x_kv for both key and value. We ignore `value` unless it is actually needed.
        if value.shape != key.shape:
            raise ValueError(f"PFNMultiHeadAttentionV2 requires key/value shapes to match; got key={tuple(key.shape)} value={tuple(value.shape)}")

        query_b = self._maybe_transpose(query, self.batch_first)
        key_b = self._maybe_transpose(key, self.batch_first)

        if not self.text_enhanced:
            attn_weight_external = None
            external_gate = None

        attn_output, probs_bqkh = self._forward_attention(
            query_b,
            key_b,
            attn_weight_external=attn_weight_external,
            external_gate=external_gate,
        )

        if need_weights:
            ps = probs_bqkh.permute(0, 3, 1, 2)  # [B, H, Lq, Lk]
            attn_weights = ps.mean(dim=1) if average_attn_weights else ps
        else:
            attn_weights = None

        attn_output = self._maybe_transpose(attn_output, self.batch_first)
        return attn_output, attn_weights
