import math
from typing import Optional

import torch
from torch import nn

from tfmplayground.attn_v2 import PFNAttentionConfig, MultiHeadAttention as PFNMultiHeadAttentionV2


class _PFNMultiHeadAttentionV2Safe(PFNMultiHeadAttentionV2):
    """MZ: All further updates are made to PFNMultiHeadAttentionV2 to make it safe.
       No update on PFNMultiHeadAttentionV2 itself should be made."""

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None, 
        *,
        cache_kv: bool = False,
        add_input: bool = False,
        allow_inplace: bool = False,  # ignored
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_cached_kv: bool = False,
        attn_weight_external: torch.Tensor | None = None,
        external_gate: float | None = None,
    ) -> torch.Tensor:
        assert not (cache_kv and use_cached_kv), "Cannot cache and use cached keys and values at the same time."
        assert not x.requires_grad or (not self.has_cached_kv and not cache_kv), (
            "Saving keys and values is only supported during inference."
        )
        x, x_kv, x_shape_after_transpose = self._rearrange_inputs_to_flat_batch(x, x_kv)

        nhead_kv = 1 if reuse_first_head_kv else self._nhead_kv

        if cache_kv:
            self._k_cache = self._v_cache = self._kv_cache = None
            if x_kv is not None:
                batch_size, seqlen_kv = x_kv.shape[:2]
            else:
                batch_size, seqlen_kv = x.shape[:2]

            if self._w_kv is not None or self._w_qkv is not None:
                self._kv_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    2,
                    1 if only_cache_first_head_kv else nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                self._k_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
                self._v_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_v,
                    device=x.device,
                    dtype=x.dtype,
                )

        q, k, v, kv, qkv = self.compute_qkv(
            x,
            x_kv,
            self._k_cache,
            self._v_cache,
            self._kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
        )
        attention_head_outputs = self.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout_p,
            self.softmax_scale,
            attn_weight_external=attn_weight_external,
            external_gate=external_gate,
        )
        output = torch.einsum("... h d, h d s -> ... s", attention_head_outputs, self._w_out)
        if add_input:
            output = output + x
        if allow_inplace:
            return output
        return output.reshape(x_shape_after_transpose[:-1] + output.shape[-1:])


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
        # MZ: added text_enhanced flag to decide if the encoder layer should be initialized with text enhanced attention
        text_enhanced: bool = False,
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
        self.config = config or PFNAttentionConfig(emsize=embed_dim, nhead=num_heads)
        self.core = _PFNMultiHeadAttentionV2Safe(
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
        external_gate: float | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise NotImplementedError("Masks/causal attention are not implemented for PFNMultiHeadAttentionV2.")

        query_b = self._maybe_transpose(query, self.batch_first)
        key_b = self._maybe_transpose(key, self.batch_first)

        attn_output = self.core(
            query_b,
            x_kv=key_b,
            reuse_first_head_kv=self.reuse_first_head_kv,
            attn_weight_external=attn_weight_external,
            external_gate=external_gate,
        )

        attn_weights = None
        if need_weights:
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
            if qkv is not None:
                q, k, v = qkv.unbind(dim=-3)
            elif kv is not None:
                k, v = kv.unbind(dim=-3)

            if q is None or k is None:
                raise RuntimeError("Unable to compute attention weights because Q/K are missing.")

            logits = torch.einsum("b q h d, b k h d -> b h q k", q, k)
            logits = logits / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(logits, dim=-1)
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

        # attn_output = self._maybe_transpose(attn_output, not self.batch_first)
        attn_output = self._maybe_transpose(attn_output, self.batch_first)
        return attn_output, attn_weights
