import math
from typing import Optional

import torch
from torch import nn

from tfmplayground.attn_v2 import PFNAttentionConfig, MultiHeadAttention as PFNMultiHeadAttentionV2


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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is not value:
            raise ValueError("PFN attention requires key and value to be identical tensors.")
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise NotImplementedError("Masks/causal attention are not implemented for PFNMultiHeadAttentionV2.")

        query_b = self._maybe_transpose(query, self.batch_first)
        key_b = self._maybe_transpose(key, self.batch_first)

        attn_output = self.core(
            query_b,
            x_kv=key_b,
            reuse_first_head_kv=self.reuse_first_head_kv,
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

        attn_output = self._maybe_transpose(attn_output, not self.batch_first)
        return attn_output, attn_weights
