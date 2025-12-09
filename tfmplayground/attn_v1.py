import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override


def _gqa_is_supported() -> bool:
    """Check whether torch's scaled_dot_product_attention supports group query attention."""
    torch_version = torch.__version__.split(".")
    if int(torch_version[0]) < 2 or not torch.cuda.is_available():
        return False

    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    has_enable_gqa = torch_major > 2 or (torch_major == 2 and torch_minor >= 5)
    if not has_enable_gqa:
        return False

    device = torch.cuda.current_device()
    nvidia_compute_capability = torch.cuda.get_device_capability(device)
    return nvidia_compute_capability[0] >= 8


USE_TORCH_2_GQA = _gqa_is_supported()


@dataclass
class PFNAttentionConfig:
    """Minimal config needed by PFNMultiHeadAttention."""

    emsize: int
    nhead: int
    attention_init_gain: float = 1.0
    recompute_attn: bool = False


class Attention(nn.Module):
    """Base class for attention layers."""

    @override
    def forward(  # pragma: no cover - interface only
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        *,
        cache_kv: bool = False,
        use_cached_kv: bool = False,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


class PFNMultiHeadAttention(Attention):
    _input_size: int
    _output_size: int
    _nhead: int
    _nhead_kv: int
    _d_k: int
    _d_v: int
    _share_kv_across_n_heads: int
    dropout_p: float | None
    softmax_scale: float | None
    _k_cache: torch.Tensor | None
    _v_cache: torch.Tensor | None
    _kv_cache: torch.Tensor | None
    _w_q: torch.nn.Parameter | None
    _w_k: torch.nn.Parameter | None
    _w_v: torch.nn.Parameter | None
    _w_kv: torch.nn.Parameter | None
    _w_qkv: torch.nn.Parameter | None
    _w_out: torch.nn.Parameter

    @property
    def w_q(self) -> torch.nn.Parameter | None:
        return self._w_q

    @property
    def w_k(self) -> torch.nn.Parameter | None:
        return self._w_k

    @property
    def w_v(self) -> torch.nn.Parameter | None:
        return self._w_v

    @property
    def w_qkv(self) -> torch.nn.Parameter | None:
        return self._w_qkv

    @property
    def w_kv(self) -> torch.nn.Parameter | None:
        return self._w_kv

    @property
    def w_out(self) -> torch.nn.Parameter:
        return self._w_out

    @property
    def has_cached_kv(self) -> bool:
        assert (self._k_cache is None) == (self._v_cache is None)
        assert self._kv_cache is None or (self._k_cache is None and self._v_cache is None)
        return (self._k_cache is not None and self._v_cache is not None) or self._kv_cache is not None

    def empty_kv_cache(self) -> None:
        self._k_cache = None
        self._v_cache = None
        self._kv_cache = None

    def set_parameters(
        self,
        w_out: torch.nn.Parameter,
        w_q: torch.nn.Parameter | None = None,
        w_k: torch.nn.Parameter | None = None,
        w_v: torch.nn.Parameter | None = None,
        w_kv: torch.nn.Parameter | None = None,
        w_qkv: torch.nn.Parameter | None = None,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
    ) -> None:
        assert (precomputed_k is None) == (precomputed_v is None)
        assert (precomputed_kv is None) or (precomputed_k is None)
        assert (precomputed_kv is None and precomputed_k is None) != (
            w_qkv is None and w_kv is None and w_k is None and w_v is None
        )
        assert (w_qkv is None) != (w_q is None)
        assert (w_qkv is None) or (w_kv is None and w_k is None and w_v is None)
        assert w_kv is None or (w_k is None and w_v is None)
        assert (w_k is None) == (w_v is None)

        def assert_tensor_shape(
            tensor: torch.Tensor | None,
            expected_shape: list[int | None],
        ) -> None:
            if tensor is None:
                return
            actual_shape = tensor.size()
            err = f"Tensor {actual_shape=} does not match {expected_shape=}."
            assert len(actual_shape) == len(expected_shape), err
            for actual_dim, expected_dim in zip(actual_shape, expected_shape):
                if expected_dim is not None:
                    assert actual_dim == expected_dim, err

        assert_tensor_shape(precomputed_k, [None, None, self._nhead_kv, self._d_k])
        assert_tensor_shape(precomputed_v, [None, None, self._nhead_kv, self._d_v])
        assert_tensor_shape(precomputed_kv, [None, None, 2, self._nhead_kv, self._d_k])
        assert_tensor_shape(w_q, [1, self._nhead, self._d_k, self._input_size])
        assert_tensor_shape(w_k, [self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_v, [self._nhead_kv, self._d_v, self._input_size])
        assert_tensor_shape(w_kv, [2, self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_qkv, [3, self._nhead, self._d_k, self._input_size])
        assert_tensor_shape(w_out, [self._nhead, self._d_v, self._output_size])

        self.register_parameter("_w_out", w_out)
        self.register_parameter("_w_q", w_q)
        self.register_parameter("_w_k", w_k)
        self.register_parameter("_w_v", w_v)
        self.register_parameter("_w_kv", w_kv)
        self.register_parameter("_w_qkv", w_qkv)

        self.register_buffer("_k_cache", precomputed_k)
        self.register_buffer("_v_cache", precomputed_v)
        self.register_buffer("_kv_cache", precomputed_kv)

    def newly_initialized_input_weight(
        self,
        dims: list[int],
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.nn.Parameter:
        assert 3 <= len(dims) <= 4
        weight = torch.nn.Parameter(torch.empty(*dims, device=device, dtype=dtype))
        d, input_size = dims[-2:]
        std = math.sqrt(2.0 / float(nhead * d + input_size)) * self.init_gain
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(weight, -a, a)
        return weight

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_k: int,
        d_v: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        config: PFNAttentionConfig,
        share_kv_across_n_heads: int = 1,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = False,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
    ):
        super().__init__()
        assert config.nhead % share_kv_across_n_heads == 0
        self._input_size = config.emsize
        self._output_size = config.emsize
        self._d_k = d_k
        self._d_v = d_v
        self._nhead = config.nhead
        self._nhead_kv = config.nhead // share_kv_across_n_heads
        self._device = device
        self._dtype = dtype
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.init_gain = config.attention_init_gain

        w_out = torch.nn.Parameter(
            torch.empty(config.nhead, d_v, self._output_size, device=device, dtype=dtype)
        )
        if initialize_output_to_zero:
            torch.nn.init.zeros_(w_out)
        else:
            torch.nn.init.xavier_uniform_(w_out)

        assert precomputed_k is None == precomputed_v is None
        has_precomputed_kv = precomputed_kv is not None or precomputed_k is not None
        w_q = None
        w_k = None
        w_v = None
        w_kv = None
        w_qkv = None
        if d_k == d_v and self._nhead == self._nhead_kv and not has_precomputed_kv:
            w_qkv = self.newly_initialized_input_weight(
                [3, self._nhead, self._d_k, self._input_size],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
        else:
            w_q = self.newly_initialized_input_weight(
                [1, self._nhead, self._d_k, self._input_size],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
            if not has_precomputed_kv:
                if d_k == d_v:
                    w_kv = self.newly_initialized_input_weight(
                        [2, self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    w_k = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                    w_v = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_v, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
        self.set_parameters(
            w_out,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            precomputed_k,
            precomputed_v,
            precomputed_kv,
        )
        if config.recompute_attn:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        add_input: bool = False,
        allow_inplace: bool = False,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_cached_kv: bool = False,
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

        output: torch.Tensor = self._compute(
            x,
            x_kv,
            self._k_cache,
            self._v_cache,
            self._kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
        )
        if add_input:
            output = output + x
        if allow_inplace:
            return output
        return output.reshape(x_shape_after_transpose[:-1] + output.shape[-1:])

    def compute_qkv(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        assert not (cache_kv and use_cached_kv), "You cannot both cache new KV and use the cached KV at once."
        if reuse_first_head_kv:
            assert x is not x_kv, (
                "x and x_kv must be different tensors. That means reuse_first_head_kv"
                "is not compatible with self attention only cross attention."
            )
        if x_kv is None:
            x_kv = x

        k = v = kv = None
        if use_cached_kv:
            assert self.has_cached_kv, "You try to use cached keys and values but the cache is empty."
            k = k_cache
            v = v_cache
            kv = kv_cache

        assert (k is None) == (v is None)

        if self._w_qkv is None:
            w_q, w_kv = self._w_q[0], self._w_kv
        else:
            w_q, w_kv = self._w_qkv[0], self._w_qkv[1:]

        if self._w_qkv is not None and x is x_kv and kv is None and k is None and v is None:
            batch_shape = x.shape[:-1]
            j, nhead, d_k, input_size = self._w_qkv.shape
            w_flat = self._w_qkv.reshape(-1, input_size)
            qkv_flat = torch.matmul(x, w_flat.T)
            qkv = qkv_flat.reshape(*batch_shape, j, nhead, d_k)
            q = None
        else:
            qkv = None
            q = torch.einsum("... s, h d s -> ... h d", x, w_q)

        if kv is None and k is None and v is None and qkv is None:
            if w_kv is not None:
                if reuse_first_head_kv:
                    orig_num_heads = w_kv.shape[1]
                    w_kv = w_kv[:, :1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in kv.shape]
                    expand_shape[-2] = orig_num_heads
                    kv = kv.expand(*expand_shape)
            else:
                w_k = self._w_k
                w_v = self._w_v
                if reuse_first_head_kv:
                    orig_num_heads = w_k.shape[0]
                    w_k = w_k[:1]
                    w_v = w_v[:1]
                k = torch.einsum("... s, h d s -> ... h d", x_kv, w_k)
                v = torch.einsum("... s, h d s -> ... h d", x_kv, w_v)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in k.shape]
                    expand_shape[-2] = orig_num_heads
                    k = k.expand(*expand_shape)
                    v = v.expand(*expand_shape)

        if cache_kv:
            if k_cache is not None:
                k_cache[:] = k
            if v_cache is not None:
                v_cache[:] = v
            if kv_cache is not None:
                if kv_cache.shape[-2] == 1:
                    kv_cache[:] = kv[..., :1, :]
                else:
                    kv_cache[:] = kv

        return q, k, v, kv, qkv

    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
    ) -> torch.Tensor:
        q, k, v, kv, qkv = self.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
        )
        attention_head_outputs = PFNMultiHeadAttention.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout_p,
            self.softmax_scale,
        )
        return torch.einsum("... h d, h d s -> ... s", attention_head_outputs, self._w_out)

    def _rearrange_inputs_to_flat_batch(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size]:
        x_shape_after_transpose = x.shape
        if x_kv is not None:
            assert x.shape[:-2] == x_kv.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        if x_kv is not None:
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
        return x, x_kv, x_shape_after_transpose

    @staticmethod
    def broadcast_kv_across_heads(kv: torch.Tensor, share_kv_across_n_heads: int) -> torch.Tensor:
        if share_kv_across_n_heads == 1:
            return kv

        nhead, d = kv.shape[-2:]
        kv = kv[..., None, :].expand(*([-1] * (kv.dim() - 1)), share_kv_across_n_heads, -1)
        return kv.reshape(*kv.shape[:-3], nhead * share_kv_across_n_heads, d)

    @staticmethod
    def scaled_dot_product_attention_chunked(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float | None = None,
        max_batch_size: int = 65_000,
        **extra_inputs,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        output_chunks = []

        for start_idx in range(0, batch_size, max_batch_size):
            end_idx = min(start_idx + max_batch_size, batch_size)
            q_chunk = q[start_idx:end_idx]
            k_chunk = k[start_idx:end_idx]
            v_chunk = v[start_idx:end_idx]
            chunk_output = torch.nn.functional.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                dropout_p=dropout_p,
                **extra_inputs,
            )
            output_chunks.append(chunk_output)

        return torch.cat(output_chunks, dim=0)

    @staticmethod
    def compute_attention_heads(
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        kv: torch.Tensor | None,
        qkv: torch.Tensor | None,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        attn_weight_external: torch.Tensor | None = None,
        external_gate: float | None = None,
    ) -> torch.Tensor:
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
        assert (attn_weight_external is None) == (external_gate is None)

        batch_size, seqlen_q, nhead, d_k = q.shape
        _, _, nhead_kv, d_v = v.shape
        share_kv_across_n_heads = nhead // nhead_kv
        if dropout_p is None:
            dropout_p = 0.0

        use_external = attn_weight_external is not None and external_gate is not None

        if torch.__version__.startswith("1.") or use_external:
            k = PFNMultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
            v = PFNMultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)
            logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
            logits *= torch.sqrt(torch.tensor(1.0 / d_k, device=k.device, dtype=k.dtype)) if softmax_scale is None else softmax_scale
            ps = torch.softmax(logits, dim=2)
            ps = torch.dropout(ps, dropout_p, train=True)
            if use_external:
                text_enhanced = attn_weight_external.unsqueeze(1).repeat(1, nhead, 1, 1)
                ps = external_gate * ps + (1.0 - external_gate) * text_enhanced
            attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)
        else:
            extra_inputs = {}
            if softmax_scale is not None:
                extra_inputs["scale"] = softmax_scale

            if USE_TORCH_2_GQA:
                extra_inputs["enable_gqa"] = True
            else:
                k = PFNMultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
                v = PFNMultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)

            attention_head_outputs = PFNMultiHeadAttention.scaled_dot_product_attention_chunked(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                **extra_inputs,
            ).transpose(1, 2)

        return attention_head_outputs.reshape(batch_size, seqlen_q, nhead, d_v)

    @staticmethod
    def convert_torch_nn_multihead_attention_state_dict(
        state_dict: dict,
        nhead: int,
        *,
        disable_stacked_w_qkv: bool = False,
    ) -> dict:
        in_proj_weight = state_dict["in_proj_weight"]
        out_proj_weight = state_dict["out_proj.weight"]

        embed_dim = in_proj_weight.shape[1]
        assert embed_dim % nhead == 0
        assert in_proj_weight.shape[0] == 3 * embed_dim
        assert out_proj_weight.shape == (embed_dim, embed_dim)
        in_proj_weight = in_proj_weight.reshape(3, nhead, -1, embed_dim)

        new_state_dict = {}
        if disable_stacked_w_qkv:
            new_state_dict["_w_q"], new_state_dict["_w_kv"] = torch.split(in_proj_weight, [1, 2])
            new_state_dict["_w_q"] = new_state_dict["_w_q"].squeeze(0)
        else:
            new_state_dict["_w_qkv"] = in_proj_weight
        new_state_dict["_w_out"] = out_proj_weight.T.reshape(nhead, -1, embed_dim)
        return new_state_dict


class PFNMultiheadAttentionWrapper(nn.Module):
    """Wrapper that mimics torch.nn.MultiheadAttention while delegating to PFNMultiHeadAttention."""

    def __init__(
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
            raise ValueError("PFNMultiHeadAttention does not support bias terms.")
        if add_bias_kv or add_zero_attn:
            raise ValueError("add_bias_kv/add_zero_attn are not supported by PFN attention.")
        if kdim is not None and kdim != embed_dim:
            raise ValueError("kdim must equal embed_dim for PFNMultiHeadAttention.")
        if vdim is not None and vdim != embed_dim:
            raise ValueError("vdim must equal embed_dim for PFNMultiHeadAttention.")

        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.reuse_first_head_kv = reuse_first_head_kv
        self.config = config or PFNAttentionConfig(emsize=embed_dim, nhead=num_heads)
        self.core = PFNMultiHeadAttention(
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
            raise NotImplementedError("Masks/causal attention are not implemented for PFNMultiHeadAttention.")

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
