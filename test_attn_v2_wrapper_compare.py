"""Compare torch.nn.MultiheadAttention vs PFNMultiheadAttentionV2Wrapper.

This is a lightweight sanity-check script that:
1) Creates a torch MHA and a PFN wrapper with the same (embed_dim, num_heads).
2) Copies torch weights into the PFN core via the provided conversion helper.
3) Runs a forward pass and checks shapes + numerical closeness.
4) Demonstrates optional external (text-enhanced) attention on the PFN wrapper.

Note: We load modules via importlib to avoid importing tfmplayground/__init__.py,
which may require optional dependencies (pfns, h5py) in some environments.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_tfmplayground_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    # Stub the package so absolute imports like `tfmplayground.attn_v2` work.
    pkg = types.ModuleType("tfmplayground")
    pkg.__path__ = [str(repo_root / "tfmplayground")]
    sys.modules.setdefault("tfmplayground", pkg)

    attn_v2 = _load_tfmplayground_module(
        "tfmplayground.attn_v2", repo_root / "tfmplayground" / "attn_v2.py"
    )
    wrapper_mod = _load_tfmplayground_module(
        "tfmplayground.attn_v2_wrapper", repo_root / "tfmplayground" / "attn_v2_wrapper.py"
    )

    PFNMultiHeadAttentionV2 = attn_v2.PFNMultiHeadAttentionV2
    PFNMultiheadAttentionV2Wrapper = wrapper_mod.PFNMultiheadAttentionV2Wrapper

    torch.manual_seed(0)

    embed_dim = 16
    num_heads = 4
    batch_first = True
    dropout = 0.0

    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=False,
        batch_first=batch_first,
    ).eval()

    pfn_mha = PFNMultiheadAttentionV2Wrapper(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=False,
        batch_first=batch_first,
        text_enhanced=True,
    ).eval()

    # Copy weights (so the two implementations are comparable).
    converted = PFNMultiHeadAttentionV2.convert_torch_nn_multihead_attention_state_dict(
        {
            "in_proj_weight": torch_mha.in_proj_weight.detach(),
            "out_proj.weight": torch_mha.out_proj.weight.detach(),
        },
        nhead=num_heads,
    )
    missing, unexpected = pfn_mha.core.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected key mismatch when loading converted weights. missing={missing} unexpected={unexpected}")

    # Inputs: B x L x E (batch_first=True)
    B, Lq, Lk = 2, 6, 5
    query = torch.randn(B, Lq, embed_dim)
    key = torch.randn(B, Lk, embed_dim)
    value = key  # PFN attention requires key/value to match

    torch_out, torch_w = torch_mha(query, key, value, need_weights=True, average_attn_weights=True)
    pfn_out, pfn_w = pfn_mha(query, key, value, need_weights=True, average_attn_weights=True)

    print("torch_out:", tuple(torch_out.shape), "pfn_out:", tuple(pfn_out.shape))
    print("torch_attn:", tuple(torch_w.shape), "pfn_attn:", tuple(pfn_w.shape))
    print("max|diff|:", (torch_out - pfn_out).abs().max().item())

    # Demonstrate external (text-enhanced) attention (test->train style weight matrix).
    # Shape expected: (B, Lq, Lk) or (Lq, Lk) or (1, Lq, Lk).
    attn_weight_external = torch.softmax(torch.randn(1, Lq, Lk), dim=-1)
    external_gate = torch.sigmoid(torch.randn(num_heads))  # per-head gate (H,)

    pfn_out_ext, _ = pfn_mha(
        query,
        key,
        value,
        need_weights=False,
        attn_weight_external=attn_weight_external,
        external_gate=external_gate,
    )
    print("pfn_out_ext:", tuple(pfn_out_ext.shape), "max|diff vs pfn_out|:", (pfn_out_ext - pfn_out).abs().max().item())

    # Show that text_enhanced=False ignores the external attention inputs.
    pfn_mha_off = PFNMultiheadAttentionV2Wrapper(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=False,
        batch_first=batch_first,
        text_enhanced=False,
    ).eval()
    pfn_mha_off.core.load_state_dict(converted, strict=False)
    pfn_out_off, _ = pfn_mha_off(
        query,
        key,
        value,
        need_weights=False,
        attn_weight_external=attn_weight_external,
        external_gate=external_gate,
    )
    print("pfn_out_off:", tuple(pfn_out_off.shape), "max|diff vs pfn_out|:", (pfn_out_off - pfn_out).abs().max().item())


if __name__ == "__main__":
    main()
