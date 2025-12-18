"""Quick sanity-check for NanoTabPFNRegressor's external text attention helpers.

This script tests:
- NanoTabPFNRegressor._compute_text_similarity_external  -> (L, N_test, N_train)
- NanoTabPFNRegressor._compute_attn_weight_external      -> (1, N_test, N_train)

It does not require a full model forward pass; it only needs torch + numpy.
If `pfns` is not installed, we stub the minimal module so `tfmplayground.interface`
can be imported.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _ensure_pfns_stub() -> None:
    """Allow importing tfmplayground.interface without installing pfns."""
    if "pfns" in sys.modules:
        return
    pfns_mod = types.ModuleType("pfns")
    bar_mod = types.ModuleType("pfns.bar_distribution")

    class FullSupportBarDistribution:  # noqa: D401
        """Stub; only needed for import-time."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
            raise RuntimeError("This is a stub. Install `pfns` to use the full regressor.")

    bar_mod.FullSupportBarDistribution = FullSupportBarDistribution
    sys.modules["pfns"] = pfns_mod
    sys.modules["pfns.bar_distribution"] = bar_mod


def _load_interface_module():
    """Load tfmplayground/interface.py without importing tfmplayground/__init__.py."""
    repo_root = Path(__file__).resolve().parent

    # Stub the package so absolute imports like `tfmplayground.model` work.
    pkg = types.ModuleType("tfmplayground")
    pkg.__path__ = [str(repo_root / "tfmplayground")]
    sys.modules.setdefault("tfmplayground", pkg)

    # Stub tfmplayground.utils to avoid importing optional deps like h5py/pfns bucket helpers.
    utils_mod = types.ModuleType("tfmplayground.utils")

    def get_default_device():  # noqa: ANN001
        return "cpu"

    utils_mod.get_default_device = get_default_device
    sys.modules.setdefault("tfmplayground.utils", utils_mod)

    interface_path = repo_root / "tfmplayground" / "interface.py"
    spec = importlib.util.spec_from_file_location("tfmplayground.interface", interface_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load interface module from {interface_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tfmplayground.interface"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _ensure_pfns_stub()
    interface = _load_interface_module()
    NanoTabPFNRegressor = interface.NanoTabPFNRegressor

    # Create an instance without running __init__ (we only need device + train_text).
    reg = NanoTabPFNRegressor.__new__(NanoTabPFNRegressor)
    reg.device = torch.device("cpu")

    n_train, n_test, num_text_features, d = 5, 2, 4, 8  # (N, L, D)
    rng = np.random.default_rng(0)
    train_text = rng.standard_normal(size=(n_train, num_text_features, d), dtype=np.float32)
    test_text = rng.standard_normal(size=(n_test, num_text_features, d), dtype=np.float32)

    # Make test row 0 identical to train row 0 to get cosine similarity ~1.0 for all text features.
    test_text[0] = train_text[0]

    reg.train_text = train_text

    sim = reg._compute_text_similarity_external(test_text)  # (L, N_test, N_train)
    attn = reg._compute_attn_weight_external(test_text)  # (1, N_test, N_train)

    print("sim shape:", tuple(sim.shape), "(expected L, N_test, N_train)")
    print("attn shape:", tuple(attn.shape), "(expected 1, N_test, N_train)")
    print("attn row-sums (should be ~1):", attn.sum(dim=-1))

    # Inspect the "perfect match" similarities for test row 0 vs train row 0.
    sim0 = sim[:, 0, 0].detach().cpu().numpy()
    print("cosine(sim) for test[0] vs train[0] across features:", np.round(sim0, 4))

    # For attention, train[0] should get the largest weight for test[0] (usually by a lot).
    attn0 = attn[0, 0].detach().cpu().numpy()
    print("attn for test[0] over train rows:", np.round(attn0, 4))
    print("argmax train idx for test[0]:", int(attn0.argmax()))


if __name__ == "__main__":
    main()
