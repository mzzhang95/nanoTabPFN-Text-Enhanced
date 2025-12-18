"""
Quick check: load the climate lagged-embedding dataset and compute the external text
attention matrix (cosine-softmax) used by NanoTabPFNRegressor._compute_attn_weight_external.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

DATA_PATH = "data/climate_ttc/climate_2014_2023_final_with_embeddings_lag_3.csv"
EMBED_PREFIX = "embedding_text_lag"
# We use lags 1, 2, 3 to compute the text-only similarity between the 11th row and prior 10 rows.
TEXT_LAGS_FOR_ATTENTION = [f"{EMBED_PREFIX}{lag}" for lag in (1, 2, 3)]
# Numeric columns for regular (non-text) attention. Adjust if you want to drop/keep columns.
NUMERIC_COLS = ["precip", "humidity", "windspeed"]
TARGET_COL = "temp"


def _load_model_class():
    """Load NanoTabPFNModel directly from file to avoid importing pfns via package __init__."""
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "tfmplayground" / "model.py"
    # Stub the package to avoid executing tfmplayground/__init__.py (which imports pfns).
    pkg = types.ModuleType("tfmplayground")
    pkg.__path__ = [str(repo_root / "tfmplayground")]
    sys.modules.setdefault("tfmplayground", pkg)
    spec = importlib.util.spec_from_file_location("ntpfn_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NanoTabPFNModel


NanoTabPFNModel = _load_model_class()


def load_embeddings(path: str, max_rows: int | None = None) -> Dict[str, np.ndarray]:
    """Load embedding_text_lag* columns into a dict {col_name: np.ndarray[num_rows, dim]}."""
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)

    emb_cols = [c for c in df.columns if c.startswith(EMBED_PREFIX)]
    if not emb_cols:
        raise ValueError(f"No columns starting with '{EMBED_PREFIX}' found in {path}")

    # Parse list-like strings back to float arrays
    for col in emb_cols:
        df[col] = df[col].apply(ast.literal_eval)

    emb_by_col: Dict[str, np.ndarray] = {}
    for col in emb_cols:
        emb_by_col[col] = np.stack(df[col].apply(lambda x: np.asarray(x, dtype=np.float32)).to_list())

    print(f"Loaded {len(df)} rows with embedding cols {emb_cols}")
    print(f"Per-lag embedding dim: {emb_by_col[emb_cols[0]].shape[1]}")
    return emb_by_col


def load_numeric_and_target(path: str, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load numeric columns (non-text, non-embedding) and target."""
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    missing_num = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_num:
        raise ValueError(f"Missing numeric cols {missing_num} in {path}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target col {TARGET_COL} in {path}")
    X_num = df[NUMERIC_COLS].astype(np.float32).to_numpy()
    y = df[TARGET_COL].astype(np.float32).to_numpy()
    return X_num, y


def _mean_cosine_sim_matrix(emb_by_col: Dict[str, np.ndarray], cols_to_use: Sequence[str]) -> torch.Tensor:
    """Compute mean cosine similarity matrix over specified embedding columns."""
    sims = []
    for col in cols_to_use:
        if col not in emb_by_col:
            raise ValueError(f"Column {col} missing in embeddings.")
        X = torch.tensor(emb_by_col[col], dtype=torch.float32)
        X_norm = torch.nn.functional.normalize(X, dim=1)
        sims.append(X_norm @ X_norm.T)
    if not sims:
        raise ValueError("No columns provided to compute similarity.")
    return torch.stack(sims).mean(dim=0)


def _masked_softmax_row(sim: torch.Tensor, query_idx: int, key_indices: list[int]) -> torch.Tensor:
    """Softmax over a subset of keys for a given query, masking everything else to -inf."""
    mask = torch.full_like(sim, float("-inf"))
    key_tensor = torch.tensor(key_indices, device=sim.device)
    mask[query_idx, key_tensor] = sim[query_idx, key_tensor]
    return torch.softmax(mask[query_idx], dim=-1)


def compute_text_attention_row(
    emb_by_col: Dict[str, np.ndarray],
    query_idx: int,
    key_indices: list[int],
    text_cols: Sequence[str],
) -> torch.Tensor:
    """Mean cosine over text lag embeddings, softmax over provided keys only."""
    sim_text = _mean_cosine_sim_matrix(emb_by_col, text_cols)
    logits = sim_text[query_idx, key_indices]
    return torch.softmax(logits, dim=-1)


def get_regular_attention_from_model(
    model: NanoTabPFNModel, # type: ignore
    X_all: np.ndarray,
    y_train: np.ndarray,
    single_eval_pos: int,
) -> torch.Tensor:
    """
    Run the model forward (no external attention) and capture the datapoint attention weights
    for the test rows attending to the train rows in the last transformer block.
    Returns attention of shape [num_test, num_train] averaged over heads.
    """
    device = torch.device("cpu")
    model = model.to(device)

    X_tensor = torch.tensor(X_all, dtype=torch.float32, device=device).unsqueeze(0)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(0)

    attn_cache: list[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        # output is tuple (attn_output, attn_weights)
        attn_cache.append(output[1].detach())

    # Hook the last block's datapoint attention
    last_block = model.transformer_encoder.transformer_blocks[-1]
    hook = last_block.self_attention_between_datapoints.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(
            (X_tensor, y_tensor),
            single_eval_pos=single_eval_pos,
            num_mem_chunks=1,
            attn_weight_external=None,
            external_gate=None,
        )
    hook.remove()

    if not attn_cache:
        raise RuntimeError("Attention hook did not capture any weights.")

    # Two calls: train->train and test->train. Pick the smaller q dimension (test side).
    attn = sorted(attn_cache, key=lambda t: t.shape[1])[0]  # shape [B, q, k]
    return attn.squeeze(0).cpu()  # [q, k]


def compute_final_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    emb_by_col_11: Dict[str, np.ndarray],
    text_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (regular_attn, text_attn, final_attn) for the first test row vs train rows."""
    # Normalize y like the interface: z-score on train
    y_mean = y_train.mean()
    y_std = y_train.std(ddof=1) + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    # Model instantiation (matches checkpoints shapes)
    model = NanoTabPFNModel(
        num_attention_heads=6,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=100,
    )

    X_all = np.concatenate((X_train, X_test))
    single_eval_pos = len(X_train)

    attn_regular = get_regular_attention_from_model(
        model,
        X_all,
        y_train_norm,
        single_eval_pos,
    )
    attn_regular = attn_regular[0]  # first (only) test row -> train rows

    attn_text = compute_text_attention_row(
        emb_by_col_11,
        query_idx=10,
        key_indices=list(range(10)),
        text_cols=text_cols,
    )

    attn_regular_np = attn_regular.numpy().reshape(-1)
    attn_text_np = attn_text.numpy().reshape(-1)
    attn_final = 0.5 * (attn_regular_np + attn_text_np)
    return attn_regular_np, attn_text_np, attn_final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute blended attention for row 11 vs rows 1-10.")
    parser.add_argument(
        "--path",
        default=DATA_PATH,
        help=f"Path to CSV with embeddings (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=128,
        help="Limit rows for a quick test (default: 128, use -1 for all rows).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    max_rows = None if args.max_rows is None or args.max_rows < 0 else args.max_rows

    emb_by_col = load_embeddings(args.path, max_rows=max_rows)
    X_num, y = load_numeric_and_target(args.path, max_rows=max_rows)

    num_rows = next(iter(emb_by_col.values())).shape[0]
    if num_rows < 11:
        raise ValueError("Need at least 11 rows to compute attention of row 11 vs previous 10 rows.")

    text_cols = [c for c in TEXT_LAGS_FOR_ATTENTION if c in emb_by_col]
    if len(text_cols) < len(TEXT_LAGS_FOR_ATTENTION):
        missing = set(TEXT_LAGS_FOR_ATTENTION) - set(text_cols)
        raise ValueError(f"Missing expected text lag columns: {missing}")

    # Use first 10 rows as train, 11th as test
    X_train, y_train = X_num[:10], y[:10]
    X_test = X_num[10:11]
    emb_by_col_11 = {k: v[:11] for k, v in emb_by_col.items()}

    attn_regular, attn_text, attn_final = compute_final_attention(
        X_train,
        y_train,
        X_test,
        emb_by_col_11,
        text_cols=text_cols,
    )

    print("Regular attention (row 11 -> rows 1-10):")
    for i, w in enumerate(attn_regular):
        print(f"  {i}: {float(w):.4f}")

    print("\nText attention (mean over lags 1-3) (row 11 -> rows 1-10):")
    for i, w in enumerate(attn_text):
        print(f"  {i}: {float(w):.4f}")

    print("\nFinal attention = 0.5 * regular + 0.5 * text:")
    for i, w in enumerate(attn_final):
        print(f"  {i}: {float(w):.4f}")


if __name__ == "__main__":
    main()
