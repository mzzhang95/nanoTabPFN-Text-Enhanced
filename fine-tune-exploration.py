"""
Fine-tune the model's trainable `external_gate` (text-attention blending gate).

The goal: only update the gate parameters that control how much the model trusts
external text attention when predicting *test rows from train context*.

This script is written in a “step-by-step” style. Each step is explained in code
and comments so you can port it to a notebook cell-by-cell.

What the gate is in THIS repo
-----------------------------
In `tfmplayground/model.py`, the transformer stack is set up as:
- layers 0..(L-2): normal attention
- last layer: `text_enhanced=True` and has `external_gate` (per-head logits)

During the last layer's datapoint attention:
- train rows attend to train rows (no external attention)
- test rows attend to train rows, optionally mixing in `attn_weight_external`
  computed from text embeddings

The gate parameter stored on the last block is a vector of logits with shape (H,),
one per attention head. The forward pass applies `sigmoid()` to obtain values in
(0, 1), which are then used as a blending coefficient.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import schedulefree  # type: ignore
except Exception:  # noqa: BLE001
    schedulefree = None

# If `pfns` isn't installed, importing `tfmplayground.interface` fails.
# We keep a tiny stub here so this script can run in minimal environments.
try:
    import pfns.bar_distribution  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001
    import sys
    import types

    pfns_mod = types.ModuleType("pfns")
    bar_mod = types.ModuleType("pfns.bar_distribution")

    class FullSupportBarDistribution(torch.nn.Module):
        def __init__(self, bucket_edges: torch.Tensor):
            super().__init__()
            if bucket_edges.dim() != 1:
                raise ValueError("bucket_edges must be 1D")
            self.register_buffer("bucket_edges", bucket_edges.clone())

        def float(self):
            self.bucket_edges = self.bucket_edges.float()
            return self

        def mean(self, logits: torch.Tensor) -> torch.Tensor:
            centers = 0.5 * (self.bucket_edges[:-1] + self.bucket_edges[1:])
            centers = centers.to(device=logits.device, dtype=logits.dtype)
            probs = torch.softmax(logits, dim=-1)
            return (probs * centers).sum(dim=-1)

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if targets.dim() == logits.dim():
                targets = targets.squeeze(-1)
            edges = self.bucket_edges.to(device=logits.device, dtype=logits.dtype)
            boundaries = edges[1:-1]
            bin_idx = torch.bucketize(targets, boundaries).clamp(0, edges.numel() - 2)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_p_k = log_probs.gather(dim=-1, index=bin_idx.unsqueeze(-1)).squeeze(-1)
            widths = edges[1:] - edges[:-1]
            width_k = widths.gather(dim=0, index=bin_idx)
            return -(log_p_k - torch.log(width_k))

    bar_mod.FullSupportBarDistribution = FullSupportBarDistribution
    sys.modules.setdefault("pfns", pfns_mod)
    sys.modules.setdefault("pfns.bar_distribution", bar_mod)

from tfmplayground import model
from tfmplayground.interface import NanoTabPFNRegressor  # noqa: E402


# -----------------------------
# Step 0) Reproducibility
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)


# -----------------------------
# Step 1) Configuration
# -----------------------------
DATA_PATH = "data/climate_ttc/climate_2014_2023_final_with_embeddings_lag_3.csv"
DATE_COLUMN = "date"
TARGET_COLUMN = "temp"
NUMERIC_FEATURES = ["precip_lag0","precip_lag1","precip_lag2","precip_lag3",
                    "humidity_lag0","humidity_lag1","humidity_lag2","humidity_lag3",
                    "windspeed_lag0","windspeed_lag1","windspeed_lag2","windspeed_lag3",
                    "temp_lag0","temp_lag1","temp_lag2","temp_lag3"]

# Text lags to use (your regressor averages similarity across the L dimension).
TEXT_EMBEDDING_LAGS = [1, 2, 3]

# Use chronological splits to avoid leakage in time series.
MAX_ROWS = None
# Fraction-based split: first CONTEXT_RATIO for context, next TUNE_RATIO for tuning, remainder eval.
CONTEXT_RATIO = 0.2
TUNE_RATIO = 0.6
# Base context size used inside the rolling/batching fine-tune loop.
# Following the example: if N_tune >= 200, we use base 200; else we use half of tune.
BASE_CONTEXT_IN_TUNE = 200
# How many new tune rows to add per batch step.
TUNE_BATCH_SIZE = 4
# Optional: cap the training context length during tuning to avoid OOM.
# If None, use all available context; if an int, keep only the last K rows of
# [global_context + past_tune] when building each training window.
MAX_CONTEXT_FOR_TUNE = 1000

# Fine-tuning hyperparameters (we tune only a few numbers: per-head gate logits).
EPOCHS = 50
LEARNING_RATE = 1e-4  # kept for backward compatibility; see GATE_LR below
LOG_EVERY = 5  # epoch-level logging cadence
STEP_LOG_EVERY = 10  # step-level logging inside each epoch

# Use a higher LR for the gate to encourage it to move away from ~0.5 if helpful.
GATE_LR = 5e-1

# Optional: clamp gate *logits* to avoid extreme saturation of sigmoid.
GATE_LOGIT_CLAMP = 10.0

# Optional: encourage gate probabilities away from 0.5 (toward extremes).
# Minimizing gate*(1-gate) pushes sigmoids toward 0 or 1. Set to 0.0 to disable.
GATE_REG_STRENGTH = 0.10


# -----------------------------
# Step 2) Data loading utilities
# -----------------------------

import torch
from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.attn_v2 import PFNMultiHeadAttentionV2
from tfmplayground.interface import NanoTabPFNRegressor

def load_nanotabpfn_regressor(
    model_path="checkpoints/nanotabpfn_regressor.pth",
    dist_path="checkpoints/nanotabpfn_regressor_buckets.pth",
    device=None,
    nhead=8,  # checkpoint uses 128-d embeddings → 8x16 heads
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location="cpu")

    # Infer architecture from the checkpoint
    embed = state["feature_encoder.linear_layer.weight"].shape[0]   # 128
    hidden = state["decoder.linear1.weight"].shape[0]               # 512
    num_outputs = state["decoder.linear2.weight"].shape[0]          # 100
    num_layers = len([k for k in state if k.endswith("norm3.weight")])

    # Rename old prefixes (self_attn → self_attention)
    for k in list(state.keys()):
        if ".self_attn_between_datapoints." in k:
            state[k.replace(".self_attn_between_datapoints.", ".self_attention_between_datapoints.")] = state.pop(k)
        elif ".self_attn_between_features." in k:
            state[k.replace(".self_attn_between_features.", ".self_attention_between_features.")] = state.pop(k)

    # Convert torch MHA weights to PFN format
    for i in range(num_layers):
        for kind in ["self_attention_between_datapoints", "self_attention_between_features"]:
            prefix = f"transformer_encoder.transformer_blocks.{i}.{kind}"
            in_proj = state.pop(f"{prefix}.in_proj_weight")
            out_proj = state.pop(f"{prefix}.out_proj.weight")
            state.pop(f"{prefix}.in_proj_bias", None)
            state.pop(f"{prefix}.out_proj.bias", None)
            converted = PFNMultiHeadAttentionV2.convert_torch_nn_multihead_attention_state_dict(
                {"in_proj_weight": in_proj, "out_proj.weight": out_proj},
                nhead=nhead,
            )
            for k, v in converted.items():
                state[f"{prefix}.core.{k}"] = v

    model = NanoTabPFNModel(
        embedding_size=embed,
        num_attention_heads=nhead,
        mlp_hidden_size=hidden,
        num_layers=num_layers,
        num_outputs=num_outputs,
    )

    # Fill new params (external_gate) if missing
    for k, v in model.state_dict().items():
        if k.endswith("external_gate") and k not in state:
            state[k] = v

    model.load_state_dict(state, strict=False)

    bucket_edges = torch.load(dist_path, map_location=device)
    dist = FullSupportBarDistribution(bucket_edges).float()

    return NanoTabPFNRegressor(model=model.to(device), dist=dist, device=device, num_mem_chunks=8, use_text_attn=True)

def _parse_embedding_column(series: pd.Series) -> np.ndarray:
    """Parse a CSV embedding column (stringified list) to a float32 array (N, D)."""
    parsed = [np.asarray(ast.literal_eval(v), dtype=np.float32) for v in series.astype(str).tolist()]
    return np.stack(parsed, axis=0)


def load_climate_dataset(
    *,
    path: str,
    max_rows: int | None,
    embedding_lags: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numeric features + target + text embeddings.

    Returns:
    - X: (N, F) numeric features
    - y: (N,) target
    - text: (N, L, D) text embeddings (lags are treated as separate text features)
    """
    df = pd.read_csv(path)

    # Sort by date to make chronological splits meaningful.
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    for col in [*NUMERIC_FEATURES, TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    X = df[NUMERIC_FEATURES].astype(np.float32).to_numpy()
    y = df[TARGET_COLUMN].astype(np.float32).to_numpy()

    embedding_cols = [f"embedding_text_lag{lag}" for lag in embedding_lags]
    for col in embedding_cols:
        if col not in df.columns:
            raise ValueError(f"Missing embedding column '{col}' in {path}")

    # Each lag becomes a separate text feature: (N, L, D)
    by_lag = [_parse_embedding_column(df[col]) for col in embedding_cols]  # list[(N, D)]
    text = np.stack(by_lag, axis=1)  # (N, L, D)
    return X, y, text


# -----------------------------
# Step 3) Chronological splitting
# -----------------------------
def time_split(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    *,
    context_ratio: float,
    tune_ratio: float,
) -> tuple[np.ndarray, ...]:
    """
    Split sequentially into context / tune / eval.

    - context: first `context_ratio` fraction (chronological)
    - tune: next `tune_ratio` fraction
    - eval: remainder
    """
    n = len(y)
    n_context = int(n * context_ratio)
    n_tune = int(n * tune_ratio)
    n_eval = n - n_context - n_tune
    if n_context <= 0 or n_tune <= 0 or n_eval <= 0:
        raise ValueError(f"Invalid split sizes: n={n}, context={n_context}, tune={n_tune}, eval={n_eval}")

    X_context, y_context, text_context = X[:n_context], y[:n_context], text[:n_context]
    X_tune, y_tune, text_tune = (
        X[n_context : n_context + n_tune],
        y[n_context : n_context + n_tune],
        text[n_context : n_context + n_tune],
    )
    X_eval, y_eval, text_eval = X[n_context + n_tune :], y[n_context + n_tune :], text[n_context + n_tune :]

    return (
        X_context,
        y_context,
        text_context,
        X_tune,
        y_tune,
        text_tune,
        X_eval,
        y_eval,
        text_eval,
    )


# -----------------------------
# Step 4) Evaluation helper (uses reg.predict)
# -----------------------------
def evaluate(reg: NanoTabPFNRegressor, *, X: np.ndarray, y: np.ndarray, text: np.ndarray, label: str) -> None:
    """
    End-to-end evaluation using the regressor API.

    Important: `predict()` runs under `torch.no_grad()`, so it is for evaluation only.
    """
    preds = reg.predict(X, text)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    print(f"{label} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")


def evaluate_no_text_attn(reg: NanoTabPFNRegressor, *, X: np.ndarray, y: np.ndarray, label: str) -> None:
    """
    Baseline evaluation WITHOUT using any text attention.

    Implementation detail:
    - `NanoTabPFNRegressor.predict` supports `use_text_attn=False` and `text_test=None`.
    - This forces the model to behave like the original TabPFN-style in-context predictor.
    """
    preds = reg.predict(X, None, use_text_attn=False)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    print(f"{label} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")


# -----------------------------
# Step 5) Gate-only parameter selection
# -----------------------------
def get_last_layer_gate_param(reg: NanoTabPFNRegressor) -> torch.nn.Parameter:
    """
    Return the trainable gate parameter (per-head logits) from the last transformer block.

    Only the last transformer block is `text_enhanced=True` in this repo.
    """
    last_block = reg.model.transformer_encoder.transformer_blocks[-1]
    if getattr(last_block, "external_gate", None) is None:
        raise RuntimeError("Last block has no external_gate. Is text_enhanced enabled in the model?")
    return last_block.external_gate


def freeze_all_but_last_gate(reg: NanoTabPFNRegressor) -> torch.nn.Parameter:
    """Freeze all parameters and enable gradients only for the last block's external_gate."""
    for p in reg.model.parameters():
        p.requires_grad_(False)
    gate = get_last_layer_gate_param(reg)
    gate.requires_grad_(True)
    return gate


def gate_stats(gate_logits: torch.Tensor) -> str:
    """Human-readable summary for per-head gate logits and their sigmoid values."""
    logits = gate_logits.detach().float().cpu().reshape(-1)
    gate = torch.sigmoid(logits)
    return (
        f"logits(mean/min/max)={logits.mean().item():.4f}/{logits.min().item():.4f}/{logits.max().item():.4f} | "
        f"sigmoid(mean/min/max)={gate.mean().item():.4f}/{gate.min().item():.4f}/{gate.max().item():.4f}"
    )


# -----------------------------
# Step 6) Fine-tuning loop (calls reg.model directly)
# Rolling window: at step i, train on context + tune[0:i], predict tune[i].
# -----------------------------
def fine_tune_external_gate(
    reg: NanoTabPFNRegressor,
    *,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    text_tune: np.ndarray,
) -> None:
    """
    Fine-tune only the last layer's gate on the tune split.

    Why call the model directly?
    - `reg.predict()` disables gradients, so we can't optimize with it.
    - We still re-use everything from `reg.fit()`:
        - `feature_preprocessor`, `X_train` (preprocessed), `y_train_n` (normalized)
        - text attention computation `_compute_attn_weight_external(text_tune)`

    Rolling window (per step inside each epoch):
    - Train set = context + all prior tune rows
    - Test row = current tune row
    - Loss = MSE on that current row (raw space)
    """
    # (1) Select tunable params (gate only) and build optimizer.
    gate = freeze_all_but_last_gate(reg)
    # Use Schedule-Free AdamW (same optimizer used in tfmplayground/train.py).
    # If the `schedulefree` package isn't installed, fall back to AdamW.
    if schedulefree is None:
        print("WARNING: `schedulefree` not installed; falling back to torch.optim.AdamW.")
        optimizer = torch.optim.AdamW([gate], lr=GATE_LR, weight_decay=0.0)
    else:
        # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
        optimizer = schedulefree.AdamWScheduleFree([gate], lr=GATE_LR, weight_decay=0.0)
    print("Tuning last-layer external_gate:", gate_stats(gate))

    # (2) Precompute transformed tune features and normalized tune targets (using context stats).
    X_tune_proc = reg.feature_preprocessor.transform(X_tune)
    y_tune_norm = (y_tune - reg.y_train_mean) / reg.y_train_std

    # Backup original train_text so we can temporarily extend it per step.
    train_text_orig = reg.train_text

    # (3) Rolling-window gradient loop (with base context then growing window)
    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_tune) / TUNE_BATCH_SIZE))
    for epoch in range(1, EPOCHS + 1):
        for step_idx in range(num_steps):
            optimizer.zero_grad()

            start = step_idx * TUNE_BATCH_SIZE
            end = min(len(y_tune), start + TUNE_BATCH_SIZE)

            # Base context portion inside the tune set (e.g., first 200 tune rows) + accumulated history.
            base_limit = min(BASE_CONTEXT_IN_TUNE, len(y_tune))
            past_limit = max(base_limit, start)

            # Train set for this step: global context + tune[0:past_limit]
            X_train_full = np.concatenate((reg.X_train, X_tune_proc[:past_limit]))
            y_train_full = np.concatenate((reg.y_train_n, y_tune_norm[:past_limit]))

            # Apply optional context cap to avoid long sequences (OOM protection).
            if MAX_CONTEXT_FOR_TUNE is not None:
                X_train_step = X_train_full[-MAX_CONTEXT_FOR_TUNE:]
                y_train_step = y_train_full[-MAX_CONTEXT_FOR_TUNE:]
            else:
                X_train_step = X_train_full
                y_train_step = y_train_full

            # Test batch for this step: tune[start:end]
            X_test_step = X_tune_proc[start:end]
            y_target = torch.tensor(y_tune[start:end], dtype=torch.float32, device=reg.device)

            # Build model inputs
            X_concat = np.concatenate((X_train_step, X_test_step))
            X_tensor = torch.tensor(X_concat, dtype=torch.float32, device=reg.device).unsqueeze(0)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            single_eval_pos = len(X_train_step)

            # Update train_text for this step (context + past tune rows), then cap length if needed
            combined_text = np.concatenate((train_text_orig, text_tune[:past_limit]), axis=0)
            if MAX_CONTEXT_FOR_TUNE is not None:
                combined_text = combined_text[-MAX_CONTEXT_FOR_TUNE:]
            reg.train_text = combined_text
            attn_weight_external = reg._compute_attn_weight_external(text_tune[start:end])  # (1, B, N_train_step)

            logits = reg.model(
                (X_tensor, y_context_tensor),
                single_eval_pos=single_eval_pos,
                num_mem_chunks=1,
                attn_weight_external=attn_weight_external,
            ).squeeze(0)

            preds_n = reg.dist.mean(logits)
            preds = preds_n * reg.y_train_std + reg.y_train_mean
            preds = preds.reshape(-1)

            loss = torch.nn.functional.mse_loss(preds, y_target)
            if GATE_REG_STRENGTH and GATE_REG_STRENGTH > 0:
                gate_prob = torch.sigmoid(gate)
                gate_reg = (gate_prob * (1 - gate_prob)).mean()
                loss = loss + GATE_REG_STRENGTH * gate_reg

            loss.backward()
            optimizer.step()

            # Optional stability clamp in logit space (NOT in [0,1] space).
            if GATE_LOGIT_CLAMP is not None:
                with torch.no_grad():
                    gate.clamp_(-GATE_LOGIT_CLAMP, GATE_LOGIT_CLAMP)

            if (step_idx + 1) % STEP_LOG_EVERY == 0 or (step_idx == num_steps - 1):
                preds_np = preds.detach().cpu().numpy()
                mse = mean_squared_error(y_target.detach().cpu().numpy(), preds_np)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_target.detach().cpu().numpy(), preds_np)
                print(
                    f"Epoch {epoch:02d} Step {step_idx+1:03d}/{num_steps} "
                    f"(rows {start}–{end-1}) | MAE: {mae:.4f} | RMSE: {rmse:.4f} | {gate_stats(gate)}"
                )

    # Restore original train_text to avoid side effects.
    reg.train_text = train_text_orig

    # (4) Freeze again to avoid surprises if you re-use `reg` later.
    for p in reg.model.parameters():
        p.requires_grad_(False)


def main() -> None:
    # Step 1: load the dataset
    X, y, text = load_climate_dataset(
        path=DATA_PATH,
        max_rows=MAX_ROWS,
        embedding_lags=TEXT_EMBEDDING_LAGS,
    )

    # Step 2: split chronologically
    (
        X_context,
        y_context,
        text_context,
        X_tune,
        y_tune,
        text_tune,
        X_eval,
        y_eval,
        text_eval,
    ) = time_split(X, y, text, context_ratio=CONTEXT_RATIO, tune_ratio=TUNE_RATIO)
    print(f"Split sizes: context={len(y_context)} tune={len(y_tune)} eval={len(y_eval)}")

    # Step 3: initialize regressor + store context set
    reg = load_nanotabpfn_regressor()
    # Optional: force a specific GPU (e.g., GPU 1). Adjust as needed.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
        reg.device = device
        reg.model = reg.model.to(device)
        reg.dist = reg.dist.to(device)
    reg.fit(X_context, y_context, text_context)

    # Step 4: baseline eval
    print("\n== Baseline (before tuning) ==")
    evaluate_no_text_attn(reg, X=X_eval, y=y_eval, label="Eval (no text attn)")
    evaluate(reg, X=X_eval, y=y_eval, text=text_eval, label="Eval (with text attn)")
    print("Initial gate:", gate_stats(get_last_layer_gate_param(reg)))

    # Step 5: fine-tune gate on tune split
    print("\n== Fine-tuning external_gate ==")
    fine_tune_external_gate(reg, X_tune=X_tune, y_tune=y_tune, text_tune=text_tune)

    # Step 6: eval after tuning
    print("\n== After tuning ==")
    evaluate_no_text_attn(reg, X=X_eval, y=y_eval, label="Eval (no text attn)")
    evaluate(reg, X=X_eval, y=y_eval, text=text_eval, label="Eval (with text attn)")
    print("Tuned gate:", gate_stats(get_last_layer_gate_param(reg)))


if __name__ == "__main__":
    main()
