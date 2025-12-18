import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tfmplayground.interface import NanoTabPFNRegressor

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters for this quick exploration
MAX_ROWS = 800  # limit rows to keep runs fast
EPOCHS = 50
LEARNING_RATE = 5e-2
LOG_EVERY = 10

# Dataset produced by data/create_embeddings_with_lags.py
DATA_PATH = "data/climate_ttc/climate_2014_2023_final_with_embeddings_lag_3.csv"
TARGET_COLUMN = "temp"
NUMERIC_FEATURES = ["precip", "humidity", "windspeed"]


def load_climate_with_embeddings(max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the lagged-embedding climate CSV and return numeric X, target y, and concatenated text embeddings."""
    df = pd.read_csv(DATA_PATH)
    if max_rows is not None:
        df = df.head(max_rows)

    embedding_cols = [c for c in df.columns if c.startswith("embedding_text_lag")]
    if not embedding_cols:
        raise ValueError(f"No embedding columns found in {DATA_PATH}.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing in {DATA_PATH}.")

    # Parse list-like strings into numeric arrays
    for col in embedding_cols:
        df[col] = df[col].apply(ast.literal_eval)

    y = df[TARGET_COLUMN].astype(np.float32).to_numpy()
    X_num = df[NUMERIC_FEATURES].astype(np.float32).to_numpy()

    def concat_embeddings(row: pd.Series) -> np.ndarray:
        return np.concatenate([np.asarray(row[c], dtype=np.float32) for c in embedding_cols])

    X_text = np.stack([concat_embeddings(row) for _, row in df[embedding_cols].iterrows()])
    return X_num, y, X_text


def prepare_splits() -> tuple[np.ndarray, ...]:
    """Create context/tune/eval splits from the climate dataset with text embeddings."""
    X, y, X_text = load_climate_with_embeddings(MAX_ROWS)

    # Context data feeds the in-context learning part of the model.
    X_context, X_tmp, y_context, y_tmp, X_text_context, X_text_tmp = train_test_split(
        X, y, X_text, test_size=0.4, random_state=42
    )
    # Tune set optimizes the external_gate, eval set measures generalization.
    X_tune, X_eval, y_tune, y_eval, X_text_tune, X_text_eval = train_test_split(
        X_tmp, y_tmp, X_text_tmp, test_size=0.5, random_state=42
    )

    return (
        X_context,
        y_context,
        X_tune,
        y_tune,
        X_eval,
        y_eval,
        X_text_context,
        X_text_tune,
        X_text_eval,
    )


def freeze_all_but_external_gate(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Only allow gradients for external_gate parameters."""
    gate_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if "external_gate" in name:
            param.requires_grad_(True)
            gate_params.append(param)
        else:
            param.requires_grad_(False)
    if not gate_params:
        raise RuntimeError("No external_gate parameters found on the model.")
    return gate_params


def evaluate(reg: NanoTabPFNRegressor, X: np.ndarray, y: np.ndarray, X_text: np.ndarray, label: str) -> None:
    preds = reg.predict(X, X_text)
    print(
        f"{label} | MSE: {mean_squared_error(y, preds):.4f} | R2: {r2_score(y, preds):.4f}"
    )


def fine_tune_external_gate(
    reg: NanoTabPFNRegressor,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_text_tune: np.ndarray,
) -> None:
    """Optimize only the external_gate parameters on the tune split."""
    gate_params = freeze_all_but_external_gate(reg.model)
    optimizer = torch.optim.Adam(gate_params, lr=LEARNING_RATE)

    # Precompute tensors that stay constant during the tiny fine-tuning loop.
    X_tune_proc = reg.feature_preprocessor.transform(X_tune)
    X_concat = np.concatenate((reg.X_train, X_tune_proc))
    X_tensor = torch.tensor(X_concat, dtype=torch.float32, device=reg.device).unsqueeze(0)
    y_context_tensor = torch.tensor(reg.y_train_n, dtype=torch.float32, device=reg.device).unsqueeze(0)
    y_target = torch.tensor(y_tune, dtype=torch.float32, device=reg.device)
    attn_weight_external = reg._compute_attn_weight_external(X_text_tune)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = reg.model(
            (X_tensor, y_context_tensor),
            single_eval_pos=len(reg.X_train),
            num_mem_chunks=1,
            attn_weight_external=attn_weight_external,
            external_gate=None,
        ).squeeze(0)

        preds_n = reg.dist.mean(logits)
        preds = preds_n * reg.y_train_std + reg.y_train_mean
        preds = preds.reshape(-1)

        loss = torch.nn.functional.mse_loss(preds, y_target)
        loss.backward()
        optimizer.step()

        # Keep the gate inside [0, 1] to make the blend interpretable.
        with torch.no_grad():
            for param in gate_params:
                param.clamp_(0.0, 1.0)

        if epoch == 1 or epoch % LOG_EVERY == 0:
            preds_np = preds.detach().cpu().numpy()
            gate_val = torch.stack([p.detach() for p in gate_params]).mean().item()
            print(
                f"Epoch {epoch:02d} | tune MSE: {loss.item():.4f} | tune R2: {r2_score(y_tune, preds_np):.4f} | gate≈{gate_val:.3f}"
            )

    # Freeze everything again after tuning
    for param in reg.model.parameters():
        param.requires_grad_(False)


def main() -> None:
    splits = prepare_splits()
    (
        X_context,
        y_context,
        X_tune,
        y_tune,
        X_eval,
        y_eval,
        X_text_context,
        X_text_tune,
        X_text_eval,
    ) = splits

    reg = NanoTabPFNRegressor(num_mem_chunks=1)
    reg.fit(X_context, y_context, X_text_context)

    print("== Baseline (before fine-tuning external_gate) ==")
    evaluate(reg, X_eval, y_eval, X_text_eval, "Eval")

    print("\n== Fine-tuning external_gate on the tune split ==")
    fine_tune_external_gate(reg, X_tune, y_tune, X_text_tune)

    print("\n== After fine-tuning external_gate ==")
    evaluate(reg, X_eval, y_eval, X_text_eval, "Eval")

    gate_values = [p.detach().cpu().item() for name, p in reg.model.named_parameters() if "external_gate" in name]
    print(f"\nLearned external_gate values: {[round(v, 4) for v in gate_values]}")


if __name__ == "__main__":
    main()
