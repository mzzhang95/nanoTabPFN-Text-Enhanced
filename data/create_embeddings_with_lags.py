"""
Create embeddings with lagged text features for the climate CSV. Should be extended to other dataframes easily in the future.

Note: onlt raw data should be uploaded to our repo.

MZ: Missing text is filled with the literal string ``n/a`` so we never use an all-zero
vector placeholder for unavailable text features., e.g. for the first few rows where lagged text is not available.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Fixed configuration (edit here if needed)
csv_path = "./climate_ttc/climate_2014_2023_final.csv"
text_column = "text"
model_name = "Qwen/Qwen3-Embedding-0.6B"
lag_days = 3  # number of prior days to include (in addition to current)
batch_size = 16
max_length = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
output_csv_path = f"./climate_ttc/climate_2014_2023_final_with_embeddings_lag_{lag_days}.csv"
na_text = "n/a"


def sanitize_text_series(series: pd.Series) -> pd.Series:
    # MZ: Replace NaN or empty strings with na_text. Make it a function for reusability and flexibility.
    series = series.fillna("").astype(str).str.strip()
    return series.mask(series == "", na_text)



def embed_texts(texts: List[str]) -> torch.Tensor:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        # MZ: TODO: try both with and without normalization
        normalize_embeddings=False,
    )
    return embeddings.cpu()


def main():
    # read df
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}.")
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # create lagged text columns.
    lag_text_cols = []
    for lag in range(lag_days + 1):
        col_name = f"{text_column}_lag{lag}"
        if lag == 0:
            df[col_name] = df[text_column]
        else:
            # get lag feaute by shifting lag days
            df[col_name] = df[text_column].shift(lag)
        df[col_name] = sanitize_text_series(df[col_name])
        lag_text_cols.append(col_name)

    print(
        f"Loaded {len(df)} rows from {csv_path} on device {device} "
        f"with lags 0..{lag_days}."
    )

    # create embeddings for cols
    for col_name in lag_text_cols:
        print(f"Encoding column '{col_name}'...")
        col_embeddings = embed_texts(df[col_name].tolist())
        df[f"embedding_{col_name}"] = col_embeddings.tolist()

    df.to_csv(output_csv_path, index=False)
    print(f"Wrote CSV with embeddings to {output_csv_path}")


if __name__ == "__main__":
    main()
