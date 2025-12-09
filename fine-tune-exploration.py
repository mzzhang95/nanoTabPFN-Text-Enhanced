import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tfmplayground.interface import NanoTabPFNRegressor

# Test if the model could be loaded successfully
# Data (tiny slice for a quick run)
X, y = fetch_california_housing(return_X_y=True)
X, y = X[:5], y[:5]  # tiny sample
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Use numeric features as stand-in text features for this test
X_text_train = X_train.copy()
X_text_test = X_test.copy()

# Model
reg = NanoTabPFNRegressor(num_mem_chunks=1)

# Fit (note the X_text argument)
reg.fit(X_train, y_train, X_text_train)

# Predict (note X_text_test)
with torch.no_grad():
    preds = reg.predict(X_test, X_text_test)

print("y_test:", y_test)
print("preds :", preds)
print("r2    :", r2_score(y_test, preds))

# Set only external_gate to true
for name, p in reg.model.named_parameters():
    if "external_gate" in name:
        print("setting", name, "as tunable")
        p.requires_grad_(True)
    else:
        p.requires_grad_(False)