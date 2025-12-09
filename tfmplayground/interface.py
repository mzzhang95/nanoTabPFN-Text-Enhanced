import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device
from tfmplayground.attn_v2 import MultiHeadAttention as PFNMultiHeadAttentionV2

def init_model_from_state_dict_file(file_path):
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model = NanoTabPFNModel(
    num_attention_heads=6,# state_dict['architecture']['num_attention_heads'],
    embedding_size=192, # state_dict['architecture']['embedding_size'],
    mlp_hidden_size=768, # state_dict['architecture']['mlp_hidden_size'],
    num_layers=6, # state_dict['architecture']['num_layers'],
    num_outputs=100, # state_dict['architecture']['num_outputs'],
)

    # Map torch MHA weights into PFN format for each block
    for i in range(model.num_layers):
        for kind in ["self_attention_between_datapoints", "self_attention_between_features"]:
            prefix_old = f"transformer_encoder.transformer_blocks.{i}.{kind}"
            prefix_new = f"{prefix_old}.core"
            # pull torch weights
            in_proj = state_dict.pop(f"{prefix_old}.in_proj_weight")
            out_proj = state_dict.pop(f"{prefix_old}.out_proj.weight")
            # optional bias keys can be popped/ignored if present
            state_dict.pop(f"{prefix_old}.in_proj_bias", None)
            state_dict.pop(f"{prefix_old}.out_proj.bias", None)
            converted = PFNMultiHeadAttentionV2.convert_torch_nn_multihead_attention_state_dict(
                {"in_proj_weight": in_proj, "out_proj.weight": out_proj},
                nhead=model.num_attention_heads,
            )
            for k, v in converted.items():
                state_dict[f"{prefix_new}.{k}"] = v
    # Fill in new external_gate params with their initialized values if missing
    model_init_state = model.state_dict()
    for key, tensor in model_init_state.items():
        if key.endswith("external_gate") and key not in state_dict:
            state_dict[key] = tensor

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing, "unexpected:", unexpected)
    return model

# def init_model_from_state_dict_file(file_path):
#     """
#     reads model architecture from state dict, instantiates the architecture and loads the weights
#     """
#     # print(file_path)
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
#     #print(state_dict.keys())

#     model = NanoTabPFNModel(
#         num_attention_heads=6,# state_dict['architecture']['num_attention_heads'],
#         embedding_size=192, # state_dict['architecture']['embedding_size'],
#         mlp_hidden_size=768, # state_dict['architecture']['mlp_hidden_size'],
#         num_layers=6, # state_dict['architecture']['num_layers'],
#         num_outputs=100, # state_dict['architecture']['num_outputs'],
#     )
#     # MZ: Added a new param, by pass checks
#     # model.load_state_dict(state_dict)
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     print("missing:", missing, "unexpected:", unexpected)
#     return model

# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x

def to_numeric(x):
    return x.apply(pd.to_numeric, errors='coerce').to_numpy()

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(to_numeric)), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, device: None|str|torch.device = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            model = 'checkpoints/nanotabpfn.pth'
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train))+1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').numpy()


class NanoTabPFNRegressor():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, dist: FullSupportBarDistribution|str|None = None, device: str|torch.device|None = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = 'checkpoints/nanotabpfn_regressor.pth'
            dist = 'checkpoints/nanotabpfn_regressor_buckets.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks
        self.external_gate = 0.5  # weight for blending external attention if available

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_text: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        # MZ: the fit function only stores the training data, no forward pass is done here
        # MZ: treat numerical features same as before
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        # MZ: our text enhanced module
        self.X_text_train = np.array(X_text)

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray, X_text_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        MZ: Pass text embeddings of test data for textenhanced attention calculation.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        # MZ: treat numerical features same as before
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        attn_weight_external = self._compute_attn_weight_external(X_text_test)

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor),
                single_eval_pos=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
                attn_weight_external=attn_weight_external,
                external_gate=self.external_gate,
            ).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
    
    def _compute_attn_weight_external(self, X_text_test: np.ndarray) -> torch.Tensor:
        """
        Computes external attention weights from text features for train+test rows.
        Returns a tensor of shape (1, L, L) on the model device.
        """
        X_text_full = np.concatenate((self.X_text_train, np.array(X_text_test)))
        X_tensor = torch.tensor(X_text_full, dtype=torch.float32, device=self.device)  # [L, F_text]
        X_norm = torch.nn.functional.normalize(X_tensor, dim=1)
        sim = X_norm @ X_norm.T  # [L, L]
        attn = torch.softmax(sim, dim=-1).unsqueeze(0)  # [1, L, L]
        return attn

    def build_fake_attn_weight(self, num_test_rows: int, fill_value: float = 0.0) -> torch.Tensor:
        """
        Utility to create a fake external attention weight when no text is available.
        Produces a uniform (softmaxed) matrix of shape (1, L, L) on the correct device.
        """
        train_len = len(self.X_train)
        total_len = train_len + num_test_rows
        base = torch.full(
            (1, total_len, total_len),
            fill_value,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.softmax(base, dim=-1)
