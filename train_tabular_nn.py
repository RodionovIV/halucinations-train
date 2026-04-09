"""
Tabular Neural Networks for hallucination detection.

Implements several modern architectures designed for tabular data:
  - TabMLP         – Residual MLP with skip connections
  - TabTransformer – Attention over feature embeddings
  - AutoInt        – Multi-head self-attention for feature interaction learning

All models trained with:
  - Stratified K-Fold CV
  - BCE loss with positive class weight
  - AdamW + cosine LR scheduler
  - Metric: PR-AUC (Average Precision)

Usage:
    python train_tabular_nn.py --data final_features.csv --label_col label
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from features import FEATURE_SETS, available_cols

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tabular neural network experiments for hallucination detection."
    )
    p.add_argument("--data",       default="final_features.csv")
    p.add_argument("--label_col",  default="label")
    p.add_argument("--n_splits",   type=int,   default=5)
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch",      type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output",     default="results_tabular_nn.csv")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="auto")
    p.add_argument(
        "--feature_set", default="all_features",
        help=f"Feature set key from features.py. Choices: {list(FEATURE_SETS.keys())}",
    )
    p.add_argument(
        "--validate", default=None,
        help="Optional path to a validation CSV. Models trained on the full "
             "training set will be evaluated on it."
    )
    return p.parse_args()


def get_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():    return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabMLP(nn.Module):
    """Deep Residual MLP for tabular data."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_residual: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_residual)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


class FeatureTokenizer(nn.Module):
    """Project each scalar feature to a d-dimensional embedding."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # Each feature gets its own linear projection: scalar → d_model
        self.weights = nn.Parameter(torch.empty(n_features, d_model))
        self.biases  = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  (B, F, d_model)
        return x.unsqueeze(-1) * self.weights + self.biases


class TabTransformer(nn.Module):
    """
    TabTransformer: each numerical feature is first tokenised, then processed
    with multi-head self-attention before MLP head.
    (Huang et al., 2020 — adapted for regression-free tabular use)
    """

    def __init__(self, input_dim: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 mlp_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(),                                  # (B, F*d_model)
            nn.LayerNorm(input_dim * d_model),
            nn.Linear(input_dim * d_model, mlp_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)          # (B, F, d_model)
        h      = self.transformer(tokens)   # (B, F, d_model)
        return self.head(h).squeeze(-1)


class AutoInt(nn.Module):
    """
    AutoInt: interacting features with multi-head self-attention (Song et al., 2019).
    Concatenates attention output with a residual shortcut from raw features.
    """

    def __init__(self, input_dim: int, d_model: int = 32,
                 n_heads: int = 2, n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim * d_model, bias=False)
        self.d_model    = d_model
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        # Each feature gets a d_model embedding via linear proj
        h = self.input_proj(x).view(B, F, self.d_model)   # (B, F, d_model)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_eval(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    n_splits: int,
    seed: int,
    device: torch.device,
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos_weight = torch.tensor(
        [(y == 0).sum() / max(1, (y == 1).sum())],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        imp = SimpleImputer(strategy="median").fit(X_tr)
        scl = StandardScaler().fit(imp.transform(X_tr))

        X_tr_t  = torch.tensor(scl.transform(imp.transform(X_tr)),
                               dtype=torch.float32, device=device)
        X_val_t = torch.tensor(scl.transform(imp.transform(X_val)),
                               dtype=torch.float32, device=device)
        y_tr_t  = torch.tensor(y_tr, dtype=torch.float32, device=device)

        model = model_cls(input_dim=X.shape[1], **model_kwargs).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.05)

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), batch_size):
                b     = perm[i : i + batch_size]
                loss  = criterion(model(X_tr_t[b]), y_tr_t[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t).cpu().numpy()
        proba_val = 1.0 / (1.0 + np.exp(-logits_val))
        fold_scores.append(average_precision_score(y_val, proba_val))

    return float(np.mean(fold_scores))


def _val_predict(
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> float:
    """Train on full training data and return PR-AUC on the held-out val set."""
    torch.manual_seed(seed)
    imp = SimpleImputer(strategy="median").fit(X_train)
    scl = StandardScaler().fit(imp.transform(X_train))

    X_tr_t  = torch.tensor(scl.transform(imp.transform(X_train)),
                           dtype=torch.float32, device=device)
    X_val_t = torch.tensor(scl.transform(imp.transform(X_val)),
                           dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_train, dtype=torch.float32, device=device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max(1, (y_train == 1).sum())],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = model_cls(input_dim=X_train.shape[1], **model_kwargs).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(X_tr_t), batch_size):
            b    = perm[i : i + batch_size]
            loss = criterion(model(X_tr_t[b]), y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_t).cpu().numpy()
    proba_val = 1.0 / (1.0 + np.exp(-logits_val))
    return float(average_precision_score(y_val, proba_val))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(f"ERROR: label column '{args.label_col}' not found.")

    mask = df[args.label_col].isin([0, 1])
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"[INFO] Dropping {n_dropped} rows with label not in {{0, 1}}")
        df = df[mask].reset_index(drop=True)

    y = df[args.label_col].astype(int).values

    if args.feature_set not in FEATURE_SETS:
        sys.exit(f"Unknown feature set. Choices: {list(FEATURE_SETS.keys())}")
    feat_cols = FEATURE_SETS[args.feature_set]
    avail     = available_cols(feat_cols, df.columns)
    if not avail:
        sys.exit(f"No feature columns found in dataset for set '{args.feature_set}'.")

    X = df[avail].values.astype(np.float32)
    print(
        f"Feature set  : {args.feature_set}  ({len(avail)} features)\n"
        f"Dataset      : {len(df):,} samples | pos rate {y.mean():.3%}"
    )

    # Load validation dataset if provided
    X_val_ext, y_val_ext = None, None
    if args.validate:
        print(f"Loading validation set {args.validate} …")
        df_val = pd.read_csv(args.validate)
        if args.label_col not in df_val.columns:
            sys.exit(f"ERROR: label column '{args.label_col}' not found in validation dataset.")
        val_mask = df_val[args.label_col].isin([0, 1])
        n_val_dropped = (~val_mask).sum()
        if n_val_dropped:
            print(f"[INFO] Validation: dropping {n_val_dropped} rows with label not in {{0, 1}}")
            df_val = df_val[val_mask].reset_index(drop=True)
        y_val_ext = df_val[args.label_col].astype(int).values
        avail_val = available_cols(feat_cols, df_val.columns)
        if avail_val:
            X_val_ext = df_val[avail_val].values.astype(np.float32)
        print(
            f"Validation: {len(df_val):,} samples | positive rate: {y_val_ext.mean():.3%}"
        )

    # ---- Model catalogue --------------------------------------------------
    # Limit TabTransformer / AutoInt feature dim to avoid GPU OOM
    MAX_FEATS_ATTN = 64   # attention over too many features is slow and noisy

    models_to_run = [
        ("TabMLP_32_4res",       TabMLP,       {"hidden_dim": 32,  "n_residual": 4}),
        ("TabMLP_128_4res",      TabMLP,       {"hidden_dim": 128, "n_residual": 4}),
        ("TabMLP_256_6res",      TabMLP,       {"hidden_dim": 256, "n_residual": 6}),
    ]

    if len(avail) <= MAX_FEATS_ATTN:
        models_to_run += [
            ("TabTransformer_d64_h4", TabTransformer, {"d_model": 64, "n_heads": 4, "n_layers": 2}),
            ("AutoInt_d32_h2",        AutoInt,        {"d_model": 32, "n_heads": 2, "n_layers": 3}),
        ]
    else:
        print(
            f"[INFO] Skipping attention-based models "
            f"({len(avail)} features > MAX_FEATS_ATTN={MAX_FEATS_ATTN}). "
            "Re-run with a smaller feature set if desired."
        )

    # ---- Training loop ----------------------------------------------------
    results = []
    print(f"\n{'='*60}")
    for model_name, model_cls, model_kwargs in models_to_run:
        pr_auc = train_eval(
            model_cls, model_kwargs, X, y,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            n_splits=args.n_splits, seed=args.seed, device=device,
        )
        val_suffix = ""
        row = {
            "model":        model_name,
            "feature_set":  args.feature_set,
            "n_features":   len(avail),
            "pr_auc":       round(pr_auc, 5),
        }
        if X_val_ext is not None:
            val_pr = _val_predict(
                model_cls, model_kwargs, X, y, X_val_ext, y_val_ext,
                epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                seed=args.seed, device=device,
            )
            val_suffix = f"   [val] PR-AUC={val_pr:.4f}"
            row["val_pr_auc"] = round(val_pr, 5)
        print(f"  {model_name:<30}  PR-AUC={pr_auc:.4f}{val_suffix}")
        results.append(row)

    df_res = (
        pd.DataFrame(results)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(args.output)
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
