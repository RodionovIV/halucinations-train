"""
MHAD (Detecting Hallucination through Deep Internal Representation Analysis)
experiments with MLP-based classifiers.

Based on: "Detecting Hallucination in Large Language Models Through
           Deep Internal Representation Analysis"
           Zhang et al., IJCAI 2025

Original method:
  1. Use linear probes to rank neurons / layers by hallucination discriminability.
  2. Concatenate hidden states at the FIRST and LAST generated tokens.
  3. Apply a small MLP binary classifier on the resulting "hallucination awareness vector".

This adaptation:
  - Step 1 replicated via per-column permutation importance from a shallow LR.
  - Steps 2–3 use the 4-stat summaries (norm, mean, std, maxabs) available in the CSV.
  - MLP runs on multiple feature-set candidates with hyperparameter sweeps.
  - Metric: PR-AUC.

Usage:
    python train_mhad_mlp.py --data final_features.csv --label_col label
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from features import (
    ALL_LAYER_FEATURES,
    FEATURE_SETS,
    HIDDEN_FIRST,
    HIDDEN_LAST,
    LAYERDIFF_FIRST,
    LAYERDIFF_LAST,
    META_FEATURES,
    UNCERTAINTY_FEATURES,
    available_cols,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MHAD-style MLP probes for hallucination detection."
    )
    p.add_argument("--data",      default="final_features.csv")
    p.add_argument("--label_col", default="label")
    p.add_argument("--n_splits",  type=int, default=5)
    p.add_argument("--epochs",    type=int, default=50,
                   help="Training epochs per fold per config")
    p.add_argument("--batch",     type=int, default=256)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--top_k_neurons", type=int, default=32,
                   help="Keep top-k features after linear-probe neuron selection")
    p.add_argument("--output",    default="results_mhad_mlp.csv")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# MLP architectures
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Small MLP as used in MHAD."""

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 dropout: float = 0.3):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Neuron selection  (MHAD Step-1 adaptation)
# ---------------------------------------------------------------------------

def select_top_k_features(X: np.ndarray, y: np.ndarray,
                           top_k: int, seed: int) -> np.ndarray:
    """
    Rank features by |coef| from a logistic regression trained on the full data
    (simulates MHAD's linear-probe neuron importance ranking).
    Returns indices of the top-k most discriminative features.
    """
    imp = SimpleImputer(strategy="median").fit(X)
    X_imp = imp.transform(X)
    scl = StandardScaler().fit(X_imp)
    X_sc = scl.transform(X_imp)

    lr = LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=seed,
    )
    lr.fit(X_sc, y)
    importance = np.abs(lr.coef_[0])
    return np.argsort(importance)[::-1][:top_k]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_eval_mlp(
    X: np.ndarray, y: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    epochs: int, batch_size: int, lr: float,
    n_splits: int, seed: int, device: torch.device,
) -> float:
    """Stratified-CV training; returns mean PR-AUC."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores = []

    pos_weight = torch.tensor(
        [(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Impute + scale on training fold only
        imp = SimpleImputer(strategy="median").fit(X_tr)
        scl = StandardScaler().fit(imp.transform(X_tr))

        X_tr_t  = torch.tensor(scl.transform(imp.transform(X_tr)),
                               dtype=torch.float32, device=device)
        X_val_t = torch.tensor(scl.transform(imp.transform(X_val)),
                               dtype=torch.float32, device=device)
        y_tr_t  = torch.tensor(y_tr, dtype=torch.float32, device=device)

        model = MLP(X.shape[1], hidden_dims, dropout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.1
        )

        # Training
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), batch_size):
                idx_b = perm[i : i + batch_size]
                logits = model(X_tr_t[idx_b])
                loss   = criterion(logits, y_tr_t[idx_b])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t).cpu().numpy()
        proba_val = 1.0 / (1.0 + np.exp(-logits_val))   # sigmoid
        score = average_precision_score(y_val, proba_val)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(f"ERROR: label column '{args.label_col}' not found.")
    y = df[args.label_col].astype(int).values

    print(
        f"Dataset: {len(df):,} samples | "
        f"positive rate: {y.mean():.3%}"
    )

    # -----------------------------------------------------------------------
    # Feature-set definitions  (mirroring MHAD architecture intent)
    # -----------------------------------------------------------------------
    feature_sets_mhad = {
        # Core MHAD: first-token + last-token hidden states (the "awareness vector")
        "mhad_first+last":      HIDDEN_FIRST + HIDDEN_LAST + LAYERDIFF_FIRST + LAYERDIFF_LAST,
        # MHAD + logprob uncertainty (our dataset's rich uncertainty side)
        "mhad_full":            HIDDEN_FIRST + HIDDEN_LAST + LAYERDIFF_FIRST + LAYERDIFF_LAST
                                + UNCERTAINTY_FEATURES + META_FEATURES,
        # All layer features (comprehensive)
        "sep_all_layers":       ALL_LAYER_FEATURES,
        # Combined everything
        "all_features":         ALL_LAYER_FEATURES + UNCERTAINTY_FEATURES + META_FEATURES,
    }

    # Architecture sweep
    arch_variants = {
        "shallow_32":     ([32], 0.2),
        "medium_128_64":  ([128, 64], 0.3),
        "deep_256_128_64":([256, 128, 64], 0.3),
        "wide_512_256":   ([512, 256], 0.4),
    }

    results = []

    for feat_name, feat_cols in feature_sets_mhad.items():
        avail = available_cols(feat_cols, df.columns)
        if not avail:
            print(f"\n[SKIP] {feat_name}: no columns found")
            continue

        X_full = df[avail].values.astype(np.float32)

        print(f"\n{'='*60}")
        print(f"Feature set: {feat_name}  ({len(avail)} features)")
        print(f"{'='*60}")

        # ---- MHAD Step-1: top-k neuron selection ----------------------------
        top_k = min(args.top_k_neurons, X_full.shape[1])
        top_idx = select_top_k_features(X_full, y, top_k, args.seed)
        X_topk  = X_full[:, top_idx]

        for arch_name, (hidden_dims, dropout) in arch_variants.items():
            for use_topk, X_in, suffix in [
                (False, X_full,  ""),
                (True,  X_topk, f"_top{top_k}"),
            ]:
                config_name = f"{feat_name} | {arch_name}{suffix}"
                pr_auc = train_eval_mlp(
                    X_in, y,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    lr=args.lr,
                    n_splits=args.n_splits,
                    seed=args.seed,
                    device=device,
                )
                print(
                    f"  {arch_name}{suffix:<12}  "
                    f"PR-AUC={pr_auc:.4f}"
                )
                results.append({
                    "feature_set":  feat_name,
                    "architecture": arch_name + suffix,
                    "n_features":   X_in.shape[1],
                    "pr_auc":       round(pr_auc, 5),
                })

    df_res = (
        pd.DataFrame(results)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(args.output)
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\nTop-10 by PR-AUC:")
    print(df_res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
