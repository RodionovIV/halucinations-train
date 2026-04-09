"""
SEP (Semantic Entropy Probes) experiments.

Based on: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
          Kossen et al., arXiv 2406.15927 (Oxford OATML, 2024)

Original method:
  - Extract the hidden-state vector h_l_p ∈ R^d at layer l, token position p
    (SLT = second-last token, or TBG = token before generation).
  - Concatenate vectors from 5 adjacent layers.
  - Train a logistic regression probe to predict binarised semantic entropy.
  - Sweep layers to find the most discriminative region.

This adaptation:
  - Instead of full hidden-state vectors, we use 4 aggregated statistics
    (norm, mean, std, maxabs) per (layer × position) cell.
  - We train probes on each layer, each position, and combinations.
  - We sweep regularisation strength C (analogous to the paper's ablation).
  - Metric: PR-AUC (Average Precision).

Usage:
    python train_sep_probes.py --data final_features.csv --label_col label
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    ALL_LAYERDIFF_FEATURES,
    ALL_LAYER_FEATURES,
    HIDDEN_FIRST,
    HIDDEN_LAST,
    HIDDEN_MEAN,
    HIDDEN_MINLOGPROB,
    LAYER_FEATURES_BY_LAYER,
    LAYERDIFF_FIRST,
    LAYERDIFF_LAST,
    LAYERS,
    POSITIONS,
    UNCERTAINTY_FEATURES,
    META_FEATURES,
    available_cols,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SEP-style per-layer logistic regression probes."
    )
    p.add_argument("--data",      default="final_features.csv")
    p.add_argument("--label_col", default="label")
    p.add_argument("--n_splits",  type=int, default=5)
    p.add_argument("--output",    default="results_sep_probes.csv")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Probe factory and evaluation
# ---------------------------------------------------------------------------

def make_probe(C: float = 1.0, seed: int = 42) -> Pipeline:
    """Logistic Regression probe identical to the SEP paper setup."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, C=C, solver="lbfgs",
            class_weight="balanced", random_state=seed,
        )),
    ])


def eval_probe(probe, X: np.ndarray, y: np.ndarray,
               n_splits: int, seed: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = cross_val_predict(probe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(average_precision_score(y, y_proba))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(f"ERROR: label column '{args.label_col}' not in dataset.")

    y = df[args.label_col].astype(int).values
    print(
        f"Dataset: {len(df):,} samples | "
        f"positive rate: {y.mean():.3%}"
    )

    results = []
    header_printed = set()

    def record(probe_name, cols, layer, position, pr_auc):
        results.append({
            "probe":      probe_name,
            "layer":      layer,
            "position":   position,
            "n_features": len(cols),
            "pr_auc":     round(pr_auc, 5),
        })

    def section(title):
        if title not in header_printed:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
            header_printed.add(title)

    # -----------------------------------------------------------------------
    # 1. Per-layer, per-position probes  (direct SEP layer sweep)
    # -----------------------------------------------------------------------
    section("Per-layer / Per-position Probes  (SEP layer sweep)")
    for layer in LAYERS:
        for pos in POSITIONS:
            cols = available_cols(LAYER_FEATURES_BY_LAYER[layer][pos], df.columns)
            if not cols:
                continue
            X = df[cols].values.astype(np.float32)
            pr = eval_probe(make_probe(seed=args.seed), X, y,
                            args.n_splits, args.seed)
            print(f"  layer{layer}_{pos:<14}  PR-AUC={pr:.4f}")
            record(f"layer{layer}_{pos}", cols, layer, pos, pr)

    # -----------------------------------------------------------------------
    # 2. All-positions probe per layer  (concatenation of all 4 positions)
    # -----------------------------------------------------------------------
    section("All-positions Probe per Layer")
    for layer in LAYERS:
        all_pos_cols = available_cols(
            [c for pos in POSITIONS for c in LAYER_FEATURES_BY_LAYER[layer][pos]],
            df.columns,
        )
        if not all_pos_cols:
            continue
        X = df[all_pos_cols].values.astype(np.float32)
        pr = eval_probe(make_probe(seed=args.seed), X, y, args.n_splits, args.seed)
        print(f"  layer{layer}_all_positions    PR-AUC={pr:.4f}")
        record(f"layer{layer}_all_positions", all_pos_cols, layer, "all", pr)

    # -----------------------------------------------------------------------
    # 3. Inter-layer differences
    # -----------------------------------------------------------------------
    section("Inter-layer Difference Probes")
    for pos, diff_group in [
        ("first",      LAYERDIFF_FIRST),
        ("last",       LAYERDIFF_LAST),
        ("minlogprob", [c for c in ALL_LAYERDIFF_FEATURES if "_minlogprob_" in c]),
        ("mean",       [c for c in ALL_LAYERDIFF_FEATURES if "_mean_" in c]),
        ("all",        ALL_LAYERDIFF_FEATURES),
    ]:
        cols = available_cols(diff_group, df.columns)
        if not cols:
            continue
        X = df[cols].values.astype(np.float32)
        pr = eval_probe(make_probe(seed=args.seed), X, y, args.n_splits, args.seed)
        print(f"  layerdiff_{pos:<14}  PR-AUC={pr:.4f}")
        record(f"layerdiff_{pos}", cols, "diff", pos, pr)

    # -----------------------------------------------------------------------
    # 4. SEP-style: first + last only
    #    (analogous to the paper's SLT position using final-token hidden state)
    # -----------------------------------------------------------------------
    section("SEP Best-Setup: First + Last Positions (all layers)")
    cols_fl = available_cols(
        HIDDEN_FIRST + HIDDEN_LAST + LAYERDIFF_FIRST + LAYERDIFF_LAST,
        df.columns,
    )
    X_fl = df[cols_fl].values.astype(np.float32)
    pr = eval_probe(make_probe(seed=args.seed), X_fl, y, args.n_splits, args.seed)
    print(f"  all_layers_first+last         PR-AUC={pr:.4f}")
    record("all_layers_first+last", cols_fl, "all", "first+last", pr)

    # -----------------------------------------------------------------------
    # 5. Combined probe: all layer + diff features (full SEP vector)
    # -----------------------------------------------------------------------
    section("Combined All-Layer Probe")
    cols_all = available_cols(ALL_LAYER_FEATURES, df.columns)
    X_all = df[cols_all].values.astype(np.float32)
    pr = eval_probe(make_probe(seed=args.seed), X_all, y, args.n_splits, args.seed)
    print(f"  all_layers_combined           PR-AUC={pr:.4f}")
    record("all_layers_combined", cols_all, "all", "all", pr)

    # -----------------------------------------------------------------------
    # 6. SEP + uncertainty features  (best combined)
    # -----------------------------------------------------------------------
    section("SEP + Uncertainty Signal (Logprob / Entropy) Probe")
    cols_combined = available_cols(
        ALL_LAYER_FEATURES + UNCERTAINTY_FEATURES + META_FEATURES, df.columns
    )
    X_comb = df[cols_combined].values.astype(np.float32)
    pr = eval_probe(make_probe(seed=args.seed), X_comb, y, args.n_splits, args.seed)
    print(f"  sep+uncertainty               PR-AUC={pr:.4f}")
    record("sep+uncertainty", cols_combined, "all", "all", pr)

    # -----------------------------------------------------------------------
    # 7. Regularisation sweep (on first+last probe — analogous to paper ablation)
    # -----------------------------------------------------------------------
    section("Regularisation Sweep C  (first+last combined probe)")
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        pr = eval_probe(make_probe(C=C, seed=args.seed), X_fl, y,
                        args.n_splits, args.seed)
        print(f"  C={C:<8.3f}  PR-AUC={pr:.4f}")
        record(f"first+last_C={C}", cols_fl, "all", "first+last", pr)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df_res = (
        pd.DataFrame(results)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(args.output)
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\nTop-10 probes by PR-AUC:")
    print(df_res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
