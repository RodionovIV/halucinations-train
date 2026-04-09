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
    ALL_LAYER_FEATURES,
    ALL_LAYERDIFF_FEATURES,
    HIDDEN_FIRST,
    HIDDEN_LAST,
    LAYER_FEATURES_BY_LAYER,
    LAYERDIFF_FIRST,
    LAYERDIFF_LAST,
    LAYERS,
    META_FEATURES,
    POSITIONS,
    UNCERTAINTY_FEATURES,
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
    p.add_argument("--data", default="final_features.csv")
    p.add_argument("--label_col", default="label")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--output", default="results_sep_probes.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--validate",
        default=None,
        help="Optional path to a validation CSV. Models fitted on the full "
        "training set will be evaluated on it.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Probe factory and evaluation
# ---------------------------------------------------------------------------


def make_probe(C: float = 1.0, seed: int = 42) -> Pipeline:
    """Logistic Regression probe identical to the SEP paper setup."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def eval_probe(probe, X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = cross_val_predict(probe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(average_precision_score(y, y_proba))


def eval_probe_val(
    probe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Fit on full training data and evaluate on held-out validation set."""
    probe.fit(X_train, y_train)
    y_proba_val = probe.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, y_proba_val))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(f"ERROR: label column '{args.label_col}' not in dataset.")

    mask = df[args.label_col].isin([0, 1])
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"[INFO] Dropping {n_dropped} rows with label not in {{0, 1}}")
        df = df[mask].reset_index(drop=True)

    y = df[args.label_col].astype(int).values
    print(f"Dataset: {len(df):,} samples | positive rate: {y.mean():.3%}")

    # Load validation dataset if provided
    df_val, y_val_arr = None, None
    if args.validate:
        print(f"Loading validation set {args.validate} …")
        df_val = pd.read_csv(args.validate)
        if args.label_col not in df_val.columns:
            sys.exit(
                f"ERROR: label column '{args.label_col}' not found in validation dataset."
            )
        val_mask = df_val[args.label_col].isin([0, 1])
        n_val_dropped = (~val_mask).sum()
        if n_val_dropped:
            print(
                f"[INFO] Validation: dropping {n_val_dropped} rows with label not in {{0, 1}}"
            )
            df_val = df_val[val_mask].reset_index(drop=True)
        y_val_arr = df_val[args.label_col].astype(int).values
        print(
            f"Validation: {len(df_val):,} samples | positive rate: {y_val_arr.mean():.3%}"
        )

    results = []
    header_printed = set()

    def record(probe_name, cols, layer, position, pr_auc, val_pr_auc=None):
        row = {
            "probe": probe_name,
            "layer": layer,
            "position": position,
            "n_features": len(cols),
            "pr_auc": round(pr_auc, 5),
        }
        if val_pr_auc is not None:
            row["val_pr_auc"] = round(val_pr_auc, 5)
        results.append(row)

    def section(title):
        if title not in header_printed:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
            header_printed.add(title)

    def run_probe(probe_name, cols, layer, position, X_tr, C=1.0):
        """CV eval + optional validation eval, then record."""
        pr = eval_probe(
            make_probe(C=C, seed=args.seed), X_tr, y, args.n_splits, args.seed
        )
        val_pr = None
        val_suffix = ""
        if df_val is not None:
            avail_val = available_cols(
                cols if isinstance(cols, list) else list(cols), df_val.columns
            )
            if avail_val:
                X_val = df_val[avail_val].values.astype(np.float32)
                val_pr = eval_probe_val(
                    make_probe(C=C, seed=args.seed), X_tr, y, X_val, y_val_arr
                )
                val_suffix = f"   [val] PR-AUC={val_pr:.4f}"
        return pr, val_pr, val_suffix

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
            pr, val_pr, val_suffix = run_probe(
                f"layer{layer}_{pos}", cols, layer, pos, X
            )
            print(f"  layer{layer}_{pos:<14}  PR-AUC={pr:.4f}{val_suffix}")
            record(f"layer{layer}_{pos}", cols, layer, pos, pr, val_pr)

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
        pr, val_pr, val_suffix = run_probe(
            f"layer{layer}_all_positions", all_pos_cols, layer, "all", X
        )
        print(f"  layer{layer}_all_positions    PR-AUC={pr:.4f}{val_suffix}")
        record(f"layer{layer}_all_positions", all_pos_cols, layer, "all", pr, val_pr)

    # -----------------------------------------------------------------------
    # 3. Inter-layer differences
    # -----------------------------------------------------------------------
    section("Inter-layer Difference Probes")
    for pos, diff_group in [
        ("first", LAYERDIFF_FIRST),
        ("last", LAYERDIFF_LAST),
        ("minlogprob", [c for c in ALL_LAYERDIFF_FEATURES if "_minlogprob_" in c]),
        ("mean", [c for c in ALL_LAYERDIFF_FEATURES if "_mean_" in c]),
        ("all", ALL_LAYERDIFF_FEATURES),
    ]:
        cols = available_cols(diff_group, df.columns)
        if not cols:
            continue
        X = df[cols].values.astype(np.float32)
        pr, val_pr, val_suffix = run_probe(f"layerdiff_{pos}", cols, "diff", pos, X)
        print(f"  layerdiff_{pos:<14}  PR-AUC={pr:.4f}{val_suffix}")
        record(f"layerdiff_{pos}", cols, "diff", pos, pr, val_pr)

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
    pr, val_pr, val_suffix = run_probe(
        "all_layers_first+last", cols_fl, "all", "first+last", X_fl
    )
    print(f"  all_layers_first+last         PR-AUC={pr:.4f}{val_suffix}")
    record("all_layers_first+last", cols_fl, "all", "first+last", pr, val_pr)

    # -----------------------------------------------------------------------
    # 5. Combined probe: all layer + diff features (full SEP vector)
    # -----------------------------------------------------------------------
    section("Combined All-Layer Probe")
    cols_all = available_cols(ALL_LAYER_FEATURES, df.columns)
    X_all = df[cols_all].values.astype(np.float32)
    pr, val_pr, val_suffix = run_probe(
        "all_layers_combined", cols_all, "all", "all", X_all
    )
    print(f"  all_layers_combined           PR-AUC={pr:.4f}{val_suffix}")
    record("all_layers_combined", cols_all, "all", "all", pr, val_pr)

    # -----------------------------------------------------------------------
    # 6. SEP + uncertainty features  (best combined)
    # -----------------------------------------------------------------------
    section("SEP + Uncertainty Signal (Logprob / Entropy) Probe")
    cols_combined = available_cols(
        ALL_LAYER_FEATURES + UNCERTAINTY_FEATURES + META_FEATURES, df.columns
    )
    X_comb = df[cols_combined].values.astype(np.float32)
    pr, val_pr, val_suffix = run_probe(
        "sep+uncertainty", cols_combined, "all", "all", X_comb
    )
    print(f"  sep+uncertainty               PR-AUC={pr:.4f}{val_suffix}")
    record("sep+uncertainty", cols_combined, "all", "all", pr, val_pr)

    # -----------------------------------------------------------------------
    # 7. Regularisation sweep (on first+last probe — analogous to paper ablation)
    # -----------------------------------------------------------------------
    section("Regularisation Sweep C  (first+last combined probe)")
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        pr, val_pr, val_suffix = run_probe(
            f"first+last_C={C}", cols_fl, "all", "first+last", X_fl, C=C
        )
        print(f"  C={C:<8.3f}  PR-AUC={pr:.4f}{val_suffix}")
        record(f"first+last_C={C}", cols_fl, "all", "first+last", pr, val_pr)

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
