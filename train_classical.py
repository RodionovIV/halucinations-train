"""
Train classical ML models (LR, RF, Gradient Boosting, XGBoost, LightGBM)
on multiple feature-set combinations for hallucination detection.

Covers both MHAD and SEP feature groups as defined in features.py.
Target metric: PR-AUC (Average Precision).

Usage:
    python train_classical.py --data final_features.csv --label_col label
    python train_classical.py --data final_features.csv --label_col label \
        --feature_set mhad_full --output results_mhad_classical.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_SETS, available_cols

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not found – XGBoost model will be skipped.", file=sys.stderr)

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print(
        "[WARN] lightgbm not found – LightGBM model will be skipped.", file=sys.stderr
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train classical ML models on hallucination detection features."
    )
    p.add_argument("--data", default="final_features.csv", help="Path to CSV dataset")
    p.add_argument(
        "--label_col",
        default="label",
        help="Name of the binary target column (0=ok, 1=hallucination)",
    )
    p.add_argument(
        "--n_splits", type=int, default=5, help="Number of stratified CV folds"
    )
    p.add_argument(
        "--output", default="results_classical.csv", help="Path for the results CSV"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature_set",
        default=None,
        help=(
            "Run only this feature set (or leave empty to run all). "
            f"Choices: {list(FEATURE_SETS.keys())}"
        ),
    )
    p.add_argument(
        "--validate",
        default=None,
        help="Optional path to a validation CSV (same structure as --data). "
        "Models trained on the full training set will be evaluated on it.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------


def build_models(seed: int) -> dict:
    models = {
        "LogReg_L2": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=1,  # will be set per fold; here as placeholder
                        eval_metric="aucpr",
                        n_jobs=-1,
                        random_state=seed,
                        verbosity=0,
                        use_label_encoder=False,
                    ),
                ),
            ]
        )

    if HAS_LGB:
        models["LightGBM"] = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        is_unbalance=True,
                        n_jobs=-1,
                        random_state=seed,
                        verbose=-1,
                    ),
                ),
            ]
        )

    return models


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> dict:
    """5-fold CV → PR-AUC and ROC-AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "pr_auc": average_precision_score(y, y_proba),
        "roc_auc": roc_auc_score(y, y_proba),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(
            f"ERROR: label column '{args.label_col}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    mask = df[args.label_col].isin([0, 1])
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"[INFO] Dropping {n_dropped} rows with label not in {{0, 1}}")
        df = df[mask].reset_index(drop=True)

    y = df[args.label_col].astype(int).values
    print(
        f"Dataset: {len(df):,} samples | positive rate: {y.mean():.3%} "
        f"| positives: {y.sum():,}"
    )

    # Select which feature sets to run
    if args.feature_set:
        if args.feature_set not in FEATURE_SETS:
            sys.exit(
                f"Unknown feature set '{args.feature_set}'. "
                f"Choices: {list(FEATURE_SETS.keys())}"
            )
        sets_to_run = {args.feature_set: FEATURE_SETS[args.feature_set]}
    else:
        sets_to_run = FEATURE_SETS

    # Load validation dataset if provided
    df_val, y_val = None, None
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
        y_val = df_val[args.label_col].astype(int).values
        print(
            f"Validation: {len(df_val):,} samples | positive rate: {y_val.mean():.3%} "
            f"| positives: {y_val.sum():,}"
        )

    models = build_models(args.seed)
    results = []
    val_results = []

    for feat_name, feat_cols in sets_to_run.items():
        avail = available_cols(feat_cols, df.columns)
        if not avail:
            print(f"\n[SKIP] '{feat_name}': no columns found in dataset")
            continue

        X = df[avail].values.astype(np.float32)
        print(f"\n{'=' * 60}")
        print(f"Feature set : {feat_name}  ({len(avail)} features)")
        print(f"{'=' * 60}")

        for model_name, model in models.items():
            try:
                metrics = evaluate_model(model, X, y, args.n_splits, args.seed)
                pr_auc = metrics["pr_auc"]
                roc_auc = metrics["roc_auc"]
                row = {
                    "feature_set": feat_name,
                    "model": model_name,
                    "pr_auc": round(pr_auc, 5),
                    "roc_auc": round(roc_auc, 5),
                    "n_features": len(avail),
                }
                print(
                    f"  {model_name:<22}  PR-AUC={pr_auc:.4f}   ROC-AUC={roc_auc:.4f}"
                )

                if df_val is not None:
                    avail_val = available_cols(feat_cols, df_val.columns)
                    if avail_val:
                        X_val = df_val[avail_val].values.astype(np.float32)
                        model.fit(X, y)
                        y_proba_val = model.predict_proba(X_val)[:, 1]
                        val_pr = average_precision_score(y_val, y_proba_val)
                        val_roc = roc_auc_score(y_val, y_proba_val)
                        print(
                            f"  {'':22}  "
                            f"[val] PR-AUC={val_pr:.4f}   ROC-AUC={val_roc:.4f}"
                        )
                        row["val_pr_auc"] = round(val_pr, 5)
                        row["val_roc_auc"] = round(val_roc, 5)

                results.append(row)
            except Exception as exc:
                print(f"  {model_name:<22}  ERROR: {exc}")

    # Save and summarise
    df_res = (
        pd.DataFrame(results)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(args.output)
    df_res.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print("\nTop-10 configurations by PR-AUC:")
    print(df_res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
