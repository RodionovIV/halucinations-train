"""
Aggregate results from all experiment scripts and produce a summary report.

Usage:
    python summarise_results.py
    python summarise_results.py --results_dir . --output summary.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=".",
                   help="Directory containing *results*.csv files")
    p.add_argument("--output",      default="summary_results.png")
    return p.parse_args()


def load_all(results_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(results_dir.glob("results_*.csv")):
        df = pd.read_csv(csv_path)
        if "pr_auc" not in df.columns:
            continue
        # Normalise: add an 'experiment' column from filename
        name = csv_path.stem.replace("results_", "")
        df["experiment"] = name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No results_*.csv files found in {results_dir}. "
            "Run the training scripts first."
        )
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    df = load_all(results_dir)

    # -----------------------------------------------------------------------
    # Compute a short description for each row
    # -----------------------------------------------------------------------
    label_parts = []
    for _, row in df.iterrows():
        parts = [row.get("experiment", "")]
        for col in ("model", "feature_set", "architecture", "probe"):
            val = row.get(col, None)
            if pd.notna(val) and val:
                parts.append(str(val))
        label_parts.append(" | ".join(parts[:3]))
    df["label"] = label_parts

    # -----------------------------------------------------------------------
    # Top-20 overall
    # -----------------------------------------------------------------------
    top = (
        df.sort_values("pr_auc", ascending=False)
        .drop_duplicates("label")
        .head(20)
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("TOP-20 CONFIGURATIONS (by PR-AUC)")
    print("=" * 70)
    show_cols = [c for c in ["experiment", "model", "feature_set",
                              "architecture", "probe", "pr_auc", "roc_auc"]
                 if c in top.columns]
    print(top[show_cols].to_string(index=False))

    # -----------------------------------------------------------------------
    # Best per experiment
    # -----------------------------------------------------------------------
    best = (
        df.sort_values("pr_auc", ascending=False)
        .groupby("experiment", as_index=False)
        .first()
        .sort_values("pr_auc", ascending=False)
    )
    print("\n" + "=" * 70)
    print("BEST per EXPERIMENT")
    print("=" * 70)
    print(best[show_cols].to_string(index=False))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Hallucination Detection – PR-AUC Summary", fontsize=14)

    # Panel 1: top-20 bar chart
    ax1 = axes[0]
    sns.barplot(
        data=top, y="label", x="pr_auc", palette="viridis",
        orient="h", ax=ax1,
    )
    ax1.set_xlabel("PR-AUC")
    ax1.set_ylabel("")
    ax1.set_title("Top-20 Configurations")
    ax1.axvline(x=df["pr_auc"].mean(), color="red", linestyle="--",
                label=f"Mean={df['pr_auc'].mean():.3f}")
    ax1.legend(fontsize=9)

    # Panel 2: best per experiment
    ax2 = axes[1]
    sns.barplot(
        data=best, y="experiment", x="pr_auc", palette="rocket",
        orient="h", ax=ax2,
    )
    ax2.set_xlabel("PR-AUC")
    ax2.set_ylabel("")
    ax2.set_title("Best per Experiment Group")

    # Annotate bars with values
    for ax in axes:
        for p in ax.patches:
            val = p.get_width()
            if not pd.isna(val) and val > 0:
                ax.annotate(
                    f"{val:.3f}",
                    (val + 0.003, p.get_y() + p.get_height() / 2),
                    va="center", fontsize=8,
                )

    plt.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
