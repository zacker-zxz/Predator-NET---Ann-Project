"""
Feature Analysis and Visualization Artifacts for PREDATOR NET.

Generates and saves:
  - artifacts/features/feature_rankings.csv
  - artifacts/features/selected_features.json
  - artifacts/features/correlation_matrix.png
  - artifacts/features/feature_importance.png
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def save_feature_artifacts(
    rankings_df: pd.DataFrame,
    selected_features: List[str],
    corr_matrix: pd.DataFrame,
    output_dir: Path,
    top_n_plots: int = 25,
) -> None:
    """
    Save feature rankings, selected feature list, correlation matrix heatmap,
    and feature importance bar plot.

    Args:
        rankings_df: DataFrame containing feature rankings and scores.
        selected_features: List of selected feature names.
        corr_matrix: Full or filtered correlation matrix DataFrame.
        output_dir: Path to directory where artifacts will be saved.
        top_n_plots: Number of top features to include in visualization plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save feature rankings CSV
    rankings_csv_path = output_dir / "feature_rankings.csv"
    rankings_df.to_csv(rankings_csv_path, index=False)
    logger.info(f"Saved feature rankings to {rankings_csv_path}")

    # 2. Save selected features JSON
    selected_json_path = output_dir / "selected_features.json"
    with open(selected_json_path, "w") as f:
        json.dump(
            {
                "num_features": len(selected_features),
                "selected_features": selected_features,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved selected features ({len(selected_features)}) to {selected_json_path}")

    # 3. Plot and save correlation matrix heatmap (for top N features or selected)
    corr_plot_path = output_dir / "correlation_matrix.png"
    subset_cols = [c for c in selected_features[:top_n_plots] if c in corr_matrix.columns]
    if len(subset_cols) > 1:
        plt.figure(figsize=(14, 12))
        sub_corr = corr_matrix.loc[subset_cols, subset_cols]
        mask = np.triu(np.ones_like(sub_corr, dtype=bool))
        sns.heatmap(
            sub_corr,
            mask=mask,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(f"Feature Correlation Matrix (Top {len(subset_cols)} Features)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=200)
        plt.close()
        logger.info(f"Saved correlation matrix plot to {corr_plot_path}")

    # 4. Plot and save feature importance bar chart
    importance_plot_path = output_dir / "feature_importance.png"
    top_rankings = rankings_df.head(top_n_plots)
    score_col = "mutual_info" if "mutual_info" in top_rankings.columns else top_rankings.columns[1]

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_rankings,
        x=score_col,
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Top {len(top_rankings)} Features by {score_col.replace('_', ' ').title()}", fontsize=14, pad=15)
    plt.xlabel(score_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved feature importance plot to {importance_plot_path}")
