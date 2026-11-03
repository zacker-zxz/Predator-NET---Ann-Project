"""
Feature Selection Module for PREDATOR NET.

Implements three feature selection methods per Spec §11:
  1. Correlation Filtering: Removes redundant features with |corr| > threshold (default 0.95).
  2. Mutual Information: Ranks features by mutual information with the target label.
  3. Optional Model-Based Importance: Fast tree-based feature ranking.

Enforces Spec Rule §70 #3: Never hard-code the input feature count.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier

logger = logging.getLogger(__name__)


def remove_collinear_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    threshold: float = 0.95,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Identify and remove collinear features exceeding the correlation threshold.

    If labels `y` are provided, when a pair (A, B) exceeds threshold, the feature
    with higher absolute correlation to `y` is kept. Otherwise, the second feature is dropped.

    Args:
        X: Feature DataFrame.
        y: Optional target label Series for informed tie-breaking.
        threshold: Absolute correlation threshold (default 0.95).

    Returns:
        Tuple of (retained_features, dropped_features, correlation_matrix).
    """
    corr_matrix = X.corr(method="pearson").abs()
    cols = list(X.columns)
    dropped = set()

    # Precompute correlation with target if available
    target_corr = {}
    if y is not None:
        for c in cols:
            target_corr[c] = abs(X[c].corr(y))

    for i in range(len(cols)):
        col_i = cols[i]
        if col_i in dropped:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j in dropped:
                continue

            r = corr_matrix.loc[col_i, col_j]
            if not np.isnan(r) and r >= threshold:
                # Decide which one to drop
                if y is not None and col_i in target_corr and col_j in target_corr:
                    if target_corr[col_i] >= target_corr[col_j]:
                        drop_col = col_j
                    else:
                        drop_col = col_i
                else:
                    drop_col = col_j

                dropped.add(drop_col)
                logger.debug(
                    f"Collinear pair ({col_i}, {col_j}) r={r:.4f}. Dropping '{drop_col}'."
                )

    retained = [c for c in cols if c not in dropped]
    logger.info(
        f"Collinear filter (r >= {threshold}): kept {len(retained)}/{len(cols)} features "
        f"(dropped {len(dropped)})."
    )
    return retained, list(dropped), corr_matrix


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int = 50000,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute Mutual Information scores between each feature and the binary target.

    Subsamples data if len(X) > n_samples for computational tractability.

    Args:
        X: Feature DataFrame.
        y: Target label Series.
        n_samples: Maximum samples for MI computation.
        random_state: Random seed.

    Returns:
        pd.Series indexed by feature name with MI scores sorted descending.
    """
    if len(X) > n_samples:
        logger.info(f"Subsampling {n_samples} flows for Mutual Information computation.")
        idx = X.sample(n=n_samples, random_state=random_state).index
        X_sub = X.loc[idx]
        y_sub = y.loc[idx]
    else:
        X_sub = X
        y_sub = y

    # Replace any infinite values or NaNs that might remain
    X_clean = np.nan_to_num(X_sub.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = y_sub.to_numpy(dtype=np.int32)

    mi_scores = mutual_info_classif(
        X_clean,
        y_clean,
        discrete_features=False,
        random_state=random_state,
    )

    mi_series = pd.Series(mi_scores, index=X.columns, name="mutual_info").sort_values(
        ascending=False
    )
    logger.info(f"Computed MI scores for {len(mi_series)} features. Top feature: {mi_series.index[0]} ({mi_series.iloc[0]:.4f})")
    return mi_series


def compute_tree_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int = 50000,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute feature importance using a lightweight ExtraTrees classifier.

    Args:
        X: Feature DataFrame.
        y: Target label Series.
        n_samples: Maximum samples for fitting.
        random_state: Random seed.

    Returns:
        pd.Series with feature importances sorted descending.
    """
    if len(X) > n_samples:
        idx = X.sample(n=n_samples, random_state=random_state).index
        X_sub = X.loc[idx]
        y_sub = y.loc[idx]
    else:
        X_sub = X
        y_sub = y

    X_clean = np.nan_to_num(X_sub.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = y_sub.to_numpy(dtype=np.int32)

    clf = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=12,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_clean, y_clean)

    importance_series = pd.Series(
        clf.feature_importances_, index=X.columns, name="tree_importance"
    ).sort_values(ascending=False)
    return importance_series


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.95,
    mi_threshold: float = 0.001,
    top_k: Optional[int] = None,
    use_tree_importance: bool = False,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Execute full feature selection pipeline:
      1. Collinear feature filtering.
      2. Mutual information ranking.
      3. Optional tree-based importance.
      4. Selection of features exceeding MI threshold or top_k.

    Args:
        X: Input feature DataFrame.
        y: Target label Series.
        correlation_threshold: Threshold to drop redundant features.
        mi_threshold: Minimum mutual information score to keep a feature.
        top_k: Optional maximum number of features to select (if None, keep all above threshold).
        use_tree_importance: Whether to also calculate tree importance.
        random_state: Random seed.

    Returns:
        Tuple of (selected_feature_names, rankings_df, correlation_matrix).
    """
    # Step 1: Remove collinear features
    retained_cols, dropped_cols, corr_matrix = remove_collinear_features(
        X, y=y, threshold=correlation_threshold
    )

    X_filtered = X[retained_cols]

    # Step 2: Compute Mutual Information
    mi_scores = compute_mutual_information(
        X_filtered, y, random_state=random_state
    )

    rankings_df = pd.DataFrame({
        "feature": mi_scores.index,
        "mutual_info": mi_scores.values,
    }).set_index("feature")

    # Step 3: Optional Tree Importance
    if use_tree_importance:
        tree_imp = compute_tree_importance(
            X_filtered, y, random_state=random_state
        )
        rankings_df["tree_importance"] = tree_imp

    # Step 4: Filter by MI threshold
    valid_features = rankings_df[rankings_df["mutual_info"] >= mi_threshold].index.tolist()

    if top_k is not None and len(valid_features) > top_k:
        selected_features = valid_features[:top_k]
    else:
        selected_features = valid_features

    logger.info(
        f"Selected {len(selected_features)} features after collinearity filtering and MI scoring."
    )
    return selected_features, rankings_df.reset_index(), corr_matrix
