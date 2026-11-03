"""
CIC-IDS2017 Data Quality Analyzer and Cleaner.

Spec §6: Detect NaN, ±Inf, duplicated rows, constant columns,
report class distribution and data types. Save a data quality report.

Spec §9: Do NOT casually remove outliers — in cybersecurity data,
anomalous values can be the actual signal representing attacks.

Every transformation is logged. The cleaner does NOT silently alter data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_data_quality(
    df: pd.DataFrame,
    label_column: str = "label",
) -> dict[str, Any]:
    """
    Perform comprehensive data quality analysis on the loaded dataset.

    This does NOT modify the DataFrame. It only reports issues.

    Args:
        df: The loaded CIC-IDS2017 DataFrame.
        label_column: Name of the label column.

    Returns:
        Dictionary containing all quality metrics.
    """
    report: dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
    }

    logger.info("=" * 60)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Data types
    # -----------------------------------------------------------------------
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    report["data_types"] = dtypes

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    report["numeric_columns"] = numeric_cols
    report["non_numeric_columns"] = non_numeric_cols
    report["num_numeric"] = len(numeric_cols)
    report["num_non_numeric"] = len(non_numeric_cols)

    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Non-numeric columns: {len(non_numeric_cols)} → {non_numeric_cols}")

    # -----------------------------------------------------------------------
    # 2. Missing values (NaN)
    # -----------------------------------------------------------------------
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    report["missing_values"] = {
        "total_missing_cells": int(nan_counts.sum()),
        "columns_with_missing": {
            col: int(count) for col, count in nan_cols.items()
        },
        "num_columns_with_missing": len(nan_cols),
    }

    if len(nan_cols) > 0:
        logger.warning(
            f"Missing values found in {len(nan_cols)} column(s), "
            f"total missing cells: {nan_counts.sum()}"
        )
        for col, count in nan_cols.items():
            logger.warning(f"  {col}: {count} missing ({count/len(df)*100:.2f}%)")
    else:
        logger.info("No missing values detected")

    # -----------------------------------------------------------------------
    # 3. Infinite values
    # -----------------------------------------------------------------------
    inf_report: dict[str, dict[str, int]] = {}
    total_inf = 0

    for col in numeric_cols:
        pos_inf = int(np.isinf(df[col]).sum()) if df[col].dtype != object else 0
        neg_inf = 0
        if df[col].dtype != object:
            pos_inf_count = int((df[col] == np.inf).sum())
            neg_inf_count = int((df[col] == -np.inf).sum())
            if pos_inf_count > 0 or neg_inf_count > 0:
                inf_report[col] = {
                    "positive_infinity": pos_inf_count,
                    "negative_infinity": neg_inf_count,
                }
                total_inf += pos_inf_count + neg_inf_count

    report["infinite_values"] = {
        "total_infinite_cells": total_inf,
        "columns_with_infinity": inf_report,
        "num_columns_with_infinity": len(inf_report),
    }

    if total_inf > 0:
        logger.warning(
            f"Infinite values found in {len(inf_report)} column(s), "
            f"total: {total_inf}"
        )
        for col, counts in inf_report.items():
            logger.warning(
                f"  {col}: +Inf={counts['positive_infinity']}, "
                f"-Inf={counts['negative_infinity']}"
            )
    else:
        logger.info("No infinite values detected")

    # -----------------------------------------------------------------------
    # 4. Duplicate rows
    # -----------------------------------------------------------------------
    num_duplicates = int(df.duplicated().sum())
    report["duplicate_rows"] = {
        "count": num_duplicates,
        "percentage": round(num_duplicates / len(df) * 100, 4) if len(df) > 0 else 0,
    }

    if num_duplicates > 0:
        logger.warning(
            f"Duplicate rows: {num_duplicates} "
            f"({num_duplicates/len(df)*100:.2f}%)"
        )
    else:
        logger.info("No duplicate rows detected")

    # -----------------------------------------------------------------------
    # 5. Constant columns (zero variance)
    # -----------------------------------------------------------------------
    constant_cols = []
    for col in numeric_cols:
        # Drop NaN/Inf for variance calculation
        clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) == 0 or clean.nunique() <= 1:
            constant_cols.append(col)

    # Also check non-numeric columns
    for col in non_numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    report["constant_columns"] = {
        "columns": constant_cols,
        "count": len(constant_cols),
    }

    if constant_cols:
        logger.warning(f"Constant/near-constant columns: {constant_cols}")
    else:
        logger.info("No constant columns detected")

    # -----------------------------------------------------------------------
    # 6. Label distribution
    # -----------------------------------------------------------------------
    if label_column in df.columns:
        label_dist = df[label_column].value_counts()
        label_report = {
            str(label): int(count)
            for label, count in label_dist.items()
        }
        report["label_distribution"] = {
            "column": label_column,
            "classes": label_report,
            "num_classes": len(label_report),
        }

        logger.info(f"Label distribution ({label_column}):")
        for label, count in label_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {label}: {count} ({pct:.2f}%)")
    else:
        report["label_distribution"] = {
            "column": label_column,
            "error": f"Column '{label_column}' not found",
        }
        logger.error(f"Label column '{label_column}' not found in DataFrame")

    # -----------------------------------------------------------------------
    # 7. Basic statistics for numeric columns
    # -----------------------------------------------------------------------
    try:
        stats = df[numeric_cols].describe().to_dict()
        # Convert numpy types to Python types for JSON serialization
        clean_stats = {}
        for col, col_stats in stats.items():
            clean_stats[col] = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in col_stats.items()
                if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
            }
        report["numeric_statistics"] = clean_stats
    except Exception as e:
        logger.warning(f"Could not compute numeric statistics: {e}")
        report["numeric_statistics"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("DATA QUALITY ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return report


def clean_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    remove_duplicates: bool = True,
    handle_infinity: str = "nan",
    handle_nan: str = "drop",
    remove_constant_cols: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply cleaning transformations to the dataset.

    Every transformation is logged. The original DataFrame is NOT modified
    in-place; a new cleaned DataFrame is returned.

    Args:
        df: Raw DataFrame.
        label_column: Name of the label column.
        remove_duplicates: Whether to remove exact duplicate rows.
        handle_infinity: How to handle ±Inf. Options: 'nan' (convert to NaN),
                         'clip' (clip to float max), 'drop' (drop rows).
        handle_nan: How to handle NaN after infinity handling.
                    Options: 'drop' (drop rows), 'median' (fill with median).
        remove_constant_cols: Whether to remove zero-variance columns.

    Returns:
        Tuple of (cleaned DataFrame, cleaning_log dict).
    """
    cleaning_log: dict[str, Any] = {
        "initial_shape": list(df.shape),
        "transformations": [],
    }

    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()

    # -----------------------------------------------------------------------
    # Step 1: Handle infinite values
    # -----------------------------------------------------------------------
    inf_mask = cleaned[numeric_cols].isin([np.inf, -np.inf])
    total_inf = int(inf_mask.sum().sum())

    if total_inf > 0:
        if handle_infinity == "nan":
            cleaned[numeric_cols] = cleaned[numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            )
            msg = f"Replaced {total_inf} infinite values with NaN"
        elif handle_infinity == "clip":
            for col in numeric_cols:
                max_val = np.finfo(np.float32).max
                cleaned[col] = cleaned[col].clip(-max_val, max_val)
            msg = f"Clipped {total_inf} infinite values to float32 range"
        elif handle_infinity == "drop":
            rows_before = len(cleaned)
            mask = ~inf_mask.any(axis=1)
            cleaned = cleaned[mask].reset_index(drop=True)
            msg = f"Dropped {rows_before - len(cleaned)} rows with infinite values"
        else:
            raise ValueError(f"Unknown infinity handling: {handle_infinity}")

        logger.info(f"  [CLEAN] {msg}")
        cleaning_log["transformations"].append(msg)

    # -----------------------------------------------------------------------
    # Step 2: Handle NaN values
    # -----------------------------------------------------------------------
    nan_count = int(cleaned.isna().sum().sum())
    if nan_count > 0:
        if handle_nan == "drop":
            rows_before = len(cleaned)
            cleaned = cleaned.dropna().reset_index(drop=True)
            dropped = rows_before - len(cleaned)
            msg = f"Dropped {dropped} rows with NaN values (from {rows_before} to {len(cleaned)})"
        elif handle_nan == "median":
            for col in numeric_cols:
                if cleaned[col].isna().any():
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val)
            msg = f"Filled NaN in numeric columns with column medians ({nan_count} cells)"
        else:
            raise ValueError(f"Unknown NaN handling: {handle_nan}")

        logger.info(f"  [CLEAN] {msg}")
        cleaning_log["transformations"].append(msg)

    # -----------------------------------------------------------------------
    # Step 3: Remove duplicate rows
    # -----------------------------------------------------------------------
    if remove_duplicates:
        rows_before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        removed = rows_before - len(cleaned)
        if removed > 0:
            msg = f"Removed {removed} duplicate rows (from {rows_before} to {len(cleaned)})"
            logger.info(f"  [CLEAN] {msg}")
            cleaning_log["transformations"].append(msg)
        else:
            logger.info("  [CLEAN] No duplicate rows to remove")

    # -----------------------------------------------------------------------
    # Step 4: Remove constant columns
    # -----------------------------------------------------------------------
    if remove_constant_cols:
        constant_cols = []
        for col in cleaned.columns:
            if col == label_column:
                continue  # Never remove the label column
            if cleaned[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)
            msg = f"Removed {len(constant_cols)} constant column(s): {constant_cols}"
            logger.info(f"  [CLEAN] {msg}")
            cleaning_log["transformations"].append(msg)
            cleaning_log["removed_constant_columns"] = constant_cols

    # -----------------------------------------------------------------------
    # Step 5: Create binary label
    # -----------------------------------------------------------------------
    if label_column in cleaned.columns:
        unique_labels = cleaned[label_column].unique()
        logger.info(f"  Unique labels: {unique_labels}")

        # Map to binary: BENIGN → 0, everything else → 1 (ATTACK)
        benign_variants = {"benign", "normal", "Benign", "Normal", "BENIGN", "NORMAL"}
        cleaned["binary_label"] = cleaned[label_column].apply(
            lambda x: 0 if str(x).strip() in benign_variants else 1
        )

        benign_count = int((cleaned["binary_label"] == 0).sum())
        attack_count = int((cleaned["binary_label"] == 1).sum())
        msg = (
            f"Created binary_label: BENIGN={benign_count}, "
            f"ATTACK={attack_count} "
            f"({attack_count/(benign_count+attack_count)*100:.2f}% attack)"
        )
        logger.info(f"  [CLEAN] {msg}")
        cleaning_log["transformations"].append(msg)
        cleaning_log["binary_distribution"] = {
            "benign": benign_count,
            "attack": attack_count,
        }

    cleaning_log["final_shape"] = list(cleaned.shape)
    cleaning_log["rows_removed"] = (
        cleaning_log["initial_shape"][0] - cleaning_log["final_shape"][0]
    )

    logger.info(
        f"  Cleaning complete: {cleaning_log['initial_shape']} → "
        f"{cleaning_log['final_shape']}"
    )

    return cleaned, cleaning_log


def save_quality_report(
    report: dict[str, Any],
    cleaning_log: dict[str, Any],
    file_info: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Save the complete data quality report as JSON.

    Args:
        report: Quality analysis results from analyze_data_quality().
        cleaning_log: Cleaning transformation log from clean_dataset().
        file_info: Per-file loading info from loader.
        output_path: Path to save the JSON report.

    Returns:
        Path to the saved report file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_report = {
        "dataset": "CIC-IDS2017",
        "files": file_info,
        "quality_analysis": report,
        "cleaning_log": cleaning_log,
        "preprocessing_decisions": {
            "infinity_handling": "Replaced with NaN, then dropped rows",
            "nan_handling": "Dropped rows with remaining NaN",
            "duplicates": "Removed exact duplicate rows",
            "constant_columns": "Removed zero-variance columns",
            "outlier_policy": (
                "NOT removed — in cybersecurity data, anomalous values "
                "can be the actual signal representing attacks"
            ),
            "label_encoding": "BENIGN → 0, all attack types → 1",
        },
    }

    # Ensure JSON-serializable (handle numpy types)
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    def _deep_convert(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_convert(v) for v in d]
        return _convert(d)

    full_report = _deep_convert(full_report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)

    logger.info(f"Data quality report saved to {output_path}")
    return output_path
