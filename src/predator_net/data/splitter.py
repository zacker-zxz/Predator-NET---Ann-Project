"""
Data Splitting Protocols for PREDATOR NET.

Implements three evaluation protocols per Spec §7:
  - PROTOCOL A (Stratified Split): Preserves class balance across train/val/test.
  - PROTOCOL B (Leakage-Aware Split): Groups flows by 5-tuple, connection, or duplicate
    hash to prevent identical connections from leaking across train/val/test splits.
  - PROTOCOL C (Temporal Split): Chronologically partitions traffic by timestamp.
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

logger = logging.getLogger(__name__)


def compute_five_tuple_group(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Construct a five-tuple grouping series if required columns exist in the DataFrame.
    5-tuple: (Source IP, Destination IP, Source Port, Destination Port, Protocol)
    """
    cols = {c.lower(): c for c in df.columns}
    five_tuple_keys = ["source_ip", "destination_ip", "source_port", "destination_port", "protocol"]

    if all(k in cols for k in five_tuple_keys):
        logger.info("Constructing 5-tuple connection identifier for leakage-aware grouping.")
        return (
            df[cols["source_ip"]].astype(str)
            + "_"
            + df[cols["destination_ip"]].astype(str)
            + "_"
            + df[cols["source_port"]].astype(str)
            + "_"
            + df[cols["destination_port"]].astype(str)
            + "_"
            + df[cols["protocol"]].astype(str)
        )
    elif "flow_id" in cols:
        logger.info("Using 'flow_id' for leakage-aware connection grouping.")
        return df[cols["flow_id"]].astype(str)
    return None


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = "binary_label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Protocol A: Standard Stratified Split.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found for stratification.")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Second split: val vs test
    val_share_of_val_test = val_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=(1.0 - val_share_of_val_test),
        stratify=val_test_df[label_col],
        random_state=random_state,
    )

    _log_split_stats("Protocol A (Stratified)", train_df, val_df, test_df, label_col)
    return train_df, val_df, test_df


def leakage_aware_split(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = "binary_label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Protocol B: Leakage-Aware Group Split.
    Guarantees no group (e.g. 5-tuple, flow_id, or duplicate-hash) spans across train/val/test.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    # Determine grouping Series
    if group_col is not None and group_col in df.columns:
        groups = df[group_col]
    else:
        inferred_groups = compute_five_tuple_group(df)
        if inferred_groups is not None:
            groups = inferred_groups
        else:
            logger.warning(
                "No 5-tuple or flow_id found. Generating hash-based duplicate groups for leakage prevention."
            )
            # Hash numerical features to group exact/near duplicates
            feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != label_col]
            groups = df[feature_cols].apply(lambda row: hash(tuple(row[:5])), axis=1)

    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_state)
    train_idx, val_test_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    val_test_df = df.iloc[val_test_idx].copy()
    val_test_groups = groups.iloc[val_test_idx]

    val_share = val_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_share), random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss2.split(val_test_df, groups=val_test_groups))

    val_df = val_test_df.iloc[val_rel_idx].copy()
    test_df = val_test_df.iloc[test_rel_idx].copy()

    # Verify no group overlap
    train_grps = set(groups.iloc[train_idx])
    val_grps = set(val_test_groups.iloc[val_rel_idx])
    test_grps = set(val_test_groups.iloc[test_rel_idx])

    assert len(train_grps.intersection(val_grps)) == 0, "Leakage detected: train & val share groups!"
    assert len(train_grps.intersection(test_grps)) == 0, "Leakage detected: train & test share groups!"
    assert len(val_grps.intersection(test_grps)) == 0, "Leakage detected: val & test share groups!"

    _log_split_stats("Protocol B (Leakage-Aware)", train_df, val_df, test_df, label_col)
    return train_df, val_df, test_df


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = "binary_label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Protocol C: Temporal Chronological Split.
    Partitions traffic by time: earlier -> train, middle -> val, latest -> test.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found for temporal split.")

    sorted_df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    _log_split_stats("Protocol C (Temporal)", train_df, val_df, test_df, label_col)
    return train_df, val_df, test_df


def split_data(
    df: pd.DataFrame,
    protocol: str = "stratified",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = "binary_label",
    group_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Unified entry point for data splitting.
    """
    proto = protocol.lower()
    if proto in ["stratified", "protocol_a", "a"]:
        return stratified_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_col=label_col,
            random_state=random_state,
        )
    elif proto in ["leakage_aware", "protocol_b", "b", "group"]:
        return leakage_aware_split(
            df,
            group_col=group_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_col=label_col,
            random_state=random_state,
        )
    elif proto in ["temporal", "protocol_c", "c"]:
        col = timestamp_col if timestamp_col else "timestamp"
        return temporal_split(
            df,
            timestamp_col=col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            label_col=label_col,
        )
    else:
        raise ValueError(
            f"Unknown split protocol '{protocol}'. Choose 'stratified', 'leakage_aware', or 'temporal'."
        )


def _log_split_stats(
    protocol_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> None:
    """Helper to log split sizes and label distributions."""
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"=== {protocol_name} Split Summary (Total: {total:,} rows) ===")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = (split[label_col] == 1).sum() if label_col in split.columns else 0
        neg = len(split) - pos
        pos_pct = (pos / len(split) * 100) if len(split) > 0 else 0
        logger.info(
            f"  {name:5s}: {len(split):8,d} rows ({len(split)/total*100:5.1f}%) | "
            f"Benign: {neg:7,d} | Attack: {pos:7,d} ({pos_pct:5.2f}%)"
        )
