"""
Feature Engineering for PREDATOR NET.

Supports two feature extraction modes per Spec §5 and §10:
  - MODE A (Behavioral): Flow statistics, durations, packet counts, byte counts,
    inter-arrival times (IAT), TCP flags, subflow statistics, active/idle stats,
    and destination port. Excludes raw host identifiers (IPs, flow IDs, timestamps).
  - MODE B (Enriched): Extends Mode A by incorporating ports and securely encoded
    identifiers (frequency-encoded or hashed subnets). Never maps raw IPs to arbitrary
    integers (Spec §5).
"""

import hashlib
import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known non-feature identification / leakage columns to exclude in Mode A
METADATA_COLUMNS = {
    "flow_id",
    "source_ip",
    "destination_ip",
    "source_port",
    "timestamp",
    "label",
    "binary_label",
    "attack_cat",
}

# Core behavioral column patterns
BEHAVIORAL_PATTERNS = [
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "total_length_of_fwd",
    "total_length_of_bwd",
    "fwd_packet_length",
    "bwd_packet_length",
    "flow_bytes",
    "flow_packets",
    "flow_iat",
    "fwd_iat",
    "bwd_iat",
    "fwd_psh",
    "bwd_psh",
    "fwd_urg",
    "bwd_urg",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets/s",
    "bwd_packets/s",
    "packet_length",
    "fin_flag",
    "syn_flag",
    "rst_flag",
    "psh_flag",
    "ack_flag",
    "urg_flag",
    "cwe_flag",
    "ece_flag",
    "down/up_ratio",
    "average_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "subflow_fwd",
    "subflow_bwd",
    "init_win_bytes",
    "act_data_pkt",
    "min_seg_size",
    "active_",
    "idle_",
    "destination_port",
    "protocol",
]


def hash_ip(ip_str: str, num_buckets: int = 1000) -> int:
    """
    Hash an IP address into a fixed bucket ID to avoid memorization.
    
    Per Spec §5: Raw IP strings must NOT simply be converted to arbitrary
    integer IDs. Hashing into a fixed bucket preserves uniqueness without
    false ordinal ordering.
    """
    if pd.isna(ip_str) or not str(ip_str).strip():
        return 0
    digest = hashlib.md5(str(ip_str).strip().encode("utf-8")).hexdigest()
    return int(digest, 16) % num_buckets


def extract_features(
    df: pd.DataFrame,
    mode: str = "behavioral",
    target_col: str = "binary_label",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract features and labels according to the specified mode.

    Args:
        df: Input DataFrame containing cleaned network flow data.
        mode: "behavioral" (Mode A) or "enriched" (Mode B).
        target_col: Column name containing the target labels.

    Returns:
        Tuple of (X DataFrame, y Series, list of feature column names).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].copy()
    feature_df = pd.DataFrame(index=df.index)

    if mode.lower() == "behavioral":
        # Mode A: Select all numeric behavioral columns, excluding identifier columns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in METADATA_COLUMNS:
                continue
            # Keep if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_df[col] = df[col].astype(np.float32)

        logger.info(
            f"Mode A (Behavioral) extracted {feature_df.shape[1]} features for {len(df)} flows."
        )

    elif mode.lower() == "enriched":
        # Mode B: Behavioral features plus hashed identifiers and ports
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["label", "binary_label", "attack_cat"]:
                continue
            if col_lower in ["source_ip", "destination_ip"]:
                # Hashed encoding per Spec §5
                feature_df[f"{col_lower}_hashed"] = (
                    df[col].apply(hash_ip).astype(np.float32)
                )
            elif pd.api.types.is_numeric_dtype(df[col]):
                feature_df[col] = df[col].astype(np.float32)

        logger.info(
            f"Mode B (Enriched) extracted {feature_df.shape[1]} features for {len(df)} flows."
        )

    else:
        raise ValueError(f"Unknown feature mode '{mode}'. Choose 'behavioral' or 'enriched'.")

    feature_cols = list(feature_df.columns)
    return feature_df, y, feature_cols
