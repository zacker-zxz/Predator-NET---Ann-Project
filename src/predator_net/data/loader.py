"""
CIC-IDS2017 Dataset Loader.

Spec §6: Create a robust loader that:
  1. Locates all CIC-IDS2017 CSV files
  2. Reads files safely
  3. Strips whitespace from column names
  4. Normalizes inconsistent column names
  5. Detects the label column automatically
  6. Reports duplicate columns

Spec §5: NEVER assume an 'IP Address' column exists.
Every transformation is logged.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name normalization mapping
# ---------------------------------------------------------------------------
# CIC-IDS2017 CSVs have inconsistent column naming across files.
# This mapping normalizes them to a canonical form.
COLUMN_NORMALIZATION = {
    # Common variations → canonical name
    "flow id": "flow_id",
    "flowid": "flow_id",
    "source ip": "source_ip",
    "src ip": "source_ip",
    "source port": "source_port",
    "src port": "source_port",
    "destination ip": "destination_ip",
    "dst ip": "destination_ip",
    "destination port": "destination_port",
    "dst port": "destination_port",
    "protocol": "protocol",
    "timestamp": "timestamp",
    "flow duration": "flow_duration",
    "total fwd packets": "total_fwd_packets",
    "total backward packets": "total_bwd_packets",
    "total length of fwd packets": "total_length_fwd_packets",
    "total length of bwd packets": "total_length_bwd_packets",
    "fwd packet length max": "fwd_packet_length_max",
    "fwd packet length min": "fwd_packet_length_min",
    "fwd packet length mean": "fwd_packet_length_mean",
    "fwd packet length std": "fwd_packet_length_std",
    "bwd packet length max": "bwd_packet_length_max",
    "bwd packet length min": "bwd_packet_length_min",
    "bwd packet length mean": "bwd_packet_length_mean",
    "bwd packet length std": "bwd_packet_length_std",
    "flow bytes/s": "flow_bytes_per_s",
    "flow bytes\\s": "flow_bytes_per_s",
    "flow packets/s": "flow_packets_per_s",
    "flow packets\\s": "flow_packets_per_s",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat max": "flow_iat_max",
    "flow iat min": "flow_iat_min",
    "fwd iat total": "fwd_iat_total",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "fwd iat max": "fwd_iat_max",
    "fwd iat min": "fwd_iat_min",
    "bwd iat total": "bwd_iat_total",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "bwd iat max": "bwd_iat_max",
    "bwd iat min": "bwd_iat_min",
    "fwd psh flags": "fwd_psh_flags",
    "bwd psh flags": "bwd_psh_flags",
    "fwd urg flags": "fwd_urg_flags",
    "bwd urg flags": "bwd_urg_flags",
    "fwd header length": "fwd_header_length",
    "bwd header length": "bwd_header_length",
    "fwd packets/s": "fwd_packets_per_s",
    "fwd packets\\s": "fwd_packets_per_s",
    "bwd packets/s": "bwd_packets_per_s",
    "bwd packets\\s": "bwd_packets_per_s",
    "min packet length": "min_packet_length",
    "max packet length": "max_packet_length",
    "packet length mean": "packet_length_mean",
    "packet length std": "packet_length_std",
    "packet length variance": "packet_length_variance",
    "fin flag count": "fin_flag_count",
    "syn flag count": "syn_flag_count",
    "rst flag count": "rst_flag_count",
    "psh flag count": "psh_flag_count",
    "ack flag count": "ack_flag_count",
    "urg flag count": "urg_flag_count",
    "cwe flag count": "cwe_flag_count",
    "ece flag count": "ece_flag_count",
    "down/up ratio": "down_up_ratio",
    "average packet size": "average_packet_size",
    "avg fwd segment size": "avg_fwd_segment_size",
    "avg bwd segment size": "avg_bwd_segment_size",
    "fwd header length.1": "fwd_header_length_1",
    "fwd avg bytes/bulk": "fwd_avg_bytes_per_bulk",
    "fwd avg packets/bulk": "fwd_avg_packets_per_bulk",
    "fwd avg bulk rate": "fwd_avg_bulk_rate",
    "bwd avg bytes/bulk": "bwd_avg_bytes_per_bulk",
    "bwd avg packets/bulk": "bwd_avg_packets_per_bulk",
    "bwd avg bulk rate": "bwd_avg_bulk_rate",
    "subflow fwd packets": "subflow_fwd_packets",
    "subflow fwd bytes": "subflow_fwd_bytes",
    "subflow bwd packets": "subflow_bwd_packets",
    "subflow bwd bytes": "subflow_bwd_bytes",
    "init_win_bytes_forward": "init_win_bytes_forward",
    "init_win_bytes_backward": "init_win_bytes_backward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_forward",
    "active mean": "active_mean",
    "active std": "active_std",
    "active max": "active_max",
    "active min": "active_min",
    "idle mean": "idle_mean",
    "idle std": "idle_std",
    "idle max": "idle_max",
    "idle min": "idle_min",
    "label": "label",
}

# Possible label column names in CIC-IDS2017
LABEL_CANDIDATES = ["label", "Label", " Label", "class", "Class", "attack", "Attack"]


def _normalize_column_name(col: str) -> str:
    """
    Normalize a single column name:
      1. Strip whitespace
      2. Lowercase
      3. Look up in normalization map
      4. Fallback: replace spaces/special chars with underscores
    """
    stripped = col.strip()
    lower = stripped.lower()

    if lower in COLUMN_NORMALIZATION:
        return COLUMN_NORMALIZATION[lower]

    # Fallback normalization: lowercase, replace spaces & slashes with underscores
    normalized = re.sub(r"[\s/\\]+", "_", lower)
    normalized = re.sub(r"[^a-z0-9_.]", "", normalized)
    return normalized


normalize_column_name = _normalize_column_name


def find_csv_files(raw_dir: str | Path) -> list[Path]:
    """
    Locate all CSV files in the raw data directory.

    Args:
        raw_dir: Path to the directory containing CIC-IDS2017 CSV files.

    Returns:
        Sorted list of Path objects for each CSV file found.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no CSV files are found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_path}\n"
            f"Please download CIC-IDS2017 and place CSV files in '{raw_path}'"
        )

    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        # Also try case-insensitive and subdirectories
        csv_files = sorted(raw_path.rglob("*.csv"))

    if not csv_files:
        raise ValueError(
            f"No CSV files found in {raw_path}. "
            f"Please place CIC-IDS2017 CSV files in this directory."
        )

    logger.info(f"Found {len(csv_files)} CSV file(s) in {raw_path}:")
    for f in csv_files:
        logger.info(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    return csv_files


def _detect_label_column(columns: list[str]) -> Optional[str]:
    """
    Auto-detect the label column from a list of column names.

    Tries exact matches first, then case-insensitive/stripped matches.
    """
    # Exact match
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate

    # Stripped + lowered match
    col_map = {c.strip().lower(): c for c in columns}
    for candidate in LABEL_CANDIDATES:
        key = candidate.strip().lower()
        if key in col_map:
            return col_map[key]

    return None


def load_single_csv(
    filepath: Path,
    encoding: str = "utf-8",
    nrows: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a single CIC-IDS2017 CSV file with safe handling.

    Args:
        filepath: Path to the CSV file.
        encoding: File encoding.
        nrows: Maximum rows to read.

    Returns:
        Tuple of (DataFrame, file_info_dict).
    """
    file_info = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "size_mb": filepath.stat().st_size / 1e6,
    }

    logger.info(f"Loading {filepath.name} ({file_info['size_mb']:.1f} MB)...")

    # Try reading with different encodings if needed
    try:
        df = pd.read_csv(filepath, encoding=encoding, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 failed for {filepath.name}, trying latin-1")
        df = pd.read_csv(filepath, encoding="latin-1", nrows=nrows, low_memory=False)

    file_info["rows_raw"] = len(df)
    file_info["columns_raw"] = list(df.columns)
    file_info["num_columns_raw"] = len(df.columns)

    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 1: Strip whitespace from column names
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    stripped_changes = [
        (orig, new) for orig, new in zip(original_cols, df.columns) if orig != new
    ]
    if stripped_changes:
        logger.info(f"  Stripped whitespace from {len(stripped_changes)} column name(s)")
        file_info["whitespace_stripped"] = stripped_changes

    # Step 2: Check for duplicate columns BEFORE normalization
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        logger.warning(f"  Duplicate columns found: {duplicates}")
        file_info["duplicate_columns"] = duplicates

    # Step 3: Normalize column names
    normalized_cols = [_normalize_column_name(c) for c in df.columns]
    rename_map = {
        orig: norm
        for orig, norm in zip(df.columns, normalized_cols)
        if orig != norm
    }
    if rename_map:
        logger.info(f"  Normalized {len(rename_map)} column name(s)")
        file_info["columns_renamed"] = rename_map

    df.columns = normalized_cols

    # Step 4: Detect label column
    label_col = _detect_label_column(list(df.columns))
    if label_col:
        logger.info(f"  Label column detected: '{label_col}'")
        file_info["label_column"] = label_col
    else:
        logger.warning(f"  No label column detected in {filepath.name}")
        file_info["label_column"] = None

    file_info["columns_final"] = list(df.columns)

    return df, file_info


def load_dataset(
    raw_dir: str | Path,
    encoding: str = "utf-8",
    nrows_per_file: Optional[int] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load all CIC-IDS2017 CSV files and concatenate them.

    This is the main entry point for Phase 1 data loading.

    Args:
        raw_dir: Path to directory containing CSV files.
        encoding: File encoding.
        nrows_per_file: Optional maximum rows per CSV file.

    Returns:
        Tuple of (concatenated DataFrame, list of per-file info dicts).

    Raises:
        FileNotFoundError: If raw_dir doesn't exist.
        ValueError: If no CSVs found or label column is inconsistent.
    """
    csv_files = find_csv_files(raw_dir)

    all_dfs = []
    all_file_info = []

    for filepath in csv_files:
        df, info = load_single_csv(filepath, encoding=encoding, nrows=nrows_per_file)
        all_dfs.append(df)
        all_file_info.append(info)

    # Concatenate all files
    logger.info(f"Concatenating {len(all_dfs)} file(s)...")
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns"
    )

    # Verify label column consistency
    label_cols = [info.get("label_column") for info in all_file_info]
    unique_labels = set(lc for lc in label_cols if lc is not None)

    if len(unique_labels) == 0:
        raise ValueError(
            "No label column detected in any CSV file. "
            "Expected one of: " + ", ".join(LABEL_CANDIDATES)
        )
    elif len(unique_labels) > 1:
        logger.warning(
            f"Multiple label column names found across files: {unique_labels}. "
            f"Using the first detected: '{label_cols[0]}'"
        )

    return combined, all_file_info
