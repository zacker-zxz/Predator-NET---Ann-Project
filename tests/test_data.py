"""
Unit tests for data loading and cleaning modules.

Tests loader.py and cleaner.py using synthetic data that mimics
CIC-IDS2017 structure (no actual dataset required).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predator_net.data.loader import (
    _normalize_column_name,
    _detect_label_column,
    find_csv_files,
    load_single_csv,
    load_dataset,
)
from predator_net.data.cleaner import (
    analyze_data_quality,
    clean_dataset,
    save_quality_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a synthetic CIC-IDS2017-like CSV."""
    df = pd.DataFrame({
        " Flow Duration": [100, 200, 300, 400, 500],
        " Total Fwd Packets": [10, 20, 30, 40, 50],
        " Total Backward Packets": [5, 10, 15, 20, 25],
        " Flow Bytes/s": [1000.0, np.inf, 3000.0, 4000.0, 5000.0],
        " Flow Packets/s": [100.0, 200.0, np.nan, 400.0, 500.0],
        " Destination Port": [80, 443, 22, 80, 443],
        " Label": ["BENIGN", "BENIGN", "DDoS", "BENIGN", "PortScan"],
    })
    csv_path = tmp_path / "Monday-WorkingHours.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking cleaned CIC-IDS2017 data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "flow_duration": np.random.randint(0, 10000, n).astype(float),
        "total_fwd_packets": np.random.randint(0, 100, n).astype(float),
        "total_bwd_packets": np.random.randint(0, 50, n).astype(float),
        "flow_bytes_per_s": np.random.uniform(0, 10000, n),
        "destination_port": np.random.choice([80, 443, 22, 8080], n),
        "constant_col": np.zeros(n),
        "label": np.random.choice(["BENIGN", "DDoS", "PortScan"], n, p=[0.7, 0.2, 0.1]),
    })


@pytest.fixture
def sample_df_with_issues(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Add NaN, Inf, and duplicates to the sample DataFrame."""
    df = sample_df.copy()
    # Add NaN
    df.loc[5, "flow_bytes_per_s"] = np.nan
    df.loc[10, "flow_duration"] = np.nan
    # Add Inf
    df.loc[15, "flow_bytes_per_s"] = np.inf
    df.loc[20, "flow_bytes_per_s"] = -np.inf
    # Add duplicates
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestColumnNormalization:
    def test_strips_whitespace(self):
        assert _normalize_column_name("  Flow Duration  ") == "flow_duration"

    def test_normalizes_known_columns(self):
        assert _normalize_column_name("Flow Bytes/s") == "flow_bytes_per_s"
        assert _normalize_column_name(" Label") == "label"
        assert _normalize_column_name("Destination Port") == "destination_port"
        assert _normalize_column_name("Source IP") == "source_ip"

    def test_fallback_normalization(self):
        result = _normalize_column_name("My Custom Column")
        assert result == "my_custom_column"

    def test_handles_backslash(self):
        assert _normalize_column_name("Flow Bytes\\s") == "flow_bytes_per_s"


class TestLabelDetection:
    def test_detects_label(self):
        assert _detect_label_column(["flow_duration", "label"]) == "label"

    def test_detects_label_with_space(self):
        assert _detect_label_column(["flow_duration", " Label"]) == " Label"

    def test_returns_none_when_missing(self):
        assert _detect_label_column(["flow_duration", "packets"]) is None


class TestFindCSVFiles:
    def test_finds_csvs(self, sample_csv_dir: Path):
        files = find_csv_files(sample_csv_dir)
        assert len(files) == 1
        assert files[0].suffix == ".csv"

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            find_csv_files("/nonexistent/path")

    def test_raises_on_empty_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No CSV files found"):
            find_csv_files(tmp_path)


class TestLoadSingleCSV:
    def test_loads_and_normalizes(self, sample_csv_dir: Path):
        csv_file = list(sample_csv_dir.glob("*.csv"))[0]
        df, info = load_single_csv(csv_file)

        assert len(df) == 5
        assert "flow_duration" in df.columns
        assert "label" in df.columns
        assert info["label_column"] == "label"

    def test_strips_whitespace_from_columns(self, sample_csv_dir: Path):
        csv_file = list(sample_csv_dir.glob("*.csv"))[0]
        df, info = load_single_csv(csv_file)
        # No column should start/end with whitespace
        for col in df.columns:
            assert col == col.strip()


class TestLoadDataset:
    def test_loads_full_dataset(self, sample_csv_dir: Path):
        df, file_info = load_dataset(sample_csv_dir)
        assert len(df) == 5
        assert len(file_info) == 1


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestAnalyzeDataQuality:
    def test_reports_nan(self, sample_df_with_issues: pd.DataFrame):
        report = analyze_data_quality(sample_df_with_issues, label_column="label")
        assert report["missing_values"]["total_missing_cells"] > 0

    def test_reports_infinity(self, sample_df_with_issues: pd.DataFrame):
        report = analyze_data_quality(sample_df_with_issues, label_column="label")
        assert report["infinite_values"]["total_infinite_cells"] > 0

    def test_reports_duplicates(self, sample_df_with_issues: pd.DataFrame):
        report = analyze_data_quality(sample_df_with_issues, label_column="label")
        assert report["duplicate_rows"]["count"] > 0

    def test_reports_constant_columns(self, sample_df: pd.DataFrame):
        report = analyze_data_quality(sample_df, label_column="label")
        assert "constant_col" in report["constant_columns"]["columns"]

    def test_reports_label_distribution(self, sample_df: pd.DataFrame):
        report = analyze_data_quality(sample_df, label_column="label")
        assert report["label_distribution"]["num_classes"] > 0

    def test_no_issues_on_clean_data(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "label": ["BENIGN", "DDoS", "BENIGN"],
        })
        report = analyze_data_quality(df, label_column="label")
        assert report["missing_values"]["total_missing_cells"] == 0
        assert report["infinite_values"]["total_infinite_cells"] == 0
        assert report["duplicate_rows"]["count"] == 0


class TestCleanDataset:
    def test_removes_nan_rows(self, sample_df_with_issues: pd.DataFrame):
        cleaned, log = clean_dataset(
            sample_df_with_issues,
            label_column="label",
            handle_infinity="nan",
            handle_nan="drop",
        )
        assert not cleaned.select_dtypes(include=[np.number]).isna().any().any()

    def test_removes_infinity(self, sample_df_with_issues: pd.DataFrame):
        cleaned, log = clean_dataset(
            sample_df_with_issues,
            label_column="label",
            handle_infinity="nan",
            handle_nan="drop",
        )
        numeric = cleaned.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_removes_duplicates(self, sample_df_with_issues: pd.DataFrame):
        cleaned, log = clean_dataset(
            sample_df_with_issues,
            label_column="label",
        )
        assert cleaned.duplicated().sum() == 0

    def test_removes_constant_columns(self, sample_df: pd.DataFrame):
        cleaned, log = clean_dataset(sample_df, label_column="label")
        assert "constant_col" not in cleaned.columns

    def test_creates_binary_label(self, sample_df: pd.DataFrame):
        cleaned, log = clean_dataset(sample_df, label_column="label")
        assert "binary_label" in cleaned.columns
        assert set(cleaned["binary_label"].unique()).issubset({0, 1})

    def test_benign_maps_to_zero(self, sample_df: pd.DataFrame):
        cleaned, log = clean_dataset(sample_df, label_column="label")
        benign_mask = cleaned["label"] == "BENIGN"
        assert (cleaned.loc[benign_mask, "binary_label"] == 0).all()

    def test_attack_maps_to_one(self, sample_df: pd.DataFrame):
        cleaned, log = clean_dataset(sample_df, label_column="label")
        attack_mask = cleaned["label"] != "BENIGN"
        assert (cleaned.loc[attack_mask, "binary_label"] == 1).all()

    def test_does_not_modify_original(self, sample_df: pd.DataFrame):
        original_shape = sample_df.shape
        clean_dataset(sample_df, label_column="label")
        assert sample_df.shape == original_shape

    def test_log_contains_transformations(self, sample_df_with_issues: pd.DataFrame):
        _, log = clean_dataset(sample_df_with_issues, label_column="label")
        assert len(log["transformations"]) > 0
        assert "initial_shape" in log
        assert "final_shape" in log


class TestSaveQualityReport:
    def test_saves_valid_json(self, sample_df: pd.DataFrame, tmp_path: Path):
        report = analyze_data_quality(sample_df, label_column="label")
        _, cleaning_log = clean_dataset(sample_df, label_column="label")

        output_path = tmp_path / "report.json"
        save_quality_report(report, cleaning_log, [], output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert "quality_analysis" in loaded
        assert "cleaning_log" in loaded
        assert "preprocessing_decisions" in loaded
