"""
Unit tests for Feature Engineering, Selection, and Analysis modules.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from predator_net.features.engineering import extract_features, hash_ip
from predator_net.features.selection import (
    remove_collinear_features,
    compute_mutual_information,
    compute_tree_importance,
    select_features,
)
from predator_net.features.analysis import save_feature_artifacts


@pytest.fixture
def sample_feature_data():
    """Generates a synthetic DataFrame for feature testing."""
    rng = np.random.RandomState(42)
    n = 200
    f1 = rng.uniform(0, 10, n)
    f2 = f1 * 0.999 + rng.normal(0, 0.001, n)  # Nearly perfectly collinear with f1
    f3 = rng.uniform(50, 100, n)
    f4 = rng.normal(0, 1, n)
    # y depends strongly on f1 and f3
    prob = 1 / (1 + np.exp(-(f1 - 5 + (f3 - 75) / 10)))
    y = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "flow_duration": f1,
        "collinear_copy": f2,
        "total_fwd_packets": f3,
        "noise_feature": f4,
        "source_ip": ["192.168.1.1"] * (n // 2) + ["10.0.0.1"] * (n // 2),
        "destination_ip": ["172.16.0.5"] * n,
        "binary_label": y,
    })
    return df


class TestFeatureEngineering:
    def test_hash_ip_deterministic(self):
        h1 = hash_ip("192.168.1.1", num_buckets=1000)
        h2 = hash_ip("192.168.1.1", num_buckets=1000)
        assert h1 == h2
        assert 0 <= h1 < 1000

    def test_hash_ip_handles_empty(self):
        assert hash_ip("") == 0
        assert hash_ip(None) == 0

    def test_mode_a_excludes_ips(self, sample_feature_data):
        X, y, cols = extract_features(sample_feature_data, mode="behavioral")
        assert "source_ip" not in cols
        assert "destination_ip" not in cols
        assert "binary_label" not in cols
        assert "flow_duration" in cols
        assert X.shape[0] == len(sample_feature_data)

    def test_mode_b_hashes_ips(self, sample_feature_data):
        X, y, cols = extract_features(sample_feature_data, mode="enriched")
        assert "source_ip_hashed" in cols
        assert "destination_ip_hashed" in cols
        assert np.issubdtype(X["source_ip_hashed"].dtype, np.floating)


class TestFeatureSelection:
    def test_remove_collinear_features(self, sample_feature_data):
        X = sample_feature_data[["flow_duration", "collinear_copy", "total_fwd_packets"]]
        y = sample_feature_data["binary_label"]
        retained, dropped, corr = remove_collinear_features(X, y=y, threshold=0.95)

        assert len(dropped) == 1
        assert "collinear_copy" in dropped or "flow_duration" in dropped
        assert len(retained) == 2
        assert "total_fwd_packets" in retained

    def test_compute_mutual_information(self, sample_feature_data):
        X = sample_feature_data[["flow_duration", "total_fwd_packets", "noise_feature"]]
        y = sample_feature_data["binary_label"]
        mi = compute_mutual_information(X, y, random_state=42)

        assert len(mi) == 3
        assert mi.iloc[0] >= mi.iloc[1] >= mi.iloc[2]
        # Informative features should have higher MI than pure noise
        assert mi["flow_duration"] > mi["noise_feature"]

    def test_select_features_pipeline(self, sample_feature_data):
        X = sample_feature_data[["flow_duration", "collinear_copy", "total_fwd_packets", "noise_feature"]]
        y = sample_feature_data["binary_label"]

        selected, rankings, corr = select_features(
            X, y, correlation_threshold=0.95, mi_threshold=0.0, random_state=42
        )
        assert len(selected) <= 3  # dropped collinear
        assert isinstance(rankings, pd.DataFrame)
        assert "feature" in rankings.columns
        assert "mutual_info" in rankings.columns


class TestFeatureAnalysis:
    def test_save_feature_artifacts(self, sample_feature_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            rankings = pd.DataFrame({
                "feature": ["flow_duration", "total_fwd_packets"],
                "mutual_info": [0.45, 0.32],
            })
            selected = ["flow_duration", "total_fwd_packets"]
            corr = pd.DataFrame(
                [[1.0, 0.2], [0.2, 1.0]],
                index=selected,
                columns=selected,
            )

            save_feature_artifacts(rankings, selected, corr, out_path)

            assert (out_path / "feature_rankings.csv").exists()
            assert (out_path / "selected_features.json").exists()
            assert (out_path / "correlation_matrix.png").exists()
            assert (out_path / "feature_importance.png").exists()

            with open(out_path / "selected_features.json") as f:
                data = json.load(f)
                assert data["num_features"] == 2
