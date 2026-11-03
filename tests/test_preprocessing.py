"""
Unit tests for data splitting, scaling, and PyTorch dataset modules.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from predator_net.data.splitter import (
    stratified_split,
    leakage_aware_split,
    temporal_split,
    split_data,
)
from predator_net.data.scaler import FeatureScaler
from predator_net.data.dataset import FlowDataset, create_dataloader


@pytest.fixture
def split_test_data():
    """Generates a dataset with groups and timestamps for testing splitting."""
    rng = np.random.RandomState(42)
    n = 300
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "source_ip": [f"192.168.1.{i % 15}" for i in range(n)],
        "destination_ip": [f"10.0.0.{i % 10}" for i in range(n)],
        "source_port": [1000 + (i % 20) for i in range(n)],
        "destination_port": [80] * n,
        "protocol": [6] * n,
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "binary_label": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    return df


class TestDataSplitter:
    def test_stratified_split_ratios(self, split_test_data):
        train, val, test = stratified_split(
            split_test_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )
        total = len(split_test_data)
        assert len(train) == int(total * 0.70)
        assert len(val) == int(total * 0.15)
        assert len(test) == total - len(train) - len(val)

        # Check stratification
        train_pos_rate = train["binary_label"].mean()
        val_pos_rate = val["binary_label"].mean()
        test_pos_rate = test["binary_label"].mean()
        assert abs(train_pos_rate - val_pos_rate) < 0.05
        assert abs(train_pos_rate - test_pos_rate) < 0.05

    def test_leakage_aware_split_no_overlap(self, split_test_data):
        train, val, test = leakage_aware_split(
            split_test_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        # Verify 5-tuple groups do not overlap across splits
        def make_group(df):
            return set(
                df["source_ip"] + "_" + df["destination_ip"] + "_" +
                df["source_port"].astype(str) + "_" + df["destination_port"].astype(str)
            )

        train_grp = make_group(train)
        val_grp = make_group(val)
        test_grp = make_group(test)

        assert len(train_grp.intersection(val_grp)) == 0
        assert len(train_grp.intersection(test_grp)) == 0
        assert len(val_grp.intersection(test_grp)) == 0

    def test_temporal_split_chronological(self, split_test_data):
        train, val, test = temporal_split(
            split_test_data,
            timestamp_col="timestamp",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_invalid_ratio_raises(self, split_test_data):
        with pytest.raises(ValueError):
            stratified_split(split_test_data, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)


class TestFeatureScaler:
    def test_fit_on_train_only(self):
        """Spec §8: Scaler MUST be fitted strictly on train data."""
        X_train = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        X_test = np.array([[1000.0, 2000.0]], dtype=np.float32)

        scaler = FeatureScaler(method="standard")
        scaler.fit(X_train)

        # Mean of train should be [20, 30]
        np.testing.assert_allclose(scaler.scaler.mean_, [20.0, 30.0])

        transformed_test = scaler.transform(X_test)
        assert transformed_test.dtype == np.float32
        # Transformed test should be scaled using TRAIN statistics
        expected_test = (X_test - np.array([20.0, 30.0])) / np.array([10.0, 10.0])
        np.testing.assert_allclose(transformed_test, expected_test)

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.zeros((5, 2)))

    def test_save_and_load_scaler(self):
        X = np.random.randn(50, 4).astype(np.float32)
        scaler = FeatureScaler(method="robust")
        scaler.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)

        try:
            scaler.save(temp_path)
            loaded = FeatureScaler.load(temp_path)
            assert loaded.is_fitted
            assert loaded.method == "robust"
            np.testing.assert_allclose(scaler.transform(X), loaded.transform(X))
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestFlowDataset:
    def test_dataset_item_types_and_shapes(self):
        X = np.random.randn(100, 15).astype(np.float32)
        y = np.random.choice([0, 1], size=100).astype(np.float32)

        dataset = FlowDataset(X, y)
        assert len(dataset) == 100

        x_sample, y_sample = dataset[0]
        assert isinstance(x_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)
        assert x_sample.dtype == torch.float32
        assert y_sample.dtype == torch.float32
        assert x_sample.shape == (15,)
        assert y_sample.shape == (1,)

    def test_dataloader_batching(self):
        X = np.random.randn(64, 10).astype(np.float32)
        y = np.ones(64, dtype=np.float32)

        dataset = FlowDataset(X, y)
        loader = create_dataloader(dataset, batch_size=16, shuffle=False)

        batches = list(loader)
        assert len(batches) == 4
        bx, by = batches[0]
        assert bx.shape == (16, 10)
        assert by.shape == (16, 1)
