"""
Feature Scaling Module for PREDATOR NET.

CRITICAL RULE (Spec §8 & Rule #5):
  The scaler must NEVER be fitted using test or validation data.
  scaler.fit(X_train) ONLY.
  Never scaler.fit(all_data).
"""

import logging
from pathlib import Path
from typing import Union, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Robust wrapper around scikit-learn scalers enforcing strict fit-on-train semantics.
    """

    def __init__(self, method: str = "standard") -> None:
        """
        Args:
            method: "standard" (StandardScaler), "robust" (RobustScaler), or "minmax" (MinMaxScaler).
        """
        self.method = method.lower()
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard', 'robust', or 'minmax'.")

        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """
        Fit scaler parameters exclusively on training data.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)

        # Handle any residual NaN/inf before fitting
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler.fit(arr)
        self.is_fitted = True
        logger.info(f"Fitted {self.method} scaler on training data shape: {arr.shape}.")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform features using previously fitted parameters. Outputs float32.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before calling fit() on training data!")

        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                # Ensure column alignment if dataframe
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise KeyError(f"Missing features expected by scaler: {missing}")
                arr = X[self.feature_names].to_numpy(dtype=np.float32)
            else:
                arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        transformed = self.scaler.transform(arr).astype(np.float32)
        return transformed

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit on training data and return transformed float32 array.
        """
        return self.fit(X).transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted scaler and metadata to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "method": self.method,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved scaler state to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureScaler":
        """Load fitted scaler from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
        state = joblib.load(filepath)
        obj = cls(method=state["method"])
        obj.scaler = state["scaler"]
        obj.is_fitted = state["is_fitted"]
        obj.feature_names = state["feature_names"]
        logger.info(f"Loaded {obj.method} scaler with {len(obj.feature_names or [])} features.")
        return obj
