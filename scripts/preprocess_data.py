"""
PHASE 2 SCRIPT: Preprocessing & Feature Engineering Pipeline.

Usage:
    python scripts/preprocess_data.py [--config config/baseline.yaml] [--generate-synthetic]

This script:
  1. Loads cleaned interim data (or runs cleaner on data/raw, or generates synthetic data)
  2. Extracts behavioral/enriched features (Spec §5, §10)
  3. Executes feature selection (collinearity filter + mutual information) (Spec §11)
  4. Saves feature selection artifacts (rankings, selected features, plots)
  5. Partitions dataset using Protocol A (stratified) or Protocol B (leakage-aware) (Spec §7)
  6. Fits FeatureScaler ONLY on training data, transforms val and test (Spec §8)
  7. Persists processed data as Parquet and memory-mapped NumPy arrays (Spec §12)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
import numpy as np
import pandas as pd

from predator_net.training.seed import set_global_seed
from predator_net.data.loader import load_dataset
from predator_net.data.cleaner import clean_dataset, analyze_data_quality, save_quality_report
from predator_net.data.splitter import split_data
from predator_net.data.scaler import FeatureScaler
from predator_net.data.synthetic import generate_synthetic_cicids2017
from predator_net.features.engineering import extract_features
from predator_net.features.selection import select_features
from predator_net.features.analysis import save_feature_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PREDATOR NET — Phase 2: Preprocessing & Feature Engineering"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline.yaml",
        help="Path to baseline configuration YAML",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic CIC-IDS2017 dataset if raw data is missing",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=25000,
        help="Number of samples to generate if synthetic mode is active",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to directory containing raw CSVs (overrides config)",
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=None,
        help="Max rows to read per CSV file (for fast balanced training)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if interim dataset exists",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    if args.raw_dir:
        raw_dir = PROJECT_ROOT / args.raw_dir
    else:
        raw_dir = PROJECT_ROOT / config.get("data", {}).get("raw_dir", "data/raw")
        if not raw_dir.exists() or not (list(raw_dir.glob("*.csv")) or list(raw_dir.glob("**/*.csv"))):
            alt_dir = PROJECT_ROOT / "data" / "cic-ids-2017" / "MachineLearningCVE"
            if alt_dir.exists():
                raw_dir = alt_dir

    interim_dir = PROJECT_ROOT / config.get("data", {}).get("interim_dir", "data/interim")
    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")
    features_art_dir = PROJECT_ROOT / "artifacts" / "features"
    data_quality_dir = PROJECT_ROOT / "artifacts" / "data_quality"

    processed_dir.mkdir(parents=True, exist_ok=True)
    features_art_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 2: PREPROCESSING & FEATURE ENGINEERING")
    logger.info("=" * 70)

    # 1. Acquire / Clean dataset
    interim_file = interim_dir / "cleaned_combined.parquet"
    if not interim_file.exists():
        interim_file = interim_dir / "clean_dataset.parquet"

    if interim_file.exists() and not args.force:
        logger.info(f"Loading existing cleaned interim dataset from {interim_file}")
        df = pd.read_parquet(interim_file)
    else:
        # Check raw CSVs
        csv_files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("**/*.csv"))
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files in {raw_dir}. Loading and cleaning...")
            raw_df, file_info = load_dataset(raw_dir, nrows_per_file=args.sample_per_file)
            label_col = "label"
            for info in file_info:
                if info.get("label_column"):
                    label_col = info["label_column"]
                    break
            report = analyze_data_quality(raw_df, label_column=label_col)
            df, cleaning_log = clean_dataset(raw_df, label_column=label_col)
            save_quality_report(report, cleaning_log, file_info, data_quality_dir / "data_quality_report.json")
            df.to_parquet(interim_file, index=False)
        elif args.generate_synthetic:
            logger.info(
                f"Generating {args.synthetic_samples} synthetic CIC-IDS2017 flows for pipeline verification..."
            )
            raw_df = generate_synthetic_cicids2017(n_samples=args.synthetic_samples, seed=seed)
            # Normalize column names to canonical form
            from predator_net.data.loader import normalize_column_name
            raw_df.columns = [normalize_column_name(c) for c in raw_df.columns]
            file_info = [{
                "file": "synthetic_cicids2017.csv",
                "rows": len(raw_df),
                "columns": len(raw_df.columns),
                "label_column": "label",
            }]
            report = analyze_data_quality(raw_df, label_column="label")
            df, cleaning_log = clean_dataset(raw_df, label_column="label")
            save_quality_report(report, cleaning_log, file_info, data_quality_dir / "data_quality_report.json")
            df.to_parquet(interim_file, index=False)
        else:
            logger.error(
                f"No CSVs found in {raw_dir} and interim dataset missing. "
                "Specify --generate-synthetic to create synthetic verification data or place CSVs into data/raw/"
            )
            sys.exit(1)

    logger.info(f"Dataset shape for feature extraction: {df.shape}")

    # 2. Feature Extraction (Mode A / Mode B)
    feature_mode = config.get("features", {}).get("mode", "behavioral")
    logger.info(f"Extracting features in mode: '{feature_mode}' (Spec §5, §10)")
    X_full, y_full, all_features = extract_features(df, mode=feature_mode, target_col="binary_label")

    # 3. Feature Selection
    corr_thresh = config.get("features", {}).get("correlation_threshold", 0.95)
    use_tree_imp = config.get("features", {}).get("model_based_importance", False)

    logger.info(f"Running feature selection (corr_thresh={corr_thresh}, tree_imp={use_tree_imp})")
    selected_features, rankings_df, corr_matrix = select_features(
        X_full,
        y_full,
        correlation_threshold=corr_thresh,
        use_tree_importance=use_tree_imp,
        random_state=seed,
    )

    # 4. Save Feature Analysis Artifacts
    logger.info(f"Saving feature selection artifacts to {features_art_dir}")
    save_feature_artifacts(
        rankings_df=rankings_df,
        selected_features=selected_features,
        corr_matrix=corr_matrix,
        output_dir=features_art_dir,
    )

    # 5. Data Splitting (Protocol A / B / C)
    proto = config.get("splitting", {}).get("protocol", "stratified")
    train_r = config.get("splitting", {}).get("train_ratio", 0.70)
    val_r = config.get("splitting", {}).get("val_ratio", 0.15)
    test_r = config.get("splitting", {}).get("test_ratio", 0.15)

    # Prepare DataFrame with selected features and binary label for splitting
    filtered_df = df[selected_features].copy()
    filtered_df["binary_label"] = y_full.values
    if "timestamp" in df.columns:
        filtered_df["timestamp"] = df["timestamp"].values

    logger.info(f"Splitting data using protocol '{proto}' (train={train_r}, val={val_r}, test={test_r})")
    train_df, val_df, test_df = split_data(
        filtered_df,
        protocol=proto,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        label_col="binary_label",
        random_state=seed,
    )

    # Separate X and y
    X_train = train_df[selected_features].to_numpy(dtype=np.float32)
    y_train = train_df["binary_label"].to_numpy(dtype=np.float32)

    X_val = val_df[selected_features].to_numpy(dtype=np.float32)
    y_val = val_df["binary_label"].to_numpy(dtype=np.float32)

    X_test = test_df[selected_features].to_numpy(dtype=np.float32)
    y_test = test_df["binary_label"].to_numpy(dtype=np.float32)

    # 6. Fit Scaler ONLY on Training Data (Spec §8)
    logger.info("Fitting FeatureScaler ONLY on train data (Spec §8)...")
    scaler = FeatureScaler(method="standard")
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler
    scaler_path = features_art_dir / "scaler.joblib"
    scaler.save(scaler_path)

    # 7. Persist Processed Data
    logger.info(f"Saving processed datasets to {processed_dir}...")

    # Parquet files
    train_out_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    train_out_df["binary_label"] = y_train
    train_out_df.to_parquet(processed_dir / "train.parquet", index=False)

    val_out_df = pd.DataFrame(X_val_scaled, columns=selected_features)
    val_out_df["binary_label"] = y_val
    val_out_df.to_parquet(processed_dir / "val.parquet", index=False)

    test_out_df = pd.DataFrame(X_test_scaled, columns=selected_features)
    test_out_df["binary_label"] = y_test
    test_out_df.to_parquet(processed_dir / "test.parquet", index=False)

    # Memory-mapped NumPy arrays for high-speed PyTorch loading
    np.save(processed_dir / "train_X.npy", X_train_scaled)
    np.save(processed_dir / "train_y.npy", y_train)
    np.save(processed_dir / "val_X.npy", X_val_scaled)
    np.save(processed_dir / "val_y.npy", y_val)
    np.save(processed_dir / "test_X.npy", X_test_scaled)
    np.save(processed_dir / "test_y.npy", y_test)

    # 8. Save Metadata
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    pos_weight = float(neg_count / max(pos_count, 1.0))

    metadata = {
        "num_features": len(selected_features),
        "selected_features": selected_features,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_benign_count": int(neg_count),
        "train_attack_count": int(pos_count),
        "pos_weight": pos_weight,
        "protocol": proto,
        "scaler_method": scaler.method,
    }

    with open(features_art_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info("PHASE 2 COMPLETED SUCCESSFULLY!")
    logger.info(f"Selected Features: {len(selected_features)}")
    logger.info(f"Train / Val / Test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")
    logger.info(f"Training pos_weight: {pos_weight:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
