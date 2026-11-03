"""
PHASE 1 SCRIPT: Dataset Discovery & Quality Report.

Usage:
    python scripts/prepare_dataset.py [--raw-dir data/raw] [--config config/baseline.yaml]

This script:
  1. Locates all CIC-IDS2017 CSV files
  2. Loads and concatenates them
  3. Runs data quality analysis
  4. Applies cleaning transformations
  5. Saves a comprehensive data_quality_report.json

No model training. No feature engineering. Just data inspection.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from predator_net.data.loader import load_dataset
from predator_net.data.cleaner import analyze_data_quality, clean_dataset, save_quality_report
from predator_net.training.seed import set_global_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PREDATOR NET — Phase 1: Dataset Discovery & Quality Report"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to directory containing CIC-IDS2017 CSV files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline.yaml",
        help="Path to baseline configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/data_quality",
        help="Directory for output artifacts",
    )
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        seed = config.get("project", {}).get("seed", 42)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}
        seed = 42

    # Set seed
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 1: DATASET DISCOVERY")
    logger.info("=" * 70)

    # Resolve raw_dir relative to project root
    raw_dir = PROJECT_ROOT / args.raw_dir

    # Step 1: Load dataset
    logger.info(f"Looking for CIC-IDS2017 CSVs in: {raw_dir}")
    try:
        df, file_info = load_dataset(raw_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        logger.error(
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  CIC-IDS2017 dataset not found.                        ║\n"
            "║                                                        ║\n"
            "║  Please download from:                                 ║\n"
            "║  https://www.unb.ca/cic/datasets/ids-2017.html         ║\n"
            "║                                                        ║\n"
            "║  Place the CSV files (MachineLearningCSV/) into:       ║\n"
            f"║  {str(raw_dir):<55}║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )
        sys.exit(1)

    # Step 2: Detect label column
    label_col = None
    for info in file_info:
        if info.get("label_column"):
            label_col = info["label_column"]
            break

    if label_col is None:
        logger.error("Could not detect label column in any file!")
        sys.exit(1)

    # Step 3: Data quality analysis (non-destructive)
    logger.info("\nRunning data quality analysis...")
    quality_report = analyze_data_quality(df, label_column=label_col)

    # Step 4: Clean dataset
    logger.info("\nApplying cleaning transformations...")
    cleaned_df, cleaning_log = clean_dataset(
        df,
        label_column=label_col,
        remove_duplicates=True,
        handle_infinity="nan",
        handle_nan="drop",
        remove_constant_cols=True,
    )

    # Step 5: Save quality report
    output_dir = PROJECT_ROOT / args.output_dir
    report_path = save_quality_report(
        report=quality_report,
        cleaning_log=cleaning_log,
        file_info=file_info,
        output_path=output_dir / "data_quality_report.json",
    )

    # Save cleaned data as interim parquet for Phase 2
    interim_dir = PROJECT_ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    interim_path = interim_dir / "cleaned_combined.parquet"
    cleaned_df.to_parquet(interim_path, index=False)
    logger.info(f"Cleaned data saved to {interim_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Files loaded:          {len(file_info)}")
    logger.info(f"Raw rows:              {quality_report['total_rows']:,}")
    logger.info(f"Raw columns:           {quality_report['total_columns']}")
    logger.info(f"Cleaned rows:          {len(cleaned_df):,}")
    logger.info(f"Cleaned columns:       {len(cleaned_df.columns)}")
    logger.info(f"Rows removed:          {cleaning_log['rows_removed']:,}")
    if "binary_distribution" in cleaning_log:
        bd = cleaning_log["binary_distribution"]
        logger.info(f"Benign flows:          {bd['benign']:,}")
        logger.info(f"Attack flows:          {bd['attack']:,}")
    logger.info(f"Quality report:        {report_path}")
    logger.info(f"Interim data:          {interim_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
