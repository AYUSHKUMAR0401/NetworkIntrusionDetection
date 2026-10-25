"""
run_eda.py — Run the full EDA + feature engineering pipeline.

Usage:
    python -m src.run_eda               # auto-detect real or synthetic data
    python -m src.run_eda --synthetic    # force synthetic data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    PROCESSED_CSV,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SOURCE_FILE_COLUMN,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.run_eda")


def _generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic CIC-IDS2017-like data for demonstration.

    This data is SYNTHETIC — it does not represent real network traffic
    and must not be used to claim real-data results.
    """
    rng = np.random.RandomState(42)

    # Realistic label distribution
    attack_types = [
        ("benign", 0.60),
        ("bot", 0.05),
        ("ddos", 0.10),
        ("dos hulk", 0.08),
        ("dos goldeneye", 0.04),
        ("dos slowhttptest", 0.03),
        ("dos slowloris", 0.02),
        ("ftp-patator", 0.02),
        ("portscan", 0.03),
        ("ssh-patator", 0.02),
        ("web attack - brute force", 0.005),
        ("heartbleed", 0.003),
        ("infiltration", 0.002),
    ]
    labels, probs = zip(*attack_types)
    # Normalise probabilities
    probs = np.array(probs)
    probs = probs / probs.sum()

    original_labels = rng.choice(list(labels), size=n, p=probs)
    binary_target = np.where(original_labels == "benign", 0, 1)

    # Synthetic feature columns (mimicking CIC-IDS2017 naming)
    df = pd.DataFrame({
        LABEL_COLUMN: original_labels,
        ORIGINAL_LABEL_COLUMN: original_labels,
        BINARY_TARGET_COLUMN: binary_target,
        SOURCE_FILE_COLUMN: rng.choice(
            ["Monday.csv", "Tuesday.csv", "Wednesday.csv", "Thursday.csv", "Friday.csv"],
            size=n,
        ),
        "flow_duration": rng.exponential(50000, n).clip(0),
        "total_fwd_packets": rng.poisson(15, n).astype(float),
        "total_backward_packets": rng.poisson(12, n).astype(float),
        "total_length_of_fwd_packets": rng.exponential(3000, n),
        "total_length_of_bwd_packets": rng.exponential(2500, n),
        "fwd_packet_length_max": rng.uniform(0, 65535, n),
        "fwd_packet_length_min": rng.uniform(0, 100, n),
        "fwd_packet_length_mean": rng.uniform(50, 1500, n),
        "fwd_packet_length_std": rng.exponential(200, n),
        "bwd_packet_length_max": rng.uniform(0, 65535, n),
        "bwd_packet_length_min": rng.uniform(0, 100, n),
        "bwd_packet_length_mean": rng.uniform(50, 1500, n),
        "bwd_packet_length_std": rng.exponential(200, n),
        "flow_bytes_s": rng.exponential(50000, n),
        "flow_packets_s": rng.exponential(1000, n),
        "flow_iat_mean": rng.exponential(100000, n),
        "flow_iat_std": rng.exponential(200000, n),
        "flow_iat_max": rng.exponential(500000, n),
        "flow_iat_min": rng.uniform(0, 10000, n),
        "fwd_iat_total": rng.exponential(100000, n),
        "fwd_iat_mean": rng.exponential(50000, n),
        "fwd_iat_std": rng.exponential(100000, n),
        "fwd_iat_max": rng.exponential(200000, n),
        "fwd_iat_min": rng.uniform(0, 5000, n),
        "bwd_iat_total": rng.exponential(80000, n),
        "bwd_iat_mean": rng.exponential(40000, n),
        "bwd_iat_std": rng.exponential(80000, n),
        "bwd_iat_max": rng.exponential(200000, n),
        "bwd_iat_min": rng.uniform(0, 5000, n),
        "fwd_psh_flags": rng.randint(0, 2, n).astype(float),
        "bwd_psh_flags": rng.randint(0, 2, n).astype(float),
        "fwd_urg_flags": rng.randint(0, 2, n).astype(float),
        "bwd_urg_flags": rng.randint(0, 2, n).astype(float),
        "fwd_header_length": rng.poisson(20, n).astype(float) * 4,
        "bwd_header_length": rng.poisson(20, n).astype(float) * 4,
        "fwd_packets_s": rng.exponential(500, n),
        "bwd_packets_s": rng.exponential(400, n),
        "min_packet_length": rng.uniform(0, 100, n),
        "max_packet_length": rng.uniform(100, 65535, n),
        "packet_length_mean": rng.uniform(50, 1500, n),
        "packet_length_std": rng.exponential(300, n),
        "packet_length_variance": rng.exponential(90000, n),
        "fin_flag_count": rng.randint(0, 5, n).astype(float),
        "syn_flag_count": rng.randint(0, 5, n).astype(float),
        "rst_flag_count": rng.randint(0, 3, n).astype(float),
        "psh_flag_count": rng.randint(0, 5, n).astype(float),
        "ack_flag_count": rng.randint(0, 5, n).astype(float),
        "urg_flag_count": rng.randint(0, 2, n).astype(float),
        "cwe_flag_count": np.zeros(n),  # intentionally constant
        "ece_flag_count": rng.randint(0, 2, n).astype(float),
        "down_up_ratio": rng.uniform(0, 10, n),
        "average_packet_size": rng.uniform(50, 1500, n),
        "avg_fwd_segment_size": rng.uniform(50, 1500, n),
        "avg_bwd_segment_size": rng.uniform(50, 1500, n),
        "init_win_bytes_forward": rng.randint(-1, 65536, n).astype(float),
        "init_win_bytes_backward": rng.randint(-1, 65536, n).astype(float),
        "act_data_pkt_fwd": rng.poisson(5, n).astype(float),
        "min_seg_size_forward": rng.randint(0, 100, n).astype(float),
        "active_mean": rng.exponential(50000, n),
        "active_std": rng.exponential(50000, n),
        "active_max": rng.exponential(100000, n),
        "active_min": rng.uniform(0, 10000, n),
        "idle_mean": rng.exponential(500000, n),
        "idle_std": rng.exponential(500000, n),
        "idle_max": rng.exponential(1000000, n),
        "idle_min": rng.uniform(0, 100000, n),
    })

    # Inject a few NaN and inf values for realism
    df.loc[rng.choice(n, 5, replace=False), "flow_bytes_s"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "flow_packets_s"] = np.inf

    log.info("Generated synthetic dataset: %d rows × %d cols", len(df), len(df.columns))
    return df


def main():
    parser = argparse.ArgumentParser(description="Run AI-NIDS EDA pipeline")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic data even if processed CSV exists",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports at module level
    from src.eda import generate_eda_report
    from src.feature_engineering import (
        build_label_mapping,
        build_preprocessing_pipeline,
        diagnose_nonfinite,
        find_constant_columns,
        find_duplicate_columns,
        save_feature_metadata,
        separate_features_target,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    data_source = "synthetic"

    if not args.synthetic and PROCESSED_CSV.exists():
        log.info("Loading processed data from %s", PROCESSED_CSV)
        df = pd.read_csv(PROCESSED_CSV)
        data_source = "real_cic_ids2017_processed"
    elif not args.synthetic:
        # Try loading raw data through Phase 1 pipeline
        from src.data_loader import discover_csv_files
        raw_files = discover_csv_files()
        if raw_files:
            log.info("Raw CSVs found — running Phase 1 pipeline first...")
            from src.data_loader import load_all_csvs
            from src.preprocess import run_preprocessing
            raw_df, _ = load_all_csvs()
            df, _ = run_preprocessing(raw_df)
            data_source = "real_cic_ids2017_freshly_processed"
        else:
            log.warning("No real data found. Generating synthetic data.")
            df = _generate_synthetic_data(n=5000)
    else:
        log.info("Using synthetic data (--synthetic flag).")
        df = _generate_synthetic_data(n=5000)

    print(f"\n{'='*60}")
    print(f"  Data source: {data_source}")
    print(f"  Dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"{'='*60}\n")

    # -- EDA ------------------------------------------------------------------
    report_path = generate_eda_report(df, data_source=data_source)
    print(f"  EDA report saved -> {report_path}")

    # -- Feature engineering --------------------------------------------------
    X, y = separate_features_target(df)
    label_mapping = build_label_mapping(df)

    # Diagnostics
    constant_cols = find_constant_columns(X)
    duplicate_groups = find_duplicate_columns(X)
    nonfinite = diagnose_nonfinite(X)

    print(f"\n  Features: {X.shape[1]}")
    print(f"  Target: {y.name} ({y.value_counts().to_dict()})")
    print(f"  Constant columns: {constant_cols or 'None'}")
    print(f"  Duplicate column groups: {len(duplicate_groups)}")
    print(f"  Columns with non-finite values: {len(nonfinite)}")

    # Build pipeline (on full data for demonstration -- NOT for production training)
    pipeline, feature_names = build_preprocessing_pipeline(X, scale=True)
    print(f"  Pipeline fitted on {len(feature_names)} numeric features")

    # Save metadata
    meta_path = save_feature_metadata(
        feature_names=feature_names,
        label_mapping=label_mapping,
        constant_cols=constant_cols,
        duplicate_groups=duplicate_groups,
        nonfinite_report=nonfinite,
    )
    print(f"  Feature metadata saved -> {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 2 complete. Reports are in: {REPORTS_DIR}")
    if data_source == "synthetic":
        print("  WARNING: Results are from SYNTHETIC data -- not real CIC-IDS2017.")
        print("    Download the dataset and place CSVs in data/raw/ to run on real data.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
