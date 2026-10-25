"""
test_eda.py — Pytest tests for src/eda.py

All tests use synthetic data. No real CIC-IDS2017 data required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    SOURCE_FILE_COLUMN,
)
from src.eda import (
    class_distribution,
    generate_eda_report,
    identify_suspicious_columns,
    inspect_dataset,
    missing_value_summary,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _synth_df(n: int = 200) -> pd.DataFrame:
    """Synthetic processed DataFrame for EDA testing."""
    rng = np.random.RandomState(42)
    labels = rng.choice(["benign", "bot", "ddos"], size=n, p=[0.7, 0.2, 0.1])
    return pd.DataFrame({
        LABEL_COLUMN: labels,
        ORIGINAL_LABEL_COLUMN: labels,
        BINARY_TARGET_COLUMN: np.where(labels == "benign", 0, 1),
        SOURCE_FILE_COLUMN: ["test.csv"] * n,
        "flow_duration": rng.exponential(1000, n),
        "total_fwd_packets": rng.poisson(10, n).astype(float),
        "flow_bytes_s": rng.exponential(5000, n),
        "fwd_packet_length_mean": rng.uniform(50, 1500, n),
    })


# ── inspect ─────────────────────────────────────────────────────────────────

def test_inspect_dataset():
    df = _synth_df()
    info = inspect_dataset(df)
    assert info["n_rows"] == 200
    assert info["numeric_columns"] > 0
    assert "columns" in info


# ── missing value summary ──────────────────────────────────────────────────

def test_missing_no_issues():
    df = _synth_df()
    summary = missing_value_summary(df)
    assert summary.empty


def test_missing_with_nan():
    df = _synth_df()
    df.loc[0, "flow_duration"] = np.nan
    df.loc[1, "flow_bytes_s"] = np.inf
    summary = missing_value_summary(df)
    assert len(summary) >= 1


# ── class distribution ─────────────────────────────────────────────────────

def test_class_distribution_binary():
    df = _synth_df()
    dist = class_distribution(df)
    assert "binary" in dist
    assert 0 in dist["binary"]["counts"]
    assert 1 in dist["binary"]["counts"]


def test_class_distribution_multiclass():
    df = _synth_df()
    dist = class_distribution(df)
    assert "multiclass" in dist
    assert dist["multiclass"]["n_classes"] == 3


# ── suspicious columns ────────────────────────────────────────────────────

def test_no_suspicious_basic():
    df = _synth_df()
    suspicious = identify_suspicious_columns(df)
    assert suspicious["potential_timestamps"] == []


def test_timestamp_flagged():
    df = _synth_df()
    df["timestamp"] = pd.date_range("2017-07-03", periods=len(df), freq="s")
    suspicious = identify_suspicious_columns(df)
    assert "timestamp" in suspicious["potential_timestamps"]


# ── plotting (smoke tests — check files are created) ────────────────────────

def test_plot_class_distribution(tmp_path: Path):
    df = _synth_df()
    paths = plot_class_distribution(df, output_dir=tmp_path)
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_plot_feature_distributions(tmp_path: Path):
    df = _synth_df()
    path = plot_feature_distributions(df, output_dir=tmp_path)
    assert path is not None
    assert path.exists()


def test_plot_correlation_heatmap(tmp_path: Path):
    df = _synth_df()
    path = plot_correlation_heatmap(df, output_dir=tmp_path)
    assert path is not None
    assert path.exists()


# ── full EDA report ────────────────────────────────────────────────────────

def test_generate_eda_report(tmp_path: Path):
    df = _synth_df()
    report_path = generate_eda_report(
        df, data_source="synthetic_test", output_dir=tmp_path
    )
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "EXPLORATORY DATA ANALYSIS" in text
    assert "synthetic_test" in text

    # JSON report should also exist
    json_path = tmp_path / "eda_report.json"
    assert json_path.exists()
