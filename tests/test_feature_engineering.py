"""
test_feature_engineering.py — Pytest tests for src/feature_engineering.py

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
from src.feature_engineering import (
    apply_preprocessing_pipeline,
    build_label_mapping,
    build_preprocessing_pipeline,
    diagnose_nonfinite,
    find_constant_columns,
    find_duplicate_columns,
    save_feature_metadata,
    separate_features_target,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _processed_df(n: int = 100) -> pd.DataFrame:
    """Create a realistic processed DataFrame with multiple features."""
    rng = np.random.RandomState(42)
    labels = rng.choice(["benign", "bot", "ddos"], size=n, p=[0.7, 0.2, 0.1])
    binary = np.where(labels == "benign", 0, 1)

    return pd.DataFrame({
        LABEL_COLUMN: labels,
        ORIGINAL_LABEL_COLUMN: labels,
        BINARY_TARGET_COLUMN: binary,
        SOURCE_FILE_COLUMN: ["test.csv"] * n,
        "flow_duration": rng.exponential(1000, n),
        "total_fwd_packets": rng.poisson(10, n).astype(float),
        "total_bwd_packets": rng.poisson(8, n).astype(float),
        "fwd_packet_length_mean": rng.uniform(50, 1500, n),
        "bwd_packet_length_mean": rng.uniform(50, 1500, n),
        "flow_bytes_s": rng.exponential(5000, n),
        "flow_packets_s": rng.exponential(100, n),
        "fwd_iat_mean": rng.exponential(500, n),
    })


# ── target / feature separation ─────────────────────────────────────────────

class TestSeparateFeaturesTarget:

    def test_basic_separation(self):
        df = _processed_df()
        X, y = separate_features_target(df)

        assert BINARY_TARGET_COLUMN not in X.columns
        assert ORIGINAL_LABEL_COLUMN not in X.columns
        assert LABEL_COLUMN not in X.columns
        assert SOURCE_FILE_COLUMN not in X.columns
        assert len(y) == len(df)

    def test_target_values_match(self):
        df = _processed_df()
        X, y = separate_features_target(df)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            df[BINARY_TARGET_COLUMN].reset_index(drop=True),
        )

    def test_feature_columns_are_clean(self):
        df = _processed_df()
        X, y = separate_features_target(df)
        leaked = NON_FEATURE_COLUMNS & set(X.columns)
        assert leaked == set(), f"Leaked columns: {leaked}"

    def test_missing_target_raises(self):
        df = _processed_df().drop(columns=[BINARY_TARGET_COLUMN])
        with pytest.raises(KeyError, match="is_malicious"):
            separate_features_target(df)

    def test_feature_column_consistency(self):
        """Same input → same feature columns, same order."""
        df = _processed_df(50)
        X1, _ = separate_features_target(df.copy())
        X2, _ = separate_features_target(df.copy())
        assert list(X1.columns) == list(X2.columns)


# ── label mapping ───────────────────────────────────────────────────────────

class TestLabelMapping:

    def test_mapping_contains_all_labels(self):
        df = _processed_df()
        mapping = build_label_mapping(df)
        assert "benign" in mapping
        assert mapping["benign"] == 0
        for lbl in ["bot", "ddos"]:
            assert mapping[lbl] == 1

    def test_mapping_missing_columns_raises(self):
        df = pd.DataFrame({"feature": [1, 2]})
        with pytest.raises(KeyError):
            build_label_mapping(df)


# ── constant columns ───────────────────────────────────────────────────────

class TestConstantColumns:

    def test_detect_constant(self):
        df = _processed_df()
        df["constant_col"] = 42.0
        X, _ = separate_features_target(df)
        constants = find_constant_columns(X)
        assert "constant_col" in constants

    def test_no_constant(self):
        df = _processed_df()
        X, _ = separate_features_target(df)
        constants = find_constant_columns(X)
        assert constants == []


# ── duplicate columns ──────────────────────────────────────────────────────

class TestDuplicateColumns:

    def test_detect_duplicate_columns(self):
        df = _processed_df()
        df["flow_duration_copy"] = df["flow_duration"]
        X, _ = separate_features_target(df)
        groups = find_duplicate_columns(X)
        assert len(groups) >= 1
        found = False
        for g in groups:
            if "flow_duration" in g and "flow_duration_copy" in g:
                found = True
        assert found

    def test_no_duplicates(self):
        df = _processed_df()
        X, _ = separate_features_target(df)
        groups = find_duplicate_columns(X)
        assert groups == []


# ── non-finite diagnostics ──────────────────────────────────────────────────

class TestNonfinite:

    def test_detect_nan(self):
        df = _processed_df()
        df.loc[0, "flow_duration"] = np.nan
        X, _ = separate_features_target(df)
        report = diagnose_nonfinite(X)
        assert "flow_duration" in report
        assert report["flow_duration"]["nan"] >= 1

    def test_detect_inf(self):
        df = _processed_df()
        df.loc[0, "flow_bytes_s"] = np.inf
        df.loc[1, "flow_bytes_s"] = -np.inf
        X, _ = separate_features_target(df)
        report = diagnose_nonfinite(X)
        assert "flow_bytes_s" in report
        assert report["flow_bytes_s"]["posinf"] == 1
        assert report["flow_bytes_s"]["neginf"] == 1

    def test_clean_data(self):
        df = _processed_df()
        X, _ = separate_features_target(df)
        report = diagnose_nonfinite(X)
        assert report == {}


# ── preprocessing pipeline ──────────────────────────────────────────────────

class TestPreprocessingPipeline:

    def test_fit_and_transform(self):
        df = _processed_df(200)
        X, y = separate_features_target(df)
        # Inject some NaNs
        X.loc[0, "flow_duration"] = np.nan
        X.loc[5, "total_fwd_packets"] = np.nan

        pipeline, feature_names = build_preprocessing_pipeline(X)

        transformed = apply_preprocessing_pipeline(pipeline, X, feature_names)
        assert transformed.shape[0] == X.shape[0]
        assert list(transformed.columns) == feature_names
        # No NaN after imputation
        assert transformed.isna().sum().sum() == 0

    def test_train_test_independence(self):
        """Pipeline fitted on train must not use test statistics."""
        df = _processed_df(200)
        X, y = separate_features_target(df)

        # Simulate train/test split
        X_train = X.iloc[:150].copy()
        X_test = X.iloc[150:].copy()

        pipeline, feature_names = build_preprocessing_pipeline(X_train)

        train_transformed = apply_preprocessing_pipeline(pipeline, X_train, feature_names)
        test_transformed = apply_preprocessing_pipeline(pipeline, X_test, feature_names)

        # Train mean should be ~0, test mean may differ
        train_means = train_transformed.mean().abs()
        assert (train_means < 0.1).all(), "Train means should be near 0 after scaling"

        # Both should have same number of features
        assert list(train_transformed.columns) == list(test_transformed.columns)

    def test_missing_column_raises(self):
        df = _processed_df(100)
        X, _ = separate_features_target(df)
        pipeline, feature_names = build_preprocessing_pipeline(X)

        X_missing = X.drop(columns=["flow_duration"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            apply_preprocessing_pipeline(pipeline, X_missing, feature_names)

    def test_reproducibility(self):
        """Same data → same pipeline output."""
        df = _processed_df(100)
        X, _ = separate_features_target(df)

        pipe1, names1 = build_preprocessing_pipeline(X.copy())
        pipe2, names2 = build_preprocessing_pipeline(X.copy())

        out1 = apply_preprocessing_pipeline(pipe1, X.copy(), names1)
        out2 = apply_preprocessing_pipeline(pipe2, X.copy(), names2)

        pd.testing.assert_frame_equal(out1, out2)

    def test_no_target_leakage_in_pipeline(self):
        """Feature names from pipeline must not include target columns."""
        df = _processed_df()
        X, _ = separate_features_target(df)
        _, feature_names = build_preprocessing_pipeline(X)

        leaked = NON_FEATURE_COLUMNS & set(feature_names)
        assert leaked == set(), f"Target leaked into pipeline features: {leaked}"


# ── metadata persistence ───────────────────────────────────────────────────

class TestMetadata:

    def test_save_and_read(self, tmp_path: Path):
        import json

        path = save_feature_metadata(
            feature_names=["a", "b", "c"],
            label_mapping={"benign": 0, "bot": 1},
            constant_cols=[],
            duplicate_groups=[],
            nonfinite_report={},
            output_path=tmp_path / "meta.json",
        )
        assert path.exists()
        with open(path) as f:
            meta = json.load(f)
        assert meta["n_features"] == 3
        assert meta["label_mapping"]["benign"] == 0
