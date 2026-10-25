"""
test_preprocess.py — Pytest tests for src/preprocess.py

All tests use synthetic data. No real CIC-IDS2017 data required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    SOURCE_FILE_COLUMN,
    NON_FEATURE_COLUMNS,
)
from src.preprocess import (
    convert_numeric_features,
    create_binary_target,
    handle_missing_and_infinite,
    remove_duplicates,
    run_preprocessing,
    validate_labels,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_df(labels, values=None):
    """Quick DataFrame with label column and one numeric feature."""
    n = len(labels)
    return pd.DataFrame({
        LABEL_COLUMN: labels,
        "feature_a": values if values is not None else list(range(n)),
        SOURCE_FILE_COLUMN: ["test.csv"] * n,
    })


# ── label validation ─────────────────────────────────────────────────────────

def test_validate_labels_normalises():
    df = _make_df(["BENIGN", " Bot ", "  DDoS  "])
    df = validate_labels(df)
    assert list(df[LABEL_COLUMN]) == ["benign", "bot", "ddos"]


def test_validate_labels_missing_column():
    df = pd.DataFrame({"feature": [1, 2]})
    with pytest.raises(KeyError, match="label"):
        validate_labels(df)


def test_validate_labels_all_empty():
    df = _make_df(["", "", ""])
    with pytest.raises(ValueError, match="empty"):
        validate_labels(df)


def test_validate_labels_drops_empty_rows():
    df = _make_df(["BENIGN", "", "Bot"])
    df = validate_labels(df)
    assert len(df) == 2


# ── binary target ───────────────────────────────────────────────────────────

def test_benign_maps_to_zero():
    df = _make_df(["benign", "benign"])
    df = validate_labels(df)
    df = create_binary_target(df)
    assert list(df[BINARY_TARGET_COLUMN]) == [0, 0]


def test_attack_maps_to_one():
    df = _make_df(["benign", "bot", "ddos"])
    df = validate_labels(df)
    df = create_binary_target(df)
    assert list(df[BINARY_TARGET_COLUMN]) == [0, 1, 1]


def test_unknown_label_maps_to_malicious():
    """Unknown labels should NOT default to benign."""
    df = _make_df(["benign", "never_seen_before"])
    df = validate_labels(df)
    df = create_binary_target(df)
    assert df[BINARY_TARGET_COLUMN].iloc[1] == 1


def test_original_label_preserved():
    df = _make_df(["BENIGN", "Bot"])
    df = validate_labels(df)
    df = create_binary_target(df)
    assert list(df[ORIGINAL_LABEL_COLUMN]) == ["benign", "bot"]


# ── missing and infinite values ──────────────────────────────────────────────

def test_inf_replaced():
    df = _make_df(["benign", "bot"], values=[np.inf, -np.inf])
    df = validate_labels(df)
    df = create_binary_target(df)
    df, stats = handle_missing_and_infinite(df, strategy="median")
    assert stats["infinite_replaced"] == 2
    assert not np.isinf(df["feature_a"]).any()


def test_nan_filled_median():
    df = _make_df(["benign", "bot", "ddos"], values=[10.0, np.nan, 30.0])
    df = validate_labels(df)
    df = create_binary_target(df)
    df, stats = handle_missing_and_infinite(df, strategy="median")
    assert stats["nan_before"] == 1
    assert not df["feature_a"].isna().any()
    # Median of [10, 30] = 20
    assert df["feature_a"].iloc[1] == 20.0


def test_nan_filled_mean():
    df = _make_df(["benign", "bot", "ddos"], values=[10.0, np.nan, 30.0])
    df = validate_labels(df)
    df = create_binary_target(df)
    df, stats = handle_missing_and_infinite(df, strategy="mean")
    assert df["feature_a"].iloc[1] == 20.0


def test_nan_drop_strategy():
    df = _make_df(["benign", "bot", "ddos"], values=[10.0, np.nan, 30.0])
    df = validate_labels(df)
    df = create_binary_target(df)
    df, stats = handle_missing_and_infinite(df, strategy="drop")
    assert len(df) == 2


def test_unknown_strategy_raises():
    df = _make_df(["benign"], values=[1.0])
    df = validate_labels(df)
    df = create_binary_target(df)
    with pytest.raises(ValueError, match="Unknown"):
        handle_missing_and_infinite(df, strategy="magic")


# ── numeric conversion ──────────────────────────────────────────────────────

def test_numeric_conversion_coerces_bad_values():
    df = pd.DataFrame({
        LABEL_COLUMN: ["benign", "bot"],
        "feature_x": ["100", "not_a_number"],
        SOURCE_FILE_COLUMN: ["t.csv", "t.csv"],
    })
    df = validate_labels(df)
    df = create_binary_target(df)
    df, report = convert_numeric_features(df)
    assert df["feature_x"].iloc[0] == 100.0
    assert pd.isna(df["feature_x"].iloc[1])
    assert len(report["coercion_warnings"]) == 1


# ── duplicates ──────────────────────────────────────────────────────────────

def test_duplicates_removed():
    df = _make_df(["benign", "benign"], values=[1, 1])
    # Make exact duplicates (4 rows total)
    df = pd.concat([df, df], ignore_index=True)
    df, n = remove_duplicates(df)
    assert n == 3
    assert len(df) == 1


# ── full pipeline ───────────────────────────────────────────────────────────

def test_full_pipeline_deterministic(tmp_path: Path):
    """Running twice on identical input yields identical output."""
    df = _make_df(["BENIGN", "Bot", "DDoS", "BENIGN"], values=[1, 2, 3, 1])
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    sum1 = tmp_path / "sum1.json"
    sum2 = tmp_path / "sum2.json"

    r1, _ = run_preprocessing(df.copy(), output_csv=out1, output_summary=sum1)
    r2, _ = run_preprocessing(df.copy(), output_csv=out2, output_summary=sum2)

    pd.testing.assert_frame_equal(r1, r2)


def test_pipeline_saves_output(tmp_path: Path):
    df = _make_df(["BENIGN", "Bot"], values=[10, 20])
    out_csv = tmp_path / "processed.csv"
    out_json = tmp_path / "summary.json"
    run_preprocessing(df, output_csv=out_csv, output_summary=out_json)

    assert out_csv.exists()
    assert out_json.exists()

    saved = pd.read_csv(out_csv)
    assert BINARY_TARGET_COLUMN in saved.columns
    assert ORIGINAL_LABEL_COLUMN in saved.columns


def test_pipeline_empty_after_processing():
    """If all rows have empty labels, pipeline should raise."""
    df = _make_df(["", ""])
    with pytest.raises(ValueError, match="empty"):
        run_preprocessing(df)


def test_non_feature_columns_excluded():
    df = _make_df(["BENIGN", "Bot"], values=[1, 2])
    df = validate_labels(df)
    df = create_binary_target(df)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    assert BINARY_TARGET_COLUMN not in feature_cols
    assert ORIGINAL_LABEL_COLUMN not in feature_cols
    assert SOURCE_FILE_COLUMN not in feature_cols
    assert LABEL_COLUMN not in feature_cols
