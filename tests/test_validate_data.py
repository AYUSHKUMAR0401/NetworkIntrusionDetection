"""
test_validate_data.py — Pytest tests for src/validate_data.py

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
from src.validate_data import (
    check_class_imbalance,
    check_duplicates,
    check_infinite_values,
    check_invalid_numeric,
    check_missing_values,
    check_not_empty,
    check_required_columns,
    check_target_leakage,
    check_target_values,
    check_unexpected_labels,
    run_validation,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _good_df(n: int = 4) -> pd.DataFrame:
    """Minimal valid processed DataFrame."""
    return pd.DataFrame({
        BINARY_TARGET_COLUMN: [0, 1] * (n // 2),
        ORIGINAL_LABEL_COLUMN: ["benign", "bot"] * (n // 2),
        LABEL_COLUMN: ["benign", "bot"] * (n // 2),
        SOURCE_FILE_COLUMN: ["test.csv"] * n,
        "feature_a": list(range(n)),
        "feature_b": [float(x) for x in range(n)],
    })


# ── check_not_empty ─────────────────────────────────────────────────────────

def test_empty_df_flagged():
    issues = check_not_empty(pd.DataFrame())
    assert any(lvl == "ERROR" for lvl, _ in issues)


def test_non_empty_passes():
    assert check_not_empty(_good_df()) == []


# ── check_required_columns ─────────────────────────────────────────────────

def test_missing_target_column():
    df = _good_df().drop(columns=[BINARY_TARGET_COLUMN])
    issues = check_required_columns(df)
    assert any("is_malicious" in msg for _, msg in issues)


def test_all_required_present():
    assert check_required_columns(_good_df()) == []


# ── check_target_values ────────────────────────────────────────────────────

def test_valid_target_values():
    assert check_target_values(_good_df()) == []


def test_single_class_warning():
    df = _good_df()
    df[BINARY_TARGET_COLUMN] = 0
    issues = check_target_values(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── check_missing_values ───────────────────────────────────────────────────

def test_no_missing():
    assert check_missing_values(_good_df()) == []


def test_with_nan():
    df = _good_df()
    df.loc[0, "feature_a"] = np.nan
    issues = check_missing_values(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── check_infinite ──────────────────────────────────────────────────────────

def test_no_inf():
    assert check_infinite_values(_good_df()) == []


def test_with_inf():
    df = _good_df()
    df.loc[0, "feature_a"] = np.inf
    issues = check_infinite_values(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── check_duplicates ───────────────────────────────────────────────────────

def test_no_duplicates():
    assert check_duplicates(_good_df()) == []


def test_with_duplicates():
    df = _good_df()
    df = pd.concat([df, df], ignore_index=True)
    issues = check_duplicates(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── class imbalance ─────────────────────────────────────────────────────────

def test_balanced():
    issues = check_class_imbalance(_good_df())
    # Should have INFO but no WARNING
    assert all(lvl != "WARNING" for lvl, _ in issues if "imbalance" in _ .lower())


def test_imbalanced():
    df = pd.DataFrame({
        BINARY_TARGET_COLUMN: [0] * 100 + [1],
        ORIGINAL_LABEL_COLUMN: ["benign"] * 100 + ["bot"],
    })
    issues = check_class_imbalance(df, minority_threshold=0.05)
    assert any(lvl == "WARNING" and "imbalance" in msg.lower() for lvl, msg in issues)


# ── unexpected labels ──────────────────────────────────────────────────────

def test_known_labels_pass():
    assert check_unexpected_labels(_good_df()) == []


def test_unknown_labels_warned():
    df = _good_df()
    df.loc[0, ORIGINAL_LABEL_COLUMN] = "totally_new_attack"
    issues = check_unexpected_labels(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── invalid numeric ────────────────────────────────────────────────────────

def test_all_numeric_features():
    assert check_invalid_numeric(_good_df()) == []


def test_object_feature_warned():
    df = _good_df()
    df["feature_c"] = ["a", "b", "c", "d"]
    issues = check_invalid_numeric(df)
    assert any(lvl == "WARNING" for lvl, _ in issues)


# ── target leakage ─────────────────────────────────────────────────────────

def test_no_leakage():
    assert check_target_leakage(_good_df()) == []


# ── full validation run ────────────────────────────────────────────────────

def test_run_validation_pass(tmp_path: Path):
    report_path = tmp_path / "val_report.txt"
    passed, report = run_validation(_good_df(), report_path=report_path)
    assert passed
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "PASS" in text


def test_run_validation_fail_on_missing_col(tmp_path: Path):
    df = _good_df().drop(columns=[BINARY_TARGET_COLUMN])
    passed, report = run_validation(df, report_path=tmp_path / "val.txt")
    assert not passed
    assert len(report["errors"]) > 0
