"""
validate_data.py — Post-processing validation checks for AI-NIDS.

Runs a battery of checks on the processed DataFrame and produces a
human-readable report. Checks are classified as:

* **ERROR** — Fatal: the dataset cannot be used for modelling.
* **WARNING** — Non-fatal: the dataset can be used but something deserves
  attention.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    REPORTS_DIR,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.validate_data")


# ── individual checks ───────────────────────────────────────────────────────

def check_not_empty(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """ERROR if dataset has zero rows or zero columns."""
    issues = []
    if df.empty:
        issues.append(("ERROR", "Dataset is empty (0 rows)."))
    if df.columns.size == 0:
        issues.append(("ERROR", "Dataset has no columns."))
    return issues


def check_required_columns(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """ERROR if required target / label columns are missing."""
    issues = []
    for col in (BINARY_TARGET_COLUMN, ORIGINAL_LABEL_COLUMN):
        if col not in df.columns:
            issues.append(("ERROR", f"Required column '{col}' is missing."))
    return issues


def check_target_values(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """ERROR / WARNING on unexpected binary-target values."""
    issues = []
    if BINARY_TARGET_COLUMN not in df.columns:
        return issues  # already flagged by check_required_columns

    unique_vals = set(df[BINARY_TARGET_COLUMN].dropna().unique())
    expected = {0, 1}
    if not unique_vals:
        issues.append(("ERROR", "Binary target column is entirely NaN."))
    elif unique_vals - expected:
        issues.append(("ERROR", f"Unexpected target values: {unique_vals - expected}"))
    if unique_vals == {0}:
        issues.append(("WARNING", "Only benign samples — no malicious class."))
    elif unique_vals == {1}:
        issues.append(("WARNING", "Only malicious samples — no benign class."))
    return issues


def check_missing_values(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """WARNING if any column still has NaN values."""
    issues = []
    nan_counts = df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        total = int(cols_with_nan.sum())
        issues.append((
            "WARNING",
            f"{total} missing value(s) remain across {len(cols_with_nan)} column(s): "
            f"{dict(cols_with_nan.head(10))}",
        ))
    return issues


def check_infinite_values(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """WARNING if any numeric column contains ±inf."""
    issues = []
    numeric = df.select_dtypes(include=[np.number])
    inf_counts = numeric.isin([np.inf, -np.inf]).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if not cols_with_inf.empty:
        total = int(cols_with_inf.sum())
        issues.append((
            "WARNING",
            f"{total} infinite value(s) in {len(cols_with_inf)} column(s): "
            f"{dict(cols_with_inf.head(10))}",
        ))
    return issues


def check_duplicates(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """WARNING if duplicate rows exist."""
    issues = []
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        issues.append(("WARNING", f"{n_dupes} duplicate row(s) found."))
    return issues


def check_class_imbalance(
    df: pd.DataFrame,
    minority_threshold: float = 0.05,
) -> List[Tuple[str, str]]:
    """WARNING if minority class < threshold fraction."""
    issues = []
    if BINARY_TARGET_COLUMN not in df.columns:
        return issues
    counts = df[BINARY_TARGET_COLUMN].value_counts()
    total = counts.sum()
    if total == 0:
        return issues
    minority = counts.min()
    ratio = minority / total
    if ratio < minority_threshold:
        issues.append((
            "WARNING",
            f"Class imbalance: minority class has {minority} samples "
            f"({ratio:.2%} of total). Consider resampling in Phase 2.",
        ))
    # Always report counts
    issues.append((
        "INFO",
        f"Class distribution — {dict(counts)}",
    ))
    return issues


def check_unexpected_labels(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """WARNING if original labels contain values outside known set."""
    issues = []
    if ORIGINAL_LABEL_COLUMN not in df.columns:
        return issues

    from src.config import ALL_KNOWN_LABELS
    unique = set(df[ORIGINAL_LABEL_COLUMN].dropna().unique())
    unknown = unique - ALL_KNOWN_LABELS
    if unknown:
        issues.append((
            "WARNING",
            f"Unexpected original labels found: {unknown}",
        ))
    return issues


def check_invalid_numeric(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """WARNING if object-typed columns remain in feature columns."""
    issues = []
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    obj_cols = [c for c in feature_cols if df[c].dtype == object]
    if obj_cols:
        issues.append((
            "WARNING",
            f"{len(obj_cols)} feature column(s) are still non-numeric (object): "
            f"{obj_cols[:10]}",
        ))
    return issues


def check_target_leakage(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """ERROR if the target column appears in the feature matrix."""
    issues = []
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    leaked = NON_FEATURE_COLUMNS & set(feature_cols)
    if leaked:
        issues.append((
            "ERROR",
            f"Target / metadata columns in feature matrix: {leaked}",
        ))
    return issues


# ── orchestrator ─────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_not_empty,
    check_required_columns,
    check_target_values,
    check_missing_values,
    check_infinite_values,
    check_duplicates,
    check_class_imbalance,
    check_unexpected_labels,
    check_invalid_numeric,
    check_target_leakage,
]


def run_validation(
    df: pd.DataFrame,
    report_path: Optional[Path] = None,
) -> Tuple[bool, Dict]:
    """Run all validation checks.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset.
    report_path : Path | None
        Where to write the human-readable report (default: reports/validation_report.txt).

    Returns
    -------
    (passed, report_dict)
        ``passed`` is False if any ERROR-level issue was found.
    """
    report_path = report_path or (REPORTS_DIR / "validation_report.txt")

    all_issues: List[Tuple[str, str]] = []
    for check_fn in ALL_CHECKS:
        all_issues.extend(check_fn(df))

    errors = [msg for lvl, msg in all_issues if lvl == "ERROR"]
    warnings = [msg for lvl, msg in all_issues if lvl == "WARNING"]
    infos = [msg for lvl, msg in all_issues if lvl == "INFO"]

    passed = len(errors) == 0

    # Log
    for lvl, msg in all_issues:
        getattr(log, lvl.lower() if lvl != "ERROR" else "error")(msg)

    # Build human-readable report
    lines = [
        "=" * 70,
        " AI-NIDS DATA VALIDATION REPORT",
        f" Generated: {datetime.now().isoformat()}",
        "=" * 70,
        "",
        f"Overall result: {'PASS ✓' if passed else 'FAIL ✗'}",
        f"Errors:   {len(errors)}",
        f"Warnings: {len(warnings)}",
        f"Info:     {len(infos)}",
        "",
    ]

    if errors:
        lines.append("─── ERRORS (fatal) ───")
        for e in errors:
            lines.append(f"  ✗ {e}")
        lines.append("")

    if warnings:
        lines.append("─── WARNINGS ───")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    if infos:
        lines.append("─── INFO ───")
        for i in infos:
            lines.append(f"  ℹ {i}")
        lines.append("")

    lines.append("=" * 70)
    report_text = "\n".join(lines)

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Validation report saved → %s", report_path)

    report_dict = {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
    }
    return passed, report_dict
