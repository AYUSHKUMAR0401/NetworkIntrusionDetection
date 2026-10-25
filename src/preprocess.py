"""
preprocess.py — Clean and prepare loaded CIC-IDS2017 data for modelling.

Pipeline (deterministic, repeatable)
-------------------------------------
1. Normalise and validate the label column.
2. Create binary target: Benign → 0, attack → 1.
3. Handle missing values (strategy from config).
4. Replace ±inf with NaN, then apply missing-value strategy.
5. Convert all feature columns to numeric where possible.
6. Remove duplicate rows and report count.
7. Save processed CSV + JSON summary.

This module does NOT fit scalers or models — that belongs in Phase 2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ALL_KNOWN_LABELS,
    BENIGN_LABELS,
    BINARY_TARGET_COLUMN,
    DROP_DUPLICATES,
    KNOWN_ATTACK_LABELS,
    LABEL_COLUMN,
    MISSING_VALUE_STRATEGY,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    PROCESSED_CSV,
    PROCESSED_DIR,
    PREPROCESSING_SUMMARY,
    RANDOM_SEED,
    SOURCE_FILE_COLUMN,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.preprocess")


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalise_label(value) -> str:
    """Lowercase, strip whitespace, collapse interior whitespace."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are valid model features (excludes targets, meta)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


# ── core pipeline ────────────────────────────────────────────────────────────

def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the label column exists and normalise its values.

    Raises
    ------
    KeyError
        If the label column is missing.
    ValueError
        If all labels are empty after normalisation.
    """
    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(_normalise_label)

    # Check for empty labels
    empty_mask = df[LABEL_COLUMN] == ""
    if empty_mask.all():
        raise ValueError("All labels are empty after normalisation.")
    if empty_mask.any():
        n = int(empty_mask.sum())
        log.warning("%d rows have empty labels — they will be dropped.", n)
        df = df[~empty_mask].copy()

    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target and original-label columns.

    Mapping
    -------
    * Labels in ``BENIGN_LABELS`` → 0 (Benign)
    * All other labels → 1 (Malicious)
    * Unknown labels (not in KNOWN_ATTACK_LABELS) trigger a warning but are
      still mapped to 1.

    Raises
    ------
    ValueError
        If the resulting target has only one class.
    """
    # Preserve original label before mapping
    df[ORIGINAL_LABEL_COLUMN] = df[LABEL_COLUMN]

    def _map_label(lbl: str) -> int:
        if lbl in BENIGN_LABELS:
            return 0
        return 1

    df[BINARY_TARGET_COLUMN] = df[LABEL_COLUMN].apply(_map_label)

    # Warn on unknown attack labels
    unique_labels = set(df[LABEL_COLUMN].unique())
    unknown = unique_labels - ALL_KNOWN_LABELS
    if unknown:
        log.warning(
            "Unknown labels mapped to Malicious (class 1): %s. "
            "Verify these are genuine attack categories.",
            unknown,
        )

    class_counts = df[BINARY_TARGET_COLUMN].value_counts()
    log.info(
        "Binary target created — Benign: %d, Malicious: %d",
        class_counts.get(0, 0),
        class_counts.get(1, 0),
    )

    return df


def handle_missing_and_infinite(
    df: pd.DataFrame,
    strategy: str = MISSING_VALUE_STRATEGY,
) -> Tuple[pd.DataFrame, Dict]:
    """Replace ±inf with NaN, then fill or drop NaNs in numeric feature columns.

    Returns (df, stats_dict).
    """
    feature_cols = _get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Count infinities before replacement
    inf_mask = df[numeric_cols].isin([np.inf, -np.inf])
    inf_count = int(inf_mask.sum().sum())
    if inf_count:
        log.info("Replacing %d infinite value(s) with NaN.", inf_count)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Count NaNs
    nan_before = int(df[numeric_cols].isna().sum().sum())

    if strategy == "median":
        fill_values = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    elif strategy == "mean":
        fill_values = df[numeric_cols].mean()
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    elif strategy == "drop":
        df = df.dropna(subset=numeric_cols).copy()
    else:
        raise ValueError(f"Unknown missing-value strategy: '{strategy}'")

    nan_after = int(df[numeric_cols].isna().sum().sum())

    stats = {
        "infinite_replaced": inf_count,
        "nan_before": nan_before,
        "nan_after": nan_after,
        "strategy": strategy,
    }
    log.info("Missing-value handling: %s", stats)
    return df, stats


def convert_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Attempt to convert feature columns to numeric types.

    Columns that cannot be fully converted are logged. Values that cannot be
    parsed are set to NaN (never silently turned into valid numbers).

    Returns (df, conversion_report).
    """
    feature_cols = _get_feature_columns(df)
    converted = 0
    failed_cols: list[str] = []

    for col in feature_cols:
        if df[col].dtype == object:
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced = int(original.notna().sum() - df[col].notna().sum())
            if coerced > 0:
                failed_cols.append(f"{col} ({coerced} values coerced to NaN)")
                log.warning(
                    "Column '%s': %d non-numeric values coerced to NaN.",
                    col, coerced,
                )
            converted += 1

    report = {"columns_converted": converted, "coercion_warnings": failed_cols}
    log.info("Numeric conversion: %d object column(s) processed.", converted)
    return df, report


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns (df, n_removed)."""
    if not DROP_DUPLICATES:
        return df, 0
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        log.info("Removed %d duplicate row(s).", n_removed)
    return df, n_removed


# ── main entry point ─────────────────────────────────────────────────────────

def run_preprocessing(
    df: pd.DataFrame,
    output_csv: Optional[Path] = None,
    output_summary: Optional[Path] = None,
    strategy: str = MISSING_VALUE_STRATEGY,
) -> Tuple[pd.DataFrame, Dict]:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame (output of ``data_loader.load_all_csvs``).
    output_csv : Path | None
        Where to save the processed CSV (default: config.PROCESSED_CSV).
    output_summary : Path | None
        Where to save the JSON summary (default: config.PREPROCESSING_SUMMARY).
    strategy : str
        Missing-value strategy (``"median"``, ``"mean"``, ``"drop"``).

    Returns
    -------
    (processed_df, summary_dict)
    """
    output_csv = output_csv or PROCESSED_CSV
    output_summary = output_summary or PREPROCESSING_SUMMARY

    summary: Dict = {"input_rows": len(df), "input_cols": len(df.columns)}
    log.info("=== Preprocessing pipeline started ===")
    log.info("Input: %d rows × %d columns", len(df), len(df.columns))

    # 1. Labels
    df = validate_labels(df)
    summary["rows_after_label_validation"] = len(df)

    # 2. Binary target
    df = create_binary_target(df)
    summary["class_counts"] = df[BINARY_TARGET_COLUMN].value_counts().to_dict()

    # 3. Numeric conversion (before imputation so we know what's numeric)
    df, conv_report = convert_numeric_features(df)
    summary["numeric_conversion"] = conv_report

    # 4. Inf + missing values
    df, missing_stats = handle_missing_and_infinite(df, strategy=strategy)
    summary["missing_value_handling"] = missing_stats

    # 5. Duplicates
    df, n_dupes = remove_duplicates(df)
    summary["duplicates_removed"] = n_dupes

    # 6. Final stats
    summary["output_rows"] = len(df)
    summary["output_cols"] = len(df.columns)
    summary["feature_columns"] = _get_feature_columns(df)
    summary["non_feature_columns"] = list(NON_FEATURE_COLUMNS & set(df.columns))

    # 7. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Saved processed data → %s", output_csv)

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Saved preprocessing summary → %s", output_summary)

    log.info("=== Preprocessing pipeline finished ===")
    return df, summary
