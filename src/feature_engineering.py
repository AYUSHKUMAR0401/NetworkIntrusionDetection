"""
feature_engineering.py — Reusable feature-engineering module for AI-NIDS.

Responsibilities
----------------
* Separate target labels from model features.
* Identify and report constant / near-constant and duplicate feature columns.
* Detect non-finite numeric values.
* Handle invalid and missing values via documented policies.
* Preserve a mapping between original attack labels and binary labels.
* Exclude the target and any direct label leakage from the feature matrix.
* Build a scikit-learn preprocessing pipeline (fitted on training data only).

This module does NOT arbitrarily delete legitimate network features based
only on their names — column removal is data-driven (constant, duplicate,
or non-numeric after best-effort conversion).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    PROCESSED_DIR,
    RANDOM_SEED,
    SOURCE_FILE_COLUMN,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.feature_engineering")


# ─── Target / feature separation ────────────────────────────────────────────

def separate_features_target(
    df: pd.DataFrame,
    target_col: str = BINARY_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into feature matrix X and target vector y.

    All columns in ``NON_FEATURE_COLUMNS`` are excluded from X.

    Raises
    ------
    KeyError
        If *target_col* is missing from *df*.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Sanity: ensure no target / label leakage
    leaked = NON_FEATURE_COLUMNS & set(X.columns)
    if leaked:
        raise ValueError(f"Target / metadata leaked into feature matrix: {leaked}")

    log.info("Separated features (%d cols) and target '%s' (%d samples).",
             X.shape[1], target_col, len(y))
    return X, y


# ─── Label mapping ──────────────────────────────────────────────────────────

def build_label_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """Return a dict mapping each unique original_label → binary target value.

    Useful for documentation and reproducibility.
    """
    if ORIGINAL_LABEL_COLUMN not in df.columns or BINARY_TARGET_COLUMN not in df.columns:
        raise KeyError("Need both original_label and is_malicious columns.")

    mapping = (
        df[[ORIGINAL_LABEL_COLUMN, BINARY_TARGET_COLUMN]]
        .drop_duplicates()
        .set_index(ORIGINAL_LABEL_COLUMN)[BINARY_TARGET_COLUMN]
        .to_dict()
    )
    log.info("Label mapping (%d labels): %s", len(mapping), mapping)
    return mapping


# ─── Column diagnostics ─────────────────────────────────────────────────────

def find_constant_columns(
    X: pd.DataFrame,
    threshold: float = 0.0,
) -> List[str]:
    """Return feature columns whose variance ≤ *threshold*.

    A column with zero variance carries no predictive information.
    ``threshold=0.0`` catches truly constant columns only.
    """
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    variances = numeric.var(ddof=0)
    constant = variances[variances <= threshold].index.tolist()
    if constant:
        log.info("Constant feature columns (var ≤ %.4f): %s", threshold, constant)
    return constant


def find_duplicate_columns(X: pd.DataFrame) -> List[List[str]]:
    """Find groups of feature columns that are exact duplicates of each other.

    Returns a list of groups, where each group is a list of column names
    that are identical. Only groups with ≥ 2 members are returned.
    """
    seen: Dict[int, List[str]] = {}
    for col in X.columns:
        # Hash the column values for fast comparison
        col_hash = hash(X[col].values.tobytes())
        if col_hash in seen:
            # Verify it's a true duplicate, not just a hash collision
            ref_col = seen[col_hash][0]
            if X[col].equals(X[ref_col]):
                seen[col_hash].append(col)
            else:
                # Hash collision — start a new group
                seen[col_hash + 1] = [col]
        else:
            seen[col_hash] = [col]

    groups = [g for g in seen.values() if len(g) >= 2]
    if groups:
        log.info("Found %d group(s) of duplicate columns: %s", len(groups), groups)
    return groups


def diagnose_nonfinite(X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Report NaN, +inf, and -inf counts per numeric column.

    Returns {col_name: {"nan": n, "posinf": n, "neginf": n}} for columns
    that have at least one non-finite value.
    """
    numeric = X.select_dtypes(include=[np.number])
    result: Dict[str, Dict[str, int]] = {}

    for col in numeric.columns:
        nan_count = int(numeric[col].isna().sum())
        posinf = int((numeric[col] == np.inf).sum())
        neginf = int((numeric[col] == -np.inf).sum())
        if nan_count or posinf or neginf:
            result[col] = {"nan": nan_count, "posinf": posinf, "neginf": neginf}

    if result:
        total_issues = sum(
            v["nan"] + v["posinf"] + v["neginf"] for v in result.values()
        )
        log.info("Non-finite values found in %d column(s), %d total issues.",
                 len(result), total_issues)
    return result


# ─── sklearn preprocessing pipeline ─────────────────────────────────────────

def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
    scale: bool = True,
) -> Tuple[Pipeline, List[str]]:
    """Build and fit a scikit-learn preprocessing pipeline on training data.

    Steps
    -----
    1. Impute missing numeric values with median (training-set statistics only).
    2. Optionally scale features with StandardScaler (mean=0, std=1).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (must not contain target / label columns).
    scale : bool
        Whether to apply StandardScaler after imputation.

    Returns
    -------
    (fitted_pipeline, feature_names)
        The pipeline is fitted on *X_train*. ``feature_names`` records the
        exact column order used — inference must present columns in this order.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X_train.columns if c not in numeric_cols]

    if non_numeric_cols:
        log.warning(
            "Dropping %d non-numeric feature column(s) from pipeline: %s",
            len(non_numeric_cols), non_numeric_cols,
        )

    # Record feature order for reproducibility
    feature_names = list(numeric_cols)

    # Replace ±inf with NaN so the imputer can handle them
    X_fit = X_train[feature_names].replace([np.inf, -np.inf], np.nan)

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)
    pipeline.fit(X_fit)

    log.info("Preprocessing pipeline fitted on %d features: %s",
             len(feature_names), steps)
    return pipeline, feature_names


def apply_preprocessing_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """Transform a feature matrix through a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn pipeline.
    X : pd.DataFrame
        Feature matrix to transform. Must contain all columns in *feature_names*.
    feature_names : list[str]
        Exact column order the pipeline was fitted on.

    Returns
    -------
    pd.DataFrame with the same column names and transformed values.
    """
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns for pipeline: {missing}")

    # Replace ±inf with NaN for consistency with fit-time handling
    X_input = X[feature_names].replace([np.inf, -np.inf], np.nan)
    transformed = pipeline.transform(X_input)
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


# ─── Summary / persistence ──────────────────────────────────────────────────

def save_feature_metadata(
    feature_names: List[str],
    label_mapping: Dict[str, int],
    constant_cols: List[str],
    duplicate_groups: List[List[str]],
    nonfinite_report: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Save feature engineering metadata as JSON for reproducibility."""
    output_path = output_path or (PROCESSED_DIR / "feature_metadata.json")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "label_mapping": label_mapping,
        "constant_columns": constant_cols,
        "duplicate_column_groups": duplicate_groups,
        "nonfinite_summary": nonfinite_report,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Feature metadata saved → %s", output_path)
    return output_path
