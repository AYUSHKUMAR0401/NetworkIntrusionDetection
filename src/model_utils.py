"""
model_utils.py — Model persistence, schema validation, and utility functions.

Handles:
  • Saving / loading fitted sklearn pipelines via joblib.
  • Saving / loading JSON model metadata.
  • Validating that inference inputs match the training feature schema.
  • Recording package versions for reproducibility.

WARNING: Only load joblib artifacts from trusted sources.
Loading untrusted files can execute arbitrary code.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.config import (
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    NON_FEATURE_COLUMNS,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.model_utils")


# ── package versions ────────────────────────────────────────────────────────

def get_package_versions() -> Dict[str, str]:
    """Return versions of key packages for reproducibility metadata."""
    versions: Dict[str, str] = {}
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "joblib": "joblib",
    }
    for display_name, import_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            versions[display_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[display_name] = "not installed"
    return versions


# ── pipeline persistence ────────────────────────────────────────────────────

def save_pipeline(pipeline, path: Optional[Path] = None) -> Path:
    """Save a fitted sklearn pipeline to disk using joblib.

    Returns the path the pipeline was saved to.
    """
    path = path or MODEL_PIPELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    log.info("Saved pipeline → %s", path)
    return path


def load_pipeline(path: Optional[Path] = None):
    """Load a saved sklearn pipeline from disk.

    WARNING: Only load trusted joblib artifacts. Loading untrusted
    files can execute arbitrary code.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = path or MODEL_PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {path}")
    pipeline = joblib.load(path)
    log.info("Loaded pipeline from %s", path)
    return pipeline


# ── metadata persistence ────────────────────────────────────────────────────

def save_metadata(metadata: Dict, path: Optional[Path] = None) -> Path:
    """Save model metadata as JSON."""
    path = path or MODEL_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved model metadata → %s", path)
    return path


def load_metadata(path: Optional[Path] = None) -> Dict:
    """Load model metadata from JSON.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = path or MODEL_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── feature schema validation ───────────────────────────────────────────────

def validate_features(
    df: pd.DataFrame,
    expected_features: List[str],
) -> pd.DataFrame:
    """Validate and reorder DataFrame columns to match expected feature schema.

    Checks:
    1. No target / metadata columns are present (prevents leakage).
    2. All expected feature columns exist.
    3. Extra columns are ignored with a warning.

    Returns the DataFrame with columns in the exact training order.

    Raises
    ------
    ValueError
        If required features are missing or target columns are present.
    """
    # 1. Target / label leakage
    leaked = NON_FEATURE_COLUMNS & set(df.columns)
    if leaked:
        raise ValueError(
            f"Input contains target/metadata columns that must not be used "
            f"as features: {leaked}.  Remove these columns before prediction."
        )

    # 2. Missing features
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required feature column(s): {sorted(missing)}"
        )

    # 3. Extra columns (warning only)
    extra = set(df.columns) - set(expected_features)
    if extra:
        log.warning(
            "Input has %d extra column(s) not used by the model: %s. "
            "They will be ignored.",
            len(extra),
            sorted(extra),
        )

    # Reorder to match training feature order
    return df[expected_features].copy()
