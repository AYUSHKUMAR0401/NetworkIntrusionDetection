"""
predict.py — Prediction functionality for AI-NIDS Phase 3.

Loads a saved pipeline, validates the input feature schema, and returns
predictions mapped to Benign / Malicious.

These predictions operate on dataset records or explicitly supplied
feature rows.  This module does NOT claim to perform live network
intrusion detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import setup_logging
from src.model_utils import load_metadata, load_pipeline, validate_features

logger = setup_logging()
log = logging.getLogger("ai_nids.predict")


def predict(
    X: pd.DataFrame,
    pipeline=None,
    metadata: Optional[Dict] = None,
    pipeline_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run predictions on input data using a saved (or provided) model.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.  Must contain the columns the model was trained
        on (listed in metadata ``feature_names``).  Must **not** contain
        target or metadata columns.
    pipeline : sklearn Pipeline | None
        Pre-loaded pipeline.  Loaded from disk if ``None``.
    metadata : dict | None
        Pre-loaded metadata.  Loaded from disk if ``None``.
    pipeline_path / metadata_path : Path | None
        Override default persistence paths.

    Returns
    -------
    pd.DataFrame with columns:
        * ``prediction`` — numeric class (0 or 1).
        * ``label`` — human-readable ("Benign" or "Malicious").
        * ``probability_malicious`` — P(class 1), if the model supports it.
        * ``probability_benign`` — P(class 0), if the model supports it.
    """
    if pipeline is None:
        pipeline = load_pipeline(pipeline_path)
    if metadata is None:
        metadata = load_metadata(metadata_path)

    feature_names = metadata["feature_names"]

    # Replace ±inf with NaN (consistent with training)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Validate and reorder
    X_validated = validate_features(X, feature_names)

    # Predict
    predictions = pipeline.predict(X_validated)

    result = pd.DataFrame(
        {
            "prediction": predictions,
            "label": np.where(predictions == 0, "Benign", "Malicious"),
        },
        index=X.index,
    )

    # Probabilities (if supported)
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_validated)
            result["probability_malicious"] = proba[:, 1]
            result["probability_benign"] = proba[:, 0]
        except Exception as exc:
            log.warning("Could not compute prediction probabilities: %s", exc)

    log.info(
        "Predicted %d samples: %d Benign, %d Malicious",
        len(result),
        int((predictions == 0).sum()),
        int((predictions == 1).sum()),
    )
    return result


def predict_single(
    features: Dict[str, float],
    pipeline=None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Predict a single sample from a dict of feature values.

    Parameters
    ----------
    features : dict
        ``{feature_name: value}`` for one network flow.

    Returns
    -------
    Dict with ``prediction``, ``label``, and (optionally) probability keys.
    """
    X = pd.DataFrame([features])
    result = predict(X, pipeline=pipeline, metadata=metadata)
    return result.iloc[0].to_dict()
