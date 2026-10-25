"""
train.py — Training pipeline for AI-NIDS Phase 3.

Trains three baseline binary classifiers (Logistic Regression, Decision
Tree, Random Forest), evaluates each on a stratified held-out test set,
selects the best model based on malicious-class F1 / recall, and saves
the winning pipeline + metadata.

Usage:
    python -m src.train               # auto-detect real or synthetic data
    python -m src.train --synthetic    # force synthetic data
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BINARY_TARGET_COLUMN,
    MIN_RECALL_THRESHOLD,
    MODELS_DIR,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    PROCESSED_CSV,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    setup_logging,
)
from src.feature_engineering import build_label_mapping, separate_features_target

logger = setup_logging()
log = logging.getLogger("ai_nids.train")


# ── pipeline construction ───────────────────────────────────────────────────

def build_model_pipelines() -> Dict[str, Pipeline]:
    """Create three baseline model pipelines.

    Each pipeline contains its own imputer and scaler so that
    preprocessing is fitted **only on training data** — no leakage.
    """
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_SEED,
                class_weight="balanced",
                max_depth=20,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                n_jobs=2,
                max_depth=30,
            )),
        ]),
    }


# ── feature preparation ─────────────────────────────────────────────────────

def prepare_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Prepare feature matrix: replace ±inf with NaN, drop non-numeric columns.

    Returns ``(X_numeric, feature_names, dropped_columns)``.
    """
    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]

    if non_numeric:
        log.warning(
            "Dropping %d non-numeric feature column(s): %s",
            len(non_numeric), non_numeric,
        )

    return X[numeric_cols].copy(), numeric_cols, non_numeric


# ── train / test split ──────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / test split.

    Falls back to a non-stratified split if stratification is
    impossible (e.g. a class has fewer samples than required).
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y,
        )
        log.info(
            "Stratified split: train=%d, test=%d",
            len(X_train), len(X_test),
        )
    except ValueError as exc:
        log.warning(
            "Stratified split failed (%s), falling back to random split.", exc,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
        )
        log.info(
            "Random split: train=%d, test=%d",
            len(X_train), len(X_test),
        )

    return X_train, X_test, y_train, y_test


# ── contamination check ─────────────────────────────────────────────────────

def check_train_test_contamination(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict:
    """Detect exact duplicate rows between train and test sets.

    CIC-IDS2017 contains many near-duplicate network flows, so some
    contamination is expected.  The returned dict documents the extent.
    """
    train_hashes = set(pd.util.hash_pandas_object(X_train).values)
    test_hashes = pd.util.hash_pandas_object(X_test).values

    duplicated = sum(1 for h in test_hashes if h in train_hashes)

    result = {
        "test_rows": len(X_test),
        "duplicated_in_train": duplicated,
        "contamination_pct": (
            round(duplicated / len(X_test) * 100, 2)
            if len(X_test) > 0 else 0.0
        ),
    }

    if duplicated:
        log.warning(
            "Train/test contamination: %d test rows (%.1f%%) are exact "
            "duplicates of training rows.  This is common in CIC-IDS2017 "
            "due to repeated network flows.  Results may be optimistically "
            "biased.",
            duplicated, result["contamination_pct"],
        )
    else:
        log.info("No exact train/test duplicate rows detected.")

    return result


# ── single-model training ───────────────────────────────────────────────────

def train_single_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Tuple[Pipeline, float]:
    """Fit a single model pipeline.  Returns ``(fitted_pipeline, seconds)``."""
    log.info("Training %s on %d samples …", model_name, len(X_train))
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    log.info("  %s trained in %.2f s", model_name, elapsed)
    return pipeline, elapsed


# ── full training orchestration ──────────────────────────────────────────────

def run_training(
    df: Optional[pd.DataFrame] = None,
    force_synthetic: bool = False,
    test_size: float = TEST_SIZE,
) -> Tuple[Dict, str, Dict]:
    """Execute the full Phase 3 training pipeline.

    Parameters
    ----------
    df : pd.DataFrame | None
        Pre-loaded data.  If ``None``, attempts to load real data from
        the processed CSV or raw directory, falling back to synthetic.
    force_synthetic : bool
        If ``True``, uses synthetic data regardless.
    test_size : float
        Fraction of data held out for evaluation.

    Returns
    -------
    (results, best_model_name, metadata)
        * ``results`` maps model name → {pipeline, train_time, metrics}.
        * ``best_model_name`` is the selected model.
        * ``metadata`` is the full metadata dict (saved to disk).
    """
    # Late imports to avoid circular dependencies
    from src.evaluate import (
        compare_models,
        evaluate_model,
        plot_confusion_matrix,
        plot_model_comparison,
        save_classification_reports,
        save_training_summary,
    )
    from src.model_utils import get_package_versions, save_metadata, save_pipeline

    # ── 1. Load data ────────────────────────────────────────────────────────
    data_source = _resolve_data(df, force_synthetic)
    df, data_source = data_source

    log.info("=" * 60)
    log.info("Phase 3 Training — source: %s", data_source)
    log.info("Dataset: %d rows × %d columns", len(df), len(df.columns))
    log.info("=" * 60)

    # ── 2. Label mapping ────────────────────────────────────────────────────
    label_mapping = build_label_mapping(df)
    log.info("Label mapping: %s", label_mapping)

    if ORIGINAL_LABEL_COLUMN in df.columns:
        orig_dist = df[ORIGINAL_LABEL_COLUMN].value_counts()
        log.info("Original label distribution:\n%s", orig_dist.to_string())

    binary_dist = df[BINARY_TARGET_COLUMN].value_counts()
    log.info("Binary class distribution: %s", binary_dist.to_dict())

    # ── 3. Both classes must exist ──────────────────────────────────────────
    unique_classes = set(df[BINARY_TARGET_COLUMN].unique())
    if unique_classes != {0, 1}:
        raise ValueError(
            f"Both binary classes (0=Benign, 1=Malicious) must be present. "
            f"Found only: {unique_classes}"
        )

    # ── 4. Feature / target separation ──────────────────────────────────────
    X, y = separate_features_target(df)
    X, feature_names, dropped_cols = prepare_features(X)
    log.info("Features: %d numeric columns", len(feature_names))

    # ── 5. Stratified split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    # ── 6. Contamination check ──────────────────────────────────────────────
    contamination = check_train_test_contamination(X_train, X_test)

    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()
    log.info("Train class distribution: %s", train_dist)
    log.info("Test  class distribution: %s", test_dist)

    # ── 7. Train all models ─────────────────────────────────────────────────
    pipelines = build_model_pipelines()
    results: Dict[str, Dict] = {}

    for name, pipeline in pipelines.items():
        fitted, train_time = train_single_model(
            pipeline, X_train, y_train, name,
        )
        metrics = evaluate_model(fitted, X_test, y_test, model_name=name)
        results[name] = {
            "pipeline": fitted,
            "train_time": train_time,
            "metrics": metrics,
        }

    # ── 8. Compare and select best ──────────────────────────────────────────
    best_name, comparison_df = compare_models(results)

    # ── 9. Generate reports ─────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        plot_confusion_matrix(
            result["metrics"]["confusion_matrix"],
            model_name=name,
            output_dir=REPORTS_DIR / "confusion_matrices",
        )

    plot_model_comparison(comparison_df, output_dir=REPORTS_DIR)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    save_classification_reports(results)
    log.info("Saved model_comparison.csv → %s", REPORTS_DIR / "model_comparison.csv")

    # ── 10. Save best pipeline ──────────────────────────────────────────────
    save_pipeline(results[best_name]["pipeline"])

    # ── 11. Build and save metadata ─────────────────────────────────────────
    best_metrics = results[best_name]["metrics"]
    best_cm = best_metrics["confusion_matrix"]

    metadata: Dict = {
        "model_type": best_name,
        "training_date": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "data_source": data_source,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "dropped_non_numeric_columns": dropped_cols,
        "label_mapping": label_mapping,
        "binary_target": {
            "column": BINARY_TARGET_COLUMN,
            "0": "Benign",
            "1": "Malicious",
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_class_distribution": {str(k): v for k, v in train_dist.items()},
        "test_class_distribution": {str(k): v for k, v in test_dist.items()},
        "contamination": contamination,
        "evaluation_metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in best_metrics.items()
        },
        "all_model_metrics": {
            name: {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in r["metrics"].items()
            }
            for name, r in results.items()
        },
        "selection_criteria": (
            f"Selected '{best_name}' based on highest malicious-class "
            f"F1-score among models meeting the minimum recall threshold "
            f"of {MIN_RECALL_THRESHOLD:.0%}."
        ),
        "min_recall_threshold": MIN_RECALL_THRESHOLD,
        "package_versions": get_package_versions(),
        "known_limitations": [
            "Random train/test split does not account for temporal "
            "correlation in network flows.",
            "CIC-IDS2017 contains many near-duplicate flows which "
            "may inflate test-set performance.",
            "Model was trained on CIC-IDS2017 only — generalisation "
            "to other networks is not validated.",
            (
                f"Data source: {data_source}."
                + (
                    " Synthetic data does NOT represent real attack "
                    "patterns."
                    if data_source == "synthetic"
                    else ""
                )
            ),
        ],
    }

    save_metadata(metadata)
    save_training_summary(results, best_name, metadata)

    # ── 12. Console summary ─────────────────────────────────────────────────
    _print_summary(results, best_name, data_source)

    return results, best_name, metadata


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_data(
    df: Optional[pd.DataFrame],
    force_synthetic: bool,
) -> Tuple[pd.DataFrame, str]:
    """Load real or synthetic data."""
    if df is not None:
        return df, "provided"

    if not force_synthetic and PROCESSED_CSV.exists():
        log.info("Loading processed data from %s", PROCESSED_CSV)
        return pd.read_csv(PROCESSED_CSV), "real_cic_ids2017_processed"

    if not force_synthetic:
        from src.data_loader import discover_csv_files
        if discover_csv_files():
            log.info("Raw CSVs found — running preprocessing first …")
            from src.data_loader import load_all_csvs
            from src.preprocess import run_preprocessing
            raw_df, _ = load_all_csvs()
            processed, _ = run_preprocessing(raw_df)
            return processed, "real_cic_ids2017_freshly_processed"

    log.warning("No real data found.  Generating synthetic data.")
    from src.run_eda import _generate_synthetic_data
    return _generate_synthetic_data(n=5000), "synthetic"


def _print_summary(results: Dict, best_name: str, data_source: str) -> None:
    """Print a human-readable training summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 Training Complete")
    print(f"  Data source: {data_source}")
    print(f"  Selected model: {best_name}")
    print(f"{'=' * 60}")

    for name, r in results.items():
        m = r["metrics"]
        marker = " <- SELECTED" if name == best_name else ""
        print(f"\n  {name}{marker}")
        print(f"    Accuracy:  {m['accuracy']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1-score:  {m['f1']:.4f}")
        if m.get("roc_auc") is not None:
            print(f"    ROC-AUC:   {m['roc_auc']:.4f}")
        print(f"    Train time:   {r['train_time']:.2f} s")
        print(f"    Predict time: {m['predict_time']:.4f} s")
        if not m["meets_recall_threshold"]:
            print(f"    [!] Recall below {MIN_RECALL_THRESHOLD:.0%} threshold")

    if data_source == "synthetic":
        print(f"\n  [!] WARNING: Results are from SYNTHETIC data.")
        print(f"    Real CIC-IDS2017 required for meaningful evaluation.")

    print(f"\n  Model saved:   {MODELS_DIR}")
    print(f"  Reports saved: {REPORTS_DIR}")
    print(f"{'=' * 60}\n")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-NIDS Phase 3: Train binary intrusion detection models",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data even if real data is available",
    )
    args = parser.parse_args()
    run_training(force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
