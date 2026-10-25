"""
test_phase3.py — Comprehensive unit tests for AI-NIDS Phase 3.

Tests:
  1. model_utils: saving/loading pipelines, metadata creation & loading, schema validation.
  2. evaluate: metric calculation, confusion matrix plotting, classification report generation, model selection logic.
  3. train: end-to-end model training workflow on synthetic data.
  4. predict: schema validation, predictions, probability scores.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.model_utils import (
    save_pipeline,
    load_pipeline,
    save_metadata,
    load_metadata,
    validate_features,
    get_package_versions,
)
from src.evaluate import (
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    plot_model_comparison,
    save_classification_reports,
    save_training_summary,
)
from src.predict import (
    predict,
    predict_single,
)
from src.train import run_training
from src.run_eda import _generate_synthetic_data
from src.feature_engineering import separate_features_target


@pytest.fixture
def sample_data():
    df = _generate_synthetic_data(n=200)
    X, y = separate_features_target(df)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, list(X.columns)


@pytest.fixture
def fitted_pipeline(sample_data):
    X, y, _ = sample_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Test model_utils
# ---------------------------------------------------------------------------

def test_save_and_load_pipeline(tmp_path, fitted_pipeline, sample_data):
    X, _, _ = sample_data
    path = tmp_path / "test_pipeline.joblib"
    save_pipeline(fitted_pipeline, path)
    assert path.exists()

    loaded_pipe = load_pipeline(path)
    orig_preds = fitted_pipeline.predict(X)
    loaded_preds = loaded_pipe.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_save_and_load_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    features = ["feat_a", "feat_b", "feat_c"]
    metrics = {"accuracy": 0.95, "f1": 0.94}
    meta_dict = {
        "model_name": "LogisticRegression",
        "feature_names": features,
        "num_features": 3,
        "label_mapping": {"benign": 0, "malicious": 1},
        "metrics": metrics,
        "data_source": "synthetic",
    }
    
    save_metadata(meta_dict, path)
    assert path.exists()

    loaded_meta = load_metadata(path)
    assert loaded_meta["model_name"] == "LogisticRegression"
    assert loaded_meta["feature_names"] == features
    assert loaded_meta["num_features"] == 3
    assert loaded_meta["metrics"]["accuracy"] == 0.95


def test_validate_features(sample_data):
    X, _, feature_cols = sample_data
    
    # Matching schema
    val_df = validate_features(X, feature_cols)
    assert list(val_df.columns) == feature_cols

    # Leaked target columns
    X_leaked = X.copy()
    X_leaked["is_malicious"] = 1
    with pytest.raises(ValueError, match="target/metadata columns"):
        validate_features(X_leaked, feature_cols)

    # Missing column
    X_missing = X.iloc[:, :-2]
    with pytest.raises(ValueError, match="Missing 2 required feature column"):
        validate_features(X_missing, feature_cols)


def test_get_package_versions():
    versions = get_package_versions()
    assert "scikit-learn" in versions
    assert "pandas" in versions
    assert "joblib" in versions


# ---------------------------------------------------------------------------
# Test evaluate
# ---------------------------------------------------------------------------

def test_evaluate_model(fitted_pipeline, sample_data):
    X, y, _ = sample_data
    res = evaluate_model(fitted_pipeline, X, pd.Series(y), model_name="TestLR")
    assert "accuracy" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "confusion_matrix" in res
    assert res["confusion_matrix"].shape == (2, 2)


def test_compare_models():
    results = {
        "ModelA": {"metrics": {"accuracy": 0.90, "precision": 0.8, "recall": 0.85, "f1": 0.87, "meets_recall_threshold": True, "predict_time": 0.01}, "train_time": 0.1},
        "ModelB": {"metrics": {"accuracy": 0.95, "precision": 0.9, "recall": 0.70, "f1": 0.80, "meets_recall_threshold": False, "predict_time": 0.01}, "train_time": 0.1},
    }
    # ModelA passes 80% recall, even though ModelB has higher accuracy
    best_name, comp_df = compare_models(results)
    assert best_name == "ModelA"
    assert len(comp_df) == 2


def test_plot_and_reports(tmp_path, fitted_pipeline, sample_data):
    X, y, _ = sample_data
    res = evaluate_model(fitted_pipeline, X, pd.Series(y), model_name="TestLR")
    
    # Plot CM
    cm_path = plot_confusion_matrix(res["confusion_matrix"], "TestLR", output_dir=tmp_path)
    assert cm_path.exists()

    # Plot comparison
    results = {"TestLR": {"metrics": res, "train_time": 0.1}}
    _, comp_df = compare_models(results)
    comp_path = plot_model_comparison(comp_df, output_dir=tmp_path)
    assert comp_path.exists()

    # Classification report
    rep_dir = tmp_path / "reports"
    save_classification_reports(results, output_dir=rep_dir)
    assert (rep_dir / "report_testlr.txt").exists()

    # Training summary JSON
    sum_path = save_training_summary(results, best_name="TestLR", metadata={"data_source": "synthetic"}, output_dir=tmp_path)
    assert sum_path.exists()


# ---------------------------------------------------------------------------
# Test predict
# ---------------------------------------------------------------------------

def test_predict_dataframe(tmp_path, fitted_pipeline, sample_data):
    X, _, feature_cols = sample_data
    meta_dict = {
        "model_name": "TestLR",
        "feature_names": feature_cols,
        "num_features": len(feature_cols),
        "label_mapping": {"benign": 0, "malicious": 1},
        "metrics": {"f1": 0.9},
        "data_source": "synthetic",
    }

    res_df = predict(X, pipeline=fitted_pipeline, metadata=meta_dict)

    assert "label" in res_df.columns
    assert "prediction" in res_df.columns
    assert "probability_malicious" in res_df.columns
    assert set(res_df["label"].unique()).issubset({"Benign", "Malicious"})

    # Single predict
    single_res = predict_single(X.iloc[0].to_dict(), pipeline=fitted_pipeline, metadata=meta_dict)
    assert "label" in single_res
    assert single_res["label"] in ("Benign", "Malicious")


# ---------------------------------------------------------------------------
# Test run_training CLI end-to-end
# ---------------------------------------------------------------------------

def test_run_training_end_to_end(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    
    monkeypatch.setattr("src.config.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.config.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.train.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.train.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.evaluate.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.model_utils.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.model_utils.MODEL_PIPELINE_PATH", models_dir / "binary_intrusion_pipeline.joblib")
    monkeypatch.setattr("src.model_utils.MODEL_METADATA_PATH", models_dir / "model_metadata.json")

    results, best_name, metadata = run_training(force_synthetic=True)
    assert best_name in ("LogisticRegression", "DecisionTree", "RandomForest")
    assert len(results) == 3
    assert (models_dir / "binary_intrusion_pipeline.joblib").exists()
    assert (models_dir / "model_metadata.json").exists()
