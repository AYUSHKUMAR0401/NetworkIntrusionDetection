"""
evaluate.py — Model evaluation and comparison for AI-NIDS Phase 3.

Computes accuracy, precision, recall, F1, confusion matrix, ROC-AUC,
and timing for each trained model.  Generates plots and comparison
reports.

Terminology for binary intrusion detection (positive class = 1 = Malicious):
  • True  Positive (TP): malicious flow correctly detected as malicious.
  • False Positive (FP): benign flow incorrectly flagged as malicious.
  • False Negative (FN): malicious flow missed (classified as benign).
  • True  Negative (TN): benign flow correctly classified as benign.

For a security system, malicious-class recall (FN minimisation) and
precision (FP minimisation) deserve particular attention.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MIN_RECALL_THRESHOLD, REPORTS_DIR, setup_logging

logger = setup_logging()
log = logging.getLogger("ai_nids.evaluate")


# ── single-model evaluation ─────────────────────────────────────────────────

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict:
    """Evaluate a fitted model pipeline on the held-out test set.

    Returns a dict containing all metrics, the confusion matrix,
    and a formatted classification report.
    """
    # Predict and time it
    start = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - start

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Benign", "Malicious"],
            zero_division=0,
        ),
        "predict_time": float(predict_time),
    }

    # ROC-AUC — requires probability or decision-function scores
    roc_auc = _safe_roc_auc(pipeline, X_test, y_test, model_name)
    metrics["roc_auc"] = roc_auc

    # Confusion-matrix decomposition
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    })

    # Check recall against configurable threshold
    metrics["meets_recall_threshold"] = metrics["recall"] >= MIN_RECALL_THRESHOLD

    log.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f%s%s",
        model_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        f" | AUC: {roc_auc:.4f}" if roc_auc is not None else "",
        "" if metrics["meets_recall_threshold"]
        else f" ⚠ recall < {MIN_RECALL_THRESHOLD:.0%}",
    )
    return metrics


def _safe_roc_auc(pipeline, X_test, y_test, model_name: str):
    """Compute ROC-AUC, returning None if not possible."""
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, y_proba))
        if hasattr(pipeline, "decision_function"):
            y_scores = pipeline.decision_function(X_test)
            return float(roc_auc_score(y_test, y_scores))
        log.warning(
            "%s does not support probability scores — ROC-AUC skipped.",
            model_name,
        )
    except ValueError as exc:
        log.warning("ROC-AUC failed for %s: %s", model_name, exc)
    return None


# ── model comparison and selection ──────────────────────────────────────────

def compare_models(results: Dict[str, Dict]) -> Tuple[str, pd.DataFrame]:
    """Compare all trained models and select the best one.

    Selection criteria
    ------------------
    1. Among models meeting ``MIN_RECALL_THRESHOLD``, select the one
       with the highest malicious-class F1-score.
    2. If no model meets the threshold, select the one with the
       highest recall (to prioritise detection).

    Returns ``(best_model_name, comparison_dataframe)``.
    """
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "train_time": result["train_time"],
            "predict_time": m["predict_time"],
            "meets_recall_threshold": m["meets_recall_threshold"],
        })

    comparison = pd.DataFrame(rows)

    # Selection logic
    eligible = comparison[comparison["meets_recall_threshold"]]

    if not eligible.empty:
        best_idx = eligible["f1"].idxmax()
        best_name = str(eligible.loc[best_idx, "model"])
        log.info(
            "Selected '%s' — highest F1 (%.4f) among models meeting "
            "recall threshold (≥ %.0f%%).",
            best_name, eligible.loc[best_idx, "f1"],
            MIN_RECALL_THRESHOLD * 100,
        )
    else:
        log.warning(
            "No model meets the minimum recall threshold of %.0f%%. "
            "Selecting model with highest recall.",
            MIN_RECALL_THRESHOLD * 100,
        )
        best_idx = comparison["recall"].idxmax()
        best_name = str(comparison.loc[best_idx, "model"])

    # Annotate highest-accuracy model if different
    highest_acc_idx = comparison["accuracy"].idxmax()
    highest_acc_name = str(comparison.loc[highest_acc_idx, "model"])
    if highest_acc_name != best_name:
        log.info(
            "Note: '%s' has the highest accuracy (%.4f) but '%s' was "
            "selected for better malicious-class detection.",
            highest_acc_name,
            comparison.loc[highest_acc_idx, "accuracy"],
            best_name,
        )

    return best_name, comparison


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save a confusion-matrix heatmap as PNG."""
    output_dir = output_dir or (REPORTS_DIR / "confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Benign (0)", "Malicious (1)"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title=f"Confusion Matrix — {model_name}",
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = output_dir / f"cm_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)
    return path


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a grouped bar chart comparing model metrics."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    if "roc_auc" in comparison_df.columns and comparison_df["roc_auc"].notna().all():
        metrics_to_plot.append("roc_auc")

    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].astype(float).values
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=metric.replace("_", " ").title(),
            color=colours[i % len(colours)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.axhline(
        y=MIN_RECALL_THRESHOLD, color="red", linestyle="--", alpha=0.5,
        label=f"Min recall ({MIN_RECALL_THRESHOLD:.0%})",
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(
        "Model Comparison — Binary Intrusion Detection",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved model comparison chart → %s", path)
    return path


# ── report persistence ──────────────────────────────────────────────────────

def save_training_summary(
    results: Dict[str, Dict],
    best_name: str,
    metadata: Dict,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save a comprehensive training summary as JSON."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "selected_model": best_name,
        "training_date": metadata.get("training_date"),
        "data_source": metadata.get("data_source"),
        "train_samples": metadata.get("train_samples"),
        "test_samples": metadata.get("test_samples"),
        "n_features": metadata.get("n_features"),
        "min_recall_threshold": MIN_RECALL_THRESHOLD,
        "models": {},
    }

    for name, result in results.items():
        m = result["metrics"]
        cm = m["confusion_matrix"]
        summary["models"][name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "train_time": result["train_time"],
            "predict_time": m["predict_time"],
            "meets_recall_threshold": m["meets_recall_threshold"],
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        }

    path = output_dir / "training_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Training summary saved → %s", path)
    return path


def save_classification_reports(
    results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Save per-model classification reports as text files."""
    output_dir = output_dir or (REPORTS_DIR / "classification_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for name, result in results.items():
        report_text = result["metrics"]["classification_report"]
        safe_name = name.lower().replace(" ", "_")
        path = output_dir / f"report_{safe_name}.txt"
        path.write_text(report_text, encoding="utf-8")
        paths.append(path)

    log.info("Saved %d classification report(s) to %s", len(paths), output_dir)
    return paths
