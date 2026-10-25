"""
eda.py — Exploratory Data Analysis for AI-NIDS.

Generates:
  • Dataset dimensions and data-type summary
  • Class distribution plots (binary + multiclass)
  • Missing-value summary
  • Feature distribution histograms (sampled)
  • Correlation heatmap for top numeric features
  • Class-imbalance analysis with implications
  • A concise written EDA report saved under reports/

Can run on either the real processed dataset or a synthetic sample.
All outputs are clearly labelled to indicate which data source was used.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    BINARY_TARGET_COLUMN,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    PROCESSED_CSV,
    PROCESSED_DIR,
    REPORTS_DIR,
    SOURCE_FILE_COLUMN,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.eda")

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ─── Dataset inspection ─────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> Dict:
    """Compute basic statistics about the dataset."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.value_counts().to_dict()
    dtypes_str = {str(k): int(v) for k, v in dtypes.items()}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    info = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": dtypes_str,
        "numeric_columns": len(numeric_cols),
        "non_numeric_columns": len(non_numeric_cols),
        "non_numeric_names": non_numeric_cols,
        "feature_columns": len(feature_cols),
        "columns": list(df.columns),
    }
    log.info("Dataset: %d rows × %d cols (%d numeric, %d non-numeric)",
             n_rows, n_cols, len(numeric_cols), len(non_numeric_cols))
    return info


# ─── Missing / infinite value summary ───────────────────────────────────────

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of columns with missing or infinite values."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

    records = []
    for col in feature_cols:
        nan_count = int(df[col].isna().sum())
        if col in numeric_cols:
            inf_count = int(df[col].isin([np.inf, -np.inf]).sum())
        else:
            inf_count = 0
        if nan_count or inf_count:
            records.append({
                "column": col,
                "missing": nan_count,
                "infinite": inf_count,
                "missing_pct": round(nan_count / len(df) * 100, 2),
            })

    summary = pd.DataFrame(records)
    if not summary.empty:
        summary = summary.sort_values("missing", ascending=False).reset_index(drop=True)
    log.info("Columns with missing/inf values: %d", len(summary))
    return summary


# ─── Class distribution ─────────────────────────────────────────────────────

def class_distribution(df: pd.DataFrame) -> Dict:
    """Compute binary and multiclass label distributions."""
    result: Dict = {}

    # Binary
    if BINARY_TARGET_COLUMN in df.columns:
        binary_counts = df[BINARY_TARGET_COLUMN].value_counts().to_dict()
        total = sum(binary_counts.values())
        result["binary"] = {
            "counts": binary_counts,
            "percentages": {k: round(v / total * 100, 2) for k, v in binary_counts.items()},
        }

    # Multiclass
    if ORIGINAL_LABEL_COLUMN in df.columns:
        multi_counts = df[ORIGINAL_LABEL_COLUMN].value_counts().to_dict()
        total = sum(multi_counts.values())
        result["multiclass"] = {
            "counts": multi_counts,
            "percentages": {k: round(v / total * 100, 4) for k, v in multi_counts.items()},
            "n_classes": len(multi_counts),
        }

    return result


# ─── Potential leakage / identifier columns ─────────────────────────────────

def identify_suspicious_columns(df: pd.DataFrame) -> Dict:
    """Flag columns that could be identifiers, timestamps, or cause leakage."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    suspicious: Dict[str, List[str]] = {
        "potential_timestamps": [],
        "potential_identifiers": [],
        "high_cardinality": [],
        "known_non_features": list(NON_FEATURE_COLUMNS & set(df.columns)),
    }

    timestamp_hints = {"timestamp", "time", "datetime", "date", "epoch"}
    id_hints = {"id", "flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip"}

    for col in feature_cols:
        col_lower = col.lower()
        if any(hint in col_lower for hint in timestamp_hints):
            suspicious["potential_timestamps"].append(col)
        if any(hint in col_lower for hint in id_hints):
            suspicious["potential_identifiers"].append(col)
        # High cardinality objects — possible IP addresses or strings
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if n_unique > 0.5 * len(df) and len(df) > 10:
                suspicious["high_cardinality"].append(
                    f"{col} ({n_unique} unique / {len(df)} rows)"
                )

    return suspicious


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate binary and multiclass class-distribution bar charts."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Binary
    if BINARY_TARGET_COLUMN in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[BINARY_TARGET_COLUMN].value_counts().sort_index()
        labels = ["Benign (0)", "Malicious (1)"]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(labels[:len(counts)], counts.values, color=colors[:len(counts)],
                       edgecolor="white", linewidth=1.2)
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * counts.max(),
                    f"{count:,}", ha="center", va="bottom", fontweight="bold")
        ax.set_title("Binary Class Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Sample Count")
        fig.tight_layout()
        path = output_dir / "binary_class_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Multiclass
    if ORIGINAL_LABEL_COLUMN in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        counts = df[ORIGINAL_LABEL_COLUMN].value_counts()
        counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(counts)),
                    edgecolor="white", linewidth=0.5)
        ax.set_title("Attack Label Distribution (Multiclass)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sample Count")
        ax.set_ylabel("")
        for i, (val, name) in enumerate(zip(counts.values, counts.index)):
            ax.text(val + 0.005 * counts.max(), i, f" {val:,}", va="center", fontsize=9)
        fig.tight_layout()
        path = output_dir / "multiclass_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    log.info("Saved %d class distribution plot(s).", len(saved))
    return saved


def plot_missing_values(
    missing_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    top_n: int = 20,
) -> Optional[Path]:
    """Bar chart of columns with the most missing values."""
    if missing_df.empty:
        log.info("No missing values to plot.")
        return None

    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    top = missing_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    ax.barh(top["column"], top["missing"], color="#e67e22", edgecolor="white")
    ax.set_xlabel("Missing Count")
    ax.set_title(f"Top {len(top)} Columns by Missing Values", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()

    path = output_dir / "missing_values.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved missing-values plot → %s", path)
    return path


def plot_feature_distributions(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    max_features: int = 16,
    sample_n: int = 10_000,
) -> Optional[Path]:
    """Histogram grid of a sample of numeric feature columns."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        log.info("No numeric features to plot.")
        return None

    # Subsample columns and rows for readability
    cols = numeric_cols[:max_features]
    sample = df[cols].sample(n=min(sample_n, len(df)), random_state=42)

    n_plots = len(cols)
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        data = sample[col].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            axes[i].set_title(col, fontsize=9)
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            continue
        axes[i].hist(data, bins=50, color="#3498db", edgecolor="white", linewidth=0.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (sample)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = output_dir / "feature_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved feature distributions → %s", path)
    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    max_features: int = 20,
    sample_n: int = 50_000,
) -> Optional[Path]:
    """Correlation heatmap for the top *max_features* numeric features."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        log.info("Not enough numeric features for a correlation heatmap.")
        return None

    cols = numeric_cols[:max_features]
    sample = df[cols].sample(n=min(sample_n, len(df)), random_state=42)
    # Replace inf before computing correlation
    sample = sample.replace([np.inf, -np.inf], np.nan)
    corr = sample.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), max(6, len(cols) * 0.5)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=len(cols) <= 12,
                fmt=".2f" if len(cols) <= 12 else "",
                linewidths=0.5, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved correlation heatmap → %s", path)
    return path


# ─── Written EDA report ─────────────────────────────────────────────────────

def generate_eda_report(
    df: pd.DataFrame,
    data_source: str = "unknown",
    output_dir: Optional[Path] = None,
) -> Path:
    """Run the full EDA and save a human-readable text report + plots.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse (processed or raw).
    data_source : str
        Label to indicate origin: ``"real_cic_ids2017"``, ``"synthetic"``, etc.
    output_dir : Path | None
        Where to save reports and plots (default: ``reports/``).

    Returns
    -------
    Path to the written EDA report.
    """
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== EDA started (source: %s) ===", data_source)

    # 1. Inspect
    info = inspect_dataset(df)

    # 2. Missing values
    missing_df = missing_value_summary(df)

    # 3. Class distribution
    class_dist = class_distribution(df)

    # 4. Suspicious columns
    suspicious = identify_suspicious_columns(df)

    # 5. Duplicate rows
    n_duplicates = int(df.duplicated().sum())

    # 6. Plots
    class_plots = plot_class_distribution(df, output_dir)
    missing_plot = plot_missing_values(missing_df, output_dir)
    dist_plot = plot_feature_distributions(df, output_dir)
    corr_plot = plot_correlation_heatmap(df, output_dir)

    # 7. Class imbalance analysis
    imbalance_analysis = _analyse_class_imbalance(class_dist)

    # 8. Build report text
    lines = [
        "=" * 70,
        " AI-NIDS EXPLORATORY DATA ANALYSIS REPORT",
        f" Generated: {datetime.now().isoformat()}",
        f" Data source: {data_source}",
        "=" * 70,
        "",
        "── DATASET DIMENSIONS ──",
        f"  Rows:             {info['n_rows']:,}",
        f"  Columns:          {info['n_cols']}",
        f"  Feature columns:  {info['feature_columns']}",
        f"  Numeric columns:  {info['numeric_columns']}",
        f"  Non-numeric cols: {info['non_numeric_columns']}",
        f"  Data types:       {info['dtypes']}",
        "",
    ]

    if info["non_numeric_columns"] > 0:
        lines.append(f"  Non-numeric column names: {info['non_numeric_names']}")
        lines.append("")

    lines += [
        "── DUPLICATE ROWS ──",
        f"  Duplicate rows: {n_duplicates:,}",
        "",
    ]

    # Missing values
    lines.append("── MISSING / INFINITE VALUES ──")
    if missing_df.empty:
        lines.append("  No missing or infinite values found.")
    else:
        lines.append(f"  Columns with issues: {len(missing_df)}")
        for _, row in missing_df.iterrows():
            lines.append(
                f"    {row['column']}: {row['missing']} missing ({row['missing_pct']}%), "
                f"{row['infinite']} infinite"
            )
    lines.append("")

    # Class distribution
    lines.append("── CLASS DISTRIBUTION ──")
    if "binary" in class_dist:
        lines.append("  Binary:")
        for k, v in class_dist["binary"]["counts"].items():
            pct = class_dist["binary"]["percentages"][k]
            label = "Benign" if k == 0 else "Malicious"
            lines.append(f"    {label} (class {k}): {v:,} ({pct}%)")

    if "multiclass" in class_dist:
        lines.append(f"  Multiclass ({class_dist['multiclass']['n_classes']} labels):")
        for label, count in class_dist["multiclass"]["counts"].items():
            pct = class_dist["multiclass"]["percentages"][label]
            lines.append(f"    {label}: {count:,} ({pct}%)")
    lines.append("")

    # Imbalance analysis
    lines.append("── CLASS IMBALANCE ANALYSIS ──")
    for line in imbalance_analysis:
        lines.append(f"  {line}")
    lines.append("")

    # Suspicious columns
    lines.append("── POTENTIAL LEAKAGE / IDENTIFIER COLUMNS ──")
    for category, cols in suspicious.items():
        if cols:
            lines.append(f"  {category}: {cols}")
    if not any(suspicious.values()):
        lines.append("  None detected.")
    lines.append("")

    # Plots generated
    lines.append("── GENERATED PLOTS ──")
    for p in class_plots:
        lines.append(f"  • {p.name}")
    if missing_plot:
        lines.append(f"  • {missing_plot.name}")
    if dist_plot:
        lines.append(f"  • {dist_plot.name}")
    if corr_plot:
        lines.append(f"  • {corr_plot.name}")
    lines.append("")

    lines.append("=" * 70)

    report_text = "\n".join(lines)
    report_path = output_dir / "eda_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("EDA report saved → %s", report_path)

    # Also save as JSON for programmatic use
    json_report = {
        "data_source": data_source,
        "generated_at": datetime.now().isoformat(),
        "dataset_info": info,
        "missing_values": missing_df.to_dict("records") if not missing_df.empty else [],
        "class_distribution": class_dist,
        "duplicate_rows": n_duplicates,
        "suspicious_columns": suspicious,
        "imbalance_analysis": imbalance_analysis,
    }
    json_path = output_dir / "eda_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)

    log.info("=== EDA finished ===")
    return report_path


def _analyse_class_imbalance(class_dist: Dict) -> List[str]:
    """Produce textual analysis of class imbalance and its implications."""
    analysis: List[str] = []

    if "binary" not in class_dist:
        analysis.append("No binary target found — cannot assess imbalance.")
        return analysis

    counts = class_dist["binary"]["counts"]
    pcts = class_dist["binary"]["percentages"]
    total = sum(counts.values())

    if len(counts) < 2:
        analysis.append("WARNING: Only one class present — cannot train a binary classifier.")
        return analysis

    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    ratio = counts[minority_class] / total if total > 0 else 0.0

    analysis.append(
        f"Majority class: {'Benign' if majority_class == 0 else 'Malicious'} "
        f"({counts[majority_class]:,} samples, {pcts[majority_class]}%)"
    )
    analysis.append(
        f"Minority class: {'Benign' if minority_class == 0 else 'Malicious'} "
        f"({counts[minority_class]:,} samples, {pcts[minority_class]}%)"
    )
    analysis.append(f"Imbalance ratio: 1:{counts[majority_class] / max(counts[minority_class], 1):.1f}")

    # Implications
    analysis.append("")
    analysis.append("Implications for evaluation:")
    if ratio < 0.1:
        analysis.append("  • SEVERE imbalance — accuracy is misleading.")
        analysis.append("  • Use precision, recall, F1, and AUC-PR as primary metrics.")
        analysis.append("  • Consider SMOTE, class weights, or threshold tuning in Phase 3.")
    elif ratio < 0.3:
        analysis.append("  • MODERATE imbalance — accuracy may overstate performance.")
        analysis.append("  • Prefer F1-score and ROC-AUC over raw accuracy.")
        analysis.append("  • Class-weighted models recommended.")
    else:
        analysis.append("  • Reasonably balanced — standard metrics are informative.")
        analysis.append("  • Still verify per-class recall for rare attack types.")

    # Multiclass note
    if "multiclass" in class_dist:
        mc_counts = class_dist["multiclass"]["counts"]
        rare = {k: v for k, v in mc_counts.items() if v < 100}
        if rare:
            analysis.append("")
            analysis.append(f"Rare attack labels (<100 samples): {list(rare.keys())}")
            analysis.append(
                "  • These labels may be too rare for reliable per-class evaluation."
            )
            analysis.append(
                "  • Do NOT remove them from the test set — they represent real attack types."
            )
            analysis.append(
                "  • Consider grouping or stratified sampling during training."
            )

    return analysis
