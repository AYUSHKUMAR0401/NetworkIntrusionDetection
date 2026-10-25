"""
config.py — Centralized configuration for AI-NIDS.

All project paths, column names, label mappings and preprocessing options
are defined here so that every module uses the same conventions.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — two levels up from src/config.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
REPORTS_DIR     = PROJECT_ROOT / "reports"
LOGS_DIR        = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"
TESTS_DIR       = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------------
# Processed output filenames
# ---------------------------------------------------------------------------
PROCESSED_CSV       = PROCESSED_DIR / "processed_data.csv"
PREPROCESSING_SUMMARY = PROCESSED_DIR / "preprocessing_summary.json"

# ---------------------------------------------------------------------------
# Label / target configuration
# ---------------------------------------------------------------------------
# The raw CIC-IDS2017 column that contains the traffic label
LABEL_COLUMN_RAW = "Label"

# Cleaned column name used after normalization (lowercase + stripped)
LABEL_COLUMN = "label"

# Binary target column created during preprocessing
BINARY_TARGET_COLUMN = "is_malicious"

# Column that stores the original (cleaned) multiclass label
ORIGINAL_LABEL_COLUMN = "original_label"

# Source-file tracking column (excluded from model features)
SOURCE_FILE_COLUMN = "_source_file"

# Columns to exclude from model feature matrix
NON_FEATURE_COLUMNS = {
    LABEL_COLUMN,
    BINARY_TARGET_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    SOURCE_FILE_COLUMN,
}

# ---------------------------------------------------------------------------
# Benign / attack label mapping
# ---------------------------------------------------------------------------
# Explicit list of labels that map to "Benign" (class 0).
# Everything NOT in this set is treated as an attack (class 1).
BENIGN_LABELS = {"benign"}

# Known attack labels in CIC-IDS2017 (for validation, not exhaustive).
# Unknown labels that are NOT benign will still be flagged as attacks
# but a warning will be issued if they are not in this set.
KNOWN_ATTACK_LABELS = {
    "bot",
    "ddos",
    "dos goldeneye",
    "dos hulk",
    "dos slowhttptest",
    "dos slowloris",
    "ftp-patator",
    "heartbleed",
    "infiltration",
    "portscan",
    "ssh-patator",
    "web attack – brute force",
    "web attack – sql injection",
    "web attack – xss",
    # Some CSV exports use slightly different casing / dashes:
    "web attack - brute force",
    "web attack - sql injection",
    "web attack - xss",
}

# Combined known labels (for validation)
ALL_KNOWN_LABELS = BENIGN_LABELS | KNOWN_ATTACK_LABELS

# ---------------------------------------------------------------------------
# Data-loading options
# ---------------------------------------------------------------------------
CSV_CHUNK_SIZE = 100_000        # rows per chunk when streaming large files
CSV_ENCODING   = "utf-8"       # default; chardet fallback is used automatically
SUPPORTED_EXTENSIONS = {".csv"}

# ---------------------------------------------------------------------------
# Model / training outputs  (Phase 3)
# ---------------------------------------------------------------------------
MODELS_DIR          = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "binary_intrusion_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Training options  (Phase 3)
# ---------------------------------------------------------------------------
TEST_SIZE            = 0.2       # fraction of data held out for evaluation
MIN_RECALL_THRESHOLD = 0.80     # minimum acceptable malicious-class recall

# ---------------------------------------------------------------------------
# Preprocessing options
# ---------------------------------------------------------------------------
# Strategy for missing numeric values: "median", "mean", or "drop"
MISSING_VALUE_STRATEGY = "median"

# Whether to drop duplicate rows
DROP_DUPLICATES = True

# Random seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "ai_nids.log"

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging to console + file.

    Returns the root project logger.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_nids")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
