"""
data_loader.py — Discover, validate and load raw CIC-IDS2017 CSV files.

Design decisions
----------------
* Column names are normalised to lowercase-with-underscores on load so that
  downstream code is never surprised by whitespace or casing differences.
* Each row is tagged with a ``_source_file`` column so that provenance is
  always traceable.
* Files whose schemas are incompatible (different column sets) are reported
  but never silently concatenated.
* Large files are read in configurable chunks to keep peak memory bounded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chardet
import numpy as np
import pandas as pd

from src.config import (
    CSV_CHUNK_SIZE,
    CSV_ENCODING,
    LABEL_COLUMN,
    LABEL_COLUMN_RAW,
    RAW_DATA_DIR,
    SOURCE_FILE_COLUMN,
    SUPPORTED_EXTENSIONS,
    setup_logging,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.data_loader")


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_column_name(col: str) -> str:
    """Lowercase, strip whitespace, replace interior spaces / slashes with '_'."""
    return col.strip().lower().replace(" ", "_").replace("/", "_")


def _detect_encoding(filepath: Path, sample_bytes: int = 100_000) -> str:
    """Use chardet to guess encoding if UTF-8 fails."""
    with open(filepath, "rb") as f:
        raw = f.read(sample_bytes)
    result = chardet.detect(raw)
    encoding = result.get("encoding", CSV_ENCODING) or CSV_ENCODING
    log.debug("Detected encoding for %s: %s (confidence %.0f%%)",
              filepath.name, encoding, (result.get("confidence", 0) or 0) * 100)
    return encoding


# ── public API ───────────────────────────────────────────────────────────────

def discover_csv_files(directory: Optional[Path] = None) -> List[Path]:
    """Return sorted list of CSV files in *directory* (default: RAW_DATA_DIR)."""
    directory = Path(directory) if directory else RAW_DATA_DIR
    if not directory.exists():
        log.warning("Data directory does not exist: %s", directory)
        return []

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    log.info("Discovered %d CSV file(s) in %s", len(files), directory)
    return files


def load_single_csv(
    filepath: Path,
    chunk_size: Optional[int] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Load one CSV, normalise columns, and tag with source filename.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to the CSV file.
    chunk_size : int | None
        If provided, file is read in chunks of this many rows and then
        concatenated. Useful for very large files.
    encoding : str | None
        Force a specific encoding. If ``None``, UTF-8 is tried first,
        then chardet-based fallback.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is empty (no columns or no rows).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # --- determine encoding ---------------------------------------------------
    enc = encoding or CSV_ENCODING
    read_kw = dict(low_memory=False)

    def _read(e: str) -> pd.DataFrame:
        if chunk_size and chunk_size > 0:
            chunks: List[pd.DataFrame] = []
            for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size, encoding=e, **read_kw)):
                chunks.append(chunk)
                log.debug("Read chunk %d (%d rows) from %s", i + 1, len(chunk), filepath.name)
            if not chunks:
                raise ValueError(f"CSV file is empty: {filepath}")
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(filepath, encoding=e, **read_kw)

    try:
        df = _read(enc)
    except UnicodeDecodeError:
        enc = _detect_encoding(filepath)
        log.info("Falling back to detected encoding '%s' for %s", enc, filepath.name)
        df = _read(enc)

    if df.empty and df.columns.size == 0:
        raise ValueError(f"CSV file is empty (no columns): {filepath}")

    # --- normalize column names -----------------------------------------------
    df.columns = [_normalize_column_name(c) for c in df.columns]

    # --- tag source file ------------------------------------------------------
    df[SOURCE_FILE_COLUMN] = filepath.name

    log.info("Loaded %s — %d rows × %d cols", filepath.name, len(df), len(df.columns))
    return df


def _check_schema_compatibility(
    dataframes: List[pd.DataFrame],
    filenames: List[str],
) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """Keep only DataFrames whose column sets match the first one.

    Returns (compatible_dfs, compatible_filenames, error_messages).
    """
    if not dataframes:
        return [], [], []

    ref_cols = set(dataframes[0].columns)
    compatible: List[pd.DataFrame] = [dataframes[0]]
    compatible_files: List[str] = [filenames[0]]
    errors: List[str] = []

    for df, name in zip(dataframes[1:], filenames[1:]):
        if set(df.columns) != ref_cols:
            missing = ref_cols - set(df.columns)
            extra = set(df.columns) - ref_cols
            msg = (
                f"Schema mismatch in '{name}': "
                f"missing={missing or '∅'}, extra={extra or '∅'}"
            )
            errors.append(msg)
            log.warning(msg)
        else:
            compatible.append(df)
            compatible_files.append(name)

    return compatible, compatible_files, errors


def load_all_csvs(
    directory: Optional[Path] = None,
    chunk_size: Optional[int] = CSV_CHUNK_SIZE,
) -> Tuple[pd.DataFrame, Dict]:
    """Discover, load, validate and combine all CSV files.

    Returns
    -------
    (combined_df, report)
        ``report`` contains keys: files_found, files_loaded, files_skipped,
        total_rows, total_cols, schema_errors, load_errors.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the directory.
    """
    files = discover_csv_files(directory)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {directory or RAW_DATA_DIR}"
        )

    dataframes: List[pd.DataFrame] = []
    filenames: List[str] = []
    load_errors: List[str] = []

    for fp in files:
        try:
            df = load_single_csv(fp, chunk_size=chunk_size)
            dataframes.append(df)
            filenames.append(fp.name)
        except Exception as exc:
            msg = f"Failed to load '{fp.name}': {exc}"
            load_errors.append(msg)
            log.error(msg)

    if not dataframes:
        raise ValueError(
            "All CSV files failed to load. Errors:\n" + "\n".join(load_errors)
        )

    compatible, compatible_files, schema_errors = _check_schema_compatibility(dataframes, filenames)

    combined = pd.concat(compatible, ignore_index=True)

    report: Dict = {
        "files_found": len(files),
        "files_loaded": len(compatible),
        "files_skipped": len(files) - len(compatible),
        "total_rows": len(combined),
        "total_cols": len(combined.columns),
        "schema_errors": schema_errors,
        "load_errors": load_errors,
        "filenames_loaded": compatible_files,
    }

    log.info(
        "Combined dataset: %d rows × %d cols from %d file(s)",
        report["total_rows"],
        report["total_cols"],
        report["files_loaded"],
    )
    return combined, report
