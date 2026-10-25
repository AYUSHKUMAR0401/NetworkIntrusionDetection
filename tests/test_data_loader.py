"""
test_data_loader.py — Pytest tests for src/data_loader.py

All tests use small, temporary synthetic CSV fixtures and never require
the real CIC-IDS2017 dataset or an Internet connection.
"""

import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    _normalize_column_name,
    discover_csv_files,
    load_all_csvs,
    load_single_csv,
)
from src.config import LABEL_COLUMN, SOURCE_FILE_COLUMN


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_csv_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a valid minimal CSV."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        " Label, Flow Duration, Total Fwd Packets\n"
        "BENIGN, 100, 5\n"
        "Bot, 200, 10\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def two_csv_dir(tmp_path: Path) -> Path:
    """Two compatible CSV files."""
    for name, label in [("a.csv", "BENIGN"), ("b.csv", "DDoS")]:
        (tmp_path / name).write_text(
            " Label, Flow Duration\n"
            f"{label}, 100\n",
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def incompatible_csv_dir(tmp_path: Path) -> Path:
    """Two CSVs with different schemas."""
    (tmp_path / "a.csv").write_text(
        " Label, Flow Duration\nBENIGN,100\n", encoding="utf-8"
    )
    (tmp_path / "b.csv").write_text(
        " Label, Extra Column\nBot,999\n", encoding="utf-8"
    )
    return tmp_path


# ── column normalisation ─────────────────────────────────────────────────────

def test_normalize_column_name():
    assert _normalize_column_name(" Flow Duration ") == "flow_duration"
    assert _normalize_column_name("Fwd Pkt Len Max") == "fwd_pkt_len_max"
    assert _normalize_column_name("Flow Bytes/s") == "flow_bytes_s"


# ── discover ─────────────────────────────────────────────────────────────────

def test_discover_finds_csv(tmp_csv_dir: Path):
    files = discover_csv_files(tmp_csv_dir)
    assert len(files) == 1
    assert files[0].suffix == ".csv"


def test_discover_empty_dir(tmp_path: Path):
    files = discover_csv_files(tmp_path)
    assert files == []


def test_discover_nonexistent_dir(tmp_path: Path):
    files = discover_csv_files(tmp_path / "nope")
    assert files == []


# ── load_single_csv ──────────────────────────────────────────────────────────

def test_load_valid_csv(tmp_csv_dir: Path):
    csv_file = list(tmp_csv_dir.glob("*.csv"))[0]
    df = load_single_csv(csv_file)
    assert len(df) == 2
    assert LABEL_COLUMN in df.columns  # normalised
    assert SOURCE_FILE_COLUMN in df.columns
    assert df[SOURCE_FILE_COLUMN].iloc[0] == csv_file.name


def test_load_with_chunks(tmp_csv_dir: Path):
    csv_file = list(tmp_csv_dir.glob("*.csv"))[0]
    df = load_single_csv(csv_file, chunk_size=1)
    assert len(df) == 2


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_single_csv(tmp_path / "does_not_exist.csv")


def test_load_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(Exception):
        load_single_csv(empty)


def test_column_names_normalised(tmp_csv_dir: Path):
    csv_file = list(tmp_csv_dir.glob("*.csv"))[0]
    df = load_single_csv(csv_file)
    for col in df.columns:
        assert col == col.lower().strip()


# ── load_all_csvs ───────────────────────────────────────────────────────────

def test_load_all_combines(two_csv_dir: Path):
    df, report = load_all_csvs(two_csv_dir, chunk_size=None)
    assert len(df) == 2
    assert report["files_loaded"] == 2
    assert report["schema_errors"] == []


def test_load_all_reports_schema_mismatch(incompatible_csv_dir: Path):
    df, report = load_all_csvs(incompatible_csv_dir, chunk_size=None)
    assert report["files_loaded"] == 1
    assert len(report["schema_errors"]) == 1


def test_load_all_no_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_all_csvs(tmp_path)


# ── encoding fallback ───────────────────────────────────────────────────────

def test_load_latin1_csv(tmp_path: Path):
    """CSV saved with latin-1 encoding should be loadable via fallback."""
    csv_file = tmp_path / "latin.csv"
    csv_file.write_bytes(
        " Label, Value\nBENIGN, 42\nBénign, 99\n".encode("latin-1")
    )
    df = load_single_csv(csv_file)
    assert len(df) == 2
