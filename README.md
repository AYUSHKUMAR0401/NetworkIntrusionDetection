# AI-NIDS — AI-Powered Network Intrusion Detection System

> **Phases 1, 2 & 3 Complete: Data Engineering, EDA, Feature Engineering & ML Model Development**
>
> This project builds a reliable data engineering and ML pipeline for classifying network traffic
> as **Benign** or **Malicious** using machine learning. Phase 1 & 2 cover data loading,
> cleaning, validation, exploratory analysis, and scikit-learn feature engineering pipelines.
> Phase 3 implements model training (Logistic Regression, Decision Tree, Random Forest),
> model evaluation, recall-prioritised selection, model persistence, and batch/single prediction modules.

⚠️ **This is NOT a real-time intrusion detector.** It is an ML data-engineering & classifier
foundation that will be extended in later phases (PCAP/Live capture & Dashboard).

---

## 📁 Folder Structure

```
AI-NIDS/
├── data/
│   ├── raw/               ← Place CIC-IDS2017 CSVs here
│   └── processed/         ← Cleaned output CSV, summary JSON, feature_metadata.json
├── models/                ← Fitted pipeline (.joblib) & model metadata (.json)
├── notebooks/             ← Jupyter exploration notebooks
├── src/
│   ├── __init__.py
│   ├── config.py          ← Paths, label mappings, options
│   ├── data_loader.py     ← CSV discovery, loading, schema checks
│   ├── preprocess.py      ← Cleaning, target creation, dedup
│   ├── validate_data.py   ← Post-processing validation checks
│   ├── eda.py             ← Exploratory Data Analysis & plot generation
│   ├── feature_engineering.py ← Preprocessing pipelines & diagnostics
│   ├── run_eda.py         ← CLI script to execute EDA & feature engineering
│   ├── model_utils.py     ← Model persistence, schema validation, package versioning
│   ├── evaluate.py        ← Evaluation metrics, confusion matrix & ROC-AUC plots
│   ├── train.py           ← Training orchestration & best model selection
│   └── predict.py         ← Prediction module (DataFrame & single sample)
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_validate_data.py
│   ├── test_eda.py
│   ├── test_feature_engineering.py
│   └── test_phase3.py
├── reports/               ← Validation, EDA & Model reports + PNG plots (auto-generated)
├── logs/                  ← Runtime logs
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🐍 Python Environment Setup (Windows)

### 1. Prerequisites

* **Python 3.10+** installed and available on `PATH`.
* `pip` (bundled with Python).

### 2. Create a virtual environment

```powershell
cd C:\Users\user\Desktop\NDS
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 📊 Dataset: CIC-IDS2017

### Where to obtain it

The **CIC-IDS2017** dataset is freely available from the Canadian Institute
for Cybersecurity:

* **Official page**: <https://www.unb.ca/cic/datasets/ids-2017.html>
* **Direct CSVs**: Look for the "MachineLearningCSV" download containing
  eight CSV files (Monday through Friday).

### Where to place raw files

Place all downloaded CSV files in:

```
data/raw/
```

The data loader will automatically discover every `.csv` file in that folder.

---

## 🚀 How to Run

### Run the preprocessing pipeline

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.preprocess          # (or call from a script)
```

To process data programmatically:

```python
from src.data_loader import load_all_csvs
from src.preprocess import run_preprocessing
from src.validate_data import run_validation

df, load_report = load_all_csvs()
processed, summary = run_preprocessing(df)
passed, val_report = run_validation(processed)
```

### Run data validation

```python
from src.validate_data import run_validation
passed, report = run_validation(processed_df)
# Report saved to reports/validation_report.txt
```

### Run EDA and feature engineering pipeline

```powershell
python -m src.run_eda               # auto-detects real CSVs or uses synthetic data
python -m src.run_eda --synthetic    # force synthetic data evaluation
```

### Run Phase 3 Model Training & Evaluation

```powershell
python -m src.train                 # auto-detects processed CSV or uses synthetic data
python -m src.train --synthetic     # force training on synthetic data
```

### Run Phase 3 Predictions

```python
from src.predict import predict, predict_single
import pandas as pd

# Batch dataframe prediction
results = predict(X_dataframe)

# Single sample prediction
single = predict_single({"flow_duration": 1500.0, "total_fwd_packets": 10.0, ...})
```

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

All 93 unit tests use small synthetic fixtures — they do **not** require the real
dataset or an Internet connection.

---

## ✅ Expected Outputs

| Output | Location |
|---|---|
| Processed dataset | `data/processed/processed_data.csv` |
| Preprocessing summary | `data/processed/preprocessing_summary.json` |
| Validation report | `reports/validation_report.txt` |
| Runtime logs | `logs/ai_nids.log` |

---

## ⚙️ Configuration

All paths, label mappings and preprocessing options live in
[`src/config.py`](src/config.py). Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `MISSING_VALUE_STRATEGY` | `"median"` | How NaNs are filled (`median`, `mean`, `drop`) |
| `DROP_DUPLICATES` | `True` | Remove exact duplicate rows |
| `CSV_CHUNK_SIZE` | `100 000` | Rows per chunk for memory-bounded reading |
| `BENIGN_LABELS` | `{"benign"}` | Labels mapped to class 0 |
| `KNOWN_ATTACK_LABELS` | *(14 labels)* | Known CIC-IDS2017 attack categories |

---

## ⚠️ Known Limitations

1. **No model training yet** — Phase 1 is data engineering only.
2. **No feature scaling** — will be applied in Phase 2 with sklearn pipelines.
3. **No real-time capture** — the system processes CSV files, not live packets.
4. **CIC-IDS2017 only** — adapting to other datasets requires updating label
   mappings in `config.py`.
5. **Single-machine** — no distributed processing.
6. **Dataset not included** — you must download CIC-IDS2017 separately (≈ 1 GB).

---

## 📝 License

This project is for educational purposes. The CIC-IDS2017 dataset is
provided by the Canadian Institute for Cybersecurity under their own terms.
