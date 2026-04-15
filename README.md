# Zero-Shot vs Fine-Tuned RoBERTa for Misinformation Detection

## Overview

This project is a graduation thesis comparing two approaches to misinformation classification on tweets:

- **Fine-Tuned RoBERTa**: `roberta-base` fine-tuned via 5-fold stratified cross-validation on labeled tweet datasets.
- **Zero-Shot LLM (Gemini Pro)**: `gemini-1.5-pro` with dataset-specific Chain-of-Thought prompts, no task-specific training.

Three real-world tweet datasets are used, each covering a different topic and annotation scheme. The goal is to determine whether a powerful zero-shot LLM can match or surpass a fine-tuned transformer without any labeled training data.

---

## Directory Structure

```
ZEROSHO_CODE/
├── data/
│   ├── raw/                        # Original unmodified source files
│   ├── processed/                  # Cleaned full datasets (after preprocessing)
│   └── gold_standard/              # Train / val / test CSVs per dataset
├── notebooks/
│   ├── manchester/
│   │   ├── 01_EDA_Manchester.ipynb
│   │   └── 02_Preprocessing_Manchester.ipynb
│   ├── monkeypox/
│   │   ├── 03_EDA_Monkeypox.ipynb
│   │   └── 04_Preprocessing_Monkeypox.ipynb
│   ├── pheme/
│   │   ├── 05_EDA_PHEME.ipynb
│   │   └── 06_Preprocessing_PHEME.ipynb
│   ├── 07_RoBERTa_Finetuning.ipynb
│   ├── 08_ZeroShot_Classification.ipynb
│   └── 09_Comparison.ipynb
├── scripts/
│   ├── preprocessing.py            # CLI preprocessing pipeline (all 3 datasets)
│   ├── train_manchester_roberta.py
│   ├── train_monkeypox_roberta.py
│   ├── train_pheme_roberta.py
│   ├── zeroshot_manchester_ollama.py
│   ├── zeroshot_monkeypox_ollama.py
│   ├── zeroshot_pheme_ollama.py
│   └── plot_training_curves.py
├── results/
│   ├── figures/                    # All saved plots (.png)
│   ├── models/                     # Saved RoBERTa model checkpoints
│   ├── predictions/                # Per-dataset summary CSVs
│   └── master_results.csv          # Aggregated comparison table (notebook 09)
├── config.py                       # Central configuration (paths, hyperparams, labels)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset    | Source tweets | Gold standard | Labels                          | Text column     |
|------------|--------------|---------------|---------------------------------|-----------------|
| Manchester | 89,147       | ~2,427        | `reliable` / `misinformation`   | `cleaned_tweet` |
| Monkeypox  | 6,287        | ~3,043        | `reliable` / `misinformation`   | `cleaned_tweet` |
| PHEME      | 62,443       | ~12,887       | `not_rumour` / `rumour`         | `cleaned_tweet` |

### Manchester
Tweets related to the 2017 Manchester Arena bombing. Labeled via the `Rumour` column (`True` → `reliable`, `Fake` → `misinformation`). Raw file: `data/raw/manchester_raw.xlsx`.

### Monkeypox
Tweets about the 2022 Monkeypox outbreak. Labels come from a binary classifier column (`binary_class`: 0 = reliable, 1 = misinformation). Combined from two CSVs: `monkeypox.csv` and `monkeypox-followup.csv`.

### PHEME
A large multi-event rumour dataset. Labels from `is_rumor` column (0.0 = `not_rumour`, 1.0 = `rumour`). Raw file: `data/raw/PHEME-rumourdetection.csv`.

---

## How to Reproduce

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (required for RoBERTa fine-tuning, notebook 07)
- Gemini API key with access to `gemini-1.5-pro` (required for notebook 08)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data files

Put the following files in `data/raw/`:
- `manchester_raw.xlsx`
- `monkeypox.csv`
- `monkeypox-followup.csv`
- `PHEME-rumourdetection.csv`

### 3. Run preprocessing notebooks (one per dataset)

Each notebook cleans text, builds a balanced gold standard, and saves train/val/test splits.

| Notebook | Dataset | Output |
|----------|---------|--------|
| `notebooks/manchester/02_Preprocessing_Manchester.ipynb` | Manchester | `data/gold_standard/manchester_*.csv` |
| `notebooks/monkeypox/04_Preprocessing_Monkeypox.ipynb`  | Monkeypox  | `data/gold_standard/monkeypox_*.csv`  |
| `notebooks/pheme/06_Preprocessing_PHEME.ipynb`           | PHEME      | `data/gold_standard/pheme_*.csv`      |

Alternatively, run all preprocessing via the CLI script:

```bash
python scripts/preprocessing.py
```

### 4. Fine-tune RoBERTa (notebook 07)

Open `notebooks/07_RoBERTa_Finetuning.ipynb`. Set the `DATASET` variable to `manchester`, `monkeypox`, or `pheme`, then run all cells. Repeat for each dataset.

- Uses stratified 5-fold CV, then trains a final model on train+val.
- Outputs: `results/models/{dataset}_roberta_final/` and `results/predictions/{dataset}_roberta_summary.csv`

### 5. Zero-shot classification with Gemini (notebook 08)

Open `notebooks/08_ZeroShot_Classification.ipynb`.

1. In cell 2, set your API key: `GEMINI_API_KEY = "your-key-here"`
2. Set the `DATASET` variable to `manchester`, `monkeypox`, or `pheme`.
3. Run all cells. Repeat for each dataset.

- Output: `results/predictions/{dataset}_zeroshot_summary.csv`

### 6. Generate comparison figures (notebook 09)

Open `notebooks/09_Comparison.ipynb` and run all cells. This loads all summary CSVs from the previous steps and produces:

- Grouped bar charts, heatmaps, confusion matrices
- Per-dataset winner analysis
- `results/master_results.csv` — the full aggregated results table

---

## API Key Setup (Gemini)

Notebook 08 requires a Google Gemini API key:

1. Obtain a key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Open `notebooks/08_ZeroShot_Classification.ipynb`
3. In **cell 2**, set:
   ```python
   GEMINI_API_KEY = "your-key-here"
   ```

The key is never written to disk or committed to version control. Do not hard-code it anywhere else.

---

## Results

After running all notebooks:

- **Master results table**: `results/master_results.csv`
- **Figures**: `results/figures/comparison_*.png`

Key metrics reported: Accuracy, Precision, Recall, F1-score (macro and per-class), and AUC-ROC where applicable.

---

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `transformers` | RoBERTa model and tokenizer |
| `torch` | PyTorch (GPU training) |
| `datasets` | HuggingFace dataset utilities |
| `scikit-learn` | Stratified splits, metrics |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Plotting |
| `google-generativeai` | Gemini API client |

---

## Configuration

All paths, label maps, and hyperparameters are centralized in `config.py`. Import it in any notebook or script:

```python
from config import DATASETS, TRAIN_PARAMS, LABEL_MAPS
```
