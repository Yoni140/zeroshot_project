# Zero-Shot vs Fine-Tuned Transformers for Misinformation Detection

## Overview

This project is a graduation thesis comparing approaches to misinformation / rumour classification on tweets:

- **Fine-tuned encoders**: `roberta-base`, `vinai/bertweet-base`, `microsoft/MiniLM-L12-H384-uncased`, and `answerdotai/ModernBERT-base` via stratified cross-validation on labeled tweet datasets.
- **Zero-Shot LLMs**: local (Ollama) and cloud models with dataset-specific prompts, no task-specific training.

Core topic datasets (Manchester, Monkeypox, PHEME) plus **PHEME per-event** splits are used so results can be compared across events as well as across models.

---

## Directory Structure

```
zeroshot_project/
├── data/
│   ├── raw/                        # Original unmodified source files
│   ├── processed/                  # Cleaned full datasets (after preprocessing)
│   └── gold_standard/              # Train / val / test CSVs per dataset
├── notebooks/
│   ├── manchester/
│   ├── monkeypox/
│   ├── pheme/
│   └── models/
├── scripts/
│   ├── preprocessing.py            # CLI preprocessing (core + PHEME events)
│   ├── train_transformer.py        # Unified fine-tuning (RoBERTa/BERTweet/MiniLM/ModernBERT)
│   ├── run_all_transformers.py     # Batch runner over datasets × models
│   ├── plot_pheme_events.py        # EDA + t-SNE plots for PHEME events
│   ├── train_*_roberta.py          # Legacy per-dataset RoBERTa scripts
│   ├── zeroshot_*.py
│   └── run_comparison.py
├── results/
│   ├── figures/                    # Plots per dataset + comparison/
│   ├── models/                     # Saved checkpoints
│   ├── predictions/                # Summary + prediction CSVs
│   └── master_results.csv
├── config.py
├── requirements.txt
└── README.md
```

---

## Datasets

### Core datasets

| Dataset    | Labels (gold)                                      | Text column     |
|------------|----------------------------------------------------|-----------------|
| Manchester | `reliable` / `misinformation` / `unrelated`        | `cleaned_tweet` |
| Monkeypox  | `reliable` / `misinformation` / `unrelated`        | `cleaned_tweet` |
| PHEME      | `not_rumour` / `rumour` / `unrelated`              | `cleaned_tweet` |

### PHEME event datasets (2-class)

Raw CSVs in `data/raw/`. Cleaned → `data/processed/{name}_clean.csv`. Gold splits → `data/gold_standard/`.

| Dataset key          | Notes |
|----------------------|-------|
| `pheme_all_events`   | All annotated PHEME source tweets combined |
| `charliehebdo`       | Charlie Hebdo attack |
| `sydneysiege`        | Sydney Lindt café siege |
| `ferguson`           | Ferguson unrest |
| `ottawashooting`     | Ottawa Parliament shooting *(filename spelling)* |
| `germanwings-crash`  | Germanwings Flight 9525 |
| `putinmissing`       | Putin missing rumours |
| `prince-toronto`     | Prince secret Toronto show *(very imbalanced)* |
| `gurlitt`            | Gurlitt art bequest |
| `ebola-essien`       | Michael Essien Ebola rumour *(single-class after clean — training skipped)* |

Event gold standards use labels `not_rumour` / `rumour` (no `unrelated`; event files have no `topic==unknown`).

---

## Encoder models

| Key          | HuggingFace id                         |
|--------------|----------------------------------------|
| `roberta`    | `roberta-base`                         |
| `bertweet`   | `vinai/bertweet-base`                  |
| `minilm`     | `microsoft/MiniLM-L12-H384-uncased`    |
| `modernbert` | `answerdotai/ModernBERT-base`          |

Training protocol (same for all encoders): stratified K-fold CV on the gold standard, then a final model on train+val evaluated on held-out test. Outputs:

- `results/predictions/{dataset}_{model}_summary.csv`
- `results/predictions/{dataset}_{model}_test_predictions.csv`
- `results/figures/{dataset}/{dataset}_{model}_*.png`
- `results/models/{dataset}_{model}_final/`

For `roberta`, legacy filenames `{dataset}_roberta_summary.csv` are also written for existing comparison scripts.

---

## How to Reproduce

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended for fine-tuning
- API keys only if running cloud zero-shot notebooks/scripts

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data files

Put core files and PHEME event CSVs in `data/raw/`:

- `manchester_raw.xlsx`, `monkeypox.csv`, `monkeypox-followup.csv`
- Optional legacy: `PHEME-rumourdetection.csv`
- Event CSVs: `pheme_all_events.csv`, `gurlitt.csv`, `germanwings-crash.csv`, `ebola-essien.csv`, `charliehebdo.csv`, `ferguson.csv`, `ottawashooting.csv`, `prince-toronto.csv`, `putinmissing.csv`, `sydneysiege.csv`

### 3. Preprocess

```bash
python scripts/preprocessing.py
```

This writes cleaned CSVs under `data/processed/` and gold train/val/test under `data/gold_standard/` for Manchester, Monkeypox, legacy PHEME (if present), and all PHEME event datasets.

### 4. EDA / projection plots (PHEME events)

```bash
python scripts/plot_pheme_events.py
```

### 5. Fine-tune encoders

Single run:

```bash
python scripts/train_transformer.py --dataset charliehebdo --model roberta
python scripts/train_transformer.py --dataset manchester --model bertweet
```

Batch (all datasets × all models; resumes with `--skip-if-done`):

```bash
python scripts/run_all_transformers.py --skip-if-done
```

Useful filters:

```bash
# Only new event datasets, all 4 encoders
python scripts/run_all_transformers.py --events-only --skip-if-done

# Only BERTweet / MiniLM / ModernBERT on everything (RoBERTa already done)
python scripts/run_all_transformers.py --new-models-only --skip-if-done
```

Legacy RoBERTa-only scripts (`scripts/train_pheme_roberta.py`, etc.) remain available.

### 6. Zero-shot LLMs (cloud)

Requires `GROQ_API_KEY` and/or `GEMINI_API_KEY` in a project-root `.env` (see `.env.example`).

```bash
# All PHEME event datasets × cloud models (gpt_oss, llama33, qwen3, gemini_flash)
python scripts/run_all_cloud.py --events-only

# Single combo
python scripts/run_all_cloud.py --dataset charliehebdo --model gemini_flash
```

Outputs: `results/predictions/{dataset}_{model}_summary.csv` (+ test predictions / checkpoints).

### 7. Comparison (all models × all datasets)

```bash
python scripts/compare_all_models.py
```

Writes `results/master_results_all_models.csv` and figures under `results/figures/comparison/` (heatmaps + per-dataset bars).

---

## Configuration

Paths, dataset registry, label maps, and encoder hub ids live in `config.py`:

```python
from config import DATASETS, PHEME_EVENT_KEYS, ENCODER_MODELS, TRAIN_PARAMS, LABEL_MAPS
```

---

## Results

- Per-run summaries: `results/predictions/*_summary.csv`
- Figures: `results/figures/{dataset}/` and `results/figures/comparison/`
- Aggregated table (when comparison is run): `results/master_results.csv`

Key metrics: Accuracy, Precision, Recall, F1 (macro / weighted / positive class).

---

## Requirements

See `requirements.txt`. Key packages: `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.
