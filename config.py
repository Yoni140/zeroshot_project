"""
config.py - Central configuration for the Zero-Shot vs RoBERTa project.

Usage:
    from config import DATASETS, LABEL_MAPS, TRAIN_PARAMS, PATHS
"""

import os

# ──────────────────────────────────────────────
# Base directory (project root, wherever config.py lives)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Top-level paths
# ──────────────────────────────────────────────
PATHS = {
    "data_raw":        os.path.join(BASE_DIR, "data", "raw"),
    "data_processed":  os.path.join(BASE_DIR, "data", "processed"),
    "gold_standard":   os.path.join(BASE_DIR, "data", "gold_standard"),
    "predictions":     os.path.join(BASE_DIR, "results", "predictions"),
    "figures":         os.path.join(BASE_DIR, "results", "figures"),
    "models":          os.path.join(BASE_DIR, "results", "models"),
    "master_results":  os.path.join(BASE_DIR, "results", "master_results.csv"),
}

# ──────────────────────────────────────────────
# Dataset definitions
# ──────────────────────────────────────────────
# Each entry describes one dataset end-to-end.
# Keys used throughout notebooks and scripts.

DATASETS = {
    "manchester": {
        "name":        "Manchester",
        "raw_file":    os.path.join(PATHS["data_raw"], "manchester_raw.xlsx"),
        "clean_file":  os.path.join(PATHS["data_processed"], "manchester_clean.csv"),
        "gold_file":   os.path.join(PATHS["gold_standard"], "manchester_gold_standard.csv"),
        "train_file":  os.path.join(PATHS["gold_standard"], "manchester_train.csv"),
        "val_file":    os.path.join(PATHS["gold_standard"], "manchester_val.csv"),
        "test_file":   os.path.join(PATHS["gold_standard"], "manchester_test.csv"),
        "text_col":    "cleaned_tweet",
        "label_col":   "label",
        "labels":      ["reliable", "misinformation"],
        "roberta_dir": os.path.join(PATHS["models"], "manchester_roberta_final"),
        "roberta_summary": os.path.join(PATHS["predictions"], "manchester_roberta_summary.csv"),
        "zeroshot_summary": os.path.join(PATHS["predictions"], "manchester_zeroshot_summary.csv"),
    },
    "monkeypox": {
        "name":        "Monkeypox",
        "raw_files": [
            os.path.join(PATHS["data_raw"], "monkeypox.csv"),
            os.path.join(PATHS["data_raw"], "monkeypox-followup.csv"),
        ],
        "clean_file":  os.path.join(PATHS["data_processed"], "monkeypox_clean.csv"),
        "gold_file":   os.path.join(PATHS["gold_standard"], "monkeypox_gold_standard.csv"),
        "train_file":  os.path.join(PATHS["gold_standard"], "monkeypox_train.csv"),
        "val_file":    os.path.join(PATHS["gold_standard"], "monkeypox_val.csv"),
        "test_file":   os.path.join(PATHS["gold_standard"], "monkeypox_test.csv"),
        "text_col":    "cleaned_tweet",
        "label_col":   "label",
        "labels":      ["reliable", "misinformation"],
        "roberta_dir": os.path.join(PATHS["models"], "monkeypox_roberta_final"),
        "roberta_summary": os.path.join(PATHS["predictions"], "monkeypox_roberta_summary.csv"),
        "zeroshot_summary": os.path.join(PATHS["predictions"], "monkeypox_zeroshot_summary.csv"),
    },
    "pheme": {
        "name":        "PHEME",
        "raw_file":    os.path.join(PATHS["data_raw"], "PHEME-rumourdetection.csv"),
        "clean_file":  os.path.join(PATHS["data_processed"], "pheme_clean.csv"),
        "gold_file":   os.path.join(PATHS["gold_standard"], "pheme_gold_standard.csv"),
        "train_file":  os.path.join(PATHS["gold_standard"], "pheme_train.csv"),
        "val_file":    os.path.join(PATHS["gold_standard"], "pheme_val.csv"),
        "test_file":   os.path.join(PATHS["gold_standard"], "pheme_test.csv"),
        "text_col":    "cleaned_tweet",
        "label_col":   "label",
        "labels":      ["not_rumour", "rumour"],
        "roberta_dir": os.path.join(PATHS["models"], "pheme_roberta_final"),
        "roberta_summary": os.path.join(PATHS["predictions"], "pheme_roberta_summary.csv"),
        "zeroshot_summary": os.path.join(PATHS["predictions"], "pheme_zeroshot_summary.csv"),
    },
}

# ──────────────────────────────────────────────
# Label maps (raw source value → unified label string)
# ──────────────────────────────────────────────
LABEL_MAPS = {
    "manchester": {
        # df['Rumour'] raw values after .capitalize()
        "True":        "reliable",
        "Fake":        "misinformation",
        "Not related": "not_related",
    },
    "monkeypox": {
        # df['binary_class'] integer values
        0: "reliable",
        1: "misinformation",
    },
    "pheme": {
        # df['is_rumor'] float values
        0.0: "not_rumour",
        1.0: "rumour",
    },
}

# ──────────────────────────────────────────────
# Training hyperparameters (RoBERTa fine-tuning)
# ──────────────────────────────────────────────
TRAIN_PARAMS = {
    "model_name":          "roberta-base",
    "learning_rate":       2e-5,
    "batch_size":          16,
    "num_epochs":          4,
    "max_len":             128,
    "n_folds":             5,
    "early_stopping_patience": 2,
    "fp16":                True,        # mixed precision (requires CUDA)
    "random_state":        42,
    "warmup_ratio":        0.1,
    "weight_decay":        0.01,
}

# ──────────────────────────────────────────────
# Preprocessing parameters
# ──────────────────────────────────────────────
PREPROCESS_PARAMS = {
    "max_chars":        350,    # maximum characters per cleaned tweet
    "min_words":        5,      # minimum word count after cleaning
    "reliable_sample":  2000,   # max reliable tweets sampled for gold standard
    "random_state":     42,
    "test_size":        0.30,   # 70% train, 15% val, 15% test
    "val_size":         0.50,   # split the 30% temp into equal val/test
}

# ──────────────────────────────────────────────
# Zero-shot (Gemini) parameters
# ──────────────────────────────────────────────
ZEROSHOT_PARAMS = {
    "model":            "gemini-1.5-pro",
    "temperature":      0.0,
    "max_retries":      5,
    "initial_backoff":  2,      # seconds before first retry
    "requests_per_min": 60,     # rate limit
}
