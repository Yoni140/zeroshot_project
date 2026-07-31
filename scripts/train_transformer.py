"""
train_transformer.py — unified fine-tuning for RoBERTa / BERTweet / MiniLM / ModernBERT.

Mirrors scripts/train_pheme_roberta.py (5-fold CV + final train+val → test).

Usage (from project root):
  python scripts/train_transformer.py --dataset charliehebdo --model roberta
  python scripts/train_transformer.py --dataset manchester --model bertweet
  python scripts/train_transformer.py --dataset pheme --model modernbert --skip-if-done
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings('ignore')

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    'roberta':    'roberta-base',
    'bertweet':   'vinai/bertweet-base',
    'minilm':     'microsoft/MiniLM-L12-H384-uncased',
    'modernbert': 'answerdotai/ModernBERT-base',
}

# Datasets that use rumour / not_rumour labels
RUMOUR_DATASETS = {
    'pheme', 'pheme_all_events', 'gurlitt', 'germanwings-crash', 'ebola-essien',
    'charliehebdo', 'ferguson', 'ottawashooting', 'prince-toronto',
    'putinmissing', 'sydneysiege',
}

MISINFO_DATASETS = {'manchester', 'monkeypox'}

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 4
N_FOLDS_DEFAULT = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'gold_standard'
PREDS_DIR = ROOT / 'results' / 'predictions'
MODELS_DIR = ROOT / 'results' / 'models'


def slug(s: str) -> str:
    """Filesystem-safe short name (keep hyphens; they are fine)."""
    return s


def infer_label_maps(df: pd.DataFrame, dataset: str):
    labels_present = sorted(df['label'].dropna().unique().tolist())
    if dataset in RUMOUR_DATASETS or set(labels_present) <= {'not_rumour', 'rumour', 'unrelated'}:
        preferred = [l for l in ['not_rumour', 'rumour', 'unrelated'] if l in labels_present]
        pos = 'rumour' if 'rumour' in preferred else preferred[-1]
    elif dataset in MISINFO_DATASETS or set(labels_present) <= {'reliable', 'misinformation', 'unrelated'}:
        preferred = [l for l in ['reliable', 'misinformation', 'unrelated'] if l in labels_present]
        pos = 'misinformation' if 'misinformation' in preferred else preferred[-1]
    else:
        preferred = labels_present
        pos = preferred[-1]

    label_map = {name: i for i, name in enumerate(preferred)}
    id2label = {i: name for name, i in label_map.items()}
    return preferred, label_map, id2label, pos


def load_tokenizer(hf_name: str, model_key: str):
    kwargs = {}
    if model_key == 'bertweet':
        kwargs['normalization'] = True
    try:
        return AutoTokenizer.from_pretrained(hf_name, **kwargs)
    except Exception:
        return AutoTokenizer.from_pretrained(hf_name, use_fast=False, **kwargs)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok, max_len=MAX_LEN):
        # BERTweet / some tokenizers dislike None
        texts = [str(t) if t is not None else '' for t in texts]
        enc = tok(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
        )
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'attention_mask': self.attention_mask[i],
            'labels': self.labels[i],
        }


def encode_labels(series, label_map):
    return np.array([label_map[l] for l in series])


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


def full_report(true, pred, pos_id, pos_name, tag):
    acc = accuracy_score(true, pred)
    f1m = f1_score(true, pred, average='macro', zero_division=0)
    f1w = f1_score(true, pred, average='weighted', zero_division=0)
    prec = precision_score(true, pred, average='macro', zero_division=0)
    rec = recall_score(true, pred, average='macro', zero_division=0)
    f1p = f1_score(true, pred, labels=[pos_id], average='macro', zero_division=0)
    print(f'[{tag}] Acc={acc:.4f}  F1m={f1m:.4f}  F1w={f1w:.4f}  F1_{pos_name}={f1p:.4f}')
    return {
        'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
        'precision_macro': prec, 'recall_macro': rec, 'f1_pos': f1p,
    }


def get_training_args(output_dir, do_eval=True, save_checkpoints=False):
    # Write trainer state under local TEMP — project path is often cloud-synced and
    # checkpoint I/O there previously hung / aborted the long batch run.
    local_out = Path(os.environ.get('LOCALAPPDATA', os.environ.get('TEMP', '.'))) / 'zeroshot_hf' / Path(str(output_dir)).name
    local_out.mkdir(parents=True, exist_ok=True)
    common = dict(
        output_dir=str(local_out),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to='none',
        dataloader_num_workers=0,
        save_total_limit=1,
    )
    if do_eval and save_checkpoints:
        return TrainingArguments(
            **common,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            greater_is_better=True,
        )
    if do_eval:
        # CV folds: evaluate each epoch but do NOT checkpoint (avoids hung disk I/O).
        return TrainingArguments(
            **common,
            eval_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=False,
        )
    return TrainingArguments(
        **common,
        eval_strategy='no',
        save_strategy='no',
    )


def choose_n_folds(y, requested=N_FOLDS_DEFAULT):
    """Reduce folds when minority class is too small for StratifiedKFold."""
    _, counts = np.unique(y, return_counts=True)
    max_folds = int(counts.min())
    if max_folds < 2:
        return 0
    return max(2, min(requested, max_folds))


def train_one(dataset: str, model_key: str, skip_if_done: bool = False):
    if model_key not in MODELS:
        raise ValueError(f'Unknown model {model_key}. Choose from {list(MODELS)}')

    hf_name = MODELS[model_key]
    ds = slug(dataset)
    tag = f'{ds}_{model_key}'

    summary_path = PREDS_DIR / f'{tag}_summary.csv'
    if skip_if_done and summary_path.exists():
        print(f'[skip] already done: {summary_path.name}')
        return

    figs_dir = ROOT / 'results' / 'figures' / ds
    for d in [PREDS_DIR, MODELS_DIR, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / f'{ds}_train.csv'
    val_path = DATA_DIR / f'{ds}_val.csv'
    test_path = DATA_DIR / f'{ds}_test.csv'
    gold_path = DATA_DIR / f'{ds}_gold_standard.csv'
    for p in [train_path, val_path, test_path, gold_path]:
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}')

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_gold = pd.read_csv(gold_path)

    label_names, label_map, id2label, pos_name = infer_label_maps(df_gold, ds)
    num_labels = len(label_names)
    pos_id = label_map[pos_name]
    text_col = 'cleaned_tweet'

    print(f'\n{"="*60}')
    print(f' DATASET={ds}  MODEL={model_key} ({hf_name})')
    print(f' labels={label_names}  pos={pos_name}')
    print(f' Train={len(df_train):,} Val={len(df_val):,} Test={len(df_test):,} Gold={len(df_gold):,}')
    print(f'{"="*60}')

    if num_labels < 2:
        print(f'[skip] {ds}: single-class data — cannot train a classifier.')
        return

    if len(df_test) == 0:
        print(f'[skip] {ds}: empty test set — cannot evaluate.')
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    tokenizer = load_tokenizer(hf_name, model_key)

    # ── CV ───────────────────────────────────────────────────────────────────
    all_texts = df_gold[text_col].values
    all_labels = encode_labels(df_gold['label'], label_map)
    n_folds = choose_n_folds(all_labels)
    cv_results = []
    all_oof_preds = np.full(len(df_gold), -1, dtype=int)

    if n_folds >= 2:
        print(f'\nStarting {n_folds}-Fold CV on {len(df_gold):,} gold samples...')
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), 1):
            print(f'\n--- Fold {fold}/{n_folds} train={len(train_idx)} val={len(val_idx)} ---')
            fold_train = TweetDataset(all_texts[train_idx], all_labels[train_idx], tokenizer)
            fold_val = TweetDataset(all_texts[val_idx], all_labels[val_idx], tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(
                hf_name, num_labels=num_labels, id2label=id2label, label2id=label_map,
                ignore_mismatched_sizes=True,
            )
            fold_dir = MODELS_DIR / f'{tag}_fold{fold}'
            trainer = Trainer(
                model=model,
                args=get_training_args(fold_dir, do_eval=True, save_checkpoints=False),
                train_dataset=fold_train,
                eval_dataset=fold_val,
                compute_metrics=compute_metrics_fn,
            )
            trainer.train()
            preds_out = trainer.predict(fold_val)
            fold_preds = np.argmax(preds_out.predictions, axis=-1)
            all_oof_preds[val_idx] = fold_preds
            metrics = full_report(all_labels[val_idx], fold_preds, pos_id, pos_name,
                                  f'{ds} fold{fold}')
            metrics['fold'] = fold
            cv_results.append(metrics)
            del model, trainer, fold_train, fold_val
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print('[warn] Skipping CV — not enough samples per class for ≥2 folds.')

    if cv_results:
        cv_df = pd.DataFrame(cv_results).set_index('fold')
        print('\nCV Summary:')
        print(cv_df.round(4).to_string())
        cv_df.to_csv(PREDS_DIR / f'{tag}_cv_results.csv')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mask = all_oof_preds >= 0
        cm = confusion_matrix(all_labels[mask], all_oof_preds[mask],
                              labels=list(range(num_labels)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names, ax=axes[0])
        axes[0].set_title(f'{ds} — OOF CM ({model_key})', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        f1_scores = cv_df['f1_macro'].tolist()
        bars = axes[1].bar(cv_df.index.tolist(), f1_scores, color='steelblue', edgecolor='black')
        axes[1].axhline(np.mean(f1_scores), color='red', linestyle='--',
                        label=f'Mean: {np.mean(f1_scores):.4f}')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('F1 Macro')
        axes[1].set_title(f'{ds} — F1 Macro per Fold ({model_key})', fontweight='bold')
        axes[1].legend()
        for bar, val in zip(bars, f1_scores):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(figs_dir / f'{tag}_cv_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        cv_mean = float(cv_df['f1_macro'].mean())
        cv_std = float(cv_df['f1_macro'].std())
    else:
        cv_mean, cv_std = float('nan'), float('nan')

    # ── Final model ──────────────────────────────────────────────────────────
    print(f'\nTraining FINAL model on train+val...')
    parts = [df_train]
    if len(df_val):
        parts.append(df_val)
    df_trainval = pd.concat(parts, ignore_index=True)
    ds_trainval = TweetDataset(df_trainval[text_col].values,
                               encode_labels(df_trainval['label'], label_map), tokenizer)
    ds_test = TweetDataset(df_test[text_col].values,
                           encode_labels(df_test['label'], label_map), tokenizer)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=num_labels, id2label=id2label, label2id=label_map,
        ignore_mismatched_sizes=True,
    )
    final_dir = MODELS_DIR / f'{tag}_final'
    final_trainer = Trainer(
        model=final_model,
        args=get_training_args(final_dir, do_eval=False),
        train_dataset=ds_trainval,
        compute_metrics=compute_metrics_fn,
    )
    final_trainer.train()

    test_out = final_trainer.predict(ds_test)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()
    test_preds = np.argmax(test_probs, axis=-1)
    test_true = encode_labels(df_test['label'], label_map)

    metrics = full_report(test_true, test_preds, pos_id, pos_name, f'{ds} TEST')
    print(classification_report(test_true, test_preds, target_names=label_names, zero_division=0))

    # Save model
    model_save = MODELS_DIR / f'{tag}_final'
    final_model.save_pretrained(str(model_save))
    tokenizer.save_pretrained(str(model_save))
    print(f'Model saved: {model_save}')

    # Summary
    summary = pd.DataFrame([{
        'dataset': ds,
        'model': hf_name,
        'model_key': model_key,
        'cv_f1_macro_mean': None if np.isnan(cv_mean) else round(cv_mean, 6),
        'cv_f1_macro_std': None if np.isnan(cv_std) else round(cv_std, 6),
        'test_accuracy': round(metrics['accuracy'], 6),
        'test_f1_macro': round(metrics['f1_macro'], 6),
        'test_f1_weighted': round(metrics['f1_weighted'], 6),
        'test_precision': round(metrics['precision_macro'], 6),
        'test_recall': round(metrics['recall_macro'], 6),
        f'test_f1_{pos_name}': round(metrics['f1_pos'], 6),
    }])
    summary.to_csv(summary_path, index=False)
    # Also write legacy-style name for roberta so existing comparison scripts work
    if model_key == 'roberta':
        summary.to_csv(PREDS_DIR / f'{ds}_roberta_summary.csv', index=False)

    # Predictions
    pred_cols = {
        'cleaned_tweet': df_test[text_col].values,
        'label': df_test['label'].values,
        'true_label_int': test_true,
        'pred_label_int': test_preds,
        'pred_label': [id2label[p] for p in test_preds],
    }
    for idx, name in id2label.items():
        pred_cols[f'prob_{name}'] = test_probs[:, idx]
    pred_cols['correct'] = test_true == test_preds
    pred_df = pd.DataFrame(pred_cols)
    pred_df.to_csv(PREDS_DIR / f'{tag}_test_predictions.csv', index=False)
    if model_key == 'roberta':
        pred_df.to_csv(PREDS_DIR / f'{ds}_roberta_test_predictions.csv', index=False)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_test = confusion_matrix(test_true, test_preds, labels=list(range(num_labels)))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f'{ds} — {model_key} Test CM', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(figs_dir / f'{tag}_test_cm.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nDone {tag}: test F1 macro={metrics["f1_macro"]:.4f}')
    del final_model, final_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True, choices=list(MODELS))
    parser.add_argument('--skip-if-done', action='store_true')
    args = parser.parse_args()
    train_one(args.dataset, args.model, skip_if_done=args.skip_if_done)


if __name__ == '__main__':
    main()
