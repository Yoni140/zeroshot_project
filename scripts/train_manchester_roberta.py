"""
Train RoBERTa on Manchester dataset (5-fold CV + final model).
Run from project root: python scripts/train_manchester_roberta.py

Improvements over original:
  1. Pre-tokenization caching  — tokenize gold_standard once, index per fold
  2. Class weighting           — WeightedTrainer with balanced CrossEntropyLoss
  3. ROC-AUC + PR-AUC          — computed and plotted on test set
  4. Per-fold metrics heatmap  — seaborn heatmap of all metrics × folds
  5. Dropped-rows logging      — logged dropna with counts/reasons
"""
import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    RobertaTokenizerFast, RobertaForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATASET     = 'manchester'
MODEL_NAME  = 'roberta-base'
SEED        = 42
N_FOLDS     = 5
MAX_LEN     = 128
BATCH_SIZE  = 16
NUM_EPOCHS  = 4

LABEL_MAP   = {'reliable': 0, 'misinformation': 1}
ID2LABEL    = {0: 'reliable', 1: 'misinformation'}
LABEL_NAMES = ['reliable', 'misinformation']
POS_LABEL   = 'misinformation'
TEXT_COL    = 'cleaned_tweet'
LABEL_COL   = 'label'
NUM_LABELS  = 2

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'gold_standard'
PREDS_DIR  = ROOT / 'results' / 'predictions'
MODELS_DIR = ROOT / 'results' / 'models'
FIGS_DIR   = ROOT / 'results' / 'figures' / 'manchester'
for d in [PREDS_DIR, MODELS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# Improvement 5: Logged dropna
# ─────────────────────────────────────────────────────────────────────────────
def drop_na_logged(df, subset, name):
    """Drop rows with NaN in `subset` columns, printing what was removed."""
    before = len(df)
    for col in subset:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  [{name}] Dropping {n_nan:,} rows where '{col}' is NaN")
    df = df.dropna(subset=subset).copy()
    after = len(df)
    dropped = before - after
    if dropped == 0:
        print(f"  [{name}] No rows dropped ({before:,} rows all valid)")
    else:
        print(f"  [{name}] Dropped {dropped:,} rows ({dropped/before*100:.1f}%) -> {after:,} remaining")
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df_train = pd.read_csv(DATA_DIR / 'manchester_train.csv')
df_val   = pd.read_csv(DATA_DIR / 'manchester_val.csv')
df_test  = pd.read_csv(DATA_DIR / 'manchester_test.csv')
df_gold  = pd.read_csv(DATA_DIR / 'manchester_gold_standard.csv')

print("=== Dropped-Rows Audit ===")
df_train = drop_na_logged(df_train, [TEXT_COL, LABEL_COL], 'train')
df_val   = drop_na_logged(df_val,   [TEXT_COL, LABEL_COL], 'val')
df_test  = drop_na_logged(df_test,  [TEXT_COL, LABEL_COL], 'test')
df_gold  = drop_na_logged(df_gold,  [TEXT_COL, LABEL_COL], 'gold')

print(f"\nTrain: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}  Gold: {len(df_gold):,}")
print(f"\nTrain label distribution:\n{df_train[LABEL_COL].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

def encode_labels(df):
    return np.array([LABEL_MAP[l] for l in df[LABEL_COL]])

# ─────────────────────────────────────────────────────────────────────────────
# Improvement 1: CachedTweetDataset — indexes into pre-built token tensors
# ─────────────────────────────────────────────────────────────────────────────
class CachedTweetDataset(Dataset):
    """Indexes into pre-computed tokenizer output — zero re-tokenization cost."""
    def __init__(self, encodings, labels, indices):
        self.encodings = encodings
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.indices   = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        idx  = self.indices[pos]
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[pos]
        return item

class SimpleTweetDataset(Dataset):
    """For the final model where we tokenize train+val + test separately."""
    def __init__(self, texts, labels, tok):
        enc = tok(list(texts), truncation=True, padding='max_length',
                  max_length=MAX_LEN, return_tensors='pt')
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'input_ids': self.input_ids[i],
                'attention_mask': self.attention_mask[i],
                'labels': self.labels[i]}

# ─────────────────────────────────────────────────────────────────────────────
# Improvement 2: WeightedTrainer — class-weighted CrossEntropyLoss
# ─────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(DEVICE)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy':    accuracy_score(labels, preds),
        'f1_macro':    f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }

def full_report(true, pred, tag, fold=None):
    tag_str = f"{tag} Fold {fold}" if fold else tag
    acc  = accuracy_score(true, pred)
    f1m  = f1_score(true, pred, average='macro')
    f1w  = f1_score(true, pred, average='weighted')
    prec = precision_score(true, pred, average='macro', zero_division=0)
    rec  = recall_score(true, pred, average='macro', zero_division=0)
    pos  = LABEL_MAP[POS_LABEL]
    f1p  = f1_score(true, pred, average='binary', pos_label=pos, zero_division=0)
    print(f"[{tag_str}] Acc={acc:.4f}  F1m={f1m:.4f}  F1_{POS_LABEL}={f1p:.4f}")
    return {'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
            'precision_macro': prec, 'recall_macro': rec, 'f1_pos': f1p}

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type='linear',
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro', greater_is_better=True,
        logging_steps=100, seed=SEED,
        fp16=torch.cuda.is_available(), report_to='none', dataloader_num_workers=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Improvement 1: Pre-tokenize ALL gold texts once before the fold loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Pre-tokenizing {len(df_gold):,} gold texts (once for all {N_FOLDS} folds) ===")
all_texts  = df_gold[TEXT_COL].values
all_labels = encode_labels(df_gold)

cached_encodings = tokenizer(
    list(all_texts),
    truncation=True,
    padding='max_length',
    max_length=MAX_LEN,
    return_tensors='pt',
)
print("Pre-tokenization complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold CV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Starting {N_FOLDS}-Fold CV on {len(df_gold):,} gold samples ===")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cv_results    = []
all_oof_preds = np.zeros(len(df_gold), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), 1):
    print(f"\n{'='*50}\n FOLD {fold}/{N_FOLDS}  train={len(train_idx):,}  val={len(val_idx):,}\n{'='*50}")

    # Datasets from cached encodings (Improvement 1)
    fold_train = CachedTweetDataset(cached_encodings, all_labels[train_idx], train_idx)
    fold_val   = CachedTweetDataset(cached_encodings, all_labels[val_idx],   val_idx)

    # Per-fold class weights from training split (Improvement 2)
    fold_cw = compute_class_weight(
        'balanced',
        classes=np.array(sorted(LABEL_MAP.values())),
        y=all_labels[train_idx],
    )
    fold_weights = torch.tensor(fold_cw, dtype=torch.float)
    print(f"  Class weights: { {ID2LABEL[c]: round(w, 4) for c, w in zip(sorted(LABEL_MAP.values()), fold_cw)} }")

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL_MAP)

    trainer = WeightedTrainer(
        class_weights=fold_weights,
        model=model,
        args=get_training_args(MODELS_DIR / f'{DATASET}_fold{fold}'),
        train_dataset=fold_train,
        eval_dataset=fold_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    preds_out  = trainer.predict(fold_val)
    fold_preds = np.argmax(preds_out.predictions, axis=-1)
    all_oof_preds[val_idx] = fold_preds

    metrics = full_report(all_labels[val_idx], fold_preds, DATASET.upper(), fold=fold)
    metrics['fold'] = fold
    cv_results.append(metrics)

    del model, trainer, fold_train, fold_val
    if torch.cuda.is_available(): torch.cuda.empty_cache()

cv_df = pd.DataFrame(cv_results).set_index('fold')
print(f"\nCV Summary:\n{cv_df.round(4).to_string()}")
cv_df.to_csv(PREDS_DIR / f'{DATASET}_cv_results.csv')

# ─────────────────────────────────────────────────────────────────────────────
# CV figures: OOF confusion matrix + F1 bar + per-fold heatmap (Improvement 4)
# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: OOF CM + F1 bar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(all_labels, all_oof_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
axes[0].set_title('Manchester - OOF Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

f1_scores = cv_df['f1_macro'].tolist()
bars = axes[1].bar(cv_df.index.tolist(), f1_scores, color='steelblue', edgecolor='black')
axes[1].axhline(np.mean(f1_scores), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {np.mean(f1_scores):.4f}')
axes[1].set_xlabel('Fold'); axes[1].set_ylabel('F1 Macro')
axes[1].set_title('Manchester - F1 Macro per Fold', fontweight='bold')
axes[1].set_ylim([max(0, min(f1_scores)-0.05), 1.0]); axes[1].legend()
for bar, val in zip(bars, f1_scores):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_cv_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Improvement 4 — per-fold metrics heatmap
heatmap_cols   = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'f1_pos']
heatmap_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall', f'F1 {POS_LABEL}']
hm_data = cv_df[heatmap_cols].copy()
hm_data.columns = heatmap_labels
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(hm_data.T, annot=True, fmt='.4f', cmap='YlGnBu',
            vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Score'})
ax.set_title('MANCHESTER - Per-Fold Metrics Heatmap (5-Fold CV)', fontweight='bold', pad=12)
ax.set_xlabel('Fold'); ax.set_ylabel('Metric')
ax.set_xticklabels([f'Fold {i}' for i in cv_df.index])
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_cv_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("CV figures saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Final model — train on full train+val, evaluate on test
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Training FINAL model on train+val ({len(df_train)+len(df_val):,} samples) ===")
df_trainval = pd.concat([df_train, df_val], ignore_index=True)
df_trainval = drop_na_logged(df_trainval, [TEXT_COL, LABEL_COL], 'trainval')

train_labels_arr = encode_labels(df_trainval)
test_labels_arr  = encode_labels(df_test)

# Tokenize train+val and test separately (they are different from gold)
ds_trainval = SimpleTweetDataset(df_trainval[TEXT_COL].values, train_labels_arr, tokenizer)
ds_test     = SimpleTweetDataset(df_test[TEXT_COL].values,     test_labels_arr,  tokenizer)

# Class weights for final training
final_cw      = compute_class_weight('balanced',
                                     classes=np.array(sorted(LABEL_MAP.values())),
                                     y=train_labels_arr)
final_weights = torch.tensor(final_cw, dtype=torch.float)
print(f"Final class weights: { {ID2LABEL[c]: round(w, 4) for c, w in zip(sorted(LABEL_MAP.values()), final_cw)} }")

final_model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL_MAP)

# Use a small eval split from trainval for early stopping (no leakage)
tv_texts_np, es_texts_np, tv_lab, es_lab = train_test_split(
    df_trainval[TEXT_COL].values, train_labels_arr,
    test_size=0.1, stratify=train_labels_arr, random_state=SEED,
)
ds_tv = SimpleTweetDataset(tv_texts_np, tv_lab, tokenizer)
ds_es = SimpleTweetDataset(es_texts_np, es_lab, tokenizer)

final_args = TrainingArguments(
    output_dir=str(MODELS_DIR / f'{DATASET}_final'),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE*2,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy='epoch', save_strategy='epoch',
    load_best_model_at_end=True, metric_for_best_model='f1_macro', greater_is_better=True,
    logging_steps=100, seed=SEED, fp16=torch.cuda.is_available(),
    report_to='none', dataloader_num_workers=0,
)
final_trainer = WeightedTrainer(
    class_weights=final_weights,
    model=final_model,
    args=final_args,
    train_dataset=ds_tv,
    eval_dataset=ds_es,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
final_trainer.train()

# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
test_out   = final_trainer.predict(ds_test)
test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()
test_preds = np.argmax(test_probs, axis=-1)

acc  = accuracy_score(test_labels_arr, test_preds)
f1m  = f1_score(test_labels_arr, test_preds, average='macro')
f1w  = f1_score(test_labels_arr, test_preds, average='weighted')
prec = precision_score(test_labels_arr, test_preds, average='macro', zero_division=0)
rec  = recall_score(test_labels_arr, test_preds, average='macro', zero_division=0)
f1p  = f1_score(test_labels_arr, test_preds, average='binary',
                pos_label=LABEL_MAP[POS_LABEL], zero_division=0)

# Improvement 3: ROC-AUC and PR-AUC
pos_probs = test_probs[:, LABEL_MAP[POS_LABEL]]
roc_auc   = roc_auc_score(test_labels_arr, pos_probs)
pr_auc    = average_precision_score(test_labels_arr, pos_probs)

print(f"\nTest: Acc={acc:.4f}  F1m={f1m:.4f}  F1_{POS_LABEL}={f1p:.4f}")
print(f"      ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
print(classification_report(test_labels_arr, test_preds, target_names=LABEL_NAMES))

# ─────────────────────────────────────────────────────────────────────────────
# Save model + predictions
# ─────────────────────────────────────────────────────────────────────────────
roberta_final_dir = MODELS_DIR / f'{DATASET}_roberta_final'
final_model.save_pretrained(str(roberta_final_dir))
tokenizer.save_pretrained(str(roberta_final_dir))
print(f"Model saved: {roberta_final_dir}")

pd.DataFrame([{
    'dataset': DATASET, 'model': 'roberta-base',
    'cv_f1_macro_mean': round(cv_df['f1_macro'].mean(), 6),
    'cv_f1_macro_std':  round(cv_df['f1_macro'].std(),  6),
    'test_accuracy':    round(acc,     6),
    'test_f1_macro':    round(f1m,     6),
    'test_f1_weighted': round(f1w,     6),
    'test_precision':   round(prec,    6),
    'test_recall':      round(rec,     6),
    f'test_f1_{POS_LABEL}': round(f1p, 6),
    'test_roc_auc':     round(roc_auc, 6),
    'test_pr_auc':      round(pr_auc,  6),
}]).to_csv(PREDS_DIR / f'{DATASET}_roberta_summary.csv', index=False)

pred_df = pd.DataFrame({
    'cleaned_tweet':      df_test[TEXT_COL].values,
    'label':              df_test[LABEL_COL].values,
    'true_label_int':     test_labels_arr,
    'pred_label_int':     test_preds,
    'pred_label':         [ID2LABEL[p] for p in test_preds],
    'prob_reliable':      test_probs[:, 0],
    'prob_misinformation': test_probs[:, 1],
    'correct':            test_labels_arr == test_preds,
})
pred_df.to_csv(PREDS_DIR / f'{DATASET}_roberta_test_predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Figures: test confusion matrix + ROC/PR curves (Improvement 3)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(test_labels_arr, test_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
ax.set_title('Manchester - RoBERTa Test Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_test_cm.png', dpi=150, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(test_labels_arr, pos_probs)
prec_c, rec_c, _ = precision_recall_curve(test_labels_arr, pos_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
axes[0].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random')
axes[0].set_xlim([0.0, 1.0]); axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('MANCHESTER - ROC Curve (Test Set)', fontweight='bold')
axes[0].legend(loc='lower right'); axes[0].grid(alpha=0.3)

baseline = test_labels_arr.sum() / len(test_labels_arr)
axes[1].plot(rec_c, prec_c, color='darkorange', lw=2, label=f'PR AUC = {pr_auc:.4f}')
axes[1].axhline(baseline, color='grey', linestyle='--', lw=1, label=f'Baseline = {baseline:.3f}')
axes[1].set_xlim([0.0, 1.0]); axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('MANCHESTER - Precision-Recall Curve (Test Set)', fontweight='bold')
axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Test figures saved.")

print(f"\n{'='*60}")
print(f" Manchester RoBERTa complete!")
print(f" CV F1:   {cv_df['f1_macro'].mean():.4f} +/- {cv_df['f1_macro'].std():.4f}")
print(f" Test F1 Macro:          {f1m:.4f}")
print(f" Test F1 Misinformation: {f1p:.4f}")
print(f" Test ROC-AUC:           {roc_auc:.4f}")
print(f" Test PR-AUC:            {pr_auc:.4f}")
print(f"{'='*60}")
