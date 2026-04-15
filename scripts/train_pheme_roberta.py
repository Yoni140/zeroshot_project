"""
Train RoBERTa on PHEME dataset (5-fold CV + final model on full train set).
Mirrors notebooks/models/07_RoBERTa_Finetuning.ipynb exactly.
Run from project root: python scripts/train_pheme_roberta.py
"""
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
DATASET     = 'pheme'
MODEL_NAME  = 'roberta-base'
SEED        = 42
N_FOLDS     = 5
MAX_LEN     = 128
BATCH_SIZE  = 16
NUM_EPOCHS  = 4

LABEL_MAP   = {'not_rumour': 0, 'rumour': 1}
ID2LABEL    = {0: 'not_rumour', 1: 'rumour'}
LABEL_NAMES = ['not_rumour', 'rumour']
POS_LABEL   = 'rumour'
TEXT_COL    = 'cleaned_tweet'
LABEL_COL   = 'label'
NUM_LABELS  = 2

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / 'data' / 'gold_standard'
PREDS_DIR = ROOT / 'results' / 'predictions'
MODELS_DIR= ROOT / 'results' / 'models'
FIGS_DIR  = ROOT / 'results' / 'figures' / 'pheme'
for d in [PREDS_DIR, MODELS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Data ─────────────────────────────────────────────────────────────────────
df_train = pd.read_csv(DATA_DIR / 'pheme_train.csv')
df_val   = pd.read_csv(DATA_DIR / 'pheme_val.csv')
df_test  = pd.read_csv(DATA_DIR / 'pheme_test.csv')
df_gold  = pd.read_csv(DATA_DIR / 'pheme_gold_standard.csv')

print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}  Gold: {len(df_gold):,}")

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

def encode_labels(df):
    return np.array([LABEL_MAP[l] for l in df[LABEL_COL]])

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok):
        enc = tok(list(texts), truncation=True, padding='max_length',
                  max_length=MAX_LEN, return_tensors='pt')
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels         = torch.tensor(labels, dtype=torch.long)
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return {'input_ids': self.input_ids[i],
                'attention_mask': self.attention_mask[i],
                'labels': self.labels[i]}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy':  accuracy_score(labels, preds),
        'f1_macro':  f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }

def full_report(true, pred, label_names, tag, fold=None):
    tag_str = f"{tag} Fold {fold}" if fold else tag
    acc  = accuracy_score(true, pred)
    f1m  = f1_score(true, pred, average='macro')
    f1w  = f1_score(true, pred, average='weighted')
    prec = precision_score(true, pred, average='macro', zero_division=0)
    rec  = recall_score(true, pred, average='macro', zero_division=0)
    pos  = LABEL_MAP[POS_LABEL]
    f1p  = f1_score(true, pred, average='binary', pos_label=pos, zero_division=0)
    print(f"\n[{tag_str}] Acc={acc:.4f}  F1m={f1m:.4f}  F1w={f1w:.4f}  F1_{POS_LABEL}={f1p:.4f}")
    return {'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
            'precision_macro': prec, 'recall_macro': rec, 'f1_pos': f1p}

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        logging_steps=100,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to='none',
        dataloader_num_workers=0,
    )

# ── 5-Fold CV ────────────────────────────────────────────────────────────────
print(f"\nStarting {N_FOLDS}-Fold CV on {len(df_gold):,} gold samples...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_texts  = df_gold[TEXT_COL].values
all_labels = encode_labels(df_gold)

cv_results     = []
all_oof_preds  = np.zeros(len(df_gold), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), 1):
    print(f"\n{'='*50}\n FOLD {fold}/{N_FOLDS}  train={len(train_idx):,}  val={len(val_idx):,}\n{'='*50}")
    fold_train = TweetDataset(all_texts[train_idx], all_labels[train_idx], tokenizer)
    fold_val   = TweetDataset(all_texts[val_idx],   all_labels[val_idx],   tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL_MAP)

    fold_dir = MODELS_DIR / f'{DATASET}_fold{fold}'
    trainer  = Trainer(
        model=model,
        args=get_training_args(fold_dir),
        train_dataset=fold_train,
        eval_dataset=fold_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    preds_out = trainer.predict(fold_val)
    fold_preds = np.argmax(preds_out.predictions, axis=-1)
    all_oof_preds[val_idx] = fold_preds

    metrics = full_report(all_labels[val_idx], fold_preds, LABEL_NAMES, DATASET.upper(), fold=fold)
    metrics['fold'] = fold
    cv_results.append(metrics)

    del model, trainer, fold_train, fold_val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── CV Summary ───────────────────────────────────────────────────────────────
cv_df = pd.DataFrame(cv_results).set_index('fold')
print(f"\n{'='*60}\n {DATASET.upper()} CV Summary\n{'='*60}")
print(cv_df.round(4).to_string())
print("\nMean +/- Std:")
print(cv_df.agg(['mean', 'std']).round(4).to_string())

cv_df.to_csv(PREDS_DIR / f'{DATASET}_cv_results.csv')
print(f"\nSaved: results/predictions/{DATASET}_cv_results.csv")

# ── CV Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(all_labels, all_oof_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
axes[0].set_title(f'PHEME - OOF Confusion Matrix (5-Fold CV)', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

f1_scores = cv_df['f1_macro'].tolist()
bars = axes[1].bar(cv_df.index.tolist(), f1_scores, color='steelblue', edgecolor='black')
axes[1].axhline(np.mean(f1_scores), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {np.mean(f1_scores):.4f}')
axes[1].set_xlabel('Fold'); axes[1].set_ylabel('F1 Macro')
axes[1].set_title('PHEME - F1 Macro per Fold', fontweight='bold')
axes[1].set_ylim([max(0, min(f1_scores) - 0.05), 1.0]); axes[1].legend()
for bar, val in zip(bars, f1_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_cv_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Final Model (train on train+val, evaluate on test) ───────────────────────
print(f"\n{'='*60}\n Training FINAL model on train+val set\n{'='*60}")
df_trainval = pd.concat([df_train, df_val], ignore_index=True)
ds_trainval = TweetDataset(df_trainval[TEXT_COL].values, encode_labels(df_trainval), tokenizer)
ds_test     = TweetDataset(df_test[TEXT_COL].values,     encode_labels(df_test),     tokenizer)

final_model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL_MAP)

final_dir = MODELS_DIR / f'{DATASET}_final'
final_args = TrainingArguments(
    output_dir=str(final_dir),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type='linear',
    eval_strategy='no',
    save_strategy='no',
    logging_steps=100,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to='none',
    dataloader_num_workers=0,
)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=ds_trainval,
    compute_metrics=compute_metrics,
)
final_trainer.train()

# ── Test Evaluation ──────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_out   = final_trainer.predict(ds_test)
test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()
test_preds = np.argmax(test_probs, axis=-1)
test_true  = encode_labels(df_test)

acc  = accuracy_score(test_true, test_preds)
f1m  = f1_score(test_true, test_preds, average='macro')
f1w  = f1_score(test_true, test_preds, average='weighted')
prec = precision_score(test_true, test_preds, average='macro', zero_division=0)
rec  = recall_score(test_true, test_preds, average='macro', zero_division=0)
pos  = LABEL_MAP[POS_LABEL]
f1p  = f1_score(test_true, test_preds, average='binary', pos_label=pos, zero_division=0)

print(f"\nTest Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Macro : {f1m:.4f}")
print(f"  F1 Weight: {f1w:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Rumour: {f1p:.4f}")
print("\nClassification Report:")
print(classification_report(test_true, test_preds, target_names=LABEL_NAMES))

# ── Save roberta_final model ─────────────────────────────────────────────────
roberta_final_dir = MODELS_DIR / f'{DATASET}_roberta_final'
final_model.save_pretrained(str(roberta_final_dir))
tokenizer.save_pretrained(str(roberta_final_dir))
print(f"Model saved: results/models/{DATASET}_roberta_final/")

# ── Save summary CSV ─────────────────────────────────────────────────────────
cv_mean = cv_df['f1_macro'].mean()
cv_std  = cv_df['f1_macro'].std()

summary = pd.DataFrame([{
    'dataset': DATASET,
    'model': 'roberta-base',
    'cv_f1_macro_mean': round(cv_mean, 6),
    'cv_f1_macro_std':  round(cv_std,  6),
    'test_accuracy':           round(acc,  6),
    'test_f1_macro':           round(f1m,  6),
    'test_f1_weighted':        round(f1w,  6),
    'test_precision':          round(prec, 6),
    'test_recall':             round(rec,  6),
    'test_f1_misinformation':  round(f1p,  6),
}])
summary.to_csv(PREDS_DIR / f'{DATASET}_roberta_summary.csv', index=False)
print(f"Summary saved: results/predictions/{DATASET}_roberta_summary.csv")

# ── Save test predictions CSV ────────────────────────────────────────────────
pred_label_names = [ID2LABEL[p] for p in test_preds]
true_label_names = df_test[LABEL_COL].values
test_pred_df = pd.DataFrame({
    'cleaned_tweet':  df_test[TEXT_COL].values,
    'label':          true_label_names,
    'true_label_int': test_true,
    'pred_label_int': test_preds,
    'pred_label':     pred_label_names,
    'prob_not_rumour': test_probs[:, 0],
    'prob_rumour':     test_probs[:, 1],
    'correct':        test_true == test_preds,
})
test_pred_df.to_csv(PREDS_DIR / f'{DATASET}_roberta_test_predictions.csv', index=False)
print(f"Predictions saved: results/predictions/{DATASET}_roberta_test_predictions.csv")

# ── Test Confusion Matrix figure ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm_test = confusion_matrix(test_true, test_preds)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
ax.set_title('PHEME - RoBERTa Test Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_roberta_test_cm.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print(f" PHEME RoBERTa training complete!")
print(f" CV F1 Macro: {cv_mean:.4f} +/- {cv_std:.4f}")
print(f" Test F1 Macro: {f1m:.4f}  |  Test F1 Rumour: {f1p:.4f}")
print(f"{'='*60}")
