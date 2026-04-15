"""
plot_training_curves.py
Generate training curve plots from saved trainer_state.json files.
Produces per-dataset figures: loss curves, F1 per epoch, CV summary, ROC/PR curves.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
MODELS    = ROOT / 'results' / 'models'
PREDS     = ROOT / 'results' / 'predictions'
FIGS_BASE = ROOT / 'results' / 'figures'

DATASETS = ['manchester', 'monkeypox', 'pheme']

LABEL_CFG = {
    'manchester': {'pos_label': 'misinformation', 'label_names': ['reliable', 'misinformation']},
    'monkeypox':  {'pos_label': 'misinformation', 'label_names': ['reliable', 'misinformation']},
    'pheme':      {'pos_label': 'rumour',          'label_names': ['not_rumour', 'rumour']},
}

# ── Helper: load all fold trainer_state logs ─────────────────────────────────
def load_fold_logs(dataset):
    """Return list of log_history lists (one per fold)."""
    fold_logs = []
    for fold_dir in sorted(MODELS.glob(f'{dataset}_fold*')):
        # pick the last checkpoint (highest step)
        checkpoints = sorted(fold_dir.glob('checkpoint-*'),
                             key=lambda p: int(p.name.split('-')[1]))
        if not checkpoints:
            continue
        state_path = checkpoints[-1] / 'trainer_state.json'
        if not state_path.exists():
            continue
        with open(state_path) as f:
            state = json.load(f)
        fold_logs.append(state['log_history'])
    return fold_logs

def load_final_logs(dataset):
    """Return log_history for the final model."""
    final_dir = MODELS / f'{dataset}_final'
    if not final_dir.exists():
        return None
    checkpoints = sorted(final_dir.glob('checkpoint-*'),
                         key=lambda p: int(p.name.split('-')[1]))
    if not checkpoints:
        return None
    state_path = checkpoints[-1] / 'trainer_state.json'
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    return state['log_history']

def parse_log(log_history):
    """Split log into train steps and eval epochs."""
    train = [e for e in log_history if 'loss' in e and 'eval_loss' not in e]
    evalu = [e for e in log_history if 'eval_loss' in e]
    return train, evalu


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training Loss Curve  (all folds + final, per dataset)
# ════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(dataset, fold_logs, final_log, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset.title()} — Training Loss Curves', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10.colors

    for ax_idx, (ax, key, ylabel) in enumerate(zip(
            axes,
            ['loss', 'eval_loss'],
            ['Train Loss (per step)', 'Validation Loss (per epoch)'])):

        for fi, log in enumerate(fold_logs):
            train, evalu = parse_log(log)
            if key == 'loss':
                xs = [e['step'] for e in train]
                ys = [e['loss'] for e in train]
            else:
                xs = [e['epoch'] for e in evalu]
                ys = [e['eval_loss'] for e in evalu]
            ax.plot(xs, ys, alpha=0.5, linewidth=1.5,
                    color=colors[fi], label=f'Fold {fi+1}')

        if final_log:
            train_f, eval_f = parse_log(final_log)
            if key == 'loss':
                xs = [e['step'] for e in train_f]
                ys = [e['loss'] for e in train_f]
            else:
                xs = [e['epoch'] for e in eval_f]
                ys = [e['eval_loss'] for e in eval_f]
            ax.plot(xs, ys, linewidth=2.5, color='black',
                    linestyle='--', label='Final Model')

        ax.set_xlabel('Step' if key == 'loss' else 'Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f'{dataset}_roberta_loss_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — F1 Macro per Epoch (all folds + final)
# ════════════════════════════════════════════════════════════════════════════
def plot_f1_curves(dataset, fold_logs, final_log, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset.title()} — Validation Metrics per Epoch', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10.colors

    for fi, log in enumerate(fold_logs):
        _, evalu = parse_log(log)
        epochs = [e['epoch'] for e in evalu]
        f1s    = [e.get('eval_f1_macro', 0) for e in evalu]
        accs   = [e.get('eval_accuracy', 0) for e in evalu]
        axes[0].plot(epochs, f1s,  alpha=0.5, linewidth=1.5, color=colors[fi], label=f'Fold {fi+1}')
        axes[1].plot(epochs, accs, alpha=0.5, linewidth=1.5, color=colors[fi], label=f'Fold {fi+1}')

    if final_log:
        _, eval_f = parse_log(final_log)
        epochs = [e['epoch'] for e in eval_f]
        f1s    = [e.get('eval_f1_macro', 0) for e in eval_f]
        accs   = [e.get('eval_accuracy', 0) for e in eval_f]
        axes[0].plot(epochs, f1s,  linewidth=2.5, color='black', linestyle='--', label='Final Model')
        axes[1].plot(epochs, accs, linewidth=2.5, color='black', linestyle='--', label='Final Model')

    for ax, title in zip(axes, ['Validation F1 Macro', 'Validation Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = out_dir / f'{dataset}_roberta_f1_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — CV Fold Summary Bar Chart (F1 per fold + mean±std)
# ════════════════════════════════════════════════════════════════════════════
def plot_cv_summary(dataset, fold_logs, out_dir):
    fold_f1s = []
    for log in fold_logs:
        _, evalu = parse_log(log)
        best = max(evalu, key=lambda e: e.get('eval_f1_macro', 0))
        fold_f1s.append(best['eval_f1_macro'])

    if not fold_f1s:
        print(f'  No fold data for {dataset}')
        return

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    labels  = [f'Fold {i+1}' for i in range(len(fold_f1s))] + ['Mean']
    values  = fold_f1s + [mean_f1]
    errors  = [0] * len(fold_f1s) + [std_f1]
    bar_colors = ['steelblue'] * len(fold_f1s) + ['darkblue']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax.errorbar(len(fold_f1s), mean_f1, yerr=std_f1,
                fmt='none', color='black', capsize=6, linewidth=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(max(0, min(values) - 0.05), 1.02)
    ax.set_ylabel('Best Val F1 Macro')
    ax.set_title(f'{dataset.title()} — 5-Fold CV Results\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}',
                 fontweight='bold')
    ax.axhline(mean_f1, color='red', linestyle='--', alpha=0.5, label=f'Mean = {mean_f1:.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f'{dataset}_roberta_cv_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — ROC + PR Curves (from test predictions)
# ════════════════════════════════════════════════════════════════════════════
def plot_roc_pr(dataset, out_dir):
    pred_path = PREDS / f'{dataset}_roberta_test_predictions.csv'
    if not pred_path.exists():
        print(f'  No test predictions for {dataset}, skipping ROC/PR')
        return

    df = pd.read_csv(pred_path)
    cfg = LABEL_CFG[dataset]
    pos = cfg['pos_label']

    # True label column — could be 'label' or 'true_label'
    if 'label' in df.columns:
        label_col = 'label'
    elif 'true_label' in df.columns:
        label_col = 'true_label'
    else:
        label_col = df.columns[0]

    # Probability column for the positive class
    prob_col = f'prob_{pos}'
    if prob_col not in df.columns:
        # fallback: search for any prob column
        for col in df.columns:
            if 'prob' in col.lower() or 'score' in col.lower():
                prob_col = col
                break
        else:
            prob_col = None

    y_true = (df[label_col] == pos).astype(int).values

    if prob_col and prob_col in df.columns:
        y_score = df[prob_col].values
    else:
        # fallback: use binary prediction
        pred_col = 'pred_label' if 'pred_label' in df.columns else df.columns[1]
        y_score = (df[pred_col] == pos).astype(int).values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap            = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset.title()} — RoBERTa Test Set ROC & PR Curves',
                 fontsize=13, fontweight='bold')

    # ROC
    axes[0].plot(fpr, tpr, color='steelblue', linewidth=2,
                 label=f'RoBERTa (AUC = {roc_auc:.4f})')
    axes[0].plot([0,1],[0,1], 'k--', alpha=0.4)
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # PR
    axes[1].plot(rec, prec, color='darkorange', linewidth=2,
                 label=f'RoBERTa (AP = {ap:.4f})')
    axes[1].axhline(y_true.mean(), color='gray', linestyle='--', alpha=0.5,
                    label=f'Baseline (AP = {y_true.mean():.4f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f'{dataset}_roberta_roc_pr.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Learning Rate Schedule (from final model)
# ════════════════════════════════════════════════════════════════════════════
def plot_lr_schedule(dataset, final_log, out_dir):
    if not final_log:
        return
    train, _ = parse_log(final_log)
    steps = [e['step'] for e in train if 'learning_rate' in e]
    lrs   = [e['learning_rate'] for e in train if 'learning_rate' in e]
    if not steps:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs, color='purple', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{dataset.title()} — Learning Rate Schedule (Warmup + Linear Decay)',
                 fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'{dataset}_roberta_lr_schedule.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    for dataset in DATASETS:
        out_dir = FIGS_BASE / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n{"="*55}')
        print(f'  {dataset.upper()}')
        print(f'{"="*55}')

        fold_logs  = load_fold_logs(dataset)
        final_log  = load_final_logs(dataset)

        if not fold_logs:
            print(f'  No fold logs found for {dataset}, skipping')
            continue

        print(f'  Found {len(fold_logs)} folds')

        plot_loss_curves(dataset, fold_logs, final_log, out_dir)
        plot_f1_curves(dataset,  fold_logs, final_log, out_dir)
        plot_cv_summary(dataset, fold_logs, out_dir)
        plot_roc_pr(dataset, out_dir)
        plot_lr_schedule(dataset, final_log, out_dir)

    print(f'\n{"="*55}')
    print('  All training plots generated!')
    print(f'{"="*55}')

if __name__ == '__main__':
    main()
