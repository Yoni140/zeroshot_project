"""
Standalone comparison script — replicates notebook 09 logic.
Run from project root: python scripts/run_comparison.py
"""
import pandas as pd
import glob
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

warnings.filterwarnings('ignore')

os.makedirs('results/figures/comparison', exist_ok=True)

# ── Load summaries ────────────────────────────────────────────────────────────
dfs = []
for f in sorted(glob.glob('results/predictions/*_roberta_summary.csv')):
    df = pd.read_csv(f)
    df['model_type'] = 'RoBERTa Fine-Tuned'
    df['dataset'] = os.path.basename(f).replace('_roberta_summary.csv', '')
    dfs.append(df)
for f in sorted(glob.glob('results/predictions/*_zeroshot_summary.csv')):
    df = pd.read_csv(f)
    df['model_type'] = 'Zero-Shot LLM'
    df['dataset'] = os.path.basename(f).replace('_zeroshot_summary.csv', '')
    dfs.append(df)

df_summary = pd.concat(dfs, ignore_index=True)
df_summary['model_type'] = df_summary['model_type'].astype(str)
df_summary['dataset'] = df_summary['dataset'].astype(str)

METRIC_COLS = ['test_f1_macro', 'test_f1_weighted', 'test_accuracy', 'test_precision', 'test_recall']
datasets = sorted(df_summary['dataset'].unique())

print('=== FULL COMPARISON TABLE ===')
pivot = df_summary.pivot_table(index='dataset', columns='model_type', values=METRIC_COLS).round(4)
print(pivot.to_string())

print('\n=== DELTA (RoBERTa - Zero-Shot) ===')
for col in METRIC_COLS:
    try:
        delta = pivot[col]['RoBERTa Fine-Tuned'] - pivot[col]['Zero-Shot LLM']
        print(f'{col}: {delta.round(4).to_dict()}')
    except Exception:
        pass

# ── CV std for error bars ─────────────────────────────────────────────────────
cv_std = {}
for f in glob.glob('results/predictions/*_cv_results.csv'):
    ds = os.path.basename(f).replace('_cv_results.csv', '')
    try:
        cv = pd.read_csv(f)
        col = 'f1_macro' if 'f1_macro' in cv.columns else 'eval_f1_macro'
        cv_std[ds] = cv[col].std()
    except Exception:
        pass
print('\nCV std per dataset:', cv_std)

# ── Figure 1: Per-dataset grouped bars with error bars ───────────────────────
colors = {'RoBERTa Fine-Tuned': '#2196F3', 'Zero-Shot LLM': '#FF9800'}
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, ds in zip(axes, datasets):
    sub = df_summary[df_summary['dataset'] == ds].set_index('model_type')
    models = ['RoBERTa Fine-Tuned', 'Zero-Shot LLM']
    f1s = [float(sub.loc[m, 'test_f1_macro']) if m in sub.index else 0.0 for m in models]
    errs = [cv_std.get(ds, 0), 0]
    bars = ax.bar(models, f1s, yerr=errs, capsize=5,
                  color=[colors[m] for m in models], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(ds.capitalize(), fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('F1 Macro' if ds == datasets[0] else '')
    ax.set_xticklabels(models, rotation=10, ha='right', fontsize=9)
    if errs[0] > 0:
        ax.annotate('Error bar = CV std (RoBERTa)', xy=(0.02, 0.02),
                    xycoords='axes fraction', fontsize=7, color='gray')
plt.suptitle('RoBERTa Fine-Tuned vs Zero-Shot LLM - F1 Macro by Dataset',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison/comparison_grouped_bars.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/comparison_grouped_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: comparison_grouped_bars.png')

# ── Figure 2: Heatmap ─────────────────────────────────────────────────────────
heat_data = df_summary.copy()
heat_data['label'] = heat_data['model_type'].str[:4] + ' | ' + heat_data['dataset']
heat_data = heat_data.set_index('label')[METRIC_COLS].astype(float)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.3, vmax=1.0, ax=ax, linewidths=0.5)
ax.set_title('Performance Heatmap - All Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison/comparison_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/comparison_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: comparison_heatmap.png')

# ── Figure 3: Confusion matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col_idx, ds in enumerate(datasets):
    for row_idx, (mtype, suffix, true_col, pred_col) in enumerate([
        ('RoBERTa Fine-Tuned', 'roberta_test_predictions', 'label', 'pred_label'),
        ('Zero-Shot LLM', 'zeroshot_test_predictions', 'true_label', 'pred_label_final'),
    ]):
        ax = axes[row_idx][col_idx]
        try:
            pred = pd.read_csv(f'results/predictions/{ds}_{suffix}.csv')
            pred.columns = pred.columns.astype(str)
            tc = next((c for c in [true_col, 'true_label', 'label', 'y_true'] if c in pred.columns))
            pc = next((c for c in [pred_col, 'pred_label', 'predicted_label', 'pred'] if c in pred.columns))
            cm = confusion_matrix(pred[tc].astype(str), pred[pc].astype(str))
            labels = sorted(pred[tc].astype(str).unique())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(f'{mtype}\n{ds.capitalize()}', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        except Exception as e:
            ax.text(0.5, 0.5, f'N/A\n{e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
plt.suptitle('Confusion Matrices - All Datasets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison/comparison_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/comparison_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: comparison_confusion_matrices.png')

# ── Figure 4: PR curves ───────────────────────────────────────────────────────
# pos_class per dataset
POS_CLASS = {'manchester': 'misinformation', 'monkeypox': 'misinformation', 'pheme': 'rumour'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, ds in zip(axes, datasets):
    pos = POS_CLASS.get(ds, 'misinformation')
    for mtype, suffix, color in [
        ('RoBERTa', 'roberta_test_predictions', '#2196F3'),
        ('Zero-Shot', 'zeroshot_test_predictions', '#FF9800'),
    ]:
        try:
            pred = pd.read_csv(f'results/predictions/{ds}_{suffix}.csv')
            pred.columns = pred.columns.astype(str)
            tc = next(c for c in ['label', 'true_label'] if c in pred.columns)
            y_true = (pred[tc].astype(str) == pos).astype(int)
            if suffix == 'roberta_test_predictions':
                prob_col = f'prob_{pos}'
                if prob_col not in pred.columns:
                    prob_col = next(c for c in pred.columns if c.startswith('prob_') and pos[:3] in c)
                probs = pred[prob_col].astype(float)
            else:
                pc = next(c for c in ['pred_label_final', 'pred_label'] if c in pred.columns)
                conf = pred['confidence'].astype(float)
                probs = conf.where(pred[pc].astype(str) == pos, 1 - conf)
            prec, rec, _ = precision_recall_curve(y_true, probs)
            ap = average_precision_score(y_true, probs)
            ax.plot(rec, prec, color=color, label=f'{mtype} (AP={ap:.3f})', linewidth=2)
        except Exception as e:
            print(f'  PR curve skipped {ds}/{mtype}: {e}')
    baseline = y_true.mean() if 'y_true' in dir() else 0.5
    ax.axhline(float(baseline), linestyle='--', color='gray', alpha=0.6,
               label=f'Baseline ({float(baseline):.2f})')
    ax.set_title(ds.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision' if ds == datasets[0] else '')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
plt.suptitle('Precision-Recall Curves by Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison/comparison_pr_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/comparison_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: comparison_pr_curves.png')

# ── McNemar test ──────────────────────────────────────────────────────────────
mcnemar_results = {}
try:
    from statsmodels.stats.contingency_tables import mcnemar
    print('\n=== McNEMAR SIGNIFICANCE TESTS ===')
    for ds in datasets:
        try:
            r = pd.read_csv(f'results/predictions/{ds}_roberta_test_predictions.csv')
            z = pd.read_csv(f'results/predictions/{ds}_zeroshot_test_predictions.csv')
            r.columns = r.columns.astype(str)
            z.columns = z.columns.astype(str)
            rl = next(c for c in ['label', 'true_label', 'y_true'] if c in r.columns)
            rp = next(c for c in ['pred_label', 'predicted_label'] if c in r.columns)
            zl = next(c for c in ['true_label', 'label', 'y_true'] if c in z.columns)
            zp = next(c for c in ['pred_label_final', 'pred_label', 'predicted_label'] if c in z.columns)
            min_len = min(len(r), len(z))
            r_correct = (r[rp].astype(str) == r[rl].astype(str)).values[:min_len]
            z_correct = (z[zp].astype(str) == z[zl].astype(str)).values[:min_len]
            b = int(((r_correct == 1) & (z_correct == 0)).sum())
            c_val = int(((r_correct == 0) & (z_correct == 1)).sum())
            table = [
                [int(((r_correct == 1) & (z_correct == 1)).sum()), b],
                [c_val, int(((r_correct == 0) & (z_correct == 0)).sum())]
            ]
            exact = (b + c_val) < 25
            result = mcnemar(table, exact=exact, correction=not exact)
            sig = 'SIGNIFICANT' if result.pvalue < 0.05 else 'not significant'
            mcnemar_results[ds] = {'pvalue': float(result.pvalue), 'significant': bool(result.pvalue < 0.05)}
            print(f'  {ds}: p={result.pvalue:.4f} -> {sig} (b={b}, c={c_val})')
        except Exception as e:
            print(f'  {ds}: McNemar failed - {e}')
except ImportError:
    print('statsmodels not installed - skipping McNemar test')

# ── Winner analysis ───────────────────────────────────────────────────────────
print('\n=== WINNER ANALYSIS ===')
for ds in datasets:
    sub = df_summary[df_summary['dataset'] == ds].set_index('model_type')
    try:
        r_f1 = float(sub.loc['RoBERTa Fine-Tuned', 'test_f1_macro'])
        z_f1 = float(sub.loc['Zero-Shot LLM', 'test_f1_macro'])
        winner = 'RoBERTa' if r_f1 > z_f1 else 'Zero-Shot'
        diff = abs(r_f1 - z_f1)
        sig_str = ''
        if ds in mcnemar_results:
            sig_str = ' (p={:.4f}, {})'.format(
                mcnemar_results[ds]['pvalue'],
                'significant' if mcnemar_results[ds]['significant'] else 'not significant'
            )
        print(f'  {ds}: {winner} wins by {diff:.4f} F1{sig_str}')
    except Exception as e:
        print(f'  {ds}: {e}')

# ── Save master results ───────────────────────────────────────────────────────
df_summary.to_csv('results/master_results.csv', index=False)
print('\nSaved: results/master_results.csv')
print('\nAll comparison outputs generated successfully.')
