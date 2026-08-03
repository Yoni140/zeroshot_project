"""
Research comparison plots: fine-tuned transformers vs zero-shot LLMs.

Builds / refreshes:
  results/master_results.csv
  results/figures/comparison/*.png

Focus datasets (both families present): manchester, monkeypox, pheme.
Other event summaries (e.g. gurlitt) are kept in master_results but excluded
from family-average plots unless an LLM counterpart exists.

Usage (from project root):
  python scripts/plot_research_comparison.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'results' / 'predictions'
GOLD = ROOT / 'data' / 'gold_standard'
FIGS = ROOT / 'results' / 'figures' / 'comparison'
FIGS.mkdir(parents=True, exist_ok=True)

CORE_DATASETS = ['manchester', 'monkeypox', 'pheme']
EXCLUDE_FROM_FAMILY_AVG = {'prince-toronto', 'ebola-essien'}

TRANSFORMER_KEYS = {'roberta', 'bertweet', 'minilm', 'modernbert'}
LLM_KEYS = {
    'llama31', 'zeroshot', 'gpt_oss', 'llama33', 'qwen3', 'gemini_flash',
}

POS_CLASS = {
    'manchester': 'misinformation',
    'monkeypox': 'misinformation',
    'pheme': 'rumour',
}

FAMILY_COLORS = {
    'Transformer (fine-tuned)': '#2196F3',
    'LLM (zero-shot)': '#FF9800',
}

DISPLAY_NAMES = {
    'roberta': 'RoBERTa',
    'bertweet': 'BERTweet',
    'minilm': 'MiniLM',
    'modernbert': 'ModernBERT',
    'llama31': 'Llama 3.1 8B',
    'zeroshot': 'Llama 3.1 8B',
    'gpt_oss': 'GPT-OSS 120B',
    'llama33': 'Llama 3.3 70B',
    'qwen3': 'Qwen3 27B',
    'gemini_flash': 'Gemini 2.5 Flash',
}


# ── loading / tagging ─────────────────────────────────────────────────────────

def _infer_model_key(stem: str, row: pd.Series) -> str:
    """stem is e.g. manchester_bertweet or manchester_zeroshot."""
    if pd.notna(row.get('model_key')) and str(row['model_key']).strip():
        return str(row['model_key']).strip()
    # filename after dataset prefix
    parts = stem.split('_', 1)
    suffix = parts[1] if len(parts) > 1 else stem
    if suffix == 'zeroshot':
        return 'llama31'
    if suffix in TRANSFORMER_KEYS or suffix in LLM_KEYS:
        return suffix
    # model column fallbacks
    model = str(row.get('model', '')).lower()
    for key in list(TRANSFORMER_KEYS) + list(LLM_KEYS):
        if key in model or key.replace('_', '') in model.replace('-', '').replace('_', ''):
            return key
    if 'roberta' in model:
        return 'roberta'
    if 'bertweet' in model:
        return 'bertweet'
    if 'minilm' in model:
        return 'minilm'
    if 'modernbert' in model:
        return 'modernbert'
    if 'llama3.1' in model or 'llama31' in model:
        return 'llama31'
    if 'gemini' in model:
        return 'gemini_flash'
    if 'qwen' in model:
        return 'qwen3'
    if 'gpt' in model and 'oss' in model:
        return 'gpt_oss'
    if 'llama' in model and '3.3' in model:
        return 'llama33'
    return suffix


def _family(model_key: str) -> str | None:
    if model_key in TRANSFORMER_KEYS:
        return 'Transformer (fine-tuned)'
    if model_key in LLM_KEYS:
        return 'LLM (zero-shot)'
    return None


def _pos_f1_column(row: pd.Series, dataset: str) -> float | np.nan:
    pos = POS_CLASS.get(dataset)
    candidates = []
    if pos:
        candidates += [f'test_f1_{pos}', f'f1_{pos}']
    candidates += [
        'test_f1_misinformation', 'f1_misinformation',
        'test_f1_rumour', 'f1_rumour',
    ]
    for c in candidates:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != '':
            try:
                return float(row[c])
            except (TypeError, ValueError):
                pass
    return np.nan


def load_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED.glob('*_summary.csv')):
        stem = path.name.replace('_summary.csv', '')
        # skip CV-only aggregates if any
        if stem.endswith('_cv_results'):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0].copy()
        # dataset from file if missing
        dataset = str(r.get('dataset', '')).strip()
        if not dataset or dataset == 'nan':
            # longest matching prefix from known names
            dataset = stem
            for ds in sorted(CORE_DATASETS + ['pheme_all_events', 'gurlitt'], key=len, reverse=True):
                if stem.startswith(ds + '_'):
                    dataset = ds
                    break
            else:
                # generic: strip last known model token
                m = re.match(
                    r'(.+?)_(roberta|bertweet|minilm|modernbert|zeroshot|'
                    r'gpt_oss|llama33|qwen3|gemini_flash)$',
                    stem,
                )
                dataset = m.group(1) if m else stem.rsplit('_', 1)[0]

        model_key = _infer_model_key(stem, r)
        family = _family(model_key)
        if family is None:
            print(f'[skip] unknown family for {path.name} (key={model_key})')
            continue

        display = r.get('model_display')
        if pd.isna(display) or not str(display).strip():
            display = DISPLAY_NAMES.get(model_key, model_key)

        rows.append({
            'dataset': dataset,
            'model_key': model_key,
            'model_display': str(display),
            'model': r.get('model', model_key),
            'family': family,
            'test_f1_macro': pd.to_numeric(r.get('test_f1_macro'), errors='coerce'),
            'test_f1_weighted': pd.to_numeric(r.get('test_f1_weighted'), errors='coerce'),
            'test_accuracy': pd.to_numeric(r.get('test_accuracy'), errors='coerce'),
            'test_precision': pd.to_numeric(r.get('test_precision'), errors='coerce'),
            'test_recall': pd.to_numeric(r.get('test_recall'), errors='coerce'),
            'test_f1_pos': _pos_f1_column(r, dataset),
            'cv_f1_macro_mean': pd.to_numeric(r.get('cv_f1_macro_mean'), errors='coerce'),
            'cv_f1_macro_std': pd.to_numeric(r.get('cv_f1_macro_std'), errors='coerce'),
            'source_file': path.name,
        })

    out = pd.DataFrame(rows)
    # one row per dataset × model_key (prefer newest / first)
    out = out.sort_values(['dataset', 'family', 'model_key']).drop_duplicates(
        ['dataset', 'model_key'], keep='first')
    return out.reset_index(drop=True)


def load_domain_stats() -> pd.DataFrame:
    rows = []
    for path in sorted(GOLD.glob('*_gold_standard.csv')):
        ds = path.name.replace('_gold_standard.csv', '')
        df = pd.read_csv(path)
        if 'label' not in df.columns or df.empty:
            continue
        counts = df['label'].value_counts()
        n = len(df)
        minority = int(counts.min()) if len(counts) else 0
        majority = int(counts.max()) if len(counts) else 0
        rows.append({
            'dataset': ds,
            'n': n,
            'n_classes': int(df['label'].nunique()),
            'minority_n': minority,
            'majority_n': majority,
            'minority_ratio': minority / n if n else 0.0,
            'imbalance_ratio': (majority / minority) if minority else np.inf,
            'label_counts': counts.to_dict(),
        })
    return pd.DataFrame(rows)


# ── helpers ───────────────────────────────────────────────────────────────────

def family_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset × family: mean/std/min/max/best of test_f1_macro."""
    g = (df.groupby(['dataset', 'family'])['test_f1_macro']
           .agg(mean='mean', std='std', min='min', max='max', n='count')
           .reset_index())
    best = (df.sort_values('test_f1_macro', ascending=False)
              .groupby(['dataset', 'family'], as_index=False)
              .first()[['dataset', 'family', 'test_f1_macro', 'model_display']])
    best = best.rename(columns={
        'test_f1_macro': 'best',
        'model_display': 'best_model',
    })
    return g.merge(best, on=['dataset', 'family'])


def datasets_with_both(df: pd.DataFrame) -> list[str]:
    ok = []
    for ds in sorted(df['dataset'].unique()):
        if ds in EXCLUDE_FROM_FAMILY_AVG:
            continue
        sub = df[df['dataset'] == ds]
        fams = set(sub['family'])
        if 'Transformer (fine-tuned)' in fams and 'LLM (zero-shot)' in fams:
            ok.append(ds)
    return ok


def datasets_with_any(df: pd.DataFrame) -> list[str]:
    """All evaluated datasets (exclude known broken single-class events)."""
    return sorted(
        ds for ds in df['dataset'].unique()
        if ds not in EXCLUDE_FROM_FAMILY_AVG
    )


def savefig(name: str):
    path = FIGS / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path.relative_to(ROOT)}')


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_domain_size_balance(domain: pd.DataFrame):
    d = domain[~domain['dataset'].isin(EXCLUDE_FROM_FAMILY_AVG)].copy()
    d = d.sort_values('n', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#7f8c8d' if ds not in CORE_DATASETS else '#3498db' for ds in d['dataset']]
    axes[0].barh(d['dataset'], d['n'], color=colors, edgecolor='black', linewidth=0.4)
    axes[0].set_xlabel('Gold-standard size (n)')
    axes[0].set_title('Dataset size', fontweight='bold')
    axes[0].invert_yaxis()

    axes[1].barh(d['dataset'], d['minority_ratio'], color=colors, edgecolor='black', linewidth=0.4)
    axes[1].axvline(0.5, color='gray', linestyle='--', linewidth=1, label='balanced (0.5)')
    axes[1].set_xlabel('Minority-class ratio')
    axes[1].set_xlim(0, 0.55)
    axes[1].set_title('Class balance', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    fig.suptitle('Domain characterization — size & balance', fontsize=14, fontweight='bold')
    savefig('01_domain_size_balance.png')


def plot_family_mean_by_dataset(df: pd.DataFrame, datasets: list[str]):
    stats = family_stats(df[df['dataset'].isin(datasets)])
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, fam in enumerate(['Transformer (fine-tuned)', 'LLM (zero-shot)']):
        means, stds = [], []
        for ds in datasets:
            row = stats[(stats['dataset'] == ds) & (stats['family'] == fam)]
            means.append(float(row['mean'].iloc[0]) if len(row) else np.nan)
            stds.append(float(row['std'].fillna(0).iloc[0]) if len(row) else 0.0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=fam, color=FAMILY_COLORS[fam], alpha=0.9,
                      edgecolor='black', linewidth=0.5)
        for bar, m in zip(bars, means):
            if np.isfinite(m):
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 1.08)
    ax.set_title('Mean macro F1 by model family (error bar = std across models)',
                 fontweight='bold')
    ax.legend(frameon=True)
    ax.axhline(0, color='black', linewidth=0.3)
    savefig('02_family_mean_f1_by_dataset.png')


def plot_family_overall(df: pd.DataFrame, datasets: list[str]):
    """Mean of per-dataset family means (each domain weighted equally)."""
    stats = family_stats(df[df['dataset'].isin(datasets)])
    rows = []
    for fam in ['Transformer (fine-tuned)', 'LLM (zero-shot)']:
        vals = stats.loc[stats['family'] == fam, 'mean']
        rows.append({
            'family': fam,
            'mean': float(vals.mean()),
            'std': float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
            'n_datasets': int(len(vals)),
        })
    agg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(agg['family'], agg['mean'], yerr=agg['std'], capsize=6,
                  color=[FAMILY_COLORS[f] for f in agg['family']],
                  edgecolor='black', alpha=0.9)
    for bar, m in zip(bars, agg['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.025,
                f'{m:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 (mean of per-dataset family means)')
    ax.set_ylim(0, 1.08)
    ax.set_title('Overall average performance by model family', fontweight='bold')
    savefig('03_family_mean_overall.png')


def plot_family_boxplot(df: pd.DataFrame, datasets: list[str]):
    sub = df[df['dataset'].isin(datasets)].copy()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        part = sub[sub['dataset'] == ds]
        order = ['Transformer (fine-tuned)', 'LLM (zero-shot)']
        sns.boxplot(data=part, x='family', y='test_f1_macro', order=order,
                    palette=FAMILY_COLORS, ax=ax, width=0.55)
        sns.stripplot(data=part, x='family', y='test_f1_macro', order=order,
                      color='black', size=6, ax=ax, jitter=0.08)
        ax.set_title(ds.capitalize(), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Macro F1' if ds == datasets[0] else '')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=15)
        for i, fam in enumerate(order):
            labs = part.loc[part['family'] == fam, 'model_display']
            vals = part.loc[part['family'] == fam, 'test_f1_macro']
            for y, lab in zip(vals, labs):
                ax.annotate(lab, (i, y), textcoords='offset points',
                            xytext=(6, 0), fontsize=6, alpha=0.75)
    fig.suptitle('Macro F1 distribution within each model family',
                 fontsize=13, fontweight='bold')
    savefig('04_family_boxplot_f1.png')


def plot_dumbbell(df: pd.DataFrame, datasets: list[str]):
    stats = family_stats(df[df['dataset'].isin(datasets)])
    fig, ax = plt.subplots(figsize=(8, 4 + 0.35 * len(datasets)))
    y = np.arange(len(datasets))
    for i, ds in enumerate(datasets):
        t = stats[(stats['dataset'] == ds) & (stats['family'] == 'Transformer (fine-tuned)')]
        l = stats[(stats['dataset'] == ds) & (stats['family'] == 'LLM (zero-shot)')]
        if t.empty or l.empty:
            continue
        tm, lm = float(t['mean'].iloc[0]), float(l['mean'].iloc[0])
        ax.plot([lm, tm], [i, i], color='#95a5a6', linewidth=2, zorder=1)
        ax.scatter([lm], [i], s=90, color=FAMILY_COLORS['LLM (zero-shot)'],
                   zorder=2, edgecolor='black', label='LLM mean' if i == 0 else None)
        ax.scatter([tm], [i], s=90, color=FAMILY_COLORS['Transformer (fine-tuned)'],
                   zorder=2, edgecolor='black', label='Transformer mean' if i == 0 else None)
        ax.text(max(tm, lm) + 0.02, i, f'Δ={tm - lm:+.3f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([d.capitalize() for d in datasets])
    ax.set_xlabel('Mean macro F1')
    ax.set_xlim(0, 1.15)
    ax.set_title('Family-mean gap (dumbbell): Transformer vs LLM', fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    savefig('05_family_dumbbell_delta.png')


def plot_delta_heatmap(df: pd.DataFrame, datasets: list[str]):
    """Δ = model F1 − best LLM on that dataset."""
    rows = []
    for ds in datasets:
        sub = df[df['dataset'] == ds]
        best_llm = sub.loc[sub['family'] == 'LLM (zero-shot)', 'test_f1_macro'].max()
        for _, r in sub.iterrows():
            rows.append({
                'dataset': ds.capitalize(),
                'model': r['model_display'],
                'family': r['family'],
                'delta': float(r['test_f1_macro'] - best_llm),
                'f1': float(r['test_f1_macro']),
            })
    heat = pd.DataFrame(rows)
    pivot = heat.pivot(index='model', columns='dataset', values='delta')
    # order models: transformers first
    fam_order = (heat.drop_duplicates('model')
                     .set_index('model')['family']
                     .map({'Transformer (fine-tuned)': 0, 'LLM (zero-shot)': 1}))
    pivot = pivot.loc[fam_order.sort_values().index]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=0.4, cbar_kws={'label': 'Δ F1 vs best LLM'})
    ax.set_title('Δ macro F1 relative to best LLM on each dataset', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    savefig('06_delta_f1_heatmap.png')


def plot_model_heatmap(df: pd.DataFrame, datasets: list[str]):
    sub = df[df['dataset'].isin(datasets)].copy()
    sub['dataset_lab'] = sub['dataset'].str.capitalize()
    pivot = sub.pivot(index='model_display', columns='dataset_lab', values='test_f1_macro')
    fam = (sub.drop_duplicates('model_display')
              .set_index('model_display')['family']
              .map({'Transformer (fine-tuned)': 0, 'LLM (zero-shot)': 1}))
    pivot = pivot.loc[fam.sort_values().index]
    cols = [d.capitalize() for d in datasets if d.capitalize() in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0.2, vmax=1.0,
                ax=ax, linewidths=0.4)
    ax.set_title('Macro F1 — all models × datasets', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    savefig('07_model_f1_heatmap.png')


def plot_best_vs_mean(df: pd.DataFrame, datasets: list[str]):
    stats = family_stats(df[df['dataset'].isin(datasets)])
    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ('Transformer mean', 'Transformer (fine-tuned)', 'mean', '#2196F3', 0),
        ('Transformer best', 'Transformer (fine-tuned)', 'best', '#0D47A1', 1),
        ('LLM mean', 'LLM (zero-shot)', 'mean', '#FF9800', 2),
        ('LLM best', 'LLM (zero-shot)', 'best', '#E65100', 3),
    ]
    for label, fam, col, color, i in series:
        vals = []
        for ds in datasets:
            row = stats[(stats['dataset'] == ds) & (stats['family'] == fam)]
            vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color,
               edgecolor='black', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title('Family mean vs best-in-family', fontweight='bold')
    savefig('08_best_vs_mean_family.png')


def plot_per_class_f1(df: pd.DataFrame, datasets: list[str]):
    stats_rows = []
    for ds in datasets:
        sub = df[df['dataset'] == ds]
        for fam in ['Transformer (fine-tuned)', 'LLM (zero-shot)']:
            vals = sub.loc[sub['family'] == fam, 'test_f1_pos'].dropna()
            if len(vals) == 0:
                continue
            stats_rows.append({
                'dataset': ds,
                'family': fam,
                'mean_pos_f1': float(vals.mean()),
                'std_pos_f1': float(vals.std()) if len(vals) > 1 else 0.0,
                'pos_label': POS_CLASS.get(ds, 'positive'),
            })
    if not stats_rows:
        print('[warn] no positive-class F1 columns — skipping per-class plot')
        return
    st = pd.DataFrame(stats_rows)
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fam in enumerate(['Transformer (fine-tuned)', 'LLM (zero-shot)']):
        means, stds, labels = [], [], []
        for ds in datasets:
            row = st[(st['dataset'] == ds) & (st['family'] == fam)]
            means.append(float(row['mean_pos_f1'].iloc[0]) if len(row) else np.nan)
            stds.append(float(row['std_pos_f1'].iloc[0]) if len(row) else 0.0)
            labels.append(POS_CLASS.get(ds, 'pos'))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      color=FAMILY_COLORS[fam], label=fam, edgecolor='black', alpha=0.9)
        for bar, m, lab in zip(bars, means, labels):
            if np.isfinite(m):
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d.capitalize()}\n({POS_CLASS[d]})' for d in datasets])
    ax.set_ylabel('Positive-class F1 (family mean)')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title('Mean positive-class F1 by model family', fontweight='bold')
    savefig('09_per_class_f1_by_family.png')


def plot_f1_vs_size(df: pd.DataFrame, domain: pd.DataFrame, datasets: list[str]):
    """Compare family-mean performance against gold-standard dataset size.

    Includes every evaluated dataset (not only those with both families).
    Missing family → no point / no bar for that family.
    """
    stats = family_stats(df[df['dataset'].isin(datasets)])
    sizes = domain[['dataset', 'n', 'minority_ratio', 'n_classes']].rename(
        columns={'n': 'gold_n'})
    merged = stats.merge(sizes, on='dataset', how='left')
    if merged.empty or merged['gold_n'].isna().all():
        print('[warn] no gold sizes — skipping F1-vs-size plot')
        return

    t = merged[merged['family'] == 'Transformer (fine-tuned)'][
        ['dataset', 'mean', 'best']
    ].rename(columns={'mean': 'transformer_mean', 'best': 'transformer_best'})
    l = merged[merged['family'] == 'LLM (zero-shot)'][
        ['dataset', 'mean', 'best']
    ].rename(columns={'mean': 'llm_mean', 'best': 'llm_best'})
    wide = t.merge(l, on='dataset', how='outer')
    wide = wide.merge(sizes, on='dataset', how='left')
    wide = wide.dropna(subset=['gold_n']).sort_values('gold_n').reset_index(drop=True)
    wide['delta_mean'] = wide['transformer_mean'] - wide['llm_mean']
    wide['delta_best'] = wide['transformer_best'] - wide['llm_best']
    wide['has_both'] = wide['transformer_mean'].notna() & wide['llm_mean'].notna()

    out_csv = ROOT / 'results' / 'family_f1_vs_size.csv'
    wide.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv.relative_to(ROOT)} ({len(wide)} datasets)')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A
    ax = axes[0]
    for _, r in wide.iterrows():
        tm, lm = r['transformer_mean'], r['llm_mean']
        if pd.notna(tm) and pd.notna(lm):
            ax.plot([r['gold_n'], r['gold_n']], [lm, tm],
                    color='#95a5a6', linewidth=1.8, zorder=1)
        if pd.notna(tm):
            ax.scatter([r['gold_n']], [tm], s=130, marker='o',
                       color=FAMILY_COLORS['Transformer (fine-tuned)'],
                       edgecolor='black', zorder=3)
        if pd.notna(lm):
            ax.scatter([r['gold_n']], [lm], s=130, marker='s',
                       color=FAMILY_COLORS['LLM (zero-shot)'],
                       edgecolor='black', zorder=3)
        y_ann = np.nanmax(np.array([tm, lm], dtype=float))
        ax.annotate(str(r['dataset']), (r['gold_n'], y_ann),
                    textcoords='offset points', xytext=(5, 5), fontsize=7)

    for col, fam, style in [
        ('transformer_mean', 'Transformer (fine-tuned)', '-'),
        ('llm_mean', 'LLM (zero-shot)', '--'),
    ]:
        part = wide.dropna(subset=[col, 'gold_n'])
        if len(part) >= 2:
            coef = np.polyfit(part['gold_n'].astype(float), part[col].astype(float), 1)
            xs = np.linspace(float(wide['gold_n'].min()), float(wide['gold_n'].max()), 50)
            ax.plot(xs, np.poly1d(coef)(xs), color=FAMILY_COLORS[fam],
                    linestyle=style, alpha=0.55, linewidth=1.5,
                    label=f'{fam.split()[0]} trend')
    ax.scatter([], [], s=110, marker='o', color=FAMILY_COLORS['Transformer (fine-tuned)'],
               edgecolor='black', label='Transformer mean')
    ax.scatter([], [], s=110, marker='s', color=FAMILY_COLORS['LLM (zero-shot)'],
               edgecolor='black', label='LLM mean')
    ax.set_xlabel('Gold-standard size (n)')
    ax.set_ylabel('Family-mean macro F1')
    ax.set_ylim(0, 1.05)
    ax.set_title('A. Family mean F1 vs size (all evaluated)', fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    # Panel B — Δ only where both families exist
    ax = axes[1]
    both = wide[wide['has_both']]
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    if len(both):
        ax.plot(both['gold_n'], both['delta_mean'], color='#6c5ce7',
                marker='D', markersize=9, linewidth=2, label='Δ mean (T − LLM)')
        ax.plot(both['gold_n'], both['delta_best'], color='#00b894',
                marker='^', markersize=9, linewidth=2, linestyle='--',
                label='Δ best (T − LLM)')
        for _, r in both.iterrows():
            ax.annotate(str(r['dataset']), (r['gold_n'], r['delta_mean']),
                        textcoords='offset points', xytext=(5, 5), fontsize=7)
            ax.annotate(f'{r["delta_mean"]:+.3f}', (r['gold_n'], r['delta_mean']),
                        textcoords='offset points', xytext=(5, -11), fontsize=7,
                        color='#6c5ce7')
    only_t = wide[~wide['has_both'] & wide['transformer_mean'].notna()]
    for _, r in only_t.iterrows():
        ax.axvline(r['gold_n'], color='#2196F3', alpha=0.25, linestyle=':')
        ax.annotate(f"{r['dataset']} (T only)", (r['gold_n'], 0),
                    ha='center', va='bottom', fontsize=6, color='#2196F3',
                    xytext=(0, 6), textcoords='offset points')
    ax.set_xlabel('Gold-standard size (n)')
    ax.set_ylabel('Macro F1 gap (Transformer − LLM)')
    ax.set_title('B. Performance gap vs size (both families)', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C — bars for every evaluated dataset, ordered by size
    ax = axes[2]
    width = 0.35
    x = np.arange(len(wide))
    for i, (_, r) in enumerate(wide.iterrows()):
        if pd.notna(r['transformer_mean']):
            ax.bar(i - width / 2, r['transformer_mean'], width,
                   color=FAMILY_COLORS['Transformer (fine-tuned)'], edgecolor='black')
            ax.text(i - width / 2, r['transformer_mean'] + 0.015,
                    f'{r["transformer_mean"]:.3f}', ha='center', va='bottom', fontsize=6)
        if pd.notna(r['llm_mean']):
            ax.bar(i + width / 2, r['llm_mean'], width,
                   color=FAMILY_COLORS['LLM (zero-shot)'], edgecolor='black')
            ax.text(i + width / 2, r['llm_mean'] + 0.015,
                    f'{r["llm_mean"]:.3f}', ha='center', va='bottom', fontsize=6)
    ax.bar([], [], color=FAMILY_COLORS['Transformer (fine-tuned)'],
           edgecolor='black', label='Transformer mean')
    ax.bar([], [], color=FAMILY_COLORS['LLM (zero-shot)'],
           edgecolor='black', label='LLM mean')
    labels = [f"{r['dataset']}\n(n={int(r['gold_n'])})" for _, r in wide.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Family-mean macro F1')
    ax.set_ylim(0, 1.08)
    ax.set_title('C. Means ordered by dataset size', fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle('Family-mean performance vs dataset size — all evaluated datasets',
                 fontsize=13, fontweight='bold')
    savefig('10_f1_vs_dataset_size.png')



def _pred_cols(pred: pd.DataFrame):
    true_c = next(c for c in ['true_label', 'label', 'y_true'] if c in pred.columns)
    pred_c = next(
        c for c in ['pred_label_final', 'pred_label', 'predicted_label']
        if c in pred.columns
    )
    return true_c, pred_c


def _find_pred_file(dataset: str, model_key: str) -> Path | None:
    candidates = [
        PRED / f'{dataset}_{model_key}_test_predictions.csv',
    ]
    if model_key == 'llama31':
        candidates.append(PRED / f'{dataset}_zeroshot_test_predictions.csv')
    for p in candidates:
        if p.exists():
            return p
    return None


def plot_agreement(df: pd.DataFrame, datasets: list[str]):
    """Best transformer vs best LLM contingency on shared test labels."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.2))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        t_row = sub[sub['family'] == 'Transformer (fine-tuned)'].sort_values(
            'test_f1_macro', ascending=False).iloc[0]
        l_row = sub[sub['family'] == 'LLM (zero-shot)'].sort_values(
            'test_f1_macro', ascending=False).iloc[0]
        t_path = _find_pred_file(ds, t_row['model_key'])
        l_path = _find_pred_file(ds, l_row['model_key'])
        if t_path is None or l_path is None:
            ax.text(0.5, 0.5, 'predictions missing', ha='center', va='center')
            ax.set_title(ds)
            continue

        t_pred = pd.read_csv(t_path)
        l_pred = pd.read_csv(l_path)
        tt, tp = _pred_cols(t_pred)
        lt, lp = _pred_cols(l_pred)
        n = min(len(t_pred), len(l_pred))
        t_ok = (t_pred[tp].astype(str).values[:n] == t_pred[tt].astype(str).values[:n])
        l_ok = (l_pred[lp].astype(str).values[:n] == l_pred[lt].astype(str).values[:n])

        both = int((t_ok & l_ok).sum())
        t_only = int((t_ok & ~l_ok).sum())
        l_only = int((~t_ok & l_ok).sum())
        neither = int((~t_ok & ~l_ok).sum())
        counts = [both, t_only, l_only, neither]
        labels = ['Both\ncorrect', 'Transformer\nonly', 'LLM\nonly', 'Both\nwrong']
        colors = ['#2ecc71', '#2196F3', '#FF9800', '#e74c3c']
        bars = ax.bar(labels, counts, color=colors, edgecolor='black')
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01,
                    str(c), ha='center', va='bottom', fontsize=9)
        ax.set_title(
            f'{ds.capitalize()}\n{t_row["model_display"]} vs {l_row["model_display"]}',
            fontsize=10, fontweight='bold')
        ax.set_ylabel('Examples' if ds == datasets[0] else '')

    fig.suptitle('Agreement breakdown — best transformer vs best LLM',
                 fontsize=13, fontweight='bold')
    savefig('11_agreement_breakdown.png')


def plot_side_by_side_cm(df: pd.DataFrame, datasets: list[str]):
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.6 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])

    for i, ds in enumerate(datasets):
        sub = df[df['dataset'] == ds]
        for j, fam in enumerate(['Transformer (fine-tuned)', 'LLM (zero-shot)']):
            ax = axes[i, j]
            row = sub[sub['family'] == fam].sort_values(
                'test_f1_macro', ascending=False).iloc[0]
            path = _find_pred_file(ds, row['model_key'])
            if path is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                continue
            pred = pd.read_csv(path)
            tc, pc = _pred_cols(pred)
            labels = sorted(pred[tc].astype(str).unique())
            cm = confusion_matrix(pred[tc].astype(str), pred[pc].astype(str), labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(f'{ds.capitalize()} — {row["model_display"]}\n({fam})',
                         fontsize=9, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True' if j == 0 else '')

    fig.suptitle('Confusion matrices — best model per family',
                 fontsize=13, fontweight='bold')
    savefig('12_best_family_confusion_matrices.png')


def plot_summary_table(df: pd.DataFrame, datasets: list[str]):
    stats = family_stats(df[df['dataset'].isin(datasets)])
    rows = []
    for ds in datasets:
        t = stats[(stats['dataset'] == ds) & (stats['family'] == 'Transformer (fine-tuned)')]
        l = stats[(stats['dataset'] == ds) & (stats['family'] == 'LLM (zero-shot)')]
        if t.empty or l.empty:
            continue
        tm, lm = float(t['mean'].iloc[0]), float(l['mean'].iloc[0])
        tb, lb = float(t['best'].iloc[0]), float(l['best'].iloc[0])
        rows.append([
            ds.capitalize(),
            f'{tm:.3f}', f'{lm:.3f}', f'{tm - lm:+.3f}',
            f'{tb:.3f} ({t["best_model"].iloc[0]})',
            f'{lb:.3f} ({l["best_model"].iloc[0]})',
            f'{tb - lb:+.3f}',
        ])

    # overall mean-of-means
    t_means = [float(stats[(stats['dataset'] == ds) &
                           (stats['family'] == 'Transformer (fine-tuned)')]['mean'].iloc[0])
               for ds in datasets]
    l_means = [float(stats[(stats['dataset'] == ds) &
                           (stats['family'] == 'LLM (zero-shot)')]['mean'].iloc[0])
               for ds in datasets]
    rows.append([
        'OVERALL (mean)',
        f'{np.mean(t_means):.3f}', f'{np.mean(l_means):.3f}',
        f'{np.mean(t_means) - np.mean(l_means):+.3f}',
        '—', '—', '—',
    ])

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.45 * len(rows)))
    ax.axis('off')
    cols = ['Dataset', 'T mean', 'LLM mean', 'Δ mean',
            'T best', 'LLM best', 'Δ best']
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    last = len(rows)
    for j in range(len(cols)):
        table[last, j].set_facecolor('#ECEFF1')
        table[last, j].set_text_props(fontweight='bold')
    ax.set_title('Transformer vs LLM — family mean and best-in-family (macro F1)',
                 fontsize=12, fontweight='bold', pad=12)
    savefig('13_family_summary_table.png')


def write_family_csv(df: pd.DataFrame, datasets: list[str]):
    stats = family_stats(df[df['dataset'].isin(datasets)])
    out = PRED.parent / 'family_average_results.csv'
    stats.to_csv(out, index=False)
    print(f'Saved: {out.relative_to(ROOT)}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading summaries…')
    df = load_summaries()
    print(f'  loaded {len(df)} model×dataset rows')
    print(df.groupby(['dataset', 'family']).size().to_string())

    domain = load_domain_stats()
    datasets_both = datasets_with_both(df)
    datasets_all = datasets_with_any(df)
    print(f'\nDatasets with both families: {datasets_both}')
    print(f'Datasets with any results:    {datasets_all}')
    if not datasets_both and not datasets_all:
        raise SystemExit('No evaluated datasets found.')

    # refresh master results (full, not RoBERTa-only)
    master_path = ROOT / 'results' / 'master_results.csv'
    df.to_csv(master_path, index=False)
    print(f'Saved: {master_path.relative_to(ROOT)}')
    # family CSV: all evaluated datasets (transformer-only rows included)
    write_family_csv(df, datasets_all)

    plot_domain_size_balance(domain)
    # Fair T-vs-LLM panels need both families
    if datasets_both:
        plot_family_mean_by_dataset(df, datasets_both)
        plot_family_overall(df, datasets_both)
        plot_family_boxplot(df, datasets_both)
        plot_dumbbell(df, datasets_both)
        plot_delta_heatmap(df, datasets_both)
        plot_model_heatmap(df, datasets_both)
        plot_best_vs_mean(df, datasets_both)
        plot_per_class_f1(df, datasets_both)
        plot_agreement(df, datasets_both)
        plot_side_by_side_cm(df, datasets_both)
        plot_summary_table(df, datasets_both)
    # Size comparison uses every evaluated dataset
    plot_f1_vs_size(df, domain, datasets_all)

    print('\nDone. Figures in results/figures/comparison/')


if __name__ == '__main__':
    main()
