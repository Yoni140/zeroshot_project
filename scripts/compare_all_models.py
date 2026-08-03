"""
compare_all_models.py — compare every available model on every dataset.

Loads all `results/predictions/*_summary.csv` files (encoders + zero-shot LLMs),
writes an aggregated master table, and produces comparison figures.

Usage:
  python scripts/compare_all_models.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / 'results' / 'predictions'
FIGS = ROOT / 'results' / 'figures' / 'comparison'
FIGS.mkdir(parents=True, exist_ok=True)

# Known model-key → display name (longest keys first when matching filenames)
MODEL_DISPLAY = {
    'modernbert': 'ModernBERT',
    'bertweet': 'BERTweet',
    'minilm': 'MiniLM',
    'roberta': 'RoBERTa',
    'gemini_flash': 'Gemini 2.5 Flash',
    'gpt_oss': 'GPT-OSS 120B',
    'llama33': 'Llama 3.3 70B',
    'qwen3': 'Qwen3.6 27B',
    'zeroshot': 'Zero-Shot (Ollama)',
    'ollama': 'Zero-Shot (Ollama)',
}

ENCODER_KEYS = {'roberta', 'bertweet', 'minilm', 'modernbert'}
LLM_KEYS = {'gemini_flash', 'gpt_oss', 'llama33', 'qwen3', 'zeroshot', 'ollama'}

METRIC_COLS = [
    'test_f1_macro', 'test_f1_weighted', 'test_accuracy',
    'test_precision', 'test_recall',
]


def parse_summary_path(path: Path):
    """Return (dataset, model_key) from `{dataset}_{model}_summary.csv`."""
    name = path.name
    if not name.endswith('_summary.csv'):
        return None
    stem = name[: -len('_summary.csv')]
    # Prefer longest model key match at the end
    for key in sorted(MODEL_DISPLAY.keys(), key=len, reverse=True):
        suffix = f'_{key}'
        if stem.endswith(suffix):
            return stem[: -len(suffix)], key
    return None


def load_all_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDS.glob('*_summary.csv')):
        parsed = parse_summary_path(path)
        if parsed is None:
            continue
        dataset, model_key = parsed
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row['dataset'] = dataset
        row['model_key'] = model_key
        row['model_display'] = MODEL_DISPLAY.get(model_key, model_key)
        if model_key in ENCODER_KEYS:
            row['family'] = 'encoder'
        elif model_key in LLM_KEYS:
            row['family'] = 'llm'
        else:
            row['family'] = 'other'
        # Normalize metric column names if missing aliases
        if 'test_precision' not in row and 'test_precision_macro' in row:
            row['test_precision'] = row['test_precision_macro']
        if 'test_recall' not in row and 'test_recall_macro' in row:
            row['test_recall'] = row['test_recall_macro']
        rows.append(row)
    if not rows:
        raise SystemExit('No summary CSVs found under results/predictions/')
    out = pd.DataFrame(rows)
    # Dedup: prefer richer rows if duplicates (same dataset+model_key)
    out = out.drop_duplicates(subset=['dataset', 'model_key'], keep='last')
    return out


def plot_heatmap(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    pivot = df.pivot_table(index='dataset', columns='model_display',
                           values=metric, aggfunc='first')
    # Stable column order: encoders then LLMs
    preferred = [
        'RoBERTa', 'BERTweet', 'MiniLM', 'ModernBERT',
        'Gemini 2.5 Flash', 'GPT-OSS 120B', 'Llama 3.3 70B', 'Qwen3.6 27B',
        'Zero-Shot (Ollama)',
    ]
    cols = [c for c in preferred if c in pivot.columns] + \
           [c for c in pivot.columns if c not in preferred]
    pivot = pivot[cols]

    h = max(6, 0.45 * len(pivot) + 2)
    w = max(10, 1.1 * len(cols) + 3)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.3, vmax=1.0, linewidths=0.4, ax=ax)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel('Model')
    ax.set_ylabel('Dataset')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_grouped_bars_by_dataset(df: pd.DataFrame, out_dir: Path):
    """One grouped-bar figure per dataset (all models that have results)."""
    for ds, sub in df.groupby('dataset'):
        sub = sub.sort_values('test_f1_macro', ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(sub) + 2), 5))
        colors = ['#2196F3' if f == 'encoder' else '#FF9800' for f in sub['family']]
        bars = ax.bar(sub['model_display'], sub['test_f1_macro'],
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, sub['test_f1_macro']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('F1 Macro')
        ax.set_title(f'{ds} — Model Comparison (F1 Macro)', fontweight='bold')
        ax.tick_params(axis='x', rotation=25)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#2196F3', edgecolor='black', label='Fine-tuned encoder'),
            Patch(facecolor='#FF9800', edgecolor='black', label='Zero-shot LLM'),
        ], loc='lower right', fontsize=8)
        plt.tight_layout()
        path = out_dir / f'comparison_{ds}_models.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')


def plot_encoder_events_heatmap(df: pd.DataFrame, out_path: Path):
    enc = df[df['family'] == 'encoder'].copy()
    if enc.empty:
        return
    plot_heatmap(
        enc, 'test_f1_macro', out_path,
        'Encoder F1 Macro — All Datasets',
    )


def plot_best_per_dataset(df: pd.DataFrame, out_path: Path):
    """Bar of best F1 per dataset, annotated with winning model."""
    best = (df.sort_values('test_f1_macro', ascending=False)
              .groupby('dataset', as_index=False)
              .first())
    best = best.sort_values('test_f1_macro', ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(best) + 2)))
    colors = ['#2196F3' if f == 'encoder' else '#FF9800' for f in best['family']]
    ax.barh(best['dataset'], best['test_f1_macro'], color=colors,
            edgecolor='black', linewidth=0.5)
    for y, (_, row) in enumerate(best.iterrows()):
        ax.text(row['test_f1_macro'] + 0.01, y,
                f"{row['model_display']} ({row['test_f1_macro']:.3f})",
                va='center', fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel('Best F1 Macro')
    ax.set_title('Best Model per Dataset', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    df = load_all_summaries()
    print(f'Loaded {len(df)} summary rows '
          f'({df["dataset"].nunique()} datasets × up to {df["model_key"].nunique()} models)')
    print(df.groupby('family')['model_key'].nunique().to_string())

    # Master table
    keep = ['dataset', 'model_key', 'model_display', 'family'] + \
           [c for c in METRIC_COLS if c in df.columns]
    # Also keep optional pos-class F1 columns
    extra = [c for c in df.columns if c.startswith('test_f1_') and c not in keep]
    master = df[keep + extra].sort_values(['dataset', 'family', 'model_key'])
    master_path = ROOT / 'results' / 'master_results_all_models.csv'
    master.to_csv(master_path, index=False)
    print(f'\nMaster table: {master_path}')

    # Also refresh classic master_results.csv with same content
    master.to_csv(ROOT / 'results' / 'master_results.csv', index=False)

    # Figures
    plot_heatmap(
        df, 'test_f1_macro',
        FIGS / 'comparison_all_models_f1_heatmap.png',
        'F1 Macro — All Models × All Datasets',
    )
    plot_heatmap(
        df, 'test_accuracy',
        FIGS / 'comparison_all_models_accuracy_heatmap.png',
        'Accuracy — All Models × All Datasets',
    )
    plot_encoder_events_heatmap(
        df, FIGS / 'comparison_encoders_f1_heatmap.png',
    )
    plot_best_per_dataset(df, FIGS / 'comparison_best_per_dataset.png')
    plot_grouped_bars_by_dataset(df, FIGS)

    # Console pivot
    print('\n=== F1 MACRO PIVOT ===')
    pivot = df.pivot_table(index='dataset', columns='model_display',
                           values='test_f1_macro', aggfunc='first')
    print(pivot.round(3).to_string())

    # Coverage gaps: datasets missing LLM results
    enc_ds = set(df.loc[df['family'] == 'encoder', 'dataset'])
    llm_ds = set(df.loc[df['family'] == 'llm', 'dataset'])
    missing_llm = sorted(enc_ds - llm_ds)
    if missing_llm:
        print('\nDatasets with encoder results but NO LLM results yet:')
        for d in missing_llm:
            print(f'  - {d}')
        print('Run: python scripts/run_all_cloud.py --events-only')


if __name__ == '__main__':
    main()
