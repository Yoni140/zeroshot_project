"""
plot_pheme_events.py — EDA + 2D projection plots for PHEME event datasets.

Produces (per dataset under results/figures/{dataset}/):
  {dataset}_01_label_dist.png
  {dataset}_02_tweet_length.png
  {dataset}_03_top_words.png
  {dataset}_05_tsne.png          # TF-IDF → TruncatedSVD → t-SNE
  {dataset}_06_gold_standard.png

Usage:
  python scripts/plot_pheme_events.py
  python scripts/plot_pheme_events.py --dataset charliehebdo
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
GOLD = ROOT / 'data' / 'gold_standard'
FIGS = ROOT / 'results' / 'figures'

EVENTS = [
    'pheme_all_events', 'gurlitt', 'germanwings-crash', 'ebola-essien',
    'charliehebdo', 'ferguson', 'ottawashooting', 'prince-toronto',
    'putinmissing', 'sydneysiege',
]

STOP = set(
    'a an the and or but if in on at to for of is are was were be been being '
    'i you he she it we they me him her us them my your his its our their this '
    'that these those with from by as not no so do does did have has had will '
    'would can could should just about into over after before more most other '
    'some such only own same than too very rt amp http https via'.split()
)

PALETTE = {'not_rumour': '#2ecc71', 'rumour': '#e74c3c', 'unrelated': '#95a5a6',
           'reliable': '#3498db', 'misinformation': '#e67e22'}


def tokens(text: str):
    return [w for w in re.findall(r"[a-z0-9']+", str(text).lower()) if w not in STOP and len(w) > 2]


def plot_label_dist(df, ds, out_dir):
    counts = df['label'].value_counts()
    colors = [PALETTE.get(l, '#7f8c8d') for l in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor='black')
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.01,
                f'{v:,}', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'{ds} — Label Distribution (cleaned)', fontweight='bold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_dir / f'{ds}_01_label_dist.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tweet_length(df, ds, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab in df['label'].unique():
        sub = df[df['label'] == lab]['word_count']
        ax.hist(sub, bins=30, alpha=0.55, label=lab, color=PALETTE.get(lab, None))
    ax.set_title(f'{ds} — Tweet Word Count by Label', fontweight='bold')
    ax.set_xlabel('Word count')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{ds}_02_tweet_length.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_words(df, ds, out_dir, top_n=15):
    labels = list(df['label'].unique())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        cnt = Counter()
        for t in df.loc[df['label'] == lab, 'cleaned_tweet']:
            cnt.update(tokens(t))
        words, vals = zip(*cnt.most_common(top_n)) if cnt else ([], [])
        ax.barh(list(words)[::-1], list(vals)[::-1], color=PALETTE.get(lab, '#7f8c8d'))
        ax.set_title(f'{lab}', fontweight='bold')
        ax.set_xlabel('Count')
    fig.suptitle(f'{ds} — Top Words by Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f'{ds}_03_top_words.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(df, ds, out_dir, max_samples=2500):
    """Project tweets with TF-IDF → SVD → t-SNE (2D)."""
    if len(df) < 5:
        print(f'  [skip tsne] {ds}: too few rows')
        return
    sample = df if len(df) <= max_samples else df.sample(max_samples, random_state=42)
    texts = sample['cleaned_tweet'].astype(str).tolist()
    labels = sample['label'].astype(str).tolist()

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        print(f'  [skip tsne] {ds}: vectorizer failed')
        return

    n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
    if n_comp < 2:
        print(f'  [skip tsne] {ds}: not enough features')
        return
    X_red = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)

    perplexity = min(30, max(5, len(sample) // 4))
    emb = TSNE(
        n_components=2, random_state=42, perplexity=perplexity,
        init='pca', learning_rate='auto',
    ).fit_transform(X_red)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in sorted(set(labels)):
        mask = np.array(labels) == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.65,
                   label=lab, c=PALETTE.get(lab, None))
    ax.set_title(f'{ds} — t-SNE Projection (TF-IDF)', fontweight='bold')
    ax.legend(markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / f'{ds}_05_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gold(ds, out_dir):
    path = GOLD / f'{ds}_gold_standard.csv'
    if not path.exists():
        return
    gold = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = gold['label'].value_counts()
    colors = [PALETTE.get(l, '#7f8c8d') for l in counts.index]
    axes[0].bar(counts.index.astype(str), counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Gold label counts', fontweight='bold')
    for split_name, color in [('train', '#3498db'), ('val', '#9b59b6'), ('test', '#e67e22')]:
        sp = GOLD / f'{ds}_{split_name}.csv'
        if sp.exists():
            n = len(pd.read_csv(sp))
            axes[1].bar(split_name, n, color=color, edgecolor='black')
            axes[1].text(split_name, n, f'{n:,}', ha='center', va='bottom')
    axes[1].set_title('Train / Val / Test sizes', fontweight='bold')
    fig.suptitle(f'{ds} — Gold Standard', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f'{ds}_06_gold_standard.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_one(ds: str):
    clean = PROC / f'{ds}_clean.csv'
    if not clean.exists():
        print(f'[skip] missing {clean}')
        return
    print(f'Plotting {ds}...')
    out_dir = FIGS / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean)
    if 'word_count' not in df.columns:
        df['word_count'] = df['cleaned_tweet'].astype(str).str.split().str.len()
    plot_label_dist(df, ds, out_dir)
    plot_tweet_length(df, ds, out_dir)
    plot_top_words(df, ds, out_dir)
    plot_tsne(df, ds, out_dir)
    plot_gold(ds, out_dir)
    print(f'  saved under {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()
    targets = [args.dataset] if args.dataset else EVENTS
    for ds in targets:
        run_one(ds)
    print('All event plots done.')


if __name__ == '__main__':
    main()
