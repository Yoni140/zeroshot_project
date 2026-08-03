"""
run_all_transformers.py — train RoBERTa / BERTweet / MiniLM / ModernBERT
across Manchester, Monkeypox, PHEME, and all PHEME event datasets.

Usage:
  python scripts/run_all_transformers.py
  python scripts/run_all_transformers.py --models bertweet minilm modernbert
  python scripts/run_all_transformers.py --datasets charliehebdo sydneysiege --models roberta
  python scripts/run_all_transformers.py --skip-if-done
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from train_transformer import MODELS, train_one  # noqa: E402

CORE_DATASETS = ['manchester', 'monkeypox', 'pheme']
EVENT_DATASETS = [
    'pheme_all_events', 'gurlitt', 'germanwings-crash', 'ebola-essien',
    'charliehebdo', 'ferguson', 'ottawashooting', 'prince-toronto',
    'putinmissing', 'sydneysiege',
]
ALL_DATASETS = CORE_DATASETS + EVENT_DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--models', nargs='+', default=None, choices=list(MODELS))
    parser.add_argument('--events-only', action='store_true',
                        help='Only PHEME event datasets (skip manchester/monkeypox/pheme)')
    parser.add_argument('--new-models-only', action='store_true',
                        help='Only bertweet/minilm/modernbert (skip roberta)')
    parser.add_argument('--skip-if-done', action='store_true')
    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None:
        datasets = EVENT_DATASETS if args.events_only else ALL_DATASETS

    models = args.models
    if models is None:
        models = ['bertweet', 'minilm', 'modernbert'] if args.new_models_only else list(MODELS)

    print(f'Datasets ({len(datasets)}): {datasets}')
    print(f'Models   ({len(models)}): {models}')

    failures = []
    for ds in datasets:
        for mk in models:
            try:
                train_one(ds, mk, skip_if_done=args.skip_if_done)
            except Exception as e:
                failures.append((ds, mk, str(e)))
                print(f'\n[FAIL] {ds} / {mk}: {e}')
                traceback.print_exc()

    print('\n' + '=' * 60)
    if failures:
        print(f'Finished with {len(failures)} failure(s):')
        for ds, mk, err in failures:
            print(f'  - {ds} / {mk}: {err}')
    else:
        print('All training jobs finished successfully.')
    print('=' * 60)


if __name__ == '__main__':
    main()
