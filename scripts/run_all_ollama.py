"""
Run Ollama zero-shot on all datasets (or only those still missing).

Mirrors scripts/run_all_cloud.py dataset/prompt config, but calls local Ollama.

Usage:
  python scripts/run_all_ollama.py --model llama3.1          # missing zeroshot runs
  python scripts/run_all_ollama.py --model llama3.3          # missing llama33 runs
  python scripts/run_all_ollama.py --model llama3.3 --events-only
  python scripts/run_all_ollama.py --dataset charliehebdo --model llama3.3
  python scripts/run_all_ollama.py --force                   # re-run even if summary exists
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PREDS_DIR = ROOT / 'results' / 'predictions'
PREDS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_TAGS = 'http://localhost:11434/api/tags'
CHECKPOINT_EVERY = 50
MAX_RETRIES = 3

# file_key: summary/predictions suffix used by compare_all_models.py
# artifact_key: raw + checkpoint filename stem (llama3.1 keeps legacy 'ollama')
MODEL_PRESETS = {
    'llama3.1': {
        'ollama_model': 'llama3.1:8b',
        'file_key': 'zeroshot',
        'artifact_key': 'ollama',
        'timeout': 120,
        'num_predict': 300,
    },
    'llama3.3': {
        'ollama_model': 'llama3.3',
        'file_key': 'llama33',
        'artifact_key': 'llama33',
        'timeout': 300,
        'num_predict': 400,
    },
}

# Active model settings — set in main() from --model
OLLAMA_MODEL = MODEL_PRESETS['llama3.1']['ollama_model']
OLLAMA_TIMEOUT = MODEL_PRESETS['llama3.1']['timeout']
NUM_PREDICT = MODEL_PRESETS['llama3.1']['num_predict']
FILE_KEY = MODEL_PRESETS['llama3.1']['file_key']
ARTIFACT_KEY = MODEL_PRESETS['llama3.1']['artifact_key']

# ── Dataset configs (same as run_all_cloud.py) ───────────────────────────────
DATASET_CONFIG = {
    'manchester': {
        'test': ROOT / 'data/gold_standard/manchester_test.csv',
        'text_col': 'cleaned_tweet',
        'label_col': 'label',
        'label_map': {'reliable': 0, 'misinformation': 1, 'unrelated': 2},
        'label_names': ['reliable', 'misinformation', 'unrelated'],
        'pos_label': 'misinformation',
        'topic': 'the 2017 Manchester Arena bombing',
        'classes': {
            'reliable': 'factually accurate, verified, or plausible news about the Manchester Arena bombing',
            'misinformation': 'false, unverified, or misleading claims about the event — rumours, conspiracy theories, or fabricated stories',
            'unrelated': 'the tweet is NOT about the Manchester Arena bombing at all — off-topic or irrelevant content',
        },
    },
    'monkeypox': {
        'test': ROOT / 'data/gold_standard/monkeypox_test.csv',
        'text_col': 'cleaned_tweet',
        'label_col': 'label',
        'label_map': {'reliable': 0, 'misinformation': 1, 'unrelated': 2},
        'label_names': ['reliable', 'misinformation', 'unrelated'],
        'pos_label': 'misinformation',
        'topic': 'the 2022 Monkeypox (Mpox) outbreak',
        'classes': {
            'reliable': 'factually accurate health information about Monkeypox symptoms, transmission, or treatment',
            'misinformation': 'false health claims, conspiracy theories, or misleading information about Monkeypox',
            'unrelated': 'the tweet is NOT genuinely about the Monkeypox outbreak — off-topic or irrelevant content',
        },
    },
    'pheme': {
        'test': ROOT / 'data/gold_standard/pheme_test.csv',
        'text_col': 'cleaned_tweet',
        'label_col': 'label',
        'label_map': {'not_rumour': 0, 'rumour': 1, 'unrelated': 2},
        'label_names': ['not_rumour', 'rumour', 'unrelated'],
        'pos_label': 'rumour',
        'topic': 'breaking news events (Charlie Hebdo attack 2015, Ferguson unrest 2014)',
        'classes': {
            'not_rumour': 'verified news, factual reporting, or confirmed information about the events',
            'rumour': 'unverified claims, speculation, or information that has not been confirmed by credible sources',
            'unrelated': 'the tweet is NOT about either tracked event (Charlie Hebdo / Ferguson) — off-topic or irrelevant content',
        },
    },
}

_EVENT_META = {
    'pheme_all_events': 'breaking news events in the PHEME rumour corpus (multiple crises)',
    'gurlitt': 'the Cornelius Gurlitt Nazi-looted art / museum bequest case',
    'germanwings-crash': 'the 2015 Germanwings Flight 9525 crash',
    'charliehebdo': 'the 2015 Charlie Hebdo attack in Paris',
    'ferguson': 'the 2014 Ferguson unrest after the shooting of Michael Brown',
    'ottawashooting': 'the 2014 Ottawa Parliament Hill / War Memorial shooting',
    'prince-toronto': 'rumours of a secret Prince concert in Toronto',
    'putinmissing': 'rumours that Vladimir Putin was missing, ill, or dead',
    'sydneysiege': 'the 2014 Sydney Lindt café siege in Martin Place',
}

for _key, _topic in _EVENT_META.items():
    DATASET_CONFIG[_key] = {
        'test': ROOT / f'data/gold_standard/{_key}_test.csv',
        'text_col': 'cleaned_tweet',
        'label_col': 'label',
        'label_map': {'not_rumour': 0, 'rumour': 1},
        'label_names': ['not_rumour', 'rumour'],
        'pos_label': 'rumour',
        'topic': _topic,
        'classes': {
            'not_rumour': f'verified news, factual reporting, or confirmed information about {_topic}',
            'rumour': (
                f'unverified claims, speculation, or information about {_topic} '
                f'that has not been confirmed by credible sources'
            ),
        },
    }

EVENT_DATASETS = list(_EVENT_META.keys())


def check_ollama() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        print(f'Ollama running. Models: {models}', flush=True)
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"ERROR: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}", flush=True)
            return False
        print(f"Model '{OLLAMA_MODEL}' is ready.", flush=True)
        return True
    except requests.exceptions.ConnectionError:
        print('ERROR: Ollama not running. Start with: ollama serve', flush=True)
        return False


def build_prompt(tweet: str, cfg: dict) -> str:
    class_lines = '\n'.join(f'- "{name}": {desc}' for name, desc in cfg['classes'].items())
    label_options = ' or '.join(f'"{name}"' for name in cfg['classes'])
    return f"""You are an expert fact-checker and misinformation analyst specializing in social media content.

Your task: Classify the following tweet about {cfg['topic']}.

CLASSES:
{class_lines}

TWEET:
\"\"\"{tweet}\"\"\"

INSTRUCTIONS:
Think step-by-step before classifying. Consider:
1. Is the tweet actually about {cfg['topic']}, or is it off-topic/unrelated?
2. If on-topic: what specific claim does the tweet make?
3. Does it present verifiable facts, or unverified/emotional claims?
4. Are there signals of rumour/misinformation: unconfirmed reports, speculation, conspiracy language, extreme emotion, lack of sources, implausible claims?
5. What is your final classification?

Respond in this exact JSON format (no extra text before or after):
{{
  "reasoning": "<your step-by-step reasoning in 2-4 sentences>",
  "label": {label_options},
  "confidence": <float between 0.0 and 1.0>
}}"""


def call_ollama(prompt: str) -> str:
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': NUM_PREDICT,
        },
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            raw = r.json().get('response', '')
            return raw if raw is not None else ''
        except requests.exceptions.Timeout:
            print(f'  [Timeout] attempt {attempt + 1}/{MAX_RETRIES}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f'  [Error] {e}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
    return ''


def parse_response(text: str, cfg: dict) -> dict:
    if not text or not text.strip():
        return {
            'label': None, 'confidence': 0.5, 'reasoning': 'Empty response',
            'parse_error': True, 'parse_method': 'empty',
        }
    try:
        clean = re.sub(r'```json\s*|```\s*', '', text).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            data = json.loads(m.group())
            label = str(data.get('label', '')).strip().lower()
            if label in cfg['label_map']:
                return {
                    'label': label,
                    'confidence': float(data.get('confidence', 0.5)),
                    'reasoning': str(data.get('reasoning', '')),
                    'parse_error': False,
                    'parse_method': 'json',
                }
    except Exception:
        pass
    for lname in sorted(cfg['label_names'], key=len, reverse=True):
        if lname in text.lower():
            return {
                'label': lname, 'confidence': 0.5, 'reasoning': text[:300],
                'parse_error': True, 'parse_method': 'keyword_fallback',
            }
    return {
        'label': None, 'confidence': 0.0, 'reasoning': text[:300],
        'parse_error': True, 'parse_method': 'default',
    }


def summary_path(dataset: str) -> Path:
    return PREDS_DIR / f'{dataset}_{FILE_KEY}_summary.csv'


def run_inference(dataset: str) -> dict:
    cfg = DATASET_CONFIG[dataset]
    label_map = cfg['label_map']

    print(f'\n{"=" * 60}', flush=True)
    print(f'  Dataset : {dataset}', flush=True)
    print(f'  Model   : ollama-zero-shot ({OLLAMA_MODEL})', flush=True)
    print(f'{"=" * 60}', flush=True)

    df_test = pd.read_csv(cfg['test'])
    df_test.dropna(subset=[cfg['text_col'], cfg['label_col']], inplace=True)
    df_test[cfg['text_col']] = df_test[cfg['text_col']].astype(str)
    print(f'  Test set: {len(df_test):,} samples', flush=True)
    print(f'  Labels  : {df_test[cfg["label_col"]].value_counts().to_dict()}', flush=True)

    raw_path = PREDS_DIR / f'{dataset}_{ARTIFACT_KEY}_raw.csv'
    checkpoint_path = PREDS_DIR / f'{dataset}_{ARTIFACT_KEY}_checkpoint.csv'

    done_indices, results = set(), []
    if checkpoint_path.exists():
        df_ckpt = pd.read_csv(checkpoint_path)
        done_indices = set(df_ckpt['index'].tolist())
        results = df_ckpt.to_dict('records')
        print(f'  Resuming from checkpoint: {len(done_indices):,} done', flush=True)

    df_todo = df_test[~df_test.index.isin(done_indices)]
    total = len(df_test)
    start_time = time.time()
    req_times = []

    for n, (i, row) in enumerate(df_todo.iterrows(), start=1):
        tweet = row[cfg['text_col']]
        true_label = row[cfg['label_col']]

        t0 = time.time()
        result = parse_response(call_ollama(build_prompt(tweet, cfg)), cfg)
        req_times.append(time.time() - t0)

        results.append({
            'index': i,
            'text': tweet,
            'true_label': true_label,
            'pred_label': result['label'],
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'parse_error': result['parse_error'],
            'parse_method': result['parse_method'],
        })

        processed = len(done_indices) + n
        if n % 10 == 0 or n == len(df_todo):
            avg = sum(req_times) / len(req_times)
            eta = avg * (total - processed)
            pct = processed / total * 100
            print(
                f'  {processed:,}/{total:,} ({pct:.1f}%) | avg {avg:.1f}s | ETA ~{eta / 60:.1f}min',
                flush=True,
            )

        if n % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f'  [Checkpoint saved: {len(results):,} rows]', flush=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(raw_path, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    null_mask = df_res['pred_label'].isnull()
    majority = df_res['true_label'].mode()[0]
    df_res['pred_label_final'] = df_res['pred_label'].fillna(majority)
    df_res.loc[null_mask, 'confidence'] = df_res.loc[null_mask, 'confidence'].replace(0.0, 0.5)

    y_true = df_res['true_label'].map(label_map).values
    y_pred = df_res['pred_label_final'].map(label_map).values
    pos_int = label_map[cfg['pos_label']]
    parse_counts = df_res['parse_method'].value_counts().to_dict()

    metrics = {
        'dataset': dataset,
        'model': FILE_KEY,
        'model_display': f'Ollama {OLLAMA_MODEL}',
        'provider': 'ollama',
        'model_id': OLLAMA_MODEL,
        'test_accuracy': accuracy_score(y_true, y_pred),
        'test_f1_macro': f1_score(y_true, y_pred, average='macro'),
        'test_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'test_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'test_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'test_f1_{cfg["pos_label"]}': f1_score(
            y_true, y_pred, labels=[pos_int], average='macro', zero_division=0
        ),
        'null_predictions': int(null_mask.sum()),
        'parse_errors': int(df_res['parse_error'].sum()),
        'parse_json': parse_counts.get('json', 0),
        'parse_keyword_fallback': parse_counts.get('keyword_fallback', 0),
        'parse_empty': parse_counts.get('empty', 0),
        'parse_default': parse_counts.get('default', 0),
    }

    df_res['true_label_int'] = df_res['true_label'].map(label_map)
    df_res['pred_label_int'] = df_res['pred_label_final'].map(label_map)

    pred_path = PREDS_DIR / f'{dataset}_{FILE_KEY}_test_predictions.csv'
    sum_path = summary_path(dataset)
    df_res.to_csv(pred_path, index=False)
    pd.DataFrame([metrics]).to_csv(sum_path, index=False)

    elapsed = time.time() - start_time
    print(f'\n  DONE in {elapsed / 60:.1f} min', flush=True)
    print(f'  Accuracy : {metrics["test_accuracy"]:.4f}', flush=True)
    print(f'  F1 Macro : {metrics["test_f1_macro"]:.4f}', flush=True)
    print(f'  Nulls    : {null_mask.sum()}', flush=True)
    print(f'  Saved    : {sum_path.name}', flush=True)
    return metrics


def apply_model_preset(preset_name: str) -> None:
    global OLLAMA_MODEL, OLLAMA_TIMEOUT, NUM_PREDICT, FILE_KEY, ARTIFACT_KEY
    preset = MODEL_PRESETS[preset_name]
    OLLAMA_MODEL = preset['ollama_model']
    OLLAMA_TIMEOUT = preset['timeout']
    NUM_PREDICT = preset['num_predict']
    FILE_KEY = preset['file_key']
    ARTIFACT_KEY = preset['artifact_key']


def main():
    global OLLAMA_MODEL, OLLAMA_TIMEOUT, NUM_PREDICT, FILE_KEY, ARTIFACT_KEY

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_PRESETS), default='llama3.1',
                        help='Ollama model preset (default: llama3.1)')
    parser.add_argument('--dataset', choices=list(DATASET_CONFIG), default=None)
    parser.add_argument('--events-only', action='store_true',
                        help='Only PHEME event datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Explicit dataset list')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if summary already exists')
    args = parser.parse_args()

    apply_model_preset(args.model)

    if not check_ollama():
        sys.exit(1)

    if args.datasets:
        datasets = args.datasets
        unknown = [d for d in datasets if d not in DATASET_CONFIG]
        if unknown:
            sys.exit(f'Unknown datasets: {unknown}. Known: {list(DATASET_CONFIG)}')
    elif args.dataset:
        datasets = [args.dataset]
    elif args.events_only:
        datasets = EVENT_DATASETS
    else:
        datasets = list(DATASET_CONFIG.keys())

    # Default: only run missing (unless --force)
    to_run = []
    for ds in datasets:
        test_path = DATASET_CONFIG[ds]['test']
        if not test_path.exists() or len(pd.read_csv(test_path)) == 0:
            print(f'  Skipping {ds} — empty / missing test file', flush=True)
            continue
        if not args.force and summary_path(ds).exists():
            print(f'  Already done — skipping {ds} ({summary_path(ds).name})', flush=True)
            continue
        to_run.append(ds)

    if not to_run:
        print(f'\nNothing to run — all requested datasets already have {FILE_KEY} results.', flush=True)
        return

    total_rows = sum(len(pd.read_csv(DATASET_CONFIG[ds]['test'])) for ds in to_run)
    print(f'\nWill run Ollama {OLLAMA_MODEL} on {len(to_run)} dataset(s) (~{total_rows:,} tweets):', flush=True)
    for ds in to_run:
        n = len(pd.read_csv(DATASET_CONFIG[ds]['test']))
        print(f'  - {ds} ({n} tweets)', flush=True)

    all_metrics = []
    session_start = time.time()

    for idx, ds in enumerate(to_run, 1):
        print(f'\n[{idx}/{len(to_run)}]', flush=True)
        try:
            all_metrics.append(run_inference(ds))
        except Exception as e:
            print(f'  ERROR on {ds}: {e}', flush=True)

    total_elapsed = time.time() - session_start
    print(f'\n\n{"=" * 60}', flush=True)
    print(f'ALL RUNS COMPLETE — {total_elapsed / 60:.1f} min total', flush=True)
    print(f'{"=" * 60}', flush=True)
    print(f'\n{"Dataset":<20} {"F1 Macro":>10} {"Accuracy":>10}', flush=True)
    print('-' * 44, flush=True)
    for m in all_metrics:
        print(
            f'{m.get("dataset", "?"):<20} {m.get("test_f1_macro", float("nan")):>10.4f} '
            f'{m.get("test_accuracy", float("nan")):>10.4f}',
            flush=True,
        )
    print(f'\nOutput files in: {PREDS_DIR}', flush=True)
    print('Next step: python scripts/compare_all_models.py', flush=True)


if __name__ == '__main__':
    main()
