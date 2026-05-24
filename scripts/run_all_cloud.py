"""
Run all 9 cloud LLM inference combinations (3 models × 3 datasets).
Usage: python scripts/run_all_cloud.py
       python scripts/run_all_cloud.py --dataset manchester --model deepseek  # single run
"""

import os, re, sys, time, json, random, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PREDS_DIR = ROOT / 'results' / 'predictions'
PREDS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASET_CONFIG = {
    'manchester': {
        'test':        ROOT / 'data/gold_standard/manchester_test.csv',
        'text_col':    'cleaned_tweet',
        'label_col':   'label',
        'label_map':   {'reliable': 0, 'misinformation': 1},
        'label_names': ['reliable', 'misinformation'],
        'pos_label':   'misinformation',
        'topic':       'the 2017 Manchester Arena bombing',
        'class_a':     'reliable',
        'class_b':     'misinformation',
        'class_a_desc': 'factually accurate, verified, or plausible news about the event',
        'class_b_desc': 'false, unverified, or misleading claims — rumours, conspiracy theories, or fabricated stories',
    },
    'monkeypox': {
        'test':        ROOT / 'data/gold_standard/monkeypox_test.csv',
        'text_col':    'cleaned_tweet',
        'label_col':   'label',
        'label_map':   {'reliable': 0, 'misinformation': 1},
        'label_names': ['reliable', 'misinformation'],
        'pos_label':   'misinformation',
        'topic':       'the 2022 Monkeypox (Mpox) outbreak',
        'class_a':     'reliable',
        'class_b':     'misinformation',
        'class_a_desc': 'factually accurate health information about Monkeypox symptoms, transmission, or treatment',
        'class_b_desc': 'false health claims, conspiracy theories, or misleading information about Monkeypox',
    },
    'pheme': {
        'test':        ROOT / 'data/gold_standard/pheme_test.csv',
        'text_col':    'cleaned_tweet',
        'label_col':   'label',
        'label_map':   {'not_rumour': 0, 'rumour': 1},
        'label_names': ['not_rumour', 'rumour'],
        'pos_label':   'rumour',
        'topic':       'breaking news events (Charlie Hebdo attack 2015, Ferguson unrest 2014)',
        'class_a':     'not_rumour',
        'class_b':     'rumour',
        'class_a_desc': 'verified news, factual reporting, or confirmed information about the events',
        'class_b_desc': 'unverified claims, speculation, or information that has not been confirmed by credible sources',
    },
}

# ── Model configs (all free on OpenRouter) ────────────────────────────────────
MODEL_CONFIG = {
    'deepseek': {
        'openrouter_id': 'deepseek/deepseek-v4-flash:free',
        'display_name':  'DeepSeek V4 Flash (284B)',
        'params_b':      284,
    },
    'qwen': {
        'openrouter_id': 'qwen/qwen3-next-80b-a3b-instruct:free',
        'display_name':  'Qwen3 80B Instruct',
        'params_b':      80,
    },
    'llama33': {
        'openrouter_id': 'meta-llama/llama-3.3-70b-instruct:free',
        'display_name':  'Llama 3.3 70B',
        'params_b':      70,
    },
}

CHECKPOINT_EVERY = 50
API_TIMEOUT      = 60
MAX_RETRIES      = 3
RETRY_BASE       = 5
MAX_TOKENS       = 400
SYSTEM_PROMPT    = (
    "You are an expert fact-checker and misinformation analyst specializing in "
    "social media content. Always respond with valid JSON only — no extra text before or after."
)


# ── API helpers ───────────────────────────────────────────────────────────────

def build_prompt(tweet: str, cfg: dict) -> str:
    return (
        f"Your task: Classify the following tweet about {cfg['topic']}.\n\n"
        f"CLASSES:\n"
        f"- \"{cfg['class_a']}\": {cfg['class_a_desc']}\n"
        f"- \"{cfg['class_b']}\": {cfg['class_b_desc']}\n\n"
        f"TWEET:\n\"\"\"{tweet}\"\"\"\n\n"
        f"INSTRUCTIONS:\nThink step-by-step before classifying. Consider:\n"
        f"1. What specific claim does the tweet make?\n"
        f"2. Does it present verifiable facts, or unverified/emotional claims?\n"
        f"3. Are there signals of misinformation: conspiracy language, extreme emotion, "
        f"lack of sources, implausible claims?\n"
        f"4. What is your final classification?\n\n"
        f"Respond in this exact JSON format (no extra text before or after):\n"
        f"{{\n"
        f"  \"reasoning\": \"<your step-by-step reasoning in 2-4 sentences>\",\n"
        f"  \"label\": \"{cfg['class_a']}\" or \"{cfg['class_b']}\",\n"
        f"  \"confidence\": <float between 0.0 and 1.0>\n"
        f"}}"
    )


def call_api(client, model_id: str, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                timeout=API_TIMEOUT,
            )
            return resp.choices[0].message.content or ''
        except RateLimitError:
            wait = RETRY_BASE * (2 ** attempt)
            print(f'  [RateLimit] waiting {wait}s...', flush=True)
            time.sleep(wait)
        except APITimeoutError:
            print(f'  [Timeout] attempt {attempt+1}/{MAX_RETRIES}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE * (attempt + 1))
        except Exception as e:
            print(f'  [Error] {e}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
    return ''


def parse_response(text: str, cfg: dict) -> dict:
    if not text or not text.strip():
        return {'label': None, 'confidence': 0.5, 'reasoning': 'Empty response',
                'parse_error': True, 'parse_method': 'empty'}
    try:
        clean = re.sub(r'```json\s*|```\s*', '', text).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            data  = json.loads(m.group())
            label = str(data.get('label', '')).strip().lower()
            if label in cfg['label_map']:
                return {'label': label, 'confidence': float(data.get('confidence', 0.5)),
                        'reasoning': str(data.get('reasoning', '')),
                        'parse_error': False, 'parse_method': 'json'}
    except Exception:
        pass
    for lname in sorted(cfg['label_names'], key=len, reverse=True):
        if lname in text.lower():
            return {'label': lname, 'confidence': 0.5, 'reasoning': text[:300],
                    'parse_error': True, 'parse_method': 'keyword_fallback'}
    return {'label': None, 'confidence': 0.0, 'reasoning': text[:300],
            'parse_error': True, 'parse_method': 'default'}


# ── Main inference runner ─────────────────────────────────────────────────────

def run_inference(dataset: str, model: str, client: OpenAI) -> dict:
    cfg       = DATASET_CONFIG[dataset]
    model_cfg = MODEL_CONFIG[model]
    label_map = cfg['label_map']
    model_id  = model_cfg['openrouter_id']

    print(f'\n{"="*60}', flush=True)
    print(f'  Dataset : {dataset.upper()}', flush=True)
    print(f'  Model   : {model_cfg["display_name"]}', flush=True)
    print(f'{"="*60}', flush=True)

    df_test = pd.read_csv(cfg['test'])
    df_test.dropna(subset=[cfg['text_col'], cfg['label_col']], inplace=True)
    df_test[cfg['text_col']] = df_test[cfg['text_col']].astype(str)
    print(f'  Test set: {len(df_test):,} samples', flush=True)

    checkpoint_path = PREDS_DIR / f'{dataset}_{model}_checkpoint.csv'
    raw_path        = PREDS_DIR / f'{dataset}_{model}_raw.csv'

    done_indices, results = set(), []
    if checkpoint_path.exists():
        df_ckpt      = pd.read_csv(checkpoint_path)
        done_indices = set(df_ckpt['index'].tolist())
        results      = df_ckpt.to_dict('records')
        print(f'  Resuming from checkpoint: {len(done_indices):,} done', flush=True)

    df_todo     = df_test[~df_test.index.isin(done_indices)]
    total       = len(df_test)
    start_time  = time.time()
    req_times   = []

    for n, (i, row) in enumerate(df_todo.iterrows(), start=1):
        tweet      = row[cfg['text_col']]
        true_label = row[cfg['label_col']]

        t0     = time.time()
        result = parse_response(call_api(client, model_id, build_prompt(tweet, cfg)), cfg)
        req_times.append(time.time() - t0)

        results.append({'index': i, 'text': tweet, 'true_label': true_label,
                        'pred_label': result['label'], 'confidence': result['confidence'],
                        'reasoning': result['reasoning'], 'parse_error': result['parse_error'],
                        'parse_method': result['parse_method']})

        processed = len(done_indices) + n
        if n % 10 == 0 or n == len(df_todo):
            avg  = sum(req_times) / len(req_times)
            eta  = avg * (total - processed)
            pct  = processed / total * 100
            print(f'  {processed:,}/{total:,} ({pct:.1f}%) | avg {avg:.1f}s | ETA ~{eta/60:.1f}min', flush=True)

        if n % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    df_res = pd.DataFrame(results)
    df_res.to_csv(raw_path, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Fill nulls with majority class
    null_mask      = df_res['pred_label'].isnull()
    majority       = df_res['true_label'].mode()[0]
    df_res['pred_label_final'] = df_res['pred_label'].fillna(majority)
    df_res.loc[null_mask, 'confidence'] = df_res.loc[null_mask, 'confidence'].replace(0.0, 0.5)

    # Metrics
    y_true        = df_res['true_label'].map(label_map).values
    y_pred        = df_res['pred_label_final'].map(label_map).values
    pos_int       = label_map[cfg['pos_label']]
    parse_counts  = df_res['parse_method'].value_counts().to_dict()

    metrics = {
        'dataset': dataset, 'model': model,
        'model_display': model_cfg['display_name'],
        'params_b': model_cfg['params_b'],
        'openrouter_id': model_id,
        'test_accuracy':    accuracy_score(y_true, y_pred),
        'test_f1_macro':    f1_score(y_true, y_pred, average='macro'),
        'test_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'test_precision':   precision_score(y_true, y_pred, average='macro', zero_division=0),
        'test_recall':      recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'test_f1_{cfg["pos_label"]}': f1_score(y_true, y_pred, pos_label=pos_int, zero_division=0),
        'null_predictions': int(null_mask.sum()),
        'parse_errors':     int(df_res['parse_error'].sum()),
        'parse_json':       parse_counts.get('json', 0),
        'parse_keyword_fallback': parse_counts.get('keyword_fallback', 0),
        'parse_empty':      parse_counts.get('empty', 0),
        'parse_default':    parse_counts.get('default', 0),
    }

    df_res['true_label_int'] = df_res['true_label'].map(label_map)
    df_res['pred_label_int'] = df_res['pred_label_final'].map(label_map)

    pred_path    = PREDS_DIR / f'{dataset}_{model}_test_predictions.csv'
    summary_path = PREDS_DIR / f'{dataset}_{model}_summary.csv'
    df_res.to_csv(pred_path, index=False)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    elapsed = time.time() - start_time
    print(f'\n  DONE in {elapsed/60:.1f} min', flush=True)
    print(f'  Accuracy : {metrics["test_accuracy"]:.4f}', flush=True)
    print(f'  F1 Macro : {metrics["test_f1_macro"]:.4f}', flush=True)
    print(f'  Nulls    : {null_mask.sum()}', flush=True)
    print(f'  Saved    : {summary_path.name}', flush=True)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASET_CONFIG), default=None)
    parser.add_argument('--model',   choices=list(MODEL_CONFIG),   default=None)
    args = parser.parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        sys.exit('ERROR: OPENROUTER_API_KEY not set. Add it to .env or set the env var.')

    client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)

    # Single run or all 9
    if args.dataset and args.model:
        combos = [(args.dataset, args.model)]
    else:
        combos = [(ds, m) for ds in DATASET_CONFIG for m in MODEL_CONFIG]

    total_combos = len(combos)
    all_metrics  = []
    session_start = time.time()

    print(f'\nRunning {total_combos} combination(s):', flush=True)
    for ds, m in combos:
        print(f'  - {ds} + {m}', flush=True)

    for idx, (ds, m) in enumerate(combos, 1):
        print(f'\n[{idx}/{total_combos}]', flush=True)

        # Skip if already done
        summary_path = PREDS_DIR / f'{ds}_{m}_summary.csv'
        if summary_path.exists():
            print(f'  Already done — skipping (delete {summary_path.name} to re-run)', flush=True)
            existing = pd.read_csv(summary_path).iloc[0].to_dict()
            all_metrics.append(existing)
            continue

        try:
            m_result = run_inference(ds, m, client)
            all_metrics.append(m_result)
        except Exception as e:
            print(f'  ERROR on {ds}+{m}: {e}', flush=True)

    # Final summary
    total_elapsed = time.time() - session_start
    print(f'\n\n{"="*60}', flush=True)
    print(f'ALL RUNS COMPLETE — {total_elapsed/60:.1f} min total', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'\n{"Dataset":<14} {"Model":<12} {"F1 Macro":>10} {"Accuracy":>10}', flush=True)
    print('-' * 50, flush=True)
    for m in all_metrics:
        ds   = m.get('dataset', '?')
        mdl  = m.get('model', '?')
        f1   = m.get('test_f1_macro', float('nan'))
        acc  = m.get('test_accuracy', float('nan'))
        print(f'{ds:<14} {mdl:<12} {f1:>10.4f} {acc:>10.4f}', flush=True)

    print(f'\nOutput files in: {PREDS_DIR}', flush=True)
    print('Next step: run notebooks/comparison/09_Comparison.ipynb', flush=True)


if __name__ == '__main__':
    main()
