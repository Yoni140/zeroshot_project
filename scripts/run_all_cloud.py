"""
Run all cloud LLM inference combinations (models × 3 datasets).
Providers: Groq (free tier) and Google Gemini (free tier / paid).
Usage: python scripts/run_all_cloud.py
       python scripts/run_all_cloud.py --dataset manchester --model gemini_flash  # single run
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
        'label_map':   {'reliable': 0, 'misinformation': 1, 'unrelated': 2},
        'label_names': ['reliable', 'misinformation', 'unrelated'],
        'pos_label':   'misinformation',
        'topic':       'the 2017 Manchester Arena bombing',
        'classes': {
            'reliable':       'factually accurate, verified, or plausible news about the Manchester Arena bombing',
            'misinformation': 'false, unverified, or misleading claims about the event — rumours, conspiracy theories, or fabricated stories',
            'unrelated':      'the tweet is NOT about the Manchester Arena bombing at all — off-topic or irrelevant content',
        },
    },
    'monkeypox': {
        'test':        ROOT / 'data/gold_standard/monkeypox_test.csv',
        'text_col':    'cleaned_tweet',
        'label_col':   'label',
        'label_map':   {'reliable': 0, 'misinformation': 1, 'unrelated': 2},
        'label_names': ['reliable', 'misinformation', 'unrelated'],
        'pos_label':   'misinformation',
        'topic':       'the 2022 Monkeypox (Mpox) outbreak',
        'classes': {
            'reliable':       'factually accurate health information about Monkeypox symptoms, transmission, or treatment',
            'misinformation': 'false health claims, conspiracy theories, or misleading information about Monkeypox',
            'unrelated':      'the tweet is NOT genuinely about the Monkeypox outbreak — off-topic or irrelevant content',
        },
    },
    'pheme': {
        'test':        ROOT / 'data/gold_standard/pheme_test.csv',
        'text_col':    'cleaned_tweet',
        'label_col':   'label',
        'label_map':   {'not_rumour': 0, 'rumour': 1, 'unrelated': 2},
        'label_names': ['not_rumour', 'rumour', 'unrelated'],
        'pos_label':   'rumour',
        'topic':       'breaking news events (Charlie Hebdo attack 2015, Ferguson unrest 2014)',
        'classes': {
            'not_rumour': 'verified news, factual reporting, or confirmed information about the events',
            'rumour':     'unverified claims, speculation, or information that has not been confirmed by credible sources',
            'unrelated':  'the tweet is NOT about either tracked event (Charlie Hebdo / Ferguson) — off-topic or irrelevant content',
        },
    },
}

# ── Provider configs ──────────────────────────────────────────────────────────
PROVIDER_CONFIG = {
    'groq': {
        'base_url':    'https://api.groq.com/openai/v1',
        'api_key_env': 'GROQ_API_KEY',
    },
    'gemini': {
        # Google's OpenAI-compatible endpoint — same OpenAI SDK, different base_url
        'base_url':    'https://generativelanguage.googleapis.com/v1beta/openai/',
        'api_key_env': 'GEMINI_API_KEY',
    },
}

# ── Model configs ─────────────────────────────────────────────────────────────
# req_delay: min seconds between requests based on provider rate limits
#   Groq free tier — 6,000 TPM limit:
#     qwen3  : /no_think suffix → ~600 tok/req → 6000/600*60 = 600s → 10s min → 12s
#     llama33: ~600 tokens/req  → 6000/600*60 = 6.0s min → 7s
#     gpt_oss: ~600 tokens/req  → 7s
#   Gemini free tier — ~10 RPM: 60/10 = 6s min → 7s
MODEL_CONFIG = {
    'gpt_oss': {
        'provider':     'groq',
        'model_id':     'openai/gpt-oss-120b',
        'display_name': 'GPT-OSS 120B',
        'params_b':     120,
        'req_delay':    7,
    },
    'llama33': {
        'provider':     'groq',
        'model_id':     'llama-3.3-70b-versatile',
        'display_name': 'Llama 3.3 70B',
        'params_b':     70,
        'req_delay':    7,
    },
    'qwen3': {
        # qwen/qwen3-32b was removed from Groq (2026-07) — qwen3.6-27b is its successor.
        # file_key stays 'qwen3' so output filenames remain consistent.
        'provider':     'groq',
        'model_id':     'qwen/qwen3.6-27b',
        'display_name': 'Qwen3.6 27B',
        'params_b':     27,
        'req_delay':    7,
        # /no_think no longer works on qwen3.6; reasoning_effort 'none' fully
        # disables thinking (verified: 7 completion tokens vs 400+ with thinking)
        'request_kwargs': {
            'extra_body': {'reasoning_effort': 'none'},
        },
    },
    'gemini_flash': {
        'provider':     'gemini',
        'model_id':     'gemini-2.5-flash',
        'display_name': 'Gemini 2.5 Flash',
        'params_b':     None,  # proprietary — parameter count not public
        'req_delay':    7,
        # 2.5 Flash thinks by default; a thinking run burns tokens inside max_tokens
        # and can truncate the JSON answer — disable it for classification
        # Note the double nesting: the OpenAI SDK's outer extra_body merges into the
        # request JSON, and Google's endpoint expects a field literally named
        # "extra_body" containing the "google" config
        'request_kwargs': {
            'extra_body': {'extra_body': {'google': {'thinking_config': {'thinking_budget': 0}}}},
        },
    },
}

CHECKPOINT_EVERY  = 50
API_TIMEOUT       = 60
MAX_RETRIES       = 6
MAX_TOKENS        = 800   # extra headroom for qwen3 <think> tokens
SYSTEM_PROMPT    = (
    "You are an expert fact-checker and misinformation analyst specializing in "
    "social media content. Always respond with valid JSON only — no extra text before or after."
)


# ── API helpers ───────────────────────────────────────────────────────────────

def build_prompt(tweet: str, cfg: dict) -> str:
    class_lines    = "\n".join(f'- "{name}": {desc}' for name, desc in cfg['classes'].items())
    label_options  = " or ".join(f'"{name}"' for name in cfg['classes'])
    return (
        f"Your task: Classify the following tweet about {cfg['topic']}.\n\n"
        f"CLASSES:\n"
        f"{class_lines}\n\n"
        f"TWEET:\n\"\"\"{tweet}\"\"\"\n\n"
        f"INSTRUCTIONS:\nThink step-by-step before classifying. Consider:\n"
        f"1. Is the tweet actually about {cfg['topic']}, or is it off-topic/unrelated?\n"
        f"2. If on-topic: what specific claim does the tweet make?\n"
        f"3. Does it present verifiable facts, or unverified/emotional claims?\n"
        f"4. Are there signals of misinformation: conspiracy language, extreme emotion, "
        f"lack of sources, implausible claims?\n"
        f"5. What is your final classification?\n\n"
        f"Respond in this exact JSON format (no extra text before or after):\n"
        f"{{\n"
        f"  \"reasoning\": \"<your step-by-step reasoning in 2-4 sentences>\",\n"
        f"  \"label\": {label_options},\n"
        f"  \"confidence\": <float between 0.0 and 1.0>\n"
        f"}}"
    )


class DailyQuotaExceeded(Exception):
    """Provider's per-day quota is exhausted — no point retrying until tomorrow."""


def call_api(client, model_cfg: dict, prompt: str) -> str:
    model_id = model_cfg['model_id']
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
                **model_cfg.get('request_kwargs', {}),
            )
            return resp.choices[0].message.content or ''
        except RateLimitError as e:
            err_str = str(e)
            # Gemini daily quota (camelCase metric names, resets midnight Pacific):
            # retrying in-loop just burns the remaining rows as nulls — stop and let
            # the checkpoint resume tomorrow. Groq's lowercase "per day (TPD/RPD)"
            # limits are rolling 24h windows, so those crawl with parsed waits instead.
            if re.search(r'PerDay', err_str):
                raise DailyQuotaExceeded(err_str[:200])
            # Parse Groq's "try again in XhYmZ.Zs", Gemini's retryDelay, or retry_after_seconds
            m = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s', err_str, re.IGNORECASE)
            if m:
                wait = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60
                        + float(m.group(3) or 0) + 5)
            else:
                m2 = re.search(r'(?:retry_after_seconds|retryDelay)\D*?([\d.]+)', err_str)
                wait = int(float(m2.group(1))) + 5 if m2 else 65
            wait = int(wait)
            print(f'  [RateLimit] waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...', flush=True)
            time.sleep(wait)
        except APITimeoutError:
            print(f'  [Timeout] attempt {attempt+1}/{MAX_RETRIES}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(10)
        except Exception as e:
            print(f'  [Error] {e}', flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
    return ''


def parse_response(text: str, cfg: dict) -> dict:
    if not text or not text.strip():
        return {'label': None, 'confidence': 0.5, 'reasoning': 'Empty response',
                'parse_error': True, 'parse_method': 'empty'}
    try:
        # Strip <think>...</think> blocks (Qwen3 reasoning mode)
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        clean = re.sub(r'```json\s*|```\s*', '', clean).strip()
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
    model_id  = model_cfg['model_id']

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
    consec_empty = 0
    MAX_CONSEC_EMPTY = 8  # sustained failures → stop cleanly instead of null-filling the dataset

    for n, (i, row) in enumerate(df_todo.iterrows(), start=1):
        tweet      = row[cfg['text_col']]
        true_label = row[cfg['label_col']]

        t0     = time.time()
        prompt = build_prompt(tweet, cfg) + model_cfg.get('prompt_suffix', '')
        try:
            raw = call_api(client, model_cfg, prompt)
        except DailyQuotaExceeded as e:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            done = len(done_indices) + n - 1
            print(f'\n  [DailyQuota] provider daily quota exhausted after {done:,}/{total:,} rows.', flush=True)
            print(f'  Checkpoint saved: {checkpoint_path.name}', flush=True)
            print(f'  Re-run the same command after the quota resets (midnight Pacific) to resume.', flush=True)
            print(f'  Details: {e}', flush=True)
            raise
        if raw:
            consec_empty = 0
        else:
            consec_empty += 1
            if consec_empty >= MAX_CONSEC_EMPTY:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                print(f'\n  [Abort] {consec_empty} consecutive empty responses — API is failing.', flush=True)
                print(f'  Checkpoint saved: {checkpoint_path.name}. Re-run to resume.', flush=True)
                raise DailyQuotaExceeded('consecutive empty responses')
        result  = parse_response(raw, cfg)
        elapsed = time.time() - t0
        # Pace requests to stay within Groq TPM limits (model-specific delay)
        req_delay = model_cfg.get('req_delay', 7)
        if elapsed < req_delay:
            time.sleep(req_delay - elapsed)
        req_times.append(time.time() - t0)  # includes sleep → accurate ETA

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
        'provider': model_cfg['provider'],
        'model_id': model_id,
        'test_accuracy':    accuracy_score(y_true, y_pred),
        'test_f1_macro':    f1_score(y_true, y_pred, average='macro'),
        'test_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'test_precision':   precision_score(y_true, y_pred, average='macro', zero_division=0),
        'test_recall':      recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'test_f1_{cfg["pos_label"]}': f1_score(y_true, y_pred, labels=[pos_int], average='macro', zero_division=0),
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

    # Build combo list: filter by --dataset and/or --model if provided
    all_datasets = list(DATASET_CONFIG.keys())
    all_models   = list(MODEL_CONFIG.keys())
    datasets     = [args.dataset] if args.dataset else all_datasets
    models       = [args.model]   if args.model   else all_models
    combos       = [(ds, m) for ds in datasets for m in models]

    # One client per provider — only for providers actually used in this run
    providers_needed = {MODEL_CONFIG[m]['provider'] for _, m in combos}
    clients = {}
    for prov in providers_needed:
        pcfg    = PROVIDER_CONFIG[prov]
        api_key = os.environ.get(pcfg['api_key_env'], '')
        if not api_key:
            sys.exit(f"ERROR: {pcfg['api_key_env']} not set. Add it to .env or set the env var.")
        clients[prov] = OpenAI(base_url=pcfg['base_url'], api_key=api_key)

    total_combos = len(combos)
    all_metrics  = []
    session_start = time.time()

    print(f'\nRunning {total_combos} combination(s):', flush=True)
    for ds, m in combos:
        print(f'  - {ds} + {m}', flush=True)

    quota_exhausted = set()  # providers whose daily quota ran out this session

    for idx, (ds, m) in enumerate(combos, 1):
        print(f'\n[{idx}/{total_combos}]', flush=True)

        provider = MODEL_CONFIG[m]['provider']
        if provider in quota_exhausted:
            print(f'  Skipping {ds}+{m} — {provider} daily quota exhausted, resume tomorrow', flush=True)
            continue

        # Skip if already done
        summary_path = PREDS_DIR / f'{ds}_{m}_summary.csv'
        if summary_path.exists():
            print(f'  Already done — skipping (delete {summary_path.name} to re-run)', flush=True)
            existing = pd.read_csv(summary_path).iloc[0].to_dict()
            all_metrics.append(existing)
            continue

        try:
            m_result = run_inference(ds, m, clients[provider])
            all_metrics.append(m_result)
        except DailyQuotaExceeded:
            quota_exhausted.add(provider)
            continue
        except Exception as e:
            print(f'  ERROR on {ds}+{m}: {e}', flush=True)

        # Cool down between datasets to let the Groq TPM rolling window clear.
        # Without this, the last burst of tweets from one dataset fills the window
        # and the next dataset starts into an immediate rate-limit storm.
        if idx < total_combos:
            cooldown = 90  # seconds — enough for a 60s TPM window to fully refresh
            print(f'\n  [cooldown] sleeping {cooldown}s before next dataset...', flush=True)
            time.sleep(cooldown)

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
