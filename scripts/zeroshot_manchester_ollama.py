"""
Run Ollama zero-shot classification on Manchester test set.
Requires: ollama serve  (with llama3.1 pulled)
Run from project root: python scripts/zeroshot_manchester_ollama.py
"""
import json, re, time, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

warnings.filterwarnings('ignore')

DATASET      = 'manchester'
OLLAMA_MODEL = 'llama3.1'
OLLAMA_URL   = 'http://localhost:11434/api/generate'
OLLAMA_TIMEOUT = 120

CFG = {
    'test':        'data/gold_standard/manchester_test.csv',
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
}
LABEL_MAP = CFG['label_map']

ROOT      = Path(__file__).parent.parent
PREDS_DIR = ROOT / 'results' / 'predictions'
FIGS_DIR  = ROOT / 'results' / 'figures' / 'manchester'
for d in [PREDS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def check_ollama():
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        print(f"Ollama running. Models: {models}")
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"WARNING: '{OLLAMA_MODEL}' not found! Run: ollama pull {OLLAMA_MODEL}")
            return False
        print(f"Model '{OLLAMA_MODEL}' is ready.")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama not running! Start with: ollama serve")
        return False

if not check_ollama():
    raise SystemExit(1)

df_test = pd.read_csv(ROOT / CFG['test'])
print(f"\nTest set: {len(df_test):,} tweets")
print(f"Label distribution: {df_test[CFG['label_col']].value_counts().to_dict()}")

def build_prompt(tweet_text: str) -> str:
    return f"""You are an expert fact-checker and misinformation analyst specializing in social media content.

Your task: Classify the following tweet about {CFG['topic']}.

CLASSES:
- "{CFG['class_a']}": {CFG['class_a_desc']}
- "{CFG['class_b']}": {CFG['class_b_desc']}

TWEET:
\"\"\"{tweet_text}\"\"\"

INSTRUCTIONS:
Think step-by-step before classifying. Consider:
1. What specific claim does the tweet make?
2. Does it present verifiable facts, or unverified/emotional claims?
3. Are there signals of misinformation: conspiracy language, extreme emotion, lack of sources, implausible claims?
4. What is your final classification?

Respond in this exact JSON format (no extra text before or after):
{{
  "reasoning": "<your step-by-step reasoning in 2-4 sentences>",
  "label": "{CFG['class_a']}" or "{CFG['class_b']}",
  "confidence": <float between 0.0 and 1.0>
}}"""

def call_ollama(prompt: str, max_retries: int = 3) -> str:
    payload = {
        'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
        'options': {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': 300}
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            return r.json().get('response', '')
        except requests.exceptions.Timeout:
            print(f"  [Timeout] attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1: time.sleep(5 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f"  [Error] {e}")
            if attempt < max_retries - 1: time.sleep(3)
    return ''

def parse_response(response_text: str) -> dict:
    if not response_text:
        return {'label': None, 'confidence': 0.0, 'reasoning': 'Empty response', 'parse_error': True}
    try:
        clean = re.sub(r'```json\s*|```\s*', '', response_text).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            data  = json.loads(match.group())
            label = str(data.get('label', '')).strip().lower()
            if label in CFG['label_map']:
                return {'label': label, 'confidence': float(data.get('confidence', 0.5)),
                        'reasoning': str(data.get('reasoning', '')), 'parse_error': False}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    text_lower = response_text.lower()
    for label_name in sorted(CFG['label_names'], key=len, reverse=True):
        if label_name in text_lower:
            return {'label': label_name, 'confidence': 0.5,
                    'reasoning': response_text[:300], 'parse_error': True}
    return {'label': None, 'confidence': 0.0, 'reasoning': response_text[:300], 'parse_error': True}

results = []
print(f"\nClassifying {len(df_test):,} tweets with Ollama ({OLLAMA_MODEL})...\n")
start_time = time.time()

for i, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Classifying"):
    result = parse_response(call_ollama(build_prompt(row[CFG['text_col']])))
    results.append({
        'index': i, 'text': row[CFG['text_col']], 'true_label': row[CFG['label_col']],
        'pred_label': result['label'], 'confidence': result['confidence'],
        'reasoning': result['reasoning'], 'parse_error': result['parse_error'],
    })

elapsed    = time.time() - start_time
df_results = pd.DataFrame(results)
null_count = df_results['pred_label'].isnull().sum()
df_results.to_csv(PREDS_DIR / f'{DATASET}_ollama_raw.csv', index=False)
print(f"\nDone! {len(df_results):,} classified in {elapsed/60:.1f} min  |  Nulls: {null_count}")

null_mask = df_results['pred_label'].isnull()
majority_class = df_results['true_label'].mode()[0]
df_results['pred_label_final'] = df_results['pred_label'].fillna(majority_class)
print(f"Prediction distribution:\n{df_results['pred_label_final'].value_counts().to_string()}")

y_true = df_results['true_label'].map(LABEL_MAP).values
y_pred = df_results['pred_label_final'].map(LABEL_MAP).values

print(f"\n{'='*60}\n MANCHESTER - Ollama {OLLAMA_MODEL} Zero-Shot Results\n{'='*60}")
print(classification_report(y_true, y_pred, target_names=CFG['label_names'], digits=4))

pos_label_int = LABEL_MAP[CFG['pos_label']]
metrics = {
    'dataset': DATASET, 'model': f'ollama-zero-shot ({OLLAMA_MODEL})',
    'test_accuracy':        accuracy_score(y_true, y_pred),
    'test_f1_macro':        f1_score(y_true, y_pred, average='macro'),
    'test_f1_weighted':     f1_score(y_true, y_pred, average='weighted'),
    'test_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
    'test_recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
    f'f1_{CFG["pos_label"]}': f1_score(y_true, y_pred, pos_label=pos_label_int, zero_division=0),
    'null_predictions': int(null_mask.sum()),
    'parse_errors':    int(df_results['parse_error'].sum()),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CFG['label_names'], yticklabels=CFG['label_names'], ax=axes[0])
axes[0].set_title('Manchester - Zero-Shot Confusion Matrix (Counts)', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
sns.heatmap(cm.astype(float)/cm.sum(axis=1,keepdims=True), annot=True, fmt='.3f', cmap='Oranges',
            xticklabels=CFG['label_names'], yticklabels=CFG['label_names'], ax=axes[1])
axes[1].set_title('Manchester - Zero-Shot Confusion Matrix (Normalized)', fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_zeroshot_cm.png', dpi=150, bbox_inches='tight')
plt.close()

df_results['correct'] = (df_results['true_label'] == df_results['pred_label_final'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
correct_conf   = df_results[df_results['correct']]['confidence']
incorrect_conf = df_results[~df_results['correct']]['confidence']
axes[0].hist(correct_conf, bins=20, alpha=0.6, color='steelblue', label=f'Correct (n={len(correct_conf)})')
axes[0].hist(incorrect_conf, bins=20, alpha=0.6, color='crimson', label=f'Incorrect (n={len(incorrect_conf)})')
axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
axes[0].set_title('Manchester - Confidence by Outcome', fontweight='bold'); axes[0].legend()
for label_name in CFG['label_names']:
    subset = df_results[df_results['pred_label_final'] == label_name]['confidence']
    axes[1].hist(subset, bins=20, alpha=0.6, label=f'{label_name} (n={len(subset)})')
axes[1].set_xlabel('Confidence'); axes[1].set_ylabel('Count')
axes[1].set_title('Manchester - Confidence by Predicted Label', fontweight='bold'); axes[1].legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / f'{DATASET}_zeroshot_confidence.png', dpi=150, bbox_inches='tight')
plt.close()

df_results['true_label_int'] = df_results['true_label'].map(LABEL_MAP)
df_results['pred_label_int'] = df_results['pred_label_final'].map(LABEL_MAP)
df_results.to_csv(PREDS_DIR / f'{DATASET}_zeroshot_test_predictions.csv', index=False)
pd.DataFrame([metrics]).to_csv(PREDS_DIR / f'{DATASET}_zeroshot_summary.csv', index=False)

print(f"\n{'='*60}")
print(f" Manchester Zero-Shot complete!")
print(f" Accuracy: {metrics['test_accuracy']:.4f}  F1 Macro: {metrics['test_f1_macro']:.4f}")
pos_key = f'f1_{CFG["pos_label"]}'
print(f" F1 Misinformation: {metrics[pos_key]:.4f}")
print(f"{'='*60}")
