"""Read completed summaries and checkpoints."""
import pandas as pd, warnings, glob
from pathlib import Path
warnings.filterwarnings('ignore')

PREDS = Path('C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/predictions')

# Completed summaries
print('=== COMPLETED SUMMARIES ===')
for f in sorted(PREDS.glob('*_summary.csv')):
    if any(m in f.name for m in ['gpt_oss','llama33','qwen3']):
        s = pd.read_csv(f).iloc[0]
        ds = s.get('dataset', f.name.split('_')[0])
        mdl = s.get('model', '?')
        f1 = s.get('test_f1_macro', float('nan'))
        acc = s.get('test_accuracy', float('nan'))
        nulls = s.get('null_predictions', '?')
        pjson = s.get('parse_json', '?')
        print(f'  {ds:12s} + {mdl:10s} | F1={f1:.4f} | Acc={acc:.4f} | nulls={nulls} | json_parses={pjson}')

# In-progress checkpoints
print('\n=== CHECKPOINTS (in progress) ===')
for f in sorted(PREDS.glob('*_checkpoint.csv')):
    df = pd.read_csv(f)
    print(f'  {f.name}: {len(df)} rows | parse_methods: {dict(df["parse_method"].value_counts())} | nulls={df["pred_label"].isna().sum()}')
