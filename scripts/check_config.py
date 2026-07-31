"""Quick config sanity check — no API calls."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Parse the module without running main()
src = open('scripts/run_all_cloud.py').read()
exec(compile(src.split("def main():")[0], 'run_all_cloud.py', 'exec'))

print('MODEL_CONFIG keys:', list(MODEL_CONFIG.keys()))
print('Groq IDs:')
for k, v in MODEL_CONFIG.items():
    print(f'  {k}: {v["groq_id"]}')
print()
print('DATASET_CONFIG keys:', list(DATASET_CONFIG.keys()))
print()
print('Combos that would run (all 9):')
for ds in DATASET_CONFIG:
    for m in MODEL_CONFIG:
        print(f'  {ds} + {m}')
