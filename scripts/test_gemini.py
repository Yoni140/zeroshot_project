"""
Smoke test for the Gemini API key and the gemini_flash model config.
Sends one real classification request and prints the parsed result.
Usage: python scripts/test_gemini.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from openai import OpenAI
from run_all_cloud import (
    DATASET_CONFIG, MODEL_CONFIG, PROVIDER_CONFIG,
    build_prompt, call_api, parse_response,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    model_cfg = MODEL_CONFIG['gemini_flash']
    prov_cfg  = PROVIDER_CONFIG[model_cfg['provider']]
    api_key   = os.environ.get(prov_cfg['api_key_env'], '')
    if not api_key:
        sys.exit(f"ERROR: {prov_cfg['api_key_env']} not set. Add it to .env "
                 f"(free key at aistudio.google.com).")

    client = OpenAI(base_url=prov_cfg['base_url'], api_key=api_key)

    cfg   = DATASET_CONFIG['manchester']
    tweet = ("BREAKING: police confirm the Manchester Arena explosion was caused "
             "by a suicide bomber. 22 dead, dozens injured.")

    print(f"Model : {model_cfg['model_id']}")
    print(f"Tweet : {tweet}\n")

    raw = call_api(client, model_cfg, build_prompt(tweet, cfg))
    if not raw:
        sys.exit('FAILED: empty response — check the key and quota in AI Studio.')

    result = parse_response(raw, cfg)
    print(f"Raw response:\n{raw}\n")
    print(f"Parsed label     : {result['label']}")
    print(f"Confidence       : {result['confidence']}")
    print(f"Parse method     : {result['parse_method']}")

    if result['parse_method'] == 'json' and result['label'] is not None:
        print('\nOK — Gemini is ready. Run:')
        print('  python scripts/run_all_cloud.py --model gemini_flash')
    else:
        print('\nWARNING: response did not parse as clean JSON — inspect the raw output above.')


if __name__ == '__main__':
    main()
