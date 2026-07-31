"""Test one tweet with each of the 3 Groq models."""
import os, time
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError

api_key = os.environ.get('GROQ_API_KEY', '')
client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=api_key)

MODELS = {
    'gpt_oss':  'openai/gpt-oss-120b',
    'llama33':  'llama-3.3-70b-versatile',
    'qwen3':    'qwen/qwen3-32b',
}

SYSTEM = "You are an expert fact-checker. Always respond with valid JSON only."
PROMPT = (
    'Classify this tweet about the 2017 Manchester Arena bombing.\n'
    'CLASSES: "reliable" or "misinformation"\n'
    'TWEET: "Praying for the victims in Manchester. Stay safe everyone."\n'
    'Respond: {"label":"reliable","confidence":0.9,"reasoning":"..."}'
)

for key, model_id in MODELS.items():
    print(f'\nTesting {key} ({model_id})...', flush=True)
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user',   'content': PROMPT}],
            temperature=0.0, max_tokens=120, timeout=30
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content
        print(f'  [OK] {elapsed:.1f}s : {repr(content[:150])}')
    except RateLimitError as e:
        print(f'  [RateLimit] {e}')
    except Exception as e:
        print(f'  [FAIL] {type(e).__name__}: {e}')
