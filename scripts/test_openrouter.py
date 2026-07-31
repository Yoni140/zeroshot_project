"""Test all 3 planned free models on OpenRouter (with proper retry)."""
import os, sys, time
from openai import OpenAI, RateLimitError

api_key = os.environ.get('OPENROUTER_API_KEY', '')
if not api_key:
    sys.exit('ERROR: OPENROUTER_API_KEY not set.')

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=api_key,
    default_headers={
        'HTTP-Referer': 'https://github.com/research',
        'X-Title': 'ZeroShot Research'
    }
)

models_to_test = [
    ('llama33',   'meta-llama/llama-3.3-70b-instruct:free'),
    ('deepseek',  'deepseek/deepseek-v4-flash:free'),
    ('qwen',      'qwen/qwen3-next-80b-a3b-instruct:free'),
]

for key, model_id in models_to_test:
    print(f'\nTesting {key}: {model_id}')
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'system', 'content': 'You are a helpful assistant. Respond with valid JSON only.'},
                          {'role': 'user',   'content': 'Reply with: {"label":"reliable","confidence":0.9,"reasoning":"test"}'}],
                temperature=0.0,
                max_tokens=80,
                timeout=40,
            )
            content = resp.choices[0].message.content
            finish = resp.choices[0].finish_reason
            print(f'  [OK] attempt {attempt+1}: {repr(content[:100])} (finish={finish})')
            break
        except RateLimitError as e:
            # Try to extract retry-after from the error
            retry_after = 35
            try:
                import re
                m = re.search(r'"retry_after_seconds"\s*:\s*([\d.]+)', str(e))
                if m:
                    retry_after = int(float(m.group(1))) + 5
            except Exception:
                pass
            print(f'  [RateLimit] attempt {attempt+1}: waiting {retry_after}s...')
            time.sleep(retry_after)
        except Exception as ex:
            print(f'  [FAIL] attempt {attempt+1}: {type(ex).__name__}: {ex}')
            break
    else:
        print(f'  [EXHAUSTED] all 5 attempts failed')
