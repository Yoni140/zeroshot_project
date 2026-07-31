"""Test each Groq model with a full-length inference prompt to diagnose rate limits."""
import os, time, re
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, RateLimitError

api_key = os.environ.get('GROQ_API_KEY', '')
client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=api_key)

SYSTEM = ("You are an expert fact-checker and misinformation analyst specializing in "
          "social media content. Always respond with valid JSON only.")

# Full-length prompt similar to actual inference
PROMPT = """Your task: Classify the following tweet about the 2017 Manchester Arena bombing.

CLASSES:
- "reliable": factually accurate, verified, or plausible news about the event
- "misinformation": false, unverified, or misleading claims — rumours, conspiracy theories, or fabricated stories

TWEET:
"Police confirm 22 dead after explosion at Manchester Arena following Ariana Grande concert. Terrorism suspected."

INSTRUCTIONS:
Think step-by-step before classifying. Consider:
1. What specific claim does the tweet make?
2. Does it present verifiable facts, or unverified/emotional claims?
3. Are there signals of misinformation: conspiracy language, extreme emotion, lack of sources, implausible claims?
4. What is your final classification?

Respond in this exact JSON format (no extra text before or after):
{
  "reasoning": "<your step-by-step reasoning in 2-4 sentences>",
  "label": "reliable" or "misinformation",
  "confidence": <float between 0.0 and 1.0>
}"""

MODELS = {
    'gpt_oss':  'openai/gpt-oss-120b',
    'llama33':  'llama-3.3-70b-versatile',
    'qwen3':    'qwen/qwen3-32b',
}

for key, model_id in MODELS.items():
    print(f'\nTesting {key} ({model_id})...', flush=True)
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user',   'content': PROMPT}],
            temperature=0.0, max_tokens=800, timeout=30
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content or ''
        print(f'  [OK] {elapsed:.1f}s | tokens_used={getattr(resp.usage, "total_tokens", "?")}')
        print(f'  Response: {content[:120]}')
    except RateLimitError as e:
        err_str = str(e)
        m = re.search(r'try again in ([\d.]+)s', err_str, re.IGNORECASE)
        retry = m.group(1) if m else '?'
        print(f'  [RateLimit] retry_after={retry}s')
        print(f'  Full error: {err_str[:300]}')
    except Exception as ex:
        print(f'  [FAIL] {type(ex).__name__}: {ex}')
