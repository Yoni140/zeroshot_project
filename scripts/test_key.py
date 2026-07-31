"""Detect whether the key is OpenRouter or Groq, then test it."""
import os, time
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('OPENROUTER_API_KEY', '')
print(f'Key prefix: {api_key[:12]}...')

from openai import OpenAI, RateLimitError

# --- Test as Groq ---
print('\n--- Testing as GROQ ---')
try:
    groq_client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=api_key)
    resp = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': 'Say: {"ok":true}'}],
        temperature=0.0, max_tokens=20, timeout=20
    )
    print(f'[GROQ OK] llama-3.3-70b-versatile: {repr(resp.choices[0].message.content)}')
    groq_ok = True
except Exception as e:
    print(f'[GROQ FAIL] {type(e).__name__}: {e}')
    groq_ok = False

# --- Test as OpenRouter ---
print('\n--- Testing as OPENROUTER ---')
try:
    or_client = OpenAI(
        base_url='https://openrouter.ai/api/v1', api_key=api_key,
        default_headers={'HTTP-Referer': 'https://github.com/research', 'X-Title': 'ZeroShot'}
    )
    resp = or_client.chat.completions.create(
        model='meta-llama/llama-3.3-70b-instruct:free',
        messages=[{'role': 'user', 'content': 'Say: {"ok":true}'}],
        temperature=0.0, max_tokens=20, timeout=20
    )
    print(f'[OR OK] llama-3.3-70b-instruct:free: {repr(resp.choices[0].message.content)}')
    or_ok = True
except Exception as e:
    print(f'[OR FAIL] {type(e).__name__}: {e}')
    or_ok = False

print(f'\n=> Groq works: {groq_ok}')
print(f'=> OpenRouter works: {or_ok}')
