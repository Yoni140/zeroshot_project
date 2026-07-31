"""List all available Groq models using openai client."""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('OPENROUTER_API_KEY', '')
client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=api_key)
models = sorted(client.models.list().data, key=lambda m: m.id)
print(f"Available Groq models ({len(models)} total):\n")
for m in models:
    print(f"  {m.id}")
