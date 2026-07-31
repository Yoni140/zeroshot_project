"""Fetch all free models from OpenRouter and print them."""
import os, sys, urllib.request, json

api_key = os.environ.get('OPENROUTER_API_KEY', '')
if not api_key:
    sys.exit('ERROR: OPENROUTER_API_KEY not set.')

url = 'https://openrouter.ai/api/v1/models'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {api_key}'})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())

free_models = []
for m in data.get('data', []):
    pricing = m.get('pricing', {})
    prompt_cost = float(pricing.get('prompt', 1))
    if prompt_cost == 0.0:
        free_models.append({
            'id': m['id'],
            'name': m.get('name', ''),
            'context': m.get('context_length', 0),
        })

free_models.sort(key=lambda x: x['id'])
print(f"Total free models: {len(free_models)}\n")
for m in free_models:
    print(f"  {m['id']}")
    print(f"    name: {m['name']}")
    print(f"    context: {m['context']:,}")
    print()
