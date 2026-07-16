import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    api_key = api_key.strip()
print(f"API Key loaded: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"API Key starts with: {api_key[:20] if api_key else 'None'}")
print(f"API Key ends with: {api_key[-10:] if api_key else 'None'}")

from anthropic import Anthropic
client = Anthropic(api_key=api_key)

# Try to list models
try:
    import anthropic
    print(f"Anthropic SDK version: {anthropic.__version__}")
except:
    pass

from anthropic import Anthropic
client = Anthropic(api_key=api_key)

# Try to list models
try:
    import anthropic
    print(f"Anthropic SDK version: {anthropic.__version__}")
except:
    pass

from anthropic import Anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    api_key = api_key.strip()

print(f"API Key: {api_key[:30]}...{api_key[-10:] if api_key else 'NONE'}")

client = Anthropic(api_key=api_key)

# Try to list models
print("\nTrying to list models...")
try:
    models = client.models.list()
    print(f"✓ Available models:")
    for model in models.data:
        print(f"  - {model.id}")
except Exception as e:
    print(f"✗ Error listing models: {e}")

# If models endpoint didn't work, try direct message creation with debug
print("\nTrying direct API call with debug...")
try:
    response = client.messages.create(
        model="claude-opus",
        max_tokens=10,
        messages=[{"role": "user", "content": "test"}]
    )
    print("✓ Success!")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
