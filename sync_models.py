import os
import requests
import yaml
import subprocess
import tempfile
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "generated_models.yaml")

load_dotenv(os.path.join(BASE_DIR, ".env"))

models = []


def require_env(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"{name} not found in .env")

    return value


# ============================================================
# OpenRouter
# ============================================================

print("Fetching OpenRouter models...")

openrouter_key = require_env("ARVIN_OPENROUTER_API_KEY")

r = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {openrouter_key}"
    },
    timeout=30,
)

r.raise_for_status()

data = r.json()

openrouter_count = 0

for m in data.get("data", []):
    pricing = m.get("pricing", {})
    architecture = m.get("architecture", {})

    try:
        prompt = float(pricing.get("prompt", 1))
        completion = float(pricing.get("completion", 1))
    except (TypeError, ValueError):
        continue

    # فقط مدل‌های کاملاً رایگان
    if prompt != 0 or completion != 0:
        continue

    # فقط مدل‌هایی که خروجی Text دارند
    output_modalities = architecture.get("output_modalities", [])

    if "text" not in output_modalities:
        continue

    # مدل‌هایی که ورودی text ندارند، برای Chat مناسب نیستند
    input_modalities = architecture.get("input_modalities", [])

    if "text" not in input_modalities:
        continue

    models.append({
        "model_name": "or-" + m["id"].replace("/", "-"),
        "litellm_params": {
            "model": "openrouter/" + m["id"],
            "api_key": "os.environ/ARVIN_OPENROUTER_API_KEY"
        }
    })

    openrouter_count += 1

print(f"OpenRouter chat models: {openrouter_count}")


# ============================================================
# Groq
# ============================================================

print("Fetching Groq models...")

groq_key = require_env("ARVIN_GROQ_API_KEY")

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization": f"Bearer {groq_key}"
    },
    timeout=30,
)

r.raise_for_status()

data = r.json()

groq_count = 0

# مدل‌هایی که endpoint اصلی Chat Completion نیستند
GROQ_NON_CHAT = {
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "canopylabs/orpheus-v1-english",
    "canopylabs/orpheus-arabic-saudi",
}

for m in data.get("data", []):
    model_id = m["id"]

    if model_id in GROQ_NON_CHAT:
        continue

    models.append({
        "model_name": "groq-" + model_id.replace("/", "-"),
        "litellm_params": {
            "model": "groq/" + model_id,
            "api_key": "os.environ/ARVIN_GROQ_API_KEY"
        }
    })

    groq_count += 1

print(f"Groq chat models: {groq_count}")


# ============================================================
# Gemini
# ============================================================

print("Adding Gemini models...")

require_env("GEMINI_API_KEY")

models.append({
    "model_name": "gemini-2.5-flash",
    "litellm_params": {
        "model": "gemini/gemini-2.5-flash",
        "api_key": "os.environ/GEMINI_API_KEY"
    }
})

gemini_count = 1

print(f"Gemini chat models: {gemini_count}")


# Hide internal models from model discovery while keeping them callable.
for model in models:
    model.setdefault("model_info", {})["discoverable"] = False


# ============================================================
# Generate YAML
# ============================================================

new_config = {
    "model_list": models
}

with tempfile.NamedTemporaryFile(
    mode="w",
    delete=False,
    dir=BASE_DIR,
    suffix=".yaml"
) as tmp:

    yaml.dump(
        new_config,
        tmp,
        sort_keys=False,
        allow_unicode=True
    )

    temp_path = tmp.name


# ============================================================
# Compare with previous config
# ============================================================

changed = True

if os.path.exists(OUTPUT):

    with open(temp_path, "rb") as new_file:
        new_data = new_file.read()

    with open(OUTPUT, "rb") as old_file:
        old_data = old_file.read()

    changed = new_data != old_data


if changed:

    os.replace(temp_path, OUTPUT)

    print()
    print("Model list changed.")
    print("Restarting LiteLLM...")

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            os.path.join(BASE_DIR, "docker-compose.yml"),
            "restart",
            "litellm"
        ],
        cwd=BASE_DIR,
        check=True
    )

    print("LiteLLM restarted.")

else:

    os.remove(temp_path)

    print()
    print("No model changes detected.")
    print("LiteLLM restart not required.")


# ============================================================
# Summary
# ============================================================

print()
print("=" * 50)
print(f"OpenRouter : {openrouter_count}")
print(f"Groq       : {groq_count}")
print(f"Gemini     : {gemini_count}")
print(f"TOTAL      : {len(models)}")
print("=" * 50)