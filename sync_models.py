import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
MODELS_OUTPUT = BASE_DIR / "generated_models.yaml"
ROUTER_OUTPUT = BASE_DIR / "generated_router.yaml"
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
GROQ_URL = "https://api.groq.com/openai/v1/models"
load_dotenv(BASE_DIR / ".env")

# Groq's models-list API does not currently expose a reliable modality field.
# Exclude known non-chat families; unknown future IDs are considered candidates.
GROQ_NON_CHAT_PATTERNS = (
    "whisper", "orpheus", "prompt-guard", "guard-", "speech",
    "transcription", "embedding", "tts",
)

GEMINI_MODEL = {
    "model_name": "gemini-2.5-flash",
    "litellm_params": {
        "model": "gemini/gemini-2.5-flash",
        "api_key": "os.environ/GEMINI_API_KEY",
    },
    "model_info": {"discoverable": True},
}


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} not found in .env")
    return value


def get_json(url, *, headers=None, params=None):
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError(f"Unexpected API response from {url}")
    return payload["data"]


def as_float(value, default=1):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_free_text_chat(model):
    pricing = model.get("pricing") or {}
    architecture = model.get("architecture") or {}
    if as_float(pricing.get("prompt")) != 0 or as_float(pricing.get("completion")) != 0:
        return False
    return (
        "text" in (architecture.get("input_modalities") or [])
        and "text" in (architecture.get("output_modalities") or [])
    )


def alias_for(provider, model_id):
    return re.sub(
        r"[^A-Za-z0-9_.:-]+", "-",
        provider + "-" + model_id.replace("/", "-"),
    ).strip("-")


def fetch_openrouter(api_key, sort, supported_parameters=None):
    params = {"sort": sort, "output_modalities": "text"}
    if supported_parameters:
        params["supported_parameters"] = supported_parameters
    return get_json(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
    )


def make_openrouter_entry(model):
    model_id = model["id"]
    return {
        "model_name": alias_for("or", model_id),
        "litellm_params": {
            "model": "openrouter/" + model_id,
            "api_key": "os.environ/ARVIN_OPENROUTER_API_KEY",
        },
        "model_info": {"discoverable": True},
    }


def make_groq_entry(model):
    model_id = model["id"]
    return {
        "model_name": alias_for("groq", model_id),
        "litellm_params": {
            "model": "groq/" + model_id,
            "api_key": "os.environ/ARVIN_GROQ_API_KEY",
        },
        "model_info": {"discoverable": True},
    }


def dedupe(entries):
    result, seen = [], set()
    for entry in entries:
        provider_model = entry["litellm_params"]["model"]
        if provider_model not in seen:
            result.append(entry)
            seen.add(provider_model)
    return result


# Keep individual models visible and directly callable.
for model in models:
    model.setdefault("model_info", {})["discoverable"] = True

def order_by_ids(models, ordered_ids):
    by_id = {m["id"]: m for m in models}
    result, seen = [], set()
    for model_id in ordered_ids:
        if model_id in by_id and model_id not in seen:
            result.append(by_id[model_id])
            seen.add(model_id)
    result.extend(m for m in models if m["id"] not in seen)
    return result


def write_if_changed(path, content):
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == content:
        return False
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=BASE_DIR, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return True


def main():
    openrouter_key = require_env("ARVIN_OPENROUTER_API_KEY")
    groq_key = require_env("ARVIN_GROQ_API_KEY")
    require_env("GEMINI_API_KEY")

    print("Fetching OpenRouter rankings and free text-chat models...")
    # These API sorts provide ordered lists; the model endpoint does not need
    # to expose numeric latency/throughput/intelligence fields for us to use them.
    or_latency_raw = fetch_openrouter(openrouter_key, "latency-low-to-high")
    or_throughput_raw = fetch_openrouter(openrouter_key, "throughput-high-to-low")
    or_intelligence_raw = fetch_openrouter(openrouter_key, "intelligence-high-to-low")
    or_reasoning_raw = fetch_openrouter(
        openrouter_key, "intelligence-high-to-low", supported_parameters="reasoning"
    )

    all_or = {}
    for batch in (or_latency_raw, or_throughput_raw, or_intelligence_raw, or_reasoning_raw):
        for model in batch:
            if model.get("id") and is_free_text_chat(model):
                all_or[model["id"]] = model
    if not all_or:
        raise RuntimeError(
            "No eligible free OpenRouter chat models found; existing generated files were kept."
        )

    or_models = list(all_or.values())
    latency_ids = [m.get("id") for m in or_latency_raw]
    throughput_ids = [m.get("id") for m in or_throughput_raw]
    intelligence_ids = [m.get("id") for m in or_intelligence_raw]
    reasoning_ids = [m.get("id") for m in or_reasoning_raw]

    or_latency = order_by_ids(or_models, latency_ids)
    or_throughput = order_by_ids(or_models, throughput_ids)
    or_intelligence = order_by_ids(or_models, intelligence_ids)

    # Balanced is a simple composite ranking: 60% provider intelligence rank,
    # 40% provider throughput rank. Ranks are normalized so missing benchmark
    # data naturally places a model later without inventing a numeric score.
    intelligence_rank = {model_id: i for i, model_id in enumerate(intelligence_ids)}
    throughput_rank = {model_id: i for i, model_id in enumerate(throughput_ids)}
    def balanced_score(model):
        model_id = model["id"]
        intelligence_position = intelligence_rank.get(model_id, len(intelligence_ids))
        throughput_position = throughput_rank.get(model_id, len(throughput_ids))
        intelligence_norm = intelligence_position / max(len(intelligence_ids), 1)
        throughput_norm = throughput_position / max(len(throughput_ids), 1)
        return (0.6 * intelligence_norm) + (0.4 * throughput_norm), model_id

    or_balanced = sorted(or_models, key=balanced_score)
    or_reasoning_models = [m for m in or_models if m["id"] in set(reasoning_ids)]
    or_reasoning = order_by_ids(or_reasoning_models, reasoning_ids) if or_reasoning_models else or_intelligence

    print("Fetching Groq active models...")
    groq_raw = get_json(
        GROQ_URL, headers={"Authorization": f"Bearer {groq_key}"}
    )
    groq_models = []
    for model in groq_raw:
        model_id = model.get("id", "")
        if not model_id or model.get("active") is False:
            continue
        if any(p in model_id.lower() for p in GROQ_NON_CHAT_PATTERNS):
            continue
        groq_models.append(model)

    # Groq's models-list API has active/context metadata, but no machine-readable
    # speed score. Context is only used as a deterministic tie-breaker, not as speed.
    groq_models.sort(
        key=lambda m: (int(m.get("context_window") or 0), str(m.get("id") or "")),
        reverse=True,
    )

    or_latency_entries = [make_openrouter_entry(m) for m in or_latency]
    or_throughput_entries = [make_openrouter_entry(m) for m in or_throughput]
    or_intelligence_entries = [make_openrouter_entry(m) for m in or_intelligence]
    or_balanced_entries = [make_openrouter_entry(m) for m in or_balanced]
    or_reasoning_entries = [make_openrouter_entry(m) for m in or_reasoning]
    groq_entries = [make_groq_entry(m) for m in groq_models]
    gemini_entries = [GEMINI_MODEL]

    # AUTO aliases remain stable; their primary deployment and fallback chain
    # are rebuilt from current provider metadata on every script run.
    category_candidates = {
        "AUTO - Fast": dedupe(or_latency_entries + groq_entries + gemini_entries),
        "AUTO - Balanced": dedupe(or_balanced_entries + groq_entries + gemini_entries),
        "AUTO - Reasoning": dedupe(or_reasoning_entries + or_intelligence_entries + groq_entries + gemini_entries),
    }

    raw_entries = (
        [make_openrouter_entry(m) for m in or_models]
        + groq_entries
        + gemini_entries
    )
    aliases = [e["model_name"] for e in raw_entries]
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("Generated aliases collided; refusing to replace existing configuration.")

    auto_entries = []
    for category, candidates in category_candidates.items():
        if not candidates:
            raise RuntimeError(f"No candidates for {category}; existing files were kept.")
        auto_entries.append({
            "model_name": category,
            "litellm_params": dict(candidates[0]["litellm_params"]),
            "model_info": {"discoverable": True},
        })

    generated_models = raw_entries + auto_entries
    alias_by_provider_model = {
        e["litellm_params"]["model"]: e["model_name"] for e in raw_entries
    }

    fallbacks = []
    for category, candidates in category_candidates.items():
        primary_model = candidates[0]["litellm_params"]["model"]
        names, seen = [], set()
        for candidate in candidates[1:]:
            provider_model = candidate["litellm_params"]["model"]
            name = alias_by_provider_model.get(provider_model)
            if name and provider_model != primary_model and name not in seen:
                names.append(name)
                seen.add(name)
        if not names:
            print(f"WARNING: {category} has no fallback candidate.")
        fallbacks.append({category: names})

    # Keep the existing hoosh-masnooi model name while generating its fallback
    # chain dynamically from the currently available models.
    hoosh_candidates = dedupe(
        category_candidates["AUTO - Balanced"]
        + category_candidates["AUTO - Fast"]
        + category_candidates["AUTO - Reasoning"]
    )
    hoosh_fallbacks, seen = [], set()
    for candidate in hoosh_candidates:
        name = alias_by_provider_model.get(candidate["litellm_params"]["model"])
        if name and name not in seen:
            hoosh_fallbacks.append(name)
            seen.add(name)
    fallbacks.insert(0, {"hoosh-masnooi": hoosh_fallbacks})

    model_yaml = yaml.safe_dump(
        {"model_list": generated_models}, sort_keys=False, allow_unicode=True
    )
    router_yaml = yaml.safe_dump(
        {"router_settings": {
            "fallbacks": fallbacks,
            "num_retries": 1,
            "allowed_fails": 1,
            "cooldown_time": 60,
        }},
        sort_keys=False,
        allow_unicode=True,
    )

    # Validate before writing. API errors and empty inventories never overwrite
    # the last known-good configuration.
    if not yaml.safe_load(model_yaml).get("model_list"):
        raise RuntimeError("Generated model list is empty.")
    if not yaml.safe_load(router_yaml)["router_settings"].get("fallbacks"):
        raise RuntimeError("Generated fallback configuration is empty.")

    models_changed = write_if_changed(MODELS_OUTPUT, model_yaml)
    router_changed = write_if_changed(ROUTER_OUTPUT, router_yaml)

    print()
    print("=" * 52)
    print(f"OpenRouter free chat models: {len(or_models)}")
    print(f"Groq active chat candidates: {len(groq_models)}")
    print(f"Gemini models: 1 (currently fixed to gemini-2.5-flash)")
    print(f"Raw models: {len(raw_entries)}")
    print(f"AUTO groups: {len(auto_entries)}")
    print("=" * 52)

    if models_changed or router_changed:
        print("Generated configuration changed; recreating LiteLLM to apply config and mounts...")
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--force-recreate", "litellm"],
            cwd=BASE_DIR,
            check=True,
        )
        print("LiteLLM restarted.")
    else:
        print("No model or fallback changes detected; restart not required.")


if __name__ == "__main__":
    main()
