"""
The model providers rag-titan knows, and their defaults.

Ollama and Anthropic have their own clients. Every other provider speaks the
OpenAI API, so one client serves them all (llm/openai_compatible_llm.py and
embeddings/openai_compatible_embedder.py); a provider is just its address,
the environment variable with its key, and its default models.

API keys are read from the environment when a client is built. They are never
put in the settings, so they cannot end up in a report, a trace or a log.
"""

import os

# base_url: None means LLM_BASE_URL / EMBED_BASE_URL must be set;
# key: the variable with the key (None: no key needed, LLM_API_KEY may still give one);
# chat, agent, embed: default models (None: you name the model, or no embeddings)
PROVIDERS = {
    "ollama": {"service": "Ollama", "chat": "gemma3:4b", "agent": "qwen3:4b",
               "embed": "nomic-embed-text"},
    "anthropic": {"service": "Anthropic", "chat": "claude-haiku-5-5",
                  "agent": "claude-haiku-5-5", "embed": None},
    "openai": {"service": "OpenAI", "base_url": "https://api.openai.com/v1",
               "key": "OPENAI_API_KEY", "chat": "gpt-5-mini",
               "embed": "text-embedding-3-small"},
    "gemini": {"service": "Gemini",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key": "GEMINI_API_KEY", "chat": "gemini-3.7-flash",
               "embed": "gemini-embedding-001"},
    "openrouter": {"service": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "key": "OPENROUTER_API_KEY", "chat": "openai/gpt-5-mini",
                   "embed": "openai/text-embedding-3-small"},
    # chat only: Hugging Face's router has no OpenAI-style embeddings endpoint
    "huggingface": {"service": "Hugging Face", "base_url": "https://router.huggingface.co/v1",
                    "key": "HF_TOKEN", "chat": "openai/gpt-oss-120b", "embed": None,
                    "no_embeddings": True},
    "lmstudio": {"service": "LM Studio", "base_url": "http://localhost:1234/v1"},
    "vllm": {"service": "vLLM", "base_url": "http://localhost:8000/v1"},
    "openai_compatible": {"service": "The OpenAI-compatible server", "base_url": None},
}

ALIASES = {"claude": "anthropic", "hf": "huggingface", "lm_studio": "lmstudio",
           "openai-compatible": "openai_compatible", "compatible": "openai_compatible"}

# the providers served by the OpenAI-compatible client
OPENAI_COMPATIBLE = [name for name in PROVIDERS if name not in ("ollama", "anthropic")]


def provider_name(name):
    """Turn "Claude", " HF ", "lm-studio" and the like into the provider's real name."""
    name = str(name).strip().lower().replace("-", "_")
    return ALIASES.get(name, ALIASES.get(name.replace("_", "-"), name))


def fill_models(settings):
    """
    Fill an empty llm_model, agent_model and embed_model with the providers' defaults.

    The agent model falls back to the chat model, so a provider without its own
    agent default (OpenAI, LM Studio, ...) uses the model you chose.
    """
    chat = PROVIDERS.get(provider_name(settings["llm_provider"]), PROVIDERS["ollama"])
    if not settings.get("llm_model"):
        settings["llm_model"] = chat.get("chat")
    if not settings.get("agent_model"):
        settings["agent_model"] = chat.get("agent") or settings["llm_model"]
    embed = PROVIDERS.get(provider_name(settings["embed_provider"]), {})
    if not settings.get("embed_model"):
        settings["embed_model"] = embed.get("embed")


def connection(name, settings, kind):
    """
    How to reach an OpenAI-compatible provider: the keyword arguments for
    OpenAICompatibleLLM / OpenAICompatibleEmbedder.

    Args:
        name: a provider in OPENAI_COMPATIBLE.
        settings: the dict from get_settings().
        kind: "llm" or "embed": LLM_BASE_URL / LLM_API_KEY or EMBED_BASE_URL /
            EMBED_API_KEY win over the provider's own address and key.
    """
    preset, prefix = PROVIDERS[name], kind.upper()
    base_url = settings.get(f"{kind}_base_url") or preset.get("base_url")
    if not base_url:
        raise ValueError(f"{prefix}_PROVIDER={name} needs the server's address: set "
                         f"{prefix}_BASE_URL, for example http://localhost:8000/v1")
    key_name = preset.get("key")
    api_key = os.getenv(f"{prefix}_API_KEY") or (os.getenv(key_name) if key_name else None)
    return {"base_url": base_url, "api_key": api_key, "service": preset["service"],
            "key_name": key_name}
