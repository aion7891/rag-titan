"""
Tests for choosing an OpenAI-compatible provider (OpenAI, Gemini, OpenRouter,
Hugging Face, LM Studio, vLLM, any server): defaults, addresses, keys, errors.
No network: the last test uses the fake HTTP server.
"""

import json

import pytest

from rag_titan import RAGTitan, get_settings
from rag_titan.embeddings import OpenAICompatibleEmbedder
from rag_titan.factory import build_agent_llm, build_embedder, build_llm
from rag_titan.llm import OpenAICompatibleLLM
from rag_titan.providers import provider_name

ENV = ["LLM_PROVIDER", "LLM_MODEL", "AGENT_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
       "EMBED_PROVIDER", "EMBED_MODEL", "EMBED_BASE_URL", "EMBED_API_KEY", "OPENAI_API_KEY",
       "GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider, url, key, model, embed", [
    ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5-mini",
     "text-embedding-3-small"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
     "gemini-3.7-flash", "gemini-embedding-001"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-5-mini",
     "openai/text-embedding-3-small"),
    ("huggingface", "https://router.huggingface.co/v1", "HF_TOKEN", "openai/gpt-oss-120b", None),
])
def test_cloud_providers_and_their_defaults(monkeypatch, provider, url, key, model, embed):
    monkeypatch.setenv(key, "secret-from-env")
    settings = get_settings({"llm_provider": provider, "embed_provider": provider
                             if embed else "ollama"})

    llm = build_llm(settings)

    assert isinstance(llm, OpenAICompatibleLLM)
    assert (llm.base_url, llm.model, llm.key_name, llm.api_key) == (url, model, key,
                                                                     "secret-from-env")
    assert build_agent_llm(settings).model == model      # agents use the same model
    if embed:
        embedder = build_embedder(settings)
        assert isinstance(embedder, OpenAICompatibleEmbedder)
        assert (embedder.base_url, embedder.model) == (url, embed)


@pytest.mark.parametrize("provider, url", [("lmstudio", "http://localhost:1234/v1"),
                                           ("vllm", "http://localhost:8000/v1")])
def test_local_servers_need_a_model_name_but_no_key(provider, url):
    with pytest.raises(ValueError, match="needs a model name: set LLM_MODEL"):
        build_llm(get_settings({"llm_provider": provider}))

    settings = get_settings({"llm_provider": provider, "llm_model": "qwen3-8b"})
    llm = build_llm(settings)
    assert (llm.base_url, llm.api_key, llm.key_name) == (url, None, None)
    assert build_agent_llm(settings).model == "qwen3-8b"


def test_any_server_needs_its_address(monkeypatch):
    with pytest.raises(ValueError, match="needs the server's address: set LLM_BASE_URL"):
        build_llm(get_settings({"llm_provider": "openai_compatible", "llm_model": "m"}))

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "my-gateway-key")
    llm = build_llm(get_settings({"llm_provider": "openai_compatible", "llm_model": "m"}))
    assert (llm.base_url, llm.api_key) == ("https://llm.example.com/v1", "my-gateway-key")


def test_chat_and_embeddings_can_use_different_servers(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://chat.local/v1")
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed.local/v1")
    settings = get_settings({"llm_provider": "vllm", "llm_model": "m",
                             "embed_provider": "lmstudio", "embed_model": "e"})

    assert build_llm(settings).base_url == "http://chat.local/v1"
    assert build_embedder(settings).base_url == "http://embed.local/v1"


def test_llm_api_key_wins_over_the_provider_s_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai-var")
    monkeypatch.setenv("LLM_API_KEY", "from-llm-var")

    assert build_llm(get_settings({"llm_provider": "openai"})).api_key == "from-llm-var"


def test_keys_never_enter_the_settings_or_the_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("LLM_API_KEY", "sk-very-secret")
    settings = get_settings({"llm_provider": "openai", "embed_provider": "openai"})

    assert "sk-very-secret" not in json.dumps(settings, default=str)
    assert "sk-very-secret" not in repr(build_llm(settings))
    assert "sk-very-secret" not in repr(build_embedder(settings))


def test_a_missing_key_names_the_variable():
    llm = build_llm(get_settings({"llm_provider": "gemini"}))

    with pytest.raises(ValueError, match="Gemini needs an API key: set the GEMINI_API_KEY"):
        llm.ask("s", "u")


@pytest.mark.parametrize("provider, service", [("huggingface", "Hugging Face"),
                                               ("anthropic", "Anthropic")])
def test_chat_only_providers_refuse_embeddings(provider, service):
    with pytest.raises(ValueError, match=f"{service} has no OpenAI-style embeddings endpoint"):
        build_embedder(get_settings({"embed_provider": provider}))


def test_an_unknown_embedding_provider_lists_the_choices():
    with pytest.raises(ValueError, match="Unknown embedding provider: 'openia'. Use one of: "
                                         "ollama, hash, openai, gemini, openrouter, lmstudio, "
                                         "vllm, openai_compatible"):
        build_embedder(get_settings({"embed_provider": "openia"}))


@pytest.mark.parametrize("written, name", [("Claude", "anthropic"), (" HF ", "huggingface"),
                                           ("lm-studio", "lmstudio"), ("LM_Studio", "lmstudio"),
                                           ("openai-compatible", "openai_compatible")])
def test_provider_names_are_read_loosely(written, name):
    assert provider_name(written) == name


def test_ollama_keeps_its_defaults():
    settings = get_settings()
    assert (settings["llm_model"], settings["agent_model"], settings["embed_model"]) == (
        "gemma3:4b", "qwen3:4b", "nomic-embed-text")


def test_ragtitan_answers_through_an_openai_compatible_server(fake_ollama, monkeypatch,
                                                              tmp_path):
    reply = {"choices": [{"message": {"role": "assistant", "content": "20 days [chunk_0]."},
                          "finish_reason": "stop"}]}
    fake_ollama.reply = (200, json.dumps(reply))
    monkeypatch.setenv("LLM_BASE_URL", fake_ollama.address + "/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-gateway")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hr.md").write_text("Every employee gets 20 leave days.", encoding="utf-8")

    rag = RAGTitan(settings={"llm_provider": "openai_compatible", "llm_model": "my-model"})
    rag.index([tmp_path / "hr.md"])

    assert rag.ask("How many leave days?") == "20 days [chunk_0]."
    request = fake_ollama.requests[-1]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer sk-gateway"
    assert request["body"]["model"] == "my-model"
