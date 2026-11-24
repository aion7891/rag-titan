"""Tests for choosing the LLM provider and its default models (no network needed)."""

from pathlib import Path

import pytest

from rag_titan import RAGTitan, get_settings
from rag_titan.factory import build_agent_llm, build_llm
from rag_titan.llm import AnthropicLLM, OllamaLLM, anthropic_llm

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_docs" / "leave_policy.md"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test without model settings from the environment."""
    for name in ["LLM_PROVIDER", "LLM_MODEL", "AGENT_MODEL", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


def test_ollama_stays_the_default():
    settings = get_settings()
    assert settings["llm_provider"] == "ollama"
    assert (settings["llm_model"], settings["agent_model"]) == ("gemma3:4b", "qwen3:4b")
    assert isinstance(build_llm(settings), OllamaLLM)


@pytest.mark.parametrize("provider", ["anthropic", "claude", " Claude "])
def test_anthropic_and_claude_build_claude_with_its_default_model(provider):
    settings = get_settings({"llm_provider": provider})

    llm, agent_llm = build_llm(settings), build_agent_llm(settings)

    assert isinstance(llm, AnthropicLLM) and isinstance(agent_llm, AnthropicLLM)
    assert llm.model == agent_llm.model == "claude-haiku-5-5"


def test_models_from_env_and_overrides_win(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    assert build_llm(get_settings()).model == "claude-sonnet-5-5"
    assert get_settings()["agent_model"] == "claude-haiku-5-5"

    settings = get_settings({"llm_provider": "ollama", "llm_model": "llama3.2"})
    assert build_llm(settings).model == "llama3.2"
    assert settings["agent_model"] == "qwen3:4b"


def test_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert build_llm(get_settings({"llm_provider": "claude"})).api_key == "sk-ant-from-env"


def test_unknown_provider_lists_the_choices():
    with pytest.raises(ValueError, match="Unknown LLM provider: 'openia'. Use one of: "
                                         "ollama, anthropic, openai, gemini, openrouter, "
                                         "huggingface, lmstudio, vllm, openai_compatible"):
        build_llm(get_settings({"llm_provider": "openia"}))


def test_rag_titan_answers_with_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    requests = []

    def fake_post(url, payload, headers=None, timeout=None, service="The API"):
        requests.append(payload)
        return {"content": [{"type": "text", "text": "You get 20 days [chunk_0]."}],
                "stop_reason": "end_turn"}

    monkeypatch.setattr(anthropic_llm, "post_json", fake_post)
    rag = RAGTitan(settings={"llm_provider": "claude", "embed_provider": "hash"})
    rag.index([str(SAMPLE)])

    answer = rag.ask("How many paid leave days do I get?")

    assert "20 days" in answer
    assert requests[0]["model"] == "claude-haiku-5-5"
    assert "leave" in requests[0]["messages"][0]["content"].lower()
    assert requests[0]["system"]
