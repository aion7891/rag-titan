"""Chat models (LLMs). Add new providers here."""

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.llm.ollama_llm import OllamaLLM

__all__ = ["BaseLLM", "OllamaLLM"]
