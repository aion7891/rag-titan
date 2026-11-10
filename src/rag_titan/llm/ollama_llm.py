"""
Chat model that runs locally with Ollama (https://ollama.com).

Before using it, install Ollama and pull a model:
    ollama pull gemma3:4b
"""

import ollama

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """
    Talks to an Ollama server.

    Args:
        model: Ollama model name, for example "gemma3:4b".
        host: Ollama server address. None means http://localhost:11434.
        think: turn on "thinking" for models that support it.
    """

    def __init__(self, model, host=None, think=False):
        super().__init__(model)
        self.think = think
        self.client = ollama.Client(host=host)

    def chat(self, messages, tools=None):
        """Send messages to Ollama. See BaseLLM.chat() for the reply format."""
        logger.debug("Calling Ollama model '%s'", self.model)

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                # only send "think" when it is on, some models do not support it
                think=True if self.think else None,
            )
        except Exception as error:
            raise RuntimeError(
                f"Ollama chat failed for model '{self.model}': {error}"
            ) from error

        return {
            "content": response.message.content or "",
            "tool_calls": self._read_tool_calls(response.message),
            "message": response.message,
        }

    def tool_result_message(self, tool_call, result):
        """Build the "tool" message Ollama expects after a tool call."""
        return {
            "role": "tool",
            "tool_name": tool_call["name"],
            "content": str(result),
        }

    def _read_tool_calls(self, message):
        """Turn Ollama's tool call objects into simple dicts."""
        tool_calls = []

        for call in message.tool_calls or []:
            tool_calls.append({
                "name": call.function.name,
                "arguments": dict(call.function.arguments or {}),
            })

        return tool_calls
