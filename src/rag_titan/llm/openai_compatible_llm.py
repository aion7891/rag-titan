"""
Chat model for any server that speaks the OpenAI Chat Completions API: OpenAI,
Gemini, OpenRouter, Hugging Face, LM Studio, vLLM, Ollama's /v1, and more.

Only the standard library is used, so nothing extra needs to be installed.
The key is sent as "Authorization: Bearer <key>" and never shown in errors.

Example:
    from rag_titan.llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM("gemma3:4b", base_url="http://localhost:11434/v1")
    print(llm.ask("Answer in one word.", "What colour is the sky?"))
"""

import json

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.llm.http_json import post_json
from rag_titan.llm.tool_schema import tools_to_schemas
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """
    Talks to POST {base_url}/chat/completions. Supports tools, so agents work too.

    Args:
        model: the model name the server knows, for example "gpt-5-mini".
        base_url: the API address up to /v1, for example "https://api.openai.com/v1".
        api_key: the key; None or "" for local servers that need none.
        service: the provider's name, used in error messages ("OpenAI", "Gemini").
        key_name: where the key comes from, named in the "no key" error,
            for example "OPENAI_API_KEY"; None means the server needs no key.
        max_tokens: longest reply allowed, in tokens; None uses the server's default.
        temperature: optional randomness from 0 to 2; None uses the model's default.
    """

    def __init__(self, model, base_url, api_key=None, service="The API", key_name=None,
                 max_tokens=None, temperature=None):
        super().__init__(model)
        if not model:
            raise ValueError(f"{service} needs a model name: set LLM_MODEL")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {base_url!r}")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError(f"max_tokens must be a whole number above 0, got {max_tokens!r}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.service, self.key_name = service, key_name
        self.max_tokens, self.temperature = max_tokens, temperature

    def __repr__(self):
        # never show the key, even when someone prints the object
        return f"OpenAICompatibleLLM(model={self.model!r}, base_url={self.base_url!r})"

    def chat(self, messages, tools=None):
        """Send messages to the model. See BaseLLM.chat() for the reply format."""
        if self.key_name and not self.api_key:
            raise ValueError(f"{self.service} needs an API key: set the {self.key_name} "
                             "environment variable (or .env)")
        logger.debug("Calling %s model '%s'", self.service, self.model)

        payload = {"model": self.model, "messages": [clean_message(m) for m in messages]}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools_to_schemas(tools)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            reply = post_json(f"{self.base_url}/chat/completions", payload, headers,
                              service=self.service)
        except RuntimeError as error:
            raise RuntimeError(
                f"{self.service} chat failed for model '{self.model}': {error}") from None
        return self._format_reply(reply)

    def tool_result_message(self, tool_call, result):
        """Build the "tool" message that answers one tool call."""
        return {"role": "tool", "tool_call_id": tool_call.get("id") or "call_0",
                "content": str(result)}

    def _format_reply(self, reply):
        """Pull the text and the tool calls out of the first choice."""
        choices = reply.get("choices") if isinstance(reply, dict) else None
        if not choices:
            raise RuntimeError(f"{self.service} sent no answer for model '{self.model}': "
                               f"{json.dumps(reply)[:200]}")
        message = choices[0].get("message") or {}
        if choices[0].get("finish_reason") == "length":
            logger.warning("%s cut the reply off at the length limit; set a larger "
                           "max_tokens for longer answers", self.service)

        tool_calls = [read_tool_call(call) for call in message.get("tool_calls") or []]
        history = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            history["tool_calls"] = message["tool_calls"]   # the next request must repeat them
        return {"content": message.get("content") or "", "tool_calls": tool_calls,
                "message": history}


def clean_message(message):
    """Keep only the fields the Chat Completions API accepts in a message."""
    if not isinstance(message, dict):
        raise ValueError(f"Each chat message must be a dict, got {type(message).__name__}")
    keys = ("role", "content", "tool_calls", "tool_call_id", "name")
    return {key: message[key] for key in keys if message.get(key) is not None}


def read_tool_call(call):
    """
    One tool call as rag-titan uses it: {"id", "name", "arguments"}.

    The API sends the arguments as a JSON string; a string that is not valid
    JSON gives {} (the tool then reports the missing arguments).
    """
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            logger.warning("Tool call '%s' had arguments that are not JSON: %r",
                           function.get("name"), arguments[:100])
            arguments = {}
    return {"id": call.get("id"), "name": function.get("name"),
            "arguments": arguments if isinstance(arguments, dict) else {}}
