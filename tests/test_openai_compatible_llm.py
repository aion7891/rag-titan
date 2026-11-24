"""
Tests for OpenAICompatibleLLM (no network, no key needed): most swap post_json for
a fake that saves the request; the last ones use the fake HTTP server.
"""

import json
import logging

import pytest

from rag_titan.agents import MathAgent
from rag_titan.llm import OpenAICompatibleLLM, openai_compatible_llm
from rag_titan.llm.openai_compatible_llm import read_tool_call

KEY = "sk-test-secret-123"
URL = "https://api.example.com/v1"


def text_reply(text, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}]}


def tool_reply(call_id, name, arguments):
    call = {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [call]}, "finish_reason": "tool_calls"}]}


def llm(**options):
    return OpenAICompatibleLLM("some-model", URL, api_key=KEY, service="Example",
                               key_name="EXAMPLE_API_KEY", **options)


@pytest.fixture
def fake_api(monkeypatch):
    """Replace post_json: save each request, answer with the next prepared reply."""
    calls, replies = [], []

    def fake_post(url, payload, headers=None, timeout=None, service="The API"):
        calls.append({"url": url, "payload": json.loads(json.dumps(payload)),
                      "headers": headers, "service": service})
        return replies.pop(0)

    monkeypatch.setattr(openai_compatible_llm, "post_json", fake_post)
    return calls, replies


def test_the_request_is_chat_completions_with_a_bearer_key(fake_api):
    calls, replies = fake_api
    replies.append(text_reply("Blue."))

    answer = llm(temperature=0, max_tokens=50).ask("Answer in one word.", "Sky colour?")

    assert answer == "Blue."
    request = calls[0]
    assert request["url"] == URL + "/chat/completions"
    assert request["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert request["service"] == "Example"
    assert request["payload"] == {
        "model": "some-model", "max_tokens": 50, "temperature": 0,
        "messages": [{"role": "system", "content": "Answer in one word."},
                     {"role": "user", "content": "Sky colour?"}]}


def test_a_local_server_needs_no_key(fake_api):
    calls, replies = fake_api
    replies.append(text_reply("Hi"))

    OpenAICompatibleLLM("gemma3:4b", "http://localhost:11434/v1/").ask("s", "u")

    assert calls[0]["headers"] == {} and calls[0]["url"] == "http://localhost:11434/v1/chat/completions"


def test_the_agent_loop_works_with_openai_tool_calls(fake_api):
    calls, replies = fake_api
    replies += [tool_reply("call_1", "add", {"a": 2, "b": 3}),
                tool_reply("call_2", "multiply", {"a": 5, "b": 4}),
                text_reply("The answer is 20.")]

    answer = MathAgent(llm()).run("What is (2 + 3) * 4?")

    assert answer == "The answer is 20."
    tools = calls[0]["payload"]["tools"]
    assert {tool["function"]["name"] for tool in tools} == {"add", "subtract", "multiply",
                                                           "divide"}
    last = calls[2]["payload"]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool", "assistant",
                                         "tool"]
    assert last[2]["tool_calls"][0]["id"] == "call_1"             # repeated for the API
    assert last[3] == {"role": "tool", "tool_call_id": "call_1", "content": "5.0"}


@pytest.mark.parametrize("arguments, expected", [
    ('{"a": 1}', {"a": 1}), ({"a": 1}, {"a": 1}), ("", {}), ("not json", {}), ("[1, 2]", {}),
])
def test_tool_arguments_are_read_safely(arguments, expected):
    call = {"id": "c", "function": {"name": "add", "arguments": arguments}}
    assert read_tool_call(call) == {"id": "c", "name": "add", "arguments": expected}


def test_extra_fields_never_reach_the_api(fake_api):
    calls, replies = fake_api
    replies.append(text_reply("ok"))

    llm().chat([{"role": "user", "content": "hi", "images": ["x"], "thinking": "..."}])

    assert calls[0]["payload"]["messages"] == [{"role": "user", "content": "hi"}]


def test_a_cut_off_reply_warns(fake_api, caplog):
    _, replies = fake_api
    replies.append(text_reply("partial", finish="length"))

    with caplog.at_level(logging.WARNING):
        assert llm().ask("s", "u") == "partial"
    assert "length limit" in caplog.text


@pytest.mark.parametrize("reply", [{"choices": []}, {"error": "busy"}, ["odd"]])
def test_a_reply_without_an_answer_is_a_clear_error(fake_api, reply):
    _, replies = fake_api
    replies.append(reply)

    with pytest.raises(RuntimeError, match="Example sent no answer for model 'some-model'"):
        llm().ask("s", "u")


def test_a_missing_key_names_the_variable():
    with pytest.raises(ValueError, match="set the EXAMPLE_API_KEY environment variable"):
        OpenAICompatibleLLM("m", URL, api_key="", service="Example",
                            key_name="EXAMPLE_API_KEY").ask("s", "u")


@pytest.mark.parametrize("options, message", [
    ({"model": "", "base_url": URL}, "needs a model name"),
    ({"model": "m", "base_url": "api.example.com"}, "must start with http"),
    ({"model": "m", "base_url": URL, "max_tokens": 0}, "max_tokens"),
])
def test_bad_options_raise(options, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleLLM(**options)


def test_the_key_is_never_shown():
    assert KEY not in repr(llm()) and KEY not in str(llm())


def test_real_http_and_an_error_without_the_key(fake_ollama):
    fake_ollama.reply = (200, json.dumps(text_reply("Hello")))
    model = OpenAICompatibleLLM("m", fake_ollama.address + "/v1/", api_key=KEY,
                                service="Example", key_name="EXAMPLE_API_KEY")

    assert model.ask("Be brief.", "hi") == "Hello"
    assert fake_ollama.requests[0]["path"] == "/v1/chat/completions"
    assert fake_ollama.requests[0]["authorization"] == f"Bearer {KEY}"

    fake_ollama.reply = (401, json.dumps({"error": {"message": "Incorrect API key"}}))
    with pytest.raises(RuntimeError) as error:
        model.ask("s", "u")
    assert "Example chat failed for model 'm'" in str(error.value)
    assert "HTTP 401: Incorrect API key" in str(error.value)
    assert KEY not in str(error.value) and error.value.__cause__ is None
