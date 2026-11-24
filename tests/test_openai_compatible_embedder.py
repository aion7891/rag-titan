"""Tests for OpenAICompatibleEmbedder (no network, no key needed)."""

import json

import numpy as np
import pytest

from rag_titan.embeddings import OpenAICompatibleEmbedder, openai_compatible_embedder

KEY = "sk-test-secret-456"
URL = "https://api.example.com/v1"


def reply(vectors, indexes=None):
    indexes = list(range(len(vectors))) if indexes is None else indexes
    return {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": v}
                                       for i, v in zip(indexes, vectors)]}


def embedder(**options):
    return OpenAICompatibleEmbedder("embed-model", URL, api_key=KEY, service="Example",
                                    key_name="EXAMPLE_API_KEY", **options)


@pytest.fixture
def fake_api(monkeypatch):
    """Replace post_json: save each request, answer with the next prepared reply."""
    calls, replies = [], []

    def fake_post(url, payload, headers=None, timeout=None, service="The API"):
        calls.append({"url": url, "payload": json.loads(json.dumps(payload)), "headers": headers})
        return replies.pop(0)

    monkeypatch.setattr(openai_compatible_embedder, "post_json", fake_post)
    return calls, replies


def test_the_request_and_the_vectors(fake_api):
    calls, replies = fake_api
    replies.append(reply([[1, 0], [0, 1]]))

    vectors = embedder().embed_many(["leave days", "laptop"])

    assert calls[0] == {"url": URL + "/embeddings", "headers": {"Authorization": f"Bearer {KEY}"},
                        "payload": {"model": "embed-model", "input": ["leave days", "laptop"]}}
    assert [v.tolist() for v in vectors] == [[1, 0], [0, 1]]
    assert vectors[0].dtype == np.float32


def test_texts_are_sent_in_batches(fake_api):
    calls, replies = fake_api
    replies += [reply([[1.0], [2.0]]), reply([[3.0], [4.0]]), reply([[5.0]])]

    vectors = embedder(batch_size=2).embed_many(["a", "b", "c", "d", "e"])

    assert [call["payload"]["input"] for call in calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert [v[0] for v in vectors] == [1, 2, 3, 4, 5]


def test_vectors_are_put_in_order_by_index(fake_api):
    _, replies = fake_api
    replies.append(reply([[2.0], [0.0], [1.0]], indexes=[2, 0, 1]))

    assert [v[0] for v in embedder().embed_many(["x", "y", "z"])] == [0, 1, 2]


def test_one_text_and_no_texts(fake_api):
    calls, replies = fake_api
    replies.append(reply([[0.5, 0.5]]))

    assert embedder().embed("hi").tolist() == [0.5, 0.5]
    assert embedder().embed_many([]) == [] and len(calls) == 1      # no request for nothing


@pytest.mark.parametrize("bad, message", [
    (reply([[1.0]]), "sent 1 embeddings for 2 texts"),
    ({"error": "busy"}, "sent 0 embeddings for 2 texts"),
    (reply([[1.0], [2.0]], indexes=[0, 0]), "wrong indexes"),
    ({"data": [{"index": 0}, {"index": 1, "embedding": [1.0]}]}, "without an embedding"),
    (reply([[1.0], [1.0, 2.0]]), "vectors of different sizes"),
])
def test_bad_replies_are_clear_errors(fake_api, bad, message):
    _, replies = fake_api
    replies.append(bad)

    with pytest.raises(RuntimeError, match=message):
        embedder().embed_many(["a", "b"])


def test_a_missing_key_names_the_variable():
    with pytest.raises(ValueError, match="set the EXAMPLE_API_KEY environment variable"):
        OpenAICompatibleEmbedder("m", URL, service="Example",
                                 key_name="EXAMPLE_API_KEY").embed_many(["a"])


@pytest.mark.parametrize("options, message", [
    ({"model": "", "base_url": URL}, "needs an embedding model name"),
    ({"model": "m", "base_url": "localhost:1234"}, "must start with http"),
    ({"model": "m", "base_url": URL, "batch_size": 0}, "batch_size"),
    ({"model": "m", "base_url": URL, "batch_size": True}, "batch_size"),
])
def test_bad_options_raise(options, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleEmbedder(**options)


def test_real_http_and_an_error_without_the_key(fake_ollama):
    fake_ollama.reply = (200, json.dumps(reply([[0.1, 0.2]])))
    model = OpenAICompatibleEmbedder("m", fake_ollama.address + "/v1", api_key=KEY,
                                     service="Example", key_name="EXAMPLE_API_KEY")

    assert model.embed("hi") == pytest.approx([0.1, 0.2])
    assert fake_ollama.requests[0]["path"] == "/v1/embeddings"
    assert fake_ollama.requests[0]["authorization"] == f"Bearer {KEY}"
    assert KEY not in repr(model)

    fake_ollama.reply = (404, json.dumps({"error": {"message": "model not found"}}))
    with pytest.raises(RuntimeError) as error:
        model.embed("hi")
    assert "Example embedding failed for model 'm'" in str(error.value)
    assert "HTTP 404: model not found" in str(error.value) and KEY not in str(error.value)
