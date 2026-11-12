"""
Tests for the standard-library Ollama client and the tool schemas.

The expected values in fixtures/ollama_requests.json were recorded from the
official `ollama` package, so these tests prove nothing changed for users.
A tiny HTTP server in a thread stands in for Ollama.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rag_titan.agents.tools import MATH_TOOLS
from rag_titan.llm.ollama_http import clean_message, parse_host, post_json
from rag_titan.llm.tool_schema import function_to_tool, tools_to_schemas

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "ollama_requests.json")
                      .read_text(encoding="utf-8"))


class FakeOllama(BaseHTTPRequestHandler):
    """Saves each request and answers with the server's `reply` (status, body)."""

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.server.requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(length)),
        })
        status, body = self.server.reply
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):
        pass                                    # keep the test output quiet


@pytest.fixture
def server():
    """Start a fake Ollama on a free port; yields the server object."""
    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    httpd.requests, httpd.reply = [], (200, '{"ok": true}')
    httpd.address = f"http://127.0.0.1:{httpd.server_port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_hosts_are_parsed_like_the_official_client(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    for host, expected in FIXTURES["hosts"].items():
        assert parse_host(host) == expected, host

    assert parse_host(None) == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "example.com:8080")
    assert parse_host(None) == "http://example.com:8080"
    assert parse_host("https://ollama.com") == "https://ollama.com:443"


def test_tool_schemas_match_the_official_client():
    assert tools_to_schemas(MATH_TOOLS) == FIXTURES["tool_schemas"]
    assert tools_to_schemas(None) == []

    ready_made = FIXTURES["tool_schemas"][0]
    assert function_to_tool(ready_made) is ready_made
    with pytest.raises(ValueError, match="function or a dict"):
        function_to_tool("add")


def test_clean_message_drops_empty_fields_only():
    message = {"role": "assistant", "content": "", "tool_calls": [{"x": 1}],
               "images": [], "thinking": None, "tool_name": "add"}

    assert clean_message(message) == {"role": "assistant", "tool_calls": [{"x": 1}],
                                      "tool_name": "add"}
    assert clean_message({"role": "user", "content": "0"})["content"] == "0"
    with pytest.raises(ValueError, match="must be a dict"):
        clean_message("hello")


def test_post_json_sends_json_and_the_api_key(server, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key")
    server.reply = (200, '{"message": {"content": "hi"}}')

    reply = post_json(server.address, "/api/chat", {"model": "m", "stream": False})

    assert reply == {"message": {"content": "hi"}}
    assert server.requests == [{"path": "/api/chat", "authorization": "Bearer secret-key",
                                "body": {"model": "m", "stream": False}}]


def test_post_json_without_a_key_sends_no_authorization(server, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    post_json(server.address, "/api/embed", {"input": ["a"]})
    assert server.requests[0]["authorization"] is None


def test_ollama_errors_are_clear(server):
    server.reply = (404, '{"error": "model \\"nope\\" not found, try pulling it first"}')
    with pytest.raises(RuntimeError, match='HTTP 404: model "nope" not found'):
        post_json(server.address, "/api/chat", {})

    server.reply = (200, "<html>not json</html>")
    with pytest.raises(RuntimeError, match="not JSON"):
        post_json(server.address, "/api/chat", {})


def test_unreachable_server_says_how_to_fix_it(server):
    address = server.address
    server.shutdown()
    server.server_close()                       # nothing listens on this port now

    with pytest.raises(RuntimeError, match="Is Ollama running"):
        post_json(address, "/api/chat", {}, timeout=5)
