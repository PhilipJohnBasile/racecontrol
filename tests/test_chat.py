"""Tests for the `chat` client subcommand (src/router/chat.py)."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from router import chat


def _sse_body(deltas: list[str]) -> bytes:
    events = []
    for delta in deltas:
        events.append(
            "data: " + json.dumps({"choices": [{"delta": {"content": delta}}]})
        )
    # closing usage event (empty choices) then the DONE sentinel -- must both
    # be tolerated by the client without emitting text.
    events.append("data: " + json.dumps({"choices": [], "usage": {"total_tokens": 3}}))
    events.append("data: [DONE]")
    return ("\n".join(events) + "\n").encode()


class _FakeRouterHandler(BaseHTTPRequestHandler):
    deltas = ["Hel", "lo ", "world"]
    status = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_request_body = json.loads(self.rfile.read(length))  # type: ignore[attr-defined]
        if self.status != 200:
            body = b'{"error": {"message": "nope"}}'
            self.send_response(self.status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = _sse_body(self.deltas)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A002 -- silence test noise
        pass


def _serve_once(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_stream_chat_concatenates_deltas_and_writes_them():
    server, base_url = _serve_once(_FakeRouterHandler)
    try:
        out = io.StringIO()
        reply = chat.stream_chat(
            [{"role": "user", "content": "hi"}], base_url=base_url, out=out,
        )
        assert reply == "Hello world"
        assert out.getvalue() == "Hello world\n"
        sent = server.last_request_body  # type: ignore[attr-defined]
        assert sent["model"] == chat.DEFAULT_MODEL
        assert sent["stream"] is True
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
    finally:
        server.shutdown()


def test_stream_chat_raises_on_non_200_with_body_excerpt():
    class Failing(_FakeRouterHandler):
        status = 503

    server, base_url = _serve_once(Failing)
    try:
        try:
            chat.stream_chat([{"role": "user", "content": "hi"}],
                             base_url=base_url, out=io.StringIO())
        except chat.ChatRequestError as error:
            assert "503" in str(error)
            assert "nope" in str(error)
        else:  # pragma: no cover
            raise AssertionError("expected ChatRequestError")
    finally:
        server.shutdown()
