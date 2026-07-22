"""Shared test doubles.

`FakeTransport` (no socket) is used by tests/test_backends.py and
tests/test_dispatch.py to check request translation and retry/fallback logic
without ever opening a connection. `start_fake_backend` (a real, tiny
`ThreadingHTTPServer`) is used by tests/test_server.py for a handful of true
end-to-end tests -- mirroring trailbrake's own tests/test_server.py
convention of a real server with a scripted fake behind it, rather than
mocking sockets. Neither ever touches a real model or a GPU.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenResponse:
    def __init__(self, status: int, headers: dict[str, str], chunks: list[bytes]) -> None:
        self.status = status
        self.headers = headers
        self._chunks = list(chunks)
        self.closed = False

    def read_chunk(self, size: int = 65536) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def chat_response_body(
    content: str = "hello",
    *,
    finish_reason: str = "stop",
    model: str = "default",
    usage: dict | None = None,
) -> bytes:
    payload = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return json.dumps(payload).encode("utf-8")


def error_body(message: str = "boom") -> bytes:
    return json.dumps({"error": {"message": message, "type": "server_error", "param": None, "code": None}}).encode()


class FakeTransport:
    """Records every `open()` call it receives and returns queued canned
    responses (FIFO) or raises a queued exception. Enough to test model
    rewriting, header/auth attachment, and retry-on-failure without a
    socket."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._responses: list[FakeOpenResponse | Exception] = []

    def queue_response(
        self,
        status: int,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        headers = headers or {"content-type": "application/json"}
        chunks = _split(body, chunk_size) if chunk_size else [body]
        self._responses.append(FakeOpenResponse(status, headers, chunks))

    def queue_error(self, error: Exception) -> None:
        self._responses.append(error)

    def open(self, **kwargs) -> FakeOpenResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeTransport: no queued response left for this call")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def _split(body: bytes, size: int) -> list[bytes]:
    if not body:
        return [b""]
    return [body[i : i + size] for i in range(0, len(body), size)]


def sse_stream_body(
    content: str = "hello",
    *,
    finish_reason: str = "stop",
    model: str = "default",
    usage: dict | None = None,
) -> bytes:
    """Builds raw SSE bytes shaped exactly like trailbrake's own `_stream_
    completion` (mlx_engine/server.py): one content-delta chunk, one
    empty-delta finish_reason chunk, then -- only if `usage` is given, i.e.
    only when `stream_options.include_usage` was true on the request this
    is standing in for -- one further `choices: []` chunk carrying it,
    then `data: [DONE]`. `chat_response_body()` above is a single buffered
    JSON document with no SSE framing at all; this is the raw-bytes
    counterpart used with `RawStreamResponse` below to exercise the
    router's real SSE relay+parse path (server.py's `_relay_stream` /
    `_extract_stream_usage_telemetry`) end to end."""
    request_id = "chatcmpl-fake-stream"
    events = [
        {"id": request_id, "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
        {"id": request_id, "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
    ]
    if usage is not None:
        events.append({
            "id": request_id, "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [], "usage": usage,
        })
    body = b"".join(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n" for event in events)
    return body + b"data: [DONE]\n\n"


class RawStreamResponse:
    """A `_ScriptedHandler`/`FakeBackend` script entry (append it directly
    to `.script`, not wrapped in a `(status, payload)` tuple like an
    ordinary buffered reply) that emits pre-formatted HTTP/1.1 chunked SSE
    bytes verbatim, instead of `_reply()`'s single buffered JSON body.
    `raw_body` is typically built by `sse_stream_body()` above. Written in
    small `wire_chunk_size`-sized HTTP chunks (default well under the
    router's own 65536-byte `read_chunk()` size) purely so this looks like
    a real multi-chunk streamed response rather than trivially landing in
    one `read_chunk()` call."""

    def __init__(self, status: int, raw_body: bytes, *, wire_chunk_size: int = 32) -> None:
        self.status = status
        self.raw_body = raw_body
        self.wire_chunk_size = wire_chunk_size


class _ScriptedHandler(BaseHTTPRequestHandler):
    script: "list[tuple[int, dict] | RawStreamResponse]" = []
    requests_received: list[dict] = []
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
            return
        self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw) if raw else {}
        type(self).requests_received.append(body)
        if not type(self).script:
            self._reply(500, {"error": {"message": "no script left"}})
            return
        item = type(self).script.pop(0)
        if isinstance(item, RawStreamResponse):
            self._reply_raw_stream(item)
            return
        status, payload = item
        self._reply(status, payload)

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _reply_raw_stream(self, response: RawStreamResponse) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        size = response.wire_chunk_size
        for start in range(0, len(response.raw_body), size):
            piece = response.raw_body[start : start + size]
            self.wfile.write(f"{len(piece):X}\r\n".encode("ascii"))
            self.wfile.write(piece)
            self.wfile.write(b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args) -> None:  # keep test output quiet
        return


class FakeBackend:
    """A tiny real HTTP server standing in for one trailbrake/iliria process.
    `script` is a list of (status, response_dict) tuples popped one per POST
    request; `requests_received` records the JSON body the router actually
    sent (so tests can assert the `model` field was rewritten correctly)."""

    def __init__(self, script: list[tuple[int, dict]]) -> None:
        handler_cls = type(
            "ScriptedHandler",
            (_ScriptedHandler,),
            {"script": list(script), "requests_received": []},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self._handler_cls = handler_cls
        # A short poll_interval (default is 0.5s) so `stop()`'s `shutdown()`
        # returns quickly -- with two of these per server test plus the
        # router itself, the default interval made the suite take ~18s of
        # pure teardown wait, not actual work.
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.02,), daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def script(self) -> list[tuple[int, dict]]:
        """The mutable, per-instance script list -- append `(status, body)`
        tuples; each POST request pops one, FIFO."""
        return self._handler_cls.script

    @property
    def requests_received(self) -> list[dict]:
        return self._handler_cls.requests_received

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
