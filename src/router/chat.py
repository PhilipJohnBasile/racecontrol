"""Minimal streaming terminal chat client for the router: `racecontrol chat`.

A *client* of the router's OpenAI-compatible surface, not part of the serving
path — it talks to a running `serve` instance (default `http://127.0.0.1:8100`)
over `/v1/chat/completions` with `stream: true` and prints deltas as they
arrive. With the default model name ("default", matching no backend id/model_id
— see `policy.resolve_manual_override`), every request flows through the policy
path, so under a canary config the chat is blind: the answering arm is decided
server-side and never revealed on the wire. Attribution lands in the decision
log, same as any other client.

Stdlib only, by design — this must work in any environment the router itself
runs in, with zero extra dependencies.
"""

from __future__ import annotations

import json
import sys
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8100"
DEFAULT_MODEL = "default"


class ChatRequestError(RuntimeError):
    """Non-200 response from the router (body excerpt included)."""


def _open_connection(base_url: str, timeout_s: float):
    parts = urlsplit(base_url)
    connection_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    return connection_cls(parts.hostname, parts.port, timeout=timeout_s)


def stream_chat(
    messages: list[dict],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    timeout_s: float = 300.0,
    out=sys.stdout,
) -> str:
    """POSTs one streaming chat completion and writes deltas to `out` as they
    arrive. Returns the full assistant text. Raises `ChatRequestError` on a
    non-200 status. SSE frames that are not `data:` lines, or whose payload is
    not JSON, are skipped (the closing `usage` event has an empty `choices`
    list and falls through harmlessly)."""
    connection = _open_connection(base_url, timeout_s)
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    })
    try:
        connection.request(
            "POST", "/v1/chat/completions", body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(2000).decode(errors="replace")
            raise ChatRequestError(f"HTTP {response.status}: {detail}")

        collected: list[str] = []
        buffer = b""
        done = False
        while not done:
            chunk = response.read(1024)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        collected.append(delta)
                        out.write(delta)
                        out.flush()
        out.write("\n")
        return "".join(collected)
    finally:
        connection.close()


def run_repl(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> int:
    """Interactive multi-turn loop. `/new` clears history, `/quit` (or EOF /
    Ctrl-C) exits. A failed request keeps the REPL alive and drops the message
    that caused it, so one transient backend error never loses the session."""
    print(f"router chat @ {base_url}  (model={model})")
    print("/new = fresh conversation, /quit = exit\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/quit":
            return 0
        if user == "/new":
            history = []
            print("(new conversation)\n")
            continue
        history.append({"role": "user", "content": user})
        try:
            reply = stream_chat(
                history, base_url=base_url, model=model, temperature=temperature,
            )
        except Exception as error:  # noqa: BLE001 -- REPL must survive transient failures
            print(f"[error: {error}]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": reply})
