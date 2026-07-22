"""The router's own OpenAI-compatible HTTP surface: a stdlib
`http.server`-based proxy (`ThreadingHTTPServer` + `BaseHTTPRequestHandler`,
zero runtime dependencies -- matching both trailbrake's and iliria's own servers;
see docs/DESIGN.md's "why stdlib" note).

Endpoints:
  GET  /health               -- router + per-backend reachability summary
  GET  /router/status        -- circuit-breaker states + decision counters
  GET  /v1/models            -- configured backends and tiers as model objects
  POST /v1/chat/completions
  POST /v1/completions       -- accepted, translated to one user message and
                                 routed the same way; see docs/DESIGN.md's
                                 noted limitation (escalation heuristics key
                                 off chat-shaped content).

Security posture follows the same threat model as iliria's own gateway, an already-
reviewed model for this exact codebase family (personal, single-user,
localhost daily-driver -- not a multi-tenant service): the Origin header is
checked before any real work happens (the Origin/CSRF check), a
socket read timeout defeats slowloris (the read-timeout), and the server binds loopback
by default with a loud warning if overridden without an API key (the loopback default).
Those findings are reapplied here verbatim rather than re-derived, because
the router is one more attacker-reachable stdlib `http.server` process in
that same threat model -- if anything more sensitive, since it can reach two
expensive compute backends from one process.

This router's own security/red-team review (the one that produced the
timing-safe API-key compare and the connection-hygiene fixes below) flagged
three further findings, now closed:

  * **BLIND-CANARY.** The X-Router-Backend/-Tier/-Canary/-Trigger/
    -Fallback-From response headers are client-visible routing metadata --
    on by default, they contaminate a blind pruned-vs-dense canary A/B (a
    client can read and dodge its own arm) and hand an attacker an
    escalation-policy injection-tuning oracle. Gated behind
    `config.server.expose_routing_headers` (default OFF); see
    `_emit_routing_headers` and `config.py`'s `ServerConfig`.
  * `/health` (unauthenticated by design, see below) no longer probes or
    reports a *disabled* backend's id/reachability -- an off, not-yet-
    promoted pruning candidate should be invisible to an unauthenticated
    caller, not just un-routable. See `_handle_health`.
  * Wildcard `cors_origins` (`"*"`) with no `api_key` configured now prints
    a loud startup warning, the same way an unprotected non-loopback bind
    already did (the loopback default). See `startup_warnings`.
"""

from __future__ import annotations

import hmac
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .backends import BackendClient
from .circuit import CircuitBreakerRegistry
from .config import RouterConfig, load_config
from .dispatch import RequestRouter, StreamOutcome
from .errors import RouterError
from .policy import Verifier, build_policy
from .telemetry import DecisionLogger, new_request_id

_MAX_BODY_BYTES = 8 * 2**20

# How much of the relayed SSE body _relay_stream keeps around (a trailing
# window, not the whole response) purely to feed _extract_stream_usage_
# telemetry below. Two full read_chunk()s' worth: generous enough that the
# closing `usage` event -- always the second-to-last SSE frame trailbrake sends,
# right before `data: [DONE]`, and only a few hundred bytes even including
# a full top-10-logit dump-sized payload -- can never fall outside this
# window even if it lands split across a read_chunk() boundary, while still
# bounding memory for an arbitrarily long completion (this is NOT the whole
# body -- that already happens for buffered responses via BackendResponse.
# body, and is not repeated here on purpose).
_SSE_TELEMETRY_TAIL_WINDOW_BYTES = 65536 * 2

# The exact usage-dict field names trailbrake's `_stream_completion` puts in the
# closing SSE `usage` event (mlx_engine/server.py) -- these are JSON keys
# read directly, not header names (contrast dispatch.py's
# _TELEMETRY_HEADER_FIELDS, which maps X-trailbrake-* header names to these same
# field names for the buffered/header path). Only these three are pulled
# out; every other usage key (prompt_tokens, completion_tokens,
# total_tokens, prompt_cache_hit, ...) is left alone here -- this router
# logs routing/telemetry, not token accounting.
_STREAM_USAGE_TELEMETRY_FIELDS = (
    "decode_tokens_per_second",
    "time_to_first_token_seconds",
    "draft_acceptance_rate",
)


def _extract_stream_usage_telemetry(raw: bytes) -> "dict[str, Any]":
    """Best-effort scan of raw (possibly partial) relayed SSE bytes for the
    closing `usage` event trailbrake's `_stream_completion` sends whenever
    `stream_options.include_usage` is true -- which, since
    `dispatch._with_stream_usage_requested` now forces that flag onto every
    upstream stream request this router makes, is every real trailbrake stream
    (see that function's docstring for why). `raw` is expected to be only
    the TAIL of the relayed body (see `_relay_stream`'s bounded rolling
    buffer, capped at `_SSE_TELEMETRY_TAIL_WINDOW_BYTES` -- not the whole
    response): the usage event is always the second-to-last SSE frame trailbrake
    sends (right before `data: [DONE]`), so a bounded tail window is
    sufficient no matter how long the completion is; this function does not
    care how much history came before whatever it's given.

    Splits on blank-line-delimited SSE frames (`b"\\n\\n"`), tries each
    `data: ...` payload as JSON, and pulls `_STREAM_USAGE_TELEMETRY_FIELDS`
    out of any frame that has a `usage` object -- ignoring `[DONE]`,
    ignoring frames with no `usage` key (ordinary content deltas, which is
    all of them on a request that never asked to see one -- see
    `_relay_stream`'s docstring note on why this event is still always
    requested upstream regardless), and treating a present-but-`None` field
    (e.g. `draft_acceptance_rate` when the drafter is not active) the same
    as absent. A frame that fails to decode as JSON -- including one cut in
    half by the start of this bounded tail window, which is expected and
    fine -- is silently skipped, not raised: this is optional, best-effort
    telemetry about a stream that has already been fully relayed to the
    client by the time this runs, and it must never break breaker/log
    finalization. If more than one frame carries a `usage` object (never
    happens with today's trailbrake, which sends exactly one), later frames win,
    same "last one standing" merge `_make_stream_finalizer` itself applies
    when layering this dict over the header-derived floor.
    """
    telemetry: "dict[str, Any]" = {}
    for frame in raw.split(b"\n\n"):
        frame = frame.strip(b"\r\n")
        if not frame.startswith(b"data:"):
            continue
        payload = frame[len(b"data:") :].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            event = json.loads(payload)
        except ValueError:  # json.JSONDecodeError and UnicodeDecodeError both subclass this
            continue
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for field in _STREAM_USAGE_TELEMETRY_FIELDS:
            value = usage.get(field)
            if value is None:
                continue
            try:
                telemetry[field] = float(value)
            except (TypeError, ValueError):
                continue
    return telemetry


def _json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _completions_to_chat(body: dict) -> dict:
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise RouterError(400, "`prompt` must be a string", param="prompt")
    rewritten = dict(body)
    rewritten["messages"] = [{"role": "user", "content": prompt}]
    rewritten.pop("prompt", None)
    return rewritten


class RouterHTTPServer(ThreadingHTTPServer):
    """`config`/`router`/`telemetry` are read once per request (see
    `RouterHandler`'s `self.server.*` accesses) and swapped as a unit by
    `apply()` -- the mechanism behind config-level instant rollback (see
    `reload_from_path` below): a human edits the TOML (e.g. zeroing the
    candidate's weight, or flipping `enabled=false`) and sends SIGHUP,
    in-flight requests finish against whatever they already looked up, new
    requests see the new config -- no restart, no dropped connections.
    This is a best-effort hot-swap (a lock around the *assignment*, not a
    full readers/writer barrier around *use*), which is judged adequate for
    this project's threat model (single-user localhost daily driver, human-
    paced config edits) -- see docs/DESIGN.md's guardrails section.
    """

    daemon_threads = True

    def __init__(self, address, config: RouterConfig, router: RequestRouter, telemetry: DecisionLogger) -> None:
        super().__init__(address, RouterHandler)
        self._reload_lock = threading.Lock()
        self.config = config
        self.router = router
        self.telemetry = telemetry

    def apply(self, config: RouterConfig, router: RequestRouter, telemetry: DecisionLogger) -> None:
        with self._reload_lock:
            self.config = config
            self.router = router
            self.telemetry = telemetry


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "racecontrol"
    # Slowloris guard -- see docs/the threat model's the read-timeout finding against
    # this exact codebase family; reapplied verbatim here.
    timeout = 60

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[router] %s - %s\n" % (self.address_string(), fmt % args))

    # -- shared helpers -------------------------------------------------

    def _start_request(self) -> None:
        """Mints this request's rtr_ correlation id at the earliest possible
        point -- the first line of every do_* method, before any header or
        body parsing that could fail. Every response this request produces
        -- success or error, `_send_json` or the dispatch-result paths --
        must carry this same id (see `_send_json`, which sends it on every
        call, and `_handle_chat`, which threads it into dispatch so a
        successful request's decision-log row agrees with the header the
        client saw). Fixes a gap where a pre-dispatch RouterError (bad auth,
        disallowed origin, malformed JSON, oversize body, unknown path --
        none of which ever reach dispatch.py's own id generation) shipped no
        correlation id at all, breaking tracing on exactly the requests most
        worth debugging."""
        self._request_id = new_request_id()

    def _send_json(self, status: int, payload: dict) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Defensive fallback (`getattr` + `or`), not a bare attribute read:
        # every real do_* entry point calls `_start_request()` first, but
        # this is the one choke point every JSON response (success and
        # error alike) passes through, so it must never raise even if some
        # future call site forgot to.
        self.send_header("X-Router-Request-Id", getattr(self, "_request_id", None) or new_request_id())
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        origins = self.server.config.server.cors_origins
        if not origin or ("*" not in origins and origin not in origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        if "*" not in origins:
            self.send_header("Vary", "Origin")

    def _check_origin(self) -> None:
        origin = self.headers.get("Origin")
        origins = self.server.config.server.cors_origins
        if origin is None or "*" in origins or origin in origins:
            return
        raise RouterError(403, "Cross-origin requests are not permitted from this origin.",
                          error_type="origin_not_allowed")

    def _require_auth(self) -> None:
        api_key = self.server.config.server.api_key
        if not api_key:
            return
        # Constant-time comparison (`==` on str short-circuits at the first
        # differing byte, which is a timing side channel an attacker can use
        # to recover the key byte-by-byte over enough requests) -- both
        # sides encoded to bytes since `hmac.compare_digest` requires the
        # same type on both sides.
        expected = f"Bearer {api_key}".encode("utf-8")
        provided = (self.headers.get("Authorization") or "").encode("utf-8")
        if not hmac.compare_digest(provided, expected):
            raise RouterError(401, "Invalid or missing API key.", error_type="authentication_error")

    def _read_json(self) -> dict:
        # The three rejections below all fire *before* any body byte is
        # read off the socket. On a persistent (HTTP/1.1 keep-alive)
        # connection, leaving the connection open afterwards would mean
        # whatever the client sends next -- headerless chunked data, or the
        # rest of an over-limit body it's still streaming -- gets read as if
        # it were the start of a brand-new request, desyncing this
        # connection's framing. Force a close instead so the socket is torn
        # down rather than reused against a stream we never finished
        # reading. (The JSON-decode/non-dict-body rejections below this,
        # by contrast, always follow a `self.rfile.read(length)` that
        # already consumed exactly the declared body -- the stream is still
        # in sync there, so no close is needed.)
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.close_connection = True
            raise RouterError(400, "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            self.close_connection = True
            raise RouterError(400, "Content-Length must be an integer") from error
        if length <= 0 or length > _MAX_BODY_BYTES:
            self.close_connection = True
            raise RouterError(413, f"Request body must be between 1 and {_MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise RouterError(400, "Request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise RouterError(400, "JSON request body must be an object")
        return payload

    def _handle_error(self, error: Exception) -> None:
        if isinstance(error, RouterError):
            self._send_json(error.status, error.to_object())
            return
        self.log_error("unhandled router error: %s", error)
        self._send_json(500, RouterError(500, "internal router error", error_type="server_error").to_object())

    # -- HTTP verbs -------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._start_request()
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("X-Router-Request-Id", self._request_id)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._start_request()
        try:
            self._check_origin()
            path = urlsplit(self.path).path
            if path == "/health":
                # Health is intentionally exempt from auth, matching
                # iliria's own convention (openai_server.py's do_GET checks
                # /health before calling require_auth()) -- a health probe
                # should not itself require a credential to answer "up".
                self._handle_health()
                return
            self._require_auth()
            if path == "/router/status":
                self._handle_status()
                return
            if path == "/v1/models":
                self._handle_models()
                return
            raise RouterError(404, "Not found", error_type="not_found")
        except Exception as error:  # noqa: BLE001
            self._handle_error(error)

    def do_POST(self) -> None:  # noqa: N802
        self._start_request()
        try:
            self._check_origin()
            self._require_auth()
            path = urlsplit(self.path).path
            if path not in ("/v1/chat/completions", "/v1/completions"):
                raise RouterError(404, "Not found", error_type="not_found")
            body = self._read_json()
            if path == "/v1/completions":
                body = _completions_to_chat(body)
            self._handle_chat(body)
        except Exception as error:  # noqa: BLE001
            self._handle_error(error)

    # -- endpoint bodies --------------------------------------------------

    def _handle_health(self) -> None:
        """Unauthenticated by design (matches iliria's own convention --
        see do_GET). Only *enabled* backends are probed and reported: a
        disabled backend -- e.g. an off, not-yet-promoted pruning candidate
        -- must not have its id or reachability exposed to an
        unauthenticated caller, and there is no reason to spend an outbound
        call probing a backend that is deliberately dark. `/router/status`
        (auth-required) still reports every configured backend's
        circuit-breaker state, disabled or not -- full topology visibility
        is fine once a caller has proven it's allowed to see it."""
        router = self.server.router
        enabled_ids = {backend.id for backend in self.server.config.backends if backend.enabled}
        backend_health = {
            backend_id: client.health()
            for backend_id, client in router.clients.items()
            if backend_id in enabled_ids
        }
        self._send_json(200, {
            "status": "ok" if any(backend_health.values()) else "degraded",
            "backends": backend_health,
        })

    def _handle_status(self) -> None:
        router = self.server.router
        breakers = {
            backend_id: router.circuit_breakers.get(backend_id).snapshot().state.value
            for backend_id in router.clients
        }
        self._send_json(200, {
            "tiers": list(self.server.config.tiers()),
            "circuit_breakers": breakers,
            "decision_counts": self.server.telemetry.counts_snapshot(),
        })

    def _handle_models(self) -> None:
        config = self.server.config
        data = [
            {"id": backend.id, "object": "model", "owned_by": "racecontrol", "tier": backend.tier}
            for backend in config.backends
        ] + [
            {"id": tier, "object": "model", "owned_by": "racecontrol", "tier": tier}
            for tier in config.tiers()
        ]
        self._send_json(200, {"object": "list", "data": data})

    def _handle_chat(self, body: dict) -> None:
        router = self.server.router
        stream = bool(body.get("stream", False))
        user = body.get("user")
        sticky_key = user if isinstance(user, str) and user else None
        wants_draft_first = getattr(router.policy, "wants_draft_first", False)

        if wants_draft_first:
            result = router.dispatch_chat_with_draft_verification(
                body, sticky_key=sticky_key, request_id=self._request_id
            )
            if stream:
                self._send_as_single_stream_flush(result.response, body, result)
            else:
                self._send_buffered(result.response, result)
            return

        if stream:
            result = router.dispatch_chat_stream(body, sticky_key=sticky_key, request_id=self._request_id)
            self._relay_stream(result.response, result)
            return

        result = router.dispatch_chat(body, sticky_key=sticky_key, request_id=self._request_id)
        self._send_buffered(result.response, result)

    # -- response delivery -------------------------------------------------

    def _emit_routing_headers(self, result) -> None:
        """X-Router-Request-Id is always sent (it is a correlation id, not
        routing metadata): it is what lets the offline outcome-join that has
        to detect a bad pruned-trailbrake canary (HTTP 200 with a worse answer)
        find this exact request's full attribution -- backend, tier, canary
        state, trigger -- in the server-side JSONL decision log
        (telemetry.py), which always records it regardless of this method.

        The routing-decision headers themselves (X-Router-Backend/-Tier/
        -Canary/-Fallback-From/-Trigger) are gated behind
        `config.server.expose_routing_headers` (default OFF -- BLIND-CANARY
        finding from this router's security review): client-visible routing
        metadata would (a) contaminate a blind pruned-vs-dense canary A/B --
        a client that can read its own canary arm off the response can
        dodge or specifically target the pruned arm -- and (b) hand an
        escalation-policy attacker an injection-tuning oracle (send a
        probe, read X-Router-Trigger back, learn exactly what phrasing
        tripped the heuristic). Opt in only for a trusted/debug deployment
        that needs the header on the wire itself, not just in the log.

        No-op entirely for an error sent before a backend was chosen
        (result is None)."""
        if result is None:
            return
        self.send_header("X-Router-Request-Id", result.request_id)
        if not self.server.config.server.expose_routing_headers:
            return
        if result.backend_id:
            self.send_header("X-Router-Backend", result.backend_id)
        self.send_header("X-Router-Tier", result.tier)
        self.send_header("X-Router-Canary", "1" if result.canary else "0")
        if result.fallback_from:
            self.send_header("X-Router-Fallback-From", result.fallback_from)
        trigger = getattr(result.decision, "trigger", None)
        if trigger:
            self.send_header("X-Router-Trigger", trigger)

    def _send_buffered(self, response, result=None) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response.body)))
        self._emit_routing_headers(result)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response.body)

    def _write_stream_data(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _relay_stream(self, opened, result=None) -> None:
        # Content-Type is the *router's* contract with the client (it chose
        # this code path because the client asked for `stream: true`), not
        # whatever the backend happened to send -- both real backends always
        # send text/event-stream for a streamed request (verified against
        # trailbrake's `_start_event_stream` and iliria's SSE path), but a proxy
        # should assert its own promise rather than parrot an upstream header
        # for something this significant to how the client parses the body.
        self.send_response(opened.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self._emit_routing_headers(result)
        self._send_cors_headers()
        self.end_headers()
        outcome = StreamOutcome.SUCCESS
        # A rolling tail of what has been relayed so far, purely so
        # _extract_stream_usage_telemetry has something to scan once the
        # stream ends -- see that function's docstring and
        # _SSE_TELEMETRY_TAIL_WINDOW_BYTES. This is NEVER what gets written
        # to the client (that remains the exact, unbuffered `chunk` below,
        # relayed the instant it arrives, same as before this existed) --
        # forwarding is not delayed or altered by one byte to make this
        # possible. Whether the client itself asked for
        # `stream_options.include_usage` or not, trailbrake's closing usage event
        # (now always requested upstream -- see dispatch.
        # _with_stream_usage_requested) is relayed to the client exactly as
        # received, unfiltered; this tail buffer only feeds this router's
        # OWN decision-log telemetry, it does not gate what the client sees.
        tail_buffer = b""
        try:
            while True:
                try:
                    chunk = opened.read_chunk(65536)
                except Exception:  # noqa: BLE001 -- any upstream read/protocol failure
                    # The backend read raised FIRST -> a mid-stream backend
                    # interruption (not a client problem). A clean end-of-body
                    # returns b"" (handled below), never an exception.
                    outcome = StreamOutcome.BACKEND_FAILURE
                    self.close_connection = True
                    break
                if not chunk:
                    break  # clean EOF on a chunked SSE stream = clean completion
                tail_buffer = (tail_buffer + chunk)[-_SSE_TELEMETRY_TAIL_WINDOW_BYTES:]
                try:
                    self._write_stream_data(chunk)
                except OSError:
                    # Our write to the client failed FIRST -> the client aborted;
                    # the backend is healthy, so leave its breaker untouched.
                    # Caught as the broad `OSError` (not just BrokenPipeError/
                    # ConnectionResetError): a client that stops reading
                    # without dropping the connection stalls this write until
                    # the socket timeout fires a plain `TimeoutError`, which
                    # IS an OSError but is not either of those two specific
                    # subclasses -- narrower catches here let a slow-reading
                    # client's TimeoutError escape uncaught, which unwound
                    # past this handler's own error path and attempted a
                    # second, protocol-invalid response on a connection
                    # already mid-stream (see the audit's DoS finding).
                    outcome = StreamOutcome.CLIENT_ABORT
                    self.close_connection = True
                    break
            if outcome is StreamOutcome.SUCCESS:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:  # see the broad-catch note above
                    outcome = StreamOutcome.CLIENT_ABORT
                    self.close_connection = True
        finally:
            opened.close()
            # Single-shot: record the stream's real breaker outcome now that the
            # body is fully relayed (deferred from header time by dispatch).
            # The parsed tail is passed regardless of `outcome` -- a backend
            # failure or client abort right at the very end (e.g. the write
            # of `[DONE]` itself failing) can still have a fully-formed usage
            # event already sitting in `tail_buffer`, and a malformed/partial
            # tail just yields `{}` (see _extract_stream_usage_telemetry's
            # docstring), so there is no failure mode here worth special-
            # casing away.
            if result is not None and result.finalize_stream is not None:
                result.finalize_stream(outcome, _extract_stream_usage_telemetry(tail_buffer))

    def _send_as_single_stream_flush(self, response, original_body: dict, result=None) -> None:
        """Draft-then-escalate always runs buffered internally (the verifier
        needs the complete draft before deciding) -- if the client asked for
        `stream: true` it is still owed SSE framing, just delivered as one
        flush instead of incrementally. See docs/DESIGN.md's documented
        limitation: no incremental token-by-token delivery under this
        policy. Falls back to a plain buffered response if the backend body
        is not the expected chat-completion shape, rather than emit
        malformed SSE."""
        try:
            payload = json.loads(response.body)
            choice = payload["choices"][0]
            content = (choice.get("message") or {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            usage = payload.get("usage")
            request_id = payload.get("id", "chatcmpl-unknown")
            created = payload.get("created")
            model = payload.get("model")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            self._send_buffered(response, result)
            return

        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self._emit_routing_headers(result)
        self._send_cors_headers()
        self.end_headers()

        events = [
            {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
            {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
        ]
        stream_options = original_body.get("stream_options") or {}
        if usage is not None and isinstance(stream_options, dict) and stream_options.get("include_usage"):
            events.append({"id": request_id, "object": "chat.completion.chunk", "created": created,
                          "model": model, "choices": [], "usage": usage})
        try:
            for event in events:
                self._write_stream_data(b"data: " + _json_bytes(event) + b"\n\n")
            self._write_stream_data(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:  # see _relay_stream's broad-catch note
            self.close_connection = True


def _build_components(
    config: RouterConfig, *, verifier: Verifier | None = None
) -> tuple[RequestRouter, DecisionLogger]:
    clients = {backend.id: BackendClient(backend) for backend in config.backends}
    circuit_breakers = CircuitBreakerRegistry(
        failure_threshold=config.circuit_breaker.failure_threshold,
        reset_after_s=config.circuit_breaker.reset_after_s,
    )
    telemetry = DecisionLogger(config.server.log_path)
    policy = build_policy(config, verifier=verifier)
    router = RequestRouter(config, policy, clients, circuit_breakers, telemetry)
    return router, telemetry


def build_server(config: RouterConfig, *, verifier: Verifier | None = None) -> RouterHTTPServer:
    router, telemetry = _build_components(config, verifier=verifier)
    return RouterHTTPServer((config.server.host, config.server.port), config, router, telemetry)


def reload_from_path(server: RouterHTTPServer, config_path: str | Path, *, verifier: Verifier | None = None) -> None:
    """Reloads `config_path` from disk and hot-swaps it into `server` --
    the "instant rollback" path a human triggers on purpose (as opposed to
    the circuit breaker's automatic, unattended one -- see
    docs/DESIGN.md). `[server].host`/`.port` in the new file are loaded but
    cannot change the already-bound listening socket; every other field
    (backend weights/enabled/rollback_target, escalation policy, circuit
    breaker tuning, fallback map) takes effect immediately.

    Deliberately fails safe: any parse/validation error is reported to
    stderr and the *currently running* config keeps serving traffic. A
    typo in an emergency-rollback edit must never be the thing that takes
    the router down.
    """
    try:
        new_config = load_config(config_path)
        router, telemetry = _build_components(new_config, verifier=verifier)
    except Exception as error:  # noqa: BLE001
        print(f"WARNING: config reload from {config_path} failed, keeping previous config: {error}",
              file=sys.stderr)
        return
    server.apply(new_config, router, telemetry)
    print(f"racecontrol: reloaded config from {config_path}", file=sys.stderr)


def install_sighup_reload(
    server: RouterHTTPServer, config_path: str | Path, *, verifier: Verifier | None = None
) -> Callable[[], None]:
    """Registers a SIGHUP handler that calls `reload_from_path` and returns a
    zero-argument callable that restores whatever handler was previously
    installed. Split out from `serve()` (which just calls this once, then
    blocks in `serve_forever()`) so the registration itself -- "does SIGHUP
    now trigger a reload of this exact file into this exact server" -- is
    directly unit-testable without needing a real blocking server loop or
    OS-level signal delivery; see tests/test_reload.py.

    A no-op restore function is returned on platforms without SIGHUP
    (e.g. Windows) rather than raising, since `serve()` should still work
    there -- just without this feature.
    """
    if not hasattr(signal, "SIGHUP"):
        return lambda: None
    previous = signal.getsignal(signal.SIGHUP)

    def _on_sighup(signum, frame):  # noqa: ARG001
        reload_from_path(server, config_path, verifier=verifier)

    signal.signal(signal.SIGHUP, _on_sighup)
    return lambda: signal.signal(signal.SIGHUP, previous)


def startup_warnings(config: RouterConfig) -> list[str]:
    """Loud, printed-to-stderr warnings for a configuration that is reachable
    -- or callable -- in a way that's unsafe without the usual protection.
    Checked once by `serve()` at real startup (not `build_server()`, which
    is also used by tests and by every reload, where printing on each
    config load/reload would just be noise). Pulled out to a pure function
    (config in, warning strings out), the same way `install_sighup_reload`
    was split out of `serve()`, so it's directly unit-testable without a
    live, blocking server loop -- see tests/test_server.py.

    Three findings:

      * the loopback default (docs/the threat model): bound beyond loopback with no
        `api_key` -- anyone who can reach the host can call it as anyone.
      * Wildcard `cors_origins` with no `api_key` -- any web origin a
        client's browser visits can call this router as that client, and
        nothing but the browser's own same-origin policy (which `"*"`
        explicitly disables) was ever standing in the way.
      * `length_routing.enabled=true` with no enabled `role="candidate"`
        backend anywhere -- not a security finding like the two above, but
        the same "reachable/callable config with nothing to actually act on
        it" shape: length-aware routing (docs/DESIGN.md's "Length-aware
        routing" section) only ever excludes a tier's candidate arm, so
        with no enabled candidate backend configured at all it is silently
        a no-op. Caught here rather than as a load-time `ConfigError`
        because it is not actually wrong -- a deployment may enable this
        ahead of turning on a candidate -- just worth a loud heads-up.

    The first two do not fire if an `api_key` is configured: a caller
    (browser-borne or not) still needs the key either way, so the added
    exposure from the host/CORS setting alone is no longer "anyone at all."
    """
    warnings: list[str] = []
    if config.server.host not in ("127.0.0.1", "localhost", "::1") and not config.server.api_key:
        warnings.append("router is listening beyond localhost without an api_key configured")
    if "*" in config.server.cors_origins and not config.server.api_key:
        warnings.append(
            "cors_origins allows '*' (any web origin) with no api_key configured -- any "
            "website a client's browser visits can call this router as that client"
        )
    if config.length_routing.enabled and not any(
        backend.role == "candidate" and backend.enabled for backend in config.backends
    ):
        warnings.append(
            "length_routing.enabled=true but no tier has an enabled role=\"candidate\" backend -- "
            "length-aware routing has nothing to guard and is currently a no-op"
        )
    return warnings


def serve(config: RouterConfig, *, config_path: str | Path | None = None, verifier: Verifier | None = None) -> None:
    server = build_server(config, verifier=verifier)
    for warning in startup_warnings(config):
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"racecontrol listening on http://{config.server.host}:{config.server.port}", file=sys.stderr)

    def _noop() -> None:
        return None

    restore_sighup = _noop
    if config_path is not None:
        restore_sighup = install_sighup_reload(server, config_path, verifier=verifier)
        print(f"racecontrol: SIGHUP will reload {config_path}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
    finally:
        restore_sighup()
        server.server_close()
