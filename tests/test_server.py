from __future__ import annotations

import http.client
import json
import re
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from router.config import (
    BackendConfig,
    CircuitBreakerConfig,
    EscalationConfig,
    LengthRoutingConfig,
    RouterConfig,
    ServerConfig,
)
from router.server import _extract_stream_usage_telemetry, build_server, startup_warnings

from .fakes import FakeBackend, RawStreamResponse, chat_response_body, error_body, sse_stream_body


def _wait_for_log_record(log_path: Path, request_id: str, *, timeout: float = 2.0) -> dict:
    """Polls `log_path` for a decision record matching `request_id`.

    A STREAMED response's log write (`finalize_stream`, called from
    `_relay_stream`'s `finally` block, in the request's own server thread)
    happens strictly AFTER the terminating SSE chunk has already been
    flushed to the client socket -- so a test client that has finished
    reading the response body has no ordering guarantee that the
    corresponding decisions.jsonl line has been written yet (contrast a
    BUFFERED response, where `_run`'s `_log` call always completes before
    `_send_buffered` writes a single response byte -- see
    DecisionLogAttributionTests above, which never needs this helper).
    Polls briefly rather than assuming either ordering; fails loudly (not a
    silent empty result) if the record never shows up at all."""
    deadline = time.monotonic() + timeout
    while True:
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                record = json.loads(line)
                if record.get("request_id") == request_id:
                    return record
        if time.monotonic() >= deadline:
            raise AssertionError(f"no decision-log record for request_id={request_id!r} within {timeout}s")
        time.sleep(0.01)


class _RouterHarness:
    def __init__(self, config: RouterConfig) -> None:
        self.server = build_server(config)
        # Short poll_interval so close()'s shutdown() returns quickly (see
        # the matching note in tests/fakes.py's FakeBackend).
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.02,), daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            sent_headers = {"Content-Type": "application/json"}
            sent_headers.update(headers or {})
            connection.request(method, path, body=payload, headers=sent_headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, data
        finally:
            connection.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _ServerTestBase(unittest.TestCase):
    """Shared two-backend (fast/deep) fixture. Subclasses configure the
    scripted response(s) they need on `self.trailbrake`/`self.iliria` before
    exercising `self.harness`."""

    server_overrides: dict = {}

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.trailbrake = FakeBackend(script=[])
        self.iliria = FakeBackend(script=[])
        backends = (
            BackendConfig(
                id="trailbrake-baseline", tier="fast", base_url=self.trailbrake.base_url,
                model_id="default", role="baseline", rollback_target=True,
            ),
            BackendConfig(
                id="iliria", tier="deep", base_url=self.iliria.base_url,
                model_id="glm-5.2-iliria", role="primary", rollback_target=True,
            ),
        )
        config = RouterConfig(
            server=ServerConfig(
                host="127.0.0.1", port=0,
                log_path=str(Path(self.tempdir.name) / "decisions.jsonl"),
                **self.server_overrides,
            ),
            escalation=EscalationConfig(),
            circuit_breaker=CircuitBreakerConfig(),
            backends=backends,
            fallback={"fast": "deep", "deep": "fast"},
        )
        self.harness = _RouterHarness(config)

    def tearDown(self) -> None:
        self.harness.close()
        self.trailbrake.stop()
        self.iliria.stop()
        self.tempdir.cleanup()


class ChatCompletionRoutingTests(_ServerTestBase):
    def test_default_request_reaches_trailbrake_and_response_is_relayed_verbatim(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "hi from trailbrake")
        self.assertEqual(len(self.iliria.requests_received), 0)

    def test_default_request_carries_enable_thinking_false_to_trailbrake(self):
        # Property 1 of the default->escalation contract, over a real HTTP
        # round trip: the no-think guarantee must actually reach the backend
        # on the wire (dispatch._with_no_think_default), not just decide the
        # right tier -- see docs/DESIGN.md's enable_thinking note.
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, _, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        self.assertIs(self.trailbrake.requests_received[0]["enable_thinking"], False)

    def test_reasoning_effort_high_reaches_iliria_over_http(self):
        # Property 2, via the `reasoning_effort` native-field trigger rather
        # than the "#deep" marker every other escalation test in this class
        # uses -- both are "explicit_marker" triggers in policy.py but were
        # previously only exercised together at the policy-unit level
        # (tests/test_policy.py), never over a real request end to end.
        self.iliria.script.append((200, json.loads(chat_response_body("deep answer"))))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"reasoning_effort": "high", "messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "deep answer")
        self.assertEqual(len(self.trailbrake.requests_received), 0)

    def test_forced_backend_override_wins_over_a_hard_signal_over_http(self):
        # Property 3, direction 1, over a real HTTP round trip: an explicit
        # client override to the trailbrake backend must win even though the
        # message would otherwise clear two escalation triggers at once
        # (reasoning_effort=high AND a hard-signal message).
        self.trailbrake.script.append((200, json.loads(chat_response_body("fast answer, forced"))))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {
                "model": "trailbrake-baseline",
                "reasoning_effort": "high",
                "messages": [{"role": "user", "content": "Why does this deadlock intermittently?"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "fast answer, forced")
        self.assertEqual(len(self.iliria.requests_received), 0)

    def test_default_response_hides_routing_headers_but_keeps_correlation_id(self):
        # BLIND-CANARY (security review): a pruned-trailbrake canary can return
        # HTTP 200 with a *worse* answer, and the routing metadata needed to
        # attribute that regression offline is real -- but it must not ride
        # on the client-visible response, or (a) a client can read its own
        # canary arm and dodge/target it, contaminating the blind A/B, and
        # (b) an attacker gets an escalation-policy injection-tuning oracle
        # via X-Router-Trigger. `expose_routing_headers` therefore defaults
        # to off: only the correlation id (not routing metadata) is exposed,
        # and RoutingHeadersOptInTests below covers the opt-in case, while
        # DecisionLogAttributionTests proves the same data still reaches the
        # server-side log either way.
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        self.assertTrue(headers.get("x-router-request-id"), "correlation id must still be exposed")
        for hidden in (
            "x-router-backend", "x-router-tier", "x-router-canary",
            "x-router-trigger", "x-router-fallback-from",
        ):
            self.assertNotIn(hidden, headers, f"{hidden} must be hidden by default (blind canary)")

    def test_explicit_marker_reaches_iliria_with_rewritten_model(self):
        self.iliria.script.append((200, json.loads(chat_response_body("deep answer"))))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "#deep why is this racing"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "deep answer")
        self.assertEqual(self.iliria.requests_received[0]["model"], "glm-5.2-iliria")

    def test_completions_endpoint_translates_prompt_to_a_user_message(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body("ok"))))

        status, _, _ = self.harness.request("POST", "/v1/completions", {"prompt": "add a getter"})

        self.assertEqual(status, 200)
        sent = self.trailbrake.requests_received[0]
        self.assertEqual(sent["messages"], [{"role": "user", "content": "add a getter"}])
        self.assertNotIn("prompt", sent)

    def test_backend_error_falls_back_to_the_other_tier(self):
        self.trailbrake.script.append((500, json.loads(error_body("engine crashed"))))
        self.iliria.script.append((200, json.loads(chat_response_body("fallback answer"))))

        status, _, body = self.harness.request("POST", "/v1/chat/completions", {"messages": []})

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "fallback answer")

    def test_4xx_from_backend_is_relayed_not_treated_as_health_failure(self):
        # A 4xx is the client's fault (bad request / unknown model / auth),
        # not the backend's: it must return to the client verbatim, WITHOUT
        # falling back to the other tier (which would reject it identically)
        # or opening the backend's circuit. Regression for the audit finding
        # "client errors poison global backend health."
        self.trailbrake.script.append((400, json.loads(error_body("unknown model 'bogus'"))))

        status, _, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "model": "bogus"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(len(self.iliria.requests_received), 0)

    def test_both_backends_down_returns_503_no_backend_available(self):
        self.trailbrake.script.append((500, json.loads(error_body())))
        self.iliria.script.append((500, json.loads(error_body())))

        status, _, body = self.harness.request("POST", "/v1/chat/completions", {"messages": []})

        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "no_backend_available")

    def test_unknown_path_is_404(self):
        status, _, _ = self.harness.request("POST", "/v1/nonsense", {})
        self.assertEqual(status, 404)

    def test_malformed_json_body_is_400(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{not json",
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()


class RoutingHeadersOptInTests(_ServerTestBase):
    """`expose_routing_headers = true` is the explicit, trusted/debug opt-in
    for the client-visible X-Router-* headers the BLIND-CANARY fix turns off
    by default (see ChatCompletionRoutingTests' default-hides-headers test).
    Everything asserted here was this router's always-on behavior before
    that fix."""

    server_overrides = {"expose_routing_headers": True}

    def test_opt_in_exposes_backend_tier_and_canary_headers(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        self.assertTrue(headers.get("x-router-request-id"))
        self.assertEqual(headers.get("x-router-backend"), "trailbrake-baseline")
        self.assertEqual(headers.get("x-router-tier"), "fast")
        self.assertEqual(headers.get("x-router-canary"), "0")

    def test_opt_in_exposes_trigger_and_fallback_from_headers(self):
        # "#deep" is an explicit_marker decision that starts in the "deep"
        # tier (iliria); iliria failing is what triggers the fallback to
        # "fast" (trailbrake), so iliria's script must be the one that 500s.
        self.iliria.script.append((500, json.loads(error_body("engine crashed"))))
        self.trailbrake.script.append((200, json.loads(chat_response_body("fallback answer"))))

        status, headers, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "#deep why is this racing"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "fallback answer")
        self.assertEqual(headers.get("x-router-trigger"), "explicit_marker")
        self.assertEqual(headers.get("x-router-fallback-from"), "deep")


class DecisionLogAttributionTests(_ServerTestBase):
    """The BLIND-CANARY fix must not blind the *server*, only the client: the
    same backend/tier/canary/trigger attribution the client no longer sees
    on the wire by default must still land in the JSONL decision log,
    joinable by the X-Router-Request-Id header that is always sent -- see
    telemetry.py. Without this, offline outcome-attribution/shadow-eval
    would have nothing to join a client-reported regression against."""

    def test_decision_log_has_full_attribution_keyed_by_the_request_id_header(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )
        self.assertEqual(status, 200)
        request_id = headers["x-router-request-id"]
        self.assertNotIn("x-router-backend", headers)  # default off -- see above

        log_path = Path(self.harness.server.config.server.log_path)
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        record = next(r for r in records if r["request_id"] == request_id)

        self.assertEqual(record["backend_id"], "trailbrake-baseline")
        self.assertEqual(record["tier"], "fast")
        self.assertEqual(record["canary"], False)
        self.assertEqual(record["status"], "ok")

    def test_decision_log_attributes_an_escalated_request_by_trigger_and_tier(self):
        # Property 4 of the default->escalation contract: the escalation
        # decision itself -- not just a default one -- must be attributable
        # in the server-side log, keyed by the same X-Router-Request-Id
        # header the client always receives (see this class's docstring).
        self.iliria.script.append((200, json.loads(chat_response_body("deep answer"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "#deep why is this racing"}]},
        )
        self.assertEqual(status, 200)
        request_id = headers["x-router-request-id"]

        log_path = Path(self.harness.server.config.server.log_path)
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        record = next(r for r in records if r["request_id"] == request_id)

        self.assertEqual(record["tier"], "deep")
        self.assertEqual(record["backend_id"], "iliria")
        self.assertEqual(record["trigger"], "explicit_marker")
        self.assertIn("#deep", record["reason"])
        self.assertEqual(record["status"], "ok")


class StreamingTests(_ServerTestBase):
    def test_streaming_request_gets_sse_content_type_and_full_body_relayed(self):
        content = "streamed content long enough to span several chunks"
        self.trailbrake.script.append((200, json.loads(chat_response_body(content))))

        status, headers, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": [], "stream": True}
        )

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])
        parsed = json.loads(body)
        self.assertEqual(parsed["choices"][0]["message"]["content"], content)

    def test_streaming_response_also_hides_routing_headers_by_default(self):
        # _relay_stream is a separate send path from _send_buffered but
        # shares the same _emit_routing_headers gate -- pin that it isn't
        # bypassed here.
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": [], "stream": True}
        )

        self.assertEqual(status, 200)
        self.assertTrue(headers.get("x-router-request-id"))
        self.assertNotIn("x-router-backend", headers)
        self.assertNotIn("x-router-tier", headers)


class StreamingTelemetryOverHttpTests(_ServerTestBase):
    """End-to-end regression for the canary red-team's HOLD finding: trailbrake's
    X-trailbrake-* telemetry is a non-stream-only mechanism (server.py's
    _telemetry_response_headers), so streamed traffic -- "the normal
    interactive mode" -- used to write ZERO telemetry to decisions.jsonl.
    `RawStreamResponse`/`sse_stream_body` (tests/fakes.py) give `self.trailbrake`
    real SSE framing (unlike `chat_response_body()`, a single buffered JSON
    document with nothing to parse), so this exercises the actual fix on
    both ends: dispatch.py's `_with_stream_usage_requested` forcing
    `stream_options.include_usage=True` upstream, and server.py's
    `_relay_stream`/`_extract_stream_usage_telemetry` parsing the resulting
    closing usage event back out."""

    def test_streamed_request_with_no_stream_options_still_gets_telemetry_logged(self):
        # THE bug: a plain stream:true request (no stream_options at all --
        # ordinary interactive usage) must now still produce a decision-log
        # record carrying decode_tokens_per_second / time_to_first_token_
        # seconds / draft_acceptance_rate, not silently none of them.
        usage = {
            "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
            "prompt_cache_hit": False,
            "decode_tokens_per_second": 50.35,
            "time_to_first_token_seconds": 0.04596,
            "draft_acceptance_rate": 0.7333,
        }
        self.trailbrake.script.append(RawStreamResponse(200, sse_stream_body("hi", usage=usage)))

        status, headers, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": [], "stream": True}
        )

        self.assertEqual(status, 200)
        self.assertIn(b"data: [DONE]", body)  # real SSE relayed, not a buffered JSON doc

        # The request trailbrake actually received was rewritten to ask for usage,
        # even though the test's own client request above never set it.
        self.assertEqual(self.trailbrake.requests_received[0]["stream_options"], {"include_usage": True})

        request_id = headers["x-router-request-id"]
        log_path = Path(self.harness.server.config.server.log_path)
        record = _wait_for_log_record(log_path, request_id)

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["decode_tokens_per_second"], 50.35)
        self.assertEqual(record["time_to_first_token_seconds"], 0.04596)
        self.assertEqual(record["draft_acceptance_rate"], 0.7333)

    def test_streamed_request_usage_event_is_still_relayed_to_the_client(self):
        # The router's own telemetry need (forcing include_usage upstream)
        # must not make it withhold anything from the client -- _relay_stream
        # forwards every byte trailbrake sends, unfiltered, exactly as before.
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                  "prompt_cache_hit": False, "decode_tokens_per_second": 12.5,
                  "time_to_first_token_seconds": 0.01, "draft_acceptance_rate": None}
        self.trailbrake.script.append(RawStreamResponse(200, sse_stream_body("hi", usage=usage)))

        _, _, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": [], "stream": True}
        )

        # Parse the relayed SSE frames back out rather than substring-match
        # the raw bytes -- robust to json.dumps' exact separator choices.
        relayed_events = [
            json.loads(line[len(b"data: "):])
            for line in body.split(b"\n\n")
            if line.startswith(b"data: ") and line != b"data: [DONE]"
        ]
        usage_events = [event for event in relayed_events if event.get("usage")]
        self.assertEqual(len(usage_events), 1)
        self.assertEqual(usage_events[0]["usage"]["decode_tokens_per_second"], 12.5)

    def test_streamed_request_without_a_usage_event_logs_no_telemetry_fields(self):
        # A backend that never sends a usage event at all (e.g. iliria,
        # which has no equivalent mechanism) must not have any of these keys
        # fabricated in its decision-log record -- same contract as the
        # existing non-stream test_telemetry_logs_exactly_one_ok_record.
        self.trailbrake.script.append(RawStreamResponse(200, sse_stream_body("hi", usage=None)))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": [], "stream": True}
        )
        self.assertEqual(status, 200)

        request_id = headers["x-router-request-id"]
        log_path = Path(self.harness.server.config.server.log_path)
        record = _wait_for_log_record(log_path, request_id)
        self.assertNotIn("decode_tokens_per_second", record)
        self.assertNotIn("draft_acceptance_rate", record)


class ModelsAndStatusTests(_ServerTestBase):
    def test_models_endpoint_lists_backends_and_tiers(self):
        status, _, body = self.harness.request("GET", "/v1/models")
        ids = {entry["id"] for entry in json.loads(body)["data"]}
        self.assertEqual(status, 200)
        self.assertEqual(ids, {"trailbrake-baseline", "iliria", "fast", "deep"})

    def test_health_reports_both_backends_reachable(self):
        status, _, body = self.harness.request("GET", "/health")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["backends"]["trailbrake-baseline"])
        self.assertTrue(payload["backends"]["iliria"])

    def test_status_endpoint_reports_circuit_breakers_closed_initially(self):
        status, _, body = self.harness.request("GET", "/router/status")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(set(payload["tiers"]), {"fast", "deep"})

    def test_status_endpoint_reflects_decision_counts_after_a_request(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body())))
        self.harness.request("POST", "/v1/chat/completions", {"messages": []})

        _, _, body = self.harness.request("GET", "/router/status")
        counts = json.loads(body)["decision_counts"]
        self.assertEqual(counts.get("fast:trailbrake-baseline:ok"), 1)


class HealthEndpointDisabledBackendTests(unittest.TestCase):
    """`/health` is intentionally unauthenticated (matches iliria's own
    convention -- see server.py's do_GET). An unauthenticated caller must
    not be able to enumerate a *disabled* backend's id or reachability
    through it -- e.g. an off, not-yet-promoted pruning candidate should be
    invisible, not just un-routable. Regression for the security review's
    /health topology-leak finding. Needs its own three-backend fixture
    (baseline + disabled candidate + iliria), so it doesn't reuse
    _ServerTestBase's fixed two-backend one."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.trailbrake = FakeBackend(script=[])
        self.candidate = FakeBackend(script=[])
        self.iliria = FakeBackend(script=[])
        backends = (
            BackendConfig(
                id="trailbrake-baseline", tier="fast", base_url=self.trailbrake.base_url,
                model_id="default", role="baseline", rollback_target=True,
            ),
            BackendConfig(
                id="trailbrake-candidate", tier="fast", base_url=self.candidate.base_url,
                model_id="default", role="candidate", enabled=False, weight=0,
            ),
            BackendConfig(
                id="iliria", tier="deep", base_url=self.iliria.base_url,
                model_id="glm-5.2-iliria", role="primary", rollback_target=True,
            ),
        )
        config = RouterConfig(
            server=ServerConfig(
                host="127.0.0.1", port=0, log_path=str(Path(self.tempdir.name) / "decisions.jsonl"),
            ),
            escalation=EscalationConfig(),
            circuit_breaker=CircuitBreakerConfig(),
            backends=backends,
            fallback={"fast": "deep", "deep": "fast"},
        )
        self.harness = _RouterHarness(config)

    def tearDown(self) -> None:
        self.harness.close()
        self.trailbrake.stop()
        self.candidate.stop()
        self.iliria.stop()
        self.tempdir.cleanup()

    def test_disabled_candidate_is_absent_from_the_unauthenticated_health_view(self):
        status, _, body = self.harness.request("GET", "/health")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertNotIn("trailbrake-candidate", payload["backends"])
        self.assertIn("trailbrake-baseline", payload["backends"])
        self.assertIn("iliria", payload["backends"])

    def test_router_status_still_reports_every_configured_backend_behind_auth(self):
        # Contrast case: /router/status requires auth (see SecurityGuardTests
        # below), so full topology visibility -- including a disabled
        # backend's circuit-breaker state -- is fine there. Only the
        # *unauthenticated* /health view is gated.
        status, _, body = self.harness.request("GET", "/router/status")
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertIn("trailbrake-candidate", payload["circuit_breakers"])


class SecurityGuardTests(_ServerTestBase):
    server_overrides = {"cors_origins": ("https://allowed.example",), "api_key": "s3cret"}

    def test_disallowed_origin_is_rejected_before_touching_a_backend(self):
        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": []},
            headers={"Origin": "https://evil.example", "Authorization": "Bearer s3cret"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"]["type"], "origin_not_allowed")
        self.assertEqual(len(self.trailbrake.requests_received), 0)

    def test_allowed_origin_passes_the_guard(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body())))
        status, _, _ = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": []},
            headers={"Origin": "https://allowed.example", "Authorization": "Bearer s3cret"},
        )
        self.assertEqual(status, 200)

    def test_missing_api_key_is_rejected(self):
        status, _, body = self.harness.request("POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["type"], "authentication_error")

    def test_correct_api_key_is_accepted(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body())))
        status, _, _ = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": []},
            headers={"Authorization": "Bearer s3cret"},
        )
        self.assertEqual(status, 200)

    def test_wrong_key_of_the_same_length_is_still_rejected(self):
        # Regression for the audit's timing-safety fix (`hmac.compare_digest`
        # in `_require_auth`): a same-length wrong key is the case a naive
        # `==` vs. a constant-time comparison could plausibly disagree on if
        # the switch were done incorrectly (e.g. comparing the wrong slices).
        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": []},
            headers={"Authorization": "Bearer s3cre7"},  # same length as "s3cret"
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["type"], "authentication_error")

    def test_wrong_key_of_a_different_length_is_rejected(self):
        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions", {"messages": []},
            headers={"Authorization": "Bearer s3cret-but-longer"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["type"], "authentication_error")

    def test_health_does_not_require_an_api_key(self):
        status, _, _ = self.harness.request("GET", "/health")
        self.assertEqual(status, 200)

    def test_router_status_does_require_an_api_key(self):
        status, _, _ = self.harness.request("GET", "/router/status")
        self.assertEqual(status, 401)


def _recv_until_closed(sock: socket.socket, timeout: float = 3.0) -> bytes:
    """Reads from `sock` until the peer closes it (a `b""` read), raising
    `TimeoutError` if that doesn't happen within `timeout` seconds. Used to
    assert the *server* actually closes a connection, not just that it sent
    some expected response bytes -- a hang here means the connection was
    kept alive when it should not have been."""
    sock.settimeout(timeout)
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


class ConnectionHygieneTests(_ServerTestBase):
    """`_read_json` rejections that fire *before* any body byte is read off
    the socket (missing/invalid Content-Length, or a declared length over
    the cap) must close the connection rather than leave a persistent
    HTTP/1.1 connection open -- otherwise whatever the client sends next
    (the rest of an over-limit body it's still streaming, or headerless
    data) gets parsed as if it were the start of a brand-new request on the
    same socket. Regression for the audit's request-handling finding."""

    def test_oversized_content_length_is_rejected_and_closes_the_connection(self):
        sock = socket.create_connection(("127.0.0.1", self.harness.port), timeout=5)
        try:
            oversized = 8 * 2**20 + 1  # one byte past _MAX_BODY_BYTES
            sock.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: test\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {oversized}\r\n"
                f"\r\n".encode("ascii")
                # Deliberately not sending `oversized` bytes of body: a real
                # attacker (or a client that trusts the size limit to fail
                # fast) may not either. The server must reject on the
                # declared length alone, before reading any body.
            )
            try:
                response = _recv_until_closed(sock, timeout=3.0)
            except TimeoutError:
                self.fail("server kept the connection open after an over-limit Content-Length")
        finally:
            sock.close()
        self.assertIn(b" 413 ", response)
        self.assertIn(b"must be between 1 and", response)

    def test_missing_content_length_is_rejected_and_closes_the_connection(self):
        sock = socket.create_connection(("127.0.0.1", self.harness.port), timeout=5)
        try:
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: test\r\n"
                b"Content-Type: application/json\r\n"
                b"\r\n"
            )
            try:
                response = _recv_until_closed(sock, timeout=3.0)
            except TimeoutError:
                self.fail("server kept the connection open after a Content-Length-less request")
        finally:
            sock.close()
        self.assertIn(b" 400 ", response)
        self.assertIn(b"Content-Length is required", response)

    def test_ordinary_malformed_json_does_not_force_a_close(self):
        # Contrast case: this rejection happens *after* the declared body is
        # fully drained, so the stream is still in sync and keep-alive is
        # safe -- the connection must stay usable for a following request.
        connection = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{not json",
                               headers={"Content-Type": "application/json"})
            first = connection.getresponse()
            self.assertEqual(first.status, 400)
            first.read()

            self.trailbrake.script.append((200, json.loads(chat_response_body("still alive"))))
            connection.request("POST", "/v1/chat/completions", body=json.dumps({"messages": []}),
                               headers={"Content-Type": "application/json"})
            second = connection.getresponse()
            self.assertEqual(second.status, 200)
            self.assertEqual(json.loads(second.read())["choices"][0]["message"]["content"], "still alive")
        finally:
            connection.close()


class RequestIdOnEveryResponseTests(_ServerTestBase):
    """Regression coverage for the request-id-on-error-paths fix: every
    response -- including a RouterError raised BEFORE dispatch ever runs
    (malformed JSON, an oversize body, an unknown path) -- must carry a
    present, non-empty, rtr_-prefixed X-Router-Request-Id header. Before
    this fix, only a response that reached `_emit_routing_headers` (i.e. a
    successfully dispatched chat completion) got one at all; every
    pre-dispatch RouterError silently shipped none, breaking tracing on
    exactly the requests most worth debugging (found by a live canary test).

    Two of the cases below (a clean success, and a backend-originated 4xx
    that -- unlike the pre-dispatch cases -- DOES reach dispatch.py's
    _run/_log before being re-raised) additionally assert the header
    matches the request_id of the corresponding decisions.jsonl row: proof
    that RouterHandler's entry-point id is threaded INTO dispatch rather
    than dispatch minting a second, disagreeing one."""

    _RTR_ID_RE = re.compile(r"^rtr_[0-9a-f]{32}$")

    def _assert_looks_like_a_request_id(self, value) -> None:
        self.assertTrue(value, "X-Router-Request-Id header missing or empty")
        self.assertRegex(value, self._RTR_ID_RE)

    def _decision_log_record(self, request_id: str) -> dict:
        log_path = Path(self.harness.server.config.server.log_path)
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        return next(r for r in records if r["request_id"] == request_id)

    def test_malformed_json_400_carries_a_request_id_header(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{not json",
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            headers = {k.lower(): v for k, v in response.getheaders()}
        finally:
            connection.close()
        self._assert_looks_like_a_request_id(headers.get("x-router-request-id"))

    def test_oversized_body_413_carries_a_request_id_header(self):
        # Raw socket, same as ConnectionHygieneTests: a real over-limit
        # client wouldn't necessarily send the declared body either, and the
        # server must reject on Content-Length alone before reading any of
        # it -- see _read_json. http.client can't express that.
        sock = socket.create_connection(("127.0.0.1", self.harness.port), timeout=5)
        try:
            oversized = 8 * 2**20 + 1  # one byte past _MAX_BODY_BYTES
            sock.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: test\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {oversized}\r\n"
                f"\r\n".encode("ascii")
            )
            response = _recv_until_closed(sock, timeout=3.0)
        finally:
            sock.close()
        self.assertIn(b" 413 ", response)
        match = re.search(rb"X-Router-Request-Id:\s*(\S+)\r\n", response, re.IGNORECASE)
        self.assertIsNotNone(match, f"no X-Router-Request-Id header in raw response: {response!r}")
        self._assert_looks_like_a_request_id(match.group(1).decode("ascii"))

    def test_unknown_path_404_carries_a_request_id_header(self):
        status, headers, _ = self.harness.request("POST", "/v1/nonsense", {})
        self.assertEqual(status, 404)
        self._assert_looks_like_a_request_id(headers.get("x-router-request-id"))

    def test_successful_request_id_header_matches_the_decision_log(self):
        self.trailbrake.script.append((200, json.loads(chat_response_body("hi from trailbrake"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "add a getter"}]},
        )

        self.assertEqual(status, 200)
        request_id = headers.get("x-router-request-id")
        self._assert_looks_like_a_request_id(request_id)
        self.assertEqual(self._decision_log_record(request_id)["status"], "ok")

    def test_backend_originated_4xx_request_id_header_matches_the_decision_log(self):
        # Contrast the three pre-dispatch cases above: a 4xx the BACKEND
        # itself returns still goes through dispatch.py's _run, which logs a
        # "client_error" record before re-raising (see _run's
        # BackendRequestFailed/client_error branch) -- so, unlike malformed
        # JSON/oversize/unknown-path, THIS error case does reach the log,
        # and the header must agree with it.
        self.trailbrake.script.append((400, json.loads(error_body("unknown model 'bogus'"))))

        status, headers, _ = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "model": "bogus"},
        )

        self.assertEqual(status, 400)
        request_id = headers.get("x-router-request-id")
        self._assert_looks_like_a_request_id(request_id)
        record = self._decision_log_record(request_id)
        self.assertEqual(record["status"], "client_error")
        self.assertEqual(record["http_status"], 400)

    def test_options_preflight_carries_a_request_id_header(self):
        status, headers, _ = self.harness.request(
            "OPTIONS", "/v1/chat/completions", None, headers={"Origin": "http://example.com"}
        )
        self.assertEqual(status, 204)
        self._assert_looks_like_a_request_id(headers.get("x-router-request-id"))


class StartupWarningsTests(unittest.TestCase):
    """`startup_warnings` is a pure function (config in, warning strings
    out), split out of `serve()` the same way `install_sighup_reload` was --
    see that function's docstring -- specifically so these don't need a
    live, blocking server loop to verify."""

    def _config(self, **server_overrides) -> RouterConfig:
        backends = (
            BackendConfig(id="trailbrake-baseline", tier="fast", base_url="http://127.0.0.1:8080",
                          model_id="default", rollback_target=True),
            BackendConfig(id="iliria", tier="deep", base_url="http://127.0.0.1:8000",
                          model_id="glm-5.2-iliria", rollback_target=True),
        )
        return RouterConfig(
            server=ServerConfig(**server_overrides),
            escalation=EscalationConfig(),
            circuit_breaker=CircuitBreakerConfig(),
            backends=backends,
            fallback={},
        )

    def test_no_warnings_for_the_safe_loopback_default(self):
        self.assertEqual(startup_warnings(self._config()), [])

    def test_warns_on_non_loopback_host_without_an_api_key(self):
        warnings = startup_warnings(self._config(host="0.0.0.0"))
        self.assertTrue(any("localhost" in w for w in warnings), warnings)

    def test_non_loopback_host_with_an_api_key_does_not_warn(self):
        warnings = startup_warnings(self._config(host="0.0.0.0", api_key="s3cret"))
        self.assertEqual(warnings, [])

    def test_warns_on_wildcard_cors_without_an_api_key(self):
        warnings = startup_warnings(self._config(cors_origins=("*",)))
        self.assertTrue(any("cors" in w.lower() or "origin" in w.lower() for w in warnings), warnings)

    def test_wildcard_cors_with_an_api_key_does_not_warn(self):
        warnings = startup_warnings(self._config(cors_origins=("*",), api_key="s3cret"))
        self.assertEqual(warnings, [])

    def test_non_wildcard_cors_without_an_api_key_does_not_warn_about_cors(self):
        # The shipped default cors_origins is a concrete allowlist, not "*" --
        # no api_key configured must not, on its own, trigger the CORS warning.
        warnings = startup_warnings(self._config())
        self.assertEqual(warnings, [])

    def test_both_findings_can_fire_together(self):
        warnings = startup_warnings(self._config(host="0.0.0.0", cors_origins=("*",)))
        self.assertEqual(len(warnings), 2)

    # -- length_routing's "nothing to guard" finding --------------------
    #
    # Not a security finding like the loopback default/wildcard-CORS above, but the same
    # "config is reachable/callable with nothing configured to act on it"
    # shape -- see docs/DESIGN.md's "Length-aware routing" section and
    # startup_warnings's own docstring.

    def _config_with_length_routing(self, *, length_routing_enabled: bool, candidate: BackendConfig | None) -> RouterConfig:
        backends = [
            BackendConfig(id="trailbrake-baseline", tier="fast", base_url="http://127.0.0.1:8080",
                          model_id="default", role="baseline", rollback_target=True),
            BackendConfig(id="iliria", tier="deep", base_url="http://127.0.0.1:8000",
                          model_id="glm-5.2-iliria", rollback_target=True),
        ]
        if candidate is not None:
            backends.append(candidate)
        return RouterConfig(
            server=ServerConfig(),
            escalation=EscalationConfig(),
            circuit_breaker=CircuitBreakerConfig(),
            backends=tuple(backends),
            fallback={},
            length_routing=LengthRoutingConfig(enabled=length_routing_enabled),
        )

    def test_length_routing_disabled_never_warns_even_with_no_candidate(self):
        warnings = startup_warnings(self._config_with_length_routing(length_routing_enabled=False, candidate=None))
        self.assertEqual(warnings, [])

    def test_length_routing_enabled_with_no_candidate_backend_anywhere_warns(self):
        warnings = startup_warnings(self._config_with_length_routing(length_routing_enabled=True, candidate=None))
        self.assertTrue(any("length_routing" in w for w in warnings), warnings)

    def test_length_routing_enabled_with_an_enabled_candidate_does_not_warn(self):
        candidate = BackendConfig(id="trailbrake-candidate", tier="fast", base_url="http://127.0.0.1:8081",
                                  model_id="default", role="candidate", enabled=True)
        warnings = startup_warnings(self._config_with_length_routing(length_routing_enabled=True, candidate=candidate))
        self.assertEqual(warnings, [])

    def test_length_routing_enabled_with_only_a_disabled_candidate_still_warns(self):
        # A disabled candidate is unreachable regardless of length routing
        # (matches backends.length_routing_excluded_ids's own "an already-
        # disabled candidate is not this feature's concern" rule) -- the
        # feature still has nothing live to guard.
        candidate = BackendConfig(id="trailbrake-candidate", tier="fast", base_url="http://127.0.0.1:8081",
                                  model_id="default", role="candidate", enabled=False)
        warnings = startup_warnings(self._config_with_length_routing(length_routing_enabled=True, candidate=candidate))
        self.assertTrue(any("length_routing" in w for w in warnings), warnings)


class EscalationRobustnessOverHttpTests(_ServerTestBase):
    """Red-team priority: the expensive router failure is an escalation
    FALSE NEGATIVE (a hard-reasoning request that silently gets a weak
    no-think answer), not a false positive or a visible error. These pin
    the two highest-value new-risk scenarios at the real HTTP layer (JSON
    parsed by the actual server, not a hand-built request dict) -- see
    tests/test_dispatch.py's MalformedRoutingSignalTests and
    EscalationFailureAttributionTests for the fuller matrix (dispatch-level
    integration tests, mocked transport)."""

    def test_malformed_reasoning_effort_with_a_hard_marker_still_escalates(self):
        # A malformed reasoning_effort (a list -- trivial for a buggy client
        # to send) used to raise an uncaught TypeError inside policy.decide()
        # before the marker check ever ran, which would have surfaced here
        # as a 500 instead of a correct escalation to iliria.
        self.iliria.script.append((200, json.loads(chat_response_body("deep answer"))))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"reasoning_effort": ["high"], "messages": [{"role": "user", "content": "#deep why is this racing"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "deep answer")

    def test_both_tiers_failing_on_a_hard_signal_is_a_visible_503_not_a_silent_200(self):
        # FAIL-CLOSED: a hard-reasoning request that cannot be served by
        # either tier must come back as an explicit 503, never a silently
        # fabricated 200 from a tier the policy never actually chose.
        self.iliria.script.append((500, json.loads(error_body())))
        self.trailbrake.script.append((500, json.loads(error_body())))

        status, _, body = self.harness.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "#deep why is this racing"}]},
        )

        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "no_backend_available")


class ExtractStreamUsageTelemetryTests(unittest.TestCase):
    """Pure-function tests for `_extract_stream_usage_telemetry` -- no
    harness, no sockets, mirroring test_dispatch.py's
    ExtractBackendTelemetryTests style for the header-based extractor."""

    def test_empty_bytes_yield_an_empty_dict(self):
        self.assertEqual(_extract_stream_usage_telemetry(b""), {})

    def test_stream_with_no_usage_event_yields_an_empty_dict(self):
        raw = sse_stream_body("hi", usage=None)
        self.assertEqual(_extract_stream_usage_telemetry(raw), {})

    def test_usage_event_fields_are_extracted(self):
        usage = {
            "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
            "decode_tokens_per_second": 50.35,
            "time_to_first_token_seconds": 0.04596,
            "draft_acceptance_rate": 0.7333,
        }
        raw = sse_stream_body("hi", usage=usage)
        self.assertEqual(
            _extract_stream_usage_telemetry(raw),
            {
                "decode_tokens_per_second": 50.35,
                "time_to_first_token_seconds": 0.04596,
                "draft_acceptance_rate": 0.7333,
            },
        )

    def test_none_valued_fields_are_omitted_not_sent_as_null(self):
        usage = {"decode_tokens_per_second": 12.5, "time_to_first_token_seconds": 0.1,
                  "draft_acceptance_rate": None}
        raw = sse_stream_body("hi", usage=usage)
        result = _extract_stream_usage_telemetry(raw)
        self.assertNotIn("draft_acceptance_rate", result)
        self.assertEqual(result["decode_tokens_per_second"], 12.5)

    def test_non_usage_keys_are_never_included(self):
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
                  "prompt_cache_hit": True, "decode_tokens_per_second": 12.5}
        raw = sse_stream_body("hi", usage=usage)
        result = _extract_stream_usage_telemetry(raw)
        self.assertEqual(result, {"decode_tokens_per_second": 12.5})

    def test_a_truncated_leading_fragment_does_not_break_extraction(self):
        # Simulates _relay_stream's bounded tail window starting mid-frame
        # (the window is a rolling suffix of everything relayed, not
        # necessarily frame-aligned at its own start) -- the garbage prefix
        # before the first real "\n\n" boundary must be silently skipped,
        # not raised, and must not stop the real usage frame after it from
        # being found.
        usage = {"decode_tokens_per_second": 33.3}
        raw = sse_stream_body("hi", usage=usage)
        truncated = b'{"choices": [{"delta": {"content": "garbage-mid-fra' + raw
        self.assertEqual(_extract_stream_usage_telemetry(truncated), {"decode_tokens_per_second": 33.3})

    def test_malformed_json_frame_is_skipped_not_raised(self):
        raw = b"data: {not json at all\n\ndata: [DONE]\n\n"
        self.assertEqual(_extract_stream_usage_telemetry(raw), {})

    def test_last_usage_bearing_frame_wins_if_more_than_one_is_present(self):
        # Never happens with today's trailbrake (exactly one usage event per
        # stream), but pins the documented "last one standing" merge rule
        # rather than leaving multi-frame behavior undefined.
        first = {"id": "x", "object": "chat.completion.chunk", "choices": [],
                  "usage": {"decode_tokens_per_second": 1.0}}
        second = {"id": "x", "object": "chat.completion.chunk", "choices": [],
                   "usage": {"decode_tokens_per_second": 2.0}}
        raw = (b"data: " + json.dumps(first).encode() + b"\n\n"
               + b"data: " + json.dumps(second).encode() + b"\n\n"
               + b"data: [DONE]\n\n")
        self.assertEqual(_extract_stream_usage_telemetry(raw), {"decode_tokens_per_second": 2.0})


if __name__ == "__main__":
    unittest.main()
