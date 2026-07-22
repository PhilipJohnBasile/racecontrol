from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from router.backends import BackendClient
from router.circuit import CircuitBreakerRegistry
from router.config import (
    BackendConfig,
    CircuitBreakerConfig,
    EscalationConfig,
    LengthRoutingConfig,
    RouterConfig,
    ServerConfig,
)
from router.dispatch import (
    RequestRouter,
    StreamOutcome,
    _extract_backend_telemetry,
    _with_stream_usage_requested,
)
from router.errors import BackendRequestFailed, NoBackendAvailable
from router.policy import AlwaysTierPolicy, DefaultPolicy, DraftThenEscalatePolicy
from router.telemetry import DecisionLogger

from .fakes import FakeTransport, chat_response_body, error_body


def _backend(**overrides) -> BackendConfig:
    defaults = dict(
        id="trailbrake-baseline", tier="fast", base_url="http://127.0.0.1:8080",
        model_id="default", weight=100, enabled=True, role="baseline", rollback_target=True,
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


class _Harness:
    """Wires a real RequestRouter against fake (socket-free) backends, with
    a real CircuitBreakerRegistry and a real DecisionLogger writing to a
    scratch directory -- everything downstream of `policy.decide()` is real;
    only the actual HTTP transport is faked. `tempdir` is closed by the
    caller (see tearDown in each TestCase below)."""

    def __init__(self, backends: tuple[BackendConfig, ...], *, policy=None, fallback=None,
                failure_threshold: int = 3, reset_after_s: float = 60.0,
                length_routing: LengthRoutingConfig | None = None):
        self.tempdir = tempfile.TemporaryDirectory()
        self.transports: dict[str, FakeTransport] = {b.id: FakeTransport() for b in backends}
        clients = {b.id: BackendClient(b, transport=self.transports[b.id]) for b in backends}
        self.config = RouterConfig(
            server=ServerConfig(log_path=str(Path(self.tempdir.name) / "decisions.jsonl")),
            escalation=EscalationConfig(),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=failure_threshold, reset_after_s=reset_after_s),
            backends=backends,
            fallback=fallback or {},
            length_routing=length_routing or LengthRoutingConfig(),
        )
        self.circuit_breakers = CircuitBreakerRegistry(failure_threshold=failure_threshold, reset_after_s=reset_after_s)
        self.telemetry = DecisionLogger(self.config.server.log_path)
        self.policy = policy or DefaultPolicy()
        self.router = RequestRouter(self.config, self.policy, clients, self.circuit_breakers, self.telemetry)

    def close(self) -> None:
        self.tempdir.cleanup()

    def log_lines(self) -> list[dict]:
        path = Path(self.config.server.log_path)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]


class BasicRoutingTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_default_request_routes_to_fast_tier(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("hi"))
        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "add a getter"}]})
        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.tier, "fast")
        self.assertFalse(result.canary)

    def test_explicit_marker_routes_to_deep_tier(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))
        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "#deep why does this race"}]})
        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")

    def test_model_is_rewritten_per_backend(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"model": "deep", "messages": []})
        sent = json.loads(self.harness.transports["iliria"].calls[0]["body"])
        self.assertEqual(sent["model"], "glm-5.2-iliria")

    def test_telemetry_logs_exactly_one_ok_record(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": []})
        lines = self.harness.log_lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["status"], "ok")
        self.assertEqual(lines[0]["backend_id"], "trailbrake-baseline")
        self.assertEqual(lines[0]["trigger"], "default")
        # A backend that never sends the optional X-trailbrake-* telemetry headers
        # (iliria, or trailbrake without the drafter flag) must not have any of
        # those keys appear in its decision-log record at all.
        self.assertNotIn("decode_tokens_per_second", lines[0])
        self.assertNotIn("draft_acceptance_rate", lines[0])

    def test_backend_telemetry_headers_are_folded_into_the_decision_log(self):
        # trailbrake's server.py sends these as optional X-trailbrake-*
        # response headers on a non-stream completion (see
        # _telemetry_response_headers) when the opt-in speculative-decoding
        # drafter is active for this request -- the router captures them
        # generically (it never asserts which backend sent them) into
        # DecisionRecord.extra.
        self.harness.transports["trailbrake-baseline"].queue_response(
            200,
            chat_response_body(),
            headers={
                "content-type": "application/json",
                "x-trailbrake-decode-tokens-per-second": "50.35",
                "x-trailbrake-ttft-seconds": "0.045960",
                "x-trailbrake-draft-acceptance-rate": "0.7333",
            },
        )
        self.harness.router.dispatch_chat({"messages": []})
        record = self.harness.log_lines()[0]
        self.assertEqual(record["decode_tokens_per_second"], 50.35)
        self.assertEqual(record["time_to_first_token_seconds"], 0.04596)
        self.assertEqual(record["draft_acceptance_rate"], 0.7333)


class NoThinkDefaultPropagationTests(unittest.TestCase):
    """Property 1 of the default->escalation contract: a plain request that
    lands on the router's own default tier must not just *route* there --
    the no-think guarantee has to actually reach the backend on the wire
    (`dispatch._with_no_think_default`), not rest on an assumption about
    trailbrake's own field-absent default (see docs/DESIGN.md's `enable_thinking`
    note, which -- before this fix -- was aspirational: nothing in the
    router ever set this field)."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_default_request_sends_enable_thinking_false_to_trailbrake(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "add a getter"}]})

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertIs(sent["enable_thinking"], False)

    def test_client_supplied_enable_thinking_is_not_overridden(self):
        # An explicit client value always wins -- the router only fills the
        # gap when the field is entirely absent, it never overrides it.
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat(
            {"enable_thinking": True, "messages": [{"role": "user", "content": "add a getter"}]}
        )

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertIs(sent["enable_thinking"], True)

    def test_escalated_request_does_not_get_the_default_tiers_no_think_flag(self):
        # The no-think default is specific to `default_tier` -- a request
        # that escalates to "deep" must reach iliria exactly as the client
        # sent it (iliria's own enable_thinking/reasoning_effort are native
        # fields the router never rewrites -- see docs/DESIGN.md).
        self.harness.transports["iliria"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "#deep why does this race"}]})

        sent = json.loads(self.harness.transports["iliria"].calls[0]["body"])
        self.assertNotIn("enable_thinking", sent)


class EscalationTriggerDispatchTests(unittest.TestCase):
    """End-to-end (real DefaultPolicy + real dispatch + fake transport)
    coverage for the two escalation triggers BasicRoutingTests' marker-only
    example doesn't exercise: the `reasoning_effort` native field, and the
    task-heuristic scoring a hard signal in ordinary prose with no marker at
    all. Both triggers are already pinned at the policy-unit level
    (tests/test_policy.py); these prove the same triggers actually carry a
    request all the way to the iliria backend, not just to the right
    RoutingDecision in isolation."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_reasoning_effort_high_escalates_to_iliria_end_to_end(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))

        result = self.harness.router.dispatch_chat(
            {"reasoning_effort": "high", "messages": [{"role": "user", "content": "add a getter"}]}
        )

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(result.decision.trigger, "explicit_marker")
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 0)

    def test_task_heuristic_alone_escalates_to_iliria_end_to_end(self):
        # No marker, no reasoning_effort -- purely the pattern-based scorer
        # (see policy.py's hardness_score) clearing the default 0.6 threshold.
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))

        result = self.harness.router.dispatch_chat(
            {"messages": [{"role": "user",
                           "content": "Why does this crash intermittently? Smells like a race condition."}]}
        )

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(result.decision.trigger, "task_heuristic")
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 0)


class ClientOverrideAcrossTiersTests(unittest.TestCase):
    """Property 3 of the default->escalation contract: an explicit client
    override (`model` naming a backend id directly) must force that exact
    backend in *both* directions, regardless of what the escalation policy
    would otherwise decide -- manual override is trigger #1 in
    policy.DefaultPolicy.decide, checked before reasoning_effort/marker/
    heuristic ever look at the request (see policy.resolve_manual_override).
    ManualBackendOverrideTests above already covers a same-tier pin
    (trailbrake-baseline vs. trailbrake-candidate); this covers the cross-tier
    direction the property is really about."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_forced_trailbrake_backend_overrides_a_hard_reasoning_signal(self):
        # Force no-think (trailbrake) even though the request would otherwise
        # clear *two* escalation triggers at once (reasoning_effort=high AND
        # a hard-signal message).
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())

        result = self.harness.router.dispatch_chat({
            "model": "trailbrake-baseline",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "Why does this deadlock intermittently?"}],
        })

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.tier, "fast")
        self.assertEqual(result.decision.trigger, "manual_override")
        self.assertEqual(len(self.harness.transports["iliria"].calls), 0)

    def test_forced_iliria_backend_overrides_an_easy_request(self):
        # Force thinking (iliria) even though the message has zero
        # escalation signal (it even carries an easy-signal marker word).
        self.harness.transports["iliria"].queue_response(200, chat_response_body())

        result = self.harness.router.dispatch_chat({
            "model": "iliria",
            "messages": [{"role": "user", "content": "rename this variable to `count`"}],
        })

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(result.decision.trigger, "manual_override")
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 0)


class CanaryTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", weight=0, role="baseline", rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", weight=100, role="candidate"),
        )
        self.harness = _Harness(backends)

    def tearDown(self):
        self.harness.close()

    def test_100_pct_canary_weight_always_hits_candidate(self):
        for _ in range(5):
            self.harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
        for _ in range(5):
            result = self.harness.router.dispatch_chat({"messages": []})
            self.assertEqual(result.backend_id, "trailbrake-candidate")
            self.assertTrue(result.canary)

    def test_disabling_candidate_is_the_instant_rollback(self):
        # Simulates flipping `enabled=false` on the candidate: even with its
        # weight left at 100, a disabled backend must never be chosen.
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", weight=0, role="baseline", rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", weight=100, enabled=False, role="candidate"),
        )
        harness = _Harness(backends)
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat({"messages": []})
            self.assertEqual(result.backend_id, "trailbrake-baseline")
            self.assertFalse(result.canary)
        finally:
            harness.close()


class LengthAwareRoutingTests(unittest.TestCase):
    """Feature W1c: length-aware arm routing -- see docs/DESIGN.md's
    "Length-aware routing" section. A tier's drafter-candidate arm is
    excluded from selection once the estimated prompt length reaches
    `length_routing.threshold_tokens`; below it, the ordinary weighted (or
    sticky) draw is completely untouched -- this is a guard-rail, not a
    promoter. `estimator="chars_div4"` (the only one shipped): a message
    whose `content` is `tokens * 4` characters estimates to exactly
    `tokens` tokens (integer division with no remainder), which is what
    `_message_of_estimated_tokens` below relies on for exact boundary
    tests."""

    def _backends(self, *, baseline_weight=50, candidate_weight=50):
        return (
            _backend(id="trailbrake-baseline", tier="fast", weight=baseline_weight, role="baseline", rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", weight=candidate_weight, role="candidate"),
        )

    @staticmethod
    def _message_of_estimated_tokens(tokens: int) -> dict:
        return {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}

    @staticmethod
    def _message_with_retrieval_shape() -> dict:
        return {
            "messages": [{
                "role": "user",
                "content": (
                    " ".join(["Context paragraph."] * 600)
                    + "\n\nBased ONLY on the document excerpt above, what is the key value? "
                    + "Answer concisely."
                ),
            }]
        }

    @staticmethod
    def _message_with_generative_shape() -> dict:
        return {
            "messages": [{
                "role": "user",
                "content": (
                    "Please implement a small patch in src/router/backends.py to "
                    "update the length-routing classifier."
                ),
            }]
        }

    # -- (a) disabled: byte-identical to before this feature existed -----

    def test_disabled_leaves_the_weighted_draw_and_decision_log_untouched(self):
        harness = _Harness(self._backends(), length_routing=LengthRoutingConfig(enabled=False))
        try:
            for _ in range(20):
                harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
                harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            picks = set()
            for _ in range(20):
                # An enormous prompt -- would trip any sane threshold if the
                # feature were on -- must not suppress the candidate while
                # length_routing.enabled is False.
                result = harness.router.dispatch_chat(self._message_of_estimated_tokens(50_000))
                picks.add(result.backend_id)
            self.assertEqual(picks, {"trailbrake-baseline", "trailbrake-candidate"})
            for record in harness.log_lines():
                self.assertFalse(
                    any(key.startswith("length_routing") for key in record),
                    f"unexpected length_routing_* key while disabled: {record!r}",
                )
        finally:
            harness.close()

    def test_enabled_kind_unaware_uses_single_threshold_only(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(
                enabled=True,
                kind_aware=False,
                threshold_tokens=100,
                kind_thresholds={
                    "generative": 10_000,
                    "retrieval": 10_000,
                    "multiturn": 10_000,
                    "unknown": 10_000,
                },
            ),
        )
        try:
            # Old scalar threshold would exclude the candidate; all per-kind
            # thresholds are higher in this config, so this is only possible
            # if kind-aware mode is fully off.
            for _ in range(10):
                harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(self._message_with_retrieval_shape())
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()

    def test_kind_aware_generative_uses_generative_threshold(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(
                enabled=True,
                kind_aware=True,
                threshold_tokens=100,
                kind_thresholds={
                    "generative": 10,
                    "retrieval": 10_000,
                    "multiturn": 10_000,
                    "unknown": 10_000,
                },
            ),
        )
        try:
            # Generative request should classify as "generative" and be
            # excluded by the lower generative threshold.
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(self._message_with_generative_shape())
            self.assertEqual(result.backend_id, "trailbrake-baseline")
            self.assertFalse(result.canary)
        finally:
            harness.close()

    def test_kind_aware_retrieval_respects_retrieval_threshold(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(
                enabled=True,
                kind_aware=True,
                threshold_tokens=10,
                kind_thresholds={
                    "generative": 10,
                    "retrieval": 10_000,
                    "multiturn": 10,
                    "unknown": 10,
                },
            ),
        )
        try:
            # Retrieval request should classify as "retrieval" and pass to
            # candidate due high retrieval threshold.
            harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(self._message_with_retrieval_shape())
            self.assertEqual(result.backend_id, "trailbrake-candidate")
            self.assertTrue(result.canary)
        finally:
            harness.close()

    def test_length_routing_kind_is_recorded_when_kind_aware(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(
                enabled=True,
                kind_aware=True,
                threshold_tokens=10,
                kind_thresholds={"generative": 10_000, "retrieval": 10_000, "multiturn": 10_000, "unknown": 10},
            ),
        )
        try:
            # Unknown kind should use the unknown threshold and record it.
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            harness.router.dispatch_chat({
                "messages": [{"role": "user", "content": "x " * 25 + "Please provide feedback and a brief summary."}]
            })
            record = harness.log_lines()[0]
            self.assertEqual(record["length_routing_kind"], "unknown")
        finally:
            harness.close()

    # -- (b) enabled + short prompt: weighted draw still reaches the candidate --

    def test_enabled_short_prompt_leaves_candidate_reachable_via_weighted_draw(self):
        harness = _Harness(
            self._backends(baseline_weight=50, candidate_weight=50),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=4096),
        )
        try:
            for _ in range(20):
                harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
                harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            picks = set()
            for _ in range(20):
                result = harness.router.dispatch_chat(self._message_of_estimated_tokens(10))  # far under 4096
                picks.add(result.backend_id)
            self.assertEqual(picks, {"trailbrake-baseline", "trailbrake-candidate"})
        finally:
            harness.close()

    # -- (c) enabled + long prompt: candidate never selected, reason logged --

    def test_enabled_long_prompt_never_selects_the_candidate_and_records_the_reason(self):
        harness = _Harness(
            # Candidate weighted 100 vs. baseline's 0 -- without the
            # exclusion, ordinary weighted selection would pick the
            # candidate essentially every time (see CanaryTests'
            # equivalent 100%-weight setup) -- so baseline winning every
            # draw here proves the EXCLUSION is doing the work, not weight.
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=4096),
        )
        try:
            for _ in range(30):
                harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            for _ in range(30):
                result = harness.router.dispatch_chat(self._message_of_estimated_tokens(5210))
                self.assertEqual(result.backend_id, "trailbrake-baseline")
                self.assertFalse(result.canary)
            record = harness.log_lines()[0]
            self.assertTrue(record["length_routing_excluded"])
            self.assertEqual(record["length_routing_estimated_tokens"], 5210)
            self.assertEqual(
                record["length_routing_reason"], "length_routing: 5210tok >= 4096 -> candidate excluded"
            )
        finally:
            harness.close()

    # -- (d) threshold boundary, exact -----------------------------------

    def test_threshold_boundary_exact_tokens_excludes_the_candidate(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=100),
        )
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(self._message_of_estimated_tokens(100))  # == threshold
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()

    def test_one_token_under_threshold_still_reaches_the_candidate(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=100),
        )
        try:
            harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(self._message_of_estimated_tokens(99))  # threshold - 1
            self.assertEqual(result.backend_id, "trailbrake-candidate")
        finally:
            harness.close()

    # -- (e) estimator math: multi-message + multimodal content shapes ---

    def test_estimate_sums_across_multiple_messages_end_to_end(self):
        # Three 200-char messages == 600 chars == 150 estimated tokens --
        # none alone would cross the 120-token threshold, only their sum
        # does, proving the estimate is computed over the whole request.
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=120),
        )
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            request = {"messages": [
                {"role": "system", "content": "s" * 200},
                {"role": "user", "content": "u" * 200},
                {"role": "assistant", "content": "a" * 200},
            ]}
            result = harness.router.dispatch_chat(request)
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()

    def test_estimate_tolerates_multimodal_content_parts_shape_end_to_end(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=50),
        )
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            request = {"messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "t" * 240},  # 240 chars -> 60 tokens, over the 50 threshold
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
                ],
            }]}
            result = harness.router.dispatch_chat(request)
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()

    # -- (f) sticky_key interaction: length exclusion applies first ------

    def test_sticky_key_that_would_hash_to_the_candidate_is_still_excluded_for_a_long_prompt(self):
        backends = self._backends(baseline_weight=50, candidate_weight=50)
        # A sticky key that hash-buckets to the candidate under ordinary
        # (length-routing-blind) selection -- same helper InTierFailoverTests
        # uses below.
        sticky_key = _find_sticky_key_for(backends, "fast", "trailbrake-candidate")

        harness = _Harness(backends, length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=4096))
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(
                self._message_of_estimated_tokens(5210), sticky_key=sticky_key
            )
            # Correctness beats stickiness: this exact key would normally
            # hash to the candidate, but a long prompt must still exclude
            # it -- see docs/DESIGN.md's sticky-key note.
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()

    def test_sticky_key_with_a_short_prompt_is_unaffected(self):
        # Same sticky key, same config -- only the prompt length differs --
        # must still land on the candidate it normally hashes to.
        backends = self._backends(baseline_weight=50, candidate_weight=50)
        sticky_key = _find_sticky_key_for(backends, "fast", "trailbrake-candidate")

        harness = _Harness(backends, length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=4096))
        try:
            harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat(
                self._message_of_estimated_tokens(10), sticky_key=sticky_key
            )
            self.assertEqual(result.backend_id, "trailbrake-candidate")
        finally:
            harness.close()

    # -- manual override bypasses length routing (documented in DESIGN.md) --

    def test_manual_backend_override_bypasses_length_routing_exclusion(self):
        # A client naming the candidate backend directly is this router's
        # established highest-precedence escape hatch -- length routing
        # must not silently defeat it.
        harness = _Harness(
            self._backends(baseline_weight=50, candidate_weight=50),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=100),
        )
        try:
            harness.transports["trailbrake-candidate"].queue_response(200, chat_response_body())
            request = self._message_of_estimated_tokens(5210)
            request["model"] = "trailbrake-candidate"
            result = harness.router.dispatch_chat(request)
            self.assertEqual(result.backend_id, "trailbrake-candidate")
            self.assertEqual(result.decision.trigger, "manual_override")
        finally:
            harness.close()

    # -- streaming integration: extra reaches the log via finalize_stream --

    def test_streamed_dispatch_records_the_length_routing_reason_on_finalize(self):
        harness = _Harness(
            self._backends(baseline_weight=0, candidate_weight=100),
            length_routing=LengthRoutingConfig(enabled=True, threshold_tokens=4096),
        )
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
            request = self._message_of_estimated_tokens(5210)
            request["stream"] = True
            result = harness.router.dispatch_chat_stream(request)
            self.assertEqual(result.backend_id, "trailbrake-baseline")
            result.finalize_stream(StreamOutcome.SUCCESS)
            record = harness.log_lines()[0]
            self.assertTrue(record["length_routing_excluded"])
            self.assertEqual(
                record["length_routing_reason"], "length_routing: 5210tok >= 4096 -> candidate excluded"
            )
        finally:
            harness.close()


class ManualBackendOverrideTests(unittest.TestCase):
    """Regression for the audit's "backend override isn't a real override":
    `resolve_manual_override` used to return only the tier, so normal
    weighted selection ran afterward and could still hand the request to a
    different backend in that tier."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", weight=0, role="baseline", rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", weight=100, role="candidate"),
        )
        self.harness = _Harness(backends)

    def tearDown(self):
        self.harness.close()

    def test_forced_backend_override_pins_exact_backend_despite_weighting(self):
        # trailbrake-candidate is weighted 100 vs. trailbrake-baseline's 0 -- ordinary
        # weighted selection would pick trailbrake-candidate essentially every
        # time. A client that names "trailbrake-baseline" directly must still get
        # trailbrake-baseline, every time.
        for _ in range(5):
            self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        for _ in range(5):
            result = self.harness.router.dispatch_chat({"model": "trailbrake-baseline", "messages": []})
            self.assertEqual(result.backend_id, "trailbrake-baseline")
            self.assertFalse(result.canary)

    def test_forced_backend_override_does_not_fail_over_to_a_different_backend_in_tier(self):
        # A forced backend that fails must not be silently replaced by a
        # different backend in the same tier (trailbrake-candidate here) -- that
        # would defeat the pin exactly like unconstrained weighted selection
        # did. Once trailbrake-baseline is excluded (tried once, this harness has
        # no fallback tier configured for "fast"), `_select_backend` reports
        # "no candidate," and the request surfaces as NoBackendAvailable
        # rather than quietly landing on trailbrake-candidate.
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        with self.assertRaises(NoBackendAvailable):
            self.harness.router.dispatch_chat({"model": "trailbrake-baseline", "messages": []})
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 1)
        self.assertEqual(len(self.harness.transports["trailbrake-candidate"].calls), 0)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_trailbrake_error_falls_back_to_iliria(self):
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body("engine crashed"))
        self.harness.transports["iliria"].queue_response(200, chat_response_body("fallback answer"))

        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "add a getter"}]})

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.fallback_from, "fast")

    def test_iliria_error_falls_back_to_trailbrake_baseline(self):
        self.harness.transports["iliria"].queue_response(503, error_body("queue full"))
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("fallback answer"))

        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "#deep help"}]})

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.fallback_from, "deep")

    def test_connection_refused_also_triggers_fallback(self):
        self.harness.transports["trailbrake-baseline"].queue_error(ConnectionRefusedError())
        self.harness.transports["iliria"].queue_response(200, chat_response_body())

        result = self.harness.router.dispatch_chat({"messages": []})
        self.assertEqual(result.backend_id, "iliria")

    def test_both_tiers_down_raises_no_backend_available(self):
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        self.harness.transports["iliria"].queue_response(500, error_body())

        with self.assertRaises(NoBackendAvailable):
            self.harness.router.dispatch_chat({"messages": []})

    def test_failure_and_fallback_are_both_logged(self):
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        self.harness.transports["iliria"].queue_response(200, chat_response_body())

        self.harness.router.dispatch_chat({"messages": []})

        statuses = [line["status"] for line in self.harness.log_lines()]
        self.assertEqual(statuses, ["backend_error", "ok"])

    def test_fallback_does_not_bounce_back_to_the_starting_tier(self):
        # fast -> deep is a legitimate hop; deep -> fast would bounce back to
        # where the request started, and must not happen even though
        # fallback={"fast":"deep","deep":"fast"} is written as a cycle.
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        self.harness.transports["iliria"].queue_response(500, error_body())

        with self.assertRaises(NoBackendAvailable):
            self.harness.router.dispatch_chat({"messages": []})

        # Exactly one attempt per tier: no infinite fast<->deep cycling.
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 1)
        self.assertEqual(len(self.harness.transports["iliria"].calls), 1)


class InTierFailoverTests(unittest.TestCase):
    """Two backends in the same tier: a failure on one retries the other
    before ever considering a cross-tier fallback."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", weight=50, role="baseline", rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", weight=50, role="candidate"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"}, failure_threshold=99)

    def tearDown(self):
        self.harness.close()

    def test_candidate_failure_retries_baseline_before_escalating(self):
        # Force the first pick deterministically via a sticky key hashed to
        # the candidate, then have it fail; baseline must be tried next,
        # and iliria must never be called at all.
        sticky_key = _find_sticky_key_for(self.harness.config.backends, "fast", "trailbrake-candidate")
        self.harness.transports["trailbrake-candidate"].queue_response(500, error_body())
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())

        result = self.harness.router.dispatch_chat({"messages": []}, sticky_key=sticky_key)

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertIsNone(result.fallback_from)  # still "fast" the whole time, not a cross-tier hop
        self.assertEqual(len(self.harness.transports["iliria"].calls), 0)


def _find_sticky_key_for(backends, tier: str, target_backend_id: str) -> str:
    """Brute-forces a sticky key that hash-buckets to `target_backend_id`
    within `tier`, so a test can deterministically force the "first pick"
    without reaching into `backends.py`'s private hashing."""
    from router.backends import select_backend

    for i in range(500):
        key = f"probe-{i}"
        chosen = select_backend(tier, backends, sticky_key=key)
        if chosen is not None and chosen.id == target_backend_id:
            return key
    raise AssertionError(f"could not find a sticky key routing to {target_backend_id!r}")


class CircuitBreakerIntegrationTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"}, failure_threshold=2)

    def tearDown(self):
        self.harness.close()

    def test_backend_is_excluded_after_reaching_failure_threshold(self):
        # Two failing requests trip the breaker; a third request should skip
        # straight to the fallback tier without even trying trailbrake-baseline.
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        self.harness.transports["iliria"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": []})  # failure #1, falls back to iliria

        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())
        self.harness.transports["iliria"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": []})  # failure #2, breaker opens

        self.harness.transports["iliria"].queue_response(200, chat_response_body())
        result = self.harness.router.dispatch_chat({"messages": []})  # breaker open: straight to iliria

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 2)  # not called a 3rd time


class StreamingDispatchTests(unittest.TestCase):
    def setUp(self):
        backends = (_backend(id="trailbrake-baseline", tier="fast", rollback_target=True),)
        self.harness = _Harness(backends)

    def tearDown(self):
        self.harness.close()

    def test_stream_dispatch_returns_an_open_response(self):
        body = chat_response_body("streamed")
        self.harness.transports["trailbrake-baseline"].queue_response(200, body, chunk_size=4)

        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})

        collected = b""
        while True:
            chunk = result.response.read_chunk()
            if not chunk:
                break
            collected += chunk
        self.assertEqual(collected, body)

    def test_stream_error_status_raises_before_any_chunk_is_relayed(self):
        # A 4xx arrives with the response headers, before any SSE chunk is
        # relayed, so it surfaces as the client's own error (relayed verbatim),
        # not a mid-stream failure. And a 4xx is a client error, so it raises
        # the 4xx directly rather than falling back: under the old
        # (health-poisoning) behavior this same single-fast-tier setup fell
        # back to an empty deep tier and raised NoBackendAvailable instead, so
        # getting BackendRequestFailed(404) here IS the proof no fallback ran.
        # Regression for the audit finding "client errors poison backend health."
        self.harness.transports["trailbrake-baseline"].queue_response(404, error_body("not found"))

        with self.assertRaises(BackendRequestFailed) as ctx:
            self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        self.assertEqual(ctx.exception.status, 404)

    def test_stream_outcome_is_deferred_then_finalized_once(self):
        # A streamed 200 records NO breaker result / log at header time -- the
        # body is relayed by the transport, so the real outcome isn't known
        # until finalize_stream is called (exactly once; a finally/cleanup
        # double-call must not double-count).
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        self.assertIsNotNone(result.finalize_stream)
        self.assertEqual(self.harness.log_lines(), [])  # nothing recorded yet
        result.finalize_stream(StreamOutcome.SUCCESS)
        result.finalize_stream(StreamOutcome.SUCCESS)  # single-shot
        self.assertEqual([r["status"] for r in self.harness.log_lines()], ["ok"])

    def test_stream_backend_failure_records_a_breaker_failure(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        result.finalize_stream(StreamOutcome.BACKEND_FAILURE)
        self.assertEqual([r["status"] for r in self.harness.log_lines()], ["stream_interrupted"])

    def test_stream_client_abort_leaves_the_backend_breaker_neutral(self):
        # A client hang-up is not the backend's fault: log it, but never trip
        # the (healthy) backend's breaker.
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        result.finalize_stream(StreamOutcome.CLIENT_ABORT)
        self.assertEqual([r["status"] for r in self.harness.log_lines()], ["client_disconnect"])
        self.assertTrue(self.harness.circuit_breakers.get("trailbrake-baseline").allow_request())

    def test_stream_telemetry_headers_are_captured_at_header_time_and_logged_on_finalize(self):
        # Headers arrive (and are captured) well before the finalizer ever
        # runs -- a streamed trailbrake response cannot carry decode-tok/s or
        # acceptance-rate headers (those are only known once generation
        # finishes, after headers must already have been sent -- see
        # trailbrake's _telemetry_response_headers docstring), but this proves
        # whatever headers DID arrive at header-time still reach the log
        # once the stream is finalized, regardless of outcome.
        self.harness.transports["trailbrake-baseline"].queue_response(
            200,
            chat_response_body("s"),
            headers={"content-type": "application/json", "x-trailbrake-draft-acceptance-rate": "0.5"},
            chunk_size=4,
        )
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        self.assertEqual(self.harness.log_lines(), [])  # nothing recorded yet
        result.finalize_stream(StreamOutcome.SUCCESS)
        record = self.harness.log_lines()[0]
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["draft_acceptance_rate"], 0.5)

    def test_stream_dispatch_forces_include_usage_upstream_even_when_client_never_set_it(self):
        # THE fix for "streamed traffic produces zero decision-log telemetry"
        # (the normal interactive case: a plain stream:true request, no
        # stream_options at all -- see _with_stream_usage_requested's
        # docstring). Without this, trailbrake never even computes the closing
        # usage event its own _stream_completion docstring describes, so no
        # amount of body-parsing on the router's side would have anything
        # to find.
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertEqual(sent["stream_options"], {"include_usage": True})

    def test_stream_dispatch_preserves_other_client_stream_options_while_forcing_include_usage(self):
        # A client value for a DIFFERENT stream_options key must survive
        # untouched -- only include_usage is ever forced.
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        self.harness.router.dispatch_chat_stream(
            {"messages": [], "stream": True, "stream_options": {"some_future_flag": "keep-me"}}
        )

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertEqual(sent["stream_options"], {"some_future_flag": "keep-me", "include_usage": True})

    def test_stream_dispatch_forces_include_usage_even_over_an_explicit_client_false(self):
        # Unlike _with_no_think_default, this is not "fill the gap the
        # client left" -- it is the router's own unconditional telemetry
        # need on its OWN backend connection, so even an explicit
        # include_usage=False from the client is overridden upstream (what
        # the router relays back to that client is a separate, unaffected
        # question -- see _relay_stream's docstring).
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        self.harness.router.dispatch_chat_stream(
            {"messages": [], "stream": True, "stream_options": {"include_usage": False}}
        )

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertIs(sent["stream_options"]["include_usage"], True)

    def test_non_stream_dispatch_never_gets_stream_options_injected(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        self.harness.router.dispatch_chat({"messages": []})

        sent = json.loads(self.harness.transports["trailbrake-baseline"].calls[0]["body"])
        self.assertNotIn("stream_options", sent)

    def test_stream_finalize_accepts_body_telemetry_parsed_by_the_transport_layer(self):
        # The other half of the fix: server.py's _relay_stream parses the
        # relayed SSE body (_extract_stream_usage_telemetry) and passes the
        # result as finalize_stream's second argument -- this is what
        # actually reaches the decision log for a real streamed request; no
        # real trailbrake response ever carries these as headers (see the test
        # right above this class's docstring note).
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        result.finalize_stream(
            StreamOutcome.SUCCESS,
            {
                "decode_tokens_per_second": 50.35,
                "time_to_first_token_seconds": 0.04596,
                "draft_acceptance_rate": 0.7333,
            },
        )
        record = self.harness.log_lines()[0]
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["decode_tokens_per_second"], 50.35)
        self.assertEqual(record["time_to_first_token_seconds"], 0.04596)
        self.assertEqual(record["draft_acceptance_rate"], 0.7333)

    def test_stream_finalize_body_telemetry_wins_over_header_telemetry_on_conflict(self):
        # Body-derived (the real, end-of-generation measurement) takes
        # precedence over header-derived (which, for a real trailbrake stream, is
        # always empty anyway -- see _extract_backend_telemetry's
        # docstring); this pins the merge direction explicitly rather than
        # leaving it implicit.
        self.harness.transports["trailbrake-baseline"].queue_response(
            200,
            chat_response_body("s"),
            headers={"content-type": "application/json", "x-trailbrake-draft-acceptance-rate": "0.5"},
            chunk_size=4,
        )
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        result.finalize_stream(StreamOutcome.SUCCESS, {"draft_acceptance_rate": 0.75})

        record = self.harness.log_lines()[0]
        self.assertEqual(record["draft_acceptance_rate"], 0.75)

    def test_stream_finalize_still_works_with_no_body_telemetry_argument_at_all(self):
        # Backward compatibility: an existing single-argument
        # finalize_stream(outcome) call site (this is exactly what every
        # other test in this class does) must keep working unchanged.
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("s"), chunk_size=4)
        result = self.harness.router.dispatch_chat_stream({"messages": [], "stream": True})
        result.finalize_stream(StreamOutcome.SUCCESS)
        record = self.harness.log_lines()[0]
        self.assertEqual(record["status"], "ok")
        self.assertNotIn("decode_tokens_per_second", record)


class WithStreamUsageRequestedTests(unittest.TestCase):
    """Pure-function tests for `_with_stream_usage_requested` -- no harness,
    no transport, mirroring ExtractBackendTelemetryTests' style below."""

    def test_non_stream_mode_is_untouched(self):
        body = {"messages": []}
        self.assertIs(_with_stream_usage_requested(body, "buffered"), body)

    def test_stream_mode_with_no_stream_options_forces_include_usage_true(self):
        result = _with_stream_usage_requested({"messages": [], "stream": True}, "stream")
        self.assertEqual(result["stream_options"], {"include_usage": True})

    def test_stream_mode_preserves_other_stream_options_keys(self):
        result = _with_stream_usage_requested(
            {"messages": [], "stream_options": {"keep": "me"}}, "stream"
        )
        self.assertEqual(result["stream_options"], {"keep": "me", "include_usage": True})

    def test_stream_mode_overrides_an_explicit_false(self):
        result = _with_stream_usage_requested(
            {"messages": [], "stream_options": {"include_usage": False}}, "stream"
        )
        self.assertIs(result["stream_options"]["include_usage"], True)

    def test_does_not_mutate_the_original_request_body(self):
        original = {"messages": [], "stream_options": {"keep": "me"}}
        _with_stream_usage_requested(original, "stream")
        self.assertEqual(original, {"messages": [], "stream_options": {"keep": "me"}})


class DraftThenEscalateDispatchTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        policy = DraftThenEscalatePolicy(DefaultPolicy(), verifier=lambda request, text: "TODO" not in text)
        self.harness = _Harness(backends, policy=policy, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_accepted_draft_is_returned_from_the_fast_tier(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("def f(): return 1"))

        result = self.harness.router.dispatch_chat_with_draft_verification({"messages": []})

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(len(self.harness.transports["iliria"].calls), 0)

    def test_rejected_draft_escalates_to_iliria(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("def f(): ...  # TODO"))
        self.harness.transports["iliria"].queue_response(200, chat_response_body("def f(): return 1"))

        result = self.harness.router.dispatch_chat_with_draft_verification({"messages": []})

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.decision.trigger, "draft_rejected")
        # Both the (rejected) draft and the final escalated answer are logged.
        statuses = [(line["backend_id"], line["status"]) for line in self.harness.log_lines()]
        self.assertEqual(statuses, [("trailbrake-baseline", "ok"), ("iliria", "ok")])


class MaxAttemptsSafetyBoundTests(unittest.TestCase):
    def test_pathological_all_failing_config_still_terminates(self):
        backends = (
            _backend(id="a", tier="fast", weight=50, role="baseline", rollback_target=True),
            _backend(id="b", tier="fast", weight=50, role="candidate"),
        )
        harness = _Harness(backends, failure_threshold=99)  # never trips, to force exhausting MAX_ATTEMPTS
        try:
            for transport in harness.transports.values():
                for _ in range(harness.router.MAX_ATTEMPTS):
                    transport.queue_response(500, error_body())
            with self.assertRaises(NoBackendAvailable):
                harness.router.dispatch_chat({"messages": []})
            total_calls = sum(len(t.calls) for t in harness.transports.values())
            self.assertLessEqual(total_calls, harness.router.MAX_ATTEMPTS)
        finally:
            harness.close()


class CircuitBreakerHalfOpenSafetyNetTests(unittest.TestCase):
    """Regression for the audit's circuit-breaker finding: `_run` must
    release a backend's one half-open circuit-breaker trial slot even when
    the backend call raises something that is neither `BackendRequestFailed`
    nor `OSError` (e.g. a raw `http.client.IncompleteRead`/`BadStatusLine`).
    Uses a tiny *real* `reset_after_s` and a short real sleep (rather than a
    fake clock) to cross the reset window, since `_Harness` wires up a real
    `CircuitBreakerRegistry` with no clock injection point."""

    def setUp(self):
        backends = (_backend(id="trailbrake-baseline", tier="fast", rollback_target=True),)
        self.harness = _Harness(backends, failure_threshold=1, reset_after_s=0.05)

    def tearDown(self):
        self.harness.close()

    def test_unexpected_exception_during_half_open_trial_does_not_wedge_the_breaker(self):
        # 1) Trip the breaker open with an ordinary (OSError) failure.
        self.harness.transports["trailbrake-baseline"].queue_error(ConnectionRefusedError())
        with self.assertRaises(NoBackendAvailable):
            self.harness.router.dispatch_chat({"messages": []})

        time.sleep(0.1)  # cross reset_after_s -> the next allow_request() claims the half-open trial

        # 2) The half-open trial itself raises a type _run does not
        # specifically recognize. It must still propagate (not be silently
        # swallowed or retried)...
        self.harness.transports["trailbrake-baseline"].queue_error(RuntimeError("unrecognized failure shape"))
        with self.assertRaises(RuntimeError):
            self.harness.router.dispatch_chat({"messages": []})

        time.sleep(0.1)  # cross reset_after_s again

        # 3) ...but the trial slot must have been released: a subsequent
        # request, after another reset window, must be allowed to try this
        # backend again rather than finding it permanently excluded. Without
        # the fix this raises NoBackendAvailable instead (the backend stays
        # wedged, since nothing ever cleared `_half_open_trial_in_flight`).
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("recovered"))
        result = self.harness.router.dispatch_chat({"messages": []})
        self.assertEqual(result.backend_id, "trailbrake-baseline")


class AlwaysTierPolicyDispatchTests(unittest.TestCase):
    def test_fixed_policy_pins_traffic_even_with_hard_markers_present(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        harness = _Harness(backends, policy=AlwaysTierPolicy("fast"))
        try:
            harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
            result = harness.router.dispatch_chat({"messages": [{"role": "user", "content": "#deep escalate please"}]})
            self.assertEqual(result.backend_id, "trailbrake-baseline")
        finally:
            harness.close()


# -- Red-team priority: the expensive failure is an escalation FALSE NEGATIVE --
#
# A hard-reasoning request that silently fails to escalate and gets a weak
# no-think answer is worse than a false-positive escalation (a merely slower
# answer) or a visible error (at least attributable/retriable): it looks
# exactly like a correct 200 OK. The classes below specifically target ways
# that could happen -- malformed/oversized/duplicate routing signals, and a
# hard-reasoning request whose escalation backend itself fails -- plus what
# FAIL-CLOSED means here: on a genuinely *malformed* signal, degrade
# gracefully (treat it as absent, never crash, so the remaining triggers
# still get a fair chance at the real text); on a *fuzzy-but-well-formed*
# signal that scores below threshold, staying fast is the intentional,
# already-red-teamed design (docs/DESIGN.md's cost-asymmetry argument,
# pinned by tests/test_policy.py's HardnessScoreTests); on an *internal*
# failure (policy bug, or both tiers down), fail visibly (raise / 5xx),
# never silently fabricate a default-tier "success".


class MalformedRoutingSignalTests(unittest.TestCase):
    """A malformed `reasoning_effort` value (a list or dict -- trivial for a
    buggy or adversarial client to send) used to raise an uncaught TypeError
    out of policy.decide() (bare `x in a_set` on an unhashable value),
    pre-empting the marker/heuristic checks entirely -- see policy.py's
    `DefaultPolicy.decide`. Fixed by requiring `isinstance(reasoning_effort,
    str)` before the set-membership check. These pin both halves: no crash,
    AND the other triggers still get their fair chance at the request's
    actual text (the concrete false-negative-prevention proof)."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_unhashable_reasoning_effort_does_not_crash_on_a_plain_request(self):
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body())
        result = self.harness.router.dispatch_chat(
            {"reasoning_effort": ["high"], "messages": [{"role": "user", "content": "add a getter"}]}
        )
        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.tier, "fast")

    def test_unhashable_reasoning_effort_does_not_suppress_a_hard_marker(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))
        result = self.harness.router.dispatch_chat({
            "reasoning_effort": {"level": "high"},
            "messages": [{"role": "user", "content": "#deep why does this race"}],
        })
        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(result.decision.trigger, "explicit_marker")

    def test_unhashable_reasoning_effort_does_not_suppress_the_task_heuristic(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))
        result = self.harness.router.dispatch_chat({
            "reasoning_effort": [1, 2, 3],
            "messages": [{"role": "user",
                          "content": "Why does this crash intermittently? Smells like a race condition."}],
        })
        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(result.decision.trigger, "task_heuristic")


class ConflictingRoutingSignalTests(unittest.TestCase):
    """Duplicate/conflicting routing signals within one request. A signal
    that does not *itself* force escalation (a low/none reasoning_effort)
    must not short-circuit or suppress a different, independent trigger (a
    hard marker) that also fires on the same request -- policy.py checks
    reasoning_effort, marker, and heuristic as three independent, sequential
    checks, not a single merged verdict, so this must hold by construction.
    Pinned here as an explicit regression guard against a future "optimize
    by early-returning on reasoning_effort" refactor that would silently
    reintroduce a false negative."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_explicit_reasoning_effort_none_does_not_suppress_a_hard_marker(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))
        result = self.harness.router.dispatch_chat({
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "#deep why does this race"}],
        })
        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.decision.trigger, "explicit_marker")

    def test_hardness_score_exactly_at_the_threshold_escalates(self):
        # `>=`, not `>`, in policy.DefaultPolicy.decide -- on an
        # exact-boundary ambiguous score (0.60 == the default threshold, see
        # "big-O" + "step by step" each weighing 0.3), the router leans
        # toward escalating rather than away from it.
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))
        message = "Explain the big-O complexity here, step by step."
        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": message}]})
        self.assertEqual(result.decision.trigger, "task_heuristic")
        self.assertEqual(result.backend_id, "iliria")


class OversizedSignalTests(unittest.TestCase):
    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast"),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_hard_signal_buried_in_a_very_large_message_still_escalates(self):
        # A well-formed but large message (filler text well under the
        # server's 8 MiB cap, see server.py's _MAX_BODY_BYTES) must not
        # dilute or hide a genuine hard signal buried inside it --
        # hardness_score/find_hard_marker must scan the whole thing, not
        # silently truncate or time out (this project's own security review
        # already checked -- and passed -- ReDoS against these patterns).
        filler = "The quick brown fox jumps over the lazy dog. " * 20000  # ~940 KB
        content = filler + " Why does this deadlock intermittently? " + filler
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep answer"))

        started = time.monotonic()
        result = self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": content}]})
        elapsed = time.monotonic() - started

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.decision.trigger, "task_heuristic")
        self.assertLess(elapsed, 2.0, "hardness scoring must not be pathologically slow on a large message")


class EscalationFailureAttributionTests(unittest.TestCase):
    """When the ESCALATION backend itself fails (error, timeout) and the
    request degrades to a fallback answer from trailbrake, the fact that this was
    a hard-reasoning request that *should* have gone to iliria must remain
    attributable in the decision (`decision.trigger` stays
    "explicit_marker"/"task_heuristic", and `fallback_from="deep"`) -- not
    silently indistinguishable from an ordinary request that never intended
    to escalate at all. Without this, a weak no-think answer for a hard
    question caused by a backend outage looks, downstream, exactly like the
    policy's own (intentional) decision to stay fast."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_iliria_error_after_hard_signal_falls_back_but_keeps_escalation_attribution(self):
        self.harness.transports["iliria"].queue_response(500, error_body("engine crashed"))
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("weak fallback answer"))

        result = self.harness.router.dispatch_chat(
            {"messages": [{"role": "user", "content": "#deep why does this race"}]}
        )

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.fallback_from, "deep")
        self.assertEqual(result.decision.trigger, "explicit_marker")  # still attributable, not "default"

    def test_iliria_timeout_after_hard_signal_falls_back_and_keeps_attribution(self):
        # A connect/idle timeout (TimeoutError, an OSError subclass) is the
        # single most realistic iliria failure mode in production -- its
        # own decode is slow enough (~1.6 tok/s) that a genuine timeout is
        # far more plausible than a connection refusal. Distinct from the
        # ConnectionRefusedError FallbackTests already covers.
        self.harness.transports["iliria"].queue_error(TimeoutError("timed out"))
        self.harness.transports["trailbrake-baseline"].queue_response(200, chat_response_body("weak fallback answer"))

        result = self.harness.router.dispatch_chat(
            {"reasoning_effort": "high", "messages": [{"role": "user", "content": "add a getter"}]}
        )

        self.assertEqual(result.backend_id, "trailbrake-baseline")
        self.assertEqual(result.fallback_from, "deep")
        self.assertEqual(result.decision.trigger, "explicit_marker")

    def test_both_tiers_failing_after_a_hard_signal_is_a_visible_error_not_a_silent_default(self):
        # FAIL-CLOSED: when neither tier can serve a hard-reasoning request
        # at all, the router must raise (surfacing as a 503 at the HTTP
        # layer -- see ChatCompletionRoutingTests' analogous 503 test),
        # never silently fabricate a default-tier "success".
        self.harness.transports["iliria"].queue_response(500, error_body())
        self.harness.transports["trailbrake-baseline"].queue_response(500, error_body())

        with self.assertRaises(NoBackendAvailable):
            self.harness.router.dispatch_chat({"messages": [{"role": "user", "content": "#deep why does this race"}]})


class EscalatedStreamingTests(unittest.TestCase):
    """StreamingDispatchTests (above) only wires a single fast-tier backend,
    so streaming behavior for an ESCALATED (hard-signal, "deep"-tier)
    request had no coverage at all before this class -- neither the happy
    path nor a partial/aborted decode. A hard-reasoning request that
    escalates correctly but then has its stream interrupted mid-decode must
    not be misattributed as an ordinary successful no-think answer."""

    def setUp(self):
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", rollback_target=True),
            _backend(id="iliria", tier="deep", model_id="glm-5.2-iliria", rollback_target=True),
        )
        self.harness = _Harness(backends, fallback={"fast": "deep", "deep": "fast"})

    def tearDown(self):
        self.harness.close()

    def test_hard_signal_stream_request_opens_against_iliria(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep streamed"), chunk_size=4)

        result = self.harness.router.dispatch_chat_stream(
            {"messages": [{"role": "user", "content": "#deep why does this race"}], "stream": True}
        )

        self.assertEqual(result.backend_id, "iliria")
        self.assertEqual(result.tier, "deep")
        self.assertEqual(len(self.harness.transports["trailbrake-baseline"].calls), 0)
        result.finalize_stream(StreamOutcome.SUCCESS)

    def test_partial_decode_failure_on_an_escalated_stream_is_logged_as_interrupted(self):
        self.harness.transports["iliria"].queue_response(200, chat_response_body("deep streamed"), chunk_size=4)

        result = self.harness.router.dispatch_chat_stream(
            {"messages": [{"role": "user", "content": "#deep why does this race"}], "stream": True}
        )
        result.finalize_stream(StreamOutcome.BACKEND_FAILURE)

        self.assertEqual([r["status"] for r in self.harness.log_lines()], ["stream_interrupted"])
        record = self.harness.log_lines()[0]
        self.assertEqual(record["trigger"], "explicit_marker")
        self.assertEqual(record["tier"], "deep")


class PolicyFailureIsClosedNotSilentTests(unittest.TestCase):
    """FAIL-CLOSED, explicitly stated: if the escalation policy itself
    raises (a genuine bug, distinct from the malformed-input cases
    MalformedRoutingSignalTests already covers), the dispatch layer must
    propagate that failure rather than swallow it into a silent
    default-tier "success" -- an internal error must be visible (surfacing
    as an HTTP 500 at the server layer; policy.decide() is called outside
    any try/except in dispatch.RequestRouter._run), never a quiet
    wrong-tier answer. This is the correct trade-off here: an explicit
    failure can be retried or alerted on; a silently-defaulted
    hard-reasoning request cannot be told apart from an intentional one."""

    class _ExplodingPolicy:
        def decide(self, request, config):
            raise ValueError("simulated policy bug")

    def test_a_policy_exception_propagates_instead_of_silently_defaulting(self):
        backends = (_backend(id="trailbrake-baseline", tier="fast", rollback_target=True),)
        harness = _Harness(backends, policy=self._ExplodingPolicy())
        try:
            with self.assertRaises(ValueError):
                harness.router.dispatch_chat({"messages": []})
            # Nothing was silently sent to a backend as a fallback guess either.
            self.assertEqual(len(harness.transports["trailbrake-baseline"].calls), 0)
        finally:
            harness.close()


class ExtractBackendTelemetryTests(unittest.TestCase):
    """Pure-function tests for the optional X-trailbrake-* header extractor -- no
    harness, no transport, no backend identity involved (see the function's
    own docstring: this module still knows neither backend's name)."""

    def test_no_headers_present_yields_an_empty_dict(self):
        self.assertEqual(_extract_backend_telemetry({}), {})

    def test_unrelated_headers_are_ignored(self):
        headers = {"content-type": "application/json", "x-router-request-id": "rtr_abc"}
        self.assertEqual(_extract_backend_telemetry(headers), {})

    def test_known_headers_are_parsed_as_floats(self):
        headers = {
            "x-trailbrake-decode-tokens-per-second": "50.35",
            "x-trailbrake-ttft-seconds": "0.04596",
            "x-trailbrake-draft-acceptance-rate": "0.7333",
        }
        self.assertEqual(
            _extract_backend_telemetry(headers),
            {
                "decode_tokens_per_second": 50.35,
                "time_to_first_token_seconds": 0.04596,
                "draft_acceptance_rate": 0.7333,
            },
        )

    def test_partial_headers_yield_only_the_present_fields(self):
        headers = {"x-trailbrake-draft-acceptance-rate": "0.9"}
        self.assertEqual(_extract_backend_telemetry(headers), {"draft_acceptance_rate": 0.9})

    def test_unparseable_value_is_skipped_not_raised(self):
        headers = {"x-trailbrake-decode-tokens-per-second": "not-a-number"}
        self.assertEqual(_extract_backend_telemetry(headers), {})


if __name__ == "__main__":
    unittest.main()
