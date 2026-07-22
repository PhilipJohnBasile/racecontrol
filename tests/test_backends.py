from __future__ import annotations

import json
import unittest

from router.backends import (
    BackendClient,
    drain,
    classify_prompt_kind,
    estimate_prompt_tokens,
    length_routing_excluded_ids,
    select_backend,
)
from router.config import BackendConfig, LengthRoutingConfig
from router.errors import BackendRequestFailed

from .fakes import FakeTransport, chat_response_body, error_body


def _backend(**overrides) -> BackendConfig:
    defaults = dict(
        id="trailbrake-baseline", tier="fast", base_url="http://127.0.0.1:8080",
        model_id="default", weight=100, enabled=True, role="baseline", rollback_target=True,
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


class ModelRewriteTests(unittest.TestCase):
    """The router's own virtual model name must never reach a backend --
    each backend gets its own configured `model_id` (trailbrake tolerates loose
    aliases; iliria requires an exact match -- see backends.py's module
    docstring)."""

    def test_client_supplied_model_is_replaced(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(model_id="glm-5.2-iliria"), transport=transport)

        client.chat_completions({"model": "deep", "messages": [{"role": "user", "content": "hi"}]})

        sent_body = json.loads(transport.calls[0]["body"])
        self.assertEqual(sent_body["model"], "glm-5.2-iliria")

    def test_missing_model_field_is_still_populated(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(), transport=transport)

        client.chat_completions({"messages": [{"role": "user", "content": "hi"}]})

        sent_body = json.loads(transport.calls[0]["body"])
        self.assertEqual(sent_body["model"], "default")

    def test_other_fields_pass_through_unchanged(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(), transport=transport)

        client.chat_completions({"model": "x", "temperature": 0.2, "max_tokens": 64})

        sent_body = json.loads(transport.calls[0]["body"])
        self.assertEqual(sent_body["temperature"], 0.2)
        self.assertEqual(sent_body["max_tokens"], 64)


class AuthHeaderTests(unittest.TestCase):
    def test_no_api_key_means_no_authorization_header(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(api_key=None), transport=transport)

        client.chat_completions({})

        self.assertNotIn("Authorization", transport.calls[0]["headers"])

    def test_configured_api_key_is_sent_as_bearer(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(api_key="secret123"), transport=transport)

        client.chat_completions({})

        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer secret123")


class TimeoutWiringTests(unittest.TestCase):
    def test_configured_timeouts_are_forwarded(self):
        transport = FakeTransport()
        transport.queue_response(200, chat_response_body())
        client = BackendClient(_backend(connect_timeout_s=1.5, idle_timeout_s=42.0), transport=transport)

        client.chat_completions({})

        self.assertEqual(transport.calls[0]["connect_timeout_s"], 1.5)
        self.assertEqual(transport.calls[0]["idle_timeout_s"], 42.0)


class ChatCompletionsErrorHandlingTests(unittest.TestCase):
    def test_4xx_raises_backend_request_failed_with_body_decoded(self):
        transport = FakeTransport()
        transport.queue_response(404, error_body("model not found"))
        client = BackendClient(_backend(), transport=transport)

        with self.assertRaises(BackendRequestFailed) as ctx:
            client.chat_completions({})
        self.assertEqual(ctx.exception.status, 404)
        self.assertIn("model not found", ctx.exception.message)
        self.assertEqual(ctx.exception.backend_id, "trailbrake-baseline")

    def test_5xx_raises_backend_request_failed(self):
        transport = FakeTransport()
        transport.queue_response(500, error_body("engine crashed"))
        client = BackendClient(_backend(), transport=transport)

        with self.assertRaises(BackendRequestFailed):
            client.chat_completions({})

    def test_response_is_closed_even_on_error(self):
        transport = FakeTransport()
        transport.queue_response(500, error_body())
        client = BackendClient(_backend(), transport=transport)
        opened_responses = []
        original_open = client.open

        def _spy_open(body):
            opened = original_open(body)
            opened_responses.append(opened)
            return opened

        client.open = _spy_open
        with self.assertRaises(BackendRequestFailed):
            client.chat_completions({})
        self.assertTrue(opened_responses[0].closed)

    def test_connection_error_propagates_uncaught(self):
        transport = FakeTransport()
        transport.queue_error(ConnectionRefusedError("no one home"))
        client = BackendClient(_backend(), transport=transport)

        with self.assertRaises(ConnectionRefusedError):
            client.chat_completions({})


class ChunkedBodyDrainTests(unittest.TestCase):
    def test_multi_chunk_response_is_fully_assembled(self):
        transport = FakeTransport()
        body = chat_response_body("a longer response body than one chunk")
        transport.queue_response(200, body, chunk_size=5)
        client = BackendClient(_backend(), transport=transport)

        response = client.chat_completions({})

        self.assertEqual(response.body, body)

    def test_drain_stops_at_empty_chunk(self):
        transport = FakeTransport()
        transport.queue_response(200, b"abc", chunk_size=1)
        opened = transport.open(
            base_url="http://x", method="GET", path="/", body=None, headers={},
            connect_timeout_s=1.0, idle_timeout_s=1.0,
        )
        self.assertEqual(drain(opened), b"abc")
        self.assertEqual(opened.read_chunk(), b"")  # exhausted, not an error


class HealthCheckTests(unittest.TestCase):
    def test_200_is_healthy(self):
        transport = FakeTransport()
        transport.queue_response(200, b'{"status":"ok"}')
        client = BackendClient(_backend(), transport=transport)
        self.assertTrue(client.health())

    def test_non_200_is_unhealthy(self):
        transport = FakeTransport()
        transport.queue_response(503, b'{"status":"down"}')
        client = BackendClient(_backend(), transport=transport)
        self.assertFalse(client.health())

    def test_connection_error_is_unhealthy_not_raised(self):
        transport = FakeTransport()
        transport.queue_error(OSError("connection refused"))
        client = BackendClient(_backend(), transport=transport)
        self.assertFalse(client.health())


class SelectBackendTests(unittest.TestCase):
    def test_single_enabled_backend_is_always_chosen(self):
        backends = (_backend(id="only"),)
        chosen = select_backend("fast", backends)
        self.assertEqual(chosen.id, "only")

    def test_disabled_backends_are_never_chosen(self):
        backends = (_backend(id="a", enabled=False), _backend(id="b", enabled=True))
        for _ in range(20):
            self.assertEqual(select_backend("fast", backends).id, "b")

    def test_wrong_tier_backends_are_ignored(self):
        backends = (_backend(id="a", tier="deep"), _backend(id="b", tier="fast"))
        self.assertEqual(select_backend("fast", backends).id, "b")

    def test_no_eligible_backend_returns_none(self):
        backends = (_backend(id="a", enabled=False),)
        self.assertIsNone(select_backend("fast", backends))

    def test_exclude_ids_removes_a_candidate(self):
        backends = (_backend(id="a"), _backend(id="b"))
        for _ in range(20):
            self.assertEqual(select_backend("fast", backends, exclude_ids=frozenset({"a"})).id, "b")

    def test_zero_weight_candidate_100_pct_of_the_time_is_never_picked(self):
        backends = (_backend(id="baseline", weight=100), _backend(id="candidate", weight=0))
        picks = {select_backend("fast", backends).id for _ in range(200)}
        self.assertEqual(picks, {"baseline"})

    def test_canary_weight_is_respected_statistically(self):
        backends = (_backend(id="baseline", weight=90), _backend(id="candidate", weight=10))
        picks = [select_backend("fast", backends).id for _ in range(4000)]
        candidate_share = picks.count("candidate") / len(picks)
        # Statistical, not exact -- generous band so this never flakes.
        self.assertTrue(0.05 < candidate_share < 0.16, candidate_share)

    def test_all_zero_weight_falls_back_to_uniform_not_broken(self):
        backends = (_backend(id="a", weight=0), _backend(id="b", weight=0))
        chosen = select_backend("fast", backends)
        self.assertIn(chosen.id, {"a", "b"})

    def test_sticky_key_is_deterministic(self):
        backends = (_backend(id="a"), _backend(id="b"), _backend(id="c"))
        first = select_backend("fast", backends, sticky_key="user-42").id
        for _ in range(20):
            self.assertEqual(select_backend("fast", backends, sticky_key="user-42").id, first)

    def test_different_sticky_keys_can_land_differently(self):
        backends = (_backend(id="a"), _backend(id="b"), _backend(id="c"), _backend(id="d"))
        picks = {select_backend("fast", backends, sticky_key=f"user-{i}").id for i in range(50)}
        # Not asserting a specific distribution, just that hashing actually
        # spreads across more than one backend for 50 distinct keys.
        self.assertGreater(len(picks), 1)

    def test_sticky_key_respects_exclude_ids(self):
        backends = (_backend(id="a"), _backend(id="b"))
        chosen = select_backend("fast", backends, sticky_key="user-42", exclude_ids=frozenset({"a"}))
        self.assertEqual(chosen.id, "b")


class EstimatePromptTokensTests(unittest.TestCase):
    """Pure-function tests for the "chars_div4" length estimator -- see
    docs/DESIGN.md's "Length-aware routing" section. No harness, no config,
    matching ExtractBackendTelemetryTests' style in tests/test_dispatch.py."""

    def test_empty_messages_list_is_zero(self):
        self.assertEqual(estimate_prompt_tokens([]), 0)

    def test_none_messages_is_zero_not_raised(self):
        self.assertEqual(estimate_prompt_tokens(None), 0)

    def test_single_message_divides_char_count_by_four(self):
        messages = [{"role": "user", "content": "12345678"}]  # 8 chars
        self.assertEqual(estimate_prompt_tokens(messages), 2)

    def test_integer_division_rounds_down(self):
        messages = [{"role": "user", "content": "12345"}]  # 5 chars -> 1, not 1.25
        self.assertEqual(estimate_prompt_tokens(messages), 1)

    def test_sums_across_every_message_not_just_the_last_user_turn(self):
        # Unlike policy.hardness_score (which only looks at recent USER
        # turns), the length estimate is meant to approximate the whole
        # prompt the backend will actually see -- system + assistant turns
        # included.
        messages = [
            {"role": "system", "content": "s" * 40},
            {"role": "user", "content": "u" * 20},
            {"role": "assistant", "content": "a" * 20},
            {"role": "user", "content": "u" * 20},
        ]
        self.assertEqual(estimate_prompt_tokens(messages), 100 // 4)

    def test_multimodal_content_parts_list_sums_text_parts_only(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 12},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
                {"type": "text", "text": "y" * 8},
            ],
        }]
        self.assertEqual(estimate_prompt_tokens(messages), 20 // 4)

    def test_image_only_content_contributes_zero(self):
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        self.assertEqual(estimate_prompt_tokens(messages), 0)

    def test_message_with_no_content_key_contributes_zero_not_raised(self):
        # e.g. an assistant tool-call message that carries `tool_calls`
        # instead of `content`.
        messages = [{"role": "assistant", "tool_calls": []}, {"role": "user", "content": "abcd"}]
        self.assertEqual(estimate_prompt_tokens(messages), 1)

    def test_content_of_an_unexpected_type_contributes_zero_not_raised(self):
        messages = [{"role": "user", "content": 12345}]
        self.assertEqual(estimate_prompt_tokens(messages), 0)

    def test_non_dict_message_is_skipped_not_raised(self):
        messages = ["not-a-message-object", {"role": "user", "content": "abcd"}]
        self.assertEqual(estimate_prompt_tokens(messages), 1)

    def test_unknown_estimator_raises(self):
        with self.assertRaises(ValueError):
            estimate_prompt_tokens([{"role": "user", "content": "hi"}], estimator="not-a-real-estimator")


def _length_routing(**overrides) -> LengthRoutingConfig:
    defaults = dict(enabled=True, threshold_tokens=4096, estimator="chars_div4")
    defaults.update(overrides)
    return LengthRoutingConfig(**defaults)


class LengthRoutingExcludedIdsTests(unittest.TestCase):
    """Pure-function tests for the length-routing exclusion rule itself --
    see docs/DESIGN.md's "Length-aware routing" section. `dispatch.py`'s
    `RequestRouter._select_backend` folds the returned ids into
    `select_backend`'s own `exclude_ids` before it runs; these tests pin
    down the *rule*, tests/test_dispatch.py's LengthAwareRoutingTests pin
    down the end-to-end wiring."""

    def setUp(self):
        self.backends = (
            _backend(id="trailbrake-baseline", tier="fast", role="baseline", weight=90, rollback_target=True),
            _backend(id="trailbrake-candidate", tier="fast", role="candidate", weight=10),
        )

    def test_disabled_never_excludes_regardless_of_length(self):
        ids, reason = length_routing_excluded_ids(
            "fast", self.backends, length_routing=_length_routing(enabled=False), estimated_tokens=999_999,
        )
        self.assertEqual(ids, frozenset())
        self.assertIsNone(reason)

    def test_below_threshold_excludes_nothing(self):
        ids, reason = length_routing_excluded_ids(
            "fast", self.backends, length_routing=_length_routing(threshold_tokens=4096), estimated_tokens=4095,
        )
        self.assertEqual(ids, frozenset())
        self.assertIsNone(reason)

    def test_at_threshold_boundary_excludes_the_candidate(self):
        # `>=`, not `>` -- matches this codebase's existing tie-breaking
        # convention (policy.DefaultPolicy: "hardness_score() >= threshold
        # escalates"; on an exact-boundary case, lean toward the safer
        # option, here "exclude the candidate").
        ids, reason = length_routing_excluded_ids(
            "fast", self.backends, length_routing=_length_routing(threshold_tokens=4096), estimated_tokens=4096,
        )
        self.assertEqual(ids, frozenset({"trailbrake-candidate"}))
        self.assertIsNotNone(reason)

    def test_one_below_threshold_boundary_excludes_nothing(self):
        ids, reason = length_routing_excluded_ids(
            "fast", self.backends, length_routing=_length_routing(threshold_tokens=4096), estimated_tokens=4095,
        )
        self.assertEqual(ids, frozenset())
        self.assertIsNone(reason)

    def test_above_threshold_excludes_the_candidate_with_exact_reason_text(self):
        ids, reason = length_routing_excluded_ids(
            "fast", self.backends, length_routing=_length_routing(threshold_tokens=4096), estimated_tokens=5210,
        )
        self.assertEqual(ids, frozenset({"trailbrake-candidate"}))
        self.assertEqual(reason, "length_routing: 5210tok >= 4096 -> candidate excluded")

    def test_no_candidate_role_backend_in_tier_is_a_no_op(self):
        backends = (_backend(id="trailbrake-baseline", tier="fast", role="baseline"),)
        ids, reason = length_routing_excluded_ids(
            "fast", backends, length_routing=_length_routing(threshold_tokens=100), estimated_tokens=999,
        )
        self.assertEqual(ids, frozenset())
        self.assertIsNone(reason)

    def test_a_disabled_candidate_is_not_reported_as_excluded(self):
        # A disabled candidate is already unreachable via select_backend's
        # own .enabled filter regardless of length routing -- attributing
        # that to length routing would be misleading telemetry (see
        # length_routing_excluded_ids's docstring).
        backends = (
            _backend(id="trailbrake-baseline", tier="fast", role="baseline"),
            _backend(id="trailbrake-candidate", tier="fast", role="candidate", enabled=False),
        )
        ids, reason = length_routing_excluded_ids(
            "fast", backends, length_routing=_length_routing(threshold_tokens=100), estimated_tokens=999,
        )
        self.assertEqual(ids, frozenset())
        self.assertIsNone(reason)

    def test_only_excludes_candidates_in_the_requested_tier(self):
        backends = self.backends + (_backend(id="iliria-candidate", tier="deep", role="candidate"),)
        ids, _reason = length_routing_excluded_ids(
            "fast", backends, length_routing=_length_routing(threshold_tokens=100), estimated_tokens=999,
        )
        self.assertEqual(ids, frozenset({"trailbrake-candidate"}))


class PromptKindClassificationTests(unittest.TestCase):
    """Dependency-free kind classifier signals used by kind-aware length routing.
    Every unit below targets one explicit signal bucket with non-overlapping
    heuristics; this keeps regression risk low when prompt heuristics evolve."""

    def test_multiturn_when_two_or_more_prior_assistant_turns(self):
        messages = (
            {"role": "assistant", "content": "ack"},
            {"role": "assistant", "content": "also ack"},
            {"role": "user", "content": "short follow-up"},
        )
        self.assertEqual(classify_prompt_kind(messages), "multiturn")

    def test_retrieval_for_long_context_plus_short_trailing_question(self):
        messages = (
            {"role": "user", "content": "x " * 900},
            {"role": "user", "content": "Based ONLY on the document excerpt above, what is the key finding?"},
        )
        self.assertEqual(classify_prompt_kind(messages), "retrieval")

    def test_generative_for_edit_or_file_path_signal(self):
        messages = (
            {
                "role": "user",
                "content": "Please implement a change in src/router/backends.py and update the routing path."
            },
        )
        self.assertEqual(classify_prompt_kind(messages), "generative")

    def test_generative_for_code_fence_signal(self):
        messages = (
            {"role": "user", "content": "Here is code to edit:\n```python\nprint('hi')\n```"},
        )
        self.assertEqual(classify_prompt_kind(messages), "generative")

    def test_unknown_when_no_signals_match(self):
        messages = (
            {"role": "user", "content": "Please provide an overview of the trade-offs of this design."},
            {"role": "assistant", "content": "Sure, here is an overview."},
        )
        self.assertEqual(classify_prompt_kind(messages), "unknown")


if __name__ == "__main__":
    unittest.main()
