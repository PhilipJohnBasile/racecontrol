from __future__ import annotations

import unittest

from router.config import BackendConfig, CircuitBreakerConfig, EscalationConfig, RouterConfig, ServerConfig
from router.policy import (
    AlwaysTierPolicy,
    DefaultPolicy,
    DraftThenEscalatePolicy,
    build_policy,
    find_hard_marker,
    hardness_score,
    resolve_manual_override,
)


def _config(**escalation_overrides) -> RouterConfig:
    backends = (
        BackendConfig(
            id="trailbrake-baseline", tier="fast", base_url="http://127.0.0.1:8080",
            model_id="default", role="baseline", rollback_target=True,
        ),
        BackendConfig(
            id="trailbrake-candidate", tier="fast", base_url="http://127.0.0.1:8081",
            model_id="default", weight=0, enabled=False, role="candidate",
        ),
        BackendConfig(
            id="iliria", tier="deep", base_url="http://127.0.0.1:8000",
            model_id="glm-5.2-iliria", role="primary", rollback_target=True,
        ),
    )
    escalation = EscalationConfig(**escalation_overrides)
    return RouterConfig(
        server=ServerConfig(),
        escalation=escalation,
        circuit_breaker=CircuitBreakerConfig(),
        backends=backends,
        fallback={"fast": "deep", "deep": "fast"},
    )


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


class HardnessScoreTests(unittest.TestCase):
    def test_ordinary_request_scores_low(self):
        score = hardness_score(_messages("Add a docstring to this function."))
        self.assertLess(score, 0.3)

    def test_boilerplate_markers_pull_score_down(self):
        score = hardness_score(_messages("Please rename this variable to `count`."))
        self.assertEqual(score, 0.0)  # clamped at the floor, not negative

    def test_debugging_hard_signal_scores_high(self):
        score = hardness_score(
            _messages("Why does this fail intermittently -- looks like a race condition?")
        )
        self.assertGreaterEqual(score, 0.6)

    def test_proof_signal_detected(self):
        score = hardness_score(_messages("Prove that this greedy algorithm is optimal."))
        self.assertGreater(score, 0.0)

    def test_no_user_message_scores_zero(self):
        self.assertEqual(hardness_score([{"role": "system", "content": "be nice"}]), 0.0)

    def test_content_with_no_text_at_all_is_ignored_not_crashed_on(self):
        # Image-only multimodal content (no "text" part) must not raise, and
        # correctly contributes no signal.
        score = hardness_score([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
        self.assertEqual(score, 0.0)

    def test_hard_signal_inside_content_parts_is_detected(self):
        # Regression for the audit's "content-parts bypass": content shaped
        # as a multimodal parts list (trivial for any real client to send,
        # not just an image-attaching one) used to make hardness_score see
        # an empty string no matter what the text part said, silently
        # defeating escalation for a hard task wrapped in that shape.
        score = hardness_score(
            [{"role": "user", "content": [
                {"type": "text", "text": "Why does this crash intermittently? Race condition."},
            ]}]
        )
        self.assertGreaterEqual(score, 0.6)

    def test_multiple_text_parts_are_joined(self):
        score = hardness_score(
            [{"role": "user", "content": [
                {"type": "text", "text": "Why does this crash intermittently?"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "Race condition, I think."},
            ]}]
        )
        self.assertGreaterEqual(score, 0.6)

    def test_strong_signal_floor_survives_easy_task_penalty(self):
        # Regression for the audit's "heuristic is overly additive": a
        # high-precision signal like "deadlock" (weight 0.4, below the 0.6
        # default threshold on its own) must establish a floor that easy-task
        # wording cannot cancel back under the threshold.
        bare = hardness_score(_messages("There's a deadlock somewhere in here."))
        self.assertGreaterEqual(bare, 0.6)
        penalized = hardness_score(_messages("Write a unit test for this deadlock."))
        self.assertGreaterEqual(penalized, 0.6)  # previously knocked down to 0.2

    def test_fuzzy_hard_signal_has_no_floor(self):
        # Only the two highest-precision patterns get a floor; a fuzzier one
        # (here, "trade-offs") stays purely additive and the easy-task
        # penalty can still pull it under the default threshold.
        score = hardness_score(_messages("Rename this and note the trade-offs."))
        self.assertLess(score, 0.6)

    def test_earlier_hard_turn_still_influences_score_via_decay(self):
        # Regression for "scoring only the last user message": a hard
        # opening turn followed by a terse in-context follow-up must not
        # silently drop straight back to zero.
        terse_only = hardness_score(_messages("ok thanks"))
        combined = hardness_score(
            [
                {"role": "user", "content": "Why does this crash intermittently? Smells like a race condition."},
                {"role": "assistant", "content": "Let me look into the logs."},
                {"role": "user", "content": "ok thanks"},
            ]
        )
        self.assertEqual(terse_only, 0.0)
        self.assertGreater(combined, terse_only)
        self.assertGreaterEqual(combined, 0.6)  # still crosses the default escalation threshold

    def test_turn_too_far_back_falls_outside_the_recent_window(self):
        # Only `_RECENT_TURNS` (3) user turns count; a hard signal 3 user
        # turns before the current one must not still be pulling the score
        # up indefinitely.
        messages = [
            {"role": "user", "content": "Why does this crash intermittently? Smells like a race condition."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "ok"},
        ]
        self.assertEqual(hardness_score(messages), 0.0)


class FindHardMarkerTests(unittest.TestCase):
    def test_finds_configured_marker(self):
        marker = find_hard_marker(_messages("#deep please think about this"), ("#deep", "/escalate"))
        self.assertEqual(marker, "#deep")

    def test_returns_none_when_absent(self):
        self.assertIsNone(find_hard_marker(_messages("just fix the typo"), ("#deep", "/escalate")))

    def test_does_not_match_marker_as_substring_of_a_longer_token(self):
        # Regression for the audit's "#deepfake matches `#deep`" substring bug.
        self.assertIsNone(find_hard_marker(_messages("posting some #deepfake content"), ("#deep", "/escalate")))

    def test_still_matches_at_start_of_message(self):
        # Token-boundary matching uses `(?<!\w)`/`(?!\w)` rather than `\b`
        # specifically so a marker beginning with a non-word char ("#")
        # still matches at position 0 of the message.
        self.assertEqual(find_hard_marker(_messages("#deep why is this slow"), ("#deep",)), "#deep")

    def test_still_matches_with_trailing_punctuation(self):
        self.assertEqual(find_hard_marker(_messages("please #deep, this is urgent"), ("#deep",)), "#deep")

    def test_finds_marker_inside_content_parts(self):
        # Regression for the audit's "content-parts bypass," marker side:
        # a multimodal-shaped message carrying the marker in a text part
        # must still be found, not silently treated as markerless.
        messages = [{"role": "user", "content": [{"type": "text", "text": "#deep please look at this"}]}]
        self.assertEqual(find_hard_marker(messages, ("#deep", "/escalate")), "#deep")


class ResolveManualOverrideTests(unittest.TestCase):
    def test_tier_name_overrides(self):
        config = _config()
        decision = resolve_manual_override({"model": "deep"}, config)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "manual_override")

    def test_backend_id_overrides(self):
        config = _config()
        decision = resolve_manual_override({"model": "trailbrake-baseline"}, config)
        self.assertEqual(decision.tier, "fast")

    def test_backend_id_override_pins_forced_backend_id(self):
        # Regression for the audit's "backend override isn't a real
        # override": naming a specific backend id must carry that exact
        # backend through the decision, not just its tier.
        config = _config()
        decision = resolve_manual_override({"model": "trailbrake-baseline"}, config)
        self.assertEqual(decision.forced_backend_id, "trailbrake-baseline")

    def test_tier_name_override_does_not_force_a_backend(self):
        # Naming a bare tier ("deep") is not a backend pin -- weighted
        # in-tier selection should still run.
        config = _config()
        decision = resolve_manual_override({"model": "deep"}, config)
        self.assertIsNone(decision.forced_backend_id)

    def test_backend_model_id_overrides(self):
        config = _config()
        decision = resolve_manual_override({"model": "glm-5.2-iliria"}, config)
        self.assertEqual(decision.tier, "deep")

    def test_unknown_model_returns_none(self):
        config = _config()
        self.assertIsNone(resolve_manual_override({"model": "gpt-4"}, config))

    def test_missing_model_returns_none(self):
        self.assertIsNone(resolve_manual_override({}, _config()))


class DefaultPolicyTests(unittest.TestCase):
    def test_manual_override_wins_over_everything(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide(
            {"model": "deep", "messages": _messages("rename this variable")}, config
        )
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "manual_override")

    def test_reasoning_effort_high_escalates(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide(
            {"reasoning_effort": "high", "messages": _messages("add a getter")}, config
        )
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "explicit_marker")

    def test_reasoning_effort_low_does_not_escalate(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide(
            {"reasoning_effort": "low", "messages": _messages("add a getter")}, config
        )
        self.assertEqual(decision.tier, "fast")

    def test_explicit_marker_escalates(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide({"messages": _messages("#deep why is this slow")}, config)
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "explicit_marker")
        self.assertIn("#deep", decision.reason)

    def test_task_heuristic_escalates_above_threshold(self):
        config = _config(heuristic_threshold=0.3)
        policy = DefaultPolicy()
        decision = policy.decide(
            {"messages": _messages("Why does this crash intermittently? Smells like a race condition.")},
            config,
        )
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "task_heuristic")

    def test_task_heuristic_disabled_never_escalates_on_score_alone(self):
        config = _config(heuristic_threshold=0.1, enable_task_heuristic=False)
        policy = DefaultPolicy()
        decision = policy.decide(
            {"messages": _messages("Why does this crash intermittently? Smells like a race condition.")},
            config,
        )
        self.assertEqual(decision.tier, "fast")
        self.assertEqual(decision.trigger, "default")

    def test_default_tier_when_nothing_matches(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide({"messages": _messages("add a getter for name")}, config)
        self.assertEqual(decision.tier, "fast")
        self.assertEqual(decision.trigger, "default")

    def test_no_messages_key_does_not_crash(self):
        config = _config()
        policy = DefaultPolicy()
        decision = policy.decide({}, config)
        self.assertEqual(decision.tier, "fast")

    def test_content_parts_shaped_request_with_marker_still_escalates(self):
        # End-to-end regression for the audit's "content-parts bypass":
        # this exact request shape previously fell through to
        # trigger="default" (tier="fast") because find_hard_marker and
        # hardness_score both saw an empty string for non-string `content`,
        # silently defeating both escalation triggers at once.
        config = _config()
        policy = DefaultPolicy()
        messages = [{"role": "user", "content": [{"type": "text", "text": "#deep why is this racing"}]}]
        decision = policy.decide({"messages": messages}, config)
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "explicit_marker")

    def test_hard_opening_turn_keeps_terse_followup_escalated(self):
        # Regression for the audit's exact scenario: "scoring only the last
        # user message will route a hard initial task to deep and a terse
        # follow-up back to fast." Pins threshold=0.6, the value the decay
        # arithmetic was calibrated against (0.8 raw * 0.8 decay = 0.64):
        # this tests the cross-turn decay mechanism, not the shipped default
        # (0.7 -- measured rationale in bench/escalation_eval/).
        config = _config(heuristic_threshold=0.6)
        policy = DefaultPolicy()
        messages = [
            {"role": "user", "content": "Why does this crash intermittently? Smells like a race condition."},
            {"role": "assistant", "content": "Let me look into the logs."},
            {"role": "user", "content": "ok thanks"},
        ]
        decision = policy.decide({"messages": messages}, config)
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "task_heuristic")

    def test_bare_floor_signal_stays_fast_at_the_shipped_default(self):
        # The shipped default threshold is 0.7, which a lone floored pattern
        # (0.600) deliberately does not clear: on the 84-case blind held-out
        # set (bench/escalation_eval/) every 0.6-threshold escalation was a
        # false positive and no hard prompt was caught, so a single scary
        # keyword must not buy a ~100x-cost deep-tier turn on its own.
        config = _config()  # shipped defaults, threshold 0.7
        policy = DefaultPolicy()
        decision = policy.decide(
            {"messages": _messages("There is a deadlock in the connection pool.")}, config
        )
        self.assertEqual(decision.tier, "fast")
        self.assertEqual(decision.trigger, "default")


class AlwaysTierPolicyTests(unittest.TestCase):
    def test_pins_to_configured_tier(self):
        policy = AlwaysTierPolicy("deep")
        decision = policy.decide({"messages": _messages("add a getter")}, _config())
        self.assertEqual(decision.tier, "deep")
        self.assertEqual(decision.trigger, "fixed_policy")

    def test_manual_override_still_wins(self):
        policy = AlwaysTierPolicy("deep")
        decision = policy.decide({"model": "fast"}, _config())
        self.assertEqual(decision.tier, "fast")
        self.assertEqual(decision.trigger, "manual_override")


class DraftThenEscalatePolicyTests(unittest.TestCase):
    def test_accepts_draft_with_no_verifier_configured(self):
        policy = DraftThenEscalatePolicy(DefaultPolicy(), verifier=None)
        self.assertTrue(policy.accepts_draft({}, "anything"))

    def test_verifier_rejection_is_honored(self):
        policy = DraftThenEscalatePolicy(DefaultPolicy(), verifier=lambda request, text: "TODO" not in text)
        self.assertFalse(policy.accepts_draft({}, "def f(): ...  # TODO"))
        self.assertTrue(policy.accepts_draft({}, "def f(): return 1"))

    def test_decide_delegates_to_inner_policy(self):
        policy = DraftThenEscalatePolicy(DefaultPolicy(), verifier=None)
        decision = policy.decide({"messages": _messages("#deep")}, _config())
        self.assertEqual(decision.tier, "deep")

    def test_wants_draft_first_flag_is_set(self):
        self.assertTrue(DraftThenEscalatePolicy(DefaultPolicy()).wants_draft_first)
        self.assertFalse(hasattr(DefaultPolicy(), "wants_draft_first"))


class BuildPolicyTests(unittest.TestCase):
    def test_default_builds_default_policy(self):
        policy = build_policy(_config(policy="default"))
        self.assertIsInstance(policy, DefaultPolicy)

    def test_tier_name_builds_always_tier_policy(self):
        policy = build_policy(_config(policy="deep"))
        self.assertIsInstance(policy, AlwaysTierPolicy)
        self.assertEqual(policy.tier, "deep")

    def test_unknown_policy_name_raises(self):
        with self.assertRaises(ValueError):
            build_policy(_config(policy="nonsense"))

    def test_draft_then_escalate_wraps_base_policy_when_enabled(self):
        policy = build_policy(_config(policy="default", enable_draft_then_escalate=True))
        self.assertIsInstance(policy, DraftThenEscalatePolicy)

    def test_draft_then_escalate_disabled_by_default(self):
        policy = build_policy(_config())
        self.assertNotIsInstance(policy, DraftThenEscalatePolicy)


if __name__ == "__main__":
    unittest.main()
