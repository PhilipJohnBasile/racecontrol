from __future__ import annotations

import unittest
from pathlib import Path

from router.config import BackendConfig, LengthRoutingConfig, load_config, parse_config, with_server_overrides
from router.errors import ConfigError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_CONFIG = _REPO_ROOT / "config" / "router.example.toml"


def _minimal_raw(**overrides) -> dict:
    raw = {
        "escalation": {"default_tier": "fast", "escalation_tier": "deep"},
        "backends": [
            {"id": "trailbrake-baseline", "tier": "fast", "base_url": "http://127.0.0.1:8080",
             "model_id": "default", "rollback_target": True},
            {"id": "iliria", "tier": "deep", "base_url": "http://127.0.0.1:8000",
             "model_id": "glm-5.2-iliria", "rollback_target": True},
        ],
    }
    raw.update(overrides)
    return raw


class ExampleConfigTests(unittest.TestCase):
    """The shipped example config must always be loadable -- it is both the
    README's quick-start artifact and the doc's reference deployment."""

    def test_shipped_example_config_loads_and_validates(self):
        config = load_config(_EXAMPLE_CONFIG)
        self.assertEqual(set(config.tiers()), {"fast", "deep"})
        self.assertEqual(config.escalation.default_tier, "fast")
        self.assertEqual(config.escalation.escalation_tier, "deep")

    def test_shipped_example_candidate_starts_disabled_and_zero_weight(self):
        # The instant-rollback default posture: canary off until a human
        # deliberately turns it on after reviewing the pruning sweep.
        config = load_config(_EXAMPLE_CONFIG)
        candidate = next(b for b in config.backends if b.id == "trailbrake-candidate")
        self.assertFalse(candidate.enabled)
        self.assertEqual(candidate.weight, 0)

    def test_shipped_example_has_a_rollback_target_per_tier(self):
        config = load_config(_EXAMPLE_CONFIG)
        for tier in config.tiers():
            targets = [b for b in config.backends_for_tier(tier) if b.rollback_target and b.enabled]
            self.assertTrue(targets, f"tier {tier!r} has no enabled rollback_target")

    def test_shipped_example_ships_with_routing_headers_off(self):
        # BLIND-CANARY: the shipped default posture must be blind, not
        # opt-in-by-omission.
        config = load_config(_EXAMPLE_CONFIG)
        self.assertFalse(config.server.expose_routing_headers)

    def test_shipped_example_ships_with_length_routing_disabled(self):
        # Ship-dark: length-aware routing (docs/DESIGN.md's "Length-aware
        # routing" section) is a guard-rail an operator opts into once a
        # real threshold is chosen, not a default posture.
        config = load_config(_EXAMPLE_CONFIG)
        self.assertFalse(config.length_routing.enabled)
        self.assertEqual(config.length_routing.threshold_tokens, 4096)
        self.assertEqual(config.length_routing.estimator, "chars_div4")
        self.assertFalse(config.length_routing.kind_aware)
        self.assertEqual(config.length_routing.kind_thresholds, {
            "generative": 4096,
            "retrieval": 8192,
            "multiturn": 4096,
            "unknown": 4096,
        })


class ParseConfigDefaultsTests(unittest.TestCase):
    def test_minimal_config_gets_sane_defaults(self):
        config = parse_config(_minimal_raw())
        self.assertEqual(config.server.host, "127.0.0.1")
        self.assertEqual(config.server.port, 8100)
        self.assertIsNone(config.server.api_key)
        self.assertEqual(config.escalation.policy, "default")
        self.assertEqual(config.circuit_breaker.failure_threshold, 3)

    def test_expose_routing_headers_defaults_to_false(self):
        # BLIND-CANARY: a config that doesn't mention this knob at all must
        # still ship blind, not accidentally on.
        config = parse_config(_minimal_raw())
        self.assertFalse(config.server.expose_routing_headers)

    def test_expose_routing_headers_can_be_enabled_via_config(self):
        raw = _minimal_raw(server={"expose_routing_headers": True})
        config = parse_config(raw)
        self.assertTrue(config.server.expose_routing_headers)

    def test_backend_defaults(self):
        config = parse_config(_minimal_raw())
        backend = config.backends[0]
        self.assertEqual(backend.weight, 100)
        self.assertTrue(backend.enabled)
        self.assertEqual(backend.role, "primary")
        self.assertEqual(backend.connect_timeout_s, 5.0)

    def test_length_routing_gets_sane_defaults_when_section_is_absent(self):
        config = parse_config(_minimal_raw())
        self.assertFalse(config.length_routing.enabled)
        self.assertEqual(config.length_routing.threshold_tokens, 4096)
        self.assertEqual(config.length_routing.estimator, "chars_div4")
        self.assertFalse(config.length_routing.kind_aware)
        self.assertEqual(config.length_routing.kind_thresholds, {
            "generative": 4096,
            "retrieval": 8192,
            "multiturn": 4096,
            "unknown": 4096,
        })

    def test_length_routing_section_is_parsed_from_raw_config(self):
        raw = _minimal_raw(length_routing={"enabled": True, "threshold_tokens": 2048, "estimator": "chars_div4"})
        config = parse_config(raw)
        self.assertTrue(config.length_routing.enabled)
        self.assertEqual(config.length_routing.threshold_tokens, 2048)
        self.assertEqual(config.length_routing.estimator, "chars_div4")
        self.assertFalse(config.length_routing.kind_aware)
        self.assertEqual(config.length_routing.kind_thresholds["retrieval"], 8192)

    def test_length_routing_section_parses_kind_aware_thresholds(self):
        raw = _minimal_raw(
            length_routing={
                "enabled": True,
                "kind_aware": True,
                "threshold_tokens": 200,
                "kind_thresholds": {
                    "generative": 1_000,
                    "retrieval": 2_000,
                    "multiturn": 3_000,
                    "unknown": 4_000,
                },
                "estimator": "chars_div4",
            }
        )
        config = parse_config(raw)
        self.assertTrue(config.length_routing.enabled)
        self.assertTrue(config.length_routing.kind_aware)
        self.assertEqual(config.length_routing.threshold_tokens, 200)
        self.assertEqual(config.length_routing.kind_thresholds["multiturn"], 3_000)

    def test_tiers_preserve_first_seen_order(self):
        raw = _minimal_raw()
        raw["backends"] = [
            {"id": "z", "tier": "deep", "base_url": "http://x", "model_id": "m", "rollback_target": True},
            {"id": "a", "tier": "fast", "base_url": "http://x", "model_id": "m", "rollback_target": True},
        ]
        config = parse_config(raw)
        self.assertEqual(config.tiers(), ("deep", "fast"))


class ValidationTests(unittest.TestCase):
    def test_no_backends_is_rejected(self):
        with self.assertRaises(ConfigError):
            parse_config({"backends": []})

    def test_duplicate_backend_ids_are_rejected(self):
        raw = _minimal_raw()
        raw["backends"][1]["id"] = "trailbrake-baseline"
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_escalation_tier_must_exist_among_backends(self):
        raw = _minimal_raw(escalation={"default_tier": "fast", "escalation_tier": "nonexistent"})
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_default_tier_must_exist_among_backends(self):
        raw = _minimal_raw(escalation={"default_tier": "nonexistent", "escalation_tier": "deep"})
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_tier_without_enabled_rollback_target_is_rejected(self):
        raw = _minimal_raw()
        raw["backends"][0]["rollback_target"] = False
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_tier_with_disabled_rollback_target_only_is_rejected(self):
        # A rollback_target that is itself disabled does not count -- the
        # tier still has no *live* safety net.
        raw = _minimal_raw()
        raw["backends"][0]["enabled"] = False
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_unknown_escalation_policy_name_is_rejected(self):
        raw = _minimal_raw(escalation={"policy": "not-a-tier", "default_tier": "fast", "escalation_tier": "deep"})
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_policy_name_matching_a_tier_is_allowed(self):
        raw = _minimal_raw(escalation={"policy": "deep", "default_tier": "fast", "escalation_tier": "deep"})
        config = parse_config(raw)
        self.assertEqual(config.escalation.policy, "deep")

    def test_missing_required_backend_field_raises_config_error(self):
        raw = _minimal_raw()
        del raw["backends"][0]["model_id"]
        with self.assertRaises(ConfigError):
            parse_config(raw)


class BackendConfigValidationTests(unittest.TestCase):
    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ConfigError):
            BackendConfig(id="a", tier="fast", base_url="http://x", model_id="m", weight=-1)

    def test_base_url_must_have_a_scheme(self):
        with self.assertRaises(ConfigError):
            BackendConfig(id="a", tier="fast", base_url="127.0.0.1:8080", model_id="m")

    def test_non_positive_connect_timeout_is_rejected(self):
        with self.assertRaises(ConfigError):
            BackendConfig(id="a", tier="fast", base_url="http://x", model_id="m", connect_timeout_s=0)

    def test_non_positive_idle_timeout_is_rejected(self):
        with self.assertRaises(ConfigError):
            BackendConfig(id="a", tier="fast", base_url="http://x", model_id="m", idle_timeout_s=-5)

    def test_https_base_url_is_accepted(self):
        backend = BackendConfig(id="a", tier="fast", base_url="https://x", model_id="m")
        self.assertEqual(backend.base_url, "https://x")


class LengthRoutingConfigValidationTests(unittest.TestCase):
    """Direct-construction validation for `LengthRoutingConfig` --
    `__post_init__` raises immediately, mirroring `BackendConfigValidationTests`'
    style above, so a malformed value can never be built at all, whether it
    came from a TOML file (see LengthRoutingParsingRejectionTests below) or
    from code constructing one directly."""

    def test_defaults_are_valid(self):
        length_routing = LengthRoutingConfig()
        self.assertFalse(length_routing.enabled)
        self.assertEqual(length_routing.threshold_tokens, 4096)
        self.assertEqual(length_routing.estimator, "chars_div4")
        self.assertFalse(length_routing.kind_aware)
        self.assertEqual(length_routing.kind_thresholds["retrieval"], 8192)

    def test_zero_threshold_is_rejected(self):
        with self.assertRaises(ConfigError):
            LengthRoutingConfig(threshold_tokens=0)

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(ConfigError):
            LengthRoutingConfig(threshold_tokens=-4096)

    def test_positive_threshold_is_accepted(self):
        length_routing = LengthRoutingConfig(threshold_tokens=1)
        self.assertEqual(length_routing.threshold_tokens, 1)

    def test_unknown_estimator_is_rejected(self):
        with self.assertRaises(ConfigError):
            LengthRoutingConfig(estimator="a-tokenizer-that-does-not-exist")

    def test_invalid_kind_threshold_rejected(self):
        with self.assertRaises(ConfigError):
            LengthRoutingConfig(kind_thresholds={"generative": 0, "retrieval": 1, "multiturn": 1, "unknown": 1})

    def test_unknown_kind_threshold_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            LengthRoutingConfig(kind_thresholds={"coding": 123, "retrieval": 100, "multiturn": 100, "unknown": 100})

    def test_known_estimator_is_accepted(self):
        length_routing = LengthRoutingConfig(estimator="chars_div4")
        self.assertEqual(length_routing.estimator, "chars_div4")


class LengthRoutingParsingRejectionTests(unittest.TestCase):
    """The same two `LengthRoutingConfig` checks, reached through
    `parse_config` (i.e. as if loaded from a TOML file) rather than by
    constructing the dataclass directly."""

    def test_non_positive_threshold_in_raw_config_is_rejected(self):
        raw = _minimal_raw(length_routing={"threshold_tokens": -1})
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_unknown_estimator_in_raw_config_is_rejected(self):
        raw = _minimal_raw(length_routing={"estimator": "not-a-real-estimator"})
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_unknown_kind_threshold_key_in_raw_config_is_rejected(self):
        raw = _minimal_raw(length_routing={"kind_thresholds": {"foo": 123}})
        with self.assertRaises(ConfigError):
            parse_config(raw)


class WithServerOverridesTests(unittest.TestCase):
    def test_no_overrides_returns_the_same_config_values(self):
        config = parse_config(_minimal_raw())
        overridden = with_server_overrides(config, host=None, port=None)
        self.assertEqual(overridden.server.host, config.server.host)
        self.assertEqual(overridden.server.port, config.server.port)

    def test_host_override_only(self):
        config = parse_config(_minimal_raw())
        overridden = with_server_overrides(config, host="0.0.0.0", port=None)
        self.assertEqual(overridden.server.host, "0.0.0.0")
        self.assertEqual(overridden.server.port, config.server.port)

    def test_port_override_only(self):
        config = parse_config(_minimal_raw())
        overridden = with_server_overrides(config, host=None, port=9999)
        self.assertEqual(overridden.server.port, 9999)
        self.assertEqual(overridden.server.host, config.server.host)

    def test_original_config_is_not_mutated(self):
        config = parse_config(_minimal_raw())
        with_server_overrides(config, host="0.0.0.0", port=9999)
        self.assertEqual(config.server.host, "127.0.0.1")
        self.assertEqual(config.server.port, 8100)


if __name__ == "__main__":
    unittest.main()
