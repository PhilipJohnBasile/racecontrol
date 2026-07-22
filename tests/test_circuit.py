from __future__ import annotations

import unittest

from router.circuit import CircuitBreaker, CircuitBreakerRegistry, CircuitState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CircuitBreakerTests(unittest.TestCase):
    def test_starts_closed_and_allows_requests(self):
        breaker = CircuitBreaker()
        self.assertEqual(breaker.snapshot().state, CircuitState.CLOSED)
        self.assertTrue(breaker.allow_request())

    def test_stays_closed_below_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.snapshot().state, CircuitState.CLOSED)
        self.assertTrue(breaker.allow_request())

    def test_opens_at_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.snapshot().state, CircuitState.OPEN)
        self.assertFalse(breaker.allow_request())

    def test_success_resets_consecutive_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        self.assertEqual(breaker.snapshot().consecutive_failures, 0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.snapshot().state, CircuitState.CLOSED)  # only 2 since the reset

    def test_stays_open_before_reset_window_elapses(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(59.0)
        self.assertFalse(breaker.allow_request())

    def test_transitions_to_half_open_after_reset_window(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(60.0)
        self.assertTrue(breaker.allow_request())
        self.assertEqual(breaker.snapshot().state, CircuitState.HALF_OPEN)

    def test_half_open_success_closes_the_circuit(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(60.0)
        self.assertTrue(breaker.allow_request())
        breaker.record_success()
        self.assertEqual(breaker.snapshot().state, CircuitState.CLOSED)
        self.assertTrue(breaker.allow_request())

    def test_half_open_failure_reopens_for_another_full_window(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(60.0)
        self.assertTrue(breaker.allow_request())
        breaker.record_failure()
        self.assertEqual(breaker.snapshot().state, CircuitState.OPEN)
        clock.advance(59.0)
        self.assertFalse(breaker.allow_request())
        clock.advance(1.0)
        self.assertTrue(breaker.allow_request())

    def test_half_open_allows_only_one_trial_at_a_time(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(60.0)
        self.assertTrue(breaker.allow_request())   # claims the one trial slot
        self.assertFalse(breaker.allow_request())  # a second concurrent caller is refused

    def test_is_available_does_not_consume_the_half_open_trial_slot(self):
        """The bug this guards against: if merely *checking* whether a
        half-open backend could be tried also claimed its one trial slot,
        a caller that peeks at several tier candidates and then dispatches
        to a *different* one would permanently wedge this backend (nobody
        ever calls record_success/record_failure for a request that was
        never sent, so `_half_open_trial_in_flight` would stay True
        forever). `is_available()` must be side-effect-free."""
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        clock.advance(60.0)

        for _ in range(5):
            self.assertTrue(breaker.is_available())

        self.assertTrue(breaker.allow_request())
        breaker.record_success()
        self.assertEqual(breaker.snapshot().state, CircuitState.CLOSED)

    def test_is_available_reflects_open_state_before_reset_window(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0, clock=clock)
        breaker.record_failure()
        self.assertFalse(breaker.is_available())
        clock.advance(60.0)
        self.assertTrue(breaker.is_available())

    def test_rejects_non_positive_failure_threshold(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_rejects_non_positive_reset_window(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(reset_after_s=0)


class CircuitBreakerRegistryTests(unittest.TestCase):
    def test_unknown_backend_is_not_excluded(self):
        registry = CircuitBreakerRegistry(failure_threshold=3, reset_after_s=60.0)
        self.assertEqual(registry.excluded_backend_ids(["never-seen"]), frozenset())

    def test_failing_backend_becomes_excluded(self):
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_after_s=60.0)
        registry.get("flaky").record_failure()
        registry.get("flaky").record_failure()
        self.assertEqual(registry.excluded_backend_ids(["flaky", "healthy"]), frozenset({"flaky"}))

    def test_get_returns_the_same_breaker_instance_across_calls(self):
        registry = CircuitBreakerRegistry(failure_threshold=3, reset_after_s=60.0)
        self.assertIs(registry.get("a"), registry.get("a"))

    def test_excluded_ids_recovers_after_reset_window(self):
        clock = _FakeClock()
        registry = CircuitBreakerRegistry(failure_threshold=1, reset_after_s=10.0, clock=clock)
        registry.get("flaky").record_failure()
        self.assertEqual(registry.excluded_backend_ids(["flaky"]), frozenset({"flaky"}))
        clock.advance(10.0)
        self.assertEqual(registry.excluded_backend_ids(["flaky"]), frozenset())


if __name__ == "__main__":
    unittest.main()
