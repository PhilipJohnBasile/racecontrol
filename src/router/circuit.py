"""Per-backend circuit breaker.

After `failure_threshold` consecutive failures, a backend is skipped (treated
as unhealthy) for `reset_after_s`, then given exactly one half-open trial
request; a success there closes the circuit again, a failure re-opens it for
another full `reset_after_s` window.

This is what makes canary rollback able to happen automatically, not just by
hand: a candidate backend that starts erroring gets excluded from selection
within `failure_threshold` requests, with no operator action. Manually
zeroing a backend's config `weight`/`enabled` (docs/DESIGN.md's other
rollback path) is the deliberate, reviewed rollback a human decides to make;
this is the unattended, immediate one that covers the gap while a human
notices.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    state: CircuitState
    consecutive_failures: int
    opened_at: float | None


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_after_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_after_s <= 0:
            raise ValueError("reset_after_s must be > 0")
        self._failure_threshold = failure_threshold
        self._reset_after_s = reset_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_trial_in_flight = False

    def is_available(self) -> bool:
        """Non-mutating peek: would `allow_request()` currently succeed?
        Used to filter *candidates* during backend selection, when several
        backends in a tier are being considered and only one will actually
        be called -- see `CircuitBreakerRegistry.excluded_backend_ids`.
        Deliberately does not consume the one half-open trial slot; only
        `allow_request()` (called once dispatch.py has committed to this
        specific backend) does that, so that merely *considering* a
        half-open backend among several candidates can never wedge it by
        claiming its one trial slot without actually using it."""
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                assert self._opened_at is not None
                return self._clock() - self._opened_at >= self._reset_after_s
            return not self._half_open_trial_in_flight

    def allow_request(self) -> bool:
        """True if a request may be attempted right now, against *this*
        specific backend, right before it is actually called. Calling this
        can itself transition OPEN -> HALF_OPEN once `reset_after_s` has
        elapsed and claims the one half-open trial slot -- matching standard
        circuit-breaker semantics. Callers that are only filtering candidates
        rather than committing to a call should use `is_available()` instead
        (see its docstring for why the distinction matters)."""
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                assert self._opened_at is not None
                if self._clock() - self._opened_at >= self._reset_after_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_trial_in_flight = True
                    return True
                return False
            # HALF_OPEN: exactly one trial request in flight at a time.
            if not self._half_open_trial_in_flight:
                self._half_open_trial_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._half_open_trial_in_flight = False
            if self._state is CircuitState.HALF_OPEN or self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            return CircuitSnapshot(self._state, self._consecutive_failures, self._opened_at)


class CircuitBreakerRegistry:
    """One `CircuitBreaker` per backend id, created lazily (a backend that
    has never failed has no breaker yet and is therefore never excluded)."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        reset_after_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_after_s = reset_after_s
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, backend_id: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(backend_id)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=self._failure_threshold,
                    reset_after_s=self._reset_after_s,
                    clock=self._clock,
                )
                self._breakers[backend_id] = breaker
            return breaker

    def excluded_backend_ids(self, backend_ids: list[str]) -> frozenset[str]:
        """Backend ids among `backend_ids` whose breaker currently disallows
        a request (a non-mutating peek -- see `CircuitBreaker.is_available`).
        Passed as `select_backend`'s `exclude_ids` so an open-circuit backend
        is skipped without touching its config."""
        return frozenset(
            backend_id for backend_id in backend_ids if not self.get(backend_id).is_available()
        )
