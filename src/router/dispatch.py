"""Ties policy + backend selection + circuit breaker + fallback + telemetry
into one per-request decide-and-call pipeline, independent of HTTP transport
mechanics. `server.py` adapts this to sockets; tests exercise this module
directly against fake `BackendClient`/`Transport` doubles with no sockets at
all -- see tests/test_dispatch.py.

Both `dispatch_chat` (buffered) and `dispatch_chat_stream` (incremental) share
one retry loop (`_run`) so the routing/fallback/circuit-breaker/telemetry
semantics can never drift between the two transport modes -- only how the
backend's body is read differs.

Bounded retry: `MAX_ATTEMPTS` caps the number of backend calls made for one
client request, regardless of how many backends/tiers are configured or how
`fallback` is wired -- a config mistake (or even a fallback cycle) can
therefore never turn one client request into an unbounded retry storm
against a slow, expensive backend. Within that bound, `_run` will not hop
back to the tier the request started in even if `fallback` is written as a
cycle (`fallback_tier in (decision.tier, tier)` below) -- see
docs/DESIGN.md's "at most one cross-tier hop" note.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .backends import (
    BackendClient,
    BackendResponse,
    OpenResponse,
    drain,
    classify_prompt_kind,
    estimate_prompt_tokens,
    length_routing_excluded_ids,
    select_backend,
)
from .circuit import CircuitBreakerRegistry
from .config import BackendConfig, RouterConfig
from .errors import BackendRequestFailed, NoBackendAvailable
from .policy import EscalationPolicy, RoutingDecision, escalation_features
from .telemetry import DecisionLogger, DecisionRecord, new_request_id, utc_now_iso


class StreamOutcome(Enum):
    """How a *streamed* response ended, reported by the transport layer once it
    has finished relaying the body -- the input to a stream's deferred breaker
    accounting (see RequestRouter._make_stream_finalizer)."""

    SUCCESS = "success"  # clean upstream completion
    BACKEND_FAILURE = "backend_failure"  # upstream read/protocol failure or idle timeout mid-body
    CLIENT_ABORT = "client_abort"  # our write to the client failed first (client hung up)


@dataclass
class DispatchResult:
    response: Any  # BackendResponse (buffered) or OpenResponse (stream)
    backend_id: str
    tier: str
    decision: RoutingDecision
    canary: bool
    fallback_from: str | None
    request_id: str
    # Set only for streamed results: a single-shot callback the transport layer
    # invokes once with the observed StreamOutcome and, optionally, telemetry
    # it parsed from the relayed SSE body (`_extract_stream_usage_telemetry`
    # in server.py) -- the second argument defaults to None so an existing
    # `finalize_stream(outcome)` call site (nothing to report, or a transport
    # that never parses bodies) still works unchanged. None for buffered
    # results, which are already finalized (breaker + log) at return time.
    finalize_stream: Callable[[StreamOutcome, "dict[str, Any] | None"], None] | None = None


def _extract_message_text(body: bytes) -> str:
    try:
        payload = json.loads(body)
        return payload["choices"][0]["message"]["content"] or ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


# Optional, generic per-response telemetry headers folded into the decision
# log's DecisionRecord.extra when a backend chooses to send them -- see
# trailbrake/src/mlx_engine/server.py's _telemetry_response_headers
# (decode tok/s, TTFT, and the opt-in speculative-decoding drafter's
# acceptance rate, for the "trailbrake-with-drafter" canary backend). This module
# still knows neither backend's name (see module docstring): a backend that
# never sends these headers -- iliria, or trailbrake without the flag set --
# simply contributes nothing here, exactly as before this existed.
#
# Headers-only by construction (see _extract_backend_telemetry's own
# docstring): trailbrake never sends these as headers on a streamed response, so
# for `mode == "stream"` this function alone always yields `{}`. The SAME
# field names are also the keys trailbrake's `_stream_completion` puts in the
# closing SSE `usage` event; the streamed equivalent of this extraction
# (`_extract_stream_usage_telemetry` in server.py, fed by `_relay_stream`'s
# incremental SSE-frame scan) produces a dict shaped identically, and
# `_make_stream_finalizer` below merges the two so a streamed
# DecisionRecord.extra ends up carrying whichever source actually had data.
_TELEMETRY_HEADER_FIELDS = {
    "x-trailbrake-decode-tokens-per-second": "decode_tokens_per_second",
    "x-trailbrake-ttft-seconds": "time_to_first_token_seconds",
    "x-trailbrake-draft-acceptance-rate": "draft_acceptance_rate",
}


def _extract_backend_telemetry(headers: dict[str, str]) -> dict[str, Any]:
    """Parses `_TELEMETRY_HEADER_FIELDS` out of one response's (already
    lowercased -- see backends.py's `_HttpOpenResponse`) headers into a flat
    dict suitable for `DecisionRecord.extra`. A header that is absent is
    just omitted; one that is present but not parseable as a float is
    skipped rather than raising -- a malformed *optional* telemetry header
    must never break routing or logging. Buffered and streamed responses
    both expose `.headers` by the time a call has succeeded (see
    `_HttpOpenResponse`/`BackendResponse`), so this applies uniformly to
    both dispatch modes -- see `_run`'s "ok" branch and
    `_make_stream_finalizer`.

    Deliberately headers-only, not body-parsing -- and, for a streamed
    response, ALWAYS `{}`: trailbrake's `_telemetry_response_headers` docstring
    explains why a streamed response cannot carry these as headers at all
    (they are only known once generation finishes, long after headers must
    already be on the wire); they land in the final SSE `usage` event
    instead. Read on its own, this function would therefore silently give
    every streamed request weaker telemetry than a buffered one -- that WAS
    this router's actual behavior for streamed traffic (the normal
    interactive mode) before `_make_stream_finalizer` started also
    accepting body-derived telemetry parsed by the transport layer
    (`server.py`'s `_relay_stream` / `_extract_stream_usage_telemetry`) and
    merging it in at finalize time (see `_with_stream_usage_requested`
    just below for the other half: without it, trailbrake never even computes
    the closing usage event trailbrake's own docstring describes, streamed or
    not). This function is kept as-is (not folded away) because it remains
    exactly correct and sufficient for the buffered path, and a defensive
    header-based capture is harmless to keep for streams too, in case a
    future backend ever sends early telemetry headers.
    """
    telemetry: dict[str, Any] = {}
    for header_name, field_name in _TELEMETRY_HEADER_FIELDS.items():
        raw = headers.get(header_name)
        if raw is None:
            continue
        try:
            telemetry[field_name] = float(raw)
        except ValueError:
            continue
    return telemetry


def _with_stream_usage_requested(request_body: dict, mode: str) -> dict:
    """Router-owned guarantee, the streaming-telemetry sibling of
    `_with_no_think_default` just below: for `mode == "stream"`, force
    `stream_options.include_usage = True` onto the OUTGOING (upstream, this
    backend call only) body, regardless of whatever the router's own client
    did or didn't set. `stream_options.include_usage` is what makes trailbrake's
    `_stream_completion` compute and send the closing SSE `usage` event at
    all (see that function's docstring) -- left to the router's actual
    client to opt into, streamed decision-log telemetry was permanently
    empty for "the normal interactive mode" (a plain `stream: true` request,
    which essentially never sets this fairly obscure field), independent of
    anything `_extract_backend_telemetry`/`_make_stream_finalizer` could
    parse. This router needs that telemetry for its OWN decision log on
    every streamed request -- a need entirely separate from whatever the
    router's client wants relayed back to it -- so this only touches the
    body sent to the BACKEND; whether the resulting usage event is itself
    relayed back to the original client is unchanged (still whatever trailbrake
    sends, exactly as `_relay_stream` has always forwarded bytes,
    unfiltered -- see that function's docstring for why unconditionally
    relaying this one additional, spec-shaped, empty-`choices` event was
    judged the right, minimal-risk trade-off over rewriting the relay to
    filter out frames the client itself didn't ask for).

    Any `stream_options` keys the client already set are preserved
    untouched (only `include_usage` is forced, even overriding an explicit
    client-supplied `False` -- unlike `_with_no_think_default`, this is not
    a default filling a gap, it is the router's own unconditional need); a
    non-stream request is never touched."""
    if mode != "stream":
        return request_body
    stream_options = request_body.get("stream_options")
    merged = {**stream_options, "include_usage": True} if isinstance(stream_options, dict) else {"include_usage": True}
    return {**request_body, "stream_options": merged}


def _with_no_think_default(request_body: dict, tier: str, default_tier: str) -> dict:
    """Router-owned guarantee behind "no-think is the default_tier default"
    (docs/DESIGN.md's `enable_thinking` note): rather than resting on an
    assumption about trailbrake's own field-absent behavior -- a fact about a
    different repo this router cannot see or pin -- force
    `enable_thinking: False` onto the outgoing body whenever this attempt is
    against `default_tier` and the client did not already set the field
    itself. An explicit client-supplied value (True or False) always wins
    and is left untouched here; a request that has escalated (or fallen
    back) to a different tier is never touched by this function at all, so
    iliria's own native `enable_thinking`/`reasoning_effort` fields are
    always forwarded exactly as the client sent them."""
    if tier != default_tier or "enable_thinking" in request_body:
        return request_body
    return {**request_body, "enable_thinking": False}


class RequestRouter:
    """One instance wired up per running server; holds no per-request state
    of its own (all mutable state -- circuit breakers, telemetry counters --
    lives in the injected collaborators, so this class is cheap to
    construct in tests)."""

    MAX_ATTEMPTS = 6

    def __init__(
        self,
        config: RouterConfig,
        policy: EscalationPolicy,
        clients: dict[str, BackendClient],
        circuit_breakers: CircuitBreakerRegistry,
        telemetry: DecisionLogger,
    ) -> None:
        self.config = config
        self.policy = policy
        self.clients = clients
        self.circuit_breakers = circuit_breakers
        self.telemetry = telemetry

    def _candidate_ids_for_tier(self, tier: str) -> list[str]:
        return [backend.id for backend in self.config.backends_for_tier(tier)]

    def _select_backend(
        self,
        tier: str,
        decision: RoutingDecision,
        *,
        excluded: frozenset[str],
        sticky_key: str | None,
        estimated_tokens: int | None,
        prompt_kind: str | None,
    ) -> "tuple[BackendConfig | None, dict[str, Any]]":
        """Normally just `backends.select_backend`'s weighted draw. But a
        manual override that named a specific backend id/model_id
        (`decision.forced_backend_id`) must pin that exact backend rather
        than merely narrowing the field to its tier -- weighted selection
        over the rest of the tier would otherwise silently defeat the
        override (asking for `trailbrake-baseline` could still return
        `trailbrake-candidate`; see docs/DESIGN.md / the audit's "Escalation
        policy" section). The pin only applies to the tier the override
        actually resolved to (`tier == decision.tier`): once that exact
        backend is excluded -- already tried, or its breaker is open -- this
        returns None just like "no candidates in this tier," which sends the
        request down the normal cross-tier fallback path rather than quietly
        handing it to a different backend in the same tier. After a fallback
        hop (`tier != decision.tier`) the pin no longer applies and ordinary
        weighted selection resumes in the fallback tier.

        `estimated_tokens` -- length-aware routing (see docs/DESIGN.md's
        "Length-aware routing" section). `None` whenever
        `config.length_routing.enabled` is False (its ship-dark default, set
        once by `_run`) -- ordinary (non-forced) selection is then IDENTICAL
        to before this feature existed. When not `None`, ordinary selection
        additionally excludes this tier's enabled `role="candidate"`
        backend(s) once `estimated_tokens` reaches
        `length_routing.threshold_tokens` (see
        `backends.length_routing_excluded_ids`) -- below threshold, nothing
        changes and the candidate keeps its normal weight share. A manual
        override (the `forced_backend_id` branch above) bypasses length
        routing entirely, same as it bypasses ordinary weighted selection:
        an explicit client directive naming an exact backend is already this
        router's established highest-precedence escape hatch (see policy.py's
        `resolve_manual_override`) -- length routing is a guard-rail against
        the drafter's own measured regime, not a new override layer above a
        client's explicit, deliberate choice.

        Returns `(backend, length_routing_extra)`. `length_routing_extra` is
        `{}` unless THIS call actually excluded a candidate for length, in
        which case it carries `length_routing_excluded` /
        `length_routing_estimated_tokens` / `length_routing_reason` for
        `_run` to fold into this attempt's `DecisionRecord.extra` --
        and `length_routing_kind` when kind-aware mode is on -- additive only,
        no existing `DecisionRecord` field is touched."""
        if decision.forced_backend_id is not None and tier == decision.tier:
            backend = next((b for b in self.config.backends if b.id == decision.forced_backend_id), None)
            if backend is not None and backend.tier == tier and backend.enabled and backend.id not in excluded:
                return backend, {}
            return None, {}

        length_routing_extra: dict[str, Any] = {}
        if estimated_tokens is not None:
            excluded_by_length, reason = length_routing_excluded_ids(
                tier,
                self.config.backends,
                length_routing=self.config.length_routing,
                kind=prompt_kind,
                estimated_tokens=estimated_tokens,
            )
            if excluded_by_length:
                excluded = excluded | excluded_by_length
                length_routing_extra = {
                    "length_routing_excluded": True,
                    "length_routing_estimated_tokens": estimated_tokens,
                    "length_routing_reason": reason,
                }
                if prompt_kind is not None:
                    length_routing_extra["length_routing_kind"] = prompt_kind

        backend = select_backend(tier, self.config.backends, exclude_ids=excluded, sticky_key=sticky_key)
        return backend, length_routing_extra

    def dispatch_chat(
        self, request_body: dict, *, sticky_key: str | None = None, request_id: str | None = None
    ) -> DispatchResult:
        """Buffered dispatch: fully reads the backend's response body before
        returning. Correct (and required) for non-streaming requests; also
        used internally by draft-then-escalate, which must see the complete
        draft before a verifier can grade it.

        `request_id` lets a caller that already minted one (server.py's
        RouterHandler, at request entry -- before parsing can fail) supply
        it here so the DecisionRecord this call logs carries the exact id
        the client sees in its response header, rather than a second,
        disagreeing one; omitted (the default), `_run` mints its own, same
        as before -- existing/test callers are unaffected."""
        return self._run(request_body, sticky_key=sticky_key, mode="buffered", request_id=request_id)

    def dispatch_chat_stream(
        self, request_body: dict, *, sticky_key: str | None = None, request_id: str | None = None
    ) -> DispatchResult:
        """Incremental dispatch: returns as soon as the chosen backend's
        response headers arrive; `result.response` is an `OpenResponse` the
        caller reads with `.read_chunk()` and must `.close()`. Error
        detection still happens before any body bytes are read (see
        docs/DESIGN.md: both trailbrake and iliria always send a small buffered
        JSON error, never SSE, when a request is rejected), so fallback on a
        pre-generation error works exactly like the buffered path; a failure
        *mid-stream*, after the client has already received bytes, cannot be
        silently retried elsewhere -- see server.py's `_relay_stream`.

        `request_id` -- see `dispatch_chat`'s docstring."""
        return self._run(request_body, sticky_key=sticky_key, mode="stream", request_id=request_id)

    def dispatch_chat_with_draft_verification(
        self, request_body: dict, *, sticky_key: str | None = None, request_id: str | None = None
    ) -> DispatchResult:
        """Only meaningful when `self.policy` is (or wraps) a
        `DraftThenEscalatePolicy` -- i.e. `getattr(self.policy,
        "wants_draft_first", False)` is True, which `server.py` checks before
        calling this instead of `dispatch_chat`. Runs the first-pass tier
        fully buffered, asks the policy to grade it, and -- on rejection --
        re-runs on `config.escalation.escalation_tier` with a forced
        decision (bypassing `policy.decide` a second time, since the policy
        already made its call).

        `request_id` (see `dispatch_chat`'s docstring) is passed to BOTH the
        draft attempt and a rejection's escalated re-run, not just whichever
        one ends up returned: the caller only ever sees one response header,
        and that header must agree with whichever of the (up to two)
        DecisionRecord rows this call logs corresponds to the result it
        actually got back. Omitted, each of the two `_run` calls mints its
        own id independently, same as before this parameter existed."""
        draft = self._run(request_body, sticky_key=sticky_key, mode="buffered", request_id=request_id)
        accepts_draft = getattr(self.policy, "accepts_draft", None)
        if accepts_draft is None:
            return draft
        draft_text = _extract_message_text(draft.response.body)
        if accepts_draft(request_body, draft_text):
            return draft

        escalation_tier = self.config.escalation.escalation_tier
        forced = RoutingDecision(
            escalation_tier, "draft_rejected", f"verifier rejected draft from backend {draft.backend_id!r}"
        )
        return self._run(
            request_body, sticky_key=sticky_key, mode="buffered", forced_decision=forced, request_id=request_id
        )

    def _run(
        self,
        request_body: dict,
        *,
        sticky_key: str | None,
        mode: str,
        forced_decision: RoutingDecision | None = None,
        request_id: str | None = None,
    ) -> DispatchResult:
        # A caller-supplied id (server.py's RouterHandler, minted at request
        # entry) wins so the response header and this call's DecisionRecord
        # row(s) agree; omitted, mint one here exactly as before -- keeps
        # every existing/test caller that doesn't pass one unchanged.
        request_id = request_id or new_request_id()
        decision = forced_decision or self.policy.decide(request_body, self.config)
        tier = decision.tier
        fallback_from: str | None = None
        tried: set[str] = set()
        last_error: Exception | None = None
        prompt_kind = (
            classify_prompt_kind(request_body.get("messages") or [])
            if self.config.length_routing.enabled and self.config.length_routing.kind_aware
            else None
        )

        # Length-aware routing (docs/DESIGN.md's "Length-aware routing"
        # section; see backends.estimate_prompt_tokens /
        # length_routing_excluded_ids). Estimated at most ONCE per client
        # request, from the ORIGINAL request_body -- not the per-attempt
        # `outgoing_body` mutated below, which only ever adds fields
        # (`enable_thinking`, `stream_options`) and never touches
        # `messages` -- so every attempt this call makes (an in-tier retry
        # after a failure, or a cross-tier fallback hop) judges the same
        # prompt-length estimate. Left `None` (rather than eagerly computed)
        # whenever the feature is off, its ship-dark default: `_select_
        # backend` treats `None` as "do not touch selection at all," which
        # is what keeps disabled behavior byte-identical to before this
        # feature existed.
        estimated_tokens = (
            estimate_prompt_tokens(request_body.get("messages") or [], estimator=self.config.length_routing.estimator)
            if self.config.length_routing.enabled
            else None
        )

        # Escalation evidence for the groundtruth contract (docs/DESIGN.md):
        # computed ONCE per client request from the ORIGINAL body, then merged
        # under every DecisionRecord this call logs -- every trigger path,
        # including manual overrides and markers, because those are the rows
        # that arrive pre-labeled. Hash/indices only, never prompt text.
        features = escalation_features(
            request_body.get("messages") or [], self.config.escalation.hard_markers
        )

        for _ in range(self.MAX_ATTEMPTS):
            excluded = self.circuit_breakers.excluded_backend_ids(self._candidate_ids_for_tier(tier)) | tried
            backend, length_routing_extra = self._select_backend(
                tier,
                decision,
                excluded=excluded,
                sticky_key=sticky_key,
                estimated_tokens=estimated_tokens,
                prompt_kind=prompt_kind,
            )
            # Features ride under the per-attempt extra so every _log site in
            # this loop (and the stream finalizer's floor) carries them.
            length_routing_extra = {**features, **length_routing_extra}

            if backend is None:
                fallback_tier = self.config.fallback.get(tier)
                if not fallback_tier or fallback_tier in (decision.tier, tier):
                    break  # no more tiers to try, or we would bounce back to where we started
                fallback_from = fallback_from or tier
                tier = fallback_tier
                continue

            breaker = self.circuit_breakers.get(backend.id)
            if not breaker.allow_request():
                # Lost a race for the one half-open trial slot (or the peek
                # used for selection is now stale): treat exactly like
                # "already tried" and let the loop pick another candidate.
                tried.add(backend.id)
                continue

            tried.add(backend.id)
            canary = backend.role == "candidate"
            client = self.clients[backend.id]
            outgoing_body = _with_no_think_default(request_body, tier, self.config.escalation.default_tier)
            outgoing_body = _with_stream_usage_requested(outgoing_body, mode)
            started = time.monotonic()
            try:
                response = self._call(client, outgoing_body, mode)
            except BackendRequestFailed as error:
                if 400 <= error.status < 500:
                    # A 4xx is a client error (bad request / unknown model /
                    # auth), not a backend-health problem: the backend is fine,
                    # the *request* is wrong, and every other backend would
                    # reject it identically. Do NOT trip the breaker and do NOT
                    # fall back -- surface the client's own 4xx verbatim. No
                    # bytes have been sent yet, so re-raising renders cleanly
                    # through the server's _handle_error.
                    latency = time.monotonic() - started
                    self._log(
                        request_id, decision, backend.id, tier, fallback_from,
                        "client_error", error.status, latency, canary=canary, extra=length_routing_extra,
                    )
                    raise
                # A 5xx means the backend itself failed -> record + fall back.
                breaker.record_failure()
                latency = time.monotonic() - started
                self._log(
                    request_id, decision, backend.id, tier, fallback_from, "backend_error",
                    error.status, latency, canary=canary, extra=length_routing_extra,
                )
                last_error = error
                continue
            except OSError as error:
                # A transport failure (connect refused / timeout / reset) is an
                # availability problem -> record + fall back.
                breaker.record_failure()
                latency = time.monotonic() - started
                self._log(
                    request_id, decision, backend.id, tier, fallback_from, "backend_error",
                    None, latency, canary=canary, extra=length_routing_extra,
                )
                last_error = error
                continue
            except Exception:
                # Anything else unrecognized (e.g. http.client.IncompleteRead
                # or BadStatusLine -- HTTPException subclasses, not OSError,
                # so the branch above never sees them) must still release
                # this backend's circuit breaker before propagating. Without
                # this, a call that raises during the one half-open trial
                # (see circuit.py's `_half_open_trial_in_flight`) never
                # reaches either `record_success` or `record_failure`, so
                # that flag stays True forever -- every future
                # `allow_request()` call for this backend then returns
                # False permanently (stuck HALF_OPEN, no timer left to
                # re-arm), excluding an otherwise-healthy backend until the
                # process restarts. Treated as a backend failure for
                # breaker purposes, same as OSError, then re-raised
                # unchanged -- this is a genuinely unexpected condition, not
                # an ordinary retry-eligible one, so it must still surface
                # to the caller exactly as it did before this fix.
                breaker.record_failure()
                latency = time.monotonic() - started
                self._log(
                    request_id, decision, backend.id, tier, fallback_from, "backend_error",
                    None, latency, canary=canary, extra=length_routing_extra,
                )
                raise
            else:
                if mode == "stream":
                    # Headers arrived, but the body is relayed by the caller, so
                    # the backend's real outcome (clean completion vs. mid-stream
                    # interruption vs. client abort) is not known yet. Record NO
                    # breaker result and log nothing here; hand back a single-shot
                    # finalizer the transport calls exactly once when the relay
                    # ends. This is what stops a hung/aborted SSE from being
                    # miscounted as a success. Any optional telemetry headers are
                    # extracted NOW (headers are already fully known -- though
                    # trailbrake never actually sends any on a stream, see
                    # _extract_backend_telemetry's docstring) and carried into
                    # the finalizer's eventual log call as its `extra` floor;
                    # the transport layer (server.py's _relay_stream) supplies
                    # the REAL streamed telemetry -- parsed from the closing SSE
                    # `usage` event -- as the finalizer's second argument once
                    # the body has actually been relayed; see
                    # _make_stream_finalizer.
                    return DispatchResult(
                        response, backend.id, tier, decision, canary, fallback_from, request_id,
                        finalize_stream=self._make_stream_finalizer(
                            breaker, request_id, decision, backend.id, tier, fallback_from, canary, started,
                            extra={**length_routing_extra, **_extract_backend_telemetry(response.headers)},
                        ),
                    )
                breaker.record_success()
                latency = time.monotonic() - started
                self._log(
                    request_id, decision, backend.id, tier, fallback_from, "ok",
                    response.status, latency, canary=canary,
                    extra={**length_routing_extra, **_extract_backend_telemetry(response.headers)},
                )
                return DispatchResult(response, backend.id, tier, decision, canary, fallback_from, request_id)

        self._log(request_id, decision, None, tier, fallback_from, "no_backend_available", None, 0.0)
        raise NoBackendAvailable(decision.tier) from last_error

    def _make_stream_finalizer(
        self, breaker, request_id, decision, backend_id, tier, fallback_from, canary, started,
        *, extra: dict[str, Any] | None = None,
    ) -> Callable[[StreamOutcome, "dict[str, Any] | None"], None]:
        """Build the single-shot callback that records a streamed response's
        breaker result + final log, deferred until the transport reports how the
        stream actually ended. SUCCESS -> one backend success; BACKEND_FAILURE
        -> one backend failure logged `stream_interrupted`; CLIENT_ABORT -> the
        backend was healthy, so its breaker is left untouched (logged
        `client_disconnect`). Guarded so a finally/cleanup double-call is a
        no-op.

        `extra` (typically `_extract_backend_telemetry(response.headers)`,
        captured by the caller back when the response headers first arrived --
        in practice always `{}` for a real trailbrake stream, see that function's
        docstring) is the FLOOR merged into whichever of the three log calls
        actually fires. `finalize`'s own `body_telemetry` parameter -- passed
        by the transport layer (server.py's `_relay_stream`) once the body has
        been fully relayed, typically `_extract_stream_usage_telemetry`'s
        result -- is layered on top of it (`{**extra, **body_telemetry}`, so a
        present body-derived value always wins on the rare chance both sources
        ever produced the same field). This is what actually gets streamed
        decode-tok/s, TTFT, and draft-acceptance-rate into the decision log:
        `extra` alone is not enough for a real backend (trailbrake never sends these
        as headers on a stream), and `_with_stream_usage_requested` upstream
        is what makes trailbrake compute the closing usage event this second
        argument is parsed from in the first place. Defaults to `None`
        (treated as `{}`) purely so existing single-argument call sites
        (`finalize_stream(outcome)`, e.g. in tests that don't care about
        telemetry) keep working unchanged."""
        state = {"done": False}

        def finalize(outcome: StreamOutcome, body_telemetry: "dict[str, Any] | None" = None) -> None:
            if state["done"]:
                return  # single-shot: never double-count
            state["done"] = True
            latency = time.monotonic() - started
            if outcome is StreamOutcome.SUCCESS:
                breaker.record_success()
                status = "ok"
            elif outcome is StreamOutcome.BACKEND_FAILURE:
                breaker.record_failure()
                status = "stream_interrupted"
            else:  # CLIENT_ABORT -- backend did its job; do not touch its breaker
                status = "client_disconnect"
            merged_extra = {**(extra or {}), **(body_telemetry or {})}
            self._log(
                request_id, decision, backend_id, tier, fallback_from, status, None, latency,
                canary=canary, extra=merged_extra,
            )

        return finalize

    @staticmethod
    def _call(client: BackendClient, request_body: dict, mode: str) -> BackendResponse | OpenResponse:
        if mode == "buffered":
            return client.chat_completions(request_body)
        opened = client.open(request_body)
        if opened.status >= 400:
            body = drain(opened)
            opened.close()
            raise BackendRequestFailed(client.config.id, opened.status, body.decode("utf-8", "replace"))
        return opened

    def _log(
        self,
        request_id: str,
        decision: RoutingDecision,
        backend_id: str | None,
        tier: str,
        fallback_from: str | None,
        status: str,
        http_status: int | None,
        latency_s: float,
        *,
        canary: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = DecisionRecord(
            request_id=request_id,
            tier=tier,
            backend_id=backend_id,
            trigger=decision.trigger,
            reason=decision.reason,
            canary=canary,
            fallback_from=fallback_from,
            status=status,
            http_status=http_status,
            latency_s=latency_s,
            created_utc=utc_now_iso(),
            extra=extra or {},
        )
        self.telemetry.log(record)
