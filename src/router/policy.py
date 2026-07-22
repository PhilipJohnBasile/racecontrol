"""Escalation policy: decides which *tier* (e.g. "fast" vs "deep") handles a
request, before any specific backend inside that tier is selected -- backend
selection (baseline vs canary-candidate weighting) is `backends.select_backend`'s
job, not this module's. Keeping the two separate is deliberate: a policy
should never need to know that "fast" currently has a canary running, and the
canary mechanism should never need to know why a request landed in "fast".

Trigger priority in the shipped default policy (`DefaultPolicy`), highest
precedence first:

  1. manual override      -- the client names a tier or backend id directly.
  2. explicit hard marker  -- `reasoning_effort in {"high","xhigh"}` (a field
     iliria's own API already implements natively, so a client that wants
     deep reasoning is speaking a real API field, not router-only vocabulary)
     or a configured literal token in the last user message (e.g. "#deep").
  3. task-type heuristic   -- a cheap, pattern-based "hardness" score against
     a configurable threshold.
  4. default tier          -- trailbrake-fast, the common case, zero extra latency.

This order is a deliberate cost/precision trade-off, not an arbitrary list:
escalating to iliria on a false positive is far more expensive (~1.6 tok/s,
streamed, minutes for a long answer) than staying on trailbrake on a false
negative (trailbrake is fast enough that a wrong "stay fast" guess just means a
mediocre answer, not a stalled one) -- see docs/DESIGN.md, "Escalation
policy," for the full argument. That asymmetry is why triggers 1-2 (cheap,
near-zero false-positive rate) run before trigger 3 (the fuzziest one), and
why the fuzzy heuristic alone is never enough to justify shipping it as a
loud, high-recall default.

`DraftThenEscalatePolicy` is the fourth, more expensive pattern this module
supports (trailbrake answers first; a verifier grades the draft; a rejected draft
is transparently re-run on iliria). It ships disabled
(`escalation.enable_draft_then_escalate = false`) -- see its docstring below.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import RouterConfig

_HIGH_EFFORT = {"high", "xhigh"}

# Patterns weighted by how strongly they suggest an algorithmically/
# analytically hard task rather than a boilerplate edit. This is a v0
# heuristic -- pattern-matched signal, not a learned classifier. Tune
# `heuristic_threshold` in config before editing these weights; see
# docs/DESIGN.md's honesty note on this trigger's precision.
#
# The third element is a *floor*: the minimum final score this message can
# have once this pattern matches, no matter how many `_EASY_SIGNAL_PATTERNS`
# also match. Without it the heuristic is purely additive, so a strong,
# unambiguous signal like "deadlock" (0.4, already the highest weight here)
# can be dragged back under the escalation threshold by ordinary boilerplate
# wording -- e.g. "write a unit test for this deadlock" scored 0.4 - 0.2 =
# 0.2. Only the two highest-precision patterns (a named concurrency/memory
# failure, or "why does X fail/crash/hang") get a floor; the fuzzier hard
# signals below (design trade-offs, "think carefully", ...) are common
# enough in ordinary requests that they stay purely additive.
#
# Measured honesty (2026-07-22, bench/escalation_eval/): on an 84-case blind
# held-out set, every escalation this table produced at the old 0.6 default
# threshold was a false positive (3/3 -- all trivial requests that merely
# *mentioned* a scary word, e.g. "add a unit test for the race condition we
# already fixed", where the floor overrode the easy signal built to catch
# exactly that), and it caught 0 of 42 genuinely hard prompts. The default
# threshold is now 0.7 (config.py), which a bare floored match (0.600) does
# not clear -- so at default config, escalation comes from explicit markers
# and manual overrides, and this table only fires on multi-pattern prompts.
# Two independently proposed vocabulary extensions were measured net-negative
# on the same set (recall bought at 100x-cost false positives, plus a
# multi-turn escalation latch) -- see bench/escalation_eval/RESULTS.md before
# adding patterns here.
_HARD_SIGNAL_FLOOR = 0.6

_HARD_SIGNAL_PATTERNS: tuple[tuple[re.Pattern[str], float, float], ...] = (
    (re.compile(r"\bwhy (is|does|did)\b.{0,40}\b(fail\w*|flak\w*|intermittent|race|crash|hang\w*)", re.I), 0.4, _HARD_SIGNAL_FLOOR),
    (re.compile(r"\brace condition\b|\bdeadlock\b|\bmemory corruption\b|\bundefined behaviou?r\b", re.I), 0.4, _HARD_SIGNAL_FLOOR),
    (re.compile(r"\bNP-hard\b|\bbig-?O\b|\basymptotic\b|\binvariant\b", re.I), 0.3, 0.0),
    (re.compile(r"\bprove\b|\bproof\b|\bcounterexample\b|\bformally\b", re.I), 0.3, 0.0),
    (re.compile(r"\broot.?cause\b|\bpost-?mortem\b", re.I), 0.25, 0.0),
    (re.compile(r"\btrade-?offs?\b|\barchitect\w* .{0,20}\bdesign\b|\bdesign\b.{0,20}\barchitect\w*", re.I), 0.25, 0.0),
    (re.compile(r"\bstep.by.step\b|\bthink (carefully|deeply)\b|\breason (carefully|deeply)\b", re.I), 0.3, 0.0),
)
_EASY_SIGNAL_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\brename\b|\badd an? getter\b|\bboilerplate\b", re.I), -0.3),
    (re.compile(r"\bfix (the )?typo\b|\bformat(ting)?\b|\blint\b", re.I), -0.3),
    (re.compile(r"\bwrite a (unit )?test for\b", re.I), -0.2),
)

# Short content hash over both tables (patterns + weights + floors), logged
# beside every `escalation_features` row -- so the `hard_hits`/`easy_hits`
# indices in old log lines stay interpretable after the tables change: an
# offline reader maps indices back to patterns via the source at whichever
# version stamped the row. Computed once at import; NOT a config knob.
_PATTERNS_VERSION = hashlib.sha256(
    repr(
        [(p.pattern, w, f) for p, w, f in _HARD_SIGNAL_PATTERNS]
        + [(p.pattern, w) for p, w in _EASY_SIGNAL_PATTERNS]
    ).encode()
).hexdigest()[:8]

# How many of the most recent user turns feed `hardness_score`, and how much
# an older turn's contribution shrinks per turn of distance from the current
# one. Fixes a real flip-flop: scoring only the single last user message
# means a hard opening question ("why does this deadlock intermittently?")
# followed by a terse in-context follow-up ("ok, also add a test for that")
# scores the follow-up alone and drops straight back to the fast tier -- see
# docs/DESIGN.md / the audit's "Escalation policy" section. `_TURN_DECAY` was
# chosen so one turn back from a floored hard signal (0.8 raw) still
# comfortably cleared the then-default 0.6 threshold (0.8 * 0.8 = 0.64),
# while two turns back (0.8 * 0.64 = 0.512) let it fall back to normal -- a
# short memory, not a permanent escalation latch. (That arithmetic predates
# the current 0.7 default threshold, under which a floored signal alone does
# not escalate on any turn; the decay still bounds how long multi-pattern
# scores persist. Measured context: bench/escalation_eval/RESULTS.md.)
_RECENT_TURNS = 3
_TURN_DECAY = 0.8


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    tier: str
    trigger: str
    reason: str
    # Set only by a manual override that named a specific backend id/model_id
    # (as opposed to a bare tier name): `dispatch.py`'s `RequestRouter` pins
    # this exact backend for the request, bypassing `backends.select_backend`'s
    # weighted draw over the rest of the tier -- see `resolve_manual_override`.
    # None for every other trigger (explicit marker, task heuristic, default),
    # which only ever pick a *tier* and leave the in-tier backend choice to
    # the weighted canary split.
    forced_backend_id: str | None = None


def _extract_text(content: object) -> str | None:
    """Plain text of one message's `content`, or None if it carries none.
    Handles both shapes the chat API allows: a bare string, and a
    multimodal content-parts list (`[{"type": "text", "text": "..."}, ...]`,
    e.g. alongside an `image_url` part). Security-relevant: earlier, only a
    bare string was read, so a request shaped as content-parts -- trivial
    for any real client to send -- made `find_hard_marker`/`hardness_score`
    see an empty string no matter what the text part said, silently
    bypassing both the explicit-marker and task-heuristic escalation
    triggers (a "#deep, please look at this race condition" wrapped in
    content-parts shape stayed on the cheap/pruned tier). Non-text parts
    (images, etc.) are ignored; multiple text parts are joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
        if parts:
            return "\n".join(parts)
    return None


def _recent_user_texts(messages: list, limit: int) -> list[str]:
    """Up to `limit` user-turn texts, most recent first. A message whose
    `content` carries no extractable text (see `_extract_text`) is skipped,
    not treated as a stopping point, so an older text-bearing turn can still
    be found."""
    texts: list[str] = []
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _extract_text(message.get("content"))
            if text is not None:
                texts.append(text)
                if len(texts) >= limit:
                    break
    return texts


def _last_user_text(messages: list) -> str:
    texts = _recent_user_texts(messages, limit=1)
    return texts[0] if texts else ""


def _single_turn_score(text: str) -> float:
    """Un-clamped hard/easy-signal score for one message's text, with
    `_HARD_SIGNAL_FLOOR` semantics applied (see `_HARD_SIGNAL_PATTERNS`) but
    not yet the [0, 1] clamp or cross-turn decay -- both are
    `hardness_score`'s job, since only it knows whether this text is the
    latest turn or an older one."""
    score = 0.0
    floor = 0.0
    for pattern, weight, pattern_floor in _HARD_SIGNAL_PATTERNS:
        if pattern.search(text):
            score += weight
            floor = max(floor, pattern_floor)
    for pattern, weight in _EASY_SIGNAL_PATTERNS:
        if pattern.search(text):
            score += weight
    return max(score, floor)


def hardness_score(messages: list) -> float:
    """Pattern-based "how much this looks like a hard reasoning/debugging
    task vs. a boilerplate edit" score, clamped to [0, 1]. Deliberately
    simple (stdlib `re` only, no ML) and deliberately named as a score, not a
    classification -- callers compare it against a configurable threshold
    rather than treating it as a verdict.

    Scores the last `_RECENT_TURNS` user turns, not just the latest one,
    each discounted by `_TURN_DECAY` per turn of distance from the current
    one, and takes the max rather than summing -- so a hard turn's signal
    persists for a couple of exchanges instead of vanishing the instant the
    client sends a short follow-up, without letting several merely-medium
    turns stack into a false escalation the way summing would."""
    score = 0.0
    decay = 1.0
    for text in _recent_user_texts(messages, limit=_RECENT_TURNS):
        score = max(score, _single_turn_score(text) * decay)
        decay *= _TURN_DECAY
    return max(0.0, min(1.0, score))


def find_hard_marker(messages: list, markers: tuple[str, ...]) -> str | None:
    """Token-boundary match of a configured marker (e.g. "#deep") against
    the last user message. Plain substring matching would let a marker match
    inside an unrelated longer token -- "#deep" inside "#deepfake" -- so a
    match requires a non-word character (or start/end of string) on both
    sides of the marker. `(?<!\\w)`/`(?!\\w)` rather than `\\b`: every
    shipped marker starts with a non-word character (#, /), and `\\b` does
    not count the very start of the string as a boundary when the adjacent
    character is already non-word, which would wrongly reject a marker
    sitting at the start of a message."""
    text = _last_user_text(messages)
    for marker in markers:
        if marker and re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text):
            return marker
    return None


def escalation_features(messages: list, markers: tuple[str, ...] = ()) -> dict:
    """Per-request escalation evidence for the decision log -- the
    groundtruth contract's feature side (docs/DESIGN.md, "The
    decision-groundtruth contract"). Logged for EVERY request, whatever
    trigger actually decided it: a manual override or explicit marker is a
    *labeled* example (the caller told us the right tier), and those labels
    are worthless without the features that accompanied them.

    Privacy: no prompt text leaves this function -- only the score, pattern
    hit indices, lengths, and a one-way fingerprint.

    - `hardness_score`: exactly `hardness_score(messages)`, the number the
      heuristic compares against the threshold. Logged even though the
      heuristic is inert at the default threshold (bench/escalation_eval/):
      a future learned classifier needs the v0 signal as a baseline feature.
    - `hard_hits` / `easy_hits`: indices into `_HARD_SIGNAL_PATTERNS` /
      `_EASY_SIGNAL_PATTERNS` that matched the LATEST user turn. Indices,
      not pattern strings, to keep rows small; `patterns_version` makes them
      interpretable after future table edits.
    - `prompt_fingerprint`: sha256 (first 16 hex) of the latest user text,
      lowercased, whitespace-collapsed, with configured markers stripped
      FIRST -- so "explain X" and "#deep explain X" collide on purpose.
      That collision is the offline labeler's join key: a fast-tier answer
      followed shortly by the same fingerprint carrying a marker is a
      labeled miss, no human annotation needed.
    - `had_marker`: whether a configured marker was present (the stripped
      fingerprint no longer shows it).
    - `latest_chars` / `user_turns`: cheap size features.
    """
    latest = _last_user_text(messages)
    stripped = latest
    for marker in markers:
        if marker:
            stripped = re.sub(rf"(?<!\w){re.escape(marker)}(?!\w)", " ", stripped)
    normalized = " ".join(stripped.lower().split())
    return {
        "hardness_score": round(hardness_score(messages), 3),
        "hard_hits": [i for i, (pattern, _, _) in enumerate(_HARD_SIGNAL_PATTERNS) if pattern.search(latest)],
        "easy_hits": [i for i, (pattern, _) in enumerate(_EASY_SIGNAL_PATTERNS) if pattern.search(latest)],
        "patterns_version": _PATTERNS_VERSION,
        "prompt_fingerprint": hashlib.sha256(normalized.encode()).hexdigest()[:16],
        "had_marker": find_hard_marker(messages, markers) is not None,
        "latest_chars": len(latest),
        "user_turns": sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user"),
    }


def resolve_manual_override(request: dict, config: RouterConfig) -> RoutingDecision | None:
    """A client that names a configured tier or backend id/model_id directly
    in `model` bypasses the policy entirely. This is the escape hatch for
    testing, debugging, and a power user who wants to force a specific tier
    -- see docs/DESIGN.md, "Model identity across three namespaces."""
    requested = request.get("model")
    if not isinstance(requested, str) or not requested:
        return None
    if requested in config.tiers():
        return RoutingDecision(requested, "manual_override", f"client requested tier {requested!r} directly")
    for backend in config.backends:
        if requested in (backend.id, backend.model_id):
            return RoutingDecision(
                backend.tier,
                "manual_override",
                f"client requested backend {backend.id!r} directly",
                forced_backend_id=backend.id,
            )
    return None


@runtime_checkable
class EscalationPolicy(Protocol):
    def decide(self, request: dict, config: RouterConfig) -> RoutingDecision: ...


class DefaultPolicy:
    """manual override -> explicit marker -> task heuristic -> default tier."""

    def decide(self, request: dict, config: RouterConfig) -> RoutingDecision:
        override = resolve_manual_override(request, config)
        if override is not None:
            return override

        escalation = config.escalation
        messages = request.get("messages") or []

        reasoning_effort = request.get("reasoning_effort")
        # `isinstance` first, not just `in _HIGH_EFFORT`: a malformed client
        # value that happens to be unhashable (a list or dict -- trivial for
        # a buggy or adversarial client to send) would otherwise raise
        # TypeError out of a bare set-membership check, which crashes this
        # request before the marker/heuristic triggers below ever get a
        # chance to look at its actual text -- turning one malformed field
        # into a silent loss of escalation coverage for a message that might
        # otherwise clearly deserve it, not just a robustness bug. A
        # non-string value can never legitimately match `_HIGH_EFFORT`
        # anyway, so this changes no behavior for any well-formed request.
        if isinstance(reasoning_effort, str) and reasoning_effort in _HIGH_EFFORT:
            return RoutingDecision(
                escalation.escalation_tier, "explicit_marker", f"reasoning_effort={reasoning_effort!r}"
            )

        marker = find_hard_marker(messages, escalation.hard_markers)
        if marker is not None:
            return RoutingDecision(escalation.escalation_tier, "explicit_marker", f"marker {marker!r} in message")

        if escalation.enable_task_heuristic:
            score = hardness_score(messages)
            if score >= escalation.heuristic_threshold:
                return RoutingDecision(
                    escalation.escalation_tier,
                    "task_heuristic",
                    f"hardness_score={score:.2f} >= threshold={escalation.heuristic_threshold:.2f}",
                )

        return RoutingDecision(escalation.default_tier, "default", "no escalation signal matched")


class AlwaysTierPolicy:
    """Fixed-tier policy (still honors a manual override). Useful for tests,
    and for an operator who wants to temporarily pin all traffic to one tier
    without touching the backend `enabled`/`weight` knobs."""

    def __init__(self, tier: str) -> None:
        self.tier = tier

    def decide(self, request: dict, config: RouterConfig) -> RoutingDecision:
        override = resolve_manual_override(request, config)
        if override is not None:
            return override
        return RoutingDecision(self.tier, "fixed_policy", f"AlwaysTierPolicy({self.tier!r})")


@runtime_checkable
class Verifier(Protocol):
    """Mirrors trailbrake's scoring harness's
    `score_response(task, response_text) -> VerifierResult` shape: a callable
    that grades a draft response and says whether it is good enough. The
    router ships no real verifier -- that requires task metadata (expected
    output, a test command, a rubric) this generic proxy does not have.
    Callers plug one in; see docs/DESIGN.md's worked example."""

    def __call__(self, request: dict, draft_response_text: str) -> bool: ...


class DraftThenEscalatePolicy:
    """Draft-then-verify: get a first-pass tier decision from `inner`
    (typically `DefaultPolicy`, so explicit markers/overrides still
    short-circuit straight to escalation -- there is no reason to spend an
    trailbrake draft on a request the client already said needs iliria); the
    caller (`dispatch.RequestRouter`) runs that tier, then calls
    `accepts_draft` with the response text. A rejection means "re-run this
    on `config.escalation.escalation_tier` instead."

    `wants_draft_first = True` is a marker `server.py` checks for -- this
    class changes the request lifecycle (one extra generation-and-verify
    round trip before any tokens reach the client) enough that it needs
    explicit handling, not just a different `decide()` return value.

    Ships disabled by default. With no `verifier` wired in, `accepts_draft`
    always returns True, which degrades this policy to "run the default
    tier, then always accept its answer" -- strictly worse than
    `DefaultPolicy` (same outcome, plus a policy-dispatch indirection) rather
    than actually safer. It is only worth enabling once a real verifier
    exists for the traffic in question (e.g. an adaptation of
    trailbrake's own `trailbrake's campaign benchmark/scoring.py` verifiers for
    coding tasks that carry a test command or expected-output rubric).
    """

    wants_draft_first = True

    def __init__(self, inner: EscalationPolicy, verifier: Verifier | None = None) -> None:
        self._inner = inner
        self._verifier = verifier

    def decide(self, request: dict, config: RouterConfig) -> RoutingDecision:
        return self._inner.decide(request, config)

    def accepts_draft(self, request: dict, draft_response_text: str) -> bool:
        if self._verifier is None:
            return True
        return self._verifier(request, draft_response_text)


def build_policy(config: RouterConfig, *, verifier: Verifier | None = None) -> EscalationPolicy:
    name = config.escalation.policy
    base: EscalationPolicy
    if name == "default":
        base = DefaultPolicy()
    elif name in config.tiers():
        base = AlwaysTierPolicy(name)
    else:
        raise ValueError(f"unknown escalation policy {name!r}")

    if config.escalation.enable_draft_then_escalate:
        return DraftThenEscalatePolicy(base, verifier)
    return base
