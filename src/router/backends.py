"""HTTP client for a single configured backend (trailbrake or iliria in the
shipped deployment; this module itself knows neither name -- it only knows
the generic OpenAI-compatible shape both already speak, per docs/DESIGN.md's
backend API notes).

Translation this layer is responsible for, given what reading both real
servers showed (`trailbrake/src/mlx_engine/server.py`,
`iliria/c/openai_server.py`):

  * `model` is rewritten to this backend's own configured `model_id` before
    the request leaves the router. trailbrake accepts loose aliases (its own model
    directory name, or the literal string "default"); iliria requires an
    exact string match against its `--model-id` (`check_model` in
    `openai_server.py`, 404 `model_not_found` otherwise). Neither backend
    should ever see the router's own virtual tier/backend names.
  * `Authorization: Bearer <api_key>` is attached only if this backend's
    config carries one -- trailbrake has no auth layer at all; iliria's is
    optional (`--api-key` / `require_auth()`).
  * Timeouts are two numbers, not one: `connect_timeout_s` (TCP handshake)
    and `idle_timeout_s` (max gap between individual socket reads once the
    response has started, applied as the connection's socket timeout). A
    single flat request timeout would be actively wrong against iliria: at
    ~1.6 tok/s a legitimate 1024-token escalation can take minutes end to
    end, but the router must still fail fast if the connection goes truly
    silent. `http.client.HTTPResponse.read(n)` on a chunked response already
    de-chunks incrementally, so a bounded `read_chunk` loop naturally waits
    only as long as `idle_timeout_s` between arrivals, never for the whole
    body -- see docs/DESIGN.md, "Why two timeouts."
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
import re
from http.client import HTTPConnection, HTTPSConnection
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from .config import BackendConfig, LengthRoutingConfig
from .errors import BackendRequestFailed


@dataclass
class BackendResponse:
    status: int
    headers: dict[str, str]
    body: bytes


@runtime_checkable
class OpenResponse(Protocol):
    status: int
    headers: dict[str, str]

    def read_chunk(self, size: int = 65536) -> bytes:
        """Returns up to `size` bytes, or `b""` at end of body."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class Transport(Protocol):
    """Everything BackendClient needs from an HTTP client. The default
    implementation (`HttpTransport`) wraps stdlib `http.client`; tests inject
    a fake so no real socket is ever opened -- see tests/fakes.py."""

    def open(
        self,
        *,
        base_url: str,
        method: str,
        path: str,
        body: bytes | None,
        headers: dict[str, str],
        connect_timeout_s: float,
        idle_timeout_s: float,
    ) -> OpenResponse: ...


class _HttpOpenResponse:
    def __init__(self, connection, response) -> None:
        self._connection = connection
        self._response = response
        self.status = response.status
        self.headers = {key.lower(): value for key, value in response.getheaders()}

    def read_chunk(self, size: int = 65536) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        self._connection.close()


class HttpTransport:
    """Real stdlib `http.client` transport."""

    def open(
        self,
        *,
        base_url: str,
        method: str,
        path: str,
        body: bytes | None,
        headers: dict[str, str],
        connect_timeout_s: float,
        idle_timeout_s: float,
    ) -> OpenResponse:
        parts = urlsplit(base_url)
        connection_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
        connection = connection_cls(parts.hostname, parts.port, timeout=connect_timeout_s)
        try:
            connection.connect()
            connection.sock.settimeout(idle_timeout_s)
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
        except Exception:
            connection.close()
            raise
        return _HttpOpenResponse(connection, response)


def drain(opened: OpenResponse) -> bytes:
    """Fully reads an OpenResponse without closing it (caller closes)."""
    chunks: list[bytes] = []
    while True:
        chunk = opened.read_chunk(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _rewrite_model(body: dict, model_id: str) -> dict:
    rewritten = dict(body)
    rewritten["model"] = model_id
    return rewritten


class BackendClient:
    def __init__(self, config: BackendConfig, *, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or HttpTransport()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def open(self, request_body: dict) -> OpenResponse:
        """Opens the connection and returns as soon as headers arrive,
        without reading the body -- the caller decides whether to drain it
        fully (non-streaming) or relay it incrementally (streaming)."""
        body = _rewrite_model(request_body, self.config.model_id)
        payload = json.dumps(body).encode("utf-8")
        return self._transport.open(
            base_url=self.config.base_url,
            method="POST",
            path="/v1/chat/completions",
            body=payload,
            headers=self._headers(),
            connect_timeout_s=self.config.connect_timeout_s,
            idle_timeout_s=self.config.idle_timeout_s,
        )

    def chat_completions(self, request_body: dict) -> BackendResponse:
        """Buffered convenience wrapper over `open()` for non-streaming call
        sites (and for tests that don't care about incremental delivery).
        Raises `BackendRequestFailed` for any >=400 response, with the
        (small, always-non-streamed on both real backends -- see
        docs/DESIGN.md) error body decoded into the exception message."""
        opened = self.open(request_body)
        try:
            data = drain(opened)
            if opened.status >= 400:
                raise BackendRequestFailed(self.config.id, opened.status, data.decode("utf-8", "replace"))
            return BackendResponse(opened.status, opened.headers, data)
        finally:
            opened.close()

    def health(self) -> bool:
        try:
            opened = self._transport.open(
                base_url=self.config.base_url,
                method="GET",
                path="/health",
                body=None,
                headers=self._headers(),
                connect_timeout_s=self.config.connect_timeout_s,
                idle_timeout_s=self.config.idle_timeout_s,
            )
        except OSError:
            return False
        try:
            return opened.status == 200
        finally:
            opened.close()


# -- Length-aware arm routing ---------------------------------------------
#
# A guard-rail on a tier's drafter-candidate arm, not a promoter. Measured
# 2026-07-20: the drafter's speedup over dense decays -- and, past some
# prompt length, inverts -- as the prompt grows (1.35x at ~4K prompt tokens,
# falling to 0.98x/1.01x at ~6.1K/7.3K; acceptance rate falling from a 0.708
# to a 0.315 median over the same range, an early 7-pair sample). A blind
# weighted canary draw (`select_backend` below) cannot see prompt length at
# all, so it keeps sending the drafter exactly where it has already measured
# a loss. See docs/DESIGN.md's "Length-aware routing" section for the full
# argument and `config.LengthRoutingConfig` for the config knobs read here.

_CHARS_PER_TOKEN = 4  # the "chars_div4" estimator's divisor -- see estimate_prompt_tokens
_MULTI_LINE_MIN_CHARS = 1200
_RETRIEVAL_MAX_TRAILING_QUESTION_WORDS = 26

_RETRIEVAL_CUE = re.compile(
    r"\b(?:based\s+(?:only|solely)\s+on|based\s+only\s+on\s+the\s+document\s+excerpt|"
    r"document\s+excerpt|source\s+document|citing\s+the\s+specific|answer\s+concisely)\b",
    re.I,
)
_MULTITURN_ROLE = "assistant"
_GENERIC_VERB_CUES = (
    "implement",
    "implementing",
    "write",
    "writing",
    "add",
    "adding",
    "change",
    "changing",
    "modify",
    "modifying",
    "fix",
    "fixing",
    "refactor",
    "create",
    "creates",
    "build",
    "building",
    "propose",
    "code-level",
)
_GENERIC_VERB_RE = re.compile(r"\b(?:" + "|".join(re.escape(verb) for verb in _GENERIC_VERB_CUES) + r")\b", re.I)
_FILE_PATH_RE = re.compile(r"\b[\w./-]+\.[a-zA-Z]{1,12}\b")
_CODE_FENCE_RE = re.compile(r"```")


def _extract_text(content: object) -> str | None:
    """Shared, robust user-message text extraction used by both length
    estimation and kind classification. Supports the same two shapes as
    `policy._extract_text` (string and multimodal content-parts list) and
    returns `None` for anything else, so malformed/unusual messages never
    change routing behavior."""
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


def _user_texts(messages: list[dict]) -> list[str]:
    user_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _extract_text(message.get("content"))
        if text is not None:
            user_texts.append(text)
    return user_texts


def _assistant_turn_count(messages: list[dict]) -> int:
    count = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == _MULTITURN_ROLE:
            count += 1
    return count


def _is_short_trailing_question(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    trailing = lines[-1]
    if not trailing.endswith("?"):
        return False
    return len(trailing.split()) <= _RETRIEVAL_MAX_TRAILING_QUESTION_WORDS


def _has_retrieval_shape(text: str) -> bool:
    if _is_short_trailing_question(text):
        return True
    if "Based ONLY on the document excerpt" in text:
        return True
    if "based only on the document excerpt" in text.lower():
        return True
    if len(text) < _MULTI_LINE_MIN_CHARS:
        return False
    return bool(_RETRIEVAL_CUE.search(text))


def _looks_generative(text: str) -> bool:
    return bool(_CODE_FENCE_RE.search(text)) or bool(_GENERIC_VERB_RE.search(text)) or bool(
        _FILE_PATH_RE.search(text)
    )


def classify_prompt_kind(messages: list) -> str:
    """Heuristic prompt-kind classifier for length-routing thresholds.

    - `multiturn`: any two or more prior assistant messages
    - `retrieval`: long-context + short trailing question
    - `generative`: code-writing/editing signals (code-fence, code-path,
      or edit verbs)

    Returns one of: `generative`, `retrieval`, `multiturn`, or `unknown`.
    The classifier is intentionally dependency-free and deliberately conservative
    to avoid adding false-positive kind routing changes."""
    if _assistant_turn_count(messages) >= 2:
        return "multiturn"

    user_turns = _user_texts(messages)
    if not user_turns:
        return "unknown"

    latest_user_text = user_turns[-1]
    if _has_retrieval_shape(latest_user_text):
        return "retrieval"
    if _looks_generative(latest_user_text):
        return "generative"
    return "unknown"


def _content_char_len(content: object) -> int:
    """Character length of one message's `content`, tolerant of both shapes
    the chat API allows -- a bare string, or a multimodal content-parts list
    (`[{"type": "text", "text": "..."}, ...]`) -- mirroring policy.py's
    `_extract_text` tolerance for the same two shapes (kept as a separate,
    file-local helper rather than a cross-module import: this estimator only
    ever needs a char *count*, not the joined text itself, so backends.py
    does not need to reach into policy.py's internals to compute it).
    Non-text parts (`image_url`, etc.) contribute 0 -- this estimator is
    deliberately text-only and approximate, see `estimate_prompt_tokens`.
    A part with no usable string "text", or a `content` of some other shape
    entirely (`None`, a number, ...), also contributes 0 rather than
    raising: a malformed or unusual message must never crash routing, it
    should just under-count that one message."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part["text"])
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    return 0


def estimate_prompt_tokens(messages: list, *, estimator: str = "chars_div4") -> int:
    """Deliberately tokenizer-free prompt-length estimate, used only to
    decide whether a tier's drafter-candidate arm is even eligible for this
    request (see `length_routing_excluded_ids` below) -- NOT precise enough
    for anything that actually depends on an exact count, like context-
    window enforcement.

    `"chars_div4"` (the only estimator implemented; `config.py`'s
    `LengthRoutingConfig.estimator` validates against
    `LENGTH_ROUTING_ESTIMATORS` at config-load time, so an unrecognized
    value can never reach here from a loaded config) sums every message's
    extractable text length (`_content_char_len`) across the WHOLE request
    -- system prompt and prior turns included, not just the latest user
    message, since what actually determines the prompt length the backend
    (and its drafter) will see is the whole thing -- and divides by 4. This
    is a rough, ~15%-ish approximation (English text commonly averages ~4
    chars/token for the tokenizers both backends use), traded deliberately
    for zero dependencies and near-zero added latency: a real tokenizer call
    on every dispatched request would add real cost to every single
    dispatch just to make one binary drafter-eligibility decision. A
    message that is not a dict, or carries no usable `content`, contributes
    0 rather than raising."""
    if estimator != "chars_div4":
        raise ValueError(f"unknown length-routing estimator {estimator!r}")
    total_chars = sum(
        _content_char_len(message.get("content")) for message in (messages or []) if isinstance(message, dict)
    )
    return total_chars // _CHARS_PER_TOKEN


def length_routing_excluded_ids(
    tier: str,
    backends: tuple[BackendConfig, ...],
    *,
    length_routing: LengthRoutingConfig,
    kind: str | None = None,
    estimated_tokens: int,
) -> "tuple[frozenset[str], str | None]":
    """Which of this tier's backend ids `select_backend` should additionally
    exclude for length-aware routing, and a human-readable reason iff an
    exclusion actually happened (`None` otherwise: disabled, no *enabled*
    `role="candidate"` backend in this tier, or the estimate is under
    threshold). Callers fold the returned ids into their own `exclude_ids`
    *before* calling `select_backend`, so the exclusion applies uniformly to
    both of that function's modes -- a weighted draw never sees the excluded
    candidate, and, just as important (see docs/DESIGN.md), a sticky-key
    hash bucket is computed over the already-narrowed candidate list, so a
    sticky user with a long prompt cannot hash their way back onto the
    candidate arm either. This function has no opinion on sticky keys at
    all -- it only decides *which ids*; the caller decides how selection
    uses that.

    Below threshold: returns `(frozenset(), None)` -- nothing is excluded,
    so the candidate keeps exactly its configured weight share of the draw,
    unchanged. This function only ever narrows the field; it never
    *promotes* the candidate for a short prompt -- doing that would trade
    the canary's blindness for a guess this router has no basis to make.

    At/above threshold: excludes every *enabled* `role="candidate"` backend
    in `tier`. A disabled candidate is left out of the returned set
    (and therefore never produces a reason) since `select_backend` already
    excludes it on its own -- attributing that exclusion to length routing
    would be misleading telemetry. If `tier` has no enabled candidate-role
    backend at all, this is a no-op regardless of `enabled`/threshold --
    there is nothing to guard against."""
    if not length_routing.enabled:
        return frozenset(), None
    candidate_ids = frozenset(
        backend.id
        for backend in backends
        if backend.tier == tier and backend.role == "candidate" and backend.enabled
    )
    if not candidate_ids:
        return frozenset(), None
    threshold_tokens = (
        length_routing.threshold_tokens if not length_routing.kind_aware else length_routing.threshold_for_kind(kind)
    )
    if estimated_tokens < threshold_tokens:
        return frozenset(), None
    reason = f"length_routing: {estimated_tokens}tok >= {threshold_tokens} -> candidate excluded"
    return candidate_ids, reason


def _weighted_choice(candidates: list[BackendConfig], *, rng: random.Random) -> BackendConfig:
    total = sum(backend.weight for backend in candidates)
    if total <= 0:
        # Every enabled candidate has weight 0 (misconfigured): fall back to
        # a uniform pick rather than raising, so a canary weight of 0 for
        # every backend in a tier never means "nothing works."
        return rng.choice(candidates)
    pick = rng.uniform(0, total)
    upto = 0.0
    for backend in candidates:
        upto += backend.weight
        if pick <= upto:
            return backend
    return candidates[-1]


def _hash_bucket(key: str, candidates: list[BackendConfig]) -> BackendConfig:
    weights = [backend.weight for backend in candidates]
    if sum(weights) <= 0:
        weights = [1] * len(candidates)
    total = sum(weights)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % total
    upto = 0
    for backend, weight in zip(candidates, weights):
        upto += weight
        if bucket < upto:
            return backend
    return candidates[-1]


def select_backend(
    tier: str,
    backends: tuple[BackendConfig, ...],
    *,
    exclude_ids: frozenset[str] = frozenset(),
    sticky_key: str | None = None,
    rng: random.Random | None = None,
) -> BackendConfig | None:
    """Picks one enabled backend within `tier`, weighted by `.weight` (the
    canary-percentage knob). `exclude_ids` lets the caller skip circuit-open
    or already-tried backends without touching config. `sticky_key`, if
    given, makes the choice a deterministic hash bucket instead of a random
    weighted draw -- the same key (e.g. the request's OpenAI-standard `user`
    field) always lands on the same backend while weights hold steady, which
    is what makes an A/B comparison across turns of one conversation
    meaningful. Returns `None` if no eligible backend exists in this tier;
    the caller (`dispatch.RequestRouter`) is responsible for cross-tier
    fallback in that case."""
    candidates = [
        backend for backend in backends if backend.tier == tier and backend.enabled and backend.id not in exclude_ids
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if sticky_key:
        return _hash_bucket(sticky_key, candidates)
    return _weighted_choice(candidates, rng=rng or random.Random())
