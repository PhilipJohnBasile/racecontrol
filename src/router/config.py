"""Router configuration: backends, escalation policy, fallback, and circuit
breaker tuning, loaded from a TOML file via the stdlib `tomllib` reader --
no third-party TOML/YAML dependency, matching this project's dependency-free
convention (see pyproject.toml's empty `dependencies`, and both trailbrake's and
iliria's own zero/near-zero runtime dependency footprint).

The core routing engine (`policy.py`, `dispatch.py`, `backends.py`) never
hardcodes "trailbrake" or "iliria" -- those are just the `id`s of the backends in
the shipped example config (`config/router.example.toml`). Everything here
is expressed in terms of free-form `tier` names (the shipped default has
exactly two: "fast" and "deep"), so the same router could front a different
pair of engines, or more than two tiers, without touching engine code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from .errors import ConfigError

DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
DEFAULT_HARD_MARKERS: tuple[str, ...] = ("#deep", "#reason", "/escalate")
# `BackendConfig.role` values this project actually uses. The routing engine
# itself only ever branches on the literal string "candidate" --
# dispatch.py's `canary = backend.role == "candidate"` and backends.py's
# `length_routing_excluded_ids` both compare against it, and everything else
# is simply "not the candidate arm." "primary" (this dataclass's own default,
# for a tier with no candidate at all) and "baseline" (the shipped example
# config's/test suite's convention for the non-candidate arm of a tier that
# *does* have a candidate, e.g. "trailbrake-baseline") are both just
# descriptive labels for that -- so all three, not just "candidate", are
# accepted here. Still kept as an explicit allow-list (not "any string"),
# for the same reason as `LENGTH_ROUTING_ESTIMATORS` below: a typo'd `role`
# (e.g. "Candidate", "canidate") must fail loudly at load time, not silently
# load as an uncategorized backend that never gets classified as a canary
# and never gets the length-routing guard-rail applied to it.
BACKEND_ROLES: frozenset[str] = frozenset({"primary", "baseline", "candidate"})
# Estimators `LengthRoutingConfig.estimator` may name -- see backends.py's
# `estimate_prompt_tokens`. Just one today; kept as an explicit allow-list
# (not "any string") so a typo'd config value fails loudly at load time
# instead of silently disabling the feature at request time.
LENGTH_ROUTING_ESTIMATORS: frozenset[str] = frozenset({"chars_div4"})
LENGTH_ROUTING_KIND_NAMES: tuple[str, ...] = ("generative", "retrieval", "multiturn", "unknown")
LENGTH_ROUTING_KIND_THRESHOLD_DEFAULTS: dict[str, int] = {
    "generative": 4096,
    "retrieval": 8192,
    "multiturn": 4096,
    "unknown": 4096,
}


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """One HTTP backend the router can dispatch to.

    `weight` is the relative share of traffic this backend receives among the
    *enabled* backends of the same tier -- this is the canary knob: two
    backends in the same tier with weights 95/5 is a 5% canary. Setting the
    candidate's `enabled=false` (not just `weight=0`) is the instant-rollback
    kill switch: it also removes the backend from in-tier failover and from
    `/router/status` as a live option, not just from fresh weighted picks.

    `rollback_target=true` marks a backend as the tier's known-good default --
    `config.py`'s validation refuses to load a config where some tier has no
    enabled `rollback_target` backend, because "the pruned model with no
    fallback" is exactly the failure mode this field exists to make
    unrepresentable.
    """

    id: str
    tier: str
    base_url: str
    model_id: str
    weight: int = 100
    enabled: bool = True
    role: str = "primary"
    rollback_target: bool = False
    api_key: str | None = None
    connect_timeout_s: float = 5.0
    idle_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ConfigError(f"backend {self.id!r}: weight must be >= 0")
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(f"backend {self.id!r}: base_url must start with http:// or https://")
        if self.connect_timeout_s <= 0 or self.idle_timeout_s <= 0:
            raise ConfigError(f"backend {self.id!r}: timeouts must be > 0")
        if self.role not in BACKEND_ROLES:
            raise ConfigError(
                f"backend {self.id!r}: role {self.role!r} is not one of {sorted(BACKEND_ROLES)}"
            )


@dataclass(frozen=True, slots=True)
class EscalationConfig:
    """See `policy.py` for what each field controls; `docs/DESIGN.md`'s
    "Escalation policy" section for why these are the shipped defaults."""

    policy: str = "default"
    default_tier: str = "fast"
    escalation_tier: str = "deep"
    hard_markers: tuple[str, ...] = DEFAULT_HARD_MARKERS
    heuristic_threshold: float = 0.6
    enable_task_heuristic: bool = True
    enable_draft_then_escalate: bool = False


@dataclass(frozen=True, slots=True)
class LengthRoutingConfig:
    """Length-aware arm routing within a tier's canary draw -- a guard-rail
    for a drafter-candidate arm whose speedup decays (and, past some prompt
    length, inverts) as the prompt grows, not a general routing feature or
    a promoter. See `backends.py`'s `estimate_prompt_tokens` /
    `length_routing_excluded_ids` for the mechanism, and docs/DESIGN.md's
    "Length-aware routing" section for the measurement that motivated it
    and the full argument for each field below.

    Ships OPT-IN (`enabled=False`, "dark"): a blind weighted canary draw
    cannot see prompt length at all, so it keeps sending a length-sensitive
    drafter exactly where it has already measured a loss -- but the right
    `threshold_tokens` is a deployment-specific tuning decision (it depends
    on which drafter, which backend, and real acceptance-rate-vs-length
    data for that pairing), not a default this router should assume for
    every deployment.

    `threshold_tokens` (default 4096) is PROVISIONAL -- derived from an
    early 7-pair measurement (1.35x speedup at ~4K prompt tokens, decaying
    to 0.98x/1.01x by ~6.1K/7.3K); refine it once the fuller matched-config
    dataset that measurement was collected en route to lands.

    `estimator` names which prompt-length estimate to use against
    `threshold_tokens`; `"chars_div4"` (the only one implemented today,
    see `config.LENGTH_ROUTING_ESTIMATORS`) is deliberately tokenizer-free
    -- see `estimate_prompt_tokens`'s docstring for the accuracy/dependency
    trade-off this makes.
    """

    enabled: bool = False
    threshold_tokens: int = 4096
    estimator: str = "chars_div4"
    # Kind-aware mode refines routing by prompt kind; `kind_aware=false`
    # keeps this feature exactly the current blind behavior so that
    # enabling length routing never changes the decision boundary unless both
    # `enabled` and `kind_aware` are true.
    kind_aware: bool = False
    # Per-kind thresholds are additive and optional; this config accepts
    # partial/legacy dictionaries and fills missing entries from defaults.
    kind_thresholds: dict[str, int] = field(default_factory=lambda: dict(LENGTH_ROUTING_KIND_THRESHOLD_DEFAULTS))

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0:
            raise ConfigError("length_routing.threshold_tokens must be > 0")
        if self.estimator not in LENGTH_ROUTING_ESTIMATORS:
            raise ConfigError(
                f"length_routing.estimator {self.estimator!r} is not one of "
                f"{sorted(LENGTH_ROUTING_ESTIMATORS)}"
            )
        if not isinstance(self.kind_thresholds, Mapping):
            raise ConfigError("length_routing.kind_thresholds must be a mapping")

        unknown_keys = sorted(set(self.kind_thresholds) - set(LENGTH_ROUTING_KIND_NAMES))
        if unknown_keys:
            raise ConfigError(
                f"length_routing.kind_thresholds has unknown keys: {unknown_keys}; "
                f"expected {LENGTH_ROUTING_KIND_NAMES}"
            )

        normalized: dict[str, int] = dict(LENGTH_ROUTING_KIND_THRESHOLD_DEFAULTS)
        for kind, threshold in self.kind_thresholds.items():
            if not isinstance(threshold, int):
                raise ConfigError(
                    f"length_routing.kind_thresholds[{kind!r}] must be an int, got {type(threshold)!r}"
                )
            if threshold <= 0:
                raise ConfigError(
                    f"length_routing.kind_thresholds[{kind!r}] must be > 0, got {threshold}"
                )
            normalized[kind] = threshold
        object.__setattr__(self, "kind_thresholds", normalized)

    def threshold_for_kind(self, kind: str | None) -> int:
        normalized_kind = kind if isinstance(kind, str) and kind in LENGTH_ROUTING_KIND_NAMES else "unknown"
        return self.kind_thresholds.get(normalized_kind, self.threshold_tokens)


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 3
    reset_after_s: float = 60.0


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    api_key: str | None = None
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    log_path: str = "var/decisions.jsonl"
    # Client-visible X-Router-Backend/-Tier/-Canary/-Trigger/-Fallback-From
    # response headers (server.py's `_emit_routing_headers`). OFF by default
    # so a pruned-vs-dense canary A/B stays *blind*: a client that could read
    # its own canary arm off the response could dodge or specifically target
    # the pruned arm, and the same headers hand an attacker an escalation-
    # policy injection-tuning oracle (probe, read X-Router-Trigger, learn
    # exactly what phrasing trips the heuristic). The same attribution
    # always lands in the server-side JSONL decision log (telemetry.py),
    # keyed by X-Router-Request-Id -- which is NOT gated by this flag, since
    # it is only a correlation id, not routing metadata -- so offline
    # outcome-attribution/shadow-eval never depends on this being on. Opt in
    # only for a trusted/debug deployment that needs the header on the wire
    # itself; see the security review this flag closes.
    expose_routing_headers: bool = False


@dataclass(frozen=True, slots=True)
class RouterConfig:
    server: ServerConfig
    escalation: EscalationConfig
    circuit_breaker: CircuitBreakerConfig
    backends: tuple[BackendConfig, ...]
    # tier -> tier to retry in in when every backend in the first tier is
    # unavailable/failed (see dispatch.py's `RequestRouter`). Symmetric
    # fast<->deep by default; `dispatch.py` refuses to hop back to the tier a
    # request started in, so this can never bounce indefinitely even if
    # written as a cycle.
    fallback: dict[str, str] = field(default_factory=dict)
    length_routing: LengthRoutingConfig = field(default_factory=LengthRoutingConfig)

    def backends_for_tier(self, tier: str) -> tuple[BackendConfig, ...]:
        return tuple(b for b in self.backends if b.tier == tier)

    def tiers(self) -> tuple[str, ...]:
        seen: list[str] = []
        for backend in self.backends:
            if backend.tier not in seen:
                seen.append(backend.tier)
        return tuple(seen)


def _load_backend(raw: dict) -> BackendConfig:
    try:
        return BackendConfig(
            id=raw["id"],
            tier=raw["tier"],
            base_url=raw["base_url"],
            model_id=raw["model_id"],
            weight=int(raw.get("weight", 100)),
            enabled=bool(raw.get("enabled", True)),
            role=raw.get("role", "primary"),
            rollback_target=bool(raw.get("rollback_target", False)),
            api_key=(raw.get("api_key") or None),
            connect_timeout_s=float(raw.get("connect_timeout_s", 5.0)),
            idle_timeout_s=float(raw.get("idle_timeout_s", 30.0)),
        )
    except KeyError as error:
        raise ConfigError(f"backend entry missing required field: {error}") from error


def parse_config(raw: dict) -> RouterConfig:
    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=int(server_raw.get("port", 8100)),
        api_key=(server_raw.get("api_key") or None),
        cors_origins=tuple(server_raw.get("cors_origins", DEFAULT_CORS_ORIGINS)),
        log_path=server_raw.get("log_path", "var/decisions.jsonl"),
        expose_routing_headers=bool(server_raw.get("expose_routing_headers", False)),
    )

    escalation_raw = raw.get("escalation", {})
    escalation = EscalationConfig(
        policy=escalation_raw.get("policy", "default"),
        default_tier=escalation_raw.get("default_tier", "fast"),
        escalation_tier=escalation_raw.get("escalation_tier", "deep"),
        hard_markers=tuple(escalation_raw.get("hard_markers", DEFAULT_HARD_MARKERS)),
        heuristic_threshold=float(escalation_raw.get("heuristic_threshold", 0.6)),
        enable_task_heuristic=bool(escalation_raw.get("enable_task_heuristic", True)),
        enable_draft_then_escalate=bool(escalation_raw.get("enable_draft_then_escalate", False)),
    )

    cb_raw = raw.get("circuit_breaker", {})
    circuit_breaker = CircuitBreakerConfig(
        failure_threshold=int(cb_raw.get("failure_threshold", 3)),
        reset_after_s=float(cb_raw.get("reset_after_s", 60.0)),
    )

    length_routing_raw = raw.get("length_routing", {})
    kind_thresholds_raw = length_routing_raw.get("kind_thresholds", {})
    length_routing = LengthRoutingConfig(
        enabled=bool(length_routing_raw.get("enabled", False)),
        threshold_tokens=int(length_routing_raw.get("threshold_tokens", 4096)),
        estimator=length_routing_raw.get("estimator", "chars_div4"),
        kind_aware=bool(length_routing_raw.get("kind_aware", False)),
        kind_thresholds=dict(kind_thresholds_raw) if isinstance(kind_thresholds_raw, Mapping) else kind_thresholds_raw,
    )

    backends = tuple(_load_backend(entry) for entry in raw.get("backends", []))
    fallback = dict(raw.get("fallback", {}))

    config = RouterConfig(
        server=server,
        escalation=escalation,
        circuit_breaker=circuit_breaker,
        backends=backends,
        fallback=fallback,
        length_routing=length_routing,
    )
    _validate(config)
    return config


def _validate(config: RouterConfig) -> None:
    if not config.backends:
        raise ConfigError("config must declare at least one [[backends]] entry")

    ids = [backend.id for backend in config.backends]
    if len(ids) != len(set(ids)):
        raise ConfigError("backend ids must be unique")

    tiers = set(config.tiers())
    for required in (config.escalation.default_tier, config.escalation.escalation_tier):
        if required not in tiers:
            raise ConfigError(
                f"escalation references tier {required!r} but no backend declares it"
            )

    for tier in tiers:
        rollback_targets = [
            backend
            for backend in config.backends_for_tier(tier)
            if backend.rollback_target and backend.enabled
        ]
        if not rollback_targets:
            raise ConfigError(
                f"tier {tier!r} has no enabled backend marked rollback_target=true "
                "(every tier needs a known-good default to fall back to -- see "
                "docs/DESIGN.md's guardrails section)"
            )

    if config.escalation.policy not in ("default", *tiers):
        raise ConfigError(
            f"escalation.policy {config.escalation.policy!r} is not 'default' or a known tier"
        )


def load_config(path: str | Path) -> RouterConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_config(raw)


def with_server_overrides(config: RouterConfig, *, host: str | None, port: int | None) -> RouterConfig:
    """Returns a copy of `config` with CLI-supplied `--host`/`--port`
    overriding the file's [server] values, if given."""
    if host is None and port is None:
        return config
    server = replace(
        config.server,
        host=host if host is not None else config.server.host,
        port=port if port is not None else config.server.port,
    )
    return replace(config, server=server)
