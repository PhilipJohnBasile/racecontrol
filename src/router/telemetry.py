"""Structured, append-only JSONL logging of every routing decision -- the
input to shadow-eval ("did the router route correctly?", per
docs/DESIGN.md's guardrails section). One JSON object per line, matching this
project family's own convention for raw evidence (trailbrake's
`bench/results/*/records.jsonl`, iliria's `validation/epoch-*` artifacts).

This module only records what the router itself can observe (which tier,
which backend, why, how long, did it succeed). Whether the *decision itself*
was the right one is an offline question -- a later shadow-eval pass reads
this file, joins it against task outcomes, and fills in `extra` fields such
as `verifier_result`; this module does not grade its own decisions.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_request_id() -> str:
    return "rtr_" + uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DecisionRecord:
    request_id: str
    tier: str
    backend_id: str | None
    trigger: str
    reason: str
    canary: bool
    fallback_from: str | None
    status: str  # "ok" | "backend_error" | "no_backend_available" | ...
    http_status: int | None
    latency_s: float
    created_utc: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "created_utc": self.created_utc,
            "tier": self.tier,
            "backend_id": self.backend_id,
            "trigger": self.trigger,
            "reason": self.reason,
            "canary": self.canary,
            "fallback_from": self.fallback_from,
            "status": self.status,
            "http_status": self.http_status,
            "latency_s": round(self.latency_s, 4),
        }
        payload.update(self.extra)
        return payload


class DecisionLogger:
    """Thread-safe JSONL appender, plus an in-memory rolling counter used by
    `GET /router/status` (a cheap live dashboard -- allowed to reset on
    restart; the file is the durable record shadow-eval reads)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def log(self, record: DecisionRecord) -> None:
        line = json.dumps(record.to_json(), separators=(",", ":"))
        key = f"{record.tier}:{record.backend_id}:{record.status}"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._counts[key] = self._counts.get(key, 0) + 1

    def counts_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)
