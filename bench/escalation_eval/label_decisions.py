#!/usr/bin/env python3
"""Offline labeler for the decision-groundtruth contract.

Implements, in code, exactly the labeling rules documented in
docs/DESIGN.md ("The decision-groundtruth contract") -- this script is the
"later shadow-eval pass" telemetry.py's docstring promises. It reads the
decision log (and the optional fence file), applies the rules, and emits
one labeled example per eligible decision. It never grades itself: every
label is derived from what the CALLER did next, not from any model's
opinion of the answer.

Rules (docs/DESIGN.md is the spec; keep them in sync):
  re_ask_miss     -- a fast-tier, default-trigger, ok row whose
                     marker-stripped fingerprint reappears within
                     --reask-window minutes carrying had_marker=true:
                     the caller re-asked with an escalation marker, so the
                     fast answer wasn't good enough. Strongest label.
  caller_said_deep / caller_said_fast
                  -- manual_override and explicit_marker rows: the caller
                     stated the correct tier outright.
  fast_enough_weak -- a fast, default, ok row whose fingerprint never
                     recurs with a marker. WEAK: silence also means the
                     caller gave up; downweight accordingly.

Excluded rows (never labeled): canary=true (synthetic traffic, per the
contract), rows inside a fence window (degraded-stack traffic), rows
without feature fields (pre-contract era), and rows whose status is not
"ok" (no answer was delivered, so the caller's next move says nothing
about answer quality).

Usage:
  python3 bench/escalation_eval/label_decisions.py \
      --decisions var/decisions.jsonl \
      [--fences var/telemetry-fences.jsonl] \
      [--reask-window-minutes 30] \
      [--out labeled.jsonl]
"""
import argparse
import json
from datetime import datetime, timezone

FEATURE_FIELDS = ("hardness_score", "prompt_fingerprint", "had_marker")


def _parse_utc(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def in_any_fence(ts: float, fences: list[dict]) -> bool:
    for fence in fences:
        start = _parse_utc(fence["from"])
        end = _parse_utc(fence["to"]) if fence.get("to") else float("inf")
        if start <= ts <= end:
            return True
    return False


def label(rows: list[dict], fences: list[dict], reask_window_s: float) -> tuple[list[dict], dict]:
    """Pure function so the tests can feed synthetic rows. Returns
    (labeled_examples, summary_counts)."""
    counts = {"total": len(rows), "excluded_canary": 0, "excluded_fenced": 0,
              "excluded_no_features": 0, "excluded_not_ok": 0,
              "re_ask_miss": 0, "caller_said_deep": 0, "caller_said_fast": 0,
              "fast_enough_weak": 0, "unlabeled": 0}
    eligible: list[tuple[float, dict]] = []
    for row in rows:
        if row.get("canary"):
            counts["excluded_canary"] += 1
            continue
        if any(field not in row for field in FEATURE_FIELDS):
            counts["excluded_no_features"] += 1
            continue
        ts = _parse_utc(row["created_utc"])
        if in_any_fence(ts, fences):
            counts["excluded_fenced"] += 1
            continue
        if row.get("status") != "ok":
            counts["excluded_not_ok"] += 1
            continue
        eligible.append((ts, row))

    eligible.sort(key=lambda pair: pair[0])
    # Marker re-asks indexed by fingerprint for the join. Only marker
    # re-asks count -- a plain repeat of the same prompt is retry noise,
    # not a stated escalation.
    marker_times: dict[str, list[float]] = {}
    for ts, row in eligible:
        if row["had_marker"]:
            marker_times.setdefault(row["prompt_fingerprint"], []).append(ts)

    out = []
    for ts, row in eligible:
        trigger = row.get("trigger")
        if trigger in ("manual_override", "explicit_marker"):
            rule = "caller_said_deep" if row.get("tier") != "fast" else "caller_said_fast"
        elif row.get("tier") == "fast" and trigger == "default":
            reasks = marker_times.get(row["prompt_fingerprint"], [])
            if any(0 < later - ts <= reask_window_s for later in reasks):
                rule = "re_ask_miss"
            else:
                rule = "fast_enough_weak"
        else:
            counts["unlabeled"] += 1
            continue
        counts[rule] += 1
        out.append({
            "request_id": row.get("request_id"),
            "created_utc": row["created_utc"],
            "label_rule": rule,
            # the label a classifier would train on: was deep the right tier?
            "wanted_deep": rule in ("re_ask_miss", "caller_said_deep"),
            "weak": rule == "fast_enough_weak",
            "features": {key: row.get(key) for key in
                         ("hardness_score", "hard_hits", "easy_hits", "patterns_version",
                          "prompt_fingerprint", "had_marker", "latest_chars", "user_turns")},
        })
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--fences", default=None)
    parser.add_argument("--reask-window-minutes", type=float, default=30.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.decisions)
    fences = load_jsonl(args.fences) if args.fences else []
    labeled, counts = label(rows, fences, args.reask_window_minutes * 60.0)
    if args.out:
        with open(args.out, "w") as fh:
            for example in labeled:
                fh.write(json.dumps(example) + "\n")
    for key, value in counts.items():
        print(f"{key:22s} {value}")
    if args.out:
        print(f"\n{len(labeled)} labeled example(s) -> {args.out}")


if __name__ == "__main__":
    main()
