#!/usr/bin/env python3
"""Score the escalation heuristic against the held-out eval set.

The eval set (eval-*.jsonl in this directory) was written BLIND to the
pattern table in `router.policy` -- the builders were forbidden from reading
it -- so these numbers measure generalization, not memorization. Re-run this
after ANY change to `_HARD_SIGNAL_PATTERNS` / `_EASY_SIGNAL_PATTERNS` or the
threshold; a change that looks good here still needs a FRESH holdout before
shipping if it was tuned against this set (tuning against your own test set
is how the two rejected proposals overfit -- see RESULTS.md).

Usage:  python3 bench/escalation_eval/harness.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from router.policy import hardness_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Positive control: prompts known to hit the two floor patterns must score
# exactly 0.600 -- if they don't, the instrument is broken and every number
# below is meaningless. (This control caught nothing being wrong the day the
# set was built; it exists so a future refactor can't silently blind us.)
FLOOR_PROMPTS = [
    "Why does this test fail intermittently?",
    "There is a deadlock in the connection pool.",
    "This looks like memory corruption.",
    "prove it formally, step by step",
]


def load_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(HERE, "eval-*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def main():
    for prompt in FLOOR_PROMPTS:
        score = hardness_score([{"role": "user", "content": prompt}])
        assert abs(score - 0.600) < 1e-9, (
            f"POSITIVE CONTROL FAILED: {prompt!r} scored {score:.3f}, expected 0.600. "
            "The instrument is broken; do not trust any number this harness prints."
        )
    print("positive control: 4/4 floor prompts score 0.600 exactly\n")

    cases = load_cases()
    scored = [
        (hardness_score([{"role": "user", "content": c["prompt"]}]), c["label"], c["prompt"])
        for c in cases
    ]
    n_hard = sum(1 for _, label, _ in scored if label == "hard")
    print(f"{len(cases)} cases ({n_hard} hard / {len(cases) - n_hard} easy)\n")
    print(f"{'thresh':>6}  {'tp':>3} {'fp':>3} {'fn':>3} {'tn':>3}   note")
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(1 for s, label, _ in scored if label == "hard" and s >= threshold)
        fp = sum(1 for s, label, _ in scored if label == "easy" and s >= threshold)
        fn = n_hard - tp
        tn = len(cases) - n_hard - fp
        note = ""
        if threshold == 0.6:
            note = "old default -- every escalation false"
        elif threshold == 0.7:
            note = "current default -- inert on this set"
        print(f"{threshold:>6}  {tp:>3} {fp:>3} {fn:>3} {tn:>3}   {note}")

    print("\nfalse positives at 0.6 (each costs ~100x a miss):")
    for s, label, prompt in scored:
        if label == "easy" and s >= 0.6:
            print(f"  {s:.2f}  {prompt[:76]}")


if __name__ == "__main__":
    main()
