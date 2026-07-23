#!/usr/bin/env python3
"""EXPERIMENTAL escalation classifier -- trainer. Offline only, zero policy
impact: nothing in src/router imports this, and no trained artifact affects
routing. It exists so the moment real labels accumulate (label_decisions.py),
retraining is one command instead of a project.

Model: logistic regression, pure stdlib, five features from the decision
log's groundtruth contract:

    [hardness_score, n_hard_hits, n_easy_hits, log1p(latest_chars), user_turns]

`had_marker` is deliberately NOT a feature -- for marker-labeled rows it IS
the label, and training on it would be leakage dressed as accuracy.

Data sources, preferred first:
  --labels FILE     real labeled examples from label_decisions.py
                    (weak negatives downweighted by --weak-weight).
  --synthetic N     a FRESH, seeded, template-generated stopgap set. It is
                    generated here, from templates written for this file --
                    it never reads bench/escalation_eval/eval-*.jsonl, so
                    the committed held-out set stays a valid test. A model
                    trained only on synthetic data is distribution-shift
                    bait (two regex proposals aced their own dev sets and
                    failed the blind set); treat any such model as an
                    experiment, never a routing policy.

Output: coefficients as JSON (--out), consumed by shadow_eval.py.
"""
import argparse
import json
import math
import random

FEATURE_NAMES = ("hardness_score", "n_hard_hits", "n_easy_hits", "log1p_chars", "user_turns")

# ---------------------------------------------------------------------------
# Fresh synthetic stopgap set. Templates authored for this file; the held-out
# eval set was not consulted. Topical overlap with real traffic is the point;
# copied prompts would be the sin.
_HARD_TEMPLATES = [
    "our {svc} worker wedges after ~{n} hours and only under load, no crash log, cpu pinned",
    "getting different results from the same query when {n} clients hit it concurrently, walk me through how to corner this",
    "the {svc} latency histogram went bimodal after the last deploy and p99 tripled, what would you rule out first",
    "why would this allocator return the same block twice under pressure? here's the free-list logic",
    "prove whether this retry scheme can livelock when both sides jitter with the same seed",
    "the checksum mismatches only on arm64 release builds, never debug, never x86 -- where do I even start",
    "derive the closed form for the expected number of probes at load factor {f} and check it against these measurements",
    "our consensus layer commits out of order roughly once per {n}k rounds, reconstruct the interleaving that allows it",
    "this lock-free queue drops an element when producer and consumer wrap simultaneously, find the window",
    "explain why increasing worker count from {n} to {m} made throughput worse, given this contention profile",
]
_EASY_TEMPLATES = [
    "rename {svc}Handler to {svc}Controller across the repo",
    "write a docstring for this two-line helper",
    "what does {svc} stand for?",
    "bump the version to {n}.{m}.0 and update the changelog heading",
    "convert this list to a markdown table",
    "add a getter for the {svc} field",
    "fix the typo in the {svc} readme",
    "hi -- quick one, what timezone does the ci run in?",
    "format this json so I can read it",
    "give me a one-line summary of what {svc} does",
]
_SLOTS = {"svc": ["billing", "ingest", "auth", "scheduler", "search", "export"],
          "n": [2, 3, 6, 8, 12, 24], "m": [4, 9, 16, 32], "f": [0.7, 0.85, 0.9]}


def synthetic_examples(count: int, seed: int, feature_fn) -> list[tuple[list[float], int, float]]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        hard = i % 2 == 0
        template = rng.choice(_HARD_TEMPLATES if hard else _EASY_TEMPLATES)
        prompt = template.format(**{k: rng.choice(v) for k, v in _SLOTS.items()})
        out.append((feature_fn(prompt), 1 if hard else 0, 1.0))
    return out


# ---------------------------------------------------------------------------
def features_from_prompt(prompt: str) -> list[float]:
    """Feature vector via the router's own escalation_features -- the same
    function that stamps the decision log, so offline training and any
    future shadow use see identical inputs."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from router.policy import escalation_features
    raw = escalation_features([{"role": "user", "content": prompt}])
    return vector_from_logged(raw)


def vector_from_logged(row: dict) -> list[float]:
    return [
        float(row.get("hardness_score", 0.0)),
        float(len(row.get("hard_hits", []))),
        float(len(row.get("easy_hits", []))),
        math.log1p(float(row.get("latest_chars", 0))),
        float(row.get("user_turns", 1)),
    ]


def train(examples: list[tuple[list[float], int, float]], epochs: int = 800,
          lr: float = 0.05, l2: float = 1e-3) -> dict:
    """Weighted logistic regression by plain gradient descent. Returns the
    model as a plain dict so the artifact is diff-able JSON, not a pickle."""
    dim = len(FEATURE_NAMES)
    # standardize features; store the scaling in the artifact
    means = [sum(x[i] for x, _, _ in examples) / len(examples) for i in range(dim)]
    stds = []
    for i in range(dim):
        var = sum((x[i] - means[i]) ** 2 for x, _, _ in examples) / len(examples)
        stds.append(math.sqrt(var) or 1.0)
    scaled = [([(x[i] - means[i]) / stds[i] for i in range(dim)], y, w) for x, y, w in examples]

    weights = [0.0] * dim
    bias = 0.0
    total_w = sum(w for _, _, w in scaled)
    for _ in range(epochs):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y, w in scaled:
            z = bias + sum(wi * xi for wi, xi in zip(weights, x))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            err = w * (p - y)
            for i in range(dim):
                grad_w[i] += err * x[i]
            grad_b += err
        for i in range(dim):
            weights[i] = weights[i] - lr * (grad_w[i] / total_w + l2 * weights[i])
        bias -= lr * grad_b / total_w
    return {"feature_names": list(FEATURE_NAMES), "weights": weights, "bias": bias,
            "means": means, "stds": stds}


def predict(model: dict, vector: list[float]) -> float:
    scaled = [(vector[i] - model["means"][i]) / model["stds"][i] for i in range(len(vector))]
    z = model["bias"] + sum(w * x for w, x in zip(model["weights"], scaled))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=None, help="labeled JSONL from label_decisions.py")
    parser.add_argument("--synthetic", type=int, default=0, help="ALSO/instead generate N fresh synthetic examples")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--weak-weight", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    examples: list[tuple[list[float], int, float]] = []
    n_real = 0
    if args.labels:
        for line in open(args.labels):
            if not line.strip():
                continue
            row = json.loads(line)
            weight = args.weak_weight if row.get("weak") else 1.0
            examples.append((vector_from_logged(row["features"]), 1 if row["wanted_deep"] else 0, weight))
        n_real = len(examples)
    if args.synthetic:
        examples += synthetic_examples(args.synthetic, args.seed, features_from_prompt)
    if len(examples) < 20:
        raise SystemExit(f"refusing to train on {len(examples)} examples -- this would be noise wearing a model's name")

    model = train(examples)
    model["trained_on"] = {"real_labeled": n_real, "synthetic": len(examples) - n_real,
                           "seed": args.seed, "EXPERIMENTAL": True,
                           "note": "synthetic-trained models are distribution-shift bait; never a routing policy"}
    with open(args.out, "w") as fh:
        json.dump(model, fh, indent=1)
    print(f"trained on {n_real} real + {len(examples) - n_real} synthetic -> {args.out}")
    print("coefficients:", {n: round(w, 3) for n, w in zip(FEATURE_NAMES, model["weights"])})


if __name__ == "__main__":
    main()
