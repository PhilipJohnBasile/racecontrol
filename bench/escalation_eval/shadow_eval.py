#!/usr/bin/env python3
"""Shadow-evaluate a trained classifier against logged decisions or the
held-out eval set. OFFLINE ONLY -- this replays history; it never routes.

Two modes:

  --decisions FILE   replay a decision log: for every feature-carrying row,
                     report what the classifier WOULD have decided vs what
                     the router actually did. Disagreements are the
                     interesting rows -- they are candidate misses (or
                     candidate false escalations) to eyeball.

  --eval-set DIR     one-shot evaluation against the committed held-out set
                     (eval-*.jsonl). This is legitimate exactly once per
                     trained artifact: evaluating a finished model against
                     a held-out set is what the set is FOR. Iterating on
                     the model because you didn't like this number burns
                     the set -- at that point it is training data, and the
                     next model needs a fresh holdout (RESULTS.md).
"""
import argparse
import glob
import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "train_classifier", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_classifier.py"))
_tc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_tc)


def load_model(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def predict_row(model: dict, logged_row: dict) -> float:
    return _tc.predict(model, _tc.vector_from_logged(logged_row))


def predict_prompt(model: dict, prompt: str) -> float:
    return _tc.predict(model, _tc.features_from_prompt(prompt))


def replay_decisions(model: dict, path: str, threshold: float) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    scored = 0
    agree = 0
    disagreements = []
    for row in rows:
        if "prompt_fingerprint" not in row:
            continue
        scored += 1
        p = predict_row(model, row)
        would_deep = p >= threshold
        was_deep = row.get("tier") != "fast"
        if would_deep == was_deep:
            agree += 1
        else:
            disagreements.append((p, row))
    print(f"{scored} feature-carrying rows; classifier agrees with actual routing on {agree} "
          f"({agree / scored:.0%})" if scored else "no feature-carrying rows yet")
    for p, row in disagreements[:10]:
        print(f"  would={'deep' if p >= threshold else 'fast'} p={p:.2f}  actual={row.get('tier')}"
              f"  trigger={row.get('trigger')}  fp={row.get('prompt_fingerprint')}")


def eval_heldout(model: dict, directory: str, threshold: float) -> None:
    cases = []
    for path in sorted(glob.glob(os.path.join(directory, "eval-*.jsonl"))):
        with open(path) as fh:
            cases += [json.loads(line) for line in fh if line.strip()]
    tp = fp = tn = fn = 0
    for case in cases:
        escalate = predict_prompt(model, case["prompt"]) >= threshold
        if case["label"] == "hard":
            tp, fn = tp + escalate, fn + (not escalate)
        else:
            fp, tn = fp + escalate, tn + (not escalate)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"held-out one-shot @ threshold {threshold}: tp={tp} fp={fp} fn={fn} tn={tn}  "
          f"precision={precision:.3f} recall={recall:.3f}")
    print(f"100x-weighted cost (fp*100 + fn): {fp * 100 + fn}   [regex baseline @0.6: 342; @0.7: 42]")
    print("Reminder: one shot per artifact. Iterate on this number and the set is burned.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--eval-set", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    model = load_model(args.model)
    if args.decisions:
        replay_decisions(model, args.decisions, args.threshold)
    if args.eval_set:
        eval_heldout(model, args.eval_set, args.threshold)
    if not args.decisions and not args.eval_set:
        raise SystemExit("nothing to do: pass --decisions and/or --eval-set")


if __name__ == "__main__":
    main()
