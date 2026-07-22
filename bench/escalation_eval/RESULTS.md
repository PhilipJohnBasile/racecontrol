# Escalation-heuristic evaluation — 2026-07-22

Why the default `heuristic_threshold` is 0.7, and why two proposed vocabulary
extensions were rejected. Reproduce with `python3 bench/escalation_eval/harness.py`.

## Method

84 prompts (42 hard / 42 easy) written **blind** to the pattern table — the
authors were forbidden from reading `policy.py`, so the set measures
generalization rather than memorization. Three subsets: 30 straightforwardly
hard, 30 easy (seeded with precision traps: trivial requests that *mention*
scary vocabulary), 24 adversarial near-misses in both directions. Ground rule
from `docs/DESIGN.md` §4: a false escalation costs ~100× a miss (deep tier
measured 0.2 tok/s under contention vs ~23 on the fast tier).

A positive control ran before every measurement: four prompts known to hit
the floor patterns must score exactly 0.600. It passed in every run cited
here, independently re-verified in a fresh interpreter with no monkeypatching.

## Result: the shipped table at the old 0.6 threshold

| | escalated | stayed fast |
|---|---|---|
| **hard (42)** | 0 | 42 |
| **easy (42)** | 3 | 39 |

**Precision 0.000, recall 0.000.** Every escalation it produced was a false
positive, each a planted precision trap — e.g. *"Add a unit test for the race
condition we already fixed"*, where the 0.6 floor overrode the `-0.2`
"write a test for" easy signal that exists to catch exactly that shape.
Hard-prompt score distribution: 38× 0.00, 4× 0.30 — nothing approached 0.6.

## The two proposed extensions (both rejected)

Two independent pattern proposals (one coverage-focused, one written as a
precision hawk) were measured on this set. Self-reported coverage did not
transfer: the coverage proposal claimed 72/72 on its own dev prompts and
measured 12/42 (28.6%) here; the precision proposal claimed 28/28 and
measured 2/42 (4.8%).

| variant | tp | fp | 100×-cost (per request) | verdict |
|---|---|---|---|---|
| baseline @0.6 | 0 | 3 | 342 | reference |
| + coverage | 12 | 8 | 830 | rejected |
| + precision | 2 | 6 | 640 | rejected |
| + both (union) | 11 | 8 | 831 | strictly dominated |

An adversarial re-measurement reproduced every cell exactly and found a
failure the single-turn metrics hide: both proposals assign weight 0.6 to
many patterns, so two firing on one prompt sum to 1.2, clamp to 1.0, and
survive two rounds of the 0.8/turn decay — **a 3-turn escalation latch**
(1.0 → 0.80 → 0.64, all ≥ 0.6). Counted in wasted deep-tier *turns*:
baseline 3, +coverage 11, union 16. The baseline never latches. Neither
proposal removed a single pre-existing false positive.

## Result: threshold 0.7 strictly dominates

Raising the threshold to 0.7 removes all three false positives at **zero**
recall cost (baseline catches nothing at either threshold). On this set the
heuristic at 0.7 is inert — which is the honest description of the shipped
default: **escalation in practice comes from manual overrides and explicit
markers** (`#deep` / `#reason` / `reasoning_effort=high`), which is
consistent with production telemetry (110 logged decisions, 0 heuristic
escalations). A bare floored pattern (0.600) cannot clear 0.7; only
multi-pattern prompts can.

## Caveats — read before quoting these numbers

- The prompts are LLM-authored. They skew toward natural engineer prose and
  contain fewer "why does X fail" constructions than the patterns expect;
  real traffic may look different. This bounds the claim to "the vocabulary
  generalizes poorly", not "the production miss rate is exactly 100%".
- n=84 with 20 positives across variants: per-variant precision estimates
  (e.g. "0.600") are unstable — do not quote them as stable properties.
- The 50/50 hard/easy split is not production traffic. Real traffic skews
  easy, which makes false-positive cost *worse* in deployment, not better.
- Any new pattern set tuned against this file needs a **fresh** holdout —
  the two rejected proposals are the demonstration of why (both aced the
  prompts they were tuned on and failed here).
