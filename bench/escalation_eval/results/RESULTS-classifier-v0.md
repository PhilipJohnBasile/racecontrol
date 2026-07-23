# Experimental classifier v0 (synthetic-trained) — one-shot result, 2026-07-23

**Status: EXPERIMENT. Not a routing policy. Its held-out shot is SPENT.**

Pipeline proven end-to-end: `label_decisions.py` (1 real label existed) →
`train_classifier.py` (+400 fresh seeded synthetic examples; the held-out
set was never read during generation) → `shadow_eval.py` one-shot against
the committed 84-case blind set.

| config | tp | fp | fn | precision | recall | 100×-cost |
|---|---|---|---|---|---|---|
| regex @0.6 (old default) | 0 | 3 | 42 | 0.000 | 0.000 | 342 |
| regex @0.7 (shipped) | 0 | 0 | 42 | — | 0.000 | 42 |
| **LR v0 synthetic @0.5** | **42** | **32** | **0** | **0.568** | **1.000** | **3200** |

Reading: the distribution-shift trap is now a measurement, not an argument.
Synthetic training buys perfect recall instantly (dominant coefficient:
log-prompt-length, 3.425 — it learned "long = hard") and pays for it with
32/42 easy prompts escalated at ~100× cost — ~76× worse than the shipped
inert config. **Recall is free; precision is the scarce good, and only real
labeled traffic can buy it.** Which is precisely why the telemetry channel,
the fences, and the labeler exist.

Rules going forward:
- This artifact's one held-out evaluation is used. Any retrained model that
  was shaped by looking at this table needs a FRESH holdout.
- Retrain trigger: meaningful real-label count from `label_decisions.py`
  (weak negatives downweighted 0.25). Real labels enter via `--labels`;
  synthetic becomes scaffolding to discard, not signal to keep.
- Nothing here touches routing. Shadow-replay disagreements
  (`shadow_eval.py --decisions`) are for eyeballing, not enforcement.
