"""The offline labeler (bench/escalation_eval/label_decisions.py) implements
docs/DESIGN.md's groundtruth labeling rules; these tests pin each rule on
synthetic rows so a rule change is a deliberate act, not drift."""
import importlib.util
import os
import sys
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "label_decisions",
    os.path.join(os.path.dirname(__file__), "..", "bench", "escalation_eval", "label_decisions.py"),
)
label_decisions = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(label_decisions)


def _row(ts, tier="fast", trigger="default", status="ok", fp="aaaa", marker=False,
         canary=False, features=True, rid="r"):
    row = {"request_id": rid, "created_utc": ts, "tier": tier, "trigger": trigger,
           "status": status, "canary": canary}
    if features:
        row.update({"hardness_score": 0.0, "hard_hits": [], "easy_hits": [],
                    "patterns_version": "deadbeef", "prompt_fingerprint": fp,
                    "had_marker": marker, "latest_chars": 10, "user_turns": 1})
    return row


class LabelRulesTests(unittest.TestCase):
    def test_reask_with_marker_labels_the_earlier_fast_row_a_miss(self):
        rows = [
            _row("2026-07-22T10:00:00Z", fp="abcd", rid="first"),
            _row("2026-07-22T10:05:00Z", fp="abcd", tier="deep",
                 trigger="explicit_marker", marker=True, rid="reask"),
        ]
        labeled, counts = label_decisions.label(rows, [], reask_window_s=1800)
        by_id = {e["request_id"]: e for e in labeled}
        self.assertEqual(by_id["first"]["label_rule"], "re_ask_miss")
        self.assertTrue(by_id["first"]["wanted_deep"])
        self.assertEqual(by_id["reask"]["label_rule"], "caller_said_deep")
        self.assertEqual(counts["re_ask_miss"], 1)

    def test_reask_outside_window_is_a_weak_negative(self):
        rows = [
            _row("2026-07-22T10:00:00Z", fp="abcd", rid="first"),
            _row("2026-07-22T12:00:00Z", fp="abcd", tier="deep",
                 trigger="explicit_marker", marker=True, rid="late"),
        ]
        labeled, _ = label_decisions.label(rows, [], reask_window_s=1800)
        by_id = {e["request_id"]: e for e in labeled}
        self.assertEqual(by_id["first"]["label_rule"], "fast_enough_weak")
        self.assertTrue(by_id["first"]["weak"])

    def test_plain_repeat_without_marker_is_not_a_miss(self):
        rows = [
            _row("2026-07-22T10:00:00Z", fp="abcd", rid="first"),
            _row("2026-07-22T10:02:00Z", fp="abcd", rid="retry"),
        ]
        labeled, counts = label_decisions.label(rows, [], reask_window_s=1800)
        self.assertEqual(counts["re_ask_miss"], 0)
        self.assertEqual(counts["fast_enough_weak"], 2)

    def test_exclusions_canary_fence_features_status(self):
        rows = [
            _row("2026-07-22T10:00:00Z", canary=True, rid="canary"),
            _row("2026-07-22T10:01:00Z", features=False, rid="prefeature"),
            _row("2026-07-22T10:02:00Z", status="backend_error", rid="err"),
            _row("2026-07-22T11:30:00Z", rid="fenced"),
            _row("2026-07-22T13:00:00Z", rid="kept"),
        ]
        fences = [{"from": "2026-07-22T11:00:00Z", "to": "2026-07-22T12:00:00Z"}]
        labeled, counts = label_decisions.label(rows, fences, reask_window_s=1800)
        self.assertEqual([e["request_id"] for e in labeled], ["kept"])
        self.assertEqual(counts["excluded_canary"], 1)
        self.assertEqual(counts["excluded_no_features"], 1)
        self.assertEqual(counts["excluded_not_ok"], 1)
        self.assertEqual(counts["excluded_fenced"], 1)

    def test_open_fence_excludes_everything_after_start(self):
        rows = [_row("2026-07-22T23:59:00Z", rid="inside_open_fence")]
        fences = [{"from": "2026-07-22T23:00:00Z", "to": None}]
        _, counts = label_decisions.label(rows, fences, reask_window_s=1800)
        self.assertEqual(counts["excluded_fenced"], 1)


if __name__ == "__main__":
    unittest.main()
