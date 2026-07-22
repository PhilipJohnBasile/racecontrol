from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from router.telemetry import DecisionLogger, DecisionRecord, new_request_id, utc_now_iso


def _record(**overrides) -> DecisionRecord:
    defaults = dict(
        request_id="rtr_test",
        tier="fast",
        backend_id="trailbrake-baseline",
        trigger="default",
        reason="no escalation signal matched",
        canary=False,
        fallback_from=None,
        status="ok",
        http_status=200,
        latency_s=0.1234567,
        created_utc="2026-07-19T00:00:00Z",
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


class HelperTests(unittest.TestCase):
    def test_request_ids_are_unique_and_prefixed(self):
        first, second = new_request_id(), new_request_id()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("rtr_"))

    def test_utc_now_iso_shape(self):
        stamp = utc_now_iso()
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class DecisionRecordTests(unittest.TestCase):
    def test_to_json_rounds_latency_and_includes_core_fields(self):
        record = _record()
        payload = record.to_json()
        self.assertEqual(payload["latency_s"], 0.1235)
        self.assertEqual(payload["tier"], "fast")
        self.assertEqual(payload["backend_id"], "trailbrake-baseline")
        self.assertEqual(payload["status"], "ok")

    def test_extra_fields_are_merged_in(self):
        record = _record(extra={"verifier_result": "pass"})
        payload = record.to_json()
        self.assertEqual(payload["verifier_result"], "pass")


class DecisionLoggerTests(unittest.TestCase):
    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "decisions.jsonl"
            DecisionLogger(path)
            self.assertTrue(path.parent.is_dir())

    def test_log_appends_one_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            logger = DecisionLogger(path)
            logger.log(_record())
            logger.log(_record(status="backend_error"))

            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["status"], "ok")
            second = json.loads(lines[1])
            self.assertEqual(second["status"], "backend_error")

    def test_log_is_append_only_across_logger_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            DecisionLogger(path).log(_record())
            DecisionLogger(path).log(_record())
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_counts_snapshot_tracks_tier_backend_status_triples(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(Path(tmp) / "decisions.jsonl")
            logger.log(_record(tier="fast", backend_id="trailbrake-baseline", status="ok"))
            logger.log(_record(tier="fast", backend_id="trailbrake-baseline", status="ok"))
            logger.log(_record(tier="deep", backend_id="iliria", status="backend_error"))

            counts = logger.counts_snapshot()
            self.assertEqual(counts["fast:trailbrake-baseline:ok"], 2)
            self.assertEqual(counts["deep:iliria:backend_error"], 1)

    def test_counts_snapshot_is_a_copy_not_live_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(Path(tmp) / "decisions.jsonl")
            logger.log(_record())
            snapshot = logger.counts_snapshot()
            logger.log(_record())
            self.assertNotEqual(snapshot, logger.counts_snapshot())

    def test_concurrent_logging_does_not_lose_or_interleave_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            logger = DecisionLogger(path)

            def _write_many():
                for _ in range(50):
                    logger.log(_record())

            threads = [threading.Thread(target=_write_many) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 200)
            for line in lines:
                json.loads(line)  # every line must be independently valid JSON


if __name__ == "__main__":
    unittest.main()
