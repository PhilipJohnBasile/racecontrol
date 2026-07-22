"""Tests for the config hot-reload path -- the human-triggered half of
"instant rollback" (the other half, the automatic circuit breaker, is
covered in tests/test_dispatch.py's CircuitBreakerIntegrationTests).

`install_sighup_reload` is exercised directly (calling the registered
handler function, and separately via a real OS signal) rather than only
through the blocking `serve()` entry point, precisely so this does not need
a live server loop running in the foreground to verify.
"""

from __future__ import annotations

import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

from router.server import build_server, install_sighup_reload, reload_from_path

_BASE_TOML = """
[server]
log_path = "{log_path}"

[escalation]
default_tier = "fast"
escalation_tier = "deep"

[[backends]]
id = "trailbrake-baseline"
tier = "fast"
base_url = "http://127.0.0.1:8080"
model_id = "default"
rollback_target = true
weight = {baseline_weight}

[[backends]]
id = "trailbrake-candidate"
tier = "fast"
base_url = "http://127.0.0.1:8081"
model_id = "default"
weight = {candidate_weight}
enabled = {candidate_enabled}
role = "candidate"

[[backends]]
id = "iliria"
tier = "deep"
base_url = "http://127.0.0.1:8000"
model_id = "glm-5.2-iliria"
rollback_target = true
"""


def _write_config(path: Path, *, baseline_weight=100, candidate_weight=0, candidate_enabled="false", log_path=None):
    path.write_text(
        _BASE_TOML.format(
            baseline_weight=baseline_weight,
            candidate_weight=candidate_weight,
            candidate_enabled=candidate_enabled,
            log_path=log_path or str(path.parent / "decisions.jsonl"),
        )
    )


class ReloadFromPathTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tempdir.name) / "router.toml"
        _write_config(self.config_path, candidate_weight=0, candidate_enabled="false")
        from router.config import load_config

        self.server = build_server(load_config(self.config_path))

    def tearDown(self):
        self.server.server_close()
        self.tempdir.cleanup()

    def test_reload_picks_up_a_raised_canary_weight(self):
        candidate = next(b for b in self.server.config.backends if b.id == "trailbrake-candidate")
        self.assertFalse(candidate.enabled)

        _write_config(self.config_path, candidate_weight=100, candidate_enabled="true",
                     log_path=self.server.config.server.log_path)
        reload_from_path(self.server, self.config_path)

        candidate = next(b for b in self.server.config.backends if b.id == "trailbrake-candidate")
        self.assertTrue(candidate.enabled)
        self.assertEqual(candidate.weight, 100)

    def test_reload_to_a_broken_config_keeps_the_old_one_serving(self):
        original_config = self.server.config
        self.config_path.write_text("this is not [ valid toml")

        reload_from_path(self.server, self.config_path)

        self.assertIs(self.server.config, original_config)

    def test_reload_to_a_config_that_fails_validation_keeps_the_old_one(self):
        original_config = self.server.config
        # Disabling the only rollback_target in "fast" fails config.py's
        # validation (a tier must always have a live safety net).
        self.config_path.write_text(
            _BASE_TOML.format(baseline_weight=100, candidate_weight=0, candidate_enabled="false",
                              log_path=self.server.config.server.log_path).replace(
                "rollback_target = true\nweight = 100", "rollback_target = false\nweight = 100"
            )
        )
        reload_from_path(self.server, self.config_path)
        self.assertIs(self.server.config, original_config)

    def test_reload_rebuilds_router_and_telemetry_together(self):
        original_router = self.server.router
        original_telemetry = self.server.telemetry
        _write_config(self.config_path, candidate_weight=50, candidate_enabled="true",
                     log_path=self.server.config.server.log_path)

        reload_from_path(self.server, self.config_path)

        self.assertIsNot(self.server.router, original_router)
        self.assertIsNot(self.server.telemetry, original_telemetry)
        # And the new router actually uses the new config, not stale state.
        self.assertIs(self.server.router.config, self.server.config)


class InstallSighupReloadTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tempdir.name) / "router.toml"
        _write_config(self.config_path, candidate_weight=0, candidate_enabled="false")
        from router.config import load_config

        self.server = build_server(load_config(self.config_path))
        self._restore = None

    def tearDown(self):
        if self._restore is not None:
            self._restore()
        self.server.server_close()
        self.tempdir.cleanup()

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "SIGHUP is POSIX-only")
    def test_registered_handler_triggers_a_reload_when_invoked_directly(self):
        self._restore = install_sighup_reload(self.server, self.config_path)
        handler = signal.getsignal(signal.SIGHUP)
        self.assertNotEqual(handler, signal.SIG_DFL)

        _write_config(self.config_path, candidate_weight=100, candidate_enabled="true",
                     log_path=self.server.config.server.log_path)
        handler(signal.SIGHUP, None)  # invoke the registered callback directly, no OS signal needed

        candidate = next(b for b in self.server.config.backends if b.id == "trailbrake-candidate")
        self.assertTrue(candidate.enabled)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "SIGHUP is POSIX-only")
    def test_restore_puts_back_the_previous_handler(self):
        previous = signal.getsignal(signal.SIGHUP)
        restore = install_sighup_reload(self.server, self.config_path)
        self.assertNotEqual(signal.getsignal(signal.SIGHUP), previous)
        restore()
        self.assertEqual(signal.getsignal(signal.SIGHUP), previous)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "SIGHUP is POSIX-only")
    def test_a_real_os_signal_triggers_the_reload(self):
        self._restore = install_sighup_reload(self.server, self.config_path)
        _write_config(self.config_path, candidate_weight=100, candidate_enabled="true",
                     log_path=self.server.config.server.log_path)

        os.kill(os.getpid(), signal.SIGHUP)

        deadline = time.monotonic() + 2.0
        candidate = next(b for b in self.server.config.backends if b.id == "trailbrake-candidate")
        while not candidate.enabled and time.monotonic() < deadline:
            time.sleep(0.01)
            candidate = next(b for b in self.server.config.backends if b.id == "trailbrake-candidate")
        self.assertTrue(candidate.enabled, "SIGHUP delivery did not trigger a reload within 2s")


if __name__ == "__main__":
    unittest.main()
