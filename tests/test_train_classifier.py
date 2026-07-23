"""Trainer/predictor sanity: JSON-roundtrip determinism and separability on
an obvious pair. Guards the artifact format shadow_eval.py consumes."""
import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "train_classifier",
    os.path.join(os.path.dirname(__file__), "..", "bench", "escalation_eval", "train_classifier.py"),
)
tc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tc)


class TrainerTests(unittest.TestCase):
    def test_trains_separates_and_roundtrips(self):
        # linearly separable toy set in the real feature space
        hard = ([0.6, 2.0, 0.0, 6.0, 1.0], 1, 1.0)
        easy = ([0.0, 0.0, 1.0, 3.0, 1.0], 0, 1.0)
        model = tc.train([hard] * 30 + [easy] * 30, epochs=300)
        self.assertGreater(tc.predict(model, hard[0]), 0.9)
        self.assertLess(tc.predict(model, easy[0]), 0.1)
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
            json.dump(model, fh)
            path = fh.name
        loaded = json.load(open(path))
        self.assertEqual(tc.predict(loaded, hard[0]), tc.predict(model, hard[0]))
        os.unlink(path)

    def test_synthetic_generation_is_seeded_and_balanced(self):
        feats = lambda p: [0.0, 0.0, 0.0, float(len(p)), 1.0]  # noqa: E731
        a = tc.synthetic_examples(50, seed=7, feature_fn=feats)
        b = tc.synthetic_examples(50, seed=7, feature_fn=feats)
        self.assertEqual(a, b)  # reproducible
        self.assertEqual(sum(y for _, y, _ in a), 25)  # balanced

    def test_refuses_tiny_training_sets(self):
        # main() guards n<20; the guard matters because a "model" trained on
        # a handful of rows is noise wearing a model's name.
        import subprocess
        import sys
        script = os.path.join(os.path.dirname(__file__), "..", "bench", "escalation_eval", "train_classifier.py")
        proc = subprocess.run([sys.executable, script, "--synthetic", "4", "--out", "/dev/null"],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
