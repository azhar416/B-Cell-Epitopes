"""Tests that the fast paths produce exactly what the slow ones did.

Each optimisation here replaced working but wasteful code, so every test pins
the behaviour rather than the implementation: the vectorised threshold sweep is
compared against the per-threshold loop it replaced, and the external arm is
checked for selecting its hyperparameters once instead of twice.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ML.CFG import CFG
from ML.evaluation import calculate_metrics, mcc_at_thresholds, select_threshold_by_mcc
from ML.selection import apply_parameters


def _mcc_loop(labels, probabilities, thresholds):
    """The original implementation: one full confusion matrix per threshold."""
    return np.array(
        [
            calculate_metrics(labels, probabilities, threshold=float(value))["mcc"]
            for value in thresholds
        ]
    )


class VectorisedThresholdSweepTests(unittest.TestCase):
    def _sample(self, seed, size=400, positive_rate=0.27, decimals=3):
        generator = np.random.default_rng(seed)
        labels = (generator.random(size) < positive_rate).astype(int)
        # Rounding creates the heavy tie structure real OOF scores have.
        probabilities = np.round(generator.beta(2, 5, size), decimals)
        return labels, probabilities

    def test_curve_and_choice_match_the_per_threshold_loop(self):
        """The reference loop is slow, so both claims share one pass over it."""
        for seed in range(4):
            labels, probabilities = self._sample(seed)
            thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
            expected_curve = _mcc_loop(labels, probabilities, thresholds)
            best = max(expected_curve)
            tied = [
                float(value)
                for value, score in zip(thresholds, expected_curve)
                if np.isclose(score, best)
            ]
            expected_threshold = min(tied, key=lambda value: abs(value - 0.5))

            with self.subTest(seed=seed):
                np.testing.assert_allclose(
                    mcc_at_thresholds(labels, probabilities, thresholds),
                    expected_curve,
                    rtol=0,
                    atol=1e-12,
                )
                threshold, mcc = select_threshold_by_mcc(labels, probabilities)
                self.assertEqual(threshold, expected_threshold)
                self.assertAlmostEqual(mcc, float(best), places=12)

    def test_endpoints_are_always_candidates(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.2, 0.3, 0.7, 0.8])
        threshold, mcc = select_threshold_by_mcc(labels, probabilities)

        # A perfect separation is reachable, so MCC must hit 1.0.
        self.assertAlmostEqual(mcc, 1.0, places=12)
        self.assertAlmostEqual(threshold, 0.7, places=12)

    def test_degenerate_scores_do_not_divide_by_zero(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.full(4, 0.5)
        threshold, mcc = select_threshold_by_mcc(labels, probabilities)

        self.assertTrue(np.isfinite(mcc))
        self.assertAlmostEqual(mcc, 0.0, places=12)
        self.assertTrue(0.0 <= threshold <= 1.0)

    def test_ties_resolve_towards_the_default_cut_off(self):
        # Thresholds 0.4 and 0.6 both split the data identically, so the one
        # nearer 0.5 must win; 0.4 and 0.6 are equidistant, so the lower wins.
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.4, 0.4, 0.6, 0.6])
        threshold, mcc = select_threshold_by_mcc(labels, probabilities)

        self.assertAlmostEqual(mcc, 1.0, places=12)
        self.assertAlmostEqual(threshold, 0.6, places=12)


class ApplyParametersTests(unittest.TestCase):
    def test_returns_a_validated_copy_with_tuning_disabled(self):
        config = CFG()
        config.MODEL_TYPE = "logistic_regression"
        config.HYPERPARAMETER_TUNING = True

        resolved = apply_parameters(config, {"LOGISTIC_REGRESSION_C": 100.0})

        self.assertEqual(resolved.LOGISTIC_REGRESSION_C, 100.0)
        self.assertFalse(resolved.HYPERPARAMETER_TUNING)
        # The original must be untouched, or fold configs would leak into
        # each other through the shared base object.
        self.assertTrue(config.HYPERPARAMETER_TUNING)
        self.assertNotEqual(config.LOGISTIC_REGRESSION_C, 100.0)

    def test_invalid_parameters_are_rejected_immediately(self):
        config = CFG()
        config.MODEL_TYPE = "logistic_regression"
        with self.assertRaises(ValueError):
            apply_parameters(config, {"LOGISTIC_REGRESSION_C": -1.0})


class ExternalArmSelectsOnceTests(unittest.TestCase):
    """The external arm and its internal CV must share one selection pass."""

    @classmethod
    def setUpClass(cls):
        import ML.train as train_module

        cls.calls: list[str] = []
        real = train_module._select_external_model_from_bcell

        def counting(*args, **kwargs):
            cls.calls.append(kwargs.get("dataset_name", "bcell"))
            return real(*args, **kwargs)

        config = CFG()
        config.TRAINING_MODE = "external_sars"
        config.MODEL_TYPE = "logistic_regression"
        config.FEATURE_SET = "proposed"
        config.USE_AAC_FEATURES = False
        config.USE_DPC_FEATURES = False
        config.USE_PROTEIN_FEATURES = False
        config.USE_BIOLOGICAL_FEATURES = True

        cls._directory = TemporaryDirectory()
        config.OUTPUT_DIR = Path(cls._directory.name)
        with patch.object(
            train_module, "_select_external_model_from_bcell", counting
        ):
            cls.report = train_module.run_training(config)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_selection_runs_exactly_once(self):
        self.assertEqual(
            len(self.calls),
            1,
            "the internal CV recomputed a selection the external arm already made",
        )

    def test_internal_cv_folds_use_the_external_selection(self):
        chosen = self.report["model_selection"]["parameters"]
        folds = self.report["internal_bcell_cv"]["folds"]

        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertEqual(fold["selected_hyperparameters"], chosen)

    def test_threshold_comes_from_the_selected_candidate(self):
        threshold = self.report["prediction_threshold"]
        self.assertTrue(0.0 <= threshold <= 1.0)
        self.assertEqual(
            threshold, self.report["internal_bcell_cv"]["selected_threshold"]
        )


if __name__ == "__main__":
    unittest.main()
