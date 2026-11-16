"""Tests for the pre-registered one-standard-error selection rule."""

import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from ML.experiment_matrix import build_experiment_matrix
from ML.pre_dataset import build_feature_frame
from ML.select_configuration import (
    GROUP_FEATURE_COUNTS,
    combination_label,
    feature_count,
    read_proposed_runs,
    select_consensus,
    select_per_model,
    tuning_protocol,
)


def _record(model, groups, mean, std, n_features=None):
    return {
        "model_type": model,
        "feature_groups": tuple(groups),
        "label": combination_label(groups),
        "n_features": n_features if n_features is not None else feature_count(groups),
        "mean": mean,
        "std": std,
    }


class FeatureCountTests(unittest.TestCase):
    def test_hardcoded_counts_match_the_real_feature_builder(self):
        """Guards against GROUP_FEATURE_COUNTS drifting from the pipeline."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "peptide_seq": ["ACDEFGHIKL", "MNPQRSTVWY"],
                "protein_seq": ["ACDEFGHIKLMNPQRSTVWY"] * 2,
                "start_position": [1, 2],
                "end_position": [10, 11],
                "chou_fasman": [1.0, 1.0],
                "emini": [1.0, 1.0],
                "kolaskar_tongaonkar": [1.0, 1.0],
                "parker": [1.0, 1.0],
                "isoelectric_point": [7.0, 7.0],
                "aromaticity": [0.1, 0.1],
                "hydrophobicity": [0.1, 0.1],
                "stability": [30.0, 30.0],
            }
        )
        for group in GROUP_FEATURE_COUNTS:
            kwargs = {
                "use_aac_features": group == "aac",
                "use_dpc_features": group == "dpc",
                "use_protein_features": group == "protein",
                "use_biological_features": group == "biological",
            }
            columns = build_feature_frame(
                frame, use_peptide_features=False, **kwargs
            ).shape[1]
            self.assertEqual(
                columns,
                GROUP_FEATURE_COUNTS[group],
                f"GROUP_FEATURE_COUNTS[{group!r}] is stale",
            )

    def test_combination_width_is_the_sum_of_its_groups(self):
        self.assertEqual(feature_count(("aac", "biological")), 30)
        self.assertEqual(feature_count(("aac", "dpc", "protein", "biological")), 453)

    def test_label_uses_canonical_order_regardless_of_input_order(self):
        self.assertEqual(combination_label(("biological", "aac")), "AAC + BIO")
        self.assertEqual(combination_label(("protein", "dpc")), "DPC + PROT")


class OneStandardErrorTests(unittest.TestCase):
    def test_simpler_combination_wins_when_inside_one_se(self):
        records = [
            # Widest combination has the top mean but only by 0.004, while its
            # SE is 0.05/sqrt(5) = 0.0224, so the simpler one must win.
            _record("rf", ("aac", "dpc", "protein", "biological"), 0.640, 0.050),
            _record("rf", ("aac", "biological"), 0.636, 0.030),
        ]

        result = select_consensus(records)

        self.assertEqual(result["best_label"], "AAC + DPC + BIO + PROT")
        self.assertEqual(result["chosen"]["label"], "AAC + BIO")
        self.assertEqual(len(result["within_one_se"]), 2)

    def test_clearly_better_combination_is_kept_even_if_complex(self):
        records = [
            _record("rf", ("aac", "dpc", "protein", "biological"), 0.700, 0.010),
            _record("rf", ("biological",), 0.600, 0.010),
        ]

        result = select_consensus(records)

        self.assertEqual(result["chosen"]["label"], "AAC + DPC + BIO + PROT")
        self.assertEqual(len(result["within_one_se"]), 1)

    def test_ties_on_size_break_towards_the_steadier_candidate(self):
        records = [
            _record("rf", ("protein",), 0.650, 0.090),
            _record("rf", ("aac",), 0.648, 0.010),
        ]

        result = select_consensus(records)

        # Both are within one SE and nearly the same width, so the lower SD wins.
        self.assertEqual(result["chosen"]["label"], "AAC")

    def test_threshold_is_mean_minus_one_standard_error(self):
        records = [_record("rf", ("aac",), 0.600, 0.100)]

        result = select_consensus(records)

        self.assertAlmostEqual(result["standard_error"], 0.1 / 5**0.5, places=12)
        self.assertAlmostEqual(result["threshold"], 0.6 - 0.1 / 5**0.5, places=12)

    def test_consensus_averages_across_classifiers(self):
        records = [
            _record("rf", ("aac",), 0.70, 0.01),
            _record("svm", ("aac",), 0.50, 0.01),
            _record("rf", ("biological",), 0.61, 0.01),
            _record("svm", ("biological",), 0.60, 0.01),
        ]

        result = select_consensus(records)

        # AAC averages 0.60 while BIO averages 0.605, so BIO leads despite
        # losing badly on random_forest alone.
        self.assertEqual(result["best_label"], "BIO")

    def test_per_model_selection_is_independent_per_classifier(self):
        records = [
            _record("rf", ("aac",), 0.70, 0.001),
            _record("rf", ("biological",), 0.50, 0.001),
            _record("svm", ("aac",), 0.50, 0.001),
            _record("svm", ("biological",), 0.70, 0.001),
        ]

        result = select_per_model(records)

        self.assertEqual(result["rf"]["chosen"]["label"], "AAC")
        self.assertEqual(result["svm"]["chosen"]["label"], "BIO")

    def test_empty_candidate_set_is_an_error(self):
        with self.assertRaises(ValueError):
            select_consensus([])


class SummaryParsingTests(unittest.TestCase):
    def _write(self, directory: Path, rows):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment_summary.csv"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_only_successful_proposed_runs_are_considered(self):
        rows = [
            {
                "experiment_id": "a",
                "status": "success",
                "model_type": "random_forest",
                "feature_set": "proposed",
                "feature_groups": "aac,biological",
                "mean_validation_roc_auc": "0.63",
                "std_validation_roc_auc": "0.02",
            },
            {  # baselines are not candidates for the proposed arm
                "experiment_id": "b",
                "status": "success",
                "model_type": "random_forest",
                "feature_set": "perbandingan_1",
                "feature_groups": "profile_defined",
                "mean_validation_roc_auc": "0.99",
                "std_validation_roc_auc": "0.01",
            },
            {  # failures must never enter the selection
                "experiment_id": "c",
                "status": "failed",
                "model_type": "random_forest",
                "feature_set": "proposed",
                "feature_groups": "dpc",
                "mean_validation_roc_auc": "0.98",
                "std_validation_roc_auc": "0.01",
            },
        ]
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "AAC + BIO")

    def test_phase_two_summaries_fall_back_to_bcell_oof_columns(self):
        rows = [
            {
                "experiment_id": "a",
                "status": "success",
                "model_type": "svm",
                "feature_set": "proposed",
                "feature_groups": "aac",
                "bcell_oof_roc_auc": "0.61",
                "bcell_oof_std_roc_auc": "0.03",
            }
        ]
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["mean"], 0.61)
        self.assertAlmostEqual(records[0]["std"], 0.03)

    def test_feature_group_order_is_normalised(self):
        rows = [
            {
                "experiment_id": "a",
                "status": "success",
                "model_type": "svm",
                "feature_set": "proposed",
                "feature_groups": "biological,aac",
                "mean_validation_roc_auc": "0.61",
                "std_validation_roc_auc": "0.03",
            }
        ]
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(records[0]["feature_groups"], ("aac", "biological"))


class TuningProtocolProvenanceTests(unittest.TestCase):
    """Phase 2 must inherit phase 1's tuning metric and seed, not a default."""

    def _records(self, *pairs):
        return [
            {"tuning_metric": metric, "random_state": seed}
            for metric, seed in pairs
        ]

    def test_uniform_sweep_reports_its_single_protocol(self):
        protocol = tuning_protocol(
            self._records(("roc_auc", "42"), ("roc_auc", "42"))
        )

        self.assertTrue(protocol["is_uniform"])
        self.assertEqual(protocol["tuning_metric"], "roc_auc")
        self.assertEqual(protocol["random_state"], "42")

    def test_mixed_metrics_are_flagged_instead_of_picked(self):
        protocol = tuning_protocol(
            self._records(("roc_auc", "42"), ("pr_auc", "42"))
        )

        self.assertFalse(protocol["is_uniform"])
        self.assertIsNone(protocol["tuning_metric"])
        self.assertEqual(protocol["tuning_metrics_seen"], ["pr_auc", "roc_auc"])

    def test_mixed_seeds_are_flagged(self):
        protocol = tuning_protocol(
            self._records(("roc_auc", "42"), ("roc_auc", "7"))
        )

        self.assertFalse(protocol["is_uniform"])
        self.assertIsNone(protocol["random_state"])

    def test_records_without_provenance_do_not_crash(self):
        protocol = tuning_protocol([{"mean": 0.6, "std": 0.01}])

        self.assertFalse(protocol["is_uniform"])
        self.assertEqual(protocol["tuning_metrics_seen"], [])

    def test_two_library_environments_are_not_uniform(self):
        """Library versions decide the fold split, so mixing them is invalid."""
        local = "python=3.14.8 sklearn=1.9.1 pandas=3.0.6"
        kaggle = "python=3.11.11 sklearn=1.2.2 pandas=2.2.3"
        records = [
            {"tuning_metric": "roc_auc", "random_state": "42", "environment": local},
            {"tuning_metric": "roc_auc", "random_state": "42", "environment": kaggle},
        ]

        protocol = tuning_protocol(records)

        self.assertFalse(protocol["is_uniform"])
        self.assertIsNone(protocol["environment"])
        self.assertEqual(protocol["environments_seen"], sorted([kaggle, local]))

    def test_one_environment_stays_uniform(self):
        same = "python=3.14.8 sklearn=1.9.1"
        records = [
            {"tuning_metric": "roc_auc", "random_state": "42", "environment": same}
        ] * 2

        protocol = tuning_protocol(records)

        self.assertTrue(protocol["is_uniform"])
        self.assertEqual(protocol["environment"], same)
        self.assertTrue(protocol["environment_recorded"])

    def test_summary_without_version_columns_is_flagged_as_unrecorded(self):
        """Older summaries predate version recording; that must not read as uniform."""
        records = [{"tuning_metric": "roc_auc", "random_state": "42"}]

        protocol = tuning_protocol(records)

        self.assertFalse(protocol["environment_recorded"])
        self.assertEqual(protocol["environments_seen"], [])

    def test_environment_columns_are_read_from_the_summary(self):
        rows = [
            {
                "experiment_id": "a",
                "status": "success",
                "model_type": "svm",
                "feature_set": "proposed",
                "feature_groups": "aac",
                "tuning_metric": "roc_auc",
                "random_state": "42",
                "python_version": "3.11.11",
                "numpy_version": "1.26.4",
                "pandas_version": "2.2.3",
                "sklearn_version": "1.2.2",
                "xgboost_version": "2.0.3",
                "mean_validation_roc_auc": "0.61",
                "std_validation_roc_auc": "0.03",
            }
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(
            records[0]["environment"],
            "python=3.11.11 numpy=1.26.4 pandas=2.2.3 sklearn=1.2.2 xgboost=2.0.3",
        )

    def test_summary_rows_carry_the_protocol_through(self):
        rows = [
            {
                "experiment_id": "a",
                "status": "success",
                "model_type": "svm",
                "feature_set": "proposed",
                "feature_groups": "aac",
                "tuning_metric": "roc_auc",
                "random_state": "42",
                "selected_hyperparameters": '{"SVM_C": 10.0}',
                "mean_validation_roc_auc": "0.61",
                "std_validation_roc_auc": "0.03",
            }
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(records[0]["tuning_metric"], "roc_auc")
        self.assertEqual(records[0]["random_state"], "42")
        self.assertTrue(tuning_protocol(records)["is_uniform"])


class FrozenCombinationRunTests(unittest.TestCase):
    def test_freezing_a_combination_yields_the_phase_two_matrix(self):
        specs = build_experiment_matrix(feature_groups=("aac", "biological"))

        proposed = [spec for spec in specs if spec.feature_set == "proposed"]
        baselines = [spec for spec in specs if spec.feature_set != "proposed"]

        self.assertEqual(len(specs), 20)
        self.assertEqual(len(proposed), 5)
        self.assertEqual(len(baselines), 15)
        self.assertEqual(
            {spec.feature_groups for spec in proposed}, {("aac", "biological")}
        )

    def test_frozen_experiment_id_matches_the_phase_one_run(self):
        full = build_experiment_matrix(("random_forest",))
        frozen = build_experiment_matrix(
            ("random_forest",), feature_groups=("biological", "aac")
        )

        frozen_proposed = [s for s in frozen if s.feature_set == "proposed"][0]
        self.assertIn(frozen_proposed.experiment_id, {s.experiment_id for s in full})
        self.assertEqual(
            frozen_proposed.experiment_id, "proposed__random_forest__aac__biological"
        )

    def test_unknown_feature_group_is_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(feature_groups=("aac", "nonsense"))


if __name__ == "__main__":
    unittest.main()
