"""Tests for the pre-registered one-standard-error selection rule."""

import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from ML.experiment_matrix import MODEL_TYPES, build_experiment_matrix
from ML.pre_dataset import build_feature_frame
from ML.experiment_matrix import (
    ALL_COMBINATIONS,
    ALWAYS_ON_GROUPS,
    GROUP_SWITCHES,
)
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
                # Real coordinates: flank features refuse to build unless the
                # peptide actually sits at start_position..end_position.
                "start_position": [1, 11],
                "end_position": [10, 20],
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
        frame["parent_protein_id"] = ["P1", "P1"]
        # Derived from GROUP_SWITCHES so a new group cannot be added to the
        # matrix without its width being checked here too.
        switches = {
            group: switch.lower() for group, switch in GROUP_SWITCHES.items()
        }
        switches["biological"] = "use_biological_features"
        switches["protein"] = "use_protein_features"
        self.assertEqual(set(switches), set(GROUP_FEATURE_COUNTS))
        for group in GROUP_FEATURE_COUNTS:
            kwargs = {name: name == switches[group] for name in switches.values()}
            columns = build_feature_frame(
                frame, use_peptide_features=False, **kwargs
            ).shape[1]
            self.assertEqual(
                columns,
                GROUP_FEATURE_COUNTS[group],
                f"GROUP_FEATURE_COUNTS[{group!r}] is stale",
            )

    def test_combination_width_is_the_sum_of_its_groups(self):
        self.assertEqual(feature_count(("aac", "prop")), 24)
        self.assertEqual(feature_count(("aac", "prop", "flank", "len")), 67)
        self.assertEqual(feature_count(("aac", "prop", "flank", "len", "dpc")), 467)

    def test_label_uses_canonical_order_regardless_of_input_order(self):
        self.assertEqual(combination_label(("prop", "aac")), "AAC + PROP")
        self.assertEqual(combination_label(("len", "flank")), "FLANK + LEN")
        self.assertEqual(combination_label(("protein", "dpc")), "DPC + PROT")


class HighestMeanTests(unittest.TestCase):
    def test_top_mean_wins_even_when_the_margin_is_inside_one_se(self):
        records = [
            # The one-standard-error rule used to hand this to the simpler
            # combination, because 0.004 is well inside 0.05/sqrt(5)=0.0224.
            # Under the argmax rule the top mean takes it regardless.
            _record("rf", ("aac", "prop", "flank", "len"), 0.640, 0.050),
            _record("rf", ("aac", "prop"), 0.636, 0.030),
        ]

        result = select_consensus(records)

        self.assertEqual(result["chosen"]["label"], "AAC + PROP + FLANK + LEN")
        self.assertEqual(result["best_label"], "AAC + PROP + FLANK + LEN")

    def test_a_margin_inside_one_se_is_flagged_as_unstable(self):
        records = [
            _record("rf", ("aac", "prop", "flank", "len"), 0.640, 0.050),
            _record("rf", ("aac", "prop"), 0.636, 0.030),
        ]

        result = select_consensus(records)

        self.assertAlmostEqual(result["runner_up_margin"], 0.004, places=12)
        self.assertEqual(result["runner_up_label"], "AAC + PROP")
        # 0.004 does not clear 0.05/sqrt(5), so the ranking is not stable.
        self.assertLess(result["runner_up_margin"], result["standard_error"])

    def test_a_margin_outside_one_se_is_not_flagged(self):
        records = [
            _record("rf", ("aac", "prop", "flank", "len"), 0.700, 0.010),
            _record("rf", ("prop",), 0.600, 0.010),
        ]

        result = select_consensus(records)

        self.assertEqual(result["chosen"]["label"], "AAC + PROP + FLANK + LEN")
        self.assertGreater(result["runner_up_margin"], result["standard_error"])

    def test_near_ties_do_not_break_towards_the_steadier_candidate(self):
        records = [
            _record("rf", ("protein",), 0.650, 0.090),
            _record("rf", ("aac",), 0.648, 0.010),
        ]

        result = select_consensus(records)

        # 0.650 > 0.648, so the noisier combination wins outright. Only an
        # EXACT tie falls to the fewest features and then the lower SD.
        self.assertEqual(result["chosen"]["label"], "PROT")
        self.assertEqual(result["n_tied_at_best"], 1)

    def test_exact_ties_break_towards_fewer_features(self):
        records = [
            _record("rf", ("aac", "prop", "flank", "len"), 0.640, 0.010),
            _record("rf", ("prop",), 0.640, 0.050),
        ]

        result = select_consensus(records)

        self.assertEqual(result["n_tied_at_best"], 2)
        self.assertEqual(result["chosen"]["label"], "PROP")

    def test_standard_error_is_the_winners_sd_over_root_folds(self):
        records = [_record("rf", ("aac",), 0.600, 0.100)]

        result = select_consensus(records)

        self.assertAlmostEqual(result["standard_error"], 0.1 / 5**0.5, places=12)
        self.assertIsNone(result["runner_up_margin"])

    def test_consensus_averages_across_classifiers(self):
        records = [
            _record("rf", ("aac",), 0.70, 0.01),
            _record("svm", ("aac",), 0.50, 0.01),
            _record("rf", ("prop",), 0.61, 0.01),
            _record("svm", ("prop",), 0.60, 0.01),
        ]

        result = select_consensus(records)

        # AAC averages 0.60 while PROP averages 0.605, so PROP leads despite
        # losing badly on random_forest alone.
        self.assertEqual(result["best_label"], "PROP")

    def test_per_model_selection_is_independent_per_classifier(self):
        records = [
            _record("rf", ("aac",), 0.70, 0.001),
            _record("rf", ("prop",), 0.50, 0.001),
            _record("svm", ("aac",), 0.50, 0.001),
            _record("svm", ("prop",), 0.70, 0.001),
        ]

        result = select_per_model(records)

        self.assertEqual(result["rf"]["chosen"]["label"], "AAC")
        self.assertEqual(result["svm"]["chosen"]["label"], "PROP")

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
                "feature_groups": "aac,prop",
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
        self.assertEqual(records[0]["label"], "AAC + PROP")

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
                "feature_groups": "prop,aac",
                "mean_validation_roc_auc": "0.61",
                "std_validation_roc_auc": "0.03",
            }
        ]
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), rows)
            records = read_proposed_runs([path], "roc_auc")

        self.assertEqual(records[0]["feature_groups"], ("aac", "prop"))


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
        self.assertFalse(protocol["environment_partially_recorded"])

    def test_mixing_recorded_and_unrecorded_runs_is_not_uniform(self):
        """The likeliest mismatch: half the sweep predates version recording."""
        records = [
            {"tuning_metric": "roc_auc", "random_state": "42"},
            {
                "tuning_metric": "roc_auc",
                "random_state": "42",
                "environment": "python=3.11.11 sklearn=1.2.2",
            },
        ]

        protocol = tuning_protocol(records)

        self.assertFalse(protocol["is_uniform"])
        self.assertTrue(protocol["environment_partially_recorded"])
        self.assertFalse(protocol["environment_recorded"])
        self.assertIsNone(protocol["environment"])

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
        specs = build_experiment_matrix(feature_groups=("aac", "prop", "len"))

        proposed = [spec for spec in specs if spec.feature_set == "proposed"]
        baselines = [spec for spec in specs if spec.feature_set != "proposed"]

        # every model x (1 frozen combination + 4 paper feature sets).
        models = len(MODEL_TYPES)
        self.assertEqual(len(specs), 5 * models)
        self.assertEqual(len(proposed), models)
        self.assertEqual(len(baselines), 4 * models)
        self.assertEqual(
            {spec.feature_groups for spec in proposed}, {("aac", "prop", "len")}
        )

    def test_frozen_experiment_id_matches_the_phase_one_run(self):
        full = build_experiment_matrix(("random_forest",))
        frozen = build_experiment_matrix(
            ("random_forest",), feature_groups=("prop", "len", "aac")
        )

        frozen_proposed = [s for s in frozen if s.feature_set == "proposed"][0]
        self.assertIn(frozen_proposed.experiment_id, {s.experiment_id for s in full})
        self.assertEqual(
            frozen_proposed.experiment_id, "proposed__random_forest__aac__prop__len"
        )

    def test_unknown_feature_group_is_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(feature_groups=("aac", "nonsense"))


class PrintedPhaseTwoCommandTests(unittest.TestCase):
    """The command select_configuration hands the user must actually run.

    It names every group of the frozen combination, including the always-on
    ones, so the matrix CLI has to accept all of them. When ALWAYS_ON_GROUPS
    was introduced the CLI still only offered FEATURE_GROUPS, and the printed
    command was rejected with "invalid choice: 'len'".
    """

    def _cli_choices(self):
        import argparse
        from unittest.mock import patch

        import ML.experiment_matrix as matrix

        captured = {}
        real_add = argparse.ArgumentParser.add_argument

        def spy(self, *args, **kwargs):
            if "--feature-groups" in args:
                captured["choices"] = tuple(kwargs.get("choices") or ())
            return real_add(self, *args, **kwargs)

        with patch.object(argparse.ArgumentParser, "add_argument", spy):
            with patch.object(matrix, "run_experiment_matrix"):
                with patch("sys.argv", ["ML.experiment_matrix", "--help"]):
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            matrix.main()
        return captured["choices"]

    def test_cli_accepts_every_group_a_frozen_combination_can_name(self):
        choices = self._cli_choices()
        for combination in ALL_COMBINATIONS:
            for group in combination:
                with self.subTest(group=group):
                    self.assertIn(group, choices)

    def test_always_on_groups_are_offered_by_the_cli(self):
        choices = self._cli_choices()
        for group in ALWAYS_ON_GROUPS:
            with self.subTest(group=group):
                self.assertIn(group, choices)


if __name__ == "__main__":
    unittest.main()
