import csv
import io
import itertools
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ML.CFG import CFG
from ML.experiment_matrix import (
    CANONICAL_ENVIRONMENT,
    ALWAYS_ON_GROUPS,
    GROUPED_FEATURE_SETS,
    canonical,
    PROTEIN_GROUPS,
    FEATURE_GROUPS,
    FEATURE_SETS,
    COMPARISON_FEATURE_SETS,
    DEFAULT_COMPARISON_FEATURE_SETS,
    MODEL_TYPES,
    ExperimentSpec,
    _configure_run,
    _merge_with_existing_summary,
    _write_summary,
    build_experiment_matrix,
    environment_differences,
    environment_record,
    report_environment,
    run_experiment_matrix,
)


class ExperimentMatrixTests(unittest.TestCase):
    def test_matrix_has_all_requested_combinations(self):
        specs = build_experiment_matrix()

        # 31 configurations per model: 15 power-set, 10 contrast,
        # 2 reconstruction, 3 paper, 1 contrast backbone.
        self.assertEqual(len(specs), 32 * len(MODEL_TYPES))
        self.assertEqual({spec.model_type for spec in specs}, set(MODEL_TYPES))
        self.assertEqual(
            {spec.feature_set for spec in specs},
            set(FEATURE_SETS) | {"kontras"},
        )
        self.assertEqual(
            len({spec.experiment_id for spec in specs}), 32 * len(MODEL_TYPES)
        )
        # Only the fixed-profile paper runs carry no groups; the contrast and
        # reconstruction arms are group-driven like the proposed arm.
        self.assertTrue(
            all(
                (spec.feature_groups is None)
                == (spec.feature_set not in GROUPED_FEATURE_SETS)
                for spec in specs
            )
        )
        proposed = [spec for spec in specs if spec.feature_set == "proposed"]
        self.assertEqual(len(proposed), len(MODEL_TYPES) * 15)
        # Nothing protein-level may reach the arm the selection rule reads.
        self.assertFalse(
            {group for spec in proposed for group in spec.feature_groups}
            & set(PROTEIN_GROUPS)
        )
        self.assertEqual(
            {
                spec.feature_groups
                for spec in proposed
                if spec.model_type == MODEL_TYPES[0]
            },
            {
                canonical(combination + ALWAYS_ON_GROUPS)
                for size in range(1, len(FEATURE_GROUPS) + 1)
                for combination in itertools.combinations(FEATURE_GROUPS, size)
            },
        )

    def test_proposed_group_switches_and_output_paths_are_distinct(self):
        base = CFG()
        output_root = Path("matrix-output")
        proposed = ExperimentSpec("proposed", "random_forest", ("aac", "prop", "len"))
        configured = _configure_run(base, proposed, output_root)

        self.assertTrue(configured.USE_AAC_FEATURES)
        self.assertFalse(configured.USE_DPC_FEATURES)
        self.assertTrue(configured.USE_PROP_FEATURES)
        self.assertFalse(configured.USE_FLANK_FEATURES)
        self.assertTrue(configured.USE_LEN_FEATURES)
        # Both historical blocks stay off whatever the base config said.
        self.assertFalse(configured.USE_PROTEIN_FEATURES)
        self.assertFalse(configured.USE_BIOLOGICAL_FEATURES)
        self.assertEqual(configured.TRAINING_MODE, base.TRAINING_MODE)
        self.assertEqual(configured.RANDOM_STATE, base.RANDOM_STATE)

        paths = {
            _configure_run(base, spec, output_root).OUTPUT_DIR
            for spec in build_experiment_matrix()
        }
        self.assertEqual(len(paths), 32 * len(MODEL_TYPES))

    def test_comparison_profiles_keep_existing_feature_switches(self):
        base = CFG()
        base.USE_AAC_FEATURES = True
        base.USE_DPC_FEATURES = False
        base.USE_PROTEIN_FEATURES = True
        base.USE_BIOLOGICAL_FEATURES = False
        configured = _configure_run(
            base,
            ExperimentSpec("perbandingan_1", "svm"),
            Path("matrix-output"),
        )

        self.assertEqual(configured.USE_AAC_FEATURES, base.USE_AAC_FEATURES)
        self.assertEqual(configured.USE_DPC_FEATURES, base.USE_DPC_FEATURES)
        self.assertEqual(configured.USE_PROTEIN_FEATURES, base.USE_PROTEIN_FEATURES)
        self.assertEqual(
            configured.USE_BIOLOGICAL_FEATURES, base.USE_BIOLOGICAL_FEATURES
        )

    def test_matrix_runner_records_all_runs_with_kaggle_tuning_metric(self):
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "mock-artifact"},
            ) as mocked_training:
                with redirect_stdout(io.StringIO()):
                    rows = run_experiment_matrix(temporary_directory)

            self.assertEqual(
                mocked_training.call_count, 32 * len(MODEL_TYPES)
            )
            self.assertTrue(all(row["status"] == "success" for row in rows))
            self.assertTrue(all(row["tuning_metric"] == "roc_auc" for row in rows))
            summary_path = Path(temporary_directory) / "experiment_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(len(summary_rows), 32 * len(MODEL_TYPES))
            self.assertEqual(
                {row["status"] for row in summary_rows}, {"success"}
            )


class CanonicalEnvironmentTests(unittest.TestCase):
    """Phase 2 must run where phase 1 ran, or the frozen configuration is void."""

    def test_record_covers_every_canonical_key(self):
        record = environment_record()

        self.assertEqual(set(record), set(CANONICAL_ENVIRONMENT))
        self.assertTrue(all(record.values()))

    def test_matching_environment_reports_no_differences(self):
        self.assertEqual(environment_differences(dict(CANONICAL_ENVIRONMENT)), {})
        with redirect_stdout(io.StringIO()):
            self.assertTrue(report_environment(dict(CANONICAL_ENVIRONMENT)))

    def test_one_changed_library_is_the_only_difference_reported(self):
        record = dict(CANONICAL_ENVIRONMENT)
        record["sklearn_version"] = "1.7.0"

        differences = environment_differences(record)

        self.assertEqual(
            differences, {"sklearn_version": (CANONICAL_ENVIRONMENT["sklearn_version"], "1.7.0")}
        )

    def test_absent_library_is_reported_rather_than_skipped(self):
        record = dict(CANONICAL_ENVIRONMENT)
        del record["xgboost_version"]

        differences = environment_differences(record)

        self.assertEqual(
            differences["xgboost_version"],
            (CANONICAL_ENVIRONMENT["xgboost_version"], "absent"),
        )

    def test_mismatch_warns_and_names_the_libraries(self):
        record = dict(CANONICAL_ENVIRONMENT)
        record["pandas_version"] = "3.0.6"
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            matched = report_environment(record)

        output = buffer.getvalue()
        self.assertFalse(matched)
        self.assertIn("WARNING", output)
        self.assertIn("pandas_version", output)
        self.assertIn("3.0.6", output)
        # The reason must be stated, not just the fact.
        self.assertIn("fold", output)

    def test_pinned_requirements_match_the_declared_environment(self):
        """A pin that drifts from CANONICAL_ENVIRONMENT would install the wrong thing."""
        text = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8")
        pins = dict(
            line.split("==", 1)
            for line in text.splitlines()
            if "==" in line and not line.strip().startswith("#")
        )
        for name in ("numpy", "pandas", "scikit-learn", "xgboost"):
            key = ("sklearn" if name == "scikit-learn" else name) + "_version"
            with self.subTest(library=name):
                self.assertEqual(pins.get(name), CANONICAL_ENVIRONMENT[key])


class SummaryMergeTests(unittest.TestCase):
    """Two invocations may share one output directory without losing rows."""

    def _write(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["experiment_id", "status"])
            writer.writeheader()
            writer.writerows(rows)

    def test_existing_rows_from_another_run_survive(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            self._write(path, [{"experiment_id": "a", "status": "success"}])

            merged = _merge_with_existing_summary(
                path, [{"experiment_id": "b", "status": "success"}]
            )

            self.assertEqual(
                {row["experiment_id"] for row in merged}, {"a", "b"}
            )

    def test_rerunning_an_experiment_replaces_its_row(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            self._write(path, [{"experiment_id": "a", "status": "failed"}])

            merged = _merge_with_existing_summary(
                path, [{"experiment_id": "a", "status": "success"}]
            )

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["status"], "success")

    def test_write_summary_round_trips_through_an_existing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            _write_summary(path, [{"experiment_id": "a", "status": "success"}])
            _write_summary(path, [{"experiment_id": "b", "status": "success"}])

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                {row["experiment_id"] for row in rows}, {"a", "b"}
            )

    def test_a_corrupt_summary_does_not_abort_the_run(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            path.write_bytes(b"\x00\x01 not,a,csv\n\x00")

            merged = _merge_with_existing_summary(
                path, [{"experiment_id": "a", "status": "success"}]
            )

            self.assertEqual([row["experiment_id"] for row in merged], ["a"])


class ComparisonProfileSelectionTests(unittest.TestCase):
    """A profile outside the default sweep must be runnable on its own.

    FS-5 (ProtT5) is far more expensive than the tabular profiles and phase 1's
    artifacts are frozen, so it has to run as its own sweep without changing
    what the default sweep plans.
    """

    def test_naming_one_profile_runs_only_that_profile(self):
        specs = build_experiment_matrix(
            comparison_feature_sets=("perbandingan_5",)
        )
        self.assertEqual(len(specs), len(MODEL_TYPES))
        self.assertEqual(
            {spec.feature_set for spec in specs}, {"perbandingan_5"}
        )
        self.assertEqual(
            {spec.model_type for spec in specs}, set(MODEL_TYPES)
        )

    def test_the_default_sweep_is_unchanged_by_the_new_parameter(self):
        # The guard that matters: phase 1's plan is 31 configurations per
        # model and its artifacts are frozen, so the default plan must not
        # move when a profile is added to the registry.
        specs = build_experiment_matrix()
        self.assertEqual(len(specs), 32 * len(MODEL_TYPES))
        self.assertNotIn(
            "perbandingan_5", {spec.feature_set for spec in specs}
        )
        self.assertEqual(
            DEFAULT_COMPARISON_FEATURE_SETS,
            tuple(name for name in FEATURE_SETS if name != "proposed"),
        )

    def test_every_default_profile_is_a_known_profile(self):
        for name in DEFAULT_COMPARISON_FEATURE_SETS:
            self.assertIn(name, COMPARISON_FEATURE_SETS)

    def test_restricting_the_profiles_switches_the_other_arms_off(self):
        specs = build_experiment_matrix(
            comparison_feature_sets=("perbandingan_5",)
        )
        arms = {spec.feature_set for spec in specs}
        self.assertNotIn("proposed", arms)
        self.assertNotIn("kontras", arms)

    def test_the_proposed_arm_can_still_be_asked_for_explicitly(self):
        specs = build_experiment_matrix(
            MODEL_TYPES,
            ("prop", "len"),
            comparison_feature_sets=("perbandingan_5",),
            include_proposed=True,
        )
        self.assertEqual(len(specs), 2 * len(MODEL_TYPES))
        self.assertEqual(
            {spec.feature_set for spec in specs},
            {"proposed", "perbandingan_5"},
        )

    def test_an_unknown_profile_is_refused(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(comparison_feature_sets=("perbandingan_9",))

    def test_naming_no_profile_is_refused(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(comparison_feature_sets=())

    def test_switching_every_arm_off_is_refused(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(
                MODEL_TYPES,
                include_comparisons=False,
                include_contrasts=False,
                include_proposed=False,
            )

    def test_identifiers_stay_unique_when_profiles_are_combined(self):
        specs = build_experiment_matrix(
            comparison_feature_sets=("perbandingan_4", "perbandingan_5"),
        )
        self.assertEqual(len(specs), 2 * len(MODEL_TYPES))
        self.assertEqual(
            len({spec.experiment_id for spec in specs}), len(specs)
        )


if __name__ == "__main__":
    unittest.main()
