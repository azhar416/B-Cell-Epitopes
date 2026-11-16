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
    FEATURE_GROUPS,
    FEATURE_SETS,
    MODEL_TYPES,
    ExperimentSpec,
    _configure_run,
    _merge_with_existing_summary,
    _write_summary,
    build_experiment_matrix,
    run_experiment_matrix,
)


class ExperimentMatrixTests(unittest.TestCase):
    def test_matrix_has_all_requested_combinations(self):
        specs = build_experiment_matrix()

        self.assertEqual(len(specs), 90)
        self.assertEqual({spec.model_type for spec in specs}, set(MODEL_TYPES))
        self.assertEqual({spec.feature_set for spec in specs}, set(FEATURE_SETS))
        self.assertEqual(len({spec.experiment_id for spec in specs}), 90)
        self.assertTrue(
            all(
                spec.feature_groups is None
                for spec in specs
                if spec.feature_set != "proposed"
            )
        )
        proposed = [spec for spec in specs if spec.feature_set == "proposed"]
        self.assertEqual(len(proposed), len(MODEL_TYPES) * 15)
        self.assertEqual(
            {
                spec.feature_groups
                for spec in proposed
                if spec.model_type == MODEL_TYPES[0]
            },
            {
                combination
                for size in range(1, len(FEATURE_GROUPS) + 1)
                for combination in itertools.combinations(FEATURE_GROUPS, size)
            },
        )

    def test_proposed_group_switches_and_output_paths_are_distinct(self):
        base = CFG()
        output_root = Path("matrix-output")
        proposed = ExperimentSpec("proposed", "random_forest", ("aac", "protein"))
        configured = _configure_run(base, proposed, output_root)

        self.assertTrue(configured.USE_AAC_FEATURES)
        self.assertFalse(configured.USE_DPC_FEATURES)
        self.assertTrue(configured.USE_PROTEIN_FEATURES)
        self.assertFalse(configured.USE_BIOLOGICAL_FEATURES)
        self.assertEqual(configured.TRAINING_MODE, base.TRAINING_MODE)
        self.assertEqual(configured.RANDOM_STATE, base.RANDOM_STATE)

        paths = {
            _configure_run(base, spec, output_root).OUTPUT_DIR
            for spec in build_experiment_matrix()
        }
        self.assertEqual(len(paths), 90)

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

            self.assertEqual(mocked_training.call_count, 90)
            self.assertTrue(all(row["status"] == "success" for row in rows))
            self.assertTrue(all(row["tuning_metric"] == "roc_auc" for row in rows))
            summary_path = Path(temporary_directory) / "experiment_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(len(summary_rows), 90)
            self.assertEqual(
                {row["status"] for row in summary_rows}, {"success"}
            )


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


if __name__ == "__main__":
    unittest.main()
