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


if __name__ == "__main__":
    unittest.main()
