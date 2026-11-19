"""Tests for the cross-model fairness guarantees and the two-phase sweep."""

import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from ML.CFG import CFG
from ML.experiment_matrix import (
    MODEL_TYPES,
    _completed_report_path,
    _configure_run,
    build_experiment_matrix,
    run_experiment_matrix,
)
from ML.selection import (
    _build_configured_model,
    _select_hyperparameters_on_validation,
    resolve_scale_pos_weight,
)
from ML.summary_table import (
    full_row_plan,
    FEATURE_COMBINATION_ORDER,
    PAPER_ORDER,
    build_table,
    read_summaries,
    render_tsv,
)


IMBALANCED_LABELS = np.r_[np.zeros(10474, dtype=int), np.ones(3884, dtype=int)]


class ScalePosWeightTests(unittest.TestCase):
    def test_balanced_matches_class_weight_balanced_ratio(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.validate()

        self.assertAlmostEqual(
            resolve_scale_pos_weight(config, IMBALANCED_LABELS),
            10474 / 3884,
            places=9,
        )

    def test_balanced_resolves_to_one_when_labels_are_already_balanced(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.validate()
        balanced = np.r_[np.zeros(500, dtype=int), np.ones(500, dtype=int)]

        self.assertEqual(resolve_scale_pos_weight(config, balanced), 1.0)

    def test_opting_out_of_imbalance_handling_disables_the_weight(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.CLASS_WEIGHT = None
        config.validate()

        self.assertEqual(resolve_scale_pos_weight(config, IMBALANCED_LABELS), 1.0)

    def test_explicit_numeric_value_is_passed_through(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.XGBOOST_SCALE_POS_WEIGHT = 4.0
        config.validate()

        self.assertEqual(resolve_scale_pos_weight(config, IMBALANCED_LABELS), 4.0)

    def test_balanced_weight_reaches_the_built_estimator(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.validate()

        model = _build_configured_model(config, IMBALANCED_LABELS)

        self.assertAlmostEqual(
            model.named_steps["classifier"].scale_pos_weight,
            10474 / 3884,
            places=9,
        )

    def test_balanced_without_labels_is_an_error_not_a_silent_one(self):
        config = CFG()
        config.MODEL_TYPE = "xgboost"
        config.validate()

        with self.assertRaises(ValueError):
            resolve_scale_pos_weight(config, None)

    def test_non_xgboost_models_are_unaffected(self):
        config = CFG()
        config.MODEL_TYPE = "random_forest"
        config.validate()

        model = _build_configured_model(config, IMBALANCED_LABELS)

        self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")


class TuningBudgetTests(unittest.TestCase):
    def test_every_model_gets_the_same_number_of_candidates(self):
        config = CFG()

        budgets = {
            model_type: len(candidates) + 1
            for model_type, candidates in config.HYPERPARAMETER_TUNING_CANDIDATES.items()
        }

        self.assertEqual(set(budgets.values()), {config.TUNING_CANDIDATES_PER_MODEL})

    def test_unequal_budgets_are_rejected(self):
        config = CFG()
        config.HYPERPARAMETER_TUNING_CANDIDATES["svm"] = [{"SVM_C": 0.1}]

        with self.assertRaisesRegex(ValueError, "unfair"):
            config.validate()

    def test_the_guard_can_be_disabled_deliberately(self):
        config = CFG()
        config.TUNING_CANDIDATES_PER_MODEL = None
        config.HYPERPARAMETER_TUNING_CANDIDATES["svm"] = [{"SVM_C": 0.1}]

        config.validate()  # must not raise


class TwoPhaseSweepTests(unittest.TestCase):
    def test_phase_one_never_requests_the_external_sars_mode(self):
        with TemporaryDirectory() as directory:
            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "mock"},
            ) as mocked:
                with redirect_stdout(io.StringIO()):
                    rows = run_experiment_matrix(
                        directory,
                        training_mode="same_dataset",
                        model_types=("random_forest",),
                    )

        modes = {call.args[0].TRAINING_MODE for call in mocked.call_args_list}
        self.assertEqual(modes, {"same_dataset"})
        # 15 power-set + 10 contrast + 2 reconstruction + 3 published.
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["training_mode"] == "same_dataset" for row in rows))

    def test_phase_two_uses_the_external_mode(self):
        with TemporaryDirectory() as directory:
            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "mock"},
            ) as mocked:
                with redirect_stdout(io.StringIO()):
                    run_experiment_matrix(
                        directory,
                        training_mode="external_sars",
                        model_types=("xgboost",),
                    )

        modes = {call.args[0].TRAINING_MODE for call in mocked.call_args_list}
        self.assertEqual(modes, {"external_sars"})

    def test_restricting_models_scales_the_matrix(self):
        self.assertEqual(len(build_experiment_matrix(("xgboost",))), 32)
        self.assertEqual(
            len(build_experiment_matrix(("xgboost", "random_forest"))), 64
        )
        self.assertEqual(
            len(build_experiment_matrix()), 32 * len(MODEL_TYPES)
        )

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(("not_a_model",))


class ReusedWinningModelTests(unittest.TestCase):
    """The reused model must be indistinguishable from an explicit refit."""

    def _fixture(self):
        rng = np.random.default_rng(0)
        features = pd.DataFrame(rng.normal(size=(300, 6)), columns=list("abcdef"))
        labels = (features["a"] + rng.normal(scale=0.5, size=300) > 0).astype(int)
        return features, labels.to_numpy()

    def test_reused_model_predicts_identically_to_a_refit(self):
        features, labels = self._fixture()
        config = CFG()
        config.MODEL_TYPE = "random_forest"
        config.RANDOM_FOREST_TREES = 25
        config.validate()

        with redirect_stdout(io.StringIO()):
            parameters, _, reused = _select_hyperparameters_on_validation(
                config=config,
                fit_features=features.iloc[:200],
                fit_labels=labels[:200],
                validation_features=features.iloc[200:],
                validation_labels=labels[200:],
            )

        refit_config = CFG()
        refit_config.MODEL_TYPE = "random_forest"
        refit_config.RANDOM_FOREST_TREES = 25
        for name, value in parameters.items():
            setattr(refit_config, name, value)
        refit_config.HYPERPARAMETER_TUNING = False
        refit_config.validate()
        refit = _build_configured_model(refit_config, labels[:200])
        refit.fit(features.iloc[:200], labels[:200])

        self.assertIsNotNone(reused)
        reused_scores = reused.predict_proba(features)[:, 1]
        refit_scores = refit.predict_proba(features)[:, 1]
        # The fitted trees are identical; n_jobs=-1 only varies the order in
        # which their votes are summed, so differences stay at machine epsilon.
        np.testing.assert_allclose(reused_scores, refit_scores, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            (reused_scores >= 0.5).astype(int),
            (refit_scores >= 0.5).astype(int),
        )

    def test_no_model_is_returned_when_tuning_is_disabled(self):
        features, labels = self._fixture()
        config = CFG()
        config.MODEL_TYPE = "random_forest"
        config.HYPERPARAMETER_TUNING = False
        config.validate()

        parameters, comparison, reused = _select_hyperparameters_on_validation(
            config=config,
            fit_features=features.iloc[:200],
            fit_labels=labels[:200],
            validation_features=features.iloc[200:],
            validation_labels=labels[200:],
        )

        self.assertIsNone(reused)
        self.assertEqual(comparison, [])
        self.assertIn("RANDOM_FOREST_TREES", parameters)


class ResumeTests(unittest.TestCase):
    def _seed_completed_run(self, root: Path, spec, base: CFG) -> Path:
        config = _configure_run(base, spec, root)
        path = _completed_report_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"artifact_directory": "reused", "mean_validation_metrics": {"roc_auc": 0.61}}),
            encoding="utf-8",
        )
        return path

    def test_completed_runs_are_skipped_and_their_metrics_reused(self):
        base = CFG()
        base.TRAINING_MODE = "same_dataset"
        specs = build_experiment_matrix(("random_forest",))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._seed_completed_run(root, specs[0], base)

            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "fresh"},
            ) as mocked:
                with redirect_stdout(io.StringIO()):
                    rows = run_experiment_matrix(
                        root,
                        training_mode="same_dataset",
                        model_types=("random_forest",),
                        resume=True,
                    )

        self.assertEqual(mocked.call_count, len(specs) - 1)
        self.assertEqual(rows[0]["artifact_directory"], "reused")
        self.assertEqual(rows[0]["mean_validation_roc_auc"], 0.61)
        self.assertTrue(all(row["status"] == "success" for row in rows))

    def test_without_resume_everything_is_rerun(self):
        base = CFG()
        base.TRAINING_MODE = "same_dataset"
        specs = build_experiment_matrix(("random_forest",))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._seed_completed_run(root, specs[0], base)

            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "fresh"},
            ) as mocked:
                with redirect_stdout(io.StringIO()):
                    rows = run_experiment_matrix(
                        root,
                        training_mode="same_dataset",
                        model_types=("random_forest",),
                    )

        self.assertEqual(mocked.call_count, len(specs))
        self.assertEqual(rows[0]["artifact_directory"], "fresh")

    def test_truncated_report_is_treated_as_incomplete(self):
        base = CFG()
        base.TRAINING_MODE = "same_dataset"
        specs = build_experiment_matrix(("random_forest",))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._seed_completed_run(root, specs[0], base)
            path.write_text('{"artifact_directory": "trunc', encoding="utf-8")

            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "fresh"},
            ) as mocked:
                with redirect_stdout(io.StringIO()):
                    run_experiment_matrix(
                        root,
                        training_mode="same_dataset",
                        model_types=("random_forest",),
                        resume=True,
                    )

        self.assertEqual(mocked.call_count, len(specs))


class UnifiedTuningProtocolTests(unittest.TestCase):
    """Phase 1 must select hyperparameters the same way phase 2 does."""

    def _run(self):
        import pathlib
        import tempfile

        from ML.train import run_training

        config = CFG()
        config.TRAINING_MODE = "same_dataset"
        config.MODEL_TYPE = "logistic_regression"
        config.USE_AAC_FEATURES = False
        config.USE_DPC_FEATURES = False
        config.USE_PROTEIN_FEATURES = False
        config.USE_BIOLOGICAL_FEATURES = True
        with tempfile.TemporaryDirectory() as directory:
            config.OUTPUT_DIR = pathlib.Path(directory)
            with redirect_stdout(io.StringIO()):
                return run_training(config)

    def test_every_fold_uses_the_same_hyperparameters(self):
        report = self._run()

        chosen = [
            json.dumps(fold["selected_hyperparameters"], sort_keys=True)
            for fold in report["fold_metrics"]
        ]

        self.assertEqual(len(chosen), 5)
        self.assertEqual(
            len(set(chosen)),
            1,
            "folds disagree on hyperparameters; selection is not global",
        )

    def test_report_names_the_globally_selected_set(self):
        report = self._run()

        selected = report["selected_hyperparameters"]

        self.assertIsNotNone(selected)
        self.assertIn("LOGISTIC_REGRESSION_C", selected)
        self.assertEqual(
            json.dumps(selected, sort_keys=True),
            json.dumps(
                report["fold_metrics"][0]["selected_hyperparameters"],
                sort_keys=True,
            ),
        )

    def test_candidates_are_compared_once_not_once_per_fold(self):
        report = self._run()

        # Eight candidates compared globally, rather than 8 x 5 = 40 rows.
        self.assertEqual(len(report["hyperparameter_comparison"]), 8)
        self.assertTrue(
            all("parameters" in row for row in report["hyperparameter_comparison"])
        )

    def test_internal_metrics_are_still_labelled_as_selection_results(self):
        report = self._run()

        self.assertIn("pooled out-of-fold", report["hyperparameter_selection_protocol"])
        self.assertIn(
            "not an independent performance estimate",
            report["validation_metrics_interpretation"],
        )
        self.assertIsNotNone(report["hyperparameter_tuning_warning"])


class TuningMetricTests(unittest.TestCase):
    def test_matrix_no_longer_overrides_the_configured_metric(self):
        base = CFG()
        base.TRAINING_MODE = "same_dataset"
        base.HYPERPARAMETER_TUNING_METRIC = "pr_auc"

        with TemporaryDirectory() as directory:
            with patch(
                "ML.experiment_matrix.run_training",
                return_value={"artifact_directory": "mock"},
            ):
                with redirect_stdout(io.StringIO()):
                    rows = run_experiment_matrix(
                        directory,
                        base_config=base,
                        model_types=("random_forest",),
                    )

        self.assertTrue(all(row["tuning_metric"] == "pr_auc" for row in rows))

    def test_default_metric_is_locked_to_roc_auc(self):
        self.assertEqual(CFG().HYPERPARAMETER_TUNING_METRIC, "roc_auc")


class SummaryTableTests(unittest.TestCase):
    def _write_summary(self, directory: Path, rows: list[dict[str, str]]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment_summary.csv"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _summary_row(self, model, feature_set, feature_groups, status="success"):
        return {
            "experiment_id": f"{feature_set}__{model}__{feature_groups}",
            "status": status,
            "model_type": model,
            "feature_set": feature_set,
            "feature_groups": feature_groups,
            "bcell_oof_roc_auc": "0.6100",
            "bcell_oof_std_roc_auc": "0.0300",
        }

    def test_table_has_one_block_per_model_in_display_order(self):
        """Blocks follow model_order, and each block holds the same rows.

        The row set comes from what the summary covers, so this builds a
        summary with the same two configurations under both models.
        """
        summaries = {
            f"{model}-{index}": self._summary_row(model, "proposed", groups)
            for model in ("random_forest", "xgboost")
            for index, groups in enumerate(("aac,len", "prop,len"))
        }

        rows = build_table(summaries, model_order=("random_forest", "xgboost"))

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0][1], "RF")
        self.assertEqual(rows[2][1], "XGBOOST")
        self.assertEqual([row[2] for row in rows[:2]], [row[2] for row in rows[2:]])

    def test_an_empty_summary_produces_no_rows(self):
        """Nothing measured means nothing to print, not a block of blanks.

        full_row_plan names every row the table *can* show, including the
        superseded BIO/PROT combinations kept so an old summary still prints
        under its own labels. Rendering all of them unconditionally put 31
        permanently blank rows per model into the published table.
        """
        self.assertEqual(build_table({}, model_order=("random_forest",)), [])

    def test_a_failed_run_keeps_its_row_but_reports_no_metrics(self):
        summaries = {
            "a": self._summary_row(
                "random_forest", "proposed", "aac,len", status="failed"
            )
        }

        rows = build_table(summaries, model_order=("random_forest",))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "AAC + LEN")
        self.assertEqual(rows[0][3], "")
        self.assertEqual(rows[0][10], "FALSE")

    def test_internal_mean_and_std_are_combined(self):
        summaries = {
            "a": {
                "experiment_id": "a",
                "status": "success",
                "model_type": "random_forest",
                "feature_set": "proposed",
                "feature_groups": "aac,len",
                "bcell_oof_roc_auc": "0.6167",
                "bcell_oof_std_roc_auc": "0.0269",
                "external_sars_threshold_0_5_roc_auc": "0.5884",
            }
        }

        rows = build_table(summaries, model_order=("random_forest",))

        self.assertEqual(rows[0][3], "0.6167 ± 0.0269")
        self.assertEqual(rows[0][10], "TRUE")
        self.assertEqual(rows[0][11], "0.5884")

    def test_same_dataset_runs_fall_back_to_mean_validation_columns(self):
        summaries = {
            "a": {
                "experiment_id": "a",
                "status": "success",
                "model_type": "svm",
                "feature_set": "perbandingan_1",
                "feature_groups": "profile_defined",
                "mean_validation_roc_auc": "0.6100",
                "std_validation_roc_auc": "0.0300",
            }
        }

        rows = build_table(summaries, model_order=("svm",))
        # Resolved by label: the row's position depends on what the summary
        # covers, not on its index in full_row_plan.
        paper_row = next(
            row for row in rows if row[2] == "FS-1 (Cihan et al. 2022, as published for B-cell)"
        )

        self.assertEqual(paper_row[2], "FS-1 (Cihan et al. 2022, as published for B-cell)")
        self.assertEqual(paper_row[3], "0.6100 ± 0.0300")
        self.assertEqual(paper_row[10], "FALSE")

    def test_failed_runs_are_not_reported_as_results(self):
        summaries = {
            "a": {
                "experiment_id": "a",
                "status": "failed",
                "model_type": "random_forest",
                "feature_set": "proposed",
                "feature_groups": "aac,len",
                "bcell_oof_roc_auc": "0.9999",
            }
        }

        rows = build_table(summaries, model_order=("random_forest",))

        self.assertEqual(rows[0][3], "")

    def test_phase_two_summary_merges_onto_phase_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = self._write_summary(
                root / "p1",
                [
                    {
                        "experiment_id": "proposed__random_forest__aac",
                        "status": "success",
                        "model_type": "random_forest",
                        "feature_set": "proposed",
                        "feature_groups": "aac,len",
                        "bcell_oof_roc_auc": "0.6167",
                        "bcell_oof_std_roc_auc": "0.0269",
                    }
                ],
            )
            phase2 = self._write_summary(
                root / "p2",
                [
                    {
                        "experiment_id": "proposed__random_forest__aac",
                        "status": "success",
                        "model_type": "random_forest",
                        "feature_set": "proposed",
                        "feature_groups": "aac,len",
                        "external_sars_threshold_0_5_roc_auc": "0.5884",
                    }
                ],
            )

            merged = read_summaries([phase1, phase2])

        rows = build_table(merged, model_order=("random_forest",))
        self.assertEqual(rows[0][3], "0.6167 ± 0.0269")
        self.assertEqual(rows[0][11], "0.5884")

    def test_tsv_is_rectangular_and_tab_separated(self):
        rows = build_table({}, model_order=("random_forest",))

        lines = render_tsv(rows).split("\n")
        widths = {len(line.split("\t")) for line in lines}

        self.assertEqual(widths, {3 + 7 + 1 + 7})


if __name__ == "__main__":
    unittest.main()
