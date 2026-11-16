"""Tests for flattening a sweep's artifacts into long-format tables."""

import csv
import gzip
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ML.collect_artifacts import (
    collect_all,
    feature_group_label,
    internal_cv_directory,
    read_runs,
    run_directory,
)

METRICS = {"roc_auc": 0.61, "mcc": 0.17, "accuracy": 0.55}


def _fold(number, offset=0.0):
    return {
        "fold": number,
        "train_size": 11492,
        "validation_size": 2866,
        "training_metrics": {k: v + 0.03 for k, v in METRICS.items()},
        "validation_metrics": {k: v + offset for k, v in METRICS.items()},
        "selected_hyperparameters": {"C": 100.0},
        "hyperparameter_comparison": [],
        "artifact_directory": "ignored",
    }


def _internal_report():
    return {
        "split_strategy": "stratified_protein_group_5_fold",
        "n_splits": 5,
        "hyperparameter_tuning": True,
        "hyperparameter_tuning_metric": "roc_auc",
        "hyperparameter_selection_protocol": "global",
        "selected_hyperparameters": {"C": 100.0},
        "n_examples": 14358,
        "n_parent_proteins": 758,
        "validation_metrics_interpretation": "selection results",
        "hyperparameter_tuning_warning": "not independent",
        "fold_metrics": [_fold(n, offset=n / 1000) for n in range(1, 6)],
        "mean_validation_metrics": METRICS,
        "std_validation_metrics": {k: 0.02 for k in METRICS},
        "mean_training_metrics": METRICS,
        "std_training_metrics": {k: 0.01 for k in METRICS},
        "validation_metrics": METRICS,
        "hyperparameter_comparison": [
            {
                "candidate_id": index,
                "parameters": {"C": float(index)},
                "selection_metric": "roc_auc",
                "selection_score": 0.60 + index / 1000,
                "selected": index == 7,
                "oof_roc_auc": 0.60 + index / 1000,
                "oof_mcc": 0.10,
            }
            for index in range(8)
        ],
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _oof_rows(count=4):
    return [
        {
            "peptide_seq": f"PEPT{index}",
            "target": index % 2,
            "parent_protein_id": "P1",
            "validation_probability": 0.4 + index / 100,
            "fold": index % 5 + 1,
        }
        for index in range(count)
    ]


class SweepBuilder:
    """Build the directory layout run_experiment_matrix produces."""

    def __init__(self, root: Path):
        self.root = root
        self.rows: list[dict] = []

    def add(self, model, feature_set, groups, mode, status="success"):
        label = feature_group_label(",".join(groups) if groups else None)
        self.rows.append(
            {
                "experiment_id": f"{feature_set}__{model}__{label}",
                "feature_set": feature_set,
                "model_type": model,
                "feature_groups": ",".join(groups) if groups else "profile_defined",
                "training_mode": mode,
                "status": status,
            }
        )
        if status != "success":
            return self
        run = self.root / feature_set / model / label / feature_set
        if mode == "same_dataset":
            internal = run / "cross_validation"
            _write_json(internal / "cross_validation_metrics.json", _internal_report())
        else:
            external = run / "external_sars"
            internal = external / "internal_bcell_same_dataset" / feature_set / "cross_validation"
            _write_json(internal / "cross_validation_metrics.json", _internal_report())
            _write_json(
                external / "external_sars_metrics.json",
                {
                    "model_selection": {
                        "parameters": {"C": 100.0},
                        "selection_metric": "roc_auc",
                        "selection_score": 0.62,
                        "candidate_count": 8,
                    },
                    "sars_labels_used_for_model_or_threshold_selection": False,
                    "bcell_rows_used_for_development": 14358,
                    "external_sars_rows": 467,
                    "external_sars_proteins": 1,
                    "exact_sequence_overlap_audit": {
                        "unique_exact_peptide_sequences": 35,
                        "same_label": 31,
                    },
                    "internal_bcell_cv": {
                        "selected_threshold": 0.4578,
                        "threshold_selection_mcc": 0.1478,
                        "threshold_selection_source": "pooled OOF",
                        "folds_at_selected_threshold": [
                            {"fold": n, **{k: v + 0.05 for k, v in METRICS.items()}}
                            for n in range(1, 6)
                        ],
                        "mean_metrics_at_selected_threshold": METRICS,
                        "std_metrics_at_selected_threshold": {k: 0.02 for k in METRICS},
                        "oof_metrics_threshold_0_5": METRICS,
                        "oof_metrics_bcell_selected_threshold": METRICS,
                    },
                    "external_sars_metrics_threshold_0_5": METRICS,
                    "external_sars_metrics_bcell_selected_threshold": METRICS,
                },
            )
            _write_csv(
                external / "external_sars_predictions.csv",
                [
                    {
                        "peptide_seq": "SARS1",
                        "target": 1,
                        "probability_positive": 0.7,
                    }
                ],
            )
        _write_csv(internal / "out_of_fold_predictions.csv", _oof_rows())
        return self

    def write_summary(self, name="experiment_summary.csv") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(key for row in self.rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        return path


class PathReconstructionTests(unittest.TestCase):
    def test_label_mirrors_the_experiment_id_component(self):
        self.assertEqual(feature_group_label("aac,biological"), "aac__biological")
        self.assertEqual(feature_group_label("profile_defined"), "profile_defined")
        self.assertEqual(feature_group_label(""), "profile_defined")
        self.assertEqual(feature_group_label(None), "profile_defined")

    def test_run_directory_is_rebuilt_relative_to_the_summary(self):
        summary = Path("/sweep/experiment_summary.csv")
        row = {
            "feature_set": "proposed",
            "model_type": "svm",
            "feature_groups": "aac,biological",
            "training_mode": "same_dataset",
        }

        self.assertEqual(
            run_directory(summary, row).as_posix().split(":")[-1].lstrip("/"),
            "sweep/proposed/svm/aac__biological/proposed/cross_validation",
        )

    def test_unknown_training_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            run_directory(Path("/s/x.csv"), {"training_mode": "nonsense"})

    def test_phase_two_nests_its_internal_cross_validation(self):
        run = Path("/sweep/proposed/svm/aac/proposed/external_sars")

        self.assertEqual(
            internal_cv_directory(run, "external_sars", "proposed"),
            run / "internal_bcell_same_dataset" / "proposed" / "cross_validation",
        )
        self.assertEqual(internal_cv_directory(run, "same_dataset", "proposed"), run)


class CollectionTests(unittest.TestCase):
    def test_failed_runs_are_not_collected(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            builder.add("svm", "proposed", ("dpc",), "same_dataset", status="failed")
            summary = builder.write_summary()

            self.assertEqual(len(read_runs([summary])), 1)

    def test_phase_one_tables_have_one_row_per_fold_and_candidate(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            for groups in (("aac",), ("dpc",), ("biological",)):
                builder.add("svm", "proposed", groups, "same_dataset")
            summary = builder.write_summary()
            out = Path(directory) / "collected"
            manifest = collect_all([summary], out)

        self.assertEqual(manifest["runs_collected"], 3)
        self.assertEqual(manifest["tables"]["runs"]["rows"], 3)
        self.assertEqual(manifest["tables"]["fold_metrics"]["rows"], 15)
        self.assertEqual(
            manifest["tables"]["hyperparameter_candidates"]["rows"], 24
        )
        self.assertEqual(manifest["tables"]["oof_predictions"]["rows"], 12)
        # Phase 1 never evaluates SARS, so that table must be absent entirely.
        self.assertNotIn("external_sars_predictions", manifest["tables"])
        self.assertFalse((out / "external_sars_predictions.csv").exists())

    def test_fold_rows_carry_training_and_validation_metrics(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            summary = builder.write_summary()
            out = Path(directory) / "collected"
            collect_all([summary], out)
            with (out / "fold_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["fold"] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertAlmostEqual(float(rows[0]["validation_roc_auc"]), 0.611)
        self.assertAlmostEqual(float(rows[0]["train_roc_auc"]), 0.64)
        self.assertEqual(json.loads(rows[0]["selected_hyperparameters"]), {"C": 100.0})

    def test_phase_two_adds_threshold_external_and_overlap_columns(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "external_sars")
            summary = builder.write_summary()
            out = Path(directory) / "collected"
            manifest = collect_all([summary], out)
            with (out / "runs.csv").open(encoding="utf-8") as handle:
                run_row = next(csv.DictReader(handle))
            with (out / "fold_metrics.csv").open(encoding="utf-8") as handle:
                fold_row = next(csv.DictReader(handle))

        self.assertEqual(manifest["tables"]["external_sars_predictions"]["rows"], 1)
        self.assertAlmostEqual(float(run_row["selected_threshold"]), 0.4578)
        self.assertEqual(run_row["sars_labels_used_for_selection"], "False")
        self.assertEqual(run_row["overlap_unique_exact_peptide_sequences"], "35")
        self.assertAlmostEqual(float(run_row["sars_at_selected_threshold_mcc"]), 0.17)
        # Phase 2 re-scores each fold at the deployment threshold.
        self.assertAlmostEqual(
            float(fold_row["validation_at_selected_threshold_mcc"]), 0.22
        )

    def test_predictions_can_be_skipped(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "external_sars")
            summary = builder.write_summary()
            out = Path(directory) / "collected"
            manifest = collect_all([summary], out, include_predictions=False)

        self.assertIn("fold_metrics", manifest["tables"])
        self.assertNotIn("oof_predictions", manifest["tables"])
        self.assertFalse((out / "oof_predictions.csv.gz").exists())

    def test_out_of_fold_table_is_written_gzipped_and_tagged_by_run(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            builder.add("random_forest", "proposed", ("aac",), "same_dataset")
            summary = builder.write_summary()
            out = Path(directory) / "collected"
            collect_all([summary], out)
            with gzip.open(out / "oof_predictions.csv.gz", "rt", encoding="utf-8") as h:
                rows = list(csv.DictReader(h))

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["model_type"] for row in rows}, {"svm", "random_forest"}
        )
        self.assertIn("validation_probability", rows[0])

    def test_a_run_missing_its_report_is_listed_not_silently_dropped(self):
        with TemporaryDirectory() as directory:
            builder = SweepBuilder(Path(directory))
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            builder.add("svm", "proposed", ("dpc",), "same_dataset")
            summary = builder.write_summary()
            # Simulate a run whose artifacts were deleted after the sweep.
            shutil.rmtree(Path(directory) / "proposed" / "svm" / "dpc")
            manifest = collect_all([summary], Path(directory) / "collected")

        self.assertEqual(manifest["runs_collected"], 1)
        self.assertEqual(
            manifest["runs_missing_report"], ["proposed__svm__dpc"]
        )

    def test_collection_survives_the_tree_being_moved(self):
        """Absolute paths recorded at run time must not be relied on."""
        with TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            builder = SweepBuilder(original)
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            builder.write_summary()

            moved = Path(directory) / "moved"
            shutil.move(str(original), str(moved))
            manifest = collect_all(
                [moved / "experiment_summary.csv"], Path(directory) / "collected"
            )

        self.assertEqual(manifest["runs_collected"], 1)
        self.assertEqual(manifest["tables"]["fold_metrics"]["rows"], 5)

    def test_later_summaries_win_so_phases_can_be_merged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            builder = SweepBuilder(root)
            builder.add("svm", "proposed", ("aac",), "same_dataset")
            phase1 = builder.write_summary("phase1.csv")

            second = SweepBuilder(root)
            second.add("svm", "proposed", ("aac",), "external_sars")
            phase2 = second.write_summary("phase2.csv")

            manifest = collect_all([phase1, phase2], root / "collected")
            with (root / "collected" / "runs.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["training_mode"], "external_sars")

    def test_missing_summary_file_is_an_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_runs([Path(directory) / "absent.csv"])


if __name__ == "__main__":
    unittest.main()
