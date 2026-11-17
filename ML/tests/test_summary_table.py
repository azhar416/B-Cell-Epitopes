"""Tests for the internal-validation table layout."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ML.summary_table import (
    FEATURE_COMBINATION_ORDER,
    SUPERSEDED_COMBINATION_ORDER,
    METRICS,
    build_internal_table,
    internal_header,
    read_summaries,
    render_rows_tsv,
)

MODELS = ("random_forest", "xgboost", "svm", "logistic_regression", "extra_trees")


def _run(model, feature_set, feature_groups, value=0.6, std=0.02):
    row = {
        "experiment_id": f"{feature_set}__{model}__"
        + ("__".join(feature_groups) if feature_groups else "profile_defined"),
        "feature_set": feature_set,
        "model_type": model,
        "feature_groups": ",".join(feature_groups)
        if feature_groups
        else "profile_defined",
        "status": "success",
    }
    for metric, _ in METRICS:
        row[f"mean_validation_{metric}"] = f"{value:.6f}"
        row[f"std_validation_{metric}"] = f"{std:.6f}"
    return row


def _write(directory, rows):
    path = Path(directory) / "experiment_summary.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


class InternalLayoutTests(unittest.TestCase):
    def _full_sweep(self):
        rows = []
        for model in MODELS:
            for combination in FEATURE_COMBINATION_ORDER:
                rows.append(_run(model, "proposed", combination))
            for paper in ("perbandingan_1", "perbandingan_2", "perbandingan_3"):
                rows.append(_run(model, paper, None))
        return rows

    def test_header_matches_the_recorded_spreadsheet_columns(self):
        self.assertEqual(
            internal_header(),
            [
                "Model",
                "Note",
                "ROC AUC",
                "PR AUC",
                "Accuracy",
                "Recall",
                "Precision",
                "F1 Score",
                "MCC",
            ],
        )

    def test_one_row_per_run_with_the_model_repeated(self):
        with TemporaryDirectory() as directory:
            path = _write(directory, self._full_sweep())
            table = build_internal_table(read_summaries([path]))

        # 5 models x (15 ablations + 3 paper feature sets).
        self.assertEqual(len(table), 90)
        self.assertEqual(len({len(row) for row in table}), 1)
        self.assertEqual(len(table[0]), 9)
        # Unlike the comparison layout, the label is never blanked out.
        self.assertTrue(all(row[0] for row in table))
        self.assertEqual([row[0] for row in table[:18]], ["RF"] * 18)

    def test_ablations_keep_the_manuscript_order(self):
        with TemporaryDirectory() as directory:
            path = _write(directory, self._full_sweep())
            table = build_internal_table(read_summaries([path]))

        self.assertEqual(
            [row[1] for row in table[:15]],
            [
                "AAC",
                "AAC + DPC",
                "AAC + BIO",
                "AAC + FLANK",
                "AAC + DPC + BIO",
                "AAC + DPC + FLANK",
                "AAC + BIO + FLANK",
                "AAC + DPC + BIO + FLANK",
                "DPC",
                "DPC + BIO",
                "DPC + FLANK",
                "DPC + BIO + FLANK",
                "BIO",
                "BIO + FLANK",
                "FLANK",
            ],
        )

    def test_metrics_render_as_mean_plus_minus_sd(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory, [_run("svm", "proposed", ("aac",), value=0.6224, std=0.0243)]
            )
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        self.assertEqual(table[0][:3], ["SVM", "AAC", "0.6224 ± 0.0243"])

    def test_unswept_feature_sets_are_left_out_entirely(self):
        """FS-4/FS-5 have no assets, so blank rows for them would be noise."""
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    _run("svm", "proposed", combination)
                    for combination in FEATURE_COMBINATION_ORDER
                ]
                + [_run("svm", "perbandingan_1", None)],
            )
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        notes = [row[1] for row in table]
        self.assertEqual(len(table), 16)
        self.assertIn("FS-1 (Cihan et al. 2022)", notes)
        self.assertNotIn("FS-4 (Liu et al. 2024)", notes)
        self.assertNotIn("FS-5 (BLMPred 2026)", notes)

    def test_a_failed_run_keeps_its_row_but_reports_no_metrics(self):
        failed = _run("svm", "proposed", ("aac",))
        failed["status"] = "failed"
        with TemporaryDirectory() as directory:
            path = _write(directory, [failed])
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        aac_row = next(row for row in table if row[1] == "AAC")
        self.assertEqual(aac_row[2:], [""] * len(METRICS))

    def test_tsv_is_one_tab_separated_line_per_row(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory, [_run("svm", "proposed", ("aac",), value=0.5, std=0.01)]
            )
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        lines = render_rows_tsv(internal_header(), table).splitlines()

        self.assertEqual(lines[0].split("\t")[:2], ["Model", "Note"])
        self.assertEqual({len(line.split("\t")) for line in lines}, {9})
        self.assertIn("0.5000 ± 0.0100", lines[1])


class SupersededSweepTests(unittest.TestCase):
    """The phase-1 sweep that used PROT must still render from its summary."""

    def test_protein_rows_appear_only_when_the_sweep_contains_them(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    _run("svm", "proposed", combination)
                    for combination in SUPERSEDED_COMBINATION_ORDER
                ],
            )
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        notes = [row[1] for row in table]
        self.assertEqual(len(notes), len(SUPERSEDED_COMBINATION_ORDER))
        self.assertIn("PROT", notes)
        self.assertIn("AAC + DPC + BIO + PROT", notes)
        self.assertNotIn("FLANK", notes)

    def test_a_flank_sweep_shows_no_protein_rows(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    _run("svm", "proposed", combination)
                    for combination in FEATURE_COMBINATION_ORDER
                ],
            )
            table = build_internal_table(read_summaries([path]), model_order=("svm",))

        notes = [row[1] for row in table]
        self.assertEqual(len(notes), 15)
        self.assertTrue(all("PROT" not in note for note in notes))


if __name__ == "__main__":
    unittest.main()
