"""Tests for the internal-validation table layout."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ML.summary_table import (
    DEFAULT_MODEL_ORDER,
    EXTERNAL_PREFIXES,
    FEATURE_COMBINATION_ORDER,
    SUPERSEDED_COMBINATION_ORDER,
    METRICS,
    build_internal_table,
    build_table,
    internal_header,
    read_summaries,
    render_rows_tsv,
)

# Mirror the real order rather than a copy of it: build_internal_table renders
# one block per model in DEFAULT_MODEL_ORDER, so a fixture that covers fewer
# models produces blank rows and the row count stops meaning anything.
MODELS = DEFAULT_MODEL_ORDER


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
            for paper in ("perbandingan_1", "perbandingan_1b",
                          "perbandingan_2", "perbandingan_3"):
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

        # one row per model per (15 power-set + 4 paper feature sets).
        self.assertEqual(len(table), 19 * len(MODELS))
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
                "AAC + LEN",
                "AAC + PROP + LEN",
                "AAC + FLANK + LEN",
                "AAC + PROP + FLANK + LEN",
                "PROP + LEN (= FS-1b less its duplicate column)",
                "PROP + FLANK + LEN",
                "FLANK + LEN",
                "AAC + DPC + LEN",
                "AAC + DPC + PROP + LEN",
                "AAC + DPC + FLANK + LEN",
                "AAC + DPC + PROP + FLANK + LEN",
                "DPC + LEN",
                "DPC + PROP + LEN",
                "DPC + PROP + FLANK + LEN",
                "DPC + FLANK + LEN",
            ],
        )

    def test_metrics_render_as_mean_plus_minus_sd(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [_run("random_forest", "proposed", ("aac", "len"),
                      value=0.6224, std=0.0243)],
            )
            table = build_internal_table(
                read_summaries([path]), model_order=("random_forest",)
            )

        self.assertEqual(table[0][:3], ["RF", "AAC + LEN", "0.6224 ± 0.0243"])

    def test_unswept_feature_sets_are_left_out_entirely(self):
        """FS-4/FS-5 have no assets, so blank rows for them would be noise."""
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    _run("random_forest", "proposed", combination)
                    for combination in FEATURE_COMBINATION_ORDER
                ]
                + [_run("random_forest", "perbandingan_1", None)],
            )
            table = build_internal_table(read_summaries([path]), model_order=("random_forest",))

        notes = [row[1] for row in table]
        self.assertEqual(len(table), 16)
        self.assertIn("FS-1 (Cihan et al. 2022, as published for B-cell)", notes)
        self.assertNotIn("FS-4 (Liu et al. 2024)", notes)
        self.assertNotIn("FS-5 (BLMPred 2026)", notes)

    def test_a_failed_run_keeps_its_row_but_reports_no_metrics(self):
        failed = _run("random_forest", "proposed", ("aac",))
        failed["status"] = "failed"
        with TemporaryDirectory() as directory:
            path = _write(directory, [failed])
            table = build_internal_table(read_summaries([path]), model_order=("random_forest",))

        aac_row = next(row for row in table if row[1] == "AAC")
        self.assertEqual(aac_row[2:], [""] * len(METRICS))

    def test_tsv_is_one_tab_separated_line_per_row(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory, [_run("random_forest", "proposed", ("aac",), value=0.5, std=0.01)]
            )
            table = build_internal_table(read_summaries([path]), model_order=("random_forest",))

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
                    _run("random_forest", "proposed", combination)
                    for combination in SUPERSEDED_COMBINATION_ORDER
                ],
            )
            table = build_internal_table(read_summaries([path]), model_order=("random_forest",))

        notes = [row[1] for row in table]
        self.assertEqual(len(notes), len(SUPERSEDED_COMBINATION_ORDER))
        self.assertIn("PROT", notes)
        self.assertIn("AAC + DPC + BIO + PROT", notes)
        # No current power-set row can appear: every one of those carries LEN.
        self.assertTrue(all("LEN" not in note for note in notes))

    def test_a_flank_sweep_shows_no_protein_rows(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    _run("random_forest", "proposed", combination)
                    for combination in FEATURE_COMBINATION_ORDER
                ],
            )
            table = build_internal_table(read_summaries([path]), model_order=("random_forest",))

        notes = [row[1] for row in table]
        self.assertEqual(len(notes), 15)
        self.assertTrue(all("PROT" not in note for note in notes))


class PerModelFrozenRowTests(unittest.TestCase):
    """Phase 2 may freeze a different combination per classifier.

    The row plan named proposed rows only from FEATURE_COMBINATION_ORDER, the
    peptide-level power set, and named every protein-group combination under
    the kontras arm. A frozen configuration carrying a protein group is in the
    PROPOSED arm, so nothing matched it: the row was dropped and the table lost
    the configuration phase 2 was run to measure.
    """

    WINNERS = {
        "random_forest": ("aac", "prop", "flank", "len", "protid"),
        "logistic_regression": ("aac", "prop", "flank", "len", "protglobal"),
        "extra_trees": ("prop", "len", "protglobal", "protlen", "pos", "protid"),
        "xgboost": ("prop", "len", "protglobal", "protlen"),
        "mlp": ("prop", "len"),
    }
    PAPERS = ("perbandingan_1", "perbandingan_1b", "perbandingan_2",
              "perbandingan_3")

    def _phase2(self, winners):
        rows = []
        for model, combination in winners.items():
            rows.append(_run(model, "proposed", combination))
            for paper in self.PAPERS:
                rows.append(_run(model, paper, None))
        with TemporaryDirectory() as directory:
            return read_summaries([_write(directory, rows)])

    def test_a_protein_group_in_the_proposed_arm_still_gets_a_row(self):
        summaries = self._phase2({"xgboost": ("prop", "len", "protglobal", "protlen")})

        rows = build_table(summaries, model_order=("xgboost",))

        proposed = [row[2] for row in rows if not row[2].startswith("FS-")]
        self.assertEqual(proposed, ["PROP + LEN + PROTGLOBAL + PROTLEN"])

    def test_each_classifier_prints_the_combination_it_won_with(self):
        summaries = self._phase2(self.WINNERS)

        rows = build_table(summaries, model_order=tuple(self.WINNERS))

        proposed = [row[2] for row in rows if not row[2].startswith("FS-")]
        self.assertEqual(
            proposed,
            [
                "AAC + PROP + FLANK + LEN + PROTID",
                "AAC + PROP + FLANK + LEN + PROTGLOBAL",
                "PROP + LEN + PROTGLOBAL + PROTLEN + POS + PROTID",
                "PROP + LEN + PROTGLOBAL + PROTLEN",
                "PROP + LEN (= FS-1b less its duplicate column)",
            ],
        )

    def test_one_model_does_not_inherit_another_model_s_row(self):
        """The plan is per model, so no classifier prints a blank it never ran.

        With one plan shared across models, every model gained all five
        proposed rows: four blank, and all five labelled after whichever
        combination the static plan happened to name.
        """
        summaries = self._phase2(self.WINNERS)

        rows = build_table(summaries, model_order=tuple(self.WINNERS))

        self.assertEqual(len(rows), len(self.WINNERS) * (1 + len(self.PAPERS)))
        for row in rows:
            with self.subTest(note=row[2]):
                self.assertNotEqual(row[3], "", "row has no internal metrics")

    def test_group_labels_follow_manuscript_order_not_the_stored_order(self):
        """feature_groups is read back sorted, which is not reading order."""
        summaries = self._phase2({"random_forest": ("protid", "len", "prop", "aac")})

        rows = build_table(summaries, model_order=("random_forest",))

        self.assertEqual(rows[0][2], "AAC + PROP + LEN + PROTID")

    def test_one_frozen_combination_for_every_model_is_unchanged(self):
        """The consensus mode is what the pre-registered protocol describes."""
        shared = {model: ("prop", "len") for model in self.WINNERS}
        summaries = self._phase2(shared)

        rows = build_table(summaries, model_order=tuple(self.WINNERS))

        notes = [row[2] for row in rows]
        self.assertEqual(len(rows), len(shared) * (1 + len(self.PAPERS)))
        self.assertEqual(
            notes.count("PROP + LEN (= FS-1b less its duplicate column)"),
            len(shared),
        )

    def test_the_internal_layout_restricts_per_model_the_same_way(self):
        summaries = self._phase2(self.WINNERS)

        rows = build_internal_table(summaries, model_order=tuple(self.WINNERS))

        self.assertEqual(len(rows), len(self.WINNERS) * (1 + len(self.PAPERS)))
        self.assertIn("PROP + LEN + PROTGLOBAL + PROTLEN", [row[1] for row in rows])


if __name__ == "__main__":
    unittest.main()


class PublicationTableTests(unittest.TestCase):
    """build_table feeds the manuscript, so its row set has to be restricted.

    It used to call _model_entries without a plan, which falls back to
    full_row_plan(). That plan deliberately names the superseded BIO/PROT
    combinations so an old summary still prints under its own labels, so the
    published table gained 31 permanently blank rows per model -- 124 in all.
    """

    def _phase2_run(self, model, feature_set, feature_groups, external=0.55):
        row = _run(model, feature_set, feature_groups)
        for metric, _ in METRICS:
            row[f"{EXTERNAL_PREFIXES['0.5']}_{metric}"] = f"{external:.6f}"
        return row

    def _rows_for(self, summaries):
        return build_table(summaries, model_order=("random_forest",))

    def test_a_phase2_sweep_shows_only_the_configurations_it_ran(self):
        frozen = ("prop", "len")
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [self._phase2_run("random_forest", "proposed", frozen)]
                + [
                    self._phase2_run("random_forest", paper, ())
                    for paper in ("perbandingan_1", "perbandingan_2", "perbandingan_3")
                ],
            )
            rows = self._rows_for(read_summaries([path]))
        notes = [row[2] for row in rows]
        self.assertEqual(len(rows), 4, notes)
        self.assertNotIn("BIO", " ".join(notes))
        self.assertNotIn("PROT", " ".join(notes))

    def test_no_superseded_combination_is_ever_printed_blank(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [self._phase2_run("random_forest", "proposed", ("prop", "len"))],
            )
            rows = self._rows_for(read_summaries([path]))
        superseded = {
            " + ".join(group.upper() for group in combination)
            for combination in SUPERSEDED_COMBINATION_ORDER
        }
        printed = {row[2] for row in rows}
        self.assertEqual(printed & superseded, set())

    def test_every_printed_row_carries_metrics(self):
        """A restricted plan means no row resolves to None."""
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [
                    self._phase2_run("random_forest", "proposed", ("prop", "len")),
                    self._phase2_run("random_forest", "perbandingan_3", ()),
                ],
            )
            rows = self._rows_for(read_summaries([path]))
        for row in rows:
            with self.subTest(note=row[2]):
                self.assertNotEqual(row[3], "", "internal ROC AUC is blank")
                self.assertEqual(row[10], "TRUE", "external flag is not set")
                self.assertNotEqual(row[11], "", "external ROC AUC is blank")

    def test_a_phase1_only_sweep_reports_no_external_numbers(self):
        with TemporaryDirectory() as directory:
            path = _write(
                directory,
                [_run("random_forest", "proposed", ("prop", "len"))],
            )
            rows = self._rows_for(read_summaries([path]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][10], "FALSE")
        self.assertEqual(rows[0][11], "")


class ReportedThresholdTests(unittest.TestCase):
    """The published table must not mix two decision thresholds in one row.

    The internal columns are the fold mean at config.PREDICTION_THRESHOLD,
    which is 0.5; the external columns used to default to the threshold that
    maximises MCC on the B-cell out-of-fold scores. Every threshold metric --
    accuracy, recall, precision, F1, MCC -- then compared a tuned cut-off
    against an untuned one across the same row.
    """

    def _row(self, *, at_05, at_selected):
        row = {
            "experiment_id": "proposed__random_forest__prop__len",
            "status": "success",
            "model_type": "random_forest",
            "feature_set": "proposed",
            "feature_groups": "prop,len",
        }
        for metric, _ in METRICS:
            row[f"bcell_oof_{metric}"] = "0.610000"
            row[f"bcell_oof_std_{metric}"] = "0.030000"
            row[f"{EXTERNAL_PREFIXES['0.5']}_{metric}"] = f"{at_05:.6f}"
            row[f"{EXTERNAL_PREFIXES['selected']}_{metric}"] = f"{at_selected:.6f}"
        return row

    def test_the_external_block_is_reported_at_threshold_0_5(self):
        summaries = {"a": self._row(at_05=0.5100, at_selected=0.7700)}

        rows = build_table(summaries, model_order=("random_forest",))

        self.assertEqual(len(rows), 1)
        # Columns 11 onwards are the external block, one cell per metric.
        self.assertEqual(rows[0][11:], ["0.5100"] * len(METRICS))

    def test_the_selected_threshold_is_still_reachable_for_a_supplement(self):
        summaries = {"a": self._row(at_05=0.5100, at_selected=0.7700)}

        rows = build_table(
            summaries, model_order=("random_forest",), threshold="selected"
        )

        self.assertEqual(rows[0][11:], ["0.7700"] * len(METRICS))

    def test_a_summary_with_only_selected_columns_reports_no_external_numbers(self):
        """Silence is the right answer, not the tuned number under a 0.5 header.

        A phase 2 sweep from before this change carries only the selected
        columns. Falling back to them would reintroduce the mix without
        anything in the table saying so.
        """
        row = self._row(at_05=0.5100, at_selected=0.7700)
        for metric, _ in METRICS:
            del row[f"{EXTERNAL_PREFIXES['0.5']}_{metric}"]

        rows = build_table({"a": row}, model_order=("random_forest",))

        self.assertEqual(rows[0][10], "FALSE")
        self.assertEqual(rows[0][11:], [""] * len(METRICS))

    def test_an_unknown_threshold_name_is_refused(self):
        with self.assertRaises(ValueError):
            build_table({}, model_order=("random_forest",), threshold="0.4")


if __name__ == "__main__":
    unittest.main()
