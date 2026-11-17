"""Flank features: residues neighbouring the peptide in its parent protein.

The point of this group is that it is peptide-local. Protein-level features
are constant within a protein, so under protein-grouped folds they identify
the protein instead of describing the peptide; these tests pin down that
flank features do not behave that way.
"""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ML.CFG import CFG
from ML.experiment_matrix import (
    ALL_COMBINATIONS,
    ALWAYS_ON_GROUPS,
    CONTRAST_COMBINATIONS,
    FEATURE_GROUPS,
    GROUP_SWITCHES,
    PAPER_PATH_START,
    PAPER_RECONSTRUCTIONS,
    PROTEIN_GROUPS,
    REPRODUCTION_CHECK_COMBINATIONS,
    ExperimentSpec,
    _configure_run,
    build_experiment_matrix,
)
from ML.pre_dataset import FLANK_WINDOW_DEFAULT, build_feature_frame
from ML.select_configuration import read_proposed_runs


#          positions 1234567890123456789012
PROTEIN = "ACDEFGHIKLMNPQRSTVWYAC"


def frame(rows):
    """Rows whose peptide really sits at the stated coordinates."""
    return pd.DataFrame(
        [
            {
                "peptide_seq": PROTEIN[start - 1 : end],
                "protein_seq": PROTEIN,
                "start_position": start,
                "end_position": end,
            }
            for start, end in rows
        ]
    )


def flank_frame(dataframe, **kwargs):
    return build_feature_frame(
        dataframe,
        use_peptide_features=False,
        use_biological_features=False,
        use_aac_features=False,
        use_dpc_features=False,
        use_flank_features=True,
        **kwargs,
    )


class FlankExtractionTests(unittest.TestCase):
    def test_window_is_taken_from_both_sides_of_the_peptide(self):
        # FGHI at 5..8: the three residues before are CDE, after are KLM.
        row = flank_frame(frame([(5, 8)]), flank_window=3).iloc[0]

        for residue in "CDE":
            self.assertAlmostEqual(row[f"flank_left_aac_{residue}"], 1 / 3)
        for residue in "KLM":
            self.assertAlmostEqual(row[f"flank_right_aac_{residue}"], 1 / 3)
        self.assertEqual(row["flank_left_available"], 3.0)
        self.assertEqual(row["flank_right_available"], 3.0)

    def test_peptide_residues_are_never_counted_in_the_flank(self):
        """The blocks must partition the sequence, never overlap it."""
        row = flank_frame(frame([(5, 8)]), flank_window=3).iloc[0]

        for residue in "FGHI":  # the peptide itself
            self.assertEqual(row[f"flank_left_aac_{residue}"], 0.0)
            self.assertEqual(row[f"flank_right_aac_{residue}"], 0.0)

    def test_each_side_is_a_rate_over_the_residues_that_exist(self):
        row = flank_frame(frame([(5, 8)]), flank_window=3).iloc[0]

        for side in ("left", "right"):
            total = sum(
                value
                for name, value in row.items()
                if name.startswith(f"flank_{side}_aac_")
            )
            self.assertAlmostEqual(total, 1.0)

    def test_window_wider_than_the_protein_truncates_without_padding(self):
        row = flank_frame(frame([(5, 8)]), flank_window=10).iloc[0]

        # Only 4 residues precede position 5; 14 follow position 8.
        self.assertEqual(row["flank_left_available"], 4.0)
        self.assertEqual(row["flank_right_available"], 10.0)
        self.assertAlmostEqual(row["flank_left_aac_A"], 1 / 4)

    def test_peptide_at_the_protein_start_has_an_empty_left_flank(self):
        row = flank_frame(frame([(1, 4)]), flank_window=3).iloc[0]

        self.assertEqual(row["flank_left_available"], 0.0)
        self.assertEqual(row["flank_right_available"], 3.0)
        self.assertEqual(
            sum(
                value
                for name, value in row.items()
                if name.startswith("flank_left_aac_")
            ),
            0.0,
        )

    def test_peptide_at_the_protein_end_has_an_empty_right_flank(self):
        row = flank_frame(frame([(19, 22)]), flank_window=3).iloc[0]

        self.assertEqual(row["flank_right_available"], 0.0)
        self.assertEqual(row["flank_left_available"], 3.0)

    def test_available_counts_separate_an_absent_flank_from_a_present_one(self):
        """An absent flank is an all-zero block; only the counts say so."""
        rows = flank_frame(frame([(1, 4), (5, 8)]), flank_window=3)

        self.assertEqual(list(rows["flank_left_available"]), [0.0, 3.0])


class FlankLayoutTests(unittest.TestCase):
    def test_split_sides_produce_separate_left_and_right_blocks(self):
        features = flank_frame(frame([(5, 8)]), flank_window=3)

        self.assertEqual(features.shape[1], 42)  # 20 per side + 2 counts
        self.assertIn("flank_left_aac_A", features.columns)
        self.assertIn("flank_right_aac_A", features.columns)
        self.assertNotIn("flank_aac_A", features.columns)

    def test_merged_sides_produce_one_block(self):
        features = flank_frame(
            frame([(5, 8)]), flank_window=3, flank_split_sides=False
        )

        self.assertEqual(features.shape[1], 22)  # 20 merged + 2 counts
        self.assertIn("flank_aac_A", features.columns)
        self.assertNotIn("flank_left_aac_A", features.columns)
        # One rate over both sides together: CDE + KLM is six residues.
        self.assertAlmostEqual(features.iloc[0]["flank_aac_C"], 1 / 6)

    def test_flank_columns_do_not_collide_with_peptide_aac(self):
        features = build_feature_frame(
            frame([(5, 8)]),
            use_peptide_features=False,
            use_biological_features=False,
            use_aac_features=True,
            use_dpc_features=False,
            use_flank_features=True,
            flank_window=3,
        )

        self.assertAlmostEqual(features.iloc[0]["aac_F"], 1 / 4)  # peptide FGHI
        self.assertNotIn("aac_X", features.columns)
        self.assertNotIn("peptide_length", features.columns)
        self.assertEqual(features.iloc[0]["flank_left_aac_F"], 0.0)


class FlankIsPeptideLocalTests(unittest.TestCase):
    """The reason PROT was replaced: it was constant within a protein."""

    def test_flank_varies_between_peptides_of_the_same_protein(self):
        features = flank_frame(frame([(5, 8), (10, 13)]), flank_window=3)
        first, second = features.iloc[0], features.iloc[1]
        left = [
            name for name in features.columns if name.startswith("flank_left_aac_")
        ]

        self.assertTrue(
            any(first[name] != second[name] for name in left),
            "flank features must differ between peptides of one protein, "
            "otherwise they identify the protein the way PROT did",
        )

    def test_protein_features_are_constant_within_a_protein(self):
        """Pins the contrast this group was introduced to fix."""
        features = build_feature_frame(
            frame([(5, 8), (10, 13)]),
            use_peptide_features=False,
            use_biological_features=False,
            use_aac_features=False,
            use_dpc_features=False,
            use_protein_features=True,
        )
        composition = [
            name for name in features.columns if name.startswith("protein_aac_")
        ]

        self.assertTrue(
            all(
                features.iloc[0][name] == features.iloc[1][name]
                for name in composition
            )
        )


class FlankValidationTests(unittest.TestCase):
    def test_coordinates_that_do_not_locate_the_peptide_are_rejected(self):
        dataframe = frame([(5, 8)])
        dataframe.loc[0, "start_position"] = 6  # peptide no longer sits here

        with self.assertRaises(ValueError) as error:
            flank_frame(dataframe, flank_window=3)

        self.assertIn("start_position", str(error.exception))

    def test_missing_protein_context_is_rejected(self):
        dataframe = frame([(5, 8)]).drop(columns=["protein_seq"])

        with self.assertRaises(ValueError) as error:
            flank_frame(dataframe, flank_window=3)

        self.assertIn("protein_seq", str(error.exception))

    def test_window_must_be_a_positive_integer(self):
        for window in (0, -1):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    flank_frame(frame([(5, 8)]), flank_window=window)
        with self.assertRaises(ValueError):
            flank_frame(frame([(5, 8)]), flank_window=True)


class FlankConfigTests(unittest.TestCase):
    def test_defaults_are_window_three_and_split_sides(self):
        config = CFG()

        self.assertEqual(config.FLANK_WINDOW, FLANK_WINDOW_DEFAULT)
        self.assertEqual(config.FLANK_WINDOW, 3)
        self.assertTrue(config.FLANK_SPLIT_SIDES)
        self.assertFalse(config.USE_FLANK_FEATURES)

    def test_flank_alone_satisfies_the_one_group_requirement(self):
        config = CFG()
        config.USE_AAC_FEATURES = False
        config.USE_DPC_FEATURES = False
        config.USE_PROTEIN_FEATURES = False
        config.USE_BIOLOGICAL_FEATURES = False
        config.USE_FLANK_FEATURES = True

        config.validate()  # must not raise

    def test_invalid_window_is_rejected_by_validate(self):
        config = CFG()
        config.FLANK_WINDOW = 0

        with self.assertRaises(ValueError):
            config.validate()


class MatrixShapeTests(unittest.TestCase):
    def test_flank_is_a_power_set_group_and_protein_is_gone(self):
        self.assertIn("flank", FEATURE_GROUPS)
        self.assertNotIn("protein", FEATURE_GROUPS)
        self.assertNotIn("biological", FEATURE_GROUPS)

    def test_full_sweep_is_thirty_one_configurations_per_model(self):
        specs = build_experiment_matrix()

        self.assertEqual(len(specs), 124)
        counts = {name: 0 for name in ("proposed", "kontras")}
        for spec in specs:
            if spec.feature_set in counts:
                counts[spec.feature_set] += 1
        self.assertEqual(counts, {"proposed": 60, "kontras": 52})

    def test_the_matrix_switches_flank_on_and_leaves_protein_off(self):
        configured = _configure_run(
            CFG(),
            ExperimentSpec("proposed", "random_forest", ("aac", "flank")),
            Path("matrix-output"),
        )

        self.assertTrue(configured.USE_FLANK_FEATURES)
        self.assertTrue(configured.USE_AAC_FEATURES)
        self.assertFalse(configured.USE_PROTEIN_FEATURES)
        self.assertFalse(configured.USE_BIOLOGICAL_FEATURES)

    def test_no_matrix_run_enables_protein_features(self):
        base, root = CFG(), Path("matrix-output")
        for spec in build_experiment_matrix():
            if spec.feature_set != "proposed":
                continue
            with self.subTest(experiment=spec.experiment_id):
                self.assertFalse(
                    _configure_run(base, spec, root).USE_PROTEIN_FEATURES
                )


class SupersededSummaryIsRejectedTests(unittest.TestCase):
    """A PROT sweep must not be merged into the current FLANK matrix."""

    def _summary(self, directory, feature_groups):
        path = Path(directory) / "experiment_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_id",
                    "status",
                    "feature_set",
                    "model_type",
                    "feature_groups",
                    "mean_validation_roc_auc",
                    "std_validation_roc_auc",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_id": "old",
                    "status": "success",
                    "feature_set": "proposed",
                    "model_type": "svm",
                    "feature_groups": feature_groups,
                    "mean_validation_roc_auc": "0.61",
                    "std_validation_roc_auc": "0.02",
                }
            )
        return path

    def test_protein_groups_raise_instead_of_collapsing_onto_another_run(self):
        with TemporaryDirectory() as directory:
            path = self._summary(directory, "aac,protein")

            with self.assertRaises(ValueError) as error:
                read_proposed_runs([path], metric="roc_auc")

        message = str(error.exception)
        self.assertIn("protein", message)
        self.assertIn("superseded", message)

    def test_current_groups_are_accepted(self):
        with TemporaryDirectory() as directory:
            path = self._summary(directory, "aac,flank")

            records = read_proposed_runs([path], metric="roc_auc")

        self.assertEqual(
            [record["feature_groups"] for record in records],
            [("aac", "flank")],
        )


class PowerSetIsPeptideLocalTests(unittest.TestCase):
    """The arm the selection rule reads must hold nothing protein-level."""

    def test_power_set_and_contrast_groups_are_disjoint(self):
        self.assertFalse(set(FEATURE_GROUPS) & set(PROTEIN_GROUPS))
        self.assertFalse(set(ALWAYS_ON_GROUPS) & set(PROTEIN_GROUPS))

    def test_power_set_is_the_full_four_group_enumeration(self):
        self.assertEqual(len(FEATURE_GROUPS), 4)
        self.assertEqual(len(ALL_COMBINATIONS), 15)

    def test_no_proposed_run_switches_on_a_protein_group(self):
        """DPC is peptide-level and belongs in the power set; these do not."""
        base, root = CFG(), Path("matrix-output")
        switches = {
            "protglobal": "USE_PROTGLOBAL_FEATURES",
            "protlen": "USE_PROTLEN_FEATURES",
            "pos": "USE_POS_FEATURES",
            "protid": "USE_PROTID_FEATURES",
        }
        self.assertEqual(set(switches), set(PROTEIN_GROUPS))
        for spec in build_experiment_matrix():
            if spec.feature_set != "proposed":
                continue
            configured = _configure_run(base, spec, root)
            for group, switch in switches.items():
                with self.subTest(experiment=spec.experiment_id, group=group):
                    self.assertFalse(getattr(configured, switch))

    def test_reproduction_checks_come_from_the_power_set(self):
        for combination in REPRODUCTION_CHECK_COMBINATIONS:
            with self.subTest(combination=combination):
                self.assertIn(combination, ALL_COMBINATIONS)

    def test_partial_sweep_keeps_experiment_ids_from_a_full_sweep(self):
        """A sweep split across sessions must merge on identical ids."""
        full = {
            spec.experiment_id
            for spec in build_experiment_matrix()
            if spec.feature_set == "proposed"
        }
        partial = {
            spec.experiment_id
            for spec in build_experiment_matrix(
                feature_combinations=REPRODUCTION_CHECK_COMBINATIONS,
                include_comparisons=False,
            )
        }

        self.assertTrue(partial.issubset(full))

    def test_a_restricted_sweep_leaves_out_the_contrast_arm(self):
        specs = build_experiment_matrix(
            ("xgboost",),
            feature_combinations=(("aac", "len"),),
            include_comparisons=False,
        )

        self.assertEqual([spec.feature_set for spec in specs], ["proposed"])

    def test_a_combination_given_out_of_order_is_canonicalised(self):
        specs = build_experiment_matrix(
            ("xgboost",),
            feature_combinations=(("len", "aac"),),
            include_comparisons=False,
            include_contrasts=False,
        )

        self.assertEqual(specs[0].experiment_id, "proposed__xgboost__aac__len")

    def test_both_restrictions_at_once_are_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(
                feature_groups=("aac",), feature_combinations=(("aac",),)
            )

    def test_unknown_and_empty_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_matrix(feature_combinations=(("biological",),))
        with self.assertRaises(ValueError):
            build_experiment_matrix(feature_combinations=())


class DropSupersededGroupsTests(unittest.TestCase):
    def _summary(self, directory, rows):
        path = Path(directory) / "experiment_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_id",
                    "status",
                    "feature_set",
                    "model_type",
                    "feature_groups",
                    "mean_validation_roc_auc",
                    "std_validation_roc_auc",
                ],
            )
            writer.writeheader()
            for groups in rows:
                writer.writerow(
                    {
                        "experiment_id": f"proposed__svm__{groups.replace(',', '__')}",
                        "status": "success",
                        "feature_set": "proposed",
                        "model_type": "svm",
                        "feature_groups": groups,
                        "mean_validation_roc_auc": "0.61",
                        "std_validation_roc_auc": "0.02",
                    }
                )
        return path

    def test_superseded_rows_are_skipped_and_counted(self):
        with TemporaryDirectory() as directory:
            path = self._summary(
                directory, ["aac", "aac,protein", "protein", "aac,biological"]
            )
            # "aac,biological" and the two protein rows all come from earlier
            # matrices; only the bare "aac" row survives.
            skipped = {}

            records = read_proposed_runs(
                [path], metric="roc_auc", drop_superseded_groups=True, skipped=skipped
            )

        self.assertEqual(skipped, {"protein": 2, "biological": 1})
        self.assertEqual(
            [record["feature_groups"] for record in records], [("aac",)]
        )

    def test_a_kept_row_is_never_overwritten_by_a_dropped_one(self):
        """The bug this guards: "aac,protein" collapsing onto "aac"."""
        with TemporaryDirectory() as directory:
            path = self._summary(directory, ["aac", "aac,protein"])

            records = read_proposed_runs(
                [path], metric="roc_auc", drop_superseded_groups=True
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["feature_groups"], ("aac",))
        self.assertEqual(records[0]["n_features"], 20)


class ReconstructionTests(unittest.TestCase):
    """FS-1 and FS-2 must be rebuildable from our own groups.

    This is a check on two independent code paths: the fixed profile in
    build_feature_frame and the group switches. If they disagree, one of them
    is wrong, and the whole comparison rests on them agreeing.
    """

    def _frame(self):
        protein = "ACDEFGHIKLMNPQRSTVWYAC"
        rows = []
        for start, end in ((5, 8), (10, 13)):
            rows.append(
                {
                    "peptide_seq": protein[start - 1 : end],
                    "protein_seq": protein,
                    "parent_protein_id": "P1",
                    "start_position": start,
                    "end_position": end,
                    "chou_fasman": 1.0,
                    "emini": 0.5,
                    "kolaskar_tongaonkar": 1.1,
                    "parker": 2.0,
                    "isoelectric_point": 7.0,
                    "aromaticity": 0.1,
                    "hydrophobicity": -0.2,
                    "stability": 30.0,
                }
            )
        return pd.DataFrame(rows)

    def _by_groups(self, groups):
        switches = {
            group: switch.lower() for group, switch in GROUP_SWITCHES.items()
        }
        kwargs = {name: False for name in switches.values()}
        kwargs.update({switches[group]: True for group in groups})
        return build_feature_frame(
            self._frame(),
            use_peptide_features=False,
            use_biological_features=False,
            use_protein_features=False,
            **kwargs,
        )

    def test_fs2_is_exactly_prop_plus_len_plus_protglobal(self):
        published = build_feature_frame(self._frame(), feature_set="perbandingan_2")
        rebuilt = self._by_groups(PAPER_RECONSTRUCTIONS["perbandingan_2"])

        self.assertEqual(set(published.columns), set(rebuilt.columns))
        for column in published.columns:
            with self.subTest(column=column):
                self.assertTrue(
                    np.allclose(
                        published[column].to_numpy(float),
                        rebuilt[column].to_numpy(float),
                        equal_nan=True,
                    )
                )

    def test_fs1_rebuild_differs_only_by_its_duplicated_column(self):
        published = build_feature_frame(self._frame(), feature_set="perbandingan_1")
        rebuilt = self._by_groups(PAPER_RECONSTRUCTIONS["perbandingan_1"])

        extra = set(published.columns) - set(rebuilt.columns)
        self.assertEqual(extra, {"position"})
        self.assertFalse(set(rebuilt.columns) - set(published.columns))
        # "position" is end - start + 1, which is the peptide length again.
        self.assertTrue(
            np.allclose(
                published["position"].to_numpy(float),
                published["peptide_length"].to_numpy(float),
            )
        )
        for column in rebuilt.columns:
            with self.subTest(column=column):
                self.assertTrue(
                    np.allclose(
                        published[column].to_numpy(float),
                        rebuilt[column].to_numpy(float),
                        equal_nan=True,
                    )
                )

    def test_fs3_is_not_reconstructible(self):
        """RFE kept subsets of PROP and PROTGLOBAL, so no union of groups fits."""
        self.assertNotIn("perbandingan_3", PAPER_RECONSTRUCTIONS)


class ContrastGroupTests(unittest.TestCase):
    def test_protid_sends_unseen_proteins_to_minus_one(self):
        """On the SARS set, which shares no protein with training, every row."""
        frame = ReconstructionTests()._frame()
        features = build_feature_frame(
            frame,
            use_peptide_features=False,
            use_biological_features=False,
            use_protein_features=False,
            use_protid_features=True,
            parent_protein_id_mapping={"SOMETHING_ELSE": 0},
        )

        self.assertEqual(list(features["parent_protein_id"]), [-1, -1])

    def test_protglobal_is_constant_within_a_protein(self):
        features = build_feature_frame(
            ReconstructionTests()._frame(),
            use_peptide_features=False,
            use_biological_features=False,
            use_protein_features=False,
            use_protglobal_features=True,
        )

        self.assertEqual(features.shape[1], 4)
        for column in features.columns:
            with self.subTest(column=column):
                self.assertEqual(features[column].nunique(), 1)


class SixPartStructureTests(unittest.TestCase):
    """The run plan: 15 selectable, 13 reported, 3 published, per model."""

    def test_every_power_set_combination_holds_the_always_on_groups(self):
        for combination in ALL_COMBINATIONS:
            with self.subTest(combination=combination):
                self.assertTrue(set(ALWAYS_ON_GROUPS).issubset(combination))
        self.assertEqual(len(ALL_COMBINATIONS), 15)

    def test_every_matrix_run_switches_len_on_except_part_five(self):
        base, root = CFG(), Path("matrix-output")
        without_len = []
        for spec in build_experiment_matrix(("xgboost",)):
            if spec.feature_groups is None:
                continue  # a fixed paper profile, which defines its own columns
            if not _configure_run(base, spec, root).USE_LEN_FEATURES:
                without_len.append(spec.feature_groups)
        # Part 4 runs each protein group bare, so its raw signal is not
        # mixed with peptide length, and part 5 drops LEN to measure it.
        self.assertEqual(
            without_len,
            [(group,) for group in PROTEIN_GROUPS] + [("aac", "prop", "flank")],
        )

    def test_part_two_is_a_cumulative_ladder(self):
        rungs = [c for c in CONTRAST_COMBINATIONS if set(c) >= set(PAPER_PATH_START)
                 and not set(c) & {"aac", "flank"}]
        self.assertEqual(len(rungs), len(PROTEIN_GROUPS))
        for earlier, later in zip(rungs, rungs[1:]):
            with self.subTest(rung=later):
                self.assertTrue(set(earlier).issubset(later))
                self.assertEqual(len(set(later) - set(earlier)), 1)

    def test_contrast_arm_never_enters_selection(self):
        selectable = {
            tuple(sorted(spec.feature_groups))
            for spec in build_experiment_matrix()
            if spec.feature_set == "proposed"
        }
        reported = {tuple(sorted(c)) for c in CONTRAST_COMBINATIONS}
        self.assertFalse(selectable & reported)

    def test_no_protein_group_can_be_frozen_for_phase_two(self):
        for combination in ALL_COMBINATIONS:
            with self.subTest(combination=combination):
                self.assertFalse(set(combination) & set(PROTEIN_GROUPS))

    def test_contrast_backbone_excludes_dpc_so_one_column_is_detectable(self):
        """A 1-column contrast against 467 columns would measure nothing."""
        from ML.select_configuration import feature_count
        from ML.experiment_matrix import CONTRAST_BACKBONE

        self.assertNotIn("dpc", CONTRAST_BACKBONE)
        self.assertLess(feature_count(CONTRAST_BACKBONE), 100)


if __name__ == "__main__":
    unittest.main()
