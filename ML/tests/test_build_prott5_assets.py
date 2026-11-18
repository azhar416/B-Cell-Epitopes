import shutil
import tempfile
import unittest
from pathlib import Path

from ML.build_prott5_assets import peptide_list
from ML.dataset import load_labeled_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


class LabelFreePeptideListTests(unittest.TestCase):
    """Phase 1 builds the SARS vectors without opening the SARS labels.

    The cache is keyed by peptide and the loader left-merges it, so the list
    only has to COVER the cleaned peptides; it does not have to equal them.
    That is what lets the label column stay shut.
    """

    def test_label_free_covers_every_cleaned_peptide(self):
        for dataset_name in ("bcell", "sars"):
            with self.subTest(dataset=dataset_name):
                free = set(peptide_list(dataset_name, DATA_DIR, label_free=True))
                cleaned = set(
                    load_labeled_dataset(dataset_name, DATA_DIR)["peptide_seq"]
                )
                self.assertTrue(
                    cleaned <= free,
                    f"{len(cleaned - free)} cleaned peptides would have no vector",
                )

    def test_label_free_returns_each_peptide_once(self):
        # The loader rejects a cache with duplicate keys, so this is the
        # difference between a usable file and a run that dies on load.
        for dataset_name in ("bcell", "sars"):
            with self.subTest(dataset=dataset_name):
                sequences = peptide_list(dataset_name, DATA_DIR, label_free=True)
                self.assertEqual(len(sequences), len(set(sequences)))

    def test_overlength_peptides_are_not_embedded(self):
        # SARS carries a 393-residue entry that cleaning discards. Encoding it
        # would pad its whole batch out to 393 for nothing.
        sequences = peptide_list("sars", DATA_DIR, label_free=True)
        self.assertLessEqual(max(len(s) for s in sequences), 20)

    def test_the_limit_is_honoured(self):
        sequences = peptide_list(
            "bcell", DATA_DIR, label_free=True, max_peptide_length=10
        )
        self.assertLessEqual(max(len(s) for s in sequences), 10)

    def test_the_labeled_path_matches_the_cleaned_frame(self):
        sequences = peptide_list("sars", DATA_DIR, label_free=False)
        cleaned = load_labeled_dataset("sars", DATA_DIR)["peptide_seq"].tolist()
        self.assertEqual(sequences, cleaned)



class DevelopmentSetExcludesSarsTests(unittest.TestCase):
    """The external claim rests on the development set sharing no peptide.

    Cleaning removes the overlap, so this is the test that would catch it
    coming back. It belongs here rather than in the external arm's tests:
    the point of removing it in the loader is that phase 1 and phase 2 share
    one development set, so phase 1 must see the removal too.
    """

    def test_no_cleaned_bcell_peptide_appears_in_sars(self):
        bcell = load_labeled_dataset("bcell", DATA_DIR)
        sars = load_labeled_dataset("sars", DATA_DIR)
        shared = set(bcell["peptide_seq"]) & set(sars["peptide_seq"])
        self.assertEqual(shared, set())

    def test_sars_is_kept_whole(self):
        # The overlap leaves the training side, never the test side: SARS is
        # one protein and already the weakest part of the evidence.
        sars = load_labeled_dataset("sars", DATA_DIR)
        self.assertEqual(len(sars), 467)

    def test_every_protein_survives_the_removal(self):
        # A protein wiped out entirely would change the grouped fold split
        # for reasons unrelated to the features.
        bcell = load_labeled_dataset("bcell", DATA_DIR)
        self.assertEqual(bcell["parent_protein_id"].nunique(), 758)

    def test_both_classes_survive_the_removal(self):
        bcell = load_labeled_dataset("bcell", DATA_DIR)
        self.assertEqual(sorted(bcell["target"].unique()), [0, 1])

    def test_loading_bcell_without_sars_is_refused(self):
        # Silently skipping the removal would put leakage back without a
        # word, so a missing SARS file has to be an error.
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(DATA_DIR / "input_bcell.csv", Path(directory))
            with self.assertRaises(FileNotFoundError):
                load_labeled_dataset("bcell", Path(directory))

if __name__ == "__main__":
    unittest.main()
