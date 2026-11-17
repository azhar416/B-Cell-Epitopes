"""Predicting with a saved bundle must rebuild exactly the columns it trained on.

The failure this guards against is silent. build_feature_frame only emits the
groups it is told to, and predict_peptides then calls
`.reindex(columns=feature_names)`, which fills anything missing with NaN. The
imputer inside the pipeline replaces those with the training median, so a
bundle whose groups predict_peptides forgets to switch on still returns
plausible probabilities -- computed from the wrong inputs. Nothing raises.
"""

import unittest

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ML.experiment_matrix import (
    GROUP_SWITCHES,
    build_experiment_matrix,
)
from ML.pre_dataset import build_feature_frame
from ML.predict import predict_peptides

PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
WINDOWS = ((5, 10), (11, 16), (17, 22), (3, 8), (9, 14), (15, 20))


def frame():
    rows = []
    for index, (start, end) in enumerate(WINDOWS):
        rows.append(
            {
                "peptide_seq": PROTEIN[start - 1 : end],
                "protein_seq": PROTEIN,
                "parent_protein_id": f"P{index % 2}",
                "start_position": start,
                "end_position": end,
                "chou_fasman": 0.9 + 0.01 * index,
                "emini": 0.4 + 0.02 * index,
                "kolaskar_tongaonkar": 1.0 + 0.01 * index,
                "parker": 1.5 + 0.1 * index,
                "isoelectric_point": 7.0 + index,
                "aromaticity": 0.1 + 0.01 * index,
                "hydrophobicity": -0.2 + 0.01 * index,
                "stability": 30.0 + index,
            }
        )
    return pd.DataFrame(rows)


class PredictRoundTripTests(unittest.TestCase):
    def _bundle_for(self, groups, directory):
        """Train a tiny model on exactly these groups and save its bundle."""
        data = frame()
        kwargs = {switch.lower(): False for switch in GROUP_SWITCHES.values()}
        kwargs.update({GROUP_SWITCHES[group].lower(): True for group in groups})
        features = build_feature_frame(
            data,
            use_peptide_features=False,
            use_biological_features=False,
            use_protein_features=False,
            **kwargs,
        )
        labels = np.array([0, 1, 0, 1, 0, 1])
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=500)),
            ]
        ).fit(features, labels)
        bundle = {
            "model": model,
            "feature_names": features.columns.tolist(),
            "feature_profile": "proposed",
            "prediction_threshold": 0.5,
            "parent_protein_id_mapping": {"P0": 0, "P1": 1},
            "use_peptide_features": False,
            "use_biological_features": False,
            "use_protein_features": False,
        }
        for group, switch in GROUP_SWITCHES.items():
            bundle[switch.lower()] = group in groups
        path = directory / "peptide_classifier.joblib"
        dump(bundle, path)
        return path, features, model

    def _check(self, groups):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            from pathlib import Path

            path, features, model = self._bundle_for(groups, Path(directory))
            expected = model.predict_proba(features)[:, 1]
            data = frame()
            result = predict_peptides(
                data["peptide_seq"].tolist(),
                model_path=path,
                protein_context_data=data,
                descriptor_data=data,
            )
        np.testing.assert_allclose(
            result["probability_positive"].to_numpy(), expected, rtol=0, atol=1e-12
        )

    def test_every_group_round_trips_on_its_own(self):
        for group in GROUP_SWITCHES:
            with self.subTest(group=group):
                self._check((group,))

    def test_every_matrix_combination_round_trips(self):
        """Covers the real sweep: part 1's power set and parts 2 to 5."""
        seen = set()
        for spec in build_experiment_matrix(("logistic_regression",)):
            if spec.feature_groups is None:
                continue
            combination = tuple(spec.feature_groups)
            if combination in seen or "dpc" in combination:
                continue  # dpc is 400 columns and adds nothing to this check
            seen.add(combination)
            with self.subTest(combination=combination):
                self._check(combination)
        self.assertGreaterEqual(len(seen), 20)

    def test_a_bundle_without_the_new_flags_still_rebuilds_its_columns(self):
        """Fallbacks must infer the groups from the saved feature names."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from joblib import load

        groups = ("aac", "prop", "len")
        with TemporaryDirectory() as directory:
            path, features, model = self._bundle_for(groups, Path(directory))
            bundle = load(path)
            for switch in GROUP_SWITCHES.values():
                bundle.pop(switch.lower(), None)
            dump(bundle, path)
            expected = model.predict_proba(features)[:, 1]
            data = frame()
            result = predict_peptides(
                data["peptide_seq"].tolist(),
                model_path=path,
                protein_context_data=data,
                descriptor_data=data,
            )
        np.testing.assert_allclose(
            result["probability_positive"].to_numpy(), expected, rtol=0, atol=1e-12
        )


class PaperProfileRoundTripTests(unittest.TestCase):
    """The fixed profiles share the branch chain the group logic sits in.

    The protein-context block was an `elif` after the three profile branches
    and is now an `if`, which only stays correct because those profiles force
    every group switch off. This pins that down.
    """

    def test_each_published_profile_round_trips(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        data = frame()
        labels = np.array([0, 1, 0, 1, 0, 1])
        for profile in ("perbandingan_1", "perbandingan_2", "perbandingan_3"):
            with self.subTest(profile=profile):
                features = build_feature_frame(data, feature_set=profile)
                model = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", keep_empty_features=True
                            ),
                        ),
                        ("scaler", StandardScaler()),
                        ("classifier", LogisticRegression(max_iter=500)),
                    ]
                ).fit(features, labels)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "peptide_classifier.joblib"
                    dump(
                        {
                            "model": model,
                            "feature_names": features.columns.tolist(),
                            "feature_profile": profile,
                            "prediction_threshold": 0.5,
                            "parent_protein_id_mapping": {"P0": 0, "P1": 1},
                        },
                        path,
                    )
                    result = predict_peptides(
                        data["peptide_seq"].tolist(),
                        model_path=path,
                        protein_context_data=data,
                        descriptor_data=data,
                    )
                np.testing.assert_allclose(
                    result["probability_positive"].to_numpy(),
                    model.predict_proba(features)[:, 1],
                    rtol=0,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
