"""Feature transformation used by the Liu et al. LBCE-BERT comparison."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ML.pre_dataset import (
    AMINO_ACIDS,
    BERT_CLS_PREFIX,
    get_bert_cls_columns,
    normalize_sequence,
)


FEATURE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "perbandingan_4"
AAC_FEATURE_COLUMNS = tuple(f"aac_{amino_acid}" for amino_acid in AMINO_ACIDS)
LIU_FEATURE_COLUMNS = (
    *AAC_FEATURE_COLUMNS,
    "aap_mean",
    "aat_mean",
)


def _read_scale_file(path: Path, k: int) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Liu feature scale file not found: {path}")
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.split()
            if len(fields) != 2 or len(fields[0]) < k:
                raise ValueError(
                    f"Invalid scale entry at {path}:{line_number}."
                )
            result[fields[0][:k]] = float(fields[1])
    expected_size = len(AMINO_ACIDS) ** k
    if len(result) != expected_size:
        raise ValueError(
            f"Expected {expected_size} unique {k}-mer scores in {path}; "
            f"found {len(result)}."
        )
    return result


class LiuBertFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create AAC, mean AAP/AAT scores, and the 130-value BERT CLS vector."""

    def __init__(self, scale_dir: str | Path = FEATURE_ASSET_DIR):
        self.scale_dir = str(scale_dir)

    def fit(self, X: pd.DataFrame, y=None):
        self.aap_scale_ = _read_scale_file(
            Path(self.scale_dir) / "aap-general.txt.normal", k=2
        )
        self.aat_scale_ = _read_scale_file(
            Path(self.scale_dir) / "aat-general.txt.normal", k=3
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.bert_columns_ = get_bert_cls_columns(X.columns)
        if not self.bert_columns_:
            raise ValueError(
                "The Liu feature transformer needs precomputed BERT CLS "
                "columns named bert_cls_000, bert_cls_001, and so on."
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        required = {"peptide_seq", *self.bert_columns_}
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(
                "perbandingan_4 input is missing columns: "
                f"{sorted(missing)}."
            )

        sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in X["peptide_seq"].items()
        ]
        aac = np.zeros((len(sequences), len(AMINO_ACIDS)), dtype=float)
        aap_mean = np.zeros(len(sequences), dtype=float)
        aat_mean = np.zeros(len(sequences), dtype=float)
        aa_indices = {amino_acid: i for i, amino_acid in enumerate(AMINO_ACIDS)}
        for row, sequence in enumerate(sequences):
            counts = Counter(sequence)
            length = len(sequence)
            for amino_acid, column in aa_indices.items():
                aac[row, column] = counts[amino_acid] / length

            pairs = [sequence[i : i + 2] for i in range(length - 1)]
            trimers = [sequence[i : i + 3] for i in range(length - 2)]
            if pairs:
                aap_mean[row] = float(
                    np.mean([self.aap_scale_.get(pair, -1.0) for pair in pairs])
                )
            if trimers:
                aat_mean[row] = float(
                    np.mean([self.aat_scale_.get(trimer, -1.0) for trimer in trimers])
                )

        bert = X.loc[:, self.bert_columns_].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=float)
        features = np.column_stack((aac, aap_mean, aat_mean, bert))
        if not np.isfinite(features).all():
            raise ValueError("perbandingan_4 features contain non-finite values.")
        return features

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            (*LIU_FEATURE_COLUMNS, *self.bert_columns_), dtype=object
        )
