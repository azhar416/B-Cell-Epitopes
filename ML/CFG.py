from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    """Configuration for preprocessing, protein-group validation, and training."""

    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    EVALUATION_DATA_DIR: Path | None = None
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "ML" / "artifacts")
    MODEL_TYPE: str = "logistic_regression"
    TRAINING_MODE: str = "same_dataset"
    FEATURE_SET: str = "proposed"
    USE_PEPTIDE_FEATURES: bool = False
    USE_PROTEIN_FEATURES: bool = False
    USE_BIOLOGICAL_FEATURES: bool = True
    PEPTIDE_FEATURE_TYPE: str = "aac"
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    USE_SMOTE: bool = True
    SMOTE_K_NEIGHBORS: int = 5
    MAX_PEPTIDE_LENGTH: int = 20
    PREDICTION_THRESHOLD: float = 0.5

    # Tree models: useful sweeps include 300/500 trees, depth 8/12/None,
    # and leaf sizes 2/5/10/20.
    RANDOM_FOREST_TREES: int = 300  # Also sets n_estimators for ExtraTrees.
    TREE_MAX_DEPTH: int | None = None
    TREE_MIN_SAMPLES_LEAF: int = 2
    TREE_MIN_SAMPLES_SPLIT: int = 2
    TREE_MAX_FEATURES: str | int | float | None = "sqrt"

    # Linear/SVM: test C values 0.1/1/10; for SVM also compare gamma="scale".
    CLASS_WEIGHT: str | None = "balanced"
    LOGISTIC_REGRESSION_C: float = 1.0
    LOGISTIC_REGRESSION_MAX_ITER: int = 2000
    SVM_C: float = 1.0
    SVM_GAMMA: str | float = "scale"
    HYPERPARAMETER_TUNING: bool = False
    HYPERPARAMETER_TUNING_METRIC: str = "roc_auc"
    HYPERPARAMETER_TUNING_CANDIDATES: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {
            "logistic_regression": [
                {"LOGISTIC_REGRESSION_C": 0.1},
                {"LOGISTIC_REGRESSION_C": 10.0},
            ],
            "svm": [
                {"SVM_C": 0.1},
                {"SVM_C": 10.0},
                {"SVM_GAMMA": "auto"},
            ],
            "random_forest": [
                {"TREE_MAX_DEPTH": 8, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": 12, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 5},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 10},
                {"RANDOM_FOREST_TREES": 500},
                {"TREE_MAX_FEATURES": "log2"},
            ],
            "extra_trees": [
                {"TREE_MAX_DEPTH": 8, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": 12, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 5},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 10},
                {"RANDOM_FOREST_TREES": 500},
                {"TREE_MAX_FEATURES": "log2"},
            ],
        }
    )

    def validate(self) -> None:
        if not isinstance(self.FEATURE_SET, str):
            raise ValueError("FEATURE_SET must be 'perbandingan_1' or 'proposed'.")
        self.FEATURE_SET = self.FEATURE_SET.strip().lower().replace(" ", "_")
        if self.FEATURE_SET not in {"perbandingan_1", "proposed"}:
            raise ValueError("FEATURE_SET must be 'perbandingan_1' or 'proposed'.")
        if self.DATASET_NAME.lower() not in {"bcell", "sars"}:
            raise ValueError("DATASET_NAME must be either 'bcell' or 'sars'.")
        if self.MODEL_TYPE.lower() not in {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "svm",
        }:
            raise ValueError(
                "MODEL_TYPE must be logistic_regression, random_forest, "
                "extra_trees, or svm."
            )
        self.MODEL_TYPE = self.MODEL_TYPE.lower()
        if not isinstance(self.TRAINING_MODE, str):
            raise ValueError("TRAINING_MODE must be a string.")
        self.TRAINING_MODE = self.TRAINING_MODE.lower()
        if self.TRAINING_MODE not in {"same_dataset", "bcell_to_sars"}:
            raise ValueError(
                "TRAINING_MODE must be 'same_dataset' or 'bcell_to_sars'."
            )
        if self.TRAINING_MODE == "bcell_to_sars" and self.VALIDATION_SIZE != 0.2:
            raise ValueError(
                "bcell_to_sars mode evaluates a fixed 20% SARS holdout; "
                "VALIDATION_SIZE must be 0.2."
            )
        for name in (
            "USE_PEPTIDE_FEATURES",
            "USE_PROTEIN_FEATURES",
            "USE_BIOLOGICAL_FEATURES",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.FEATURE_SET == "proposed":
            if self.USE_PROTEIN_FEATURES and not self.USE_PEPTIDE_FEATURES:
                raise ValueError(
                    "USE_PROTEIN_FEATURES=True requires USE_PEPTIDE_FEATURES=True."
                )
            if not any(
                (
                    self.USE_PEPTIDE_FEATURES,
                    self.USE_PROTEIN_FEATURES,
                    self.USE_BIOLOGICAL_FEATURES,
                )
            ):
                raise ValueError("At least one feature group must be enabled.")
        if not isinstance(self.PEPTIDE_FEATURE_TYPE, str):
            raise ValueError("PEPTIDE_FEATURE_TYPE must be a string.")
        if self.PEPTIDE_FEATURE_TYPE.lower() not in {"aac", "aac_dipeptide"}:
            raise ValueError(
                "PEPTIDE_FEATURE_TYPE must be 'aac' or 'aac_dipeptide'."
            )
        self.PEPTIDE_FEATURE_TYPE = self.PEPTIDE_FEATURE_TYPE.lower()
        if not 0 < self.VALIDATION_SIZE < 1:
            raise ValueError("VALIDATION_SIZE must be between 0 and 1.")
        if self.SMOTE_K_NEIGHBORS < 1:
            raise ValueError("SMOTE_K_NEIGHBORS must be positive.")
        if self.MAX_PEPTIDE_LENGTH < 1:
            raise ValueError("MAX_PEPTIDE_LENGTH must be positive.")
        if not 0 < self.PREDICTION_THRESHOLD < 1:
            raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1.")
        if self.RANDOM_FOREST_TREES < 1:
            raise ValueError("RANDOM_FOREST_TREES must be positive.")
        if self.TREE_MAX_DEPTH is not None and (
            not isinstance(self.TREE_MAX_DEPTH, int)
            or isinstance(self.TREE_MAX_DEPTH, bool)
            or self.TREE_MAX_DEPTH < 1
        ):
            raise ValueError("TREE_MAX_DEPTH must be positive or None.")
        if (
            not isinstance(self.TREE_MIN_SAMPLES_LEAF, int)
            or isinstance(self.TREE_MIN_SAMPLES_LEAF, bool)
            or self.TREE_MIN_SAMPLES_LEAF < 1
        ):
            raise ValueError("TREE_MIN_SAMPLES_LEAF must be positive.")
        if (
            not isinstance(self.TREE_MIN_SAMPLES_SPLIT, int)
            or isinstance(self.TREE_MIN_SAMPLES_SPLIT, bool)
            or self.TREE_MIN_SAMPLES_SPLIT < 2
        ):
            raise ValueError("TREE_MIN_SAMPLES_SPLIT must be at least 2.")
        if isinstance(self.TREE_MAX_FEATURES, float) and not (
            0 < self.TREE_MAX_FEATURES <= 1
        ):
            raise ValueError("Float TREE_MAX_FEATURES must be in (0, 1].")
        if isinstance(self.TREE_MAX_FEATURES, int) and self.TREE_MAX_FEATURES < 1:
            raise ValueError("Integer TREE_MAX_FEATURES must be positive.")
        if isinstance(self.TREE_MAX_FEATURES, str) and self.TREE_MAX_FEATURES not in {
            "sqrt",
            "log2",
        }:
            raise ValueError("TREE_MAX_FEATURES must be 'sqrt', 'log2', or None.")
        if self.CLASS_WEIGHT not in {
            None,
            "balanced",
            "balanced_subsample",
        }:
            raise ValueError(
                "CLASS_WEIGHT must be 'balanced', 'balanced_subsample', or None."
            )
        if (
            self.CLASS_WEIGHT == "balanced_subsample"
            and self.MODEL_TYPE.lower() not in {"random_forest", "extra_trees"}
        ):
            raise ValueError(
                "CLASS_WEIGHT='balanced_subsample' is only supported by tree models."
            )
        if self.LOGISTIC_REGRESSION_C <= 0:
            raise ValueError("LOGISTIC_REGRESSION_C must be positive.")
        if self.LOGISTIC_REGRESSION_MAX_ITER < 1:
            raise ValueError("LOGISTIC_REGRESSION_MAX_ITER must be positive.")
        if self.SVM_C <= 0:
            raise ValueError("SVM_C must be positive.")
        if not isinstance(self.SVM_GAMMA, (str, float, int)):
            raise ValueError("SVM_GAMMA must be 'scale', 'auto', or a positive number.")
        if isinstance(self.SVM_GAMMA, str) and self.SVM_GAMMA not in {"scale", "auto"}:
            raise ValueError("SVM_GAMMA must be 'scale', 'auto', or a positive number.")
        if isinstance(self.SVM_GAMMA, (float, int)) and self.SVM_GAMMA <= 0:
            raise ValueError("Numeric SVM_GAMMA must be positive.")
        if not isinstance(self.HYPERPARAMETER_TUNING, bool):
            raise ValueError("HYPERPARAMETER_TUNING must be a boolean.")
        if (
            not isinstance(self.HYPERPARAMETER_TUNING_METRIC, str)
            or self.HYPERPARAMETER_TUNING_METRIC
            not in {"roc_auc", "pr_auc", "f1", "mcc"}
        ):
            raise ValueError(
                "HYPERPARAMETER_TUNING_METRIC must be roc_auc, pr_auc, f1, or mcc."
            )
        if not isinstance(self.HYPERPARAMETER_TUNING_CANDIDATES, dict):
            raise ValueError("HYPERPARAMETER_TUNING_CANDIDATES must be a mapping.")
        model_parameters = {
            "logistic_regression": {
                "LOGISTIC_REGRESSION_C",
                "LOGISTIC_REGRESSION_MAX_ITER",
            },
            "svm": {"SVM_C", "SVM_GAMMA"},
            "random_forest": {
                "RANDOM_FOREST_TREES",
                "TREE_MAX_DEPTH",
                "TREE_MIN_SAMPLES_LEAF",
                "TREE_MIN_SAMPLES_SPLIT",
                "TREE_MAX_FEATURES",
            },
            "extra_trees": {
                "RANDOM_FOREST_TREES",
                "TREE_MAX_DEPTH",
                "TREE_MIN_SAMPLES_LEAF",
                "TREE_MIN_SAMPLES_SPLIT",
                "TREE_MAX_FEATURES",
            },
        }
        for model_type, candidates in self.HYPERPARAMETER_TUNING_CANDIDATES.items():
            if model_type not in model_parameters:
                raise ValueError(
                    "Tuning candidates contain unsupported model type "
                    f"{model_type!r}."
                )
            if not isinstance(candidates, list) or not all(
                isinstance(candidate, dict) and candidate
                for candidate in candidates
            ):
                raise ValueError(
                    f"Tuning candidates for {model_type!r} must be a list of "
                    "non-empty parameter mappings."
                )
            for candidate in candidates:
                unknown_parameters = set(candidate) - model_parameters[model_type]
                if unknown_parameters:
                    raise ValueError(
                        f"Unsupported {model_type} tuning parameters: "
                        f"{sorted(unknown_parameters)}."
                    )
        if self.HYPERPARAMETER_TUNING and not self.HYPERPARAMETER_TUNING_CANDIDATES.get(
            self.MODEL_TYPE.lower()
        ):
            raise ValueError(
                "Hyperparameter tuning is enabled, but there are no candidates "
                f"for MODEL_TYPE={self.MODEL_TYPE!r}."
            )
