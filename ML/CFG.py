from dataclasses import dataclass, field
from pathlib import Path

from ML.pre_dataset import FLANK_WINDOW_DEFAULT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    """Configuration for preprocessing, protein-group validation, and training."""

    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    EVALUATION_DATA_DIR: Path | None = None
    BERT_EMBEDDING_DIR: Path | None = None
    PROTTRANS_EMBEDDING_DIR: Path | None = None
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "ML" / "artifacts")
    MODEL_TYPE: str = "logistic_regression"
    TRAINING_MODE: str = "external_sars"
    FEATURE_SET: str = "proposed"
    USE_PEPTIDE_FEATURES: bool = False
    # Aggregated over the whole parent protein, so every peptide of a protein
    # shares one value. Under protein-grouped folds that makes it a protein
    # identifier; kept only to reproduce that negative result, and no longer
    # part of the experiment matrix.
    USE_PROTEIN_FEATURES: bool = False
    USE_BIOLOGICAL_FEATURES: bool = True
    # --- the power-set groups: all peptide-local -----------------------------
    # Composition of the residues flanking the peptide in its parent protein.
    # Peptide-local, unlike USE_PROTEIN_FEATURES.
    USE_FLANK_FEATURES: bool = False
    # Peptide length. Its own group because it is not composition, every
    # reference feature set uses it, and on its own it reaches ROC-AUC 0.575.
    USE_LEN_FEATURES: bool = False
    # The four peptide-level propensity scales (Chou-Fasman, Emini,
    # Kolaskar-Tongaonkar, Parker).
    USE_PROP_FEATURES: bool = False

    # --- contrast groups: reported, never selected ---------------------------
    # Each of these is constant within a protein, or nearly so. They are run as
    # planned contrasts to measure what protein-level information buys under
    # protein-grouped folds, and are kept out of the selection space so the
    # frozen configuration cannot contain one.
    USE_PROTGLOBAL_FEATURES: bool = False
    USE_PROTLEN_FEATURES: bool = False
    USE_POS_FEATURES: bool = False
    USE_PROTID_FEATURES: bool = False
    FLANK_WINDOW: int = FLANK_WINDOW_DEFAULT
    # Left and right kept apart: composition inside and beside an epitope
    # correlates with the label in opposite directions for several residues,
    # and one merged block would cancel that.
    FLANK_SPLIT_SIDES: bool = True
    # None preserves the legacy behavior of USE_PEPTIDE_FEATURES and
    # PEPTIDE_FEATURE_TYPE; explicit booleans make AAC/DPC independent groups.
    USE_AAC_FEATURES: bool | None = None
    USE_DPC_FEATURES: bool | None = None
    PEPTIDE_FEATURE_TYPE: str = "aac"
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    # Neural network. It is the one classifier in the sweep that runs WITHOUT
    # imbalance handling: scikit-learn 1.6.1's MLPClassifier takes neither
    # class_weight nor sample_weight, while the other four all run under
    # CLASS_WEIGHT="balanced" (XGBoost through XGBOOST_SCALE_POS_WEIGHT). The
    # development set is 10,453 negative against 3,870 positive, so expect its
    # recall to sit below the others' and do not read that as the architecture
    # losing. The asymmetry is unavoidable here and belongs in the manuscript.
    MLP_HIDDEN_LAYER_SIZES: tuple[int, ...] = (100,)
    MLP_ALPHA: float = 1e-4
    MLP_LEARNING_RATE_INIT: float = 1e-3
    MLP_MAX_ITER: int = 500
    MLP_EARLY_STOPPING: bool = True
    USE_SMOTE: bool = False
    SMOTE_K_NEIGHBORS: int = 5
    MAX_PEPTIDE_LENGTH: int = 20
    PREDICTION_THRESHOLD: float = 0.5

    # Tree models: useful sweeps include 300/500 trees, depth 8/12/None,
    # and leaf sizes 2/5/10/20. XGBoost has a separate search space below.
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
    XGBOOST_ESTIMATORS: int = 300
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    XGBOOST_SUBSAMPLE: float = 1.0
    XGBOOST_COLSAMPLE_BYTREE: float = 1.0
    # "balanced" mirrors CLASS_WEIGHT="balanced" for the other classifiers by
    # resolving to (negatives / positives) of the rows actually fitted. Without
    # it XGBoost would be the only model facing the imbalance unweighted, which
    # makes a cross-model comparison unfair.
    XGBOOST_SCALE_POS_WEIGHT: float | str = "balanced"
    HYPERPARAMETER_TUNING: bool = True
    # Locked for the publication sweep so every run — CLI, notebook, and the
    # experiment matrix — selects candidates on the same metric.
    HYPERPARAMETER_TUNING_METRIC: str = "roc_auc"
    # Fairness requirement: every model must get the SAME number of candidates,
    # otherwise "model A beats model B" partly measures search budget. Each list
    # below holds TUNING_CANDIDATES_PER_MODEL - 1 overrides; the configured
    # defaults are always evaluated as candidate 0.
    HYPERPARAMETER_TUNING_CANDIDATES: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {
            "logistic_regression": [
                {"LOGISTIC_REGRESSION_C": 0.01},
                {"LOGISTIC_REGRESSION_C": 0.1},
                {"LOGISTIC_REGRESSION_C": 0.3},
                {"LOGISTIC_REGRESSION_C": 3.0},
                {"LOGISTIC_REGRESSION_C": 10.0},
                {"LOGISTIC_REGRESSION_C": 30.0},
                {"LOGISTIC_REGRESSION_C": 100.0},
            ],
            "svm": [
                {"SVM_C": 0.1},
                {"SVM_C": 0.3},
                {"SVM_C": 3.0},
                {"SVM_C": 10.0},
                {"SVM_C": 0.1, "SVM_GAMMA": "auto"},
                {"SVM_C": 1.0, "SVM_GAMMA": "auto"},
                {"SVM_C": 10.0, "SVM_GAMMA": "auto"},
            ],
            "random_forest": [
                {"TREE_MAX_DEPTH": 8, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": 12, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 5},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 10},
                {"RANDOM_FOREST_TREES": 500},
                {"TREE_MAX_FEATURES": "log2"},
                {
                    "RANDOM_FOREST_TREES": 500,
                    "TREE_MAX_DEPTH": 12,
                    "TREE_MIN_SAMPLES_LEAF": 5,
                },
            ],
            "extra_trees": [
                {"TREE_MAX_DEPTH": 8, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": 12, "TREE_MIN_SAMPLES_LEAF": 2},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 5},
                {"TREE_MAX_DEPTH": None, "TREE_MIN_SAMPLES_LEAF": 10},
                {"RANDOM_FOREST_TREES": 500},
                {"TREE_MAX_FEATURES": "log2"},
                {
                    "RANDOM_FOREST_TREES": 500,
                    "TREE_MAX_DEPTH": 12,
                    "TREE_MIN_SAMPLES_LEAF": 5,
                },
            ],
            "xgboost": [
                {"XGBOOST_MAX_DEPTH": 3},
                {"XGBOOST_MAX_DEPTH": 9},
                {"XGBOOST_LEARNING_RATE": 0.03},
                {"XGBOOST_LEARNING_RATE": 0.3},
                {"XGBOOST_SUBSAMPLE": 0.8, "XGBOOST_COLSAMPLE_BYTREE": 0.8},
                {"XGBOOST_MAX_DEPTH": 9, "XGBOOST_LEARNING_RATE": 0.03},
                {
                    "XGBOOST_ESTIMATORS": 500,
                    "XGBOOST_MAX_DEPTH": 3,
                    "XGBOOST_LEARNING_RATE": 0.03,
                },
            ],
            "mlp": [
                {"MLP_HIDDEN_LAYER_SIZES": (50,)},
                {"MLP_HIDDEN_LAYER_SIZES": (200,)},
                {"MLP_HIDDEN_LAYER_SIZES": (100, 50)},
                {"MLP_ALPHA": 1e-2},
                {"MLP_ALPHA": 1.0},
                {"MLP_LEARNING_RATE_INIT": 1e-2},
                {"MLP_HIDDEN_LAYER_SIZES": (200, 100), "MLP_ALPHA": 1e-2},
            ],
        }
    )
    # Set to None to allow unequal budgets (not recommended for the paper).
    TUNING_CANDIDATES_PER_MODEL: int | None = 8

    @property
    def EFFECTIVE_AAC_FEATURES(self) -> bool:
        return (
            self.USE_PEPTIDE_FEATURES
            if self.USE_AAC_FEATURES is None
            else self.USE_AAC_FEATURES
        )

    @property
    def EFFECTIVE_DPC_FEATURES(self) -> bool:
        return (
            self.USE_PEPTIDE_FEATURES
            and self.PEPTIDE_FEATURE_TYPE.lower() == "aac_dipeptide"
            if self.USE_DPC_FEATURES is None
            else self.USE_DPC_FEATURES
        )

    def validate(self) -> None:
        if not isinstance(self.FEATURE_SET, str):
            raise ValueError(
                "FEATURE_SET must be 'perbandingan_1', 'perbandingan_1b', "
                "'perbandingan_2', 'perbandingan_3', 'perbandingan_4', "
                "'perbandingan_5', 'kontras', or 'proposed'."
            )
        self.FEATURE_SET = self.FEATURE_SET.strip().lower().replace(" ", "_")
        if self.FEATURE_SET not in {
            "perbandingan_1",
            "perbandingan_1b",
            "perbandingan_2",
            "perbandingan_3",
            "perbandingan_4",
            "perbandingan_5",
            "kontras",
            "proposed",
        }:
            raise ValueError(
                "FEATURE_SET must be 'perbandingan_1', 'perbandingan_1b', "
                "'perbandingan_2', 'perbandingan_3', 'perbandingan_4', "
                "'perbandingan_5', 'kontras', or 'proposed'."
            )
        if self.DATASET_NAME.lower() not in {"bcell", "sars"}:
            raise ValueError("DATASET_NAME must be either 'bcell' or 'sars'.")
        if self.MODEL_TYPE.lower() not in {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "xgboost",
            "mlp",
            "svm",
        }:
            raise ValueError(
                "MODEL_TYPE must be logistic_regression, random_forest, "
                "extra_trees, xgboost, or svm."
            )
        self.MODEL_TYPE = self.MODEL_TYPE.lower()
        if not isinstance(self.TRAINING_MODE, str):
            raise ValueError("TRAINING_MODE must be a string.")
        self.TRAINING_MODE = self.TRAINING_MODE.lower()
        if self.TRAINING_MODE not in {
            "same_dataset",
            "external_sars",
        }:
            raise ValueError(
                "TRAINING_MODE must be 'same_dataset' or 'external_sars'."
            )
        if (
            self.TRAINING_MODE == "external_sars"
            and self.DATASET_NAME.lower() != "bcell"
        ):
            raise ValueError(
                "external_sars always trains on DATASET_NAME='bcell' and "
                "evaluates on the SARS evaluation dataset."
            )
        for name in (
            "USE_PEPTIDE_FEATURES",
            "USE_PROTEIN_FEATURES",
            "USE_BIOLOGICAL_FEATURES",
            "USE_FLANK_FEATURES",
            "FLANK_SPLIT_SIDES",
            "USE_LEN_FEATURES",
            "USE_PROP_FEATURES",
            "USE_PROTGLOBAL_FEATURES",
            "USE_PROTLEN_FEATURES",
            "USE_POS_FEATURES",
            "USE_PROTID_FEATURES",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not isinstance(self.FLANK_WINDOW, int) or isinstance(
            self.FLANK_WINDOW, bool
        ):
            raise ValueError("FLANK_WINDOW must be an integer.")
        if self.FLANK_WINDOW < 1:
            raise ValueError("FLANK_WINDOW must be positive.")
        for name in ("USE_AAC_FEATURES", "USE_DPC_FEATURES"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None.")
        if not isinstance(self.PEPTIDE_FEATURE_TYPE, str):
            raise ValueError("PEPTIDE_FEATURE_TYPE must be a string.")
        if self.PEPTIDE_FEATURE_TYPE.lower() not in {"aac", "aac_dipeptide"}:
            raise ValueError(
                "PEPTIDE_FEATURE_TYPE must be 'aac' or 'aac_dipeptide'."
            )
        self.PEPTIDE_FEATURE_TYPE = self.PEPTIDE_FEATURE_TYPE.lower()
        # kontras is driven by the same group switches as proposed, so it
        # carries the same requirement.
        if self.FEATURE_SET in {"proposed", "kontras"} and not any(
            (
                self.EFFECTIVE_AAC_FEATURES,
                self.EFFECTIVE_DPC_FEATURES,
                self.USE_PROTEIN_FEATURES,
                self.USE_BIOLOGICAL_FEATURES,
                self.USE_FLANK_FEATURES,
                self.USE_LEN_FEATURES,
                self.USE_PROP_FEATURES,
                self.USE_PROTGLOBAL_FEATURES,
                self.USE_PROTLEN_FEATURES,
                self.USE_POS_FEATURES,
                self.USE_PROTID_FEATURES,
            )
        ):
            raise ValueError("At least one feature group must be enabled.")
        if self.FEATURE_SET == "perbandingan_4" and self.USE_SMOTE:
            raise ValueError(
                "USE_SMOTE is not supported for perbandingan_4 because the "
                "BERT features are precomputed and the Liu feature profile "
                "uses its own fixed AAC/AAP/AAT representation."
            )
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
        if self.USE_SMOTE and self.CLASS_WEIGHT is not None:
            raise ValueError(
                "USE_SMOTE and CLASS_WEIGHT cannot both be active simultaneously. "
                "Choose either SMOTE oversampling or class weighting to address imbalance."
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
        if self.XGBOOST_ESTIMATORS < 1:
            raise ValueError("XGBOOST_ESTIMATORS must be positive.")
        if self.XGBOOST_MAX_DEPTH < 1:
            raise ValueError("XGBOOST_MAX_DEPTH must be positive.")
        if not 0 < self.XGBOOST_LEARNING_RATE <= 1:
            raise ValueError("XGBOOST_LEARNING_RATE must be in (0, 1].")
        if not 0 < self.XGBOOST_SUBSAMPLE <= 1:
            raise ValueError("XGBOOST_SUBSAMPLE must be in (0, 1].")
        if not 0 < self.XGBOOST_COLSAMPLE_BYTREE <= 1:
            raise ValueError("XGBOOST_COLSAMPLE_BYTREE must be in (0, 1].")
        if isinstance(self.XGBOOST_SCALE_POS_WEIGHT, str):
            if self.XGBOOST_SCALE_POS_WEIGHT != "balanced":
                raise ValueError(
                    "XGBOOST_SCALE_POS_WEIGHT must be 'balanced' or a positive number."
                )
        elif (
            not isinstance(self.XGBOOST_SCALE_POS_WEIGHT, (int, float))
            or isinstance(self.XGBOOST_SCALE_POS_WEIGHT, bool)
            or self.XGBOOST_SCALE_POS_WEIGHT <= 0
        ):
            raise ValueError(
                "XGBOOST_SCALE_POS_WEIGHT must be 'balanced' or a positive number."
            )
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
            "xgboost": {
                "XGBOOST_ESTIMATORS",
                "XGBOOST_MAX_DEPTH",
                "XGBOOST_LEARNING_RATE",
                "XGBOOST_SUBSAMPLE",
                "XGBOOST_COLSAMPLE_BYTREE",
                "XGBOOST_SCALE_POS_WEIGHT",
            },
            "mlp": {
                "MLP_HIDDEN_LAYER_SIZES",
                "MLP_ALPHA",
                "MLP_LEARNING_RATE_INIT",
                "MLP_MAX_ITER",
                "MLP_EARLY_STOPPING",
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
        if self.TUNING_CANDIDATES_PER_MODEL is not None:
            if (
                not isinstance(self.TUNING_CANDIDATES_PER_MODEL, int)
                or isinstance(self.TUNING_CANDIDATES_PER_MODEL, bool)
                or self.TUNING_CANDIDATES_PER_MODEL < 1
            ):
                raise ValueError(
                    "TUNING_CANDIDATES_PER_MODEL must be a positive integer or None."
                )
            budgets = {
                model_type: len(candidates) + 1
                for model_type, candidates in (
                    self.HYPERPARAMETER_TUNING_CANDIDATES.items()
                )
            }
            unequal = {
                model_type: budget
                for model_type, budget in budgets.items()
                if budget != self.TUNING_CANDIDATES_PER_MODEL
            }
            if unequal:
                raise ValueError(
                    "Unequal hyperparameter search budgets make a cross-model "
                    "comparison unfair. Every model must evaluate "
                    f"{self.TUNING_CANDIDATES_PER_MODEL} candidates (defaults "
                    f"plus overrides); got {unequal}. Set "
                    "TUNING_CANDIDATES_PER_MODEL=None to bypass this check."
                )
        if self.HYPERPARAMETER_TUNING and not self.HYPERPARAMETER_TUNING_CANDIDATES.get(
            self.MODEL_TYPE.lower()
        ):
            raise ValueError(
                "Hyperparameter tuning is enabled, but there are no candidates "
                f"for MODEL_TYPE={self.MODEL_TYPE!r}."
            )
