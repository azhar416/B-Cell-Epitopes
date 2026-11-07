"""Classical classifiers for frozen ProtT5 sequence embeddings."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from EmbeddingML.CFG import CFG


def build_model(config: CFG) -> Pipeline:
    if config.MODEL_TYPE == "svm":
        classifier = CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf",
                C=config.SVM_C,
                gamma=config.SVM_GAMMA,
                class_weight=config.SVM_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
            ),
            method="sigmoid",
            cv=5,
            ensemble=False,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    elif config.MODEL_TYPE == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_TREES,
            max_depth=config.RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=config.RANDOM_FOREST_MIN_SAMPLES_LEAF,
            class_weight=config.RANDOM_FOREST_CLASS_WEIGHT,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", classifier),
        ]
    else:
        raise ValueError("MODEL_TYPE must be 'svm' or 'random_forest'.")
    return Pipeline(steps)