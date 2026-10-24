"""Scikit-learn model and leakage-safe feature preprocessing pipelines."""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(
    model_type: str = "logistic_regression",
    random_state: int = 42,
    random_forest_trees: int = 300,
) -> Pipeline:
    """Build an imputer/model pipeline; scaling is used only by linear models."""
    model_type = model_type.lower()
    classifiers = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=random_forest_trees,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=random_forest_trees,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    if model_type not in classifiers:
        if model_type != "svm":
            raise ValueError(
                "model_type must be logistic_regression, random_forest, "
                "extra_trees, or svm."
            )
        svm_estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(class_weight="balanced", random_state=random_state),
                ),
            ]
        )
        return Pipeline(
            [
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=svm_estimator,
                        method="sigmoid",
                        cv=3,
                        ensemble=False,
                    ),
                )
            ]
        )

    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_type == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifiers[model_type]))
    return Pipeline(steps)