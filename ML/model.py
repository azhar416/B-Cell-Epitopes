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
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 2,
    tree_min_samples_split: int = 2,
    tree_max_features: str | int | float | None = "sqrt",
    class_weight: str | None = "balanced",
    logistic_regression_c: float = 1.0,
    logistic_regression_max_iter: int = 2000,
    svm_c: float = 1.0,
    svm_gamma: str | float = "scale",
) -> Pipeline:
    """Build an imputer/model pipeline; scaling is used only by linear models."""
    model_type = model_type.lower()
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=logistic_regression_c,
            class_weight=class_weight,
            max_iter=logistic_regression_max_iter,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=random_forest_trees,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            min_samples_split=tree_min_samples_split,
            max_features=tree_max_features,
            class_weight=(
                "balanced_subsample"
                if class_weight == "balanced_subsample"
                else class_weight
            ),
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=random_forest_trees,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            min_samples_split=tree_min_samples_split,
            max_features=tree_max_features,
            class_weight=class_weight,
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
                    SVC(
                        C=svm_c,
                        gamma=svm_gamma,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
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