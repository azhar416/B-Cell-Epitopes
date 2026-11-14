"""Scikit-learn model and leakage-safe feature preprocessing pipelines."""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ML.liu_features import LiuBertFeatureTransformer


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
    xgboost_estimators: int = 300,
    xgboost_max_depth: int = 6,
    xgboost_learning_rate: float = 0.1,
    xgboost_subsample: float = 1.0,
    xgboost_colsample_bytree: float = 1.0,
    xgboost_scale_pos_weight: float = 1.0,
    feature_set: str = "proposed",
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
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "The xgboost classifier requires the optional 'xgboost' package."
            ) from error
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=xgboost_estimators,
            max_depth=xgboost_max_depth,
            learning_rate=xgboost_learning_rate,
            subsample=xgboost_subsample,
            colsample_bytree=xgboost_colsample_bytree,
            scale_pos_weight=xgboost_scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_type not in classifiers:
        if model_type != "svm":
            raise ValueError(
                "model_type must be logistic_regression, random_forest, "
                "extra_trees, xgboost, or svm."
            )
        svm_steps = []
        if feature_set == "perbandingan_4":
            svm_steps.append(("liu_features", LiuBertFeatureTransformer()))
        svm_steps.extend(
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
        svm_estimator = Pipeline(svm_steps)
        outer_steps = [
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
        return Pipeline(
            outer_steps
        )

    steps = []
    if feature_set == "perbandingan_4":
        steps.append(("liu_features", LiuBertFeatureTransformer()))
    steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if model_type == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifiers[model_type]))
    return Pipeline(steps)
