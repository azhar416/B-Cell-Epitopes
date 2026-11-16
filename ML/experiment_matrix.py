"""Run the requested feature-set, model, and proposed-feature ablation matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ML.CFG import CFG, PROJECT_ROOT
from ML.train import run_training


#: Libraries whose version can change a result. scikit-learn decides the fold
#: assignment and every estimator; numpy and pandas decide row order and
#: floating-point reductions feeding into it. Runs made under different
#: versions are not comparable, so the version is recorded per run and checked
#: for uniformity before any cross-run conclusion is drawn.
VERSIONED_LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost")

#: The environment phase 1 was run in, and therefore the one phase 2 must also
#: use. Taken from the Kaggle image both phase-1 sessions ran on at commit
#: fc2d784. It is declared here, not inferred, because Kaggle refreshes its
#: image periodically: without a fixed reference a later session could hand out
#: different folds and silently invalidate the frozen configuration.
#:
#: Measured effect of running elsewhere: on a local machine with
#: scikit-learn 1.9.1 / pandas 3.0.6 / numpy 2.5.4, the same commit, data and
#: seed gave random-forest ROC-AUC differing by up to 0.039 and PR-AUC SD by
#: 2.4x, because the StratifiedGroupKFold split came out different.
CANONICAL_ENVIRONMENT = {
    "python_version": "3.13.18",
    "numpy_version": "2.1.3",
    "pandas_version": "2.3.3",
    "sklearn_version": "1.6.1",
    "xgboost_version": "3.4.1",
}


def environment_differences(
    record: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Map each library whose version differs from canonical to (canonical, actual)."""
    record = environment_record() if record is None else record
    return {
        name: (expected, record.get(name, "absent"))
        for name, expected in CANONICAL_ENVIRONMENT.items()
        if record.get(name) != expected
    }


def report_environment(record: dict[str, str] | None = None) -> bool:
    """Print the environment against canonical. True when it matches."""
    record = environment_record() if record is None else record
    differences = environment_differences(record)
    if not differences:
        print("Environment matches the canonical phase-1 environment:")
        for name, version in record.items():
            print(f"  {name:<18} {version}")
        return True
    print(
        "WARNING: this environment is NOT the one phase 1 ran in. scikit-learn "
        "decides the fold split and pandas the row order feeding it, so results "
        "produced here are not comparable with phase 1 and must not be merged "
        "with it."
    )
    print(f"  {'library':<18} {'phase 1':<12} {'here':<12}")
    for name, (expected, actual) in differences.items():
        print(f"  {name:<18} {expected:<12} {actual:<12}")
    return False


def environment_record() -> dict[str, str]:
    """Interpreter and library versions this process is running under."""
    import platform

    record = {"python_version": platform.python_version()}
    for name in VERSIONED_LIBRARIES:
        module_name = "sklearn" if name == "scikit-learn" else name
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = "absent"
        record[f"{module_name}_version"] = str(version)
    return record


FEATURE_SETS = (
    "proposed",
    "perbandingan_1",
    "perbandingan_2",
    "perbandingan_3",
)
MODEL_TYPES = (
    "random_forest",
    "svm",
    "logistic_regression",
    "extra_trees",
    "xgboost",
)
FEATURE_GROUPS = ("aac", "dpc", "protein", "biological")


@dataclass(frozen=True)
class ExperimentSpec:
    """One immutable configuration in the complete experiment matrix."""

    feature_set: str
    model_type: str
    feature_groups: tuple[str, ...] | None = None

    @property
    def feature_group_label(self) -> str:
        if self.feature_groups is None:
            return "profile_defined"
        return "__".join(self.feature_groups)

    @property
    def experiment_id(self) -> str:
        return "__".join(
            (self.feature_set, self.model_type, self.feature_group_label)
        )


def build_experiment_matrix(
    model_types: tuple[str, ...] | None = None,
    feature_groups: tuple[str, ...] | None = None,
) -> list[ExperimentSpec]:
    """Build all proposed ablations and fixed-profile comparison runs.

    Passing feature_groups restricts the proposed arm to that single frozen
    combination, which is what phase 2 runs once a configuration is selected.
    """
    selected_models = tuple(model_types) if model_types else MODEL_TYPES
    unknown_models = [model for model in selected_models if model not in MODEL_TYPES]
    if unknown_models:
        raise ValueError(f"Unsupported model types: {sorted(unknown_models)}.")

    specs: list[ExperimentSpec] = []
    if feature_groups is None:
        proposed_combinations = [
            combination
            for size in range(1, len(FEATURE_GROUPS) + 1)
            for combination in itertools.combinations(FEATURE_GROUPS, size)
        ]
    else:
        unknown_groups = [
            group for group in feature_groups if group not in FEATURE_GROUPS
        ]
        if unknown_groups:
            raise ValueError(f"Unsupported feature groups: {sorted(unknown_groups)}.")
        if not feature_groups:
            raise ValueError("At least one feature group must be selected.")
        # Canonical order keeps the experiment_id identical to the phase-1 run.
        proposed_combinations = [
            tuple(group for group in FEATURE_GROUPS if group in set(feature_groups))
        ]

    for model_type in selected_models:
        specs.extend(
            ExperimentSpec("proposed", model_type, combination)
            for combination in proposed_combinations
        )
        specs.extend(
            ExperimentSpec(feature_set, model_type)
            for feature_set in FEATURE_SETS
            if feature_set != "proposed"
        )

    expected = len(selected_models) * (len(proposed_combinations) + len(FEATURE_SETS) - 1)
    if len(specs) != expected:
        raise RuntimeError(
            f"Expected {expected} experiment configurations; got {len(specs)}."
        )
    if len({spec.experiment_id for spec in specs}) != len(specs):
        raise RuntimeError("Experiment identifiers must be unique.")
    return specs


def _configure_run(base_config: CFG, spec: ExperimentSpec, output_root: Path) -> CFG:
    config = copy.deepcopy(base_config)
    config.FEATURE_SET = spec.feature_set
    config.MODEL_TYPE = spec.model_type
    config.OUTPUT_DIR = (
        output_root
        / spec.feature_set
        / spec.model_type
        / spec.feature_group_label
    )
    if spec.feature_set == "proposed":
        assert spec.feature_groups is not None
        active_groups = set(spec.feature_groups)
        config.USE_AAC_FEATURES = "aac" in active_groups
        config.USE_DPC_FEATURES = "dpc" in active_groups
        config.USE_PROTEIN_FEATURES = "protein" in active_groups
        config.USE_BIOLOGICAL_FEATURES = "biological" in active_groups
    return config


def _preflight(base_config: CFG, specs: list[ExperimentSpec]) -> None:
    """Validate every config and XGBoost before starting any training run."""
    for spec in specs:
        config = _configure_run(base_config, spec, base_config.OUTPUT_DIR)
        config.validate()

    try:
        importlib.import_module("xgboost")
    except Exception as error:
        raise RuntimeError(
            "The experiment matrix includes XGBoost, but importing xgboost "
            "failed. Install the ML requirements before starting the sweep."
        ) from error


def _completed_report_path(config: CFG) -> Path:
    """Where a finished run writes its top-level report."""
    profile_dir = Path(config.OUTPUT_DIR) / config.FEATURE_SET
    if config.TRAINING_MODE == "external_sars":
        return profile_dir / "external_sars" / "external_sars_metrics.json"
    return profile_dir / "cross_validation" / "cross_validation_metrics.json"


def _load_completed_report(config: CFG) -> dict[str, Any] | None:
    """Return a previous run's report, or None if it is absent or unreadable.

    A truncated JSON file means the process died mid-write, so the run is
    treated as incomplete and repeated rather than trusted.
    """
    path = _completed_report_path(config)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return report if isinstance(report, dict) else None


def _flatten_metrics(prefix: str, metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {
        f"{prefix}_{name}": value
        for name, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _summary_row(
    spec: ExperimentSpec,
    config: CFG,
    status: str,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "feature_set": spec.feature_set,
        "model_type": spec.model_type,
        "feature_groups": (
            ",".join(spec.feature_groups)
            if spec.feature_groups is not None
            else "profile_defined"
        ),
        "dataset": config.DATASET_NAME,
        "training_mode": config.TRAINING_MODE,
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "tuning_metric": (
            config.HYPERPARAMETER_TUNING_METRIC
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "random_state": config.RANDOM_STATE,
        # Recorded per run so a sweep split across machines cannot be merged
        # without the mismatch being visible.
        **environment_record(),
        "status": status,
        "output_dir": str(config.OUTPUT_DIR),
        "artifact_directory": None,
        "selected_hyperparameters": None,
        "error": error,
    }
    if report is None:
        return row

    row["artifact_directory"] = report.get("artifact_directory")
    model_selection = report.get("model_selection", {})
    if isinstance(model_selection, dict):
        row["selected_hyperparameters"] = model_selection.get("parameters")
        row["selection_metric"] = model_selection.get("selection_metric")
    if not row.get("selected_hyperparameters"):
        row["selected_hyperparameters"] = [
            fold.get("selected_hyperparameters")
            for fold in report.get("fold_metrics", [])
            if isinstance(fold, dict)
        ]
    row.update(_flatten_metrics("validation", report.get("validation_metrics")))
    row.update(
        _flatten_metrics("mean_validation", report.get("mean_validation_metrics"))
    )
    row.update(_flatten_metrics("std_validation", report.get("std_validation_metrics")))
    row.update(
        _flatten_metrics(
            "external_sars_threshold_0_5",
            report.get("external_sars_metrics_threshold_0_5"),
        )
    )
    row.update(
        _flatten_metrics(
            "external_sars_bcell_selected_threshold",
            report.get("external_sars_metrics_bcell_selected_threshold"),
        )
    )
    internal_cv = report.get("internal_bcell_cv", {})
    if isinstance(internal_cv, dict):
        row.update(
            _flatten_metrics("bcell_oof", internal_cv.get("mean_metrics"))
        )
        row.update(
            _flatten_metrics("bcell_oof_std", internal_cv.get("std_metrics"))
        )
    return row


def run_experiment_matrix(
    output_root: str | Path | None = None,
    *,
    base_config: CFG | None = None,
    training_mode: str | None = None,
    model_types: tuple[str, ...] | None = None,
    feature_groups: tuple[str, ...] | None = None,
    tuning_metric: str | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Execute the matrix and persist a progressively updated summary CSV.

    Pass training_mode="same_dataset" for phase 1 (model/feature selection on
    B-cell only, SARS never touched). Once the configuration is frozen, rerun
    the chosen specs with training_mode="external_sars" for phase 2.
    """
    # CFG.HYPERPARAMETER_TUNING_METRIC is the single source of truth; the
    # matrix no longer silently overrides it.
    base_config = CFG() if base_config is None else copy.deepcopy(base_config)
    if training_mode is not None:
        base_config.TRAINING_MODE = training_mode
    if tuning_metric is not None:
        base_config.HYPERPARAMETER_TUNING_METRIC = tuning_metric
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else PROJECT_ROOT / "ML" / "artifacts" / "experiment_matrix"
    )
    specs = build_experiment_matrix(model_types, feature_groups)
    _preflight(base_config, specs)
    # Warn rather than refuse: a deliberate robustness run in another
    # environment is legitimate. The hard gate sits where it matters, in
    # ML.select_configuration, which will not merge runs across environments.
    report_environment()
    if base_config.TRAINING_MODE == "same_dataset":
        print(
            "PHASE 1 (same_dataset): internal B-cell CV only. The SARS set is "
            "not read, so it stays a valid external test for phase 2."
        )
    else:
        print(
            "PHASE 2 (external_sars): SARS is evaluated for every run. Do not "
            "pick the winning configuration from these SARS columns; that "
            "turns the external test set into a selection set."
        )
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "experiment_summary.csv"
    rows: list[dict[str, Any]] = []

    skipped = 0
    for number, spec in enumerate(specs, start=1):
        config = _configure_run(base_config, spec, root)
        print(f"[{number}/{len(specs)}] {spec.experiment_id}")
        print(
            f"Dataset={config.DATASET_NAME}; training_mode={config.TRAINING_MODE}; "
            f"tuning_metric={config.HYPERPARAMETER_TUNING_METRIC}; "
            f"class_weight={config.CLASS_WEIGHT}; random_state={config.RANDOM_STATE}"
        )
        if resume:
            existing = _load_completed_report(config)
            if existing is not None:
                skipped += 1
                print(f"Resuming: reusing {_completed_report_path(config)}")
                rows.append(_summary_row(spec, config, "success", report=existing))
                _write_summary(summary_path, rows)
                continue
        try:
            report = run_training(config)
            row = _summary_row(spec, config, "success", report=report)
        except Exception as error:
            row = _summary_row(
                spec,
                config,
                "failed",
                error=f"{type(error).__name__}: {error}",
            )
            print(f"Run failed: {row['error']}")
        rows.append(row)
        _write_summary(summary_path, rows)

    if skipped:
        print(f"Reused {skipped} previously completed run(s).")
    return rows


def _merge_with_existing_summary(
    path: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fold ``rows`` into whatever the CSV already holds, keyed by experiment id.

    Without this, a second invocation sharing the same output directory would
    truncate the first one's results, because the file is rewritten in full on
    every run. Rows from this process win, so a rerun updates its own entries
    and leaves every other run's untouched.
    """
    if not path.exists():
        return rows
    try:
        with path.open(newline="", encoding="utf-8") as summary_file:
            existing = list(csv.DictReader(summary_file))
    except (OSError, csv.Error):
        # A corrupt or half-written summary must not block the run; the
        # per-run metrics.json files remain the authoritative record.
        return rows
    # A row without an id cannot be matched against a rerun, so it is dropped
    # rather than carried forward as an unidentifiable duplicate. This also
    # covers a file whose header is not a summary header at all.
    merged: dict[str, dict[str, Any]] = {
        str(row["experiment_id"]): row
        for row in existing
        if row.get("experiment_id")
    }
    for row in rows:
        merged[str(row["experiment_id"])] = row
    return list(merged.values())


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = _merge_with_existing_summary(path, rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 90-configuration classical ML experiment matrix."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ML" / "artifacts" / "experiment_matrix",
        help="Output root for per-run artifacts and experiment_summary.csv.",
    )
    parser.add_argument(
        "--training-mode",
        choices=("same_dataset", "external_sars"),
        default="same_dataset",
        help=(
            "same_dataset (default) runs phase 1: internal B-cell CV only, "
            "SARS untouched. external_sars runs phase 2 and evaluates SARS."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_TYPES,
        default=None,
        help="Restrict the sweep to these models (default: all five).",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=None,
        help=(
            "Run the proposed arm with only this frozen feature combination "
            "instead of all 15 ablations. Use it for phase 2, with the "
            "combination that ML.select_configuration froze."
        ),
    )
    parser.add_argument(
        "--tuning-metric",
        choices=("roc_auc", "pr_auc", "f1", "mcc"),
        default=None,
        help="Override the hyperparameter selection metric for every run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip runs that already wrote a readable report in --output-dir "
            "and reuse their metrics. Use it to continue an interrupted sweep; "
            "do NOT use it after changing the pipeline, because stale results "
            "from the old code would be kept."
        ),
    )
    args = parser.parse_args()
    rows = run_experiment_matrix(
        args.output_dir,
        training_mode=args.training_mode,
        model_types=tuple(args.models) if args.models else None,
        feature_groups=tuple(args.feature_groups) if args.feature_groups else None,
        tuning_metric=args.tuning_metric,
        resume=args.resume,
    )
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"Completed {len(rows)} runs with {failed} failures.")
    print(f"Summary: {args.output_dir.resolve() / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
