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
from collections.abc import Iterable
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
    "perbandingan_1b",
    "perbandingan_2",
    "perbandingan_3",
)
# Every published profile the pipeline can reconstruct. FEATURE_SETS above is
# the DEFAULT sweep and deliberately stays at three: adding a name there would
# change the planned run count of every notebook that already ran, and phase
# 1's artifacts are frozen. A profile outside the default is requested by name
# through comparison_feature_sets, which runs it on its own.
COMPARISON_FEATURE_SETS = (
    "perbandingan_1",
    "perbandingan_1b",
    "perbandingan_2",
    "perbandingan_3",
    "perbandingan_4",
    "perbandingan_5",
)
DEFAULT_COMPARISON_FEATURE_SETS = tuple(
    name for name in FEATURE_SETS if name != "proposed"
)
# Feature sets whose columns come from feature_groups rather than a fixed
# profile. Only "proposed" enters the one-standard-error selection; the other
# two are reported and never selected from.
GROUPED_FEATURE_SETS = ("proposed", "kontras")
# SVM was dropped from the sweep for computational budget: on 14,323 rows its
# cost grows as n^2.3 and the calibration wrapper fits it four times per model
# fit, which put phase 1 past Kaggle's 12 h session cap. CFG and model.py still
# support it, and HYPERPARAMETER_TUNING_CANDIDATES still holds its grid, so it
# can be restored by adding it back to this tuple.
# The MLP is the one member that runs without imbalance handling; see the
# note on MLP_HIDDEN_LAYER_SIZES in ML.CFG for why, and report it.
MODEL_TYPES = (
    "random_forest",
    "logistic_regression",
    "extra_trees",
    "xgboost",
    "mlp",
)
# Every group the matrix can switch on, with the CFG flag that enables it.
# The order here is canonical: it fixes the column order of every combination
# and therefore every experiment_id.
GROUP_SWITCHES = {
    "aac": "USE_AAC_FEATURES",
    "dpc": "USE_DPC_FEATURES",
    "prop": "USE_PROP_FEATURES",
    "flank": "USE_FLANK_FEATURES",
    "len": "USE_LEN_FEATURES",
    "protglobal": "USE_PROTGLOBAL_FEATURES",
    "protlen": "USE_PROTLEN_FEATURES",
    "pos": "USE_POS_FEATURES",
    "protid": "USE_PROTID_FEATURES",
}
ALL_GROUPS = tuple(GROUP_SWITCHES)

# Part 1 -- the only arm the one-standard-error rule selects from. All four
# groups are computed from peptide_seq or from the residues immediately
# flanking it, so each is usable on a protein never seen in training.
FEATURE_GROUPS = ("aac", "dpc", "prop", "flank")
# Peptide length is a property of the sampling window rather than a
# representation of the peptide's content, and all three reference feature
# sets include it. Holding it on in every run keeps the comparison matched and
# keeps the smallest candidate at 5 columns instead of 1. Part 5 measures what
# it contributes by running the backbone without it.
ALWAYS_ON_GROUPS = ("len",)
_POWER_SET = tuple(
    combination
    for size in range(1, len(FEATURE_GROUPS) + 1)
    for combination in itertools.combinations(FEATURE_GROUPS, size)
)

# Computed from the whole protein, or from the row's place in the dataset, so
# every peptide of a protein shares the value (measured intra-class
# correlation 1.000 for protglobal, protlen and protid; 0.797 for pos). Under
# protein-grouped folds they identify the protein rather than describe the
# peptide, so they are reported and never selected from.
PROTEIN_GROUPS = ("protglobal", "protlen", "pos", "protid")
# Part 2 retraces the literature's own path. Starting from the legitimate half
# of the reference sets, the first two rungs reproduce FS-2 and FS-1 exactly,
# which makes those papers points on our ladder rather than external baselines.
PAPER_PATH_START = ("prop", "len")
# Part 3 and part 5 backbone. Fixed in advance, and deliberately without DPC:
# a one-column contrast against 467 columns would measure nothing, while
# against 67 it is detectable.
CONTRAST_BACKBONE = ("aac", "prop", "flank", "len")
# Cheapest two power-set combinations, rerun when a sweep is split across
# sessions to show the halves came from the same environment.
REPRODUCTION_CHECK_COMBINATIONS = (
    ("prop", "len"),
    ("aac", "len"),
)


def canonical(groups) -> tuple[str, ...]:
    """Order groups canonically and reject unknown names."""
    wanted = {str(group) for group in groups}
    unknown = sorted(wanted.difference(ALL_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported feature groups: {unknown}.")
    if not wanted:
        raise ValueError("At least one feature group must be selected.")
    return tuple(group for group in ALL_GROUPS if group in wanted)


# Part 1, with the always-on groups baked in so _configure_run needs no
# special case: 15 combinations, 5 to 467 columns.
ALL_COMBINATIONS = tuple(
    canonical(combination + ALWAYS_ON_GROUPS) for combination in _POWER_SET
)


def _contrast_combinations() -> list[tuple[str, ...]]:
    """Parts 2 to 5: reported, never selected from."""
    combinations: list[tuple[str, ...]] = []
    rung = PAPER_PATH_START                                   # part 2
    for group in PROTEIN_GROUPS:
        rung = rung + (group,)
        combinations.append(canonical(rung))
    for group in PROTEIN_GROUPS:                              # part 3
        combinations.append(canonical(CONTRAST_BACKBONE + (group,)))
    for group in PROTEIN_GROUPS:                              # part 4
        combinations.append(canonical((group,)))
    combinations.append(                                      # part 5
        canonical(tuple(g for g in CONTRAST_BACKBONE if g not in ALWAYS_ON_GROUPS))
    )
    if len(set(combinations)) != len(combinations):
        raise RuntimeError("Contrast combinations must be distinct.")
    return combinations


CONTRAST_COMBINATIONS = tuple(_contrast_combinations())

# The first rung of part 2 reproduces FS-2 column for column, which makes that
# paper a point on our own ladder rather than an external baseline and checks
# the fixed-profile code path against the group switches.
#
# FS-1 as published for B-cell keeps the four protein-level descriptors and
# protein_length, so it lands two rungs up the ladder rather than off it. Its
# peptide-only variant, FS-1b, is PAPER_PATH_START itself.
#
# FS-3 is not here either: RFE kept subsets of PROP and PROTGLOBAL rather than
# whole groups, and added the label-encoded protein id, so no union of groups
# reproduces it.
PAPER_RECONSTRUCTIONS = {
    "perbandingan_1": canonical(
        PAPER_PATH_START + ("protglobal", "protlen")
    ),
    "perbandingan_2": canonical(PAPER_PATH_START + ("protglobal",)),
}
# What FS-1b reduces to once the SARS-CoV feature engineering is applied. It
# sits in part 1, not on the ladder.
PAPER_1_EQUIVALENT = canonical(PAPER_PATH_START)
if any(
    combination not in CONTRAST_COMBINATIONS
    for combination in PAPER_RECONSTRUCTIONS.values()
):
    raise RuntimeError("Every paper reconstruction must be a rung of part 2.")
if PAPER_1_EQUIVALENT not in ALL_COMBINATIONS:
    raise RuntimeError("The FS-1 equivalent must be a part-1 combination.")


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
    feature_combinations: Iterable[Iterable[str]] | None = None,
    include_comparisons: bool = True,
    include_contrasts: bool | None = None,
    contrast_groups: Iterable[str] | None = None,
    comparison_feature_sets: Iterable[str] | None = None,
    include_proposed: bool | None = None,
) -> list[ExperimentSpec]:
    """Build the proposed power set, the planned contrasts, and the baselines.

    A full sweep is 31 configurations per model:
      15  proposed   part 1: power set over FEATURE_GROUPS with ALWAYS_ON_GROUPS
                     held on -- the only arm the selection rule sees
      13  kontras    part 2: the literature's path, cumulative from
                             PAPER_PATH_START (its first two rungs are FS-2
                             and FS-1 exactly)
                     part 3: each PROTEIN_GROUP on CONTRAST_BACKBONE
                     part 4: each PROTEIN_GROUP alone
                     part 5: CONTRAST_BACKBONE without the always-on groups
       3  perbandingan  part 6: FS-1, FS-2, FS-3 exactly as published

    Passing comparison_feature_sets restricts part 6 to the named profiles and,
    like the proposed-arm restrictions, switches the other arms off unless they
    are asked for explicitly. That is how a profile outside the default sweep
    is run on its own, without changing what the default sweep plans.

    Passing feature_groups restricts the proposed arm to that single frozen
    combination, which is what phase 2 runs once a configuration is selected.
    Passing feature_combinations restricts it to several, and contrast_groups
    restricts the contrast arm; together they are how a sweep is split across
    sessions. Either proposed-arm restriction switches the contrast and
    reconstruction arms off unless they are asked for explicitly, because they
    belong to a full sweep rather than a partial one.
    """
    if feature_groups is not None and feature_combinations is not None:
        raise ValueError(
            "Pass feature_groups (one combination) or feature_combinations "
            "(several), not both."
        )
    restricted = feature_groups is not None or feature_combinations is not None

    if comparison_feature_sets is None:
        selected_comparisons = DEFAULT_COMPARISON_FEATURE_SETS
        comparisons_restricted = False
    else:
        selected_comparisons = tuple(dict.fromkeys(comparison_feature_sets))
        comparisons_restricted = True
        if not selected_comparisons:
            raise ValueError(
                "comparison_feature_sets must name at least one profile."
            )
        unknown_profiles = [
            name for name in selected_comparisons
            if name not in COMPARISON_FEATURE_SETS
        ]
        if unknown_profiles:
            raise ValueError(
                f"Unsupported comparison feature sets: {sorted(unknown_profiles)}."
            )

    if include_contrasts is None:
        include_contrasts = not (restricted or comparisons_restricted)
    if include_proposed is None:
        include_proposed = not comparisons_restricted

    selected_models = tuple(model_types) if model_types else MODEL_TYPES
    unknown_models = [model for model in selected_models if model not in MODEL_TYPES]
    if unknown_models:
        raise ValueError(f"Unsupported model types: {sorted(unknown_models)}.")

    if feature_combinations is not None:
        requested = [canonical(combination) for combination in feature_combinations]
        if not requested:
            raise ValueError("At least one feature combination must be selected.")
        proposed_combinations = list(dict.fromkeys(requested))
    elif feature_groups is None:
        proposed_combinations = list(ALL_COMBINATIONS)
    else:
        proposed_combinations = [canonical(feature_groups)]

    contrast_combinations: list[tuple[str, ...]] = []
    if include_contrasts:
        if contrast_groups is None:
            contrast_combinations = list(CONTRAST_COMBINATIONS)
        else:
            # A session may take a subset, keyed by the protein group each
            # contrast is about; part 5 has none, so it is named explicitly.
            wanted = set(contrast_groups)
            unknown = sorted(wanted.difference(PROTEIN_GROUPS) - {"len"})
            if unknown:
                raise ValueError(f"Unsupported contrast groups: {unknown}.")
            contrast_combinations = [
                combination
                for combination in CONTRAST_COMBINATIONS
                if (set(combination) & set(PROTEIN_GROUPS) & wanted)
                or ("len" in wanted and not set(combination) & set(PROTEIN_GROUPS))
            ]

    specs: list[ExperimentSpec] = []
    for model_type in selected_models:
        if include_proposed:
            specs.extend(
                ExperimentSpec("proposed", model_type, combination)
                for combination in proposed_combinations
            )
        specs.extend(
            ExperimentSpec("kontras", model_type, combination)
            for combination in contrast_combinations
        )
        if include_comparisons:
            specs.extend(
                ExperimentSpec(feature_set, model_type)
                for feature_set in selected_comparisons
            )

    per_model = (
        (len(proposed_combinations) if include_proposed else 0)
        + len(contrast_combinations)
        + (len(selected_comparisons) if include_comparisons else 0)
    )
    if per_model == 0:
        raise ValueError(
            "Every arm is switched off, so there is nothing to run."
        )
    expected = len(selected_models) * per_model
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
    if spec.feature_set in GROUPED_FEATURE_SETS:
        assert spec.feature_groups is not None
        active_groups = set(spec.feature_groups)
        for group, switch in GROUP_SWITCHES.items():
            setattr(config, switch, group in active_groups)
        # The two historical blocks are never part of this matrix. Setting them
        # explicitly stops a non-default base config from leaking a protein
        # identifier into a run whose groups do not name one.
        config.USE_PROTEIN_FEATURES = False
        config.USE_BIOLOGICAL_FEATURES = False
        config.USE_PEPTIDE_FEATURES = False
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
    feature_combinations: Iterable[Iterable[str]] | None = None,
    include_comparisons: bool = True,
    include_contrasts: bool | None = None,
    contrast_groups: Iterable[str] | None = None,
    comparison_feature_sets: Iterable[str] | None = None,
    include_proposed: bool | None = None,
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
    specs = build_experiment_matrix(
        model_types,
        feature_groups,
        feature_combinations=feature_combinations,
        include_comparisons=include_comparisons,
        include_contrasts=include_contrasts,
        contrast_groups=contrast_groups,
        comparison_feature_sets=comparison_feature_sets,
        include_proposed=include_proposed,
    )
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
        # Includes ALWAYS_ON_GROUPS: a frozen combination names every group it
        # contains, and select_configuration prints exactly that list, so the
        # command it hands the user has to be accepted here.
        choices=FEATURE_GROUPS + ALWAYS_ON_GROUPS,
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
