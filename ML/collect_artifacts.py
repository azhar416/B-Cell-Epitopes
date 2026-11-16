"""Flatten a sweep's scattered artifacts into a few long-format tables.

A finished sweep writes one deep directory per run: a report JSON, an
out-of-fold prediction CSV, a hyperparameter comparison CSV, five fold
directories, plots, and model bundles. That is hundreds of files, which is
awkward to inspect and, on Kaggle, lost when the session ends.

This module reads `experiment_summary.csv` as the index of runs and rewrites
everything that is a *measurement* into long-format tables:

    runs.csv                     one row per run: sizes, threshold, protocol
    fold_metrics.csv             one row per (run, fold): train and validation
    hyperparameter_candidates.csv  one row per (run, candidate): every score
    oof_predictions.csv.gz       one row per (run, peptide): OOF probability
    external_sars_predictions.csv  one row per (run, SARS peptide)
    manifest.json                what was written, with row counts

Model bundles and plots are deliberately left where they are: they are not
measurements, and the bundles alone run to gigabytes across a full sweep.

Run directories are rebuilt relative to the summary file rather than read from
the absolute `artifact_directory` recorded at run time, so a sweep stays
readable after being moved, downloaded, or re-uploaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Where each training mode puts its top-level report, relative to the run dir.
PHASE_SUBDIR = {
    "same_dataset": "cross_validation",
    "external_sars": "external_sars",
}
PHASE_REPORT = {
    "same_dataset": "cross_validation_metrics.json",
    "external_sars": "external_sars_metrics.json",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return a report, or None when it is absent or was truncated mid-write."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return report if isinstance(report, dict) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def feature_group_label(feature_groups: str | None) -> str:
    """Mirror ExperimentSpec.feature_group_label from the stored CSV column."""
    text = (feature_groups or "").strip()
    if not text or text == "profile_defined":
        return "profile_defined"
    return "__".join(part.strip() for part in text.split(",") if part.strip())


def run_directory(summary_path: Path, row: dict[str, Any]) -> Path:
    """Rebuild a run's artifact directory relative to the summary file."""
    mode = (row.get("training_mode") or "").strip()
    if mode not in PHASE_SUBDIR:
        raise ValueError(f"Unknown training_mode {mode!r} in the summary CSV.")
    feature_set = (row.get("feature_set") or "").strip()
    return (
        Path(summary_path).resolve().parent
        / feature_set
        / (row.get("model_type") or "").strip()
        / feature_group_label(row.get("feature_groups"))
        / feature_set
        / PHASE_SUBDIR[mode]
    )


def internal_cv_directory(run_dir: Path, mode: str, feature_set: str) -> Path:
    """Where the internal cross-validation artifacts sit for either mode.

    Phase 2 runs the same_dataset cross-validation in a nested directory, so
    both modes expose fold metrics, candidate scores and out-of-fold
    predictions in the same shape and can be collected by one code path.
    """
    if mode == "same_dataset":
        return run_dir
    return run_dir / "internal_bcell_same_dataset" / feature_set / "cross_validation"


class Run:
    """One sweep entry, with its reports resolved and loaded."""

    def __init__(self, summary_path: Path, row: dict[str, Any]) -> None:
        self.row = row
        self.experiment_id = (row.get("experiment_id") or "").strip()
        self.mode = (row.get("training_mode") or "").strip()
        self.feature_set = (row.get("feature_set") or "").strip()
        self.model_type = (row.get("model_type") or "").strip()
        self.feature_groups = (row.get("feature_groups") or "").strip()
        self.directory = run_directory(summary_path, row)
        self.report = _read_json(self.directory / PHASE_REPORT[self.mode])
        self.internal_directory = internal_cv_directory(
            self.directory, self.mode, self.feature_set
        )
        self.internal_report = _read_json(
            self.internal_directory / PHASE_REPORT["same_dataset"]
        )

    @property
    def found(self) -> bool:
        return self.report is not None

    def key(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "training_mode": self.mode,
            "model_type": self.model_type,
            "feature_set": self.feature_set,
            "feature_groups": self.feature_groups,
        }


def read_runs(summary_paths: Iterable[Path]) -> list[Run]:
    """Load every successful run listed in the given summary CSVs."""
    rows: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Summary file not found: {path}")
        for row in _read_csv(path):
            if (row.get("status") or "").strip() != "success":
                continue
            experiment_id = (row.get("experiment_id") or "").strip()
            if experiment_id:
                rows[experiment_id] = {**row, "__summary__": path}
    return [Run(row.pop("__summary__"), row) for row in rows.values()]


def _flatten(prefix: str, metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {
        f"{prefix}{name}": value
        for name, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def fold_metrics_table(runs: list[Run]) -> list[dict[str, Any]]:
    """One row per (run, fold), with training and validation metrics."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        report = run.internal_report
        if report is None:
            continue
        # Phase 2 also reports every fold re-scored at the deployment
        # threshold; phase 1 has no such threshold, so the columns stay empty.
        at_selected = {}
        if isinstance(run.report, dict):
            internal = run.report.get("internal_bcell_cv")
            if isinstance(internal, dict):
                at_selected = {
                    fold.get("fold"): fold
                    for fold in internal.get("folds_at_selected_threshold") or []
                    if isinstance(fold, dict)
                }
        for fold in report.get("fold_metrics") or []:
            if not isinstance(fold, dict):
                continue
            row = {
                **run.key(),
                "fold": fold.get("fold"),
                "train_size": fold.get("train_size"),
                "validation_size": fold.get("validation_size"),
                "selected_hyperparameters": json.dumps(
                    fold.get("selected_hyperparameters"), ensure_ascii=False
                ),
            }
            row.update(_flatten("train_", fold.get("training_metrics")))
            row.update(_flatten("validation_", fold.get("validation_metrics")))
            extra = at_selected.get(fold.get("fold"))
            if isinstance(extra, dict):
                row.update(
                    _flatten(
                        "validation_at_selected_threshold_",
                        {k: v for k, v in extra.items() if k != "fold"},
                    )
                )
            rows.append(row)
    return rows


def candidate_table(runs: list[Run]) -> list[dict[str, Any]]:
    """One row per (run, hyperparameter candidate), with all pooled OOF metrics.

    This is the tuning evidence: it shows the margin between the winner and the
    runner-up, which a single "selected hyperparameters" field hides.
    """
    rows: list[dict[str, Any]] = []
    for run in runs:
        report = run.internal_report
        if report is None:
            continue
        for candidate in report.get("hyperparameter_comparison") or []:
            if not isinstance(candidate, dict):
                continue
            row = {**run.key()}
            for name, value in candidate.items():
                row[name] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
            rows.append(row)
    return rows


def run_overview_table(runs: list[Run]) -> list[dict[str, Any]]:
    """One row per run: dataset sizes, protocol text, threshold, SARS metrics."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        report = run.report or {}
        internal = report.get("internal_bcell_cv")
        internal = internal if isinstance(internal, dict) else {}
        source = run.internal_report or {}
        row: dict[str, Any] = {
            **run.key(),
            "split_strategy": source.get("split_strategy"),
            "n_splits": source.get("n_splits"),
            "hyperparameter_tuning": source.get("hyperparameter_tuning"),
            "hyperparameter_tuning_metric": source.get("hyperparameter_tuning_metric"),
            "hyperparameter_selection_protocol": source.get(
                "hyperparameter_selection_protocol"
            ),
            "selected_hyperparameters": json.dumps(
                source.get("selected_hyperparameters"), ensure_ascii=False
            ),
            "n_examples": source.get("n_examples"),
            "n_parent_proteins": source.get("n_parent_proteins"),
            "validation_metrics_interpretation": source.get(
                "validation_metrics_interpretation"
            ),
            "hyperparameter_tuning_warning": source.get(
                "hyperparameter_tuning_warning"
            ),
        }
        row.update(_flatten("pooled_oof_", source.get("validation_metrics")))
        row.update(_flatten("mean_validation_", source.get("mean_validation_metrics")))
        row.update(_flatten("std_validation_", source.get("std_validation_metrics")))
        row.update(_flatten("mean_train_", source.get("mean_training_metrics")))
        row.update(_flatten("std_train_", source.get("std_training_metrics")))

        if run.mode == "external_sars":
            selection = report.get("model_selection")
            selection = selection if isinstance(selection, dict) else {}
            row.update(
                {
                    "selection_metric": selection.get("selection_metric"),
                    "selection_score": selection.get("selection_score"),
                    "candidate_count": selection.get("candidate_count"),
                    "selected_threshold": internal.get("selected_threshold"),
                    "threshold_selection_mcc": internal.get("threshold_selection_mcc"),
                    "threshold_selection_source": internal.get(
                        "threshold_selection_source"
                    ),
                    "sars_labels_used_for_selection": report.get(
                        "sars_labels_used_for_model_or_threshold_selection"
                    ),
                    "bcell_rows_used_for_development": report.get(
                        "bcell_rows_used_for_development"
                    ),
                    "external_sars_rows": report.get("external_sars_rows"),
                    "external_sars_proteins": report.get("external_sars_proteins"),
                }
            )
            row.update(
                _flatten("overlap_", report.get("exact_sequence_overlap_audit"))
            )
            row.update(
                _flatten(
                    "mean_validation_at_selected_threshold_",
                    internal.get("mean_metrics_at_selected_threshold"),
                )
            )
            row.update(
                _flatten(
                    "std_validation_at_selected_threshold_",
                    internal.get("std_metrics_at_selected_threshold"),
                )
            )
            row.update(
                _flatten("oof_at_0_5_", internal.get("oof_metrics_threshold_0_5"))
            )
            row.update(
                _flatten(
                    "oof_at_selected_threshold_",
                    internal.get("oof_metrics_bcell_selected_threshold"),
                )
            )
            row.update(
                _flatten(
                    "sars_at_0_5_",
                    report.get("external_sars_metrics_threshold_0_5"),
                )
            )
            row.update(
                _flatten(
                    "sars_at_selected_threshold_",
                    report.get("external_sars_metrics_bcell_selected_threshold"),
                )
            )
        rows.append(row)
    return rows


def _prediction_rows(
    runs: list[Run], relative_path: str, from_internal: bool
) -> Iterator[dict[str, Any]]:
    for run in runs:
        base = run.internal_directory if from_internal else run.directory
        for row in _read_csv(base / relative_path):
            yield {**run.key(), **row}


def oof_prediction_rows(runs: list[Run]) -> Iterator[dict[str, Any]]:
    """Per-peptide out-of-fold probabilities, the basis of every pooled metric."""
    return _prediction_rows(runs, "out_of_fold_predictions.csv", True)


def external_prediction_rows(runs: list[Run]) -> Iterator[dict[str, Any]]:
    return _prediction_rows(runs, "external_sars_predictions.csv", False)


def _write_rows(
    path: Path, rows: Iterable[dict[str, Any]], *, compress: bool = False
) -> int:
    """Stream rows to CSV, taking the column order from the first row.

    Streaming matters for the prediction tables: a full sweep is over a million
    rows, and materialising them all before writing is pure memory cost. Later
    rows may carry keys the first row lacked, so unseen columns are dropped
    rather than silently shifting every value right; the collectors emit
    uniform keys, so this is a guard, not an expected path.
    """
    import gzip

    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return 0
    opener = (
        (lambda: gzip.open(path, "wt", newline="", encoding="utf-8"))
        if compress
        else (lambda: path.open("w", newline="", encoding="utf-8"))
    )
    written = 0
    with opener() as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(first), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerow(first)
        written = 1
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def collect_all(
    summary_paths: Iterable[Path],
    output_dir: Path,
    *,
    include_predictions: bool = True,
) -> dict[str, Any]:
    """Write every long-format table and return a manifest describing them."""
    summary_paths = [Path(path) for path in summary_paths]
    runs = read_runs(summary_paths)
    missing = [run.experiment_id for run in runs if not run.found]
    runs = [run for run in runs if run.found]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict[str, Any]] = {}

    def record(name: str, filename: str, rows, compress=False) -> None:
        path = output_dir / filename
        count = _write_rows(path, rows, compress=compress)
        if count:
            written[name] = {"file": filename, "rows": count}
        elif path.exists():
            path.unlink()

    record("runs", "runs.csv", run_overview_table(runs))
    record("fold_metrics", "fold_metrics.csv", fold_metrics_table(runs))
    record(
        "hyperparameter_candidates",
        "hyperparameter_candidates.csv",
        candidate_table(runs),
    )
    if include_predictions:
        record(
            "oof_predictions",
            "oof_predictions.csv.gz",
            oof_prediction_rows(runs),
            compress=True,
        )
        record(
            "external_sars_predictions",
            "external_sars_predictions.csv",
            external_prediction_rows(runs),
        )

    manifest = {
        "summaries": [str(path.resolve()) for path in summary_paths],
        "runs_collected": len(runs),
        "runs_missing_report": missing,
        "tables": written,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a sweep's per-run artifacts into long-format metric tables."
        )
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="One or more experiment_summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the collected tables into.",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip the per-peptide prediction tables (much smaller output).",
    )
    args = parser.parse_args()

    manifest = collect_all(
        args.summaries,
        args.output_dir,
        include_predictions=not args.no_predictions,
    )
    print(f"Collected {manifest['runs_collected']} run(s).")
    for name, info in manifest["tables"].items():
        print(f"  {name:<28} {info['rows']:>9,} rows  -> {info['file']}")
    if manifest["runs_missing_report"]:
        print(
            f"{len(manifest['runs_missing_report'])} run(s) had no readable "
            "report and were skipped:"
        )
        for experiment_id in manifest["runs_missing_report"]:
            print(f"  {experiment_id}")
    print(f"Manifest: {(args.output_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
