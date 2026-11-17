"""Turn experiment_summary.csv files into the publication comparison table.

The layout matches the spreadsheet used to record results: one block per model,
internal metrics as "mean ± std" across the five folds, external SARS metrics as
point estimates, and one row per feature combination followed by the five
comparison papers.

Typical two-phase use:

    # phase 1 - selection, B-cell only
    python -m ML.summary_table ML/artifacts/phase1/experiment_summary.csv

    # phase 2 - add the frozen external numbers next to the internal ones
    python -m ML.summary_table \
        ML/artifacts/phase1/experiment_summary.csv \
        ML/artifacts/phase2/experiment_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    ("roc_auc", "ROC AUC"),
    ("pr_auc", "PR AUC"),
    ("accuracy", "Accuracy"),
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("f1", "F1 Score"),
    ("mcc", "MCC"),
)

MODEL_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGBOOST",
    "svm": "SVM",
    "logistic_regression": "LR",
    "extra_trees": "ET",
}
# Must match ML.experiment_matrix.MODEL_TYPES. SVM is no longer swept; its
# label stays in MODEL_LABELS so an earlier summary still renders.
DEFAULT_MODEL_ORDER = (
    "random_forest",
    "logistic_regression",
    "extra_trees",
    "xgboost",
)

GROUP_LABELS = {
    "aac": "AAC",
    "dpc": "DPC",
    "prop": "PROP",
    "flank": "FLANK",
    "len": "LEN",
    "protglobal": "PROTGLOBAL",
    "protlen": "PROTLEN",
    "pos": "POS",
    "protid": "PROTID",
    "biological": "BIO",
    "protein": "PROT",
}
# Part 1, in manuscript order: AAC-first, then PROP, then FLANK, then DPC.
# LEN is in every row, so the labels name it last rather than repeating it in
# front. This is the only arm the selection rule reads.
FEATURE_COMBINATION_ORDER = (
    ("aac", "len"),
    ("aac", "prop", "len"),
    ("aac", "flank", "len"),
    ("aac", "prop", "flank", "len"),
    ("prop", "len"),
    ("prop", "flank", "len"),
    ("flank", "len"),
    ("aac", "dpc", "len"),
    ("aac", "dpc", "prop", "len"),
    ("aac", "dpc", "flank", "len"),
    ("aac", "dpc", "prop", "flank", "len"),
    ("dpc", "len"),
    ("dpc", "prop", "len"),
    ("dpc", "prop", "flank", "len"),
    ("dpc", "flank", "len"),
)
# Parts 2 to 5, in that order. Reported, never selected from.
PROTEIN_GROUPS = ("protglobal", "protlen", "pos", "protid")
PAPER_PATH_START = ("prop", "len")
CONTRAST_BACKBONE = ("aac", "prop", "flank", "len")


def _paper_path() -> tuple[tuple[str, ...], ...]:
    rungs, rung = [], PAPER_PATH_START
    for group in PROTEIN_GROUPS:
        rung = rung + (group,)
        rungs.append(rung)
    return tuple(rungs)


PART2_PAPER_PATH = _paper_path()
PART3_ON_BACKBONE = tuple(CONTRAST_BACKBONE + (g,) for g in PROTEIN_GROUPS)
PART4_ALONE = tuple((g,) for g in PROTEIN_GROUPS)
PART5_WITHOUT_LEN = (("aac", "prop", "flank"),)
CONTRAST_COMBINATION_ORDER = (
    PART2_PAPER_PATH + PART3_ON_BACKBONE + PART4_ALONE + PART5_WITHOUT_LEN
)
# Rows that reproduce a published profile, so the table says so instead of
# leaving the reader to spot it. FS-1 reduces to PROP + LEN once its own
# feature engineering is applied, which is a part-1 row rather than a rung.
PAPER_PATH_NOTES = {
    tuple(sorted(PART2_PAPER_PATH[0])): " (= FS-2)",
    ("len", "prop"): " (= FS-1 less its duplicate column)",
}
# Group names earlier sweeps used. Kept only so a superseded summary still
# prints; the matrix no longer produces any of them.
_LEGACY_GROUPS = ("aac", "dpc", "biological", "flank", "protein")
SUPERSEDED_COMBINATION_ORDER = tuple(
    combination
    for size in range(1, len(_LEGACY_GROUPS) + 1)
    for combination in itertools.combinations(_LEGACY_GROUPS, size)
    if combination not in FEATURE_COMBINATION_ORDER
)
ALL_COMBINATION_ORDER = (
    FEATURE_COMBINATION_ORDER
    + CONTRAST_COMBINATION_ORDER
    + SUPERSEDED_COMBINATION_ORDER
)
# Only the FEATURE SET is taken from each comparison study; the classifier,
# folds, tuning budget and imbalance handling are ours. Labelling these rows
# with the original method names would misreport those studies, so they are
# named as feature sets.
PAPER_LABELS = {
    "perbandingan_1": "FS-1 (Cihan et al. 2022)",
    "perbandingan_2": "FS-2 (Jain et al. 2021)",
    "perbandingan_3": "FS-3 (Gurcan et al. 2025)",
    "perbandingan_4": "FS-4 (Liu et al. 2024)",
    "perbandingan_5": "FS-5 (BLMPred 2026)",
}
PAPER_ORDER = tuple(PAPER_LABELS)

EXTERNAL_PREFIXES = {
    "selected": "external_sars_bcell_selected_threshold",
    "0.5": "external_sars_threshold_0_5",
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def _combination_key(feature_groups: str) -> tuple[str, ...] | None:
    """Normalize a stored 'aac,dpc' label into a canonical tuple."""
    text = (feature_groups or "").strip()
    if not text or text == "profile_defined":
        return None
    return tuple(sorted(part.strip() for part in text.split(",") if part.strip()))


def read_summaries(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Merge summary CSVs by experiment_id; later files win per populated field.

    This lets a phase-1 (internal only) and a phase-2 (external) sweep be
    combined into one table without rerunning either.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Summary file not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                experiment_id = (row.get("experiment_id") or "").strip()
                if not experiment_id:
                    continue
                target = merged.setdefault(experiment_id, {})
                for key, value in row.items():
                    if value is not None and str(value).strip() != "":
                        target[key] = value
    return merged


def _format_mean_std(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _internal_cell(row: dict[str, Any], metric: str, digits: int) -> str:
    """Internal metrics live under different prefixes per training mode."""
    for mean_key, std_key in (
        (f"bcell_oof_{metric}", f"bcell_oof_std_{metric}"),
        (f"mean_validation_{metric}", f"std_validation_{metric}"),
    ):
        mean = _to_float(row.get(mean_key))
        if mean is not None:
            return _format_mean_std(mean, _to_float(row.get(std_key)), digits)
    return ""


def _external_cell(
    row: dict[str, Any],
    metric: str,
    digits: int,
    threshold: str,
) -> str:
    value = _to_float(row.get(f"{EXTERNAL_PREFIXES[threshold]}_{metric}"))
    return "" if value is None else f"{value:.{digits}f}"


def _index_summaries(
    summaries: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, tuple[str, ...] | None], dict[str, Any]]:
    """Index successful runs by (model, feature set, feature combination)."""
    by_key: dict[tuple[str, str, tuple[str, ...] | None], dict[str, Any]] = {}
    for row in summaries.values():
        if (row.get("status") or "").strip() != "success":
            continue
        key = (
            (row.get("model_type") or "").strip(),
            (row.get("feature_set") or "").strip(),
            _combination_key(row.get("feature_groups", "")),
        )
        by_key[key] = row
    return by_key


def _group_label(combination: tuple[str, ...]) -> str:
    return " + ".join(GROUP_LABELS[group] for group in combination)


def full_row_plan() -> list[tuple[str, str, tuple[str, ...] | None]]:
    """Every row the table can show, as (label, feature_set, groups).

    Four arms in order: the power set the selection rule works over, the
    planned contrasts, the reference sets rebuilt from our own groups, and the
    reference sets exactly as published. Superseded group names follow the
    power set so an earlier sweep still prints under its own labels.
    """
    plan: list[tuple[str, str, tuple[str, ...] | None]] = []
    for combination in FEATURE_COMBINATION_ORDER:
        note = PAPER_PATH_NOTES.get(tuple(sorted(combination)), "")
        plan.append((_group_label(combination) + note, "proposed", combination))
    for combination in SUPERSEDED_COMBINATION_ORDER:
        plan.append((_group_label(combination), "proposed", combination))
    for combination in CONTRAST_COMBINATION_ORDER:
        note = PAPER_PATH_NOTES.get(tuple(sorted(combination)), "")
        plan.append((_group_label(combination) + note, "kontras", combination))
    for feature_set in PAPER_ORDER:
        plan.append((PAPER_LABELS[feature_set], feature_set, None))
    return plan


def _rows_present(
    summaries: dict[str, dict[str, Any]]
) -> list[tuple[str, str, tuple[str, ...] | None]]:
    """The plan restricted to rows this sweep actually covers.

    A sweep that ran only part of the matrix, or an earlier sweep using group
    names the matrix no longer produces, would otherwise contribute a block of
    permanently blank rows.
    """
    seen = {
        (
            (row.get("feature_set") or "").strip(),
            _combination_key(row.get("feature_groups", "")),
        )
        for row in summaries.values()
    }
    return [
        entry
        for entry in full_row_plan()
        if (
            entry[1],
            None if entry[2] is None else tuple(sorted(entry[2])),
        )
        in seen
    ]


def _model_entries(
    by_key: dict[tuple[str, str, tuple[str, ...] | None], dict[str, Any]],
    model_type: str,
    plan: list[tuple[str, str, tuple[str, ...] | None]] | None = None,
) -> list[tuple[str, dict[str, Any] | None]]:
    """The rows one model contributes, resolved against the indexed summaries."""
    if plan is None:
        plan = full_row_plan()
    entries: list[tuple[str, dict[str, Any] | None]] = []
    for label, feature_set, combination in plan:
        key = (
            model_type,
            feature_set,
            None if combination is None else tuple(sorted(combination)),
        )
        entries.append((label, by_key.get(key)))
    return entries


def build_internal_table(
    summaries: dict[str, dict[str, Any]],
    *,
    model_order: tuple[str, ...] = DEFAULT_MODEL_ORDER,
    digits: int = 4,
) -> list[list[str]]:
    """Flat internal-validation table: Model, Note, then one metric per column.

    Every row repeats the model label so the block can be sorted or filtered in
    a spreadsheet without losing which model a row belongs to.
    """
    by_key = _index_summaries(summaries)
    plan = _rows_present(summaries)
    rows: list[list[str]] = []
    for model_type in model_order:
        for note, row in _model_entries(by_key, model_type, plan):
            cells = [MODEL_LABELS.get(model_type, model_type), note]
            if row is None:
                cells.extend([""] * len(METRICS))
            else:
                cells.extend(
                    _internal_cell(row, metric, digits) for metric, _ in METRICS
                )
            rows.append(cells)
    return rows


def internal_header() -> list[str]:
    return ["Model", "Note"] + [name for _, name in METRICS]


def build_table(
    summaries: dict[str, dict[str, Any]],
    *,
    model_order: tuple[str, ...] = DEFAULT_MODEL_ORDER,
    digits: int = 4,
    threshold: str = "selected",
) -> list[list[str]]:
    """Build the full table body, one list of cells per row."""
    if threshold not in EXTERNAL_PREFIXES:
        raise ValueError(f"threshold must be one of {sorted(EXTERNAL_PREFIXES)}.")

    by_key = _index_summaries(summaries)

    rows: list[list[str]] = []
    for number, model_type in enumerate(model_order, start=1):
        entries = _model_entries(by_key, model_type)

        for index, (note, row) in enumerate(entries):
            cells = [
                str(number) if index == 0 else "",
                MODEL_LABELS.get(model_type, model_type) if index == 0 else "",
                note,
            ]
            if row is None:
                cells.extend([""] * len(METRICS))
                cells.append("FALSE")
                cells.extend([""] * len(METRICS))
                rows.append(cells)
                continue

            cells.extend(_internal_cell(row, metric, digits) for metric, _ in METRICS)
            external = [
                _external_cell(row, metric, digits, threshold) for metric, _ in METRICS
            ]
            cells.append("TRUE" if any(external) else "FALSE")
            cells.extend(external)
            rows.append(cells)
    return rows


def header_rows() -> tuple[list[str], list[str]]:
    metric_names = [name for _, name in METRICS]
    top = ["Evaluasi", "", ""] + ["Internal"] + [""] * (len(METRICS) - 1)
    top += [""] + ["External"] + [""] * (len(METRICS) - 1)
    bottom = ["No", "Model", "Note"] + metric_names + [""] + metric_names
    return top, bottom


def render_tsv(rows: list[list[str]]) -> str:
    top, bottom = header_rows()
    return "\n".join("\t".join(cells) for cells in (top, bottom, *rows))


def render_rows_tsv(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join("\t".join(cells) for cells in (header, *rows))


def _render_grid(labels: list[str], rows: list[list[str]]) -> str:
    """Align a header plus body into fixed-width console columns."""
    table = [labels, *rows]
    widths = [
        max(len(row[column]) for row in table if column < len(row))
        for column in range(len(labels))
    ]
    lines = []
    for index, row in enumerate([labels, *rows]):
        padded = [
            row[column].ljust(widths[column]) if column < len(row) else " " * widths[column]
            for column in range(len(labels))
        ]
        lines.append(" | ".join(padded).rstrip())
        if index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def render_pretty(rows: list[list[str]]) -> str:
    _, bottom = header_rows()
    labels = list(bottom)
    labels[10] = "Ext?"
    return _render_grid(labels, rows)


def render_internal_pretty(rows: list[list[str]]) -> str:
    return _render_grid(internal_header(), rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render experiment_summary.csv files as the publication comparison "
            "table (internal mean ± std, external point estimates)."
        )
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help=(
            "One or more experiment_summary.csv files. Later files override "
            "populated fields, so pass phase 1 then phase 2."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the table as TSV here (paste straight into a spreadsheet).",
    )
    parser.add_argument("--digits", type=int, default=4, help="Decimal places.")
    parser.add_argument(
        "--layout",
        choices=("internal", "comparison"),
        default="internal",
        help=(
            "'internal' (default): Model, Note and the seven internal metrics "
            "as mean ± SD, one row per run. 'comparison': the wider internal "
            "plus external SARS block."
        ),
    )
    parser.add_argument(
        "--external-threshold",
        choices=tuple(EXTERNAL_PREFIXES),
        default="selected",
        help=(
            "Which SARS threshold to report: 'selected' (MCC-optimised on "
            "B-cell out-of-fold predictions) or '0.5'."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_LABELS),
        default=None,
        help="Model display order (default: RF, XGBOOST, SVM, LR, ET).",
    )
    args = parser.parse_args()

    # Windows consoles default to a legacy codepage that mangles "±".
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    summaries = read_summaries(args.summaries)
    model_order = tuple(args.models) if args.models else DEFAULT_MODEL_ORDER

    if args.layout == "internal":
        rows = build_internal_table(
            summaries, model_order=model_order, digits=args.digits
        )
        print(render_internal_pretty(rows))
        filled = sum(1 for row in rows if row[2])
        print(f"\n{filled}/{len(rows)} rows have internal metrics.")
        tsv = render_rows_tsv(internal_header(), rows)
    else:
        rows = build_table(
            summaries,
            model_order=model_order,
            digits=args.digits,
            threshold=args.external_threshold,
        )
        print(render_pretty(rows))
        total = sum(1 for row in rows if row[10] in {"TRUE", "FALSE"})
        external_done = sum(1 for row in rows if row[10] == "TRUE")
        internal_done = sum(1 for row in rows if row[3])
        print(
            f"\n{internal_done}/{total} rows have internal metrics; "
            f"{external_done}/{total} have external SARS metrics."
        )
        tsv = render_tsv(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tsv, encoding="utf-8")
        print(f"TSV written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
