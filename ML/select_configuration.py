"""Freeze the proposed feature combination from phase-1 internal CV results.

This encodes the selection rule BEFORE any results exist, so the choice of
configuration is mechanical rather than made after seeing which combination
happens to win. Run it once on the phase-1 summary, commit the JSON it writes,
and only then run phase 2.

The rule (one-standard-error rule, Hastie & Tibshirani, ESL 7.10):

  1. Score every proposed feature combination by its mean metric across the
     five folds.
  2. Take the best mean and its standard error, SE = SD / sqrt(n_folds).
  3. Keep every combination whose mean is within one SE of the best.
  4. Among those, choose the one with the fewest features; ties break towards
     the smaller standard deviation.

Step 3 is the point: with five folds the gap between the top combinations is
usually smaller than their own noise, so taking the plain argmax would select
the luckiest combination rather than the best one.

Usage:

    python -m ML.select_configuration ML/artifacts/phase1/experiment_summary.csv \
        --output ML/artifacts/frozen_configuration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


N_FOLDS = 5
# Canonical order, matching ML.experiment_matrix.FEATURE_GROUPS. It defines the
# tuple stored in the summary CSV and therefore each run's experiment_id.
FEATURE_GROUPS = ("aac", "dpc", "protein", "biological")
# Display order, matching ML.summary_table and the manuscript table, where BIO
# precedes PROT. Only affects how a combination is printed.
DISPLAY_ORDER = ("aac", "dpc", "biological", "protein")
GROUP_LABELS = {
    "aac": "AAC",
    "dpc": "DPC",
    "protein": "PROT",
    "biological": "BIO",
}
# Feature counts per group on the cleaned B-cell data; the groups are disjoint
# so a combination's width is their sum. test_selection.py asserts these still
# match build_feature_frame, so drift cannot go unnoticed.
GROUP_FEATURE_COUNTS = {
    "aac": 22,
    "dpc": 400,
    "protein": 23,
    "biological": 8,
}
METRIC_COLUMNS = {
    "roc_auc": ("mean_validation_roc_auc", "std_validation_roc_auc"),
    "pr_auc": ("mean_validation_pr_auc", "std_validation_pr_auc"),
    "mcc": ("mean_validation_mcc", "std_validation_mcc"),
    "f1": ("mean_validation_f1", "std_validation_f1"),
}
# Phase-2 (external_sars) summaries store internal metrics under another prefix.
FALLBACK_METRIC_COLUMNS = {
    "roc_auc": ("bcell_oof_roc_auc", "bcell_oof_std_roc_auc"),
    "pr_auc": ("bcell_oof_pr_auc", "bcell_oof_std_pr_auc"),
    "mcc": ("bcell_oof_mcc", "bcell_oof_std_mcc"),
    "f1": ("bcell_oof_f1", "bcell_oof_std_f1"),
}


def combination_label(groups: Sequence[str]) -> str:
    ordered = [group for group in DISPLAY_ORDER if group in set(groups)]
    return " + ".join(GROUP_LABELS[group] for group in ordered)


def feature_count(groups: Sequence[str]) -> int:
    return sum(GROUP_FEATURE_COUNTS[group] for group in set(groups))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def _metric_pair(row: dict[str, Any], metric: str) -> tuple[float, float] | None:
    for table in (METRIC_COLUMNS, FALLBACK_METRIC_COLUMNS):
        mean_key, std_key = table[metric]
        mean = _to_float(row.get(mean_key))
        if mean is not None:
            return mean, _to_float(row.get(std_key)) or 0.0
    return None


def read_proposed_runs(paths: Iterable[Path], metric: str) -> list[dict[str, Any]]:
    """Collect successful proposed-arm runs, one record per (model, combination)."""
    records: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Summary file not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("status") or "").strip() != "success":
                    continue
                if (row.get("feature_set") or "").strip() != "proposed":
                    continue
                raw_groups = (row.get("feature_groups") or "").strip()
                if not raw_groups or raw_groups == "profile_defined":
                    continue
                groups = tuple(
                    group
                    for group in FEATURE_GROUPS
                    if group in {part.strip() for part in raw_groups.split(",")}
                )
                pair = _metric_pair(row, metric)
                if pair is None:
                    continue
                mean, std = pair
                model_type = (row.get("model_type") or "").strip()
                records[(model_type, groups)] = {
                    "model_type": model_type,
                    "feature_groups": groups,
                    "label": combination_label(groups),
                    "n_features": feature_count(groups),
                    "mean": mean,
                    "std": std,
                    "tuning_metric": (row.get("tuning_metric") or "").strip()
                    or None,
                    "random_state": (row.get("random_state") or "").strip()
                    or None,
                    "selected_hyperparameters": row.get("selected_hyperparameters"),
                    "environment": _environment_of(row),
                }
    return list(records.values())


#: Columns written by ML.experiment_matrix.environment_record.
ENVIRONMENT_COLUMNS = (
    "python_version",
    "numpy_version",
    "pandas_version",
    "sklearn_version",
    "xgboost_version",
)


def _environment_of(row: dict[str, Any]) -> str | None:
    """Collapse a row's recorded library versions into one comparable string."""
    parts = [
        f"{name.removesuffix('_version')}={value.strip()}"
        for name in ENVIRONMENT_COLUMNS
        if (value := row.get(name) or "")
    ]
    return " ".join(parts) or None


def tuning_protocol(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the hyperparameter protocol phase 1 actually ran under.

    Phase 2 must tune with the same metric and seed, or its model is not the
    one phase 1 selected. The values are read back from the summary rows rather
    than assumed, and a mixed sweep is reported instead of silently averaged.
    """
    metrics = sorted({m for record in records if (m := record.get("tuning_metric"))})
    seeds = sorted({s for record in records if (s := record.get("random_state"))})
    recorded = [e for record in records if (e := record.get("environment"))]
    environments = sorted(set(recorded))
    # Rows predating version recording carry no environment. Ignoring them would
    # let a sweep that mixes recorded and unrecorded runs pass as uniform, which
    # is exactly the case where the versions are most likely to differ: the two
    # halves were produced at different times. So partial recording counts as a
    # mismatch, not as agreement.
    partially_recorded = 0 < len(recorded) < len(records)
    return {
        "tuning_metric": metrics[0] if len(metrics) == 1 else None,
        "tuning_metrics_seen": metrics,
        "random_state": seeds[0] if len(seeds) == 1 else None,
        "random_states_seen": seeds,
        "environment": (
            environments[0]
            if len(environments) == 1 and not partially_recorded
            else None
        ),
        "environments_seen": environments,
        "environment_recorded": len(recorded) == len(records) and bool(recorded),
        "environment_partially_recorded": partially_recorded,
        "is_uniform": (
            len(metrics) == 1
            and len(seeds) == 1
            and len(environments) <= 1
            and not partially_recorded
        ),
    }


def _apply_one_se(
    candidates: list[dict[str, Any]],
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    """Apply the one-standard-error rule to scored candidates."""
    if not candidates:
        raise ValueError("No candidates to select from.")
    best = max(candidates, key=lambda item: item["mean"])
    standard_error = best["std"] / math.sqrt(n_folds)
    threshold = best["mean"] - standard_error
    within = [item for item in candidates if item["mean"] >= threshold]
    # Fewest features first, then the steadier candidate, then a stable
    # alphabetical tiebreak so the result never depends on dict ordering.
    chosen = min(
        within,
        key=lambda item: (item["n_features"], item["std"], item["label"]),
    )
    return {
        "best_mean": best["mean"],
        "best_label": best["label"],
        "standard_error": standard_error,
        "threshold": threshold,
        "within_one_se": sorted(
            within, key=lambda item: (item["n_features"], item["label"])
        ),
        "chosen": chosen,
    }


def select_consensus(
    records: list[dict[str, Any]],
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    """Choose ONE combination for every classifier.

    Each combination is scored by its mean across classifiers; its spread is the
    root-mean-square of the per-classifier fold SDs, so a combination that is
    unstable for any classifier is penalised rather than averaged smooth.
    """
    by_combination: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        by_combination.setdefault(record["feature_groups"], []).append(record)

    candidates = []
    for groups, entries in by_combination.items():
        means = [entry["mean"] for entry in entries]
        stds = [entry["std"] for entry in entries]
        candidates.append(
            {
                "feature_groups": groups,
                "label": combination_label(groups),
                "n_features": feature_count(groups),
                "mean": sum(means) / len(means),
                "std": math.sqrt(sum(value**2 for value in stds) / len(stds)),
                "n_models": len(entries),
                "per_model": {
                    entry["model_type"]: entry["mean"] for entry in entries
                },
            }
        )
    return _apply_one_se(candidates, n_folds)


def select_per_model(
    records: list[dict[str, Any]],
    n_folds: int = N_FOLDS,
) -> dict[str, dict[str, Any]]:
    """Apply the rule separately inside each classifier."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model_type"], []).append(record)
    return {
        model_type: _apply_one_se(entries, n_folds)
        for model_type, entries in sorted(by_model.items())
    }


def _print_selection(title: str, result: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(
        f"  terbaik           : {result['best_label']} "
        f"(mean={result['best_mean']:.4f})"
    )
    print(
        f"  SE / ambang 1-SE  : {result['standard_error']:.4f} / "
        f"{result['threshold']:.4f}"
    )
    print(f"  dalam 1 SE        : {len(result['within_one_se'])} kombinasi")
    for item in result["within_one_se"]:
        mark = "<<<" if item["label"] == result["chosen"]["label"] else "   "
        print(
            f"    {mark} {item['label']:<28} {item['n_features']:>4} fitur  "
            f"mean={item['mean']:.4f}  sd={item['std']:.4f}"
        )
    print(f"  TERPILIH          : {result['chosen']['label']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the proposed feature combination from phase-1 internal CV "
            "using the one-standard-error rule."
        )
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="Phase-1 experiment_summary.csv file(s).",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_COLUMNS),
        default="roc_auc",
        help="Binding selection metric (default: roc_auc, as pre-registered).",
    )
    parser.add_argument(
        "--sensitivity-metric",
        choices=tuple(METRIC_COLUMNS),
        default="pr_auc",
        help="Second metric reported as a robustness check, never binding.",
    )
    parser.add_argument(
        "--scope",
        choices=("consensus", "per_model"),
        default="consensus",
        help=(
            "consensus (default) freezes ONE combination used by every "
            "classifier, which is what the main comparison table needs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the frozen configuration as JSON here, then commit it.",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    records = read_proposed_runs(args.summaries, args.metric)
    if not records:
        raise SystemExit(
            "No successful proposed-arm runs with the requested metric were "
            "found. Has phase 1 finished?"
        )
    models = sorted({record["model_type"] for record in records})
    combinations = {record["feature_groups"] for record in records}
    print(f"Membaca {len(records)} run proposed: "
          f"{len(models)} classifier x {len(combinations)} kombinasi")
    if len(combinations) != 15:
        print(
            f"PERINGATAN: ditemukan {len(combinations)} kombinasi, bukan 15. "
            "Fase 1 mungkin belum selesai."
        )

    consensus = select_consensus(records)
    _print_selection(f"KONSENSUS ({args.metric}) - mengikat", consensus)

    per_model = select_per_model(records)
    print(f"\nPER CLASSIFIER ({args.metric}) - informatif")
    for model_type, result in per_model.items():
        print(f"    {model_type:<22} -> {result['chosen']['label']}")

    sensitivity = None
    if args.sensitivity_metric != args.metric:
        other = read_proposed_runs(args.summaries, args.sensitivity_metric)
        if other:
            sensitivity = select_consensus(other)
            agrees = (
                sensitivity["chosen"]["feature_groups"]
                == consensus["chosen"]["feature_groups"]
            )
            print(f"\nSENSITIVITAS ({args.sensitivity_metric}) - tidak mengikat")
            print(f"    terpilih: {sensitivity['chosen']['label']}")
            print(
                "    "
                + (
                    "SEPAKAT dengan metrik utama."
                    if agrees
                    else "BERBEDA dari metrik utama - laporkan keduanya di naskah."
                )
            )

    chosen_groups = list(consensus["chosen"]["feature_groups"])
    protocol = tuning_protocol(records)
    if not protocol["is_uniform"]:
        print(
            "\nPERINGATAN: baris Fase 1 tidak dihasilkan di bawah satu protokol.\n"
            f"  metrik tuning terlihat : {protocol['tuning_metrics_seen']}\n"
            f"  random_state terlihat  : {protocol['random_states_seen']}"
        )
        if len(protocol["environments_seen"]) > 1:
            print(
                "  LINGKUNGAN BERBEDA -- versi pustaka menentukan pembagian fold,\n"
                "  jadi run di bawah versi berbeda TIDAK sebanding dan tidak boleh\n"
                "  digabung dalam satu aturan 1-SE:"
            )
            for environment in protocol["environments_seen"]:
                print(f"    {environment}")
        if protocol["environment_partially_recorded"]:
            print(
                "  SEBAGIAN RUN TIDAK MENCATAT VERSI PUSTAKA. Run yang mencatat\n"
                "  dan yang tidak berasal dari waktu berbeda, jadi justru di situ\n"
                "  versinya paling mungkin berbeda. Tercatat:\n"
                f"    {protocol['environments_seen'][0] if protocol['environments_seen'] else '-'}"
            )
        print("  Jalankan ulang seluruh Fase 1 di satu lingkungan.")
    elif not protocol["environment_recorded"]:
        print(
            "\nCatatan: summary ini dibuat sebelum versi pustaka dicatat, jadi "
            "keseragaman lingkungan tidak bisa diperiksa."
        )
    # The hyperparameters phase 1 chose for the frozen combination. Phase 2
    # re-derives these from the same data, so they double as a check that the
    # two phases really ran one protocol.
    chosen_key = consensus["chosen"]["feature_groups"]
    phase1_hyperparameters = {
        record["model_type"]: record.get("selected_hyperparameters")
        for record in records
        if record["feature_groups"] == chosen_key
    }
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "rule": "one_standard_error",
        "rule_reference": "Hastie & Tibshirani, Elements of Statistical Learning, 7.10",
        "n_folds": N_FOLDS,
        "binding_metric": args.metric,
        "phase1_tuning_metric": protocol["tuning_metric"],
        "phase1_random_state": protocol["random_state"],
        "phase1_environment": protocol["environment"],
        "phase1_environments_seen": protocol["environments_seen"],
        "phase1_protocol_is_uniform": protocol["is_uniform"],
        "phase1_selected_hyperparameters": phase1_hyperparameters,
        "scope": args.scope,
        "source_summaries": [str(path.resolve()) for path in args.summaries],
        "models": models,
        "n_combinations_seen": len(combinations),
        "selected_feature_groups": chosen_groups,
        "selected_label": consensus["chosen"]["label"],
        "selected_n_features": consensus["chosen"]["n_features"],
        "selected_mean": consensus["chosen"]["mean"],
        "selected_std": consensus["chosen"]["std"],
        "best_mean": consensus["best_mean"],
        "best_label": consensus["best_label"],
        "standard_error": consensus["standard_error"],
        "one_se_threshold": consensus["threshold"],
        "within_one_se": [
            {
                "label": item["label"],
                "n_features": item["n_features"],
                "mean": item["mean"],
                "std": item["std"],
            }
            for item in consensus["within_one_se"]
        ],
        "per_model_choice": {
            model_type: result["chosen"]["label"]
            for model_type, result in per_model.items()
        },
        "sensitivity_metric": args.sensitivity_metric,
        "sensitivity_choice": (
            sensitivity["chosen"]["label"] if sensitivity else None
        ),
        "sensitivity_agrees": (
            sensitivity["chosen"]["feature_groups"]
            == consensus["chosen"]["feature_groups"]
            if sensitivity
            else None
        ),
    }

    print("\n" + "=" * 70)
    print(f"KONFIGURASI BEKU: {frozen['selected_label']} "
          f"({frozen['selected_n_features']} fitur)")
    print("=" * 70)
    print("\nPerintah Fase 2:\n")
    # --tuning-metric is passed explicitly: leaving it out would let phase 2
    # fall back to the CFG default and tune against a different metric than
    # phase 1 did, without anything in the output saying so.
    metric_flag = (
        f" \\\n    --tuning-metric {protocol['tuning_metric']}"
        if protocol["tuning_metric"]
        else ""
    )
    print(
        "python -u -m ML.experiment_matrix --output-dir ML/artifacts/phase2 \\\n"
        "    --training-mode external_sars \\\n"
        f"    --feature-groups {' '.join(chosen_groups)}{metric_flag}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nKonfigurasi beku ditulis ke: {args.output.resolve()}")
        print("Commit file ini SEBELUM menjalankan Fase 2.")


if __name__ == "__main__":
    main()
