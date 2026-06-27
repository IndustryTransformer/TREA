"""Build an apples-to-apples 3W benchmark report from benchmark outputs.

This script merges:
1) Internal benchmark outputs from `examples/benchmark_3w_noninferiority.py`
2) A structured literature table (`docs/3w_literature_benchmarks.csv`)

It generates a single markdown report with:
- protocol card (from config.json when available),
- internal model ranking (same split/protocol),
- non-inferiority summary,
- per-class F1 comparison,
- literature rows split into comparable vs non-comparable.
"""

from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build apples-to-apples 3W report")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        required=True,
        help=(
            "Directory containing runs.csv, summary.csv, noninferiority.csv "
            "(and optionally config.json)."
        ),
    )
    parser.add_argument(
        "--literature-csv",
        type=str,
        default="docs/3w_literature_benchmarks.csv",
    )
    parser.add_argument(
        "--candidate-model",
        type=str,
        default=None,
        help="Optional override for candidate model in report sections.",
    )
    parser.add_argument(
        "--our-class-count",
        type=int,
        default=10,
        help="Class count expected for strict comparability.",
    )
    parser.add_argument(
        "--our-primary-metric",
        type=str,
        default="macro_f1",
        help="Primary metric name for strict comparability.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output markdown path. Defaults to "
            "<benchmark-dir>/apples_to_apples_report.md."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float | int):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _to_int(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_benchmark_data(benchmark_dir: Path) -> dict[str, Any]:
    summary_path = benchmark_dir / "summary.csv"
    runs_path = benchmark_dir / "runs.csv"
    noninf_path = benchmark_dir / "noninferiority.csv"
    config_path = benchmark_dir / "config.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs.csv: {runs_path}")

    summary_rows = _read_csv(summary_path)
    runs_rows = _read_csv(runs_path)
    noninf_rows = _read_csv(noninf_path) if noninf_path.exists() else []
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else None
    )

    for row in summary_rows:
        row["val_f1_macro_mean"] = _to_float(row.get("val_f1_macro_mean"))
        row["val_f1_macro_ci95_low"] = _to_float(row.get("val_f1_macro_ci95_low"))
        row["val_f1_macro_ci95_high"] = _to_float(row.get("val_f1_macro_ci95_high"))
        row["val_acc_mean"] = _to_float(row.get("val_acc_mean"))
        row["n_runs"] = _to_int(row.get("n_runs"))
        row["train_seconds_mean"] = _to_float(row.get("train_seconds_mean"))

    summary_rows.sort(key=lambda r: r["val_f1_macro_mean"], reverse=True)

    for row in noninf_rows:
        row["delta_mean"] = _to_float(row.get("delta_mean"))
        row["delta_ci95_low"] = _to_float(row.get("delta_ci95_low"))
        row["delta_ci95_high"] = _to_float(row.get("delta_ci95_high"))
        row["noninferior"] = _to_bool(row.get("noninferior"))
        row["superior"] = _to_bool(row.get("superior"))

    return {
        "summary": summary_rows,
        "runs": runs_rows,
        "noninferiority": noninf_rows,
        "config": config,
    }


def infer_candidate_model(
    noninf_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    candidate_override: str | None,
) -> str:
    if candidate_override:
        return candidate_override
    if noninf_rows and noninf_rows[0].get("candidate_model"):
        return str(noninf_rows[0]["candidate_model"])
    if summary_rows:
        return str(summary_rows[0]["model"])
    return "unknown"


def per_class_means_by_model(
    runs_rows: list[dict[str, str]],
) -> dict[str, dict[int, float]]:
    buckets: dict[str, dict[int, list[float]]] = {}
    for row in runs_rows:
        model = row.get("model", "")
        model_bucket = buckets.setdefault(model, {})
        for key, raw in row.items():
            if not key.startswith("f1_class_"):
                continue
            cls = int(key.split("_")[-1])
            model_bucket.setdefault(cls, []).append(_to_float(raw))

    out: dict[str, dict[int, float]] = {}
    for model, cls_map in buckets.items():
        out[model] = {}
        for cls, values in cls_map.items():
            if values:
                out[model][cls] = sum(values) / len(values)
    return out


def load_literature_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    for row in rows:
        row["year"] = _to_int(row.get("year"))
        row["metric_value"] = _to_float(row.get("metric_value"))
        row["class_count"] = _to_int(row.get("class_count"))
        row["uses_feature_engineering"] = _to_bool(row.get("uses_feature_engineering"))
        row["comparable_to_our_10class_macrof1"] = _to_bool(
            row.get("comparable_to_our_10class_macrof1")
        )
    return rows


def split_literature_rows(
    rows: list[dict[str, Any]], our_class_count: int, our_primary_metric: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparable: list[dict[str, Any]] = []
    non_comparable: list[dict[str, Any]] = []

    for row in rows:
        metric_name = str(row.get("metric_name", "")).strip().lower()
        strict_match = (
            str(row.get("task_family", "")).strip().lower() == "multiclass"
            and row.get("class_count", 0) == our_class_count
            and metric_name == our_primary_metric.lower()
        )
        explicit = bool(row.get("comparable_to_our_10class_macrof1", False))
        if strict_match or explicit:
            comparable.append(row)
        else:
            non_comparable.append(row)

    comparable.sort(key=lambda r: r["metric_value"], reverse=True)
    non_comparable.sort(key=lambda r: r["metric_value"], reverse=True)
    return comparable, non_comparable


def format_protocol_card(config: dict[str, Any] | None) -> list[str]:
    lines = ["## Protocol Card", ""]
    if not config:
        lines.append("No `config.json` found; protocol details are partially unknown.")
        lines.append("")
        return lines

    args = config.get("args", {})
    data = config.get("dataset_info", {})
    lines.extend(
        [
            f"- Created at (UTC): `{config.get('created_at_utc', 'unknown')}`",
            f"- Data split seed: `{args.get('data_split_seed', 'unknown')}`",
            f"- Window/stride: `{args.get('window_size', 'unknown')}` / "
            f"`{args.get('stride', 'unknown')}`",
            f"- Val fraction: `{args.get('val_fraction', 'unknown')}`",
            f"- Number of classes: `{data.get('num_classes', 'unknown')}`",
            f"- Train/val windows: `{data.get('num_train_samples', 'unknown')}` / "
            f"`{data.get('num_val_samples', 'unknown')}`",
            f"- Models benchmarked: `{', '.join(config.get('models', []))}`",
            f"- Seeds: `{config.get('seeds', [])}`",
        ]
    )
    lines.append("")
    return lines


def build_report(
    benchmark_dir: Path,
    summary_rows: list[dict[str, Any]],
    runs_rows: list[dict[str, str]],
    noninf_rows: list[dict[str, Any]],
    config: dict[str, Any] | None,
    literature_rows: list[dict[str, Any]],
    candidate_model: str,
    our_class_count: int,
    our_primary_metric: str,
) -> str:
    lines: list[str] = []
    lines.append("# 3W Apples-to-Apples Report")
    lines.append("")
    lines.append(f"Benchmark directory: `{benchmark_dir}`")
    lines.append(f"Candidate model: `{candidate_model}`")
    lines.append(
        f"Comparability target: `{our_class_count}` classes + `{our_primary_metric}`"
    )
    lines.append("")

    lines.extend(format_protocol_card(config))

    lines.append("## Internal Benchmark (Same Protocol)")
    lines.append("")
    lines.append(
        "| Model | Runs | Macro-F1 mean | Macro-F1 CI95 | Acc mean | Sec/run |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['model']} | {row['n_runs']} | "
            f"{row['val_f1_macro_mean']:.4f} | "
            f"[{row['val_f1_macro_ci95_low']:.4f}, "
            f"{row['val_f1_macro_ci95_high']:.4f}] | "
            f"{row['val_acc_mean']:.4f} | "
            f"{row['train_seconds_mean']:.1f} |"
        )
    lines.append("")

    if noninf_rows:
        lines.append("## Non-Inferiority Summary")
        lines.append("")
        lines.append(
            "| Comparator | Δ mean (candidate - comparator) | Δ CI95 | Non-inferior |"
        )
        lines.append("|---|---:|---:|---:|")
        for row in noninf_rows:
            lines.append(
                "| "
                f"{row['comparator_model']} | "
                f"{row['delta_mean']:+.4f} | "
                f"[{row['delta_ci95_low']:+.4f}, {row['delta_ci95_high']:+.4f}] | "
                f"{row['noninferior']} |"
            )
        lines.append("")

    per_class = per_class_means_by_model(runs_rows)
    if candidate_model in per_class and summary_rows:
        best_comp = None
        for row in summary_rows:
            model = str(row["model"])
            if model != candidate_model:
                best_comp = model
                break
        if best_comp and best_comp in per_class:
            lines.append("## Per-Class F1 Means")
            lines.append("")
            lines.append(
                f"Candidate `{candidate_model}` vs best other model `{best_comp}`."
            )
            lines.append("")
            lines.append("| Class | Candidate F1 | Comparator F1 | Delta |")
            lines.append("|---:|---:|---:|---:|")
            class_ids = sorted(
                set(per_class[candidate_model]) | set(per_class[best_comp])
            )
            for cls in class_ids:
                cand = per_class[candidate_model].get(cls, float("nan"))
                comp = per_class[best_comp].get(cls, float("nan"))
                delta = cand - comp
                lines.append(f"| {cls} | {cand:.4f} | {comp:.4f} | {delta:+.4f} |")
            lines.append("")

    comparable, non_comparable = split_literature_rows(
        literature_rows,
        our_class_count=our_class_count,
        our_primary_metric=our_primary_metric,
    )

    lines.append("## Literature: Strictly Comparable Rows")
    lines.append("")
    if comparable:
        lines.append("| Work | Year | Metric | Value | FE | Notes |")
        lines.append("|---|---:|---|---:|---:|---|")
        for row in comparable:
            lines.append(
                "| "
                f"{row['work']} | {row['year']} | "
                f"{row['metric_name']} | {row['metric_value']:.4f} | "
                f"{row['uses_feature_engineering']} | {row['notes']} |"
            )
    else:
        lines.append("No strict apples-to-apples literature rows found in the table.")
    lines.append("")

    lines.append("## Literature: Not Directly Comparable")
    lines.append("")
    lines.append("| Work | Task | Metric | Value | Why not directly comparable |")
    lines.append("|---|---|---|---:|---|")
    for row in non_comparable:
        lines.append(
            "| "
            f"{row['work']} ({row['year']}) | "
            f"{row['task_detail']} | "
            f"{row['metric_name']} | {row['metric_value']:.4f} | "
            f"{row['notes']} |"
        )
    lines.append("")

    if summary_rows:
        best_internal = summary_rows[0]
        lines.append("## Conclusion")
        lines.append("")
        lines.append(
            f"- Best internal model on this protocol: `{best_internal['model']}` "
            f"with macro-F1 `{best_internal['val_f1_macro_mean']:.4f}`."
        )
        lines.append(
            "- Use this internal table for final model claims. "
            "Use literature table only with strict comparability checks."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    lit_path = Path(args.literature_csv)
    if not lit_path.exists():
        raise FileNotFoundError(f"Literature CSV not found: {lit_path}")

    data = load_benchmark_data(benchmark_dir)
    summary_rows = data["summary"]
    runs_rows = data["runs"]
    noninf_rows = data["noninferiority"]
    config = data["config"]
    literature_rows = load_literature_rows(lit_path)

    candidate_model = infer_candidate_model(
        noninf_rows=noninf_rows,
        summary_rows=summary_rows,
        candidate_override=args.candidate_model,
    )

    report_text = build_report(
        benchmark_dir=benchmark_dir,
        summary_rows=summary_rows,
        runs_rows=runs_rows,
        noninf_rows=noninf_rows,
        config=config,
        literature_rows=literature_rows,
        candidate_model=candidate_model,
        our_class_count=args.our_class_count,
        our_primary_metric=args.our_primary_metric,
    )

    output_path = (
        Path(args.output)
        if args.output
        else benchmark_dir / "apples_to_apples_report.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
