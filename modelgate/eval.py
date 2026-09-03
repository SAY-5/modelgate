"""Offline evaluation: replay a request log against model versions.

    modelgate eval --log requests.jsonl --versions v1 v2 [--truth auto|reference|none]

Each log record is validated with the same schema the API uses, encoded with
the same feature code, and run through each version's padded forward pass,
so a replay reproduces what the service answered bit for bit and is
deterministic across runs. The report covers, per version, MAE against the
truth source and a calibration table (mean predicted versus mean actual per
predicted-ETA bucket, with the expected calibration error); and, for a pair
of versions, the divergence between their answers using the same statistics
as the shadow report. Records that carry the version they were served by are
also checked against their logged ETA, which catches a serving-time or
artifact drift between the log and the replay.

Truth sources: `auto` uses an `actual_eta_minutes` field when a record has
one; `reference` uses the synthetic formula the training data was built from
(useful for the committed fixtures); `none` skips MAE and calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from pydantic import ValidationError

from modelgate.model.data import true_eta_minutes
from modelgate.model.features import encode
from modelgate.serving.registry import ModelRegistry
from modelgate.serving.schemas import PredictRequest
from modelgate.serving.shadow import _percentile

CALIBRATION_EDGES: tuple[float, ...] = (5, 10, 15, 20, 30, 45, 60, 90)


def load_log(path: Path) -> tuple[list[dict], dict]:
    """Read JSON lines, validate inputs, and return (records, skipped counts)."""
    records: list[dict] = []
    skipped = {"malformed": 0, "invalid_input": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                payload = raw["input"]
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped["malformed"] += 1
                continue
            try:
                req = PredictRequest.model_validate(payload)
            except ValidationError:
                skipped["invalid_input"] += 1
                continue
            records.append({**raw, "input": req.model_dump()})
    return records, skipped


def truth_values(records: list[dict], mode: str) -> list[float | None]:
    if mode == "none":
        return [None] * len(records)
    if mode == "reference":
        return [true_eta_minutes(**r["input"]) for r in records]
    if mode == "auto":
        out: list[float | None] = []
        for r in records:
            value = r.get("actual_eta_minutes")
            out.append(float(value) if isinstance(value, int | float) else None)
        return out
    raise ValueError(f"unknown truth mode {mode!r}")


def replay(registry: ModelRegistry, versions: list[str], records: list[dict]) -> dict:
    """Run every record through each version. Returns {version: [eta, ...]}."""
    if not records:
        return {v: [] for v in versions}
    x = torch.tensor([encode(**r["input"]) for r in records], dtype=torch.float32)
    return {v: registry.load(v).predict_batch(x) for v in versions}


def _bucket(value: float) -> str:
    lo = 0.0
    for edge in CALIBRATION_EDGES:
        if value < edge:
            return f"{lo:g}-{edge:g}"
        lo = edge
    return f"{lo:g}+"


def accuracy_report(preds: list[float], truths: list[float | None]) -> dict:
    pairs = [(p, t) for p, t in zip(preds, truths, strict=True) if t is not None]
    if not pairs:
        return {"n": 0, "mae_minutes": None, "bias_minutes": None, "calibration": None}
    errors = [abs(p - t) for p, t in pairs]
    buckets: dict[str, list[tuple[float, float]]] = {}
    for p, t in pairs:
        buckets.setdefault(_bucket(p), []).append((p, t))
    rows = []
    ece = 0.0
    for name, items in buckets.items():
        mp = sum(p for p, _ in items) / len(items)
        mt = sum(t for _, t in items) / len(items)
        rows.append(
            {
                "bucket": name,
                "n": len(items),
                "mean_predicted": round(mp, 3),
                "mean_actual": round(mt, 3),
                "gap": round(mp - mt, 3),
            }
        )
        ece += abs(mp - mt) * len(items) / len(pairs)
    rows.sort(key=lambda r: float(r["bucket"].split("-")[0].rstrip("+")))
    within = sum(1 for e in errors if e <= 2.0) / len(pairs)
    return {
        "n": len(pairs),
        "mae_minutes": round(sum(errors) / len(errors), 4),
        "p95_abs_error_minutes": round(_percentile(errors, 95), 4),
        "bias_minutes": round(sum(p - t for p, t in pairs) / len(pairs), 4),
        "share_within_2min": round(within, 4),
        "calibration": {"expected_calibration_error": round(ece, 4), "buckets": rows},
    }


def divergence_report(a: list[float], b: list[float], threshold: float = 2.0) -> dict:
    deltas = [y - x for x, y in zip(a, b, strict=True)]
    abs_deltas = [abs(d) for d in deltas]
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    rel = [d / max(x, 1e-6) for d, x in zip(abs_deltas, a, strict=True)]
    return {
        "n": n,
        "threshold_minutes": threshold,
        "abs_delta_minutes": {
            "mean": round(sum(abs_deltas) / n, 4),
            "p50": round(_percentile(abs_deltas, 50), 4),
            "p95": round(_percentile(abs_deltas, 95), 4),
            "max": round(max(abs_deltas), 4),
        },
        "rel_delta": {"mean": round(sum(rel) / n, 4), "p95": round(_percentile(rel, 95), 4)},
        "share_beyond_threshold": round(sum(1 for d in abs_deltas if d > threshold) / n, 4),
        "bias_minutes": round(sum(deltas) / n, 4),
        "agreement_within_1min": round(sum(1 for d in abs_deltas if d <= 1.0) / n, 4),
    }


def logged_consistency(records: list[dict], preds: dict[str, list[float]]) -> dict:
    """Compare replayed answers with what the log says was served, where the version matches."""
    checked = 0
    mismatches = 0
    worst = 0.0
    for i, r in enumerate(records):
        version = r.get("version")
        logged = r.get("eta_minutes")
        if version not in preds or not isinstance(logged, int | float):
            continue
        checked += 1
        gap = abs(round(preds[version][i], 2) - float(logged))
        if gap > 0.0:
            mismatches += 1
            worst = max(worst, gap)
    return {"checked": checked, "mismatches": mismatches, "max_gap_minutes": round(worst, 4)}


def build_report(
    records: list[dict],
    preds: dict[str, list[float]],
    truths: list[float | None],
    skipped: dict,
    threshold: float = 2.0,
) -> dict:
    versions = list(preds)
    report = {
        "records": len(records),
        "skipped": skipped,
        "truth_available": sum(1 for t in truths if t is not None),
        "versions": {v: accuracy_report(preds[v], truths) for v in versions},
        "consistency": logged_consistency(records, preds),
        "divergence": None,
    }
    if len(versions) >= 2:
        a, b = versions[0], versions[1]
        report["divergence"] = {
            "from": a,
            "to": b,
            **divergence_report(preds[a], preds[b], threshold),
        }
    return report


def format_report(report: dict) -> str:
    lines = ["=" * 64, "ModelGate replay evaluation", "=" * 64]
    lines.append(
        f"records {report['records']}  skipped {report['skipped']}  "
        f"with truth {report['truth_available']}"
    )
    for version, acc in report["versions"].items():
        if acc["n"] == 0:
            lines.append(f"{version:<6} no truth available; MAE not computed")
            continue
        lines.append(
            f"{version:<6} n={acc['n']}  MAE {acc['mae_minutes']} min  "
            f"p95|err| {acc['p95_abs_error_minutes']}  bias {acc['bias_minutes']}  "
            f"within 2 min {acc['share_within_2min']}  "
            f"ECE {acc['calibration']['expected_calibration_error']}"
        )
        for row in acc["calibration"]["buckets"]:
            lines.append(
                f"       {row['bucket']:>7} min  n={row['n']:<5} "
                f"predicted {row['mean_predicted']:>7}  actual {row['mean_actual']:>7}  "
                f"gap {row['gap']:>7}"
            )
    con = report["consistency"]
    lines.append(
        f"logged vs replayed: checked {con['checked']}, mismatches {con['mismatches']}, "
        f"max gap {con['max_gap_minutes']} min"
    )
    div = report["divergence"]
    if div and div.get("n"):
        lines.append(
            f"divergence {div['from']} -> {div['to']}: n={div['n']} "
            f"mean|d| {div['abs_delta_minutes']['mean']} p95|d| {div['abs_delta_minutes']['p95']} "
            f"max {div['abs_delta_minutes']['max']} beyond {div['threshold_minutes']} min "
            f"{div['share_beyond_threshold']} bias {div['bias_minutes']} "
            f"agree<=1min {div['agreement_within_1min']}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def evaluate(
    log_path: Path,
    versions: list[str],
    artifacts_dir: Path = Path("artifacts"),
    truth: str = "auto",
    threshold: float = 2.0,
) -> dict:
    started = time.perf_counter()
    registry = ModelRegistry(artifacts_dir)
    for v in versions:
        if not registry.is_known(v):
            raise SystemExit(f"unknown version {v!r}; available: {registry.available_versions()}")
    records, skipped = load_log(log_path)
    preds = replay(registry, versions, records)
    truths = truth_values(records, truth)
    report = build_report(records, preds, truths, skipped, threshold)
    report["log"] = str(log_path)
    report["truth"] = truth
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelgate eval", description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="JSON-lines request log")
    parser.add_argument("--versions", nargs="+", required=True, help="versions to replay")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--truth", choices=("auto", "reference", "none"), default="auto")
    parser.add_argument("--threshold", type=float, default=2.0, help="divergence threshold")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = parser.parse_args(argv)
    report = evaluate(args.log, args.versions, args.artifacts, args.truth, args.threshold)
    print(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
