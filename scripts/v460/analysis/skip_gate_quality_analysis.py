from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from scripts.v460.analysis.analysis_common import Record, get_pnl
from ztb.utils.safety import safe_to_finite

DEFAULT_SKIP_GATE_MODEL_PATH = Path("models/v460/skip_gate_lgbm_pnl120.pkl")
DEFAULT_CONFIG_PATH = Path("configs/v460/fill_test.yaml")


@dataclass(frozen=True)
class SkipGateQualityReport:
    text_report: str
    json_payload: dict[str, object]
    warnings: list[str]


def _record_float(record: Record, key: str) -> float | None:
    value = safe_to_finite(record.get(key))
    return float(value) if value is not None else None


def _record_bool(record: Record, key: str) -> bool:
    return bool(record.get(key))


def _score_records(records: Sequence[Record]) -> list[Record]:
    return [record for record in records if _record_float(record, "skip_gate_score") is not None]


def _filled_records(records: Sequence[Record]) -> list[Record]:
    return [record for record in records if _record_bool(record, "filled")]


def _moments(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "skew": None,
            "kurtosis": None,
            "bimodality_coefficient": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    centered = arr - mean
    var = float(np.mean(centered ** 2))
    std = math.sqrt(var) if var > 0.0 else 0.0
    skew: float | None = None
    kurtosis: float | None = None
    bc: float | None = None
    if std > 0.0:
        m3 = float(np.mean(centered ** 3))
        m4 = float(np.mean(centered ** 4))
        skew = m3 / (std ** 3)
        kurtosis = m4 / (var ** 2)
        if kurtosis > 0.0:
            bc = float((skew * skew + 1.0) / kurtosis)
    return {
        "mean": mean,
        "std": float(arr.std()),
        "skew": skew,
        "kurtosis": kurtosis,
        "bimodality_coefficient": bc,
    }


def _histogram(values: Sequence[float]) -> dict[str, object]:
    bins = np.linspace(-2.0, 2.0, 21)
    hist, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return {
        "range": [-2.0, 2.0],
        "bins": 20,
        "edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in hist.tolist()],
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + end - 1) / 2.0 + 1.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = avg_rank
        idx = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _quintile_means(xs: Sequence[float], ys: Sequence[float]) -> list[dict[str, object]]:
    if len(xs) != len(ys) or not xs:
        return []
    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    bucket_size = max(1, math.ceil(len(pairs) / 5))
    output: list[dict[str, object]] = []
    for quintile_idx in range(5):
        chunk = pairs[quintile_idx * bucket_size:(quintile_idx + 1) * bucket_size]
        if not chunk:
            continue
        scores = [score for score, _ in chunk]
        pnls = [pnl for _, pnl in chunk]
        output.append(
            {
                "quintile": quintile_idx + 1,
                "count": len(chunk),
                "score_min": float(min(scores)),
                "score_max": float(max(scores)),
                "avg_pnl_bps": float(statistics.fmean(pnls)),
            }
        )
    return output


def _counterfactual_thresholds(records: Sequence[Record], thresholds: Sequence[float]) -> list[dict[str, object]]:
    filled = _filled_records(_score_records(records))
    output: list[dict[str, object]] = []
    for threshold in thresholds:
        kept: list[Record] = []
        removed = 0
        for record in filled:
            score = _record_float(record, "skip_gate_score")
            if score is None:
                continue
            forced = _record_bool(record, "skip_gate_forced_pass")
            if score < threshold and not forced:
                removed += 1
                continue
            kept.append(record)
        kept_pnls = [pnl for record in kept if (pnl := get_pnl(record)) is not None]
        output.append(
            {
                "threshold": float(threshold),
                "kept_count": len(kept),
                "removed_count": removed,
                "avg_pnl_bps": float(statistics.fmean(kept_pnls)) if kept_pnls else None,
            }
        )
    return output


def _feature_shift(records: Sequence[Record]) -> dict[str, dict[str, float | int | None]]:
    features = {
        "spread_bps": [],
        "orderbook_imbalance": [],
        "price_velocity_bps": [],
    }
    regime_counts: dict[str, int] = {}
    for record in records:
        for key in features:
            value = _record_float(record, key)
            if value is not None:
                features[key].append(value)
        regime = record.get("regime")
        if isinstance(regime, str):
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
    payload: dict[str, dict[str, float | int | None]] = {}
    for key, values in features.items():
        moments = _moments(values)
        payload[key] = {
            "count": len(values),
            "mean": moments["mean"],
            "std": moments["std"],
        }
    payload["regime_counts"] = {key: int(value) for key, value in sorted(regime_counts.items())}
    return payload


def _extract_model_path(config_path: Path) -> Path:
    if not config_path.exists():
        return DEFAULT_SKIP_GATE_MODEL_PATH
    with config_path.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        return DEFAULT_SKIP_GATE_MODEL_PATH
    skip_gate = payload.get("skip_gate")
    if not isinstance(skip_gate, dict):
        return DEFAULT_SKIP_GATE_MODEL_PATH
    model_path = skip_gate.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        return Path(model_path)
    return DEFAULT_SKIP_GATE_MODEL_PATH


def _model_info(config_path: Path) -> dict[str, object]:
    model_path = _extract_model_path(config_path)
    if model_path.exists():
        stat = model_path.stat()
        return {
            "model_path": str(model_path),
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime_epoch": float(stat.st_mtime),
        }
    return {"model_path": str(model_path), "exists": False}


def build_skip_gate_quality_report(
    records: Sequence[Record],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> SkipGateQualityReport:
    pre = [record for record in records if str(record.get("date_str", ""))[:10] in {"2026-04-01", "2026-04-02", "2026-04-03"}]
    post = [record for record in records if str(record.get("date_str", ""))[:10] in {"2026-04-04", "2026-04-05", "2026-04-06"}]
    if not pre or not post:
        # fallback to timestamp sort split when date_str is absent in tests
        scored = list(records)
        midpoint = len(scored) // 2
        pre = scored[:midpoint]
        post = scored[midpoint:]

    pre_scored = _score_records(pre)
    post_scored = _score_records(post)
    pre_scores = [_record_float(record, "skip_gate_score") for record in pre_scored]
    post_scores = [_record_float(record, "skip_gate_score") for record in post_scored]
    pre_scores = [float(value) for value in pre_scores if value is not None]
    post_scores = [float(value) for value in post_scores if value is not None]

    pre_pairs = [
        (_record_float(record, "skip_gate_score"), get_pnl(record))
        for record in _filled_records(pre_scored)
    ]
    post_pairs = [
        (_record_float(record, "skip_gate_score"), get_pnl(record))
        for record in _filled_records(post_scored)
    ]
    pre_xy = [(score, pnl) for score, pnl in pre_pairs if score is not None and pnl is not None]
    post_xy = [(score, pnl) for score, pnl in post_pairs if score is not None and pnl is not None]

    bypass_reasons: dict[str, int] = {}
    forced_reasons: dict[str, int] = {}
    for record in post_scored:
        reason = str(record.get("skip_gate_reason") or "NONE")
        if _record_bool(record, "skip_gate_bypassed"):
            bypass_reasons[reason] = bypass_reasons.get(reason, 0) + 1
        if _record_bool(record, "skip_gate_forced_pass"):
            forced_reasons[reason] = forced_reasons.get(reason, 0) + 1

    payload = {
        "analysis": "708_skip_gate_quality",
        "pre": {
            "count": len(pre_scores),
            "histogram": _histogram(pre_scores),
            "moments": _moments(pre_scores),
            "correlation": {
                "pearson": _pearson([x for x, _ in pre_xy], [y for _, y in pre_xy]),
                "spearman": _spearman([x for x, _ in pre_xy], [y for _, y in pre_xy]),
            },
            "quintiles": _quintile_means([x for x, _ in pre_xy], [y for _, y in pre_xy]),
            "feature_shift_inputs": _feature_shift(pre),
        },
        "post": {
            "count": len(post_scores),
            "histogram": _histogram(post_scores),
            "moments": _moments(post_scores),
            "correlation": {
                "pearson": _pearson([x for x, _ in post_xy], [y for _, y in post_xy]),
                "spearman": _spearman([x for x, _ in post_xy], [y for _, y in post_xy]),
            },
            "quintiles": _quintile_means([x for x, _ in post_xy], [y for _, y in post_xy]),
            "feature_shift_inputs": _feature_shift(post),
            "counterfactual_thresholds": _counterfactual_thresholds(post, [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]),
        },
        "bypass": {
            "reasons": dict(sorted(bypass_reasons.items(), key=lambda item: (-item[1], item[0]))),
        },
        "forced_pass": {
            "reasons": dict(sorted(forced_reasons.items(), key=lambda item: (-item[1], item[0]))),
        },
        "model": _model_info(config_path),
    }

    post_mean = payload["post"]["moments"]["mean"]
    pre_mean = payload["pre"]["moments"]["mean"]
    warnings: list[str] = []
    if isinstance(pre_mean, float) and isinstance(post_mean, float) and post_mean < pre_mean:
        warnings.append("post skip-gate score mean is below pre period")
    if forced_reasons:
        top_reason = max(forced_reasons.items(), key=lambda item: item[1])[0]
        warnings.append(f"forced_pass is dominated by {top_reason}")

    text_report = "\n".join(
        [
            "708# skip_gate quality analysis",
            f"- pre score mean/std: {payload['pre']['moments']['mean']} / {payload['pre']['moments']['std']}",
            f"- post score mean/std: {payload['post']['moments']['mean']} / {payload['post']['moments']['std']}",
            f"- post bimodality coefficient: {payload['post']['moments']['bimodality_coefficient']}",
            f"- post pearson(score,pnl): {payload['post']['correlation']['pearson']}",
            f"- post spearman(score,pnl): {payload['post']['correlation']['spearman']}",
            f"- model path: {payload['model']['model_path']}",
            f"- warnings: {', '.join(warnings) if warnings else 'none'}",
        ]
    )
    return SkipGateQualityReport(text_report=text_report, json_payload=payload, warnings=warnings)
