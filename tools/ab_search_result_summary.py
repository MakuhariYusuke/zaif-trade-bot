#!/usr/bin/env python3
import json
from pathlib import Path
from typing import TypedDict

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    list_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict, safe_to_float


class ActionAverage(TypedDict):
    HOLD: float
    BUY: float
    SELL: float


class BalanceSearchSummary(TypedDict):
    skew: object
    balance: object
    avg: ActionAverage
    count: int
    reports: list[str]


def build_summary(reports_dir: Path) -> list[BalanceSearchSummary]:
    grouped: dict[tuple[object, object], dict[str, object]] = {}
    for report_path in list_training_reports(reports_dir=reports_dir):
        payload = load_training_report(report_path)
        if payload is None:
            continue

        metadata = ensure_dict(payload.get("metadata"))
        ab_tag = metadata.get("ab_tag")
        if not isinstance(ab_tag, str) or not ab_tag.startswith("ab_balance_search_"):
            continue

        configuration = ensure_dict(payload.get("configuration"))
        environment = ensure_dict(configuration.get("environment"))
        reward_settings = ensure_dict(environment.get("reward_settings"))
        skew = reward_settings.get("skewness_penalty_value")
        balance = reward_settings.get("balance_shaping_value")

        action_distribution = extract_action_distribution_from_payload(payload)
        key = (skew, balance)
        if key not in grouped:
            grouped[key] = {"dists": [], "reports": []}

        dists = grouped[key]["dists"]
        reports = grouped[key]["reports"]
        if isinstance(dists, list):
            dists.append(action_distribution)
        if isinstance(reports, list):
            reports.append(report_path.name)

    summary: list[BalanceSearchSummary] = []
    for (skew, balance), value in grouped.items():
        distributions = value.get("dists")
        reports = value.get("reports")
        if not isinstance(distributions, list) or not isinstance(reports, list):
            continue

        averages: ActionAverage = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
        for distribution in distributions:
            dist_map = ensure_dict(distribution)
            for action in averages:
                averages[action] += safe_to_float(dist_map.get(action), 0.0)

        count = len(distributions)
        if count > 0:
            for action in averages:
                averages[action] /= count

        summary.append(
            {
                "skew": skew,
                "balance": balance,
                "avg": averages,
                "count": count,
                "reports": [str(name) for name in reports if isinstance(name, str)],
            }
        )

    summary.sort(key=lambda item: abs(item["avg"]["BUY"] - item["avg"]["SELL"]))
    return summary


def main() -> None:
    summary = build_summary(Path("reports"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
