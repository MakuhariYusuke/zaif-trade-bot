#!/usr/bin/env python3
"""
Cost estimator for training runs.

Estimates GPU, power, and cloud costs based on run metadata and rates.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

sys.path.insert(0, str(Path(__file__).parent.parent / "ztb"))
# Add the ztb package to the path

from ztb.utils.cli_common import (
    CLIFormatter,
    CLIValidator,
    CommonArgs,
    create_standard_parser,
    get_env_default,
)


def calculate_jp_residential_tiered(kwh: float) -> float:
    """Calculate electricity cost using Japanese residential tiered pricing (40A)."""
    # Basic fee for 40A contract
    basic_fee = 1246.96  # JPY/month

    # Tiered rates (JPY/kWh)
    if kwh <= 120:
        power_cost = kwh * 29.70
    elif kwh <= 300:
        power_cost = 120 * 29.70 + (kwh - 120) * 35.69
    else:
        power_cost = 120 * 29.70 + 180 * 35.69 + (kwh - 300) * 39.50

    return power_cost + basic_fee


def load_metadata(
    correlation_id: str, artifacts_dir: Path = Path("artifacts")
) -> Optional[Dict[str, Any]]:
    """Load run metadata."""
    metadata_path = artifacts_dir / correlation_id / "run_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    except (OSError, json.JSONDecodeError):
        return None


def load_summary(
    correlation_id: str, artifacts_dir: Path = Path("artifacts")
) -> Optional[Dict[str, Any]]:
    """Load summary for steps/sec calculation."""
    summary_path = artifacts_dir / correlation_id / "summary.json"
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    except (OSError, json.JSONDecodeError):
        return None


def estimate_cost(
    metadata: Dict[str, Any],
    summary: Dict[str, Any],
    gpu_rate: float,
    kwh_rate: float,
    cloud_rate: float = 0.0,
    tariff: str = "jp_residential_tiered",
    manual_kwh: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate costs."""
    # Extract duration (assume in seconds)
    duration_hours = metadata.get("duration_seconds", 0) / 3600

    # Extract GPU info
    gpu_count = metadata.get("gpu_count", 1)
    gpu_hours = duration_hours * gpu_count

    # Power consumption estimate (rough: 300W per GPU)
    power_kw = 0.3 * gpu_count
    power_kwh = power_kw * duration_hours

    # Override with manual kWh if provided
    if manual_kwh is not None:
        power_kwh = manual_kwh

    # Steps/sec from summary
    global_step = summary.get("summary", {}).get("global_step", 0)
    steps_per_sec = (
        global_step / metadata.get("duration_seconds", 1)
        if metadata.get("duration_seconds", 0) > 0
        else 0
    )

    # Calculate power cost based on tariff
    if tariff == "jp_residential_tiered":
        power_cost = calculate_jp_residential_tiered(power_kwh)
    else:
        # Simple flat rate
        power_cost = power_kwh * kwh_rate

    # Costs
    gpu_cost = gpu_hours * gpu_rate
    cloud_cost = duration_hours * cloud_rate
    total_cost = gpu_cost + power_cost + cloud_cost

    return {
        "correlation_id": metadata.get("correlation_id", "unknown"),
        "duration_hours": duration_hours,
        "gpu_count": gpu_count,
        "gpu_hours": gpu_hours,
        "power_kwh": power_kwh,
        "steps_per_sec": steps_per_sec,
        "costs": {
            "gpu_jpy": gpu_cost,
            "power_jpy": power_cost,
            "cloud_jpy": cloud_cost,
            "total_jpy": total_cost,
        },
        "rates": {
            "gpu_rate_jpy_per_hour": gpu_rate,
            "kwh_rate_jpy": kwh_rate,
            "cloud_rate_jpy_per_hour": cloud_rate,
            "tariff": tariff,
        },
    }


def save_estimate(estimate: Dict[str, Any], output_path: Path) -> None:
    """Save estimate as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(estimate, f, indent=2, ensure_ascii=False)


def generate_markdown(estimate: Dict[str, Any]) -> str:
    """Generate Markdown summary."""
    md = [f"# Cost Estimate: {estimate['correlation_id']}\n"]
    md.append(f"- Duration: {estimate['duration_hours']:.2f} hours")
    md.append(f"- GPU Count: {estimate['gpu_count']}")
    md.append(f"- GPU Hours: {estimate['gpu_hours']:.2f}")
    md.append(f"- Power Consumption: {estimate['power_kwh']:.2f} kWh")
    md.append(f"- Steps/sec: {estimate['steps_per_sec']:.2f}")
    md.append("")
    md.append("## Costs (JPY)")
    md.append(f"- GPU: ¥{estimate['costs']['gpu_jpy']:.0f}")
    md.append(f"- Power: ¥{estimate['costs']['power_jpy']:.0f}")
    md.append(f"- Cloud: ¥{estimate['costs']['cloud_jpy']:.0f}")
    md.append(f"- **Total: ¥{estimate['costs']['total_jpy']:.0f}**")
    md.append("")
    md.append("## Rates")
    md.append(f"- GPU: ¥{estimate['rates']['gpu_rate_jpy_per_hour']:.0f}/hour")
    md.append(f"- Power: ¥{estimate['rates']['kwh_rate_jpy']:.0f}/kWh")
    md.append(f"- Cloud: ¥{estimate['rates']['cloud_rate_jpy_per_hour']:.0f}/hour")

    return "\n".join(md)


def main() -> int:
    parser = create_standard_parser("Estimate training costs")
    CommonArgs.add_correlation_id(parser)
    parser.add_argument(
        "--gpu-rate",
        type=lambda x: CLIValidator.validate_positive_float(x, "gpu-rate"),
        default=get_env_default("GPU_RATE_JPY_PER_HOUR", 300),
        help=CLIFormatter.format_help("GPU rate in JPY per hour", 300),
    )
    parser.add_argument(
        "--kwh-rate",
        type=lambda x: CLIValidator.validate_positive_float(x, "kwh-rate"),
        default=get_env_default("KWH_RATE_JPY", 35),
        help=CLIFormatter.format_help(
            "Electricity rate in JPY per kWh (ignored for tiered tariffs)", 35
        ),
    )
    parser.add_argument(
        "--cloud-rate",
        type=lambda x: CLIValidator.validate_positive_float(x, "cloud-rate"),
        default=get_env_default("CLOUD_RATE_JPY_PER_HOUR", 0),
        help=CLIFormatter.format_help("Cloud infrastructure rate in JPY per hour", 0),
    )
    parser.add_argument(
        "--tariff",
        choices=["simple", "jp_residential_tiered"],
        default="jp_residential_tiered",
        help=CLIFormatter.format_help(
            "Electricity tariff model", "jp_residential_tiered"
        ),
    )
    parser.add_argument(
        "--kwh",
        type=lambda x: CLIValidator.validate_positive_float(x, "kwh"),
        help="Override estimated kWh consumption",
    )
    CommonArgs.add_artifacts_dir(parser)
    parser.add_argument("--out", help="Output file (JSON if .json, Markdown if .md)")

    args = parser.parse_args()

    metadata = load_metadata(args.correlation_id, Path(args.artifacts_dir))
    if not metadata:
        print(f"Metadata not found for {args.correlation_id}", file=sys.stderr)
        return 1

    summary = load_summary(args.correlation_id, Path(args.artifacts_dir))
    if not summary:
        print(f"Summary not found for {args.correlation_id}", file=sys.stderr)
        return 1

    estimate = estimate_cost(
        metadata,
        summary,
        args.gpu_rate,
        args.kwh_rate,
        args.cloud_rate,
        args.tariff,
        args.kwh,
    )

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".json":
            save_estimate(estimate, out_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(generate_markdown(estimate), encoding="utf-8")
        print(f"Cost estimate saved to {out_path}")
    else:
        # Print JSON to stdout
        print(json.dumps(estimate, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
