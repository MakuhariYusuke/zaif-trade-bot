#!/usr/bin/env python3
"""
Artifact rollup and executive excerpt generator for Zaif Trade Bot.

Aggregates latest metrics.json, eval_*    su    summary = {
    summary = {
        'correlation_id': correlation_id,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata,
        'metrics': metrics,
        'evaluations': evals,
        'summary': summary_dict,
    }on_id': correlation_id,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata,
        'metrics': metrics,
        'evaluations': evals,
        'summary': summary_dict,
    }test': 1}mmary': summary_dict,     correlation_id=correlation_id,
        timestamp=datetime.now().isoformat(),
        metadata=metadata,
        metrics=metrics,
        evaluations=evals,
        summary=summary_dict,
    )n every N minutes or one-shot.
Produces artifacts/{corr}/summary.json and summary.md (short executive excerpt).
Validates with schema/results_schema.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

jsonschema: Optional[Any] = None

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None
    HAS_JSONSCHEMA = False

# Add ztb to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ztb.training.eval_gates import EvalGates, GateResult, GateStatus


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return cast(Dict[str, Any], json.load(f))


def load_cost_estimate(correlation_id: str) -> Optional[Dict[str, Any]]:
    """Load cost estimate if available."""
    artifacts_dir = Path("artifacts") / correlation_id
    cost_file = artifacts_dir / "reports" / "cost_estimate.json"
    if cost_file.exists():
        return load_json_file(cost_file)
    return None


def aggregate_metrics(correlation_id: str) -> Dict[str, Any]:
    """Aggregate latest metrics from artifacts."""
    artifacts_dir = Path("artifacts") / correlation_id

    # Find latest metrics.json
    metrics_files = list(artifacts_dir.glob("**/metrics.json"))
    metrics = {}
    latest_metrics_path = None
    if metrics_files:
        latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
        metrics = load_json_file(latest_metrics)
        latest_metrics_path = latest_metrics

    # Find latest eval_* files
    eval_files = list(artifacts_dir.glob("**/eval_*.json"))
    evals = {}
    for eval_file in eval_files:
        key = eval_file.stem  # eval_something
        evals[key] = load_json_file(eval_file)

    # Find gates.json
    gates_files = list(artifacts_dir.glob("**/gates.json"))
    gates_data = {}
    if gates_files:
        latest_gates = max(gates_files, key=lambda p: p.stat().st_mtime)
        gates_data = load_json_file(latest_gates)

    # Find run_metadata.json
    metadata_files = list(artifacts_dir.glob("**/run_metadata.json"))
    metadata = {}
    if metadata_files:
        latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
        metadata = load_json_file(latest_metadata)
    else:
        # Dummy metadata for schema
        metadata = {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "run_id": correlation_id,
            "type": "training",
        }

    # Extract key metrics
    global_step = metrics.get("global_step", 0)
    wall_clock_duration = metrics.get("wall_clock_duration", 0)
    rss_peaks = metrics.get("rss_peaks", [])
    vram_peaks = metrics.get("vram_peaks", [])

    # Best of evaluations
    best_sharpe = None
    best_dsr = None
    best_p_value = None
    acceptance_gate = {}

    for eval_name, eval_data in evals.items():
        if "sharpe" in eval_data:
            if best_sharpe is None or eval_data["sharpe"] > best_sharpe:
                best_sharpe = eval_data["sharpe"]
        if "dsr" in eval_data:
            if best_dsr is None or eval_data["dsr"] > best_dsr:
                best_dsr = eval_data["dsr"]
        if "bootstrap_p_value" in eval_data:
            if best_p_value is None or eval_data["bootstrap_p_value"] < best_p_value:
                best_p_value = eval_data["bootstrap_p_value"]
        if "acceptance" in eval_data:
            acceptance_gate = eval_data["acceptance"]

    # Calculate acceptance from gates if available
    if gates_data:
        # Get latest gate results
        latest_gates_data = gates_data
        gate_results = {}
        for name, result_dict in latest_gates_data.get("gate_results", {}).items():
            gate_results[name] = GateResult(
                name=name,
                status=GateStatus(result_dict["status"]),
                reason=result_dict["reason"],
                value=result_dict.get("value"),
                threshold=result_dict.get("threshold"),
            )
        eval_gates = EvalGates(enabled=True)
        acceptance_gate = eval_gates.get_acceptance_summary(gate_results)

    # Aggregate
    metrics_json = str(latest_metrics_path) if latest_metrics_path else None
    eval_files_str = [str(f) for f in eval_files]
    gates_json = str(latest_gates) if gates_files else None
    metadata_json = str(latest_metadata) if metadata_files else None
    artifact_paths = {
        "metrics_json": metrics_json,
        "eval_files": eval_files_str,
        "gates_json": gates_json,
        "metadata_json": metadata_json,
    }
    summary_dict = dict(
        global_step=global_step,
        wall_clock_duration_hours=(
            wall_clock_duration / 3600 if wall_clock_duration else 0
        ),
        rss_peak_mb=max(rss_peaks) if rss_peaks else 0,
        vram_peak_mb=max(vram_peaks) if vram_peaks else 0,
        best_sharpe=best_sharpe,
        best_dsr=best_dsr,
        best_bootstrap_p_value=best_p_value,
        acceptance_gate=acceptance_gate,
        artifact_paths=artifact_paths,
    )
    summary = {
        "correlation_id": correlation_id,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "metrics": metrics,
        "evaluations": evals,
        "summary": summary_dict,
        "global_step": global_step,
        "wall_clock_duration_hours": (
            wall_clock_duration / 3600 if wall_clock_duration else 0
        ),
        "rss_peak_mb": max(rss_peaks) if rss_peaks else 0,
        "vram_peak_mb": max(vram_peaks) if vram_peaks else 0,
        "best_sharpe": best_sharpe,
        "best_dsr": best_dsr,
        "best_bootstrap_p_value": best_p_value,
        "acceptance_gate": acceptance_gate,
        "artifact_paths": artifact_paths,
    }
    # Load cost estimate if available
    cost_estimate = load_cost_estimate(correlation_id)
    if cost_estimate:
        summary["cost_estimate"] = cost_estimate

    return summary


def generate_executive_summary(summary: Dict[str, Any]) -> str:
    """Generate short executive excerpt in Markdown."""
    md = f"# Training Summary for {summary.get('correlation_id', 'Unknown')}\n\n"
    md += f"Generated: {summary['timestamp']}\n\n"

    summ = summary.get("summary", {})

    md += "## Key Metrics\n\n"
    md += f"- Global Step: {summ.get('global_step', 'N/A')}\n"
    md += (
        f"- Wall-Clock Duration: {summ.get('wall_clock_duration_hours', 0):.1f} hours\n"
    )
    md += f"- RSS Peak: {summ.get('rss_peak_mb', 0):.1f} MB\n"
    md += f"- VRAM Peak: {summ.get('vram_peak_mb', 0):.1f} MB\n\n"

    md += "## Performance\n\n"
    md += f"- Best Sharpe Ratio: {summ.get('best_sharpe', 'N/A')}\n"
    md += f"- Best DSR: {summ.get('best_dsr', 'N/A')}\n"
    md += f"- Best Bootstrap P-Value: {summ.get('best_bootstrap_p_value', 'N/A')}\n\n"

    acceptance = summ.get("acceptance_gate", {})
    if acceptance:
        md += "## Acceptance Gate\n\n"
        for key, value in acceptance.items():
            md += f"- {key}: {value}\n"
        md += "\n"

    # Add cost estimate if available
    cost_est = summary.get("cost_estimate", {})
    if cost_est:
        costs = cost_est.get("costs", {})
        md += "## Cost Estimate\n\n"
        md += f"- Duration: {cost_est.get('duration_hours', 0):.1f} hours\n"
        md += f"- Power: {cost_est.get('power_kwh', 0):.1f} kWh\n"
        md += f"- GPU Cost: ¥{costs.get('gpu_jpy', 0):.0f}\n"
        md += f"- Power Cost: ¥{costs.get('power_jpy', 0):.0f}\n"
        md += f"- Total Cost: ¥{costs.get('total_jpy', 0):.0f}\n\n"

    paths = summ.get("artifact_paths", {})
    if paths:
        md += "## Artifact Paths\n\n"
        if paths.get("metrics_json"):
            md += f"- Metrics: {paths['metrics_json']}\n"
        if paths.get("eval_files"):
            md += f"- Evaluations: {', '.join(paths['eval_files'])}\n"
        if paths.get("metadata_json"):
            md += f"- Metadata: {paths['metadata_json']}\n"

    return md


def validate_summary(summary: Dict[str, Any]) -> bool:
    """Validate summary against schema."""
    if not HAS_JSONSCHEMA:
        print("jsonschema not available, skipping validation", file=sys.stderr)
        return True

    schema_path = Path("schema/results_schema.json")
    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}", file=sys.stderr)
        return False

    with open(schema_path, "r") as f:
        schema = json.load(f)

    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=summary, schema=schema)  # type: ignore[union-attr]
            return True
        except jsonschema.ValidationError as e:  # type: ignore[union-attr]
            print(f"Schema validation error: {e}", file=sys.stderr)
            return False
    else:
        print("jsonschema not available, skipping validation", file=sys.stderr)  # type: ignore[unreachable]
        return True


def redact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secrets from summary."""
    # Placeholder: remove sensitive fields
    redacted = cast(Dict[str, Any], json.loads(json.dumps(summary)))  # Deep copy
    # Remove any fields that might contain secrets
    if "metadata" in redacted and "config" in redacted["metadata"]:
        config = redacted["metadata"]["config"]
        # Remove API keys, passwords, etc.
        for key in list(config.keys()):
            if (
                "key" in key.lower()
                or "secret" in key.lower()
                or "password" in key.lower()
            ):
                del config[key]
    return redacted


def process_rollup(correlation_id: str) -> int:
    """Process rollup for given correlation_id."""
    summary = aggregate_metrics(correlation_id)

    if not summary:
        print(f"No artifacts found for {correlation_id}", file=sys.stderr)
        return 1

    # Redact
    summary = redact_summary(summary)

    # Validate
    if not validate_summary(summary):
        return 1

    # Write summary.json
    artifacts_dir = Path("artifacts") / correlation_id
    summary_json = artifacts_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Write summary.md
    summary_md = artifacts_dir / "summary.md"
    md_content = generate_executive_summary(summary)
    with open(summary_md, "w") as f:
        f.write(md_content)

    print(f"Rollup complete: {summary_json}, {summary_md}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollup artifacts and generate summary"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for this session"
    )
    parser.add_argument(
        "--interval-minutes", type=int, help="Run every N minutes (default: one-shot)"
    )

    args = parser.parse_args()

    if args.interval_minutes:
        import time

        while True:
            process_rollup(args.correlation_id)
            time.sleep(args.interval_minutes * 60)
    else:
        sys.exit(process_rollup(args.correlation_id))


if __name__ == "__main__":
    main()
