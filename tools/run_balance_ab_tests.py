#!/usr/bin/env python3
"""
Balance-Focused AB Test Runner

Runs AB tests with multiple balance_shaping_value configurations
to find the optimal balance between BUY, SELL, and HOLD actions.

Target: BUY ~60%, SELL ~33%, HOLD ~7% (based on previous success)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def create_balance_configs(
    base_config_path: str,
    balance_values: List[float],
    penalty_values: List[float],
    output_dir: str = "config/v447/balance_test",
) -> List[str]:
    """
    Create multiple config files with different balance_shaping_value and balance_penalty.

    Args:
        base_config_path: Path to base configuration
        balance_values: List of balance_shaping_value to test
        penalty_values: List of balance_penalty to test
        output_dir: Directory to save generated configs

    Returns:
        List of generated config file paths
    """
    base_config = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_configs = []

    for balance_val in balance_values:
        for penalty_val in penalty_values:
            # Update config
            config = base_config.copy()

            # Update curriculum config
            if "curriculum" not in config:
                config["curriculum"] = {}

            config["curriculum"]["type"] = "forced_balance"
            config["curriculum"]["balance_shaping_value"] = balance_val
            config["curriculum"]["balance_penalty"] = penalty_val

            # Update model name to reflect parameters
            model_name = f"sac_v447_balance_{int(balance_val*100):02d}_penalty_{int(penalty_val)}"
            config["training"]["model_name"] = model_name

            # Save config
            config_filename = f"{model_name}.json"
            config_path = output_path / config_filename
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            generated_configs.append(str(config_path))
            print(f"✅ Created: {config_path}")

    return generated_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balance-focused AB tests")
    from ztb.utils.cli import add_common_cli_args

    add_common_cli_args(parser)
    parser.add_argument(
        "--base-config",
        default="config/v447/sac_v447_1m_multiframe_config.json",
        help="Base configuration file",
    )
    parser.add_argument(
        "--balance-values",
        nargs="+",
        type=float,
        default=[0.03, 0.04, 0.05, 0.06, 0.07],
        help="Balance shaping values to test",
    )
    parser.add_argument(
        "--penalty-values",
        nargs="+",
        type=float,
        default=[3.0, 4.0, 5.0],
        help="Balance penalty values to test",
    )
    parser.add_argument(
        "--output-dir",
        default="config/v447/balance_test",
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--timesteps", type=int, default=2000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--seeds", type=int, default=3, help="Number of random seeds to test"
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--run", action="store_true", help="Run AB tests after generating configs"
    )

    args = parser.parse_args()
    from ztb.utils.cli import configure_logging_from_args

    configure_logging_from_args(args)

    print("=" * 80)
    print("Balance-Focused AB Test Configuration Generator")
    print("=" * 80)
    print(f"Base config: {args.base_config}")
    print(f"Balance values: {args.balance_values}")
    print(f"Penalty values: {args.penalty_values}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Generate configs
    config_paths = create_balance_configs(
        args.base_config, args.balance_values, args.penalty_values, args.output_dir
    )

    print(f"\n✅ Generated {len(config_paths)} configuration files")

    if args.run:
        print("\n" + "=" * 80)
        print("Running AB Tests")
        print("=" * 80)

        # Import and run AB test
        from subprocess import run

        cmd = [
            sys.executable,
            "tools/ab_test_runner.py",
            "--configs",
            *config_paths,
            "--seeds",
            str(args.seeds),
            "--jobs",
            str(args.jobs),
            "--timesteps",
            str(args.timesteps),
        ]

        print(f"Command: {' '.join(cmd)}")
        print()

        result = run(cmd)

        if result.returncode == 0:
            print("\n✅ AB tests completed successfully")
        else:
            print(f"\n❌ AB tests failed with exit code {result.returncode}")
            return int(result.returncode)
    else:
        print("\nTo run AB tests with these configs:")
        print("  python tools/ab_test_runner.py \\")
        print(f"    --configs {' '.join(config_paths)} \\")
        print(
            f"    --seeds {args.seeds} --jobs {args.jobs} --timesteps {args.timesteps}"
        )


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
