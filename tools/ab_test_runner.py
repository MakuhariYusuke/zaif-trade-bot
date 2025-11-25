#!/usr/bin/env python3
"""
Simple AB test runner for short training runs.

Usage (Windows cmd.exe):
    python tools\ab_test_runner.py --configs "config1.json" "config2.json" --seeds 3

Note: When passing multiple config files in Windows 'cmd.exe', wrap each path in double
quotes to avoid cmd splitting or character escaping issues. You can pass more than two
configs now; the script aggregates action_distribution across seeds for each config.

It runs each config for each seed (2k steps per config), then aggregates final
action_distribution from each generated training report and prints summary.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ztb.cache.memory_cache import default_memory_manager
from ztb.experiments.base import ExperimentResult
from ztb.utils.parallel_experiments import run_parallel_experiments

WRAPPER_PATH = Path(__file__).resolve().parent / "run_child_trainer_wrapper.py"


def check_torch_available() -> bool:
    """Check if torch is importable in this environment. Returns True if torch is available."""
    try:
        import importlib

        torch = importlib.import_module("torch")
        # If python successfully imports torch but the DLL initialization fails it'll raise OSError
        # We can still run CPU-only workloads if installed as CPU-only. Return True if import ok.
        return True
    except Exception as e:
        print("Warning: PyTorch import failed in this environment.")
        print(
            "PyTorch is required for training runs. If you only need CPU-mode, install a CPU-only PyTorch wheel, e.g.:"
        )
        print("  pip install --upgrade pip")
        print(
            "  pip install " "torch" " --index-url https://download.pytorch.org/whl/cpu"
        )
        print(
            "Otherwise, install the correct torch wheel for your platform (see https://pytorch.org/get-started/locally/)"
        )
        print(f"Import error: {e}")
        return False


def build_child_command(
    config_path: Path,
    seed: int,
    timesteps: int,
    fast_mode: bool,
    log_level: str = "WARNING",
) -> List[str]:
    """Construct the wrapper command for launching child trainers."""
    cmd = [
        sys.executable,
        "-u",
        str(WRAPPER_PATH),
        "--config",
        config_path.as_posix(),
        "--seed",
        str(seed),
        "--timesteps",
        str(timesteps),
        "--log-level",
        log_level,
    ]
    if fast_mode:
        cmd.append("--fast-mode")
    return cmd


def launch_child_training(cmd: List[str], seed: int, config_path: Path) -> None:
    """Launch the child process and persist its PID for debugging."""
    print("Running:", " ".join(cmd))
    p = subprocess.Popen(cmd)
    try:
        child_pid = p.pid
        print(f"Started training subprocess with PID {child_pid} for seed {seed}")
        # Log to file for easier verification
        try:
            log_path = Path("logs") / "ab_training_child_pids.jsonl"
            log_path.parent.mkdir(exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "child_pid": child_pid,
                            "config": config_path.as_posix(),
                            "seed": seed,
                            "time": str(datetime.now()),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

        p.wait()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
    except Exception:
        p.kill()
        raise


def run_training(
    config_path: Path, seed: int, timesteps: int = 2000, fast_mode: bool = False
) -> None:
    cmd = build_child_command(config_path, seed, timesteps, fast_mode)
    launch_child_training(cmd, seed, config_path)


class ABTrainingExperiment:
    """Small wrapper that runs the trainer for a single config+seed and returns an ExperimentResult."""

    def __init__(self, config: Dict[str, object]):
        self.config = config
        # Ensure compatibility with ParallelExperimentRunner (expecting experiment_name attr)
        self.experiment_name = f"ab_{Path(self.config.get('config_path', '')).stem}_seed_{self.config.get('seed', 0)}"

    def execute(self) -> ExperimentResult:
        # Clear memory cache before training to prevent leak
        import gc

        try:
            default_memory_manager.optimize_memory_usage()
            gc.collect()
        except Exception:
            pass  # Ignore cleanup errors

        cfg_path = Path(self.config.get("config_path"))
        seed = int(self.config.get("seed", 0))
        timesteps = int(self.config.get("timesteps", 2000))
        fast_mode = bool(self.config.get("fast_mode", False))
        cmd = build_child_command(cfg_path, seed, timesteps, fast_mode)
        launch_child_training(cmd, seed, cfg_path)

        # attempt to collect reports and return simple ExperimentResult
        model_name = json.loads(cfg_path.read_text(encoding="utf-8"))["training"][
            "model_name"
        ]
        reports = find_reports_for_model(model_name)
        metrics: Dict[str, object] = {}
        if reports:
            rp = reports[-1]
            metrics = {"action_distribution": extract_action_distribution(rp)}

        return ExperimentResult(
            experiment_name=f"ab_{model_name}_seed_{seed}",
            timestamp="",
            status="completed",
            config=self.config,  # type: ignore
            metrics=metrics,
            artifacts={},
        )


def find_reports_for_model(model_name: str) -> List[Path]:
    r = Path("reports")
    matches = []
    for p in r.glob("training_report_*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Drill down to training.model_name if present
        try:
            name = obj.get("configuration", {}).get("training", {}).get("model_name")
        except Exception:
            name = None
        if name == model_name:
            matches.append(p)
    return matches


def extract_action_distribution(report_path: Path) -> Dict[str, float]:
    obj = json.loads(report_path.read_text(encoding="utf-8"))
    return obj.get("training_stats", {}).get("action_distribution", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="+", required=True, help="One or more config paths"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
        help="Total timesteps for each run (overrides template)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast-mode defaults - small datalimit and minimal features",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers to use for runs (1 = sequential)",
    )
    args = parser.parse_args()

    configs = [Path(p) for p in args.configs]  # Changed to accept multiple configs
    for cfg in configs:
        if not cfg.exists():
            print("Config not found:", cfg)
            sys.exit(2)

    # Build run tasks for seeds
    tasks: List[Dict[str, object]] = []
    for cfg in configs:
        for seed in range(1, args.seeds + 1):
            tasks.append(
                {
                    "config_path": cfg.as_posix(),
                    "seed": seed,
                    "timesteps": args.timesteps,
                    "fast_mode": args.fast_mode,
                }
            )

    if args.jobs > 1 and tasks:
        # Use existing parallel experiments utility to run the tasks
        print(f"Running {len(tasks)} tasks with {args.jobs} parallel workers")
        run_parallel_experiments(ABTrainingExperiment, tasks, max_workers=args.jobs)
    else:
        # Check for torch availability and warn; training subprocess will still attempt to import
        if not check_torch_available():
            print(
                "Proceeding without PyTorch — CPU-only training will likely fail without torch installed."
            )
        for t in tasks:
            run_training(
                Path(t["config_path"]),
                int(t["seed"]),
                timesteps=int(t.get("timesteps", 2000)),
                fast_mode=bool(t.get("fast_mode", False)),
            )

    # Aggregate and print
    for cfg in configs:
        model_name = json.loads(cfg.read_text(encoding="utf-8"))["training"][
            "model_name"
        ]
        reports = find_reports_for_model(model_name)
        print("\nResults for", model_name)
        print("Found reports:", len(reports))
        all_dist = {"HOLD": [], "BUY": [], "SELL": []}
        for rp in reports:
            d = extract_action_distribution(rp)
            for k in all_dist.keys():
                all_dist[k].append(float(d.get(k, 0.0)))

        # compute mean
        import statistics

        if not all_dist["HOLD"]:
            print("No distributions found for", model_name)
            continue

        means = {k: statistics.mean(v) for k, v in all_dist.items()}
        stds = {
            k: statistics.pstdev(v) for k, v in all_dist.items()
        }  # population stdev as sample size small
        print("  Mean action distribution:")
        print(f"  HOLD: {means['HOLD']:.3f} ± {stds['HOLD']:.3f}")
        print(f"  BUY:  {means['BUY']:.3f} ± {stds['BUY']:.3f}")
        print(f"  SELL: {means['SELL']:.3f} ± {stds['SELL']:.3f}")

        # After run, if jobs used, print quick PID summary from logs for verification
        if args.jobs > 1:
            pid_log = Path("logs") / "parallel_worker_pids.jsonl"
            child_log = Path("logs") / "ab_training_child_pids.jsonl"
            if pid_log.exists():
                pids = set()
                with open(pid_log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            pids.add(obj.get("pid"))
                        except Exception:
                            continue
                print(
                    f"Parallel worker PIDs recorded: {len(pids)} -> {sorted(list(pids))}"
                )
            if child_log.exists():
                child_pids = set()
                with open(child_log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            child_pids.add(obj.get("child_pid"))
                        except Exception:
                            continue
                print(
                    f"Training subprocess PIDs recorded: {len(child_pids)} -> {sorted(list(child_pids))}"
                )


if __name__ == "__main__":
    main()
