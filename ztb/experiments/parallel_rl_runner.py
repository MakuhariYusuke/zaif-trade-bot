#!/usr/bin/env python3
"""
Parallel execution script for reinforcement learning experiments
Runs generalization and aggressive strategies in parallel
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local module imports
# current_dir: ztb/experiments の親ディレクトリ (ztb)
current_dir = Path(__file__).parent.parent
# project_root: プロジェクトのルートディレクトリ (zaif-trade-bot)
project_root = current_dir.parent  # Go up one more level to project root
sys.path.insert(0, str(project_root))
from ztb.utils import LoggerManager
from ztb.utils.parallel_experiments import ResourceMonitor


class ParallelRLExperimentRunner:
    """Parallel runner for reinforcement learning experiments"""

    def __init__(self):
        self.logger_manager = LoggerManager(experiment_id="parallel_rl_experiments")
        self.resource_monitor = ResourceMonitor()
        self.log_dir = Path("results/parallel_runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Experiment configurations
        self.experiments = [
            {
                'name': 'run100k_generalization',
                'strategy': 'generalization',
                'steps': 100000,
                'timeout_seconds': 21600  # 6 hours
            },
            {
                'name': 'run100k_aggressive',
                'strategy': 'aggressive',
                'steps': 100000,
                'timeout_seconds': 21600  # 6 hours
            }
        ]

        self.processes = {}
        self.results = {}
        self.start_time = time.time()

    def run_parallel_experiments(self):
        """Run experiments in parallel"""
        self.start_time = time.time()
        total_timeout = 8 * 3600  # 8 hours total timeout

        self.logger_manager.log_experiment_start("parallel_rl_experiments", {
            'experiments': self.experiments,
            'parallel_execution': True,
            'total_timeout_seconds': total_timeout
        })

        # Start resource monitoring
        self.resource_monitor.log_resources("parallel_rl_start")

        try:
            # Start experiments in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for exp_config in self.experiments:
                    future = executor.submit(self._run_single_experiment, exp_config)
                    futures[future] = exp_config['name']

                # Monitor progress with timeout
                completed_experiments = 0
                start_parallel_time = time.time()

                try:
                    for future in as_completed(futures.keys(), timeout=300):  # 5 minute timeout
                        exp_name = futures[future]

                        # Check overall timeout
                        elapsed = time.time() - start_parallel_time
                        if elapsed > total_timeout:
                            self.logger_manager.enqueue_notification(
                                f"Overall timeout reached after {elapsed:.1f}s. Terminating remaining experiments."
                            )
                            self._terminate_all_processes()
                            break

                        try:
                            result = future.result(timeout=10)  # 10 second timeout for result
                            self.results[exp_name] = result
                            completed_experiments += 1
                            self.logger_manager.log_experiment_end({
                                'experiment_name': exp_name,
                                'status': 'completed',
                                'result': result
                            })
                        except Exception as e:
                            # 致命的なエラーの場合は強制終了
                            if isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError, OSError)):
                                self.logger_manager.enqueue_notification(
                                    f"Fatal error in experiment {exp_name}: {type(e).__name__}. Terminating all processes."
                                )
                                self._terminate_all_processes()
                                sys.exit(1)

                            self.logger_manager.log_error(str(e), f"Experiment: {exp_name}")
                            self.results[exp_name] = {'status': 'failed', 'error': str(e)}
                            completed_experiments += 1

                except TimeoutError:
                    # Experiments are taking too long, terminate them
                    elapsed = time.time() - start_parallel_time
                    self.logger_manager.enqueue_notification(
                        f"Experiments timed out after 5 minutes of no progress (elapsed: {elapsed:.1f}s). Terminating all processes."
                    )
                    self._terminate_all_processes()
                    # Mark remaining experiments as timed out
                    for exp_name in futures.values():
                        if exp_name not in self.results:
                            self.results[exp_name] = {'status': 'timeout', 'error': f'Timeout after {elapsed:.1f} seconds'}

            # Generate summary report
            self._generate_summary_report()

        finally:
            # Clean up any remaining processes
            self._terminate_all_processes()

            # Stop resource monitoring
            self.resource_monitor.log_resources("parallel_rl_end")

            total_time = time.time() - self.start_time
            self.logger_manager.log_experiment_end({
                'parallel_execution': True,
                'total_time_seconds': total_time,
                'experiments_completed': len([r for r in self.results.values() if r.get('status') not in ['failed', 'timeout']]),
                'experiments_failed': len([r for r in self.results.values() if r.get('status') in ['failed', 'timeout']]),
                'timeout_reached': total_time > total_timeout
            })

    def _run_single_experiment(self, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment"""
        import sys  # Add this to bind sys in local scope
        exp_name = exp_config['name']
        strategy = exp_config['strategy']
        steps = exp_config['steps']
        timeout_seconds = exp_config.get('timeout_seconds', 21600)  # Default 6 hours

        self.logger_manager.enqueue_notification(f"Starting experiment: {exp_name} (strategy: {strategy}, timeout: {timeout_seconds}s)")

        # Prepare command
        cmd = [
            sys.executable,
            "ztb/experiments/ml_reinforcement_1k.py",
            "--strategy", strategy,
            "--steps", str(steps),
            "--name", exp_name
        ]

        # Log file for this experiment
        log_file = self.log_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # Log file for this experiment
        log_file = self.log_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        process = None  # Ensure process is defined for exception handling

        try:
            # Start process
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )

            self.processes[exp_name] = process
            time.sleep(2)
            if process.poll() is not None:
                # Process already finished (likely failed to start)
                return_code = process.returncode
                error_msg = f"Process failed to start or exited immediately with code {return_code}"
                self.logger_manager.log_error(error_msg, exp_name)
                return {
                    'status': 'failed',
                    'error': error_msg,
                    'return_code': return_code,
                    'log_file': str(log_file)
                }

            # Wait for completion with progress monitoring and timeout
            start_time = time.time()
            last_progress_time = start_time

            while process.poll() is None:
                current_time = time.time()
                elapsed = current_time - start_time

                # Check timeout
                if elapsed > timeout_seconds:
                    self.logger_manager.enqueue_notification(f"Experiment {exp_name} timed out after {elapsed:.1f}s. Terminating process.")
                    process.terminate()
                    time.sleep(5)  # Give it 5 seconds to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    return {
                        'status': 'timeout',
                        'error': f'Timeout after {elapsed:.1f} seconds',
                        'execution_time_seconds': elapsed,
                        'log_file': str(log_file)
                    }

                time.sleep(10)  # Check every 10 seconds

                # Log progress every 5 minutes
                if current_time - last_progress_time > 300:
                    self.logger_manager.enqueue_notification(f"{exp_name}: Still running (elapsed: {elapsed:.1f}s)")
                    last_progress_time = current_time

                    # Log progress (read last few lines of log file)
                    if log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = lines[-1].strip()
                                    if 'Step' in last_line and '/' in last_line:
                                        self.logger_manager.enqueue_notification(f"{exp_name}: {last_line}")
                        except:
                            pass

            # Get return code
            return_code = process.returncode

            # Read final results from log file
            final_metrics = self._extract_metrics_from_log(log_file)

            execution_time = time.time() - start_time

            result = {
                'status': 'success' if return_code == 0 else 'failed',
                'return_code': return_code,
                'log_file': str(log_file),
                'execution_time_seconds': execution_time,
                'metrics': final_metrics
            }

            self.logger_manager.enqueue_notification(f"Experiment {exp_name} completed with status: {result['status']}")
            return result

        except Exception as e:
            # 致命的なエラーの場合は強制終了
            if isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError, OSError)):
                self.logger_manager.enqueue_notification(
                    f"Fatal error in experiment {exp_name}: {type(e).__name__}. Terminating process."
                )
                import sys
                sys.exit(1)

            error_msg = f"Failed to run experiment {exp_name}: {str(e)}"
            self.logger_manager.log_error(error_msg, exp_name)
            return {
                'status': 'failed',
                'error': error_msg,
                'log_file': str(log_file)
            }

    def _extract_metrics_from_log(self, log_file: Path) -> Dict[str, Any]:
        """Extract metrics from experiment log file"""
        metrics = {}

        try:
            with open(log_file, 'r') as f:
                content = f.read()

                # Look for metrics in the log
                lines = content.split('\n')
                for line in lines:
                    if 'avg_reward:' in line:
                        # Extract reward info
                        parts = line.split('avg_reward:')
                        if len(parts) > 1:
                            try:
                                metrics['avg_reward'] = float(parts[1].split(',')[0].strip())
                            except:
                                pass
                    elif 'sharpe_ratio' in line and 'metrics:' in line:
                        # sharpe_ratio extraction not implemented
                        # TODO: Implement sharpe_ratio extraction if log format is known
                        metrics['sharpe_ratio'] = None  # 未実装

        except Exception as e:
            self.logger_manager.log_error(f"Failed to extract metrics from {log_file}: {e}")

        return metrics

    def _generate_summary_report(self):
        """Generate summary report of all experiments"""
        summary_file = self.log_dir / f"parallel_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            'execution_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_time_seconds': time.time() - self.start_time,
                'experiments_run': len(self.experiments),
                'experiments_completed': len([r for r in self.results.values() if r.get('status') == 'success']),
                'experiments_failed': len([r for r in self.results.values() if r.get('status') == 'failed'])
            },
            'resource_usage': {
                'peak_cpu_percent': self.resource_monitor.peak_cpu_percent,
                'peak_memory_mb': self.resource_monitor.peak_memory_mb,
                'avg_cpu_percent': self.resource_monitor.avg_cpu_percent
            },
            'experiment_results': self.results,
            'performance_comparison': self._compare_strategies()
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger_manager.enqueue_notification(f"Parallel execution summary saved to: {summary_file}")

        # Print summary to console
        print("\n" + "="*80)
        print("PARALLEL RL EXPERIMENTS SUMMARY")
        print("="*80)
        print(f"Total execution time: {summary['execution_summary']['total_time_seconds']:.1f} seconds")
        print(f"Experiments completed: {summary['execution_summary']['experiments_completed']}")
        print(f"Experiments failed: {summary['execution_summary']['experiments_failed']}")
        print("\nResource Usage:")
        resource_usage = summary['resource_usage']
        print(f"  Peak CPU: {resource_usage.get('peak_cpu_percent', 0):.1f}%")
        print(f"  Peak Memory: {resource_usage.get('peak_memory_mb', 0):.1f} MB")
        print(f"  Average CPU: {resource_usage.get('avg_cpu_percent', 0):.1f}%")

        print("\nStrategy Comparison:")
        comparison = summary['performance_comparison']
        for strategy, metrics in comparison.items():
            print(f"  {strategy}:")
            print(f"    Status: {metrics.get('status', 'unknown')}")
            print(f"    Avg Reward: {metrics.get('avg_reward', 0):.4f}")
            print(f"    Execution Time: {metrics.get('execution_time', 0):.1f}s")

    def _compare_strategies(self) -> Dict[str, Any]:
        """Compare performance between strategies"""
        comparison = {}

        for exp_name, result in self.results.items():
            if 'generalization' in exp_name:
                strategy = 'generalization'
            elif 'aggressive' in exp_name:
                strategy = 'aggressive'
            else:
                continue

            comparison[strategy] = {
                'status': result.get('status'),
                'execution_time': result.get('execution_time_seconds', 0),
                'avg_reward': result.get('metrics', {}).get('avg_reward', 0),
                'log_file': result.get('log_file')
            }

        return comparison

    def _terminate_all_processes(self):
        """Terminate all running experiment processes"""
        for exp_name, process in self.processes.items():
            try:
                if process.poll() is None:
                    self.logger_manager.enqueue_notification(f"Terminating process for {exp_name}")
                    process.terminate()
                    time.sleep(2)
            except Exception as e:
                # プロセス終了時のエラーは、既にプロセスが終了している場合や、権限の問題などで発生することがあります。
                self.logger_manager.log_error(f"Error terminating process {exp_name}: {e}", exp_name)


def main():
    """Main entry point"""
    print("Starting parallel reinforcement learning experiments...")
    print("Experiments: generalization (100k steps), aggressive (100k steps)")

    runner = ParallelRLExperimentRunner()
    runner.run_parallel_experiments()

    print("\nParallel execution completed!")


if __name__ == "__main__":
    main()