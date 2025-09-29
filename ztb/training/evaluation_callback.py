"""Custom evaluation callback with DSR trials and bootstrap resampling."""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from .eval_gates import EvalGates

logger = logging.getLogger(__name__)


class DSREvaluationCallback(EvalCallback):
    """Evaluation callback with Deflated Sharpe Ratio and bootstrap resampling."""

    def __init__(
        self,
        eval_env,
        callback_on_new_best: Optional[Callable] = None,
        callback_after_eval: Optional[Callable] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        bootstrap_samples: int = 1000,
        dsr_trials: int = 50,
        gates_enabled: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.bootstrap_samples = bootstrap_samples
        self.dsr_trials = dsr_trials
        self.eval_returns_history: List[float] = []
        self.eval_gates = EvalGates(enabled=gates_enabled)
        self.gate_results_history: List[Dict] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self.evaluations_results) > 0:
                    rewards = self.evaluations_results[-1]
                    self.eval_returns_history.extend(rewards)

                    # Calculate metrics
                    mean_return = np.mean(rewards)
                    std_return = np.std(rewards)

                    # Bootstrap resampling for confidence intervals
                    bootstrap_means = self._bootstrap_resample(
                        rewards, self.bootstrap_samples
                    )
                    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

                    # Deflated Sharpe Ratio
                    dsr = self._calculate_dsr(rewards, self.dsr_trials)

                    # Run evaluation gates
                    gate_results = self.eval_gates.evaluate_all(
                        final_eval_reward=mean_return,
                        rewards=self.eval_returns_history,
                        steps=self.evaluations_timesteps
                        if hasattr(self, "evaluations_timesteps")
                        else [],
                        baseline=0.0,
                    )
                    self.gate_results_history.append(
                        {
                            "timestep": self.num_timesteps,
                            "gate_results": {
                                name: {
                                    "status": r.status.value,
                                    "reason": r.reason,
                                    "value": r.value,
                                }
                                for name, r in gate_results.items()
                            },
                        }
                    )

                    # Log metrics
                    logger.info(f"Evaluation at step {self.num_timesteps}:")
                    logger.info(f"  Mean return: {mean_return:.4f}")
                    logger.info(f"  Std return: {std_return:.4f}")
                    logger.info(f"  Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    logger.info(f"  DSR: {dsr:.4f}")
                    if gate_results:
                        logger.info("  Gates:")
                        for name, result in gate_results.items():
                            logger.info(
                                f"    {name}: {result.status.value.upper()} - {result.reason}"
                            )

                    # Store in kwargs for logging
                    kwargs.update(
                        {
                            "eval/mean_return": mean_return,
                            "eval/std_return": std_return,
                            "eval/bootstrap_ci_lower": ci_lower,
                            "eval/bootstrap_ci_upper": ci_upper,
                            "eval/dsr": dsr,
                        }
                    )

                # Save to disk
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

                # Save gate results to JSON
                if self.gate_results_history:
                    gates_path = Path(self.log_path).parent / "gates.json"
                    with open(gates_path, "w", encoding="utf-8") as f:
                        json.dump(
                            self.gate_results_history, f, indent=2, ensure_ascii=False
                        )

            # Check if best model
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print(
                            f"New best mean reward: {mean_reward:.2f} (previous: {self.best_mean_reward:.2f})"
                        )
                    if self.best_model_save_path is not None:
                        self.model.save(self.best_model_save_path)
                    self.best_mean_reward = mean_reward
                    # Trigger callback on new best
                    if self.callback_on_new_best is not None:
                        self.callback_on_new_best(self)

            # Trigger callback after eval
            if self.callback_after_eval is not None:
                self.callback_after_eval(self)

        return True

    def _bootstrap_resample(self, data: List[float], n_samples: int) -> np.ndarray:
        """Perform bootstrap resampling to estimate confidence intervals."""
        bootstrap_means = []
        n = len(data)
        for _ in range(n_samples):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        return np.array(bootstrap_means)

    def _calculate_dsr(self, rewards: List[float], n_trials: int) -> float:
        """Calculate Deflated Sharpe Ratio."""
        if len(rewards) < 2:
            return 0.0

        # Sharpe ratio of the strategy
        mean_return = np.mean(rewards)
        std_return = np.std(rewards)
        sharpe = mean_return / std_return if std_return > 0 else 0.0

        # Simulate random trials to estimate overfitting
        trial_sharpes = []
        for _ in range(n_trials):
            # Random portfolio (mean return, same volatility)
            random_returns = np.random.normal(mean_return, std_return, len(rewards))
            random_sharpe = np.mean(random_returns) / np.std(random_returns)
            trial_sharpes.append(random_sharpe)

        # Deflated Sharpe Ratio: adjust for the probability of false discovery
        max_trial_sharpe = max(trial_sharpes)
        if sharpe > max_trial_sharpe:
            # Strategy beats random trials
            p_value = sum(1 for s in trial_sharpes if s >= sharpe) / n_trials
            dsr = sharpe * (1 - p_value)
        else:
            dsr = 0.0

        return dsr
