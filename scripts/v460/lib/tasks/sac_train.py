"""
G2 SAC Training — task implementation.

001# P3-1: run_experiment.py へ sac_train タスクを追加.
017# P0: ph2 並行で枠実装.

SAC 訓練を統一 SACTrainer 経由で実行し、
G2-train Gate 判定用の指標を収集する.

Usage (via run_experiment.py):
  python scripts/v460/run_experiment.py \\
    --config configs/v460/experiments/g2_sac_train.yaml \\
    --seed 42
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from scripts.v460.lib.config_access import as_int, section
from ztb.types.common import ConfigSection

logger = logging.getLogger(__name__)


class TrainingEnvProtocol(Protocol):
    observation_space: object
    action_space: object

    def reset(self) -> tuple[object, object]:
        ...

    def step(self, action: object) -> tuple[object, float, bool, bool, object]:
        ...

    def close(self) -> None:
        ...


class SACTrainModelProtocol(Protocol):
    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False) -> object:
        ...

    def predict(self, observation: object, deterministic: bool = True) -> tuple[object, object]:
        ...

    def save(self, path: str) -> None:
        ...


def task_sac_train(cfg: ConfigSection) -> dict[str, object]:
    """G2 SAC training task.

    000# §3.4 Go条件:
      - gross > 0 の seed 比率 ≥ 3/4 (75%)
      - IC の seed 間 σ ≤ 0.03
      - 30K 以降で ROI 変動 ≤ 5%
      - worst-seed ROI > −2%

    測定方法: 4 seed × 50K steps.

    Returns:
        results dict with training metrics for G2 gate judgment.
    """
    training_cfg = section(cfg, "training")
    sac_cfg = section(cfg, "sac_hyperparameters")
    data_cfg = section(cfg, "data")
    total_timesteps_raw = training_cfg.get("total_timesteps", 50_000)
    total_timesteps = as_int(total_timesteps_raw, 50_000)
    seed_raw = cfg.get("seed", 42)
    seed = as_int(seed_raw, 42)

    logger.info(f"SAC Training | seed={seed} | timesteps={total_timesteps}")

    # ── Data loading ──
    data_path_raw = data_cfg.get("v460_features_path") or data_cfg.get("ohlcv_path")
    data_path = str(data_path_raw) if data_path_raw else ""
    if not data_path:
        raise ValueError("data.v460_features_path or data.ohlcv_path is required")

    logger.info(f"Data path: {data_path}")

    # ── SAC Training (Guard: SB3 must be available) ──
    try:
        from stable_baselines3 import SAC  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 is required for sac_train task. "
            "Install with: pip install stable-baselines3"
        ) from e

    from scripts.v460.lib.data_loader import load_parquet

    df = load_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # ── H4: Replay buffer を total_timesteps に合わせて動的調整 ──
    # デフォルト 1M は 50K 訓練で 20 倍過剰 → obs_dim × buffer_size でメモリ浪費
    raw_buffer_value = sac_cfg.get("buffer_size", 1_000_000)
    raw_buffer = as_int(raw_buffer_value, 1_000_000)
    sac_cfg = dict(sac_cfg)  # 元 cfg を汚さないようコピー
    sac_cfg["buffer_size"] = min(raw_buffer, max(total_timesteps * 2, 10_000))
    if cast(int, sac_cfg["buffer_size"]) != raw_buffer:
        logger.info(
            f"Replay buffer adjusted: {raw_buffer:,} → {sac_cfg['buffer_size']:,} "
            f"(aligned to 2× timesteps)"
        )

    env: TrainingEnvProtocol | None = None
    try:
        # ── Environment setup ──
        env, env_info = _create_training_env(df, cfg)
        logger.info(f"Environment created: obs_dim={env_info['obs_dim']}, action_dim={env_info['action_dim']}")

        # ── Model creation ──
        model = _create_sac_model(env, sac_cfg, seed)
        logger.info("SAC model created")

        # ── Training ──
        start_time = time.time()
        checkpoint_metrics = _train_with_checkpoints(model, total_timesteps, cfg)
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")

        # ── Evaluation ──
        eval_metrics = _evaluate_trained_model(model, env, cfg)

        # ── Save model ──
        output_cfg = section(cfg, "output")
        model_dir_raw = output_cfg.get("model_dir", "models/v460")
        model_dir = Path(str(model_dir_raw))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"sac_v460_seed{seed}.zip"
        model.save(str(model_path))
        logger.info(f"Model saved: {model_path}")

        # ── Save schema metadata alongside model ──
        _save_model_schema(model, env, env_info, cfg, seed)

    finally:
        # C2: 環境を確実にクローズしてメモリ解放
        if env is not None:
            try:
                env.close()
                logger.debug("Environment closed")
            except Exception as e:
                logger.warning(f"Failed to close environment: {e}")
        # DataFrame 参照を明示的に解放
        del df

    # ── Results ──
    results: dict[str, object] = {
        "algorithm": "sac",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_time_sec": round(elapsed, 1),
        "model_path": str(model_path),
        "env_info": env_info,
        "checkpoint_metrics": checkpoint_metrics,
        "eval_metrics": eval_metrics,
    }

    return results


# ======================================================================
# Internal helpers
# ======================================================================


def _create_training_env(
    df: pd.DataFrame, cfg: ConfigSection
) -> tuple[TrainingEnvProtocol, dict[str, int | str]]:
    """訓練環境を作成.

    現時点では HeavyTradingEnv を使用 (016# F2: 環境切替は別チケット).
    FastIntradayEnvV456 への移行は段階的に行う.

    356# B3: feature_columns を EnvironmentConfig.feature_names に注入.
    """
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    env_cfg = section(cfg, "environment")
    feature_cfg = section(cfg, "features")
    selected_raw = feature_cfg.get("selected", [])
    feature_columns = [str(col) for col in selected_raw] if isinstance(selected_raw, list) else []

    # Construct EnvironmentConfig
    env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()

    # 356# B3: 明示的に feature_names を注入
    # config.features.selected で指定した特徴量のみを observation space に含める
    if feature_columns:
        env_config.feature_names = feature_columns

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
    )

    obs_dim = env.observation_space.shape[0] if env.observation_space.shape else 0
    action_dim = (
        env.action_space.shape[0]
        if hasattr(env.action_space, "shape") and env.action_space.shape
        else int(getattr(env.action_space, "n", 0))
    )

    env_info = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_type": "HeavyTradingEnv",
        "feature_columns_count": len(feature_columns),
        "feature_columns_injected": bool(feature_columns),  # 356# B3 可観測性
    }

    return cast(TrainingEnvProtocol, env), env_info


def _create_sac_model(
    env: TrainingEnvProtocol,
    sac_cfg: ConfigSection,
    seed: int,
) -> SACTrainModelProtocol:
    """SB3 SAC モデルを作成."""
    from stable_baselines3 import SAC

    # Defaults aligned with unified SACTrainer constants
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_cfg.get("learning_rate", 3e-4),
        buffer_size=sac_cfg.get("buffer_size", 1_000_000),
        learning_starts=sac_cfg.get("learning_starts", 1_000),
        batch_size=sac_cfg.get("batch_size", 256),
        tau=sac_cfg.get("tau", 0.005),
        gamma=sac_cfg.get("gamma", 0.99),
        train_freq=sac_cfg.get("train_freq", 1),
        gradient_steps=sac_cfg.get("gradient_steps", 1),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        verbose=0,
        seed=seed,
    )

    return cast(SACTrainModelProtocol, model)


def _train_with_checkpoints(
    model: SACTrainModelProtocol,
    total_timesteps: int,
    cfg: ConfigSection,
) -> list[dict[str, int]]:
    """チェックポイント毎に指標を収集しながら訓練.

    000# §3.4: 「30K 以降で ROI 変動 ≤ 5%」の判定に必要.
    """
    training_cfg = section(cfg, "training")
    checkpoint_interval_raw = training_cfg.get("checkpoint_interval", 10_000)
    checkpoint_interval = as_int(checkpoint_interval_raw, 10_000)
    checkpoint_metrics: list[dict[str, int]] = []

    remaining = total_timesteps
    trained = 0

    while remaining > 0:
        steps = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        trained += steps
        remaining -= steps

        # Collect checkpoint metrics (placeholder — env metrics are env-dependent)
        metrics: dict[str, int] = {
            "timesteps": trained,
        }
        checkpoint_metrics.append(metrics)
        logger.info(f"  Checkpoint @ {trained}/{total_timesteps} steps")

    return checkpoint_metrics


def _evaluate_trained_model(
    model: SACTrainModelProtocol,
    env: TrainingEnvProtocol,
    cfg: ConfigSection,
) -> dict[str, object]:
    """訓練済みモデルを評価 (in-sample).

    G2 判定に必要な指標を収集:
      - gross_pnl
      - roi
      - trade_count
    """
    eval_cfg = section(cfg, "evaluation")
    n_eval_episodes_raw = eval_cfg.get("n_episodes", 1)
    n_eval_episodes = as_int(n_eval_episodes_raw, 1)

    total_reward = 0.0
    total_steps = 0

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

        total_reward += ep_reward

    eval_metrics: dict[str, object] = {
        "mean_reward": total_reward / max(n_eval_episodes, 1),
        "total_steps": total_steps,
        "n_episodes": n_eval_episodes,
    }

    # Extract environment-specific metrics if available
    get_metrics_fn = getattr(env, "get_metrics", None)
    if callable(get_metrics_fn):
        env_metrics = get_metrics_fn()
        if isinstance(env_metrics, dict):
            eval_metrics.update({str(k): v for k, v in env_metrics.items()})

    return eval_metrics


def _save_model_schema(
    model: SACTrainModelProtocol,
    env: TrainingEnvProtocol,
    env_info: dict[str, int | str],
    cfg: ConfigSection,
    seed: int,
) -> None:
    """モデルのスキーマメタデータを保存 (017# F4/F9 対応).

    推論時に ActionPrediction が observation_space から
    正しい次元を解決できるようにする.
    """
    try:
        from ztb.training.core.feature_schema_manager import FeatureSchemaManager

        model_name = f"sac_v460_seed{seed}"
        manager = FeatureSchemaManager(model_name=model_name)

        feature_cfg = section(cfg, "features")
        selected_raw = feature_cfg.get("selected", [])
        feature_names = [str(col) for col in selected_raw] if isinstance(selected_raw, list) else []
        if not feature_names:
            logger.warning("No feature names in config — schema will be minimal")
            feature_names = [f"feature_{i}" for i in range(env_info["obs_dim"])]

        sac_hyperparameters = section(cfg, "sac_hyperparameters")

        training_config: dict[str, object] = {
            "algorithm": "sac",
            "version": "v460",
            "env_type": env_info["env_type"],
            "obs_dim": env_info["obs_dim"],
            "action_dim": env_info["action_dim"],
            "seed": seed,
            "sac_hyperparameters": sac_hyperparameters,
        }

        manager.save_schema(
            features=feature_names,
            config=training_config,
        )
        logger.info(f"Schema saved for model: {model_name}")

    except Exception as e:
        logger.warning(f"Failed to save model schema (non-fatal): {e}")
