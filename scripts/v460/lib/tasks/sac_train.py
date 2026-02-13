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
from typing import Any, Dict

logger = logging.getLogger(__name__)


def task_sac_train(cfg: dict) -> dict:
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
    seed = cfg.get("seed", 42)
    training_cfg = cfg.get("training", {})
    sac_cfg = cfg.get("sac_hyperparameters", {})
    data_cfg = cfg["data"]
    total_timesteps = training_cfg.get("total_timesteps", 50_000)

    logger.info(f"SAC Training | seed={seed} | timesteps={total_timesteps}")

    # ── Data loading ──
    data_path = data_cfg.get("v460_features_path") or data_cfg.get("ohlcv_path")
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
    raw_buffer = sac_cfg.get("buffer_size", 1_000_000)
    sac_cfg = dict(sac_cfg)  # 元 cfg を汚さないようコピー
    sac_cfg["buffer_size"] = min(raw_buffer, max(total_timesteps * 2, 10_000))
    if sac_cfg["buffer_size"] != raw_buffer:
        logger.info(
            f"Replay buffer adjusted: {raw_buffer:,} → {sac_cfg['buffer_size']:,} "
            f"(aligned to 2× timesteps)"
        )

    env: Any = None
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
        model_dir = Path(cfg.get("output", {}).get("model_dir", "models/v460"))
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
    results: Dict[str, Any] = {
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
    df: "Any", cfg: dict
) -> tuple["Any", Dict[str, Any]]:
    """訓練環境を作成.

    現時点では HeavyTradingEnv を使用 (016# F2: 環境切替は別チケット).
    FastIntradayEnvV456 への移行は段階的に行う.
    """
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    env_cfg = cfg.get("environment", {})
    feature_columns = cfg.get("features", {}).get("selected", [])

    # Construct EnvironmentConfig
    env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()

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
    }

    return env, env_info


def _create_sac_model(
    env: "Any",
    sac_cfg: dict,
    seed: int,
) -> "Any":
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

    return model


def _train_with_checkpoints(
    model: "Any",
    total_timesteps: int,
    cfg: dict,
) -> list[Dict[str, Any]]:
    """チェックポイント毎に指標を収集しながら訓練.

    000# §3.4: 「30K 以降で ROI 変動 ≤ 5%」の判定に必要.
    """
    checkpoint_interval = cfg.get("training", {}).get("checkpoint_interval", 10_000)
    checkpoint_metrics: list[Dict[str, Any]] = []

    remaining = total_timesteps
    trained = 0

    while remaining > 0:
        steps = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        trained += steps
        remaining -= steps

        # Collect checkpoint metrics (placeholder — env metrics are env-dependent)
        metrics: Dict[str, Any] = {
            "timesteps": trained,
        }
        checkpoint_metrics.append(metrics)
        logger.info(f"  Checkpoint @ {trained}/{total_timesteps} steps")

    return checkpoint_metrics


def _evaluate_trained_model(
    model: "Any",
    env: "Any",
    cfg: dict,
) -> Dict[str, Any]:
    """訓練済みモデルを評価 (in-sample).

    G2 判定に必要な指標を収集:
      - gross_pnl
      - roi
      - trade_count
    """
    n_eval_episodes = cfg.get("evaluation", {}).get("n_episodes", 1)

    total_reward = 0.0
    total_steps = 0

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

        total_reward += ep_reward

    eval_metrics: Dict[str, Any] = {
        "mean_reward": total_reward / max(n_eval_episodes, 1),
        "total_steps": total_steps,
        "n_episodes": n_eval_episodes,
    }

    # Extract environment-specific metrics if available
    env_metrics = getattr(env, "get_metrics", lambda: {})()
    if isinstance(env_metrics, dict):
        eval_metrics.update(env_metrics)

    return eval_metrics


def _save_model_schema(
    model: "Any",
    env: "Any",
    env_info: Dict[str, Any],
    cfg: dict,
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

        feature_names = cfg.get("features", {}).get("selected", [])
        if not feature_names:
            logger.warning("No feature names in config — schema will be minimal")
            feature_names = [f"feature_{i}" for i in range(env_info["obs_dim"])]

        training_config = {
            "algorithm": "sac",
            "version": "v460",
            "env_type": env_info["env_type"],
            "obs_dim": env_info["obs_dim"],
            "action_dim": env_info["action_dim"],
            "seed": seed,
            "sac_hyperparameters": cfg.get("sac_hyperparameters", {}),
        }

        manager.save_schema(
            features=feature_names,
            config=training_config,
        )
        logger.info(f"Schema saved for model: {model_name}")

    except Exception as e:
        logger.warning(f"Failed to save model schema (non-fatal): {e}")
