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
from typing import cast

import torch
import pandas as pd

from scripts.v460.lib.config_access import as_int, section
from scripts.v460.lib.sac_common import (
    SACModelProtocol,
    TrainingEnvProtocol,
    adjust_buffer_size,
    cleanup_envs,
    evaluate_model_oos,
    extract_roi_from_env,
    train_val_split,
)
from ztb.types.common import ConfigSection

logger = logging.getLogger(__name__)

# Protocol aliases (後方互換)
SACTrainModelProtocol = SACModelProtocol


def task_sac_train(cfg: ConfigSection) -> dict[str, object]:
    """G2 SAC training task.

    000# §3.4 Go条件:
      - gross > 0 の seed 比率 ≥ 3/4 (75%)
      - ROI の seed 間 σ ≤ 0.03  (363# A4: ic → roi)
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

    # ── 363# A3: Time-series train/val split ──
    # 361# F1 / 362# G1: 同一データで train+eval は in-sample 過学習
    val_ratio_raw = training_cfg.get("val_ratio", 0.2)
    val_ratio = float(val_ratio_raw) if val_ratio_raw else 0.2
    train_df, val_df = train_val_split(df, val_ratio)
    train_size = len(train_df)
    val_size = len(val_df)
    del df  # 全体 DataFrame を早期解放
    logger.info(
        f"Train/Val split: {train_size} / {val_size} rows "
        f"(val_ratio={val_ratio:.2f})"
    )

    # ── H4: Replay buffer を total_timesteps に合わせて動的調整 ──
    # デフォルト 1M は 50K 訓練で 20 倍過剰 → obs_dim × buffer_size でメモリ浪費
    raw_buffer_value = sac_cfg.get("buffer_size", 1_000_000)
    raw_buffer = as_int(raw_buffer_value, 1_000_000)
    # S-3: 元 cfg を汚さないようコピー (sac_params で shadowing 回避)
    sac_params = dict(sac_cfg)
    sac_params["buffer_size"] = adjust_buffer_size(raw_buffer, total_timesteps)
    if cast(int, sac_params["buffer_size"]) != raw_buffer:
        logger.info(
            f"Replay buffer adjusted: {raw_buffer:,} → {sac_params['buffer_size']:,} "
            f"(aligned to 2× timesteps)"
        )

    env: TrainingEnvProtocol | None = None
    eval_env: TrainingEnvProtocol | None = None
    try:
        # ── Training environment (in-sample) ──
        env, env_info = _create_training_env(train_df, cfg)
        logger.info(f"Training env: obs_dim={env_info['obs_dim']}, action_dim={env_info['action_dim']}")

        # ── Model creation (or warm-start from pretrained) ──
        # 365# P1: Warm-start incremental training with replay buffer
        pretrained_path_raw = training_cfg.get("pretrained_model_path")
        pretrained_buffer_raw = training_cfg.get("pretrained_buffer_path")
        pretrained_path = str(pretrained_path_raw) if pretrained_path_raw else ""
        pretrained_buffer = str(pretrained_buffer_raw) if pretrained_buffer_raw else ""

        if pretrained_path and Path(pretrained_path).exists():
            from stable_baselines3 import SAC as SB3_SAC

            model = SB3_SAC.load(pretrained_path, env=env)
            logger.info(f"Warm-start: loaded pretrained model from {pretrained_path}")

            if pretrained_buffer and Path(pretrained_buffer).exists():
                model.load_replay_buffer(pretrained_buffer)
                logger.info(f"Warm-start: loaded replay buffer from {pretrained_buffer}")

            # Use incremental timesteps for warm-start (default: total_timesteps)
            incremental_raw = training_cfg.get("incremental_timesteps")
            if incremental_raw is not None:
                total_timesteps = as_int(incremental_raw, total_timesteps)
                logger.info(f"Warm-start: incremental training for {total_timesteps} steps")
        else:
            model = _create_sac_model(env, sac_params, seed)
        logger.info("SAC model ready")

        # ── Training ──
        start_time = time.time()
        checkpoint_metrics = _train_with_checkpoints(model, env, total_timesteps, cfg)
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")

        # ── 363# A3: Out-of-sample evaluation ──
        eval_env, _ = _create_training_env(val_df, cfg)
        logger.info(f"Validation env: {val_size} rows (out-of-sample)")
        eval_metrics = _evaluate_trained_model(model, eval_env, cfg)

        # ── Save model ──
        output_cfg = section(cfg, "output")
        model_dir_raw = output_cfg.get("model_dir", "models/v460")
        model_dir = Path(str(model_dir_raw))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"sac_v460_seed{seed}.zip"
        model.save(str(model_path))
        logger.info(f"Model saved: {model_path}")

        # ── 365# P1: Save replay buffer for warm-start ──
        buffer_path = model_dir / f"sac_v460_seed{seed}.buffer.pkl"
        try:
            model.save_replay_buffer(str(buffer_path))
            logger.info(f"Replay buffer saved: {buffer_path}")
        except Exception as e:
            logger.warning(f"Replay buffer save failed (non-critical): {e}")

        # ── Save schema metadata alongside model ──
        _save_model_schema(model, env, env_info, cfg, seed)

    finally:
        # C2: 環境を確実にクローズしてメモリ解放
        cleanup_envs(eval_env, env)
        # DataFrame 参照を明示的に解放
        del train_df, val_df

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
        "train_val_split": {
            "val_ratio": val_ratio,
            "train_rows": train_size,
            "val_rows": val_size,
        },
    }

    return results


# ======================================================================
# Internal helpers
# ======================================================================


def _create_training_env(
    df: pd.DataFrame, cfg: ConfigSection
) -> tuple[TrainingEnvProtocol, dict[str, int | str | bool]]:
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

    env_info: dict[str, int | str | bool] = {
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
    env: TrainingEnvProtocol,
    total_timesteps: int,
    cfg: ConfigSection,
) -> list[dict[str, int | float]]:
    """チェックポイント毎に指標を収集しながら訓練.

    000# §3.4: 「30K 以降で ROI 変動 ≤ 5%」の判定に必要.
    359# L-3: 各チェックポイントで ROI を記録し、E3 convergence 判定を有効化.
    """
    training_cfg = section(cfg, "training")
    checkpoint_interval_raw = training_cfg.get("checkpoint_interval", 10_000)
    checkpoint_interval = as_int(checkpoint_interval_raw, 10_000)
    checkpoint_metrics: list[dict[str, int | float]] = []

    remaining = total_timesteps
    trained = 0

    while remaining > 0:
        steps = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        trained += steps
        remaining -= steps

        # 359# L-3: Checkpoint ROI — 1-episode deterministic eval
        roi = _checkpoint_eval_roi(model, env)

        metrics: dict[str, int | float] = {
            "timesteps": trained,
            "roi": roi,
        }
        checkpoint_metrics.append(metrics)
        logger.info(
            f"  Checkpoint @ {trained}/{total_timesteps} steps | roi={roi:.4f}"
        )

    return checkpoint_metrics


def _checkpoint_eval_roi(
    model: SACTrainModelProtocol,
    env: TrainingEnvProtocol,
) -> float:
    """1-episode deterministic eval で ROI を算出 (in-sample convergence 用).

    359# L-3: _train_with_checkpoints から各チェックポイントで呼ばれ、
    E3 convergence 判定に必要な ROI 時系列を生成する.

    Note:
        363# A3: 訓練と同一の train_env を使う (in-sample convergence monitoring).
        最終 G2 gate 評価 (E1/E4) は train/val split 済みの val_env で実施.
        eval 中の env.reset() が SB3 _last_obs と衝突するが、
        50K steps 中 ~10 遷移 (0.02%) と影響軽微.
    """
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return extract_roi_from_env(env)


def _evaluate_trained_model(
    model: SACTrainModelProtocol,
    env: TrainingEnvProtocol,
    cfg: ConfigSection,
) -> dict[str, object]:
    """訓練済みモデルを評価.

    363# A3: val_env (out-of-sample) を渡して G2 gate の信頼性を担保.
    361# F1 / 362# G1: in-sample 評価は過学習を検出できない.

    内部は sac_common.evaluate_model_oos に委譲。
    372# audit fix: 複数エピソードの ROI / trade_count を正しく集約.
    """
    eval_cfg = section(cfg, "evaluation")
    n_eval_episodes_raw = eval_cfg.get("n_episodes", 1)
    n_eval_episodes = as_int(n_eval_episodes_raw, 1)

    return evaluate_model_oos(model, env, n_episodes=n_eval_episodes)


def _save_model_schema(
    model: SACTrainModelProtocol,
    env: TrainingEnvProtocol,
    env_info: dict[str, int | str | bool],
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
            obs_dim = int(env_info["obs_dim"])  # S-2: int 保証
            feature_names = [f"feature_{i}" for i in range(obs_dim)]

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
