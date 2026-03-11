"""SAC 訓練共通ユーティリティ.

sac_train.py / sac_retrain_scheduler.py の重複ロジックを統合。
- Protocol 定義
- ROI 抽出
- OOS 評価
- Train/Val 分割
- 環境 cleanup
- Buffer サイズ調整
- SB3 stub 回避インポート

References:
  - 375# §2.9: training_metrics ラベル整合
  - 372# audit: 複数エピソード ROI 集約
  - 378# SB3 stub 回避
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Protocols (SB3 SAC / 訓練環境の最小インターフェース)
# ════════════════════════════════════════════════════════════════


class SACModelProtocol(Protocol):
    """SB3 SAC 最小プロトコル."""

    def learn(
        self, total_timesteps: int, reset_num_timesteps: bool = False
    ) -> object: ...

    def predict(
        self, observation: object, deterministic: bool = True
    ) -> tuple[object, object]: ...

    def save(self, path: str) -> None: ...

    def save_replay_buffer(self, path: str) -> None: ...

    def load_replay_buffer(self, path: str) -> None: ...


class TrainingEnvProtocol(Protocol):
    """Training env 最小プロトコル."""

    observation_space: object
    action_space: object

    def reset(self) -> tuple[object, object]: ...

    def step(
        self, action: object
    ) -> tuple[object, float, bool, bool, object]: ...

    def close(self) -> None: ...


# ════════════════════════════════════════════════════════════════
# ROI 抽出
# ════════════════════════════════════════════════════════════════


def extract_roi_from_env(env: TrainingEnvProtocol) -> float:
    """環境から ROI を算出 (duck-typing).

    HeavyTradingEnv は portfolio_value / initial_portfolio_value を保持しており、
    Protocol の最小インターフェースには含まない属性をランタイムで安全に取得する.
    """
    portfolio_value = getattr(env, "portfolio_value", None)
    initial_value = getattr(env, "initial_portfolio_value", None)
    if (
        portfolio_value is not None
        and initial_value is not None
        and float(initial_value) > 0
    ):
        return (float(portfolio_value) - float(initial_value)) / float(
            initial_value
        )
    return 0.0


# ════════════════════════════════════════════════════════════════
# OOS 評価
# ════════════════════════════════════════════════════════════════


def evaluate_model_oos(
    model: SACModelProtocol,
    env: TrainingEnvProtocol,
    n_episodes: int = 1,
) -> dict[str, float | int]:
    """OOS 評価 — 複数エピソードで ROI / trade_count を正しく集約.

    372# audit fix: env.reset() は trades_count を 0 にリセットするため、
    各エピソード終了時に個別に取得して集約する。

    Note:
        旧 sac_train._evaluate_trained_model は n_episodes > 1 のとき
        最終エピソードの ROI のみ返すバグがあった。本関数で統一。
    """
    total_reward = 0.0
    episode_rois: list[float] = []
    total_trades = 0

    for _ in range(max(n_episodes, 1)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rois.append(extract_roi_from_env(env))
        total_trades += int(getattr(env, "trades_count", 0))

    n_eps = max(n_episodes, 1)
    avg_roi = sum(episode_rois) / len(episode_rois) if episode_rois else 0.0

    return {
        "gross_roi": avg_roi,
        "mean_reward": total_reward / n_eps,
        "trade_count": total_trades,
        "n_episodes": n_eps,
        "gross_pnl": float(getattr(env, "total_pnl", 0.0)),
    }


# ════════════════════════════════════════════════════════════════
# Train/Val split
# ════════════════════════════════════════════════════════════════


def train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """時系列インデックス順で train/val 分割.

    363# A3: 時系列データは shuffle 禁止。末尾 val_ratio% を OOS に使う。

    Args:
        df: 元データ (時系列ソート済み前提)
        val_ratio: OOS 比率 (0.0–0.5 にクランプ)

    Returns:
        (train_df, val_df)
    """
    ratio = max(0.0, min(float(val_ratio), 0.5))
    split_idx = int(len(df) * (1.0 - ratio))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    return train_df, val_df


# ════════════════════════════════════════════════════════════════
# 環境 cleanup
# ════════════════════════════════════════════════════════════════


def cleanup_envs(*envs: TrainingEnvProtocol | None) -> None:
    """複数の環境を安全にクローズ (finally ブロック用)."""
    for env in envs:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Failed to close environment: {e}")


# ════════════════════════════════════════════════════════════════
# Buffer サイズ調整
# ════════════════════════════════════════════════════════════════


def adjust_buffer_size(raw_buffer: int, total_timesteps: int) -> int:
    """Replay buffer サイズを total_timesteps に合わせて動的調整.

    H4: デフォルト 1M は 50K 訓練で 20 倍過剰 → obs_dim × buffer_size でメモリ浪費.
    """
    return min(raw_buffer, max(total_timesteps * 2, 10_000))


# ════════════════════════════════════════════════════════════════
# SB3 stub 回避インポート
# ════════════════════════════════════════════════════════════════


def import_real_sb3() -> object:
    """378# SB3 stub 回避: プロジェクトルートの stub パッケージを回避して
    本物の SB3 (site-packages) をロードする.

    プロジェクトルートに stub ``stable_baselines3/`` ディレクトリがあると
    PYTHONPATH="." / sitecustomize.py 経由で stub が優先される問題を解消。

    Returns:
        本物の ``stable_baselines3`` モジュール (SAC は ``.SAC`` でアクセス)
    """
    import importlib

    # stub 関連モジュールを全て除去
    _sb3_keys = [
        k
        for k in sys.modules
        if k == "stable_baselines3" or k.startswith("stable_baselines3.")
    ]
    for k in _sb3_keys:
        sys.modules.pop(k, None)

    _project_root = str(Path(__file__).resolve().parents[3])
    _removed_paths: list[str] = []
    for _p in list(sys.path):
        if "site-packages" not in _p and (
            _p == "."
            or _p == _project_root
            or _p.rstrip("/\\") == _project_root.rstrip("/\\")
        ):
            sys.path.remove(_p)
            _removed_paths.append(_p)
    try:
        sb3 = importlib.import_module("stable_baselines3")
        if not hasattr(sb3, "__version__"):
            raise ImportError(
                "Loaded stub stable_baselines3 instead of real SB3. "
                f"Path: {getattr(sb3, '__file__', 'unknown')}"
            )
        logger.info(f"SB3 loaded: v{sb3.__version__} from {sb3.__file__}")
    finally:
        for _p in reversed(_removed_paths):
            if _p not in sys.path:
                sys.path.insert(0, _p)

    return sb3
