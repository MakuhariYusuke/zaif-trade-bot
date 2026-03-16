"""SAC 訓練共通ユーティリティ.

sac_train.py / sac_retrain_scheduler.py の重複ロジックを統合。
- Protocol 定義
- ROI 抽出
- OOS 評価
- Train/Val 分割
- 環境 cleanup
- Buffer サイズ調整

References:
  - 396# evaluate_model_oos G3 拡張: PF/Sharpe/MaxDD/alignment 追加
  - 384# pipeline fixes: CRITICAL-1/2, HIGH-1/2
  - 375# §2.9: training_metrics ラベル整合
  - 372# audit: 複数エピソード ROI 集約
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

import numpy as np
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
    max_steps_per_episode: int | None = None,
) -> dict[str, float | int]:
    """OOS 評価 — holdout 全データで 1-pass deterministic eval.

    384# CRITICAL-2 fix:
      - max_steps_per_episode を撤廃 (デフォルト None = 全走査)。
        旧デフォルト 10K は 243K OOS の 4% しか評価せず、G2 gate 不正確。
      - n_episodes > 1 は random_start=True の場合のみ意味がある。
        random_start=False (デフォルト) では reset() → step=0 で同一データを
        繰り返すだけなので、G2 gate には n_episodes=1 を推奨。
      - gross_pnl を全エピソード集約 (旧: 最終エピソードのみ)。

    372# audit fix: env.reset() は trades_count を 0 にリセットするため、
    各エピソード終了時に個別に取得して集約する。

    396# G3 拡張 (389# P1-1 準拠):
      既存 run_g3_judgment() が期待する seed_metrics 形式を直接出力。
      env.reward_history / portfolio_value_history は 100 件に切り詰められ、
      pnl_history は env に存在しないため、step ループ内で直接収集する。
      - profit_factor: ステップ realized PnL の正/|負| 比
      - max_drawdown: equity curve の最大下落率 (0-1)
      - sharpe_annual: 日次リターンの年率 Sharpe
      - avg_gross_per_trade, avg_fee_per_trade: 取引あたり収益/費用
      - reward_profit_corr: reward 累積と PnL 累積の相関 (alignment 指標)
    """
    total_reward = 0.0
    episode_rois: list[float] = []
    episode_pnls: list[float] = []
    total_trades = 0

    # 396# G3 拡張: ステップごとに直接取得
    # env.reward_history / portfolio_value_history は 100件に切り詰められるため
    # OOS 全走査 (~243K steps) の指標計算には使えない。
    # ステップループ内で直接 env 属性を読み取り、自前で蓄積する。
    all_portfolio_values: list[float] = []
    all_pnl_steps: list[float] = []
    all_reward_steps: list[float] = []

    effective_max = max_steps_per_episode if max_steps_per_episode is not None else int(1e9)

    for _ in range(max(n_episodes, 1)):
        obs, _ = env.reset()
        done = False
        steps = 0
        prev_total_pnl = float(getattr(env, "total_pnl", 0.0))

        while not done and steps < effective_max:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            # 396# ステップごとのデータ収集 (env 履歴切り詰め問題の回避)
            # 398# FIX: production env (heavy_env/core.py) は balance/unrealized_pnl
            # を持たない。portfolio_value プロパティを使う。
            pv = float(getattr(env, "portfolio_value", 0.0))
            if pv == 0.0:
                # fallback: training env (heavy_trading_env.py) 互換
                pv = float(getattr(env, "balance", 0.0)) + float(getattr(env, "unrealized_pnl", 0.0))
            all_portfolio_values.append(pv)
            all_reward_steps.append(float(reward))

            # step-level realized PnL delta (total_pnl の変化量)
            cur_total_pnl = float(getattr(env, "total_pnl", 0.0))
            all_pnl_steps.append(cur_total_pnl - prev_total_pnl)
            prev_total_pnl = cur_total_pnl

        episode_rois.append(extract_roi_from_env(env))
        episode_pnls.append(float(getattr(env, "total_pnl", 0.0)))
        total_trades += int(getattr(env, "trades_count", 0))

    n_eps = max(n_episodes, 1)
    avg_roi = sum(episode_rois) / len(episode_rois) if episode_rois else 0.0
    avg_pnl = sum(episode_pnls) / len(episode_pnls) if episode_pnls else 0.0

    result: dict[str, float | int] = {
        "gross_roi": avg_roi,
        "mean_reward": total_reward / n_eps,
        "trade_count": total_trades,
        "n_episodes": n_eps,
        "gross_pnl": avg_pnl,
    }

    # 396# G3 指標計算
    g3 = _compute_g3_metrics(
        all_portfolio_values, all_pnl_steps, all_reward_steps, total_trades
    )
    result.update(g3)

    # 425# P0-3: Multi-slice OOS 評価 (先頭/中間/末尾 3分割)
    # OOS 後半での performance 劣化を検出するための診断指標。
    # F6 checkpoint eval が OOS 先頭 5K step のみで best_model を選定
    # する問題の diagnostic データを提供する。
    n_steps = len(all_portfolio_values)
    if n_steps >= 4320:  # 3日分 (1440×3) 以上あれば 3分割意味がある
        slice_size = n_steps // 3
        slice_metrics_list: list[dict[str, float]] = []
        slice_labels = ["early", "mid", "late"]
        for i in range(3):
            s_start = i * slice_size
            s_end = s_start + slice_size if i < 2 else n_steps
            s_pv = all_portfolio_values[s_start:s_end]
            s_pnl = all_pnl_steps[s_start:s_end]
            s_reward = all_reward_steps[s_start:s_end]
            # per-trade count は分割不能なので 0 で代替
            s_g3 = _compute_g3_metrics(s_pv, s_pnl, s_reward, 0)
            s_g3["label"] = slice_labels[i]  # type: ignore[assignment]
            s_g3["steps"] = float(s_end - s_start)
            slice_metrics_list.append(s_g3)
        result["slice_metrics"] = slice_metrics_list  # type: ignore[assignment]

    return result


def _compute_g3_metrics(
    portfolio_values: list[float],
    pnl_steps: list[float],
    reward_steps: list[float],
    total_trades: int,
) -> dict[str, float]:
    """G3 Gate 用指標を計算.

    389# P1-1: 既存 run_g3_judgment() が期待する seed_metrics 形式に合わせる。
    392# P0-3: reward_profit_corr を追加 (reward-profit alignment 指標)。
    """
    metrics: dict[str, float] = {}

    # --- Profit Factor ---
    # pnl_steps は各ステップの realized PnL (trade 発生時のみ非ゼロ)
    if pnl_steps:
        pos_pnl = sum(p for p in pnl_steps if p > 0)
        neg_pnl = abs(sum(p for p in pnl_steps if p < 0))
        metrics["pf"] = pos_pnl / neg_pnl if neg_pnl > 0 else float("inf") if pos_pnl > 0 else 0.0
    else:
        metrics["pf"] = 0.0

    # --- Max Drawdown ---
    if portfolio_values and len(portfolio_values) > 1:
        peak = portfolio_values[0]
        max_dd = 0.0
        for pv in portfolio_values:
            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        metrics["max_drawdown"] = max_dd
    else:
        metrics["max_drawdown"] = 0.0

    # --- Sharpe (年率) ---
    # 1分足なので ~1440 steps/day。日次リターンに集約してから年率化。
    if portfolio_values and len(portfolio_values) > 1440:
        pv_arr = np.array(portfolio_values, dtype=np.float64)
        # 日次リターン: 1440 steps ごとに集約
        n_days = len(pv_arr) // 1440
        if n_days >= 2:
            daily_pv = pv_arr[::1440][:n_days + 1]  # 日次サンプリング
            daily_returns = np.diff(daily_pv) / daily_pv[:-1]
            daily_returns = daily_returns[np.isfinite(daily_returns)]
            if len(daily_returns) >= 2:
                mean_r = float(np.mean(daily_returns))
                std_r = float(np.std(daily_returns, ddof=1))
                metrics["sharpe_annual"] = (
                    (mean_r / std_r) * math.sqrt(365) if std_r > 1e-10 else 0.0
                )
            else:
                metrics["sharpe_annual"] = 0.0
        else:
            metrics["sharpe_annual"] = 0.0
    else:
        metrics["sharpe_annual"] = 0.0

    # --- Per-trade averages (G3 E3: avg gross > avg fee) ---
    # fee=0 (maker 0%) 前提では avg_fee=0 だが形式的に出力
    if total_trades > 0 and pnl_steps:
        trade_pnls = [p for p in pnl_steps if p != 0.0]
        if trade_pnls:
            # 408# B4: abs() を除去。avg_gross = 取引あたり平均損益（手数料控除前）。
            # abs() だと損失の絶対値が加算され意味的に誤り。
            # G3 E3「avg_gross > avg_fee」は「平均利益 > 平均コスト」を意図。
            metrics["avg_gross_per_trade"] = sum(trade_pnls) / len(trade_pnls)
        else:
            metrics["avg_gross_per_trade"] = 0.0
        metrics["avg_fee_per_trade"] = 0.0  # maker 0%
    else:
        metrics["avg_gross_per_trade"] = 0.0
        metrics["avg_fee_per_trade"] = 0.0

    # --- Reward-Profit Alignment (392# P0-3) ---
    # reward 累積と PnL 累積の相関を計算。負相関 = 学習目標が利益からズレている
    if reward_steps and pnl_steps and len(reward_steps) == len(pnl_steps):
        n = min(len(reward_steps), len(pnl_steps))
        if n >= 100:
            # ダウンサンプリングして計算コスト削減 (1000点)
            step = max(1, n // 1000)
            r_cum = np.cumsum(reward_steps[:n])[::step]
            p_cum = np.cumsum(pnl_steps[:n])[::step]
            if len(r_cum) >= 2 and np.std(r_cum) > 1e-10 and np.std(p_cum) > 1e-10:
                corr = float(np.corrcoef(r_cum, p_cum)[0, 1])
                metrics["reward_profit_corr"] = corr if np.isfinite(corr) else 0.0
            else:
                metrics["reward_profit_corr"] = 0.0
        else:
            metrics["reward_profit_corr"] = 0.0
    else:
        metrics["reward_profit_corr"] = 0.0

    return metrics


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

    # 408# B5: 空 DataFrame ガード — HeavyTradingEnv に空データを渡すとクラッシュ
    if len(train_df) == 0:
        raise ValueError(
            f"train_val_split produced empty train_df "
            f"(df rows={len(df)}, val_ratio={val_ratio})"
        )
    if len(val_df) == 0:
        raise ValueError(
            f"train_val_split produced empty val_df "
            f"(df rows={len(df)}, val_ratio={val_ratio}). "
            f"Set val_ratio > 0 or provide more data."
        )

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


# 384# import_real_sb3() は削除 — 379# でスタブを _sb3_test_stub/ にリネーム済み。
# pip版 SB3 2.7.0 が正常にロードされるため、sys.path/modules 操作は不要。
# 旧コール元 (sac_retrain_scheduler.py) は直接 `from stable_baselines3 import SAC` に移行。
