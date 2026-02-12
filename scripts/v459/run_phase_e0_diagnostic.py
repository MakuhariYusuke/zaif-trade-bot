#!/usr/bin/env python3
"""
Phase E0 診断スクリプト — SAC 学習品質の多面的評価

114#/115# Phase E0 定義:
  Q1: SAC は方向予測を学習しているか (多面IC)
  Q2: 10K→50K の劣化は過学習か (チェックポイント学習曲線)
  Q3: threshold は eval 時のみの効果か (threshold感度分析)

Usage:
  python scripts/v459/run_phase_e0_diagnostic.py
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.v459.run_phase_c import (
    DATA_PATH,
    INITIAL_BALANCE,
    SAC_DEFAULT,
    REWARD_BASE,
    build_config,
    get_experiment_configs,
    _find_vec_normalize,
    _reset_risk_controllers,
)
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.env_metrics import resolve_env, unwrap_env

OUTPUT_DIR = project_root / "results" / "phase_e0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = OUTPUT_DIR / "e0_diagnostic.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# チェックポイント保存間隔 (Q2: 学習曲線用)
CHECKPOINT_INTERVAL = 10_000
TOTAL_TIMESTEPS = 50_000


# ============================================================================
# Q1: 多面的 IC (Information Coefficient) 評価
# ============================================================================

def run_diagnostic_eval(
    model: Any,
    raw_env: Any,
    max_eval_steps: int,
    threshold: float,
    normalize_fn: Any = None,
    eval_dd_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """拡張 eval: Gate2 メトリクスに加え、step-level の action/price/position を収集。

    run_phase_c._run_deterministic_eval を参考にしつつ、
    IC 診断に必要な追加データを収集する。
    """
    _reset_risk_controllers(raw_env, eval_dd_threshold=eval_dd_threshold)
    # Q3: eval 時 threshold を動的に変更
    if hasattr(raw_env, "action_threshold"):
        raw_env.action_threshold = threshold
        raw_env.negative_action_threshold = -threshold
    obs_raw, _ = raw_env.reset(seed=42, options={"random_start": False})
    obs = normalize_fn(obs_raw.copy()) if normalize_fn else obs_raw
    done = False

    # step-level 収集
    actions: List[float] = []
    prices: List[float] = []
    positions: List[int] = []
    step_indices: List[int] = []  # raw_env.current_step (データ中の絶対位置)
    balances: List[float] = [float(raw_env.portfolio_value)]
    trade_pnls: List[float] = []

    prev_trades_count = int(raw_env.trades_count)
    prev_realized_pnl = float(raw_env.realized_pnl)
    step_count = 0

    while not done and step_count < max_eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_scalar = float(action.flatten()[0]) if hasattr(action, "flatten") else float(action)

        # step前の状態を記録
        step_indices.append(int(raw_env.current_step))
        prices.append(float(raw_env._resolve_price()))

        obs_raw, _reward, terminated, truncated, info = raw_env.step(action)
        done = terminated or truncated
        obs = normalize_fn(obs_raw.copy()) if normalize_fn else obs_raw

        actions.append(action_scalar)
        positions.append(int(info.get("position_after", 0)))
        balances.append(float(raw_env.portfolio_value))
        step_count += 1

        # 取引クローズ検出
        current_tc = int(raw_env.trades_count)
        current_rp = float(raw_env.realized_pnl)
        if current_tc > prev_trades_count:
            trade_pnls.append(current_rp - prev_realized_pnl)
        prev_trades_count = current_tc
        prev_realized_pnl = current_rp

    return {
        "actions": np.array(actions, dtype=np.float64),
        "prices": np.array(prices, dtype=np.float64),
        "positions": np.array(positions, dtype=np.int32),
        "step_indices": np.array(step_indices, dtype=np.int64),
        "balances": np.array(balances, dtype=np.float64),
        "trade_pnls": trade_pnls,
        "eval_steps": step_count,
        "eval_trades": int(raw_env.total_trades),
        "eval_gross_pnl": float(getattr(raw_env, "gross_pnl", 0.0)),
        "eval_total_fees": float(getattr(raw_env, "total_fees", 0.0)),
        "eval_net_roi": float(
            (balances[-1] - balances[0]) / balances[0] * 100
        ) if balances[0] > 0 else 0.0,
        "threshold": threshold,
    }


def compute_ic_multi_horizon(
    actions: np.ndarray,
    prices: np.ndarray,
    horizons: List[int] = [1, 5, 15],
) -> Dict[str, Any]:
    """Q1: 複数 horizon の IC (Pearson + Spearman)。"""
    results = {}
    for h in horizons:
        if len(prices) <= h:
            continue
        price_changes = prices[h:] - prices[:-h]
        acts = actions[:len(price_changes)]

        if np.std(acts) < 1e-10 or np.std(price_changes) < 1e-10:
            results[f"h{h}"] = {"pearson": 0.0, "spearman": 0.0, "p_value": 1.0, "n": len(acts)}
            continue

        pearson = float(np.corrcoef(acts, price_changes)[0, 1])
        sp_corr, sp_p = spearmanr(acts, price_changes)
        results[f"h{h}"] = {
            "pearson": round(pearson, 6),
            "spearman": round(float(sp_corr), 6),
            "p_value": round(float(sp_p), 6),
            "n": len(acts),
        }
    return results


def compute_ic_by_action_bin(
    actions: np.ndarray,
    prices: np.ndarray,
    bins: List[Tuple[float, float]] = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)],
) -> Dict[str, Any]:
    """Q1: |action| bin 別の平均 price_change (1-step ahead)。"""
    if len(prices) < 2:
        return {}
    price_changes = prices[1:] - prices[:-1]
    abs_actions = np.abs(actions[:len(price_changes)])
    results = {}
    for lo, hi in bins:
        mask = (abs_actions >= lo) & (abs_actions < hi)
        n = int(mask.sum())
        if n > 0:
            avg_pc = float(np.mean(price_changes[mask]))
            # action の符号と price_change の整合性
            signed_actions = actions[:len(price_changes)][mask]
            directional_pnl = float(np.mean(signed_actions * price_changes[mask]))
        else:
            avg_pc = 0.0
            directional_pnl = 0.0
        results[f"abs_{lo:.1f}_{hi:.1f}"] = {
            "count": n,
            "avg_price_change": round(avg_pc, 4),
            "avg_directional_pnl": round(directional_pnl, 4),
        }
    return results


def compute_ic_by_session(
    actions: np.ndarray,
    prices: np.ndarray,
    timestamps: pd.Series,
    step_indices: np.ndarray,
) -> Dict[str, Any]:
    """Q1: JST 時間帯別 IC (4セッション)。"""
    if len(prices) < 2:
        return {}
    price_changes_1 = prices[1:] - prices[:-1]
    acts = actions[:len(price_changes_1)]

    # step_indices → timestamp
    ts = timestamps.iloc[step_indices[:len(price_changes_1)]]
    hours_jst = (ts.dt.hour + 9) % 24  # UTC→JST

    sessions = {"00-06": (0, 6), "06-12": (6, 12), "12-18": (12, 18), "18-24": (18, 24)}
    results = {}
    for name, (h_lo, h_hi) in sessions.items():
        mask = (hours_jst >= h_lo) & (hours_jst < h_hi)
        n = int(mask.sum())
        if n > 10 and np.std(acts[mask]) > 1e-10:
            sp_corr, sp_p = spearmanr(acts[mask], price_changes_1[mask])
            results[name] = {
                "spearman": round(float(sp_corr), 6),
                "p_value": round(float(sp_p), 6),
                "n": n,
            }
        else:
            results[name] = {"spearman": 0.0, "p_value": 1.0, "n": n}
    return results


# ============================================================================
# Q2: 学習曲線 (チェックポイント別 eval)
# ============================================================================

def evaluate_checkpoint(
    checkpoint_path: str,
    raw_env: Any,
    max_eval_steps: int,
    threshold: float,
    normalize_fn: Any = None,
) -> Dict[str, Any]:
    """チェックポイントを読み込んで eval し、Gate2 指標を返す。"""
    from stable_baselines3 import SAC
    model = SAC.load(checkpoint_path)
    result = run_diagnostic_eval(
        model, raw_env, max_eval_steps, threshold,
        normalize_fn=normalize_fn, eval_dd_threshold=1.0,
    )
    # IC も計算
    ic = compute_ic_multi_horizon(result["actions"], result["prices"])
    summary = {
        "checkpoint": checkpoint_path,
        "eval_steps": result["eval_steps"],
        "eval_trades": result["eval_trades"],
        "eval_net_roi": round(result["eval_net_roi"], 4),
        "eval_gross_pnl": round(result["eval_gross_pnl"], 2),
        "eval_total_fees": round(result["eval_total_fees"], 2),
        "ic": ic,
    }
    if result["trade_pnls"]:
        summary["avg_net_pnl_per_trade"] = round(float(np.mean(result["trade_pnls"])), 2)
    del model
    gc.collect()
    return summary


# ============================================================================
# Q3: Threshold 感度分析
# ============================================================================

def threshold_sensitivity(
    model: Any,
    raw_env: Any,
    max_eval_steps: int,
    thresholds: List[float],
    normalize_fn: Any = None,
) -> List[Dict[str, Any]]:
    """Q3: 同一モデルで eval 時の threshold だけ変えて比較。"""
    results = []
    for thr in thresholds:
        eval_data = run_diagnostic_eval(
            model, raw_env, max_eval_steps, thr,
            normalize_fn=normalize_fn, eval_dd_threshold=1.0,
        )
        ic_h1 = compute_ic_multi_horizon(eval_data["actions"], eval_data["prices"], [1])
        abs_above = float(np.mean(np.abs(eval_data["actions"]) > thr)) if len(eval_data["actions"]) > 0 else 0.0
        results.append({
            "threshold": thr,
            "eval_trades": eval_data["eval_trades"],
            "eval_net_roi": round(eval_data["eval_net_roi"], 4),
            "eval_gross_pnl": round(eval_data["eval_gross_pnl"], 2),
            "eval_total_fees": round(eval_data["eval_total_fees"], 2),
            "abs_above_threshold_ratio": round(abs_above, 4),
            "ic_h1_spearman": ic_h1.get("h1", {}).get("spearman", 0.0),
        })
    return results


# ============================================================================
# メインフロー
# ============================================================================

def train_with_checkpoints(seed: int = 42) -> Tuple[Any, Any, Any, Any]:
    """d2_thr80 相当の設定で訓練し、10K 間隔でチェックポイント保存。

    Returns: (trainer, model, raw_env, vec_normalize)
    """
    exp_configs = get_experiment_configs()
    exp_def = exp_configs["d2_thr80"]

    config = build_config("e0_diag", seed, exp_def)
    # チェックポイント保存設定
    config["training"]["total_timesteps"] = TOTAL_TIMESTEPS
    config["training"]["checkpoint_freq"] = CHECKPOINT_INTERVAL
    config["model_name"] = "e0_diag_model"

    logger.info(f"Training d2_thr80 config (seed={seed}, steps={TOTAL_TIMESTEPS})...")
    trainer = SACTrainer(config=config, logger=logger)
    trainer.train()

    # env 取得
    vec_env = resolve_env(trainer)
    if vec_env is None and hasattr(trainer, "model") and hasattr(trainer.model, "get_env"):
        vec_env = trainer.model.get_env()
    raw_env = unwrap_env(vec_env) if vec_env is not None else None

    vec_normalize = _find_vec_normalize(vec_env)
    if vec_normalize is not None:
        vec_normalize.training = False

    model = trainer.model if hasattr(trainer, "model") else trainer.algorithm_trainer.model

    return trainer, model, raw_env, vec_normalize


def run_e0_diagnostic() -> Dict[str, Any]:
    """Phase E0 診断の全工程を実行。"""
    start_time = time.time()
    report: Dict[str, Any] = {
        "phase": "E0",
        "timestamp": datetime.now().isoformat(),
        "questions": {},
    }

    # --- 訓練 ---
    logger.info("=" * 60)
    logger.info("Phase E0: 訓練開始 (d2_thr80 config)")
    logger.info("=" * 60)
    trainer, model, raw_env, vec_normalize = train_with_checkpoints(seed=42)

    if raw_env is None:
        logger.error("env not found — aborting")
        return report

    normalize_fn = vec_normalize.normalize_obs if vec_normalize else None
    max_eval_steps = min(
        getattr(raw_env, "n_steps", TOTAL_TIMESTEPS),
        TOTAL_TIMESTEPS,
    )
    threshold_train = 0.80  # d2_thr80 の threshold

    # --- タイムスタンプデータの読み込み (時間帯分析用) ---
    df_ts = pd.read_parquet(DATA_PATH, columns=["timestamp"])
    timestamps = pd.to_datetime(df_ts["timestamp"], utc=True)

    # ================================================================
    # Q1: 多面的 IC 評価
    # ================================================================
    logger.info("=" * 60)
    logger.info("Q1: 多面的 IC 評価 (方向予測力の診断)")
    logger.info("=" * 60)

    eval_data = run_diagnostic_eval(
        model, raw_env, max_eval_steps, threshold_train,
        normalize_fn=normalize_fn, eval_dd_threshold=1.0,
    )

    q1: Dict[str, Any] = {
        "eval_summary": {
            "steps": eval_data["eval_steps"],
            "trades": eval_data["eval_trades"],
            "net_roi": round(eval_data["eval_net_roi"], 4),
            "gross_pnl": round(eval_data["eval_gross_pnl"], 2),
            "total_fees": round(eval_data["eval_total_fees"], 2),
        },
    }

    # IC by horizon
    q1["ic_multi_horizon"] = compute_ic_multi_horizon(
        eval_data["actions"], eval_data["prices"], horizons=[1, 5, 15, 60],
    )
    logger.info(f"  IC results: {json.dumps(q1['ic_multi_horizon'], indent=2)}")

    # IC by action bin
    q1["ic_by_action_bin"] = compute_ic_by_action_bin(
        eval_data["actions"], eval_data["prices"],
    )
    logger.info(f"  Action bin: {json.dumps(q1['ic_by_action_bin'], indent=2)}")

    # IC by session (time of day)
    q1["ic_by_session"] = compute_ic_by_session(
        eval_data["actions"], eval_data["prices"],
        timestamps, eval_data["step_indices"],
    )
    logger.info(f"  Session IC: {json.dumps(q1['ic_by_session'], indent=2)}")

    # Action 分布統計
    acts = eval_data["actions"]
    q1["action_distribution"] = {
        "mean": round(float(np.mean(acts)), 6),
        "std": round(float(np.std(acts)), 6),
        "abs_mean": round(float(np.mean(np.abs(acts))), 6),
        "pct_hold": round(float(np.mean(np.abs(acts) < threshold_train)), 4),
        "pct_buy": round(float(np.mean(acts > threshold_train)), 4),
        "pct_sell": round(float(np.mean(acts < -threshold_train)), 4),
    }

    # Q1 判定
    best_ic = max(
        abs(v.get("spearman", 0))
        for v in q1["ic_multi_horizon"].values()
    ) if q1["ic_multi_horizon"] else 0.0
    q1["verdict"] = (
        "EDGE_DETECTED" if best_ic > 0.05
        else "TINY_EDGE" if best_ic > 0.02
        else "NO_EDGE"
    )
    logger.info(f"  Q1 verdict: {q1['verdict']} (best |IC| = {best_ic:.4f})")
    report["questions"]["Q1"] = q1

    # ================================================================
    # Q2: 学習曲線 (チェックポイント別)
    # ================================================================
    logger.info("=" * 60)
    logger.info("Q2: 学習曲線 (チェックポイント別 eval)")
    logger.info("=" * 60)

    checkpoint_dir = project_root / "models" / "checkpoints"
    q2_results = []
    for step in range(CHECKPOINT_INTERVAL, TOTAL_TIMESTEPS + 1, CHECKPOINT_INTERVAL):
        ckpt_path = checkpoint_dir / f"sac_checkpoint_{step}_steps.zip"
        if ckpt_path.exists():
            logger.info(f"  Evaluating checkpoint: {step} steps")
            ckpt_result = evaluate_checkpoint(
                str(ckpt_path), raw_env, max_eval_steps, threshold_train,
                normalize_fn=normalize_fn,
            )
            ckpt_result["training_steps"] = step
            q2_results.append(ckpt_result)
            logger.info(
                f"    trades={ckpt_result['eval_trades']} "
                f"ROI={ckpt_result['eval_net_roi']:.4f}% "
                f"gross={ckpt_result['eval_gross_pnl']:.0f} "
                f"IC_h1={ckpt_result['ic'].get('h1', {}).get('spearman', 0):.4f}"
            )
        else:
            logger.warning(f"  Checkpoint not found: {ckpt_path}")

    # Q2 判定: 過学習 = 初期ckptのほうが成績良い
    q2: Dict[str, Any] = {"checkpoints": q2_results}
    if len(q2_results) >= 2:
        rois = [r["eval_net_roi"] for r in q2_results]
        best_step_idx = int(np.argmax(rois))
        q2["best_checkpoint_steps"] = q2_results[best_step_idx]["training_steps"]
        q2["roi_trend"] = "degrading" if rois[-1] < rois[0] else "improving"
        q2["verdict"] = (
            "OVERFITTING" if best_step_idx < len(rois) - 1 and rois[-1] < rois[best_step_idx] - 0.1
            else "UNDERFITTING" if rois[-1] > rois[0] and rois[-1] < 0
            else "STABLE"
        )
    else:
        q2["verdict"] = "INSUFFICIENT_DATA"
    logger.info(f"  Q2 verdict: {q2.get('verdict', 'N/A')}")
    report["questions"]["Q2"] = q2

    # ================================================================
    # Q3: Threshold 感度分析
    # ================================================================
    logger.info("=" * 60)
    logger.info("Q3: Threshold 感度分析")
    logger.info("=" * 60)

    q3_results = threshold_sensitivity(
        model, raw_env, max_eval_steps,
        thresholds=[0.30, 0.50, 0.70, 0.80, 0.90],
        normalize_fn=normalize_fn,
    )
    for r in q3_results:
        logger.info(
            f"  thr={r['threshold']:.2f}: trades={r['eval_trades']} "
            f"ROI={r['eval_net_roi']:.4f}% "
            f"gross={r['eval_gross_pnl']:.0f} "
            f"fees={r['eval_total_fees']:.0f} "
            f"active_ratio={r['abs_above_threshold_ratio']:.2%}"
        )

    q3: Dict[str, Any] = {"sensitivity": q3_results}
    # Q3 判定: threshold で ROI が大きく変わるか
    roi_range = max(r["eval_net_roi"] for r in q3_results) - min(r["eval_net_roi"] for r in q3_results)
    best_thr = max(q3_results, key=lambda r: r["eval_net_roi"])
    q3["best_threshold"] = best_thr["threshold"]
    q3["roi_range_pt"] = round(roi_range, 4)
    q3["verdict"] = (
        "THRESHOLD_DEPENDENT" if roi_range > 0.5
        else "THRESHOLD_INSENSITIVE"
    )
    logger.info(f"  Q3 verdict: {q3['verdict']} (ROI range = {roi_range:.4f}pt)")
    report["questions"]["Q3"] = q3

    # ================================================================
    # まとめ
    # ================================================================
    elapsed = time.time() - start_time
    report["elapsed_seconds"] = round(elapsed, 1)

    # E2 方向の推奨
    q1_verdict = q1.get("verdict", "NO_EDGE")
    q2_verdict = q2.get("verdict", "STABLE")
    q3_verdict = q3.get("verdict", "THRESHOLD_INSENSITIVE")

    if q1_verdict == "NO_EDGE":
        e2_direction = "ALGORITHM_CHANGE_OR_SUPERVISED_PRETRAINING"
    elif q1_verdict == "TINY_EDGE" and q3_verdict == "THRESHOLD_DEPENDENT":
        e2_direction = "COST_REDUCTION_AND_EDGE_AMPLIFICATION"
    elif q2_verdict == "OVERFITTING":
        e2_direction = "REGULARIZATION_OR_STEP_REDUCTION"
    else:
        e2_direction = "REWARD_REDESIGN_OR_FEATURE_EXPANSION"

    report["recommendation"] = {
        "e2_direction": e2_direction,
        "q1": q1_verdict,
        "q2": q2_verdict,
        "q3": q3_verdict,
    }
    logger.info("=" * 60)
    logger.info(f"E0 診断完了 ({elapsed:.0f}秒)")
    logger.info(f"推奨 E2 方向: {e2_direction}")
    logger.info("=" * 60)

    # 結果保存
    output_file = OUTPUT_DIR / f"e0_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # numpy 配列を除外してシリアライズ可能にする
    def _clean(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2, ensure_ascii=False)
    logger.info(f"結果保存: {output_file}")

    # クリーンアップ
    del trainer, model
    gc.collect()

    return report


if __name__ == "__main__":
    try:
        run_e0_diagnostic()
    except Exception:
        logger.exception("E0 diagnostic failed")
        sys.exit(1)
