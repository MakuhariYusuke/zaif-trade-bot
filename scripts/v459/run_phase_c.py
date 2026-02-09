#!/usr/bin/env python3
"""
Phase C 統一実験ランナー — Gate 2 KPI全収集版

0番: Gate 2基準 (ROI>5%, PF>1.20, Sharpe>1.0, MaxDD<15%, WinRate>35%)
66番: 計測基盤が一度も測定していなかった → 本スクリプトで解消
91番: γ=0.80最優先、コスト負け(H2)対策、v451 Golden Era回帰
100# §12: C0計測統一 + C1コスト圧縮を統合実行

Usage:
  # 単一実験
  python scripts/v459/run_phase_c.py --single-run --experiment gamma_080 --seed 42

  # バッチ実行（C0+C1統合）
  python scripts/v459/run_phase_c.py --batch c0_c1
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.metrics.metrics import (
    calculate_all_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_trainer_env_metrics,
)
from ztb.utils.env_metrics import resolve_env, unwrap_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 定数
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "results" / "phase_c"

INITIAL_BALANCE = 100000.0
TOTAL_TIMESTEPS = 50000

# ============================================================================
# SAC基本設定（P1-1ベース）
# ============================================================================

SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
}

# P1-1ベース報酬設定（use_simple_reward=True, ペナルティ全無効）
REWARD_BASE = {
    "use_simple_reward": True,
    "balance_penalty": 0.0,
    "balance_penalty_tolerance": 1.0,
    "position_penalty_scale": 0.0,
    "position_penalty_exponent": 1.0,
    "inventory_penalty_scale": 0.0,
    "trade_frequency_penalty": 0.0,
    "trade_cooldown_penalty": 0.0,
    "consecutive_trade_penalty": 0.0,
    "hold_penalty_multiplier": 1.0,
    "volatility_penalty_scale": 0.0,
    "consistency_penalty": 0.0,
    "redundant_trade_penalty": 0.0,
    "profit_weight": 1.0,
    "reward_scale": 100.0,
    "confidence_penalty_factor": 0.0,
    "balance_shaping_enabled": False,
    "action_entropy_shaping_enabled": False,
    "long_position_reward_multiplier": 1.0,
    "short_position_reward_multiplier": 1.0,
    "long_position_penalty_multiplier": 1.0,
    "short_position_penalty_multiplier": 1.0,
}


# ============================================================================
# 実験定義 — C0+C1+91# H1統合
# ============================================================================

def get_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Phase C 全実験定義。
    
    命名規則: {phase}_{variable}_{value}
    91#優先順: H1(gamma) ⭐⭐⭐ → H2(cost/threshold) ⭐⭐⭐
    """
    configs = {}

    # --- C0: P1-1再現 (Gate 2 KPI計測付き、ベースライン) ---
    configs["c0_baseline_p1"] = {
        "description": "P1-1再現 + Gate2 KPI全計測",
        "sac_overrides": {},
        "reward_overrides": {},
        "env_overrides": {},
    }

    # --- C1-H1: γ感度 (91# 最優先仮説) ---
    for gamma in [0.80, 0.90, 0.95]:
        gamma_key = f"{gamma:.2f}".replace('.', '')
        configs[f"c1_gamma_{gamma_key}"] = {
            "description": f"γ={gamma} (91# H1: v451={0.80})",
            "sac_overrides": {"gamma": gamma},
            "reward_overrides": {},
            "env_overrides": {},
        }

    # --- C1-H2: threshold感度 (取引コスト削減) ---
    for threshold in [0.50, 0.60, 0.70]:
        configs[f"c1_threshold_{int(threshold*100)}"] = {
            "description": f"threshold={threshold} (H2: 過剰取引抑制)",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {"continuous_to_discrete_threshold": threshold},
        }

    # --- C1-H1+H2: γ=0.80 + best threshold (組合せ) ---
    for threshold in [0.50, 0.60, 0.70]:
        configs[f"c1_gamma080_threshold_{int(threshold*100)}"] = {
            "description": f"γ=0.80 + threshold={threshold}",
            "sac_overrides": {"gamma": 0.80},
            "reward_overrides": {},
            "env_overrides": {"continuous_to_discrete_threshold": threshold},
        }

    # --- C1: min_holding_period (現行デフォルト=3) ---
    for mhp in [5, 10, 15]:
        configs[f"c1_holding_{mhp}"] = {
            "description": f"min_holding_period={mhp} (現行=3)",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {"min_holding_period": mhp},
        }

    # --- C1: v451復元 (91# Golden Era) ---
    configs["c1_v451_golden"] = {
        "description": "v451 Golden Era: γ=0.80, scale=1.0, loss_mult=1.2",
        "sac_overrides": {"gamma": 0.80},
        "reward_overrides": {
            "reward_scale": 1.0,  # P1-1は100.0
            "custom_reward_params": {"type": "pnl_centered"},  # V457RewardCalculator
        },
        "env_overrides": {},
    }

    # --- C1': 最小因果分離 (103# §3.2) ---
    # reward_scale {100, 1000} × ent_coef {auto, 0.01}
    configs["c1p_scale1000"] = {
        "description": "reward_scale=1000 (10× base), ent_coef=auto",
        "sac_overrides": {},
        "reward_overrides": {"reward_scale": 1000.0},
        "env_overrides": {},
    }
    configs["c1p_ent001"] = {
        "description": "ent_coef=0.01 (exploitation促進), reward_scale=100",
        "sac_overrides": {"ent_coef": 0.01},
        "reward_overrides": {},
        "env_overrides": {},
    }
    configs["c1p_scale1000_ent001"] = {
        "description": "reward_scale=1000 + ent_coef=0.01 (combo)",
        "sac_overrides": {"ent_coef": 0.01},
        "reward_overrides": {"reward_scale": 1000.0},
        "env_overrides": {},
    }

    return configs


# ============================================================================
# バッチ定義
# ============================================================================

BATCHES = {
    # C0+C1統合: seed=42でスクリーニング → 最良条件を4seedsで展開
    "c0_c1": [
        "c0_baseline_p1",
        "c1_gamma_080", "c1_gamma_090", "c1_gamma_095",
        "c1_threshold_50", "c1_threshold_60", "c1_threshold_70",
        "c1_gamma080_threshold_50", "c1_gamma080_threshold_60", "c1_gamma080_threshold_70",
        "c1_holding_5", "c1_holding_10", "c1_holding_15",
        "c1_v451_golden",
    ],
    # C0': 評価経路整合修正 + baseline再検証
    "c0_prime": [
        "c0_baseline_p1",
    ],
    # C1': 最小因果分離 (103# §3.2)
    # reward_scale 100 vs 1000 × ent_coef auto vs 0.01
    "c1_prime": [
        "c0_baseline_p1",          # 基準 (scale=100, ent=auto)
        "c1p_scale1000",           # scale=1000, ent=auto
        "c1p_ent001",              # scale=100, ent=0.01
        "c1p_scale1000_ent001",    # scale=1000, ent=0.01
    ],
    # screening後のフルseed展開（実行時に動的指定）
    "full_seeds": [],
}


# ============================================================================
# 実験実行
# ============================================================================

def build_config(
    experiment_name: str,
    seed: int,
    exp_def: Dict[str, Any],
) -> Dict[str, Any]:
    """実験設定dict を構築"""
    sac_params = SAC_DEFAULT.copy()
    sac_params.update(exp_def.get("sac_overrides", {}))

    reward_params = REWARD_BASE.copy()
    reward_params.update(exp_def.get("reward_overrides", {}))

    env_overrides = exp_def.get("env_overrides", {})

    env_config = {
        "use_continuous_actions": True,
        "action_space_type": "continuous",
        "initial_portfolio_value": INITIAL_BALANCE,
        "transaction_cost": 0.001,
        "reward_settings": reward_params,
        "feature_set": "v451",  # MTF無効化（OOM回避）。v451 parquetに適合
        "correlation_reduction": False,  # 相関削減スキップ（O(m⁴)→O(1)高速化）
    }
    env_config.update(env_overrides)

    config = {
        "experiment_name": experiment_name,
        "training": {
            "algorithm": "SAC",
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": 5000,
            "n_eval_episodes": 3,
            "log_interval": 100,
            "seed": seed,
            "sac_hyperparameters": sac_params,
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60,
            },
            "environment": env_config,
            "walk_forward": {"enabled": False},
        },
        "reward": reward_params,
    }
    return config


def _find_vec_normalize(env: Any) -> Any:
    """VecEnvラッパースタックからVecNormalizeを探す。"""
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        return None
    current = env
    for _ in range(10):
        if isinstance(current, VecNormalize):
            return current
        if hasattr(current, "venv"):
            current = current.venv
        else:
            break
    return None


def _reset_risk_controllers(env: Any) -> None:
    """DrawdownController等のリスク管理状態をリセット。

    学習中にemergency_stopがラッチされるとeval時に全取引がブロックされる。
    env.reset()はposition_managerのリスク管理をリセットしないため、
    eval前に明示的にリセットする必要がある。
    """
    # position_manager.risk_manager.drawdown_controller
    pm = getattr(env, "position_manager", None)
    if pm is not None:
        rm = getattr(pm, "risk_manager", None)
        if rm is not None and hasattr(rm, "reset"):
            rm.reset()
            logger.info("Risk manager reset for eval (emergency_stop cleared)")
            return
    # フォールバック: drawdown_controllerを直接探す
    for attr_path in [
        ("risk_manager", "drawdown_controller"),
        ("position_manager", "risk_manager", "drawdown_controller"),
    ]:
        obj = env
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "reset"):
            obj.reset()
            logger.info(f"DrawdownController reset via {'.'.join(attr_path)}")
            return


def _run_deterministic_eval(
    model: Any,
    raw_env: Any,
    max_eval_steps: int,
    threshold: float,
    normalize_fn: Any = None,
    label: str = "eval",
) -> Dict[str, Any]:
    """1エピソードのdeterministic評価を実行。

    Args:
        normalize_fn: obs→正規化obs変換関数。Noneなら生obs使用。
    """
    # DrawdownControllerのemergency_stopラッチを解除
    # (学習中の15%DDでラッチされ、reset()で解除されない場合の安全策)
    _reset_risk_controllers(raw_env)

    obs_raw, _ = raw_env.reset(seed=42)
    obs = normalize_fn(obs_raw.copy()) if normalize_fn else obs_raw
    done = False
    balances = [float(raw_env.portfolio_value)]
    actions: list[float] = []
    step_count = 0

    while not done and step_count < max_eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_scalar = float(action.flatten()[0]) if hasattr(action, "flatten") else float(action)
        actions.append(action_scalar)

        obs_raw, _reward, terminated, truncated, _info = raw_env.step(action)
        done = terminated or truncated
        obs = normalize_fn(obs_raw.copy()) if normalize_fn else obs_raw
        balances.append(float(raw_env.portfolio_value))
        step_count += 1

    balances_arr = np.array(balances, dtype=np.float64)
    result = compute_gate2_metrics_from_balances(balances_arr)

    result["eval_steps"] = step_count
    result["eval_trades"] = int(raw_env.total_trades)
    result["eval_gross_pnl"] = float(getattr(raw_env, "gross_pnl", 0.0))
    result["eval_net_roi"] = float(
        (balances_arr[-1] - balances_arr[0]) / balances_arr[0] * 100
    )
    result["eval_total_fees"] = float(getattr(raw_env, "total_fees", 0.0))
    result["eval_buy_count"] = int(getattr(raw_env, "buy_count", 0))
    result["eval_sell_count"] = int(getattr(raw_env, "sell_count", 0))

    if actions:
        arr = np.array(actions)
        result["action_stats"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "abs_above_threshold": float(np.mean(np.abs(arr) > threshold)),
        }

    result["eval_method"] = label
    return result


def _deterministic_eval_gate2(
    trainer: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """学習後にdeterministic=Trueで1エピソード評価し、Gate2 KPIを計算。

    103# C0' 修正: VecNormalize正規化ミスマッチの解消。
    学習時は VecNormalize で obs が正規化されるが、旧コードは
    unwrap_env() で生 obs をモデルに入力していた。
    → Eval-A: VecNormalize.normalize_obs() で手動正規化して raw env 経由で評価
    → Eval-B: 生 obs のまま評価（旧方式、比較用）
    """
    try:
        # モデル取得
        model = None
        if hasattr(trainer, "model") and trainer.model is not None:
            model = trainer.model
        elif hasattr(trainer, "algorithm_trainer"):
            at = trainer.algorithm_trainer
            if hasattr(at, "model") and at.model is not None:
                model = at.model

        if model is None:
            return {"gate2_available": False, "gate2_error": "model not found"}

        # 学習envを再利用（MTF特徴量再計算を回避）
        from ztb.utils.env_metrics import resolve_env, unwrap_env

        vec_env = resolve_env(trainer)
        if vec_env is None and hasattr(model, "get_env"):
            vec_env = model.get_env()

        raw_env = unwrap_env(vec_env) if vec_env is not None else None

        if raw_env is None:
            return {"gate2_available": False, "gate2_error": "env not found for eval"}

        # VecNormalize検出 → 正規化関数を構築
        vec_normalize = _find_vec_normalize(vec_env)
        if vec_normalize is not None:
            vec_normalize.training = False  # 統計更新を停止
            logger.info(
                "VecNormalize detected — running Eval-A (normalized) + Eval-B (raw)"
            )
        else:
            logger.warning("VecNormalize not found — running Eval-B (raw) only")

        # 評価パラメータ
        eval_timesteps = config.get("training", {}).get(
            "total_timesteps", TOTAL_TIMESTEPS
        )
        max_eval_steps = min(
            getattr(raw_env, "n_steps", eval_timesteps),
            eval_timesteps,
        )
        threshold = config.get("training", {}).get("environment", {}).get(
            "continuous_to_discrete_threshold", 0.3333
        )

        # ===== Eval-A: VecNormalize 正規化 obs =====
        normalize_fn = vec_normalize.normalize_obs if vec_normalize else None
        gate2 = _run_deterministic_eval(
            model, raw_env, max_eval_steps, threshold,
            normalize_fn=normalize_fn,
            label="normalized" if vec_normalize else "raw",
        )

        # ===== Eval-B: 生 obs (旧方式、比較用) =====
        if vec_normalize is not None:
            eval_b = _run_deterministic_eval(
                model, raw_env, max_eval_steps, threshold,
                normalize_fn=None,
                label="raw",
            )
            gate2["eval_b_comparison"] = {
                "eval_trades": eval_b["eval_trades"],
                "eval_net_roi": eval_b["eval_net_roi"],
                "eval_buy_count": eval_b.get("eval_buy_count", 0),
                "eval_sell_count": eval_b.get("eval_sell_count", 0),
                "action_stats": eval_b.get("action_stats", {}),
            }
            logger.info(
                f"Eval-A trades={gate2['eval_trades']} | "
                f"Eval-B trades={eval_b['eval_trades']} "
                f"(VecNormalize mismatch check)"
            )

        # VecNormalize 状態を復元
        if vec_normalize is not None:
            vec_normalize.training = True

        return gate2

    except Exception as e:
        logger.error(f"Deterministic eval failed: {e}", exc_info=True)
        return {"gate2_available": False, "gate2_error": str(e)}


def compute_gate2_metrics_from_balances(balances: np.ndarray) -> Dict[str, Any]:
    """balance配列からGate 2 KPIを計算。
    
    0番 §5.2 基準:
    - Net ROI > 5%
    - PF > 1.20
    - Sharpe > 1.0
    - MaxDD < 15%
    - WinRate > 35%
    """
    if balances is None or len(balances) < 10:
        return {"gate2_available": False, "gate2_error": "insufficient balance data"}
    
    returns = np.diff(balances) / np.maximum(balances[:-1], 1e-10)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    gate2: Dict[str, Any] = {"gate2_available": True}
    
    try:
        gate2["sharpe"] = float(sharpe_ratio(returns, period_per_year=525600))
    except Exception:
        gate2["sharpe"] = 0.0
    
    try:
        gate2["max_drawdown"] = float(max_drawdown(balances))
    except Exception:
        gate2["max_drawdown"] = 0.0
    
    try:
        gate2["profit_factor"] = float(profit_factor(returns))
    except Exception:
        gate2["profit_factor"] = 0.0
    
    try:
        gate2["win_rate"] = float(win_rate(returns))
    except Exception:
        gate2["win_rate"] = 0.0
    
    gate2["mtm_roi"] = float((balances[-1] - balances[0]) / balances[0] * 100)
    gate2["balance_samples"] = len(balances)
    gate2["final_balance"] = float(balances[-1])
    gate2["initial_balance_sampled"] = float(balances[0])
    
    gate2["gate2_pass"] = (
        gate2["mtm_roi"] > 5.0
        and gate2["profit_factor"] > 1.20
        and gate2["sharpe"] > 1.0
        and abs(gate2["max_drawdown"]) < 15.0
        and gate2["win_rate"] > 0.35
    )
    
    return gate2


def compute_gate2_metrics(env: Any) -> Dict[str, Any]:
    """Gate 2 KPI を環境のportfolio_value_historyから計算。
    
    0番 §5.2 基準:
    - Net ROI > 5%
    - PF > 1.20
    - Sharpe > 1.0
    - MaxDD < 15%
    - WinRate > 35%
    
    Args:
        env: unwrap済みのHeavyTradingEnv（またはNone）
    """
    if env is None:
        return {"gate2_available": False, "gate2_error": "env is None"}
    
    # portfolio_value_history は deque(maxlen=512) → 全ステップ不足
    # statistics_calculator.portfolio_value_history は deque(maxlen=None) → 全ステップあり
    balances: Optional[np.ndarray] = None
    
    # 優先1: statistics_calculator (全ステップ保持, maxlen=None)
    sc = getattr(env, "statistics_calculator", None)
    if sc is not None:
        pvh = getattr(sc, "portfolio_value_history", None)
        if pvh is not None and len(pvh) > 10:
            balances = np.array(pvh, dtype=np.float64)
    
    # フォールバック: core.py の portfolio_value_history (最後512ステップ)
    if balances is None:
        pvh = getattr(env, "portfolio_value_history", None)
        if pvh is not None and len(pvh) > 10:
            balances = np.array(pvh, dtype=np.float64)
    
    if balances is None or len(balances) < 10:
        return {"gate2_available": False, "gate2_error": "insufficient balance history"}
    
    # balance → step returns
    returns = np.diff(balances) / np.maximum(balances[:-1], 1e-10)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gate 2 KPI計算
    gate2: Dict[str, Any] = {"gate2_available": True}
    
    try:
        gate2["sharpe"] = float(sharpe_ratio(returns, period_per_year=525600))  # 1分足→年換算
    except Exception:
        gate2["sharpe"] = 0.0
    
    try:
        gate2["max_drawdown"] = float(max_drawdown(balances))
    except Exception:
        gate2["max_drawdown"] = 0.0
    
    try:
        gate2["profit_factor"] = float(profit_factor(returns))
    except Exception:
        gate2["profit_factor"] = 0.0
    
    try:
        gate2["win_rate"] = float(win_rate(returns))
    except Exception:
        gate2["win_rate"] = 0.0
    
    # ROI (mark-to-market)
    gate2["mtm_roi"] = float((balances[-1] - balances[0]) / balances[0] * 100)
    gate2["balance_samples"] = len(balances)
    gate2["final_balance"] = float(balances[-1])
    gate2["initial_balance_sampled"] = float(balances[0])
    
    # Gate 2 判定
    gate2["gate2_pass"] = (
        gate2["mtm_roi"] > 5.0
        and gate2["profit_factor"] > 1.20
        and gate2["sharpe"] > 1.0
        and abs(gate2["max_drawdown"]) < 15.0
        and gate2["win_rate"] > 0.35
    )
    
    return gate2


def run_single_experiment(
    experiment_name: str,
    seed: int,
    exp_def: Dict[str, Any],
) -> Dict[str, Any]:
    """1回の実験: 学習→Gate2 KPI収集→結果返却"""
    start_time = time.time()
    
    logger.warning(f"\n{'='*60}")
    logger.warning(f"実験: {experiment_name} (seed={seed})")
    logger.warning(f"  {exp_def.get('description', '')}")
    logger.warning(f"{'='*60}")
    
    config = build_config(experiment_name, seed, exp_def)
    trainer = None
    
    try:
        trainer = SACTrainer(config=config, logger=logger)
        _result = trainer.train()
        elapsed = time.time() - start_time
        
        # 基本メトリクス
        result: Dict[str, Any] = {
            "experiment": experiment_name,
            "seed": seed,
            "description": exp_def.get("description", ""),
            "elapsed_seconds": round(elapsed, 1),
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "gamma": config["training"]["sac_hyperparameters"]["gamma"],
                "reward_scale": config["reward"].get("reward_scale", 100.0),
                "threshold": config["training"]["environment"].get(
                    "continuous_to_discrete_threshold", 0.3333
                ),
                "min_holding_period": config["training"]["environment"].get(
                    "min_holding_period", "default(3)"
                ),
                "transaction_cost": config["training"]["environment"]["transaction_cost"],
            },
        }
        
        # P1互換メトリクス（学習時の最終エピソード値）
        metrics = extract_trainer_env_metrics(trainer, include_optional=True)
        if metrics:
            result.update(metrics)
            roi = compute_balance_roi(metrics)
            if roi is not None:
                result["net_roi"] = roi
            if "gross_pnl" in metrics and metrics.get("initial_balance", 0) > 0:
                result["gross_roi"] = metrics["gross_pnl"] / metrics["initial_balance"] * 100
        
        # ★ Gate 2 KPI: 学習後にdeterministic評価を実行
        # VecEnvリセットで学習中の履歴はクリアされるため、
        # 別途1エピソードをdeterministic=Trueで走らせて計測
        gate2 = _deterministic_eval_gate2(trainer, config)
        result["gate2"] = gate2
        
        # サマリログ
        logger.warning(f"  完了: {elapsed:.0f}秒")
        logger.warning(f"  [Training] Net ROI: {result.get('net_roi', 'N/A')}")
        logger.warning(f"  [Training] Trades: {result.get('total_trades', 'N/A')}")
        logger.warning(f"  [Training] Gross PnL: {result.get('gross_pnl', 'N/A')}")
        logger.warning(f"  [Training] Fees: {result.get('total_fees', 'N/A')}")
        if gate2.get("gate2_available"):
            a_trades = gate2.get("eval_trades", "?")
            a_stats = gate2.get("action_stats", {})
            a_above = a_stats.get("abs_above_threshold", 0)
            logger.warning(
                f"  [Gate2 Eval-A] trades={a_trades} "
                f"action_mean={a_stats.get('mean', 0):.4f} "
                f"action_std={a_stats.get('std', 0):.4f} "
                f"|a|>thr={a_above:.1%}"
            )
            logger.warning(
                f"  [Gate2 Eval-A] PF={gate2['profit_factor']:.3f} "
                f"Sharpe={gate2['sharpe']:.3f} "
                f"MaxDD={gate2['max_drawdown']:.2f}% "
                f"WinRate={gate2['win_rate']:.1%} "
                f"{'PASS' if gate2['gate2_pass'] else 'FAIL'}"
            )
            eval_b = gate2.get("eval_b_comparison", {})
            if eval_b:
                b_stats = eval_b.get("action_stats", {})
                logger.warning(
                    f"  [Gate2 Eval-B] trades={eval_b.get('eval_trades', '?')} "
                    f"action_mean={b_stats.get('mean', 0):.4f} "
                    f"action_std={b_stats.get('std', 0):.4f} "
                    f"|a|>thr={b_stats.get('abs_above_threshold', 0):.1%} "
                    f"(raw obs, 旧方式)"
                )
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"実験失敗: {experiment_name} - {e}", exc_info=True)
        return {
            "experiment": experiment_name,
            "seed": seed,
            "elapsed_seconds": round(elapsed, 1),
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        if trainer is not None:
            try:
                trainer.cleanup_training_environment()
            except Exception:
                pass
            try:
                if hasattr(trainer, 'model') and trainer.model is not None:
                    del trainer.model
            except Exception:
                pass
            del trainer
        # メモリリーク防止: GC + torch cache clear
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_batch(batch_name: str, seeds: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """バッチ実行"""
    if seeds is None:
        seeds = [42]  # スクリーニングはseed=42のみ
    
    experiments = get_experiment_configs()
    batch_exps = BATCHES.get(batch_name, [])
    
    if not batch_exps:
        logger.error(f"バッチ '{batch_name}' が見つかりません")
        return []
    
    all_results: List[Dict[str, Any]] = []
    total = len(batch_exps) * len(seeds)
    
    logger.warning(f"\n{'='*70}")
    logger.warning(f"Phase C バッチ: {batch_name}")
    logger.warning(f"  実験数: {len(batch_exps)} × {len(seeds)} seeds = {total} runs")
    logger.warning(f"{'='*70}")
    
    for i, exp_name in enumerate(batch_exps, 1):
        if exp_name not in experiments:
            logger.warning(f"スキップ: {exp_name} (定義なし)")
            continue
        exp_def = experiments[exp_name]
        
        for seed in seeds:
            logger.warning(f"\n[{i}/{len(batch_exps)}] {exp_name} seed={seed}")
            result = run_single_experiment(exp_name, seed, exp_def)
            all_results.append(result)
    
    return all_results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """結果サマリテーブルを出力"""
    logger.warning(f"\n{'='*120}")
    logger.warning("Phase C RESULTS SUMMARY")
    logger.warning(f"{'='*120}")
    
    header = (
        f"{'Experiment':<35} {'γ':>5} {'Thr':>5} {'ROI%':>8} "
        f"{'GrossPnL':>10} {'Fees':>8} {'Trades':>7} "
        f"{'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'G2':>4}"
    )
    logger.warning(header)
    logger.warning("-" * 120)
    
    for r in results:
        if not r.get("success"):
            logger.warning(f"{r['experiment']:<35} FAILED: {r.get('error', '?')[:60]}")
            continue
        
        g2 = r.get("gate2", {})
        cfg = r.get("config", {})
        
        line = (
            f"{r['experiment']:<35} "
            f"{cfg.get('gamma', '?'):>5} "
            f"{cfg.get('threshold', 0.33):>5.2f} "
            f"{r.get('net_roi', 0):>7.2f}% "
            f"{r.get('gross_pnl', 0):>+10.0f} "
            f"{r.get('total_fees', 0):>8.0f} "
            f"{r.get('total_trades', 0):>7} "
            f"{g2.get('profit_factor', 0):>6.3f} "
            f"{g2.get('sharpe', 0):>7.3f} "
            f"{g2.get('max_drawdown', 0):>6.2f}% "
            f"{g2.get('win_rate', 0)*100:>5.1f}% "
            f"{'OK' if g2.get('gate2_pass') else 'NG':>4}"
        )
        logger.warning(line)
    
    # Gate 2 基準リマインダ
    logger.warning(f"\n{'─'*60}")
    logger.warning("Gate 2 基準 (0番§5.2):")
    logger.warning("  ROI>5% | PF>1.20 | Sharpe>1.0 | MaxDD<15% | WinRate>35%")
    logger.warning(f"{'─'*60}")


def save_results(results: List[Dict[str, Any]], batch_name: str) -> Path:
    """結果をJSON保存"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{batch_name}_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "batch": batch_name,
            "timestamp": timestamp,
            "results": results,
            "gate2_criteria": {
                "roi": "> 5%",
                "profit_factor": "> 1.20",
                "sharpe": "> 1.0",
                "max_drawdown": "< 15%",
                "win_rate": "> 35%",
            },
        }, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


# ============================================================================
# メイン
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase C 統一実験ランナー")
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--experiment", type=str, default="c0_baseline_p1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42",
                       help="カンマ区切りseed例: 42,123,456,789")
    args = parser.parse_args()
    
    if args.single_run:
        experiments = get_experiment_configs()
        if args.experiment not in experiments:
            logger.error(f"実験 '{args.experiment}' が見つかりません")
            logger.error(f"利用可能: {list(experiments.keys())}")
            sys.exit(1)
        
        result = run_single_experiment(
            args.experiment, args.seed, experiments[args.experiment]
        )
        # stdout最終行にJSON出力（subprocess対応）
        print(json.dumps(result, ensure_ascii=False, default=str))
    
    elif args.batch:
        seeds = [int(s) for s in args.seeds.split(",")]
        results = run_batch(args.batch, seeds)
        print_summary_table(results)
        filepath = save_results(results, args.batch)
        logger.warning(f"\n結果保存: {filepath}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
