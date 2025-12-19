#!/usr/bin/env python3
"""
Run SAC v454 Regime-Specific Grid Search
Generic tool to find optimal Z/TP/SL parameters for any regime.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Add project root to path (but avoid local shims shadowing installed SB3)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `stable_baselines3` is required for backtests; ensure we import the real
# site-packages version even if a local shim package exists in the repo.
def _is_project_root_path(entry: str) -> bool:
    if not entry:
        try:
            return Path.cwd().resolve() == PROJECT_ROOT
        except Exception:
            return False
    try:
        return Path(entry).resolve() == PROJECT_ROOT
    except Exception:
        return False

sys.path[:] = [p for p in sys.path if not _is_project_root_path(p)]

import torch  # noqa: F401
from stable_baselines3 import SAC
import pandas as pd

sys.path.append(str(PROJECT_ROOT))

from ztb.analysis.market_regime_classifier import RegimeType
from ztb.utils.logging_utils import setup_logging
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

_REGIME_ALIASES: dict[str, str] = {
    "strong_bull": "strong_bull_trend",
    "strong_bear": "strong_bear_trend",
}

ConfigDict = dict[str, object]


def _ensure_dict(parent: ConfigDict, key: str) -> ConfigDict:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    parent[key] = value
    return value


def _ensure_regime_filter(config: ConfigDict) -> ConfigDict:
    env_cfg = _ensure_dict(config, "environment")
    hybrid_cfg = _ensure_dict(env_cfg, "hybrid_config")
    regime_filter = _ensure_dict(hybrid_cfg, "regime_filter")

    if "enabled" not in regime_filter:
        regime_filter["enabled"] = True
    if "mode" not in regime_filter:
        regime_filter["mode"] = "soft"
    if "force_exit" not in regime_filter:
        regime_filter["force_exit"] = True
    if "excluded_regimes" not in regime_filter:
        regime_filter["excluded_regimes"] = []
    if "regime_constraints" not in regime_filter:
        regime_filter["regime_constraints"] = {}
    return regime_filter


def _restrict_to_regime(regime_filter: ConfigDict, regime_name: str) -> None:
    exclusions = [r.value for r in RegimeType if r.value != regime_name]
    if exclusions:
        regime_filter["excluded_regimes"] = exclusions


def _parse_float_list(value: str) -> list[float]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    if not items:
        raise ValueError("Empty grid list")
    return items


def _normalize_regime_name(regime_name: str) -> str:
    normalized = str(regime_name or "").strip()
    if not normalized:
        raise ValueError("--regime must be non-empty")
    return _REGIME_ALIASES.get(normalized, normalized)


def _apply_regime_params(
    config: ConfigDict,
    regime_name: str,
    *,
    entry_zscore_threshold: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    entry_action_source: str | None = None,
) -> None:
    regime_filter = _ensure_regime_filter(config)
    constraints_raw = regime_filter.get("regime_constraints")
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}
        regime_filter["regime_constraints"] = constraints_raw

    if regime_name not in constraints_raw or not isinstance(
        constraints_raw.get(regime_name), dict
    ):
        constraints_raw[regime_name] = {}

    target_regime = constraints_raw[regime_name]

    if entry_zscore_threshold is not None:
        target_regime["entry_zscore_threshold"] = float(entry_zscore_threshold)
    if stop_loss_pct is not None:
        target_regime["stop_loss_pct"] = float(stop_loss_pct)
    if take_profit_pct is not None:
        target_regime["take_profit_pct"] = float(take_profit_pct)
    if entry_action_source is not None:
        target_regime["entry_action_source"] = str(entry_action_source)


def _compute_max_drawdown_pct(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values[1:]:
        if value > peak:
            peak = value
            continue
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return max_drawdown * 100.0

def _run_episode(
    *,
    env: HeavyTradingEnv,
    model: SAC,
    deterministic: bool = True,
) -> dict[str, object]:
    obs, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, done, truncated, _info = env.step(action)

    final_balance = float(getattr(env, "portfolio_value", 0.0))
    initial_balance = float(getattr(env, "initial_portfolio_value", 1.0))
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100.0

    results: dict[str, object] = {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": float(total_return_pct),
    }

    if hasattr(env, "get_statistics"):
        stats = env.get_statistics()
        if isinstance(stats, dict):
            results.update(stats)

    portfolio_values = list(getattr(env, "portfolio_value_history", []) or [])
    results["max_drawdown_pct"] = _compute_max_drawdown_pct(portfolio_values)

    return results

def run_grid_search(
    model_path: str,
    config_path: str,
    data_path: str,
    regime_name: str,
    z_grid: list[float],
    sl_grid: list[float],
    tp_grid: list[float],
    report_path: str,
    deterministic: bool = True,
    restrict_to_regime: bool = False,
) -> int:
    regime_name = _normalize_regime_name(regime_name)

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config: ConfigDict = json.load(f)

    regime_filter = _ensure_regime_filter(config)
    regime_filter["enabled"] = True
    if restrict_to_regime:
        _restrict_to_regime(regime_filter, regime_name)

    # Load data
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True, drop=False)

    # Initialize environment
    env = HeavyTradingEnv(df=df, config=config)
    
    # Load model
    model = SAC.load(model_path, env=env)

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    completed: set[tuple[float, float, float]] = set()
    if report_file.exists():
        try:
            existing = json.loads(report_file.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for row in existing:
                    if not isinstance(row, dict):
                        continue
                    results.append(row)
                    try:
                        key = (
                            round(float(row.get("entry_zscore_threshold")), 6),
                            round(float(row.get("stop_loss_pct")), 6),
                            round(float(row.get("take_profit_pct")), 6),
                        )
                        completed.add(key)
                    except Exception:
                        continue
                if results:
                    logger.info(f"Resuming from {report_path} ({len(results)} completed)")
        except Exception as e:
            logger.warning(f"Failed to load existing report from {report_path}: {e}")

    combos = [
        (z, sl, tp)
        for z in z_grid
        for sl in sl_grid
        for tp in tp_grid
    ]
    
    logger.info(f"Starting Grid Search for regime: {regime_name}")
    logger.info(f"Total combinations: {len(combos)}")
    logger.info(f"Z: {z_grid}")
    logger.info(f"SL: {sl_grid}")
    logger.info(f"TP: {tp_grid}")

    for idx, (z, sl, tp) in enumerate(combos, start=1):
        combo_key = (round(float(z), 6), round(float(sl), 6), round(float(tp), 6))
        if combo_key in completed:
            logger.info(
                f"[{idx:>3}/{len(combos)}] z={z:.2f} sl={sl:.4f} tp={tp:.4f} -> skipped (already computed)"
            )
            continue

        # Apply params
        _apply_regime_params(
            config,
            regime_name,
            entry_zscore_threshold=z,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            entry_action_source="zscore",  # Force Z-score source for grid search
        )
        
        # Update env config in-place
        if isinstance(getattr(env.config, "hybrid_config", None), dict):
            env.config.hybrid_config = config.get("environment", {}).get(
                "hybrid_config"
            )

        metrics = _run_episode(env=env, model=model, deterministic=deterministic)
        
        row = {
            "regime": regime_name,
            "entry_zscore_threshold": z,
            "stop_loss_pct": sl,
            "take_profit_pct": tp,
            "total_return_pct": metrics.get("total_return_pct"),
            "final_balance": metrics.get("final_balance"),
            "total_trades": metrics.get("total_trades"),
            "win_rate": metrics.get("win_rate"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "portfolio_volatility": metrics.get("portfolio_volatility"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        }
        results.append(row)
        completed.add(combo_key)
        logger.info(
            f"[{idx:>3}/{len(combos)}] z={z:.2f} sl={sl:.4f} tp={tp:.4f} -> return={row['total_return_pct']:.2f}% trades={row.get('total_trades')}"
        )

        # Save intermediate results (allows resume if interrupted)
        results_sorted = sorted(
            results,
            key=lambda r: float(r.get("total_return_pct") or -1e18),
            reverse=True,
        )
        report_file.write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")

    # Final sort + save
    results_sorted = sorted(
        results, key=lambda r: float(r.get("total_return_pct") or -1e18), reverse=True
    )
    report_file.write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")
    logger.info(f"Grid results saved to {report_path}")
    
    # Print Top 3
    logger.info("Top 3 Results:")
    for i, r in enumerate(results_sorted[:3]):
        logger.info(
            f"{i+1}. Return: {r['total_return_pct']:.2f}% | Trades: {r['total_trades']} | Win: {r.get('win_rate', 0.0):.1%} | Sharpe: {r.get('sharpe_ratio', 0.0):.3f} | DD: {r.get('max_drawdown_pct', 0.0):.2f}% | Z={r['entry_zscore_threshold']} TP={r['take_profit_pct']} SL={r['stop_loss_pct']}"
        )

    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAC v454 Regime Grid Search")
    parser.add_argument(
        "--regime",
        required=True,
        help="Target regime name (e.g., strong_bull, strong_bear, strong_bull_trend, strong_bear_trend)",
    )
    parser.add_argument("--z-grid", type=_parse_float_list, required=True, help="Comma-separated z thresholds")
    parser.add_argument("--sl-grid", type=_parse_float_list, required=True, help="Comma-separated stop-loss pcts")
    parser.add_argument("--tp-grid", type=_parse_float_list, required=True, help="Comma-separated take-profit pcts")
    parser.add_argument("--report-path", default="backtest_results/regime_grid_results.json")
    parser.add_argument("--model-path", default="models/sac_v454_inverse_confidence.zip")
    parser.add_argument("--config-path", default="config/v454/sac_v454_config.json")
    parser.add_argument("--data-path", default="data/btc_jpy_1m_v454.csv")
    parser.add_argument(
        "--restrict-to-regime",
        action="store_true",
        help="Exclude all non-target regimes during the grid run",
    )
    
    args = parser.parse_args()
    
    return run_grid_search(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        regime_name=args.regime,
        z_grid=args.z_grid,
        sl_grid=args.sl_grid,
        tp_grid=args.tp_grid,
        report_path=args.report_path,
        restrict_to_regime=args.restrict_to_regime,
    )

if __name__ == "__main__":
    raise SystemExit(main())
