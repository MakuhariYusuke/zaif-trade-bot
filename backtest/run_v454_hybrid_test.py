#!/usr/bin/env python3
"""
Run SAC v454 Hybrid Strategy Backtest
Incorporating v453 success factors (Regime Filters, Threshold Optimization)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from ztb.utils.file_utils import get_project_root

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Add project root to path (but avoid local shims shadowing installed SB3)
PROJECT_ROOT = get_project_root()

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

from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from ztb.utils.data_utils import load_csv_data
from ztb.utils.logging_utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def _slice_df(df: pd.DataFrame, start: int = 0, end: int | None = None) -> pd.DataFrame:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is not None and end <= start:
        raise ValueError("--end must be > --start")
    return df.iloc[start:end] if end is not None else df.iloc[start:]


def _apply_high_vol_ranging_params(
    config: dict[str, Any],
    *,
    entry_zscore_threshold: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> None:
    env_cfg = config.get("environment", {})
    hybrid_cfg = env_cfg.get("hybrid_config", {})
    regime_filter = hybrid_cfg.get("regime_filter", {}) if isinstance(hybrid_cfg, dict) else {}
    constraints = regime_filter.get("regime_constraints", {}) if isinstance(regime_filter, dict) else {}
    hvr = constraints.get("high_volatility_ranging") if isinstance(constraints, dict) else None

    if not isinstance(hvr, dict):
        raise KeyError(
            "Missing environment.hybrid_config.regime_filter.regime_constraints.high_volatility_ranging"
        )

    if entry_zscore_threshold is not None:
        hvr["entry_zscore_threshold"] = float(entry_zscore_threshold)
    if stop_loss_pct is not None:
        hvr["stop_loss_pct"] = float(stop_loss_pct)
    if take_profit_pct is not None:
        hvr["take_profit_pct"] = float(take_profit_pct)


def _run_episode(
    *,
    env: HeavyTradingEnv,
    model: SAC,
    deterministic: bool = True,
    collect_regime_diagnostics: bool = False,
) -> dict[str, Any]:
    obs, _ = env.reset()
    done = False
    truncated = False

    # --- Optional regime diagnostics ---
    step_counts_by_regime: dict[str, int] = defaultdict(int)
    trade_actions_by_regime: dict[str, int] = defaultdict(int)
    buy_actions_by_regime: dict[str, int] = defaultdict(int)
    sell_actions_by_regime: dict[str, int] = defaultdict(int)
    entry_count_by_regime: dict[str, int] = defaultdict(int)
    entry_sizes_by_regime: dict[str, list[float]] = defaultdict(list)
    prev_position = float(getattr(env, "position", 0.0))

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, done, truncated, info = env.step(action)

        if collect_regime_diagnostics:
            regime_val = info.get("market_regime")
            if hasattr(regime_val, "value"):
                regime_key = str(regime_val.value)
            elif regime_val is None:
                regime_key = "unknown"
            else:
                regime_key = str(regime_val)

            step_counts_by_regime[regime_key] += 1

            effective_action = info.get("effective_action")
            if effective_action is not None and int(effective_action) != 0:
                trade_actions_by_regime[regime_key] += 1
                if int(effective_action) == 1:
                    buy_actions_by_regime[regime_key] += 1
                elif int(effective_action) in (-1, 2):
                    sell_actions_by_regime[regime_key] += 1

            new_position = float(getattr(env, "position", 0.0))
            if (prev_position == 0.0 and new_position != 0.0) or (
                prev_position * new_position < 0.0 and new_position != 0.0
            ):
                entry_count_by_regime[regime_key] += 1
                entry_sizes_by_regime[regime_key].append(abs(new_position))
            prev_position = new_position

    final_balance = float(getattr(env, "portfolio_value", 0.0))
    initial_balance = float(getattr(env, "initial_portfolio_value", 1.0))
    total_return = (final_balance - initial_balance) / initial_balance * 100.0

    results: dict[str, Any] = {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": float(total_return),
    }

    if hasattr(env, "get_statistics"):
        stats = env.get_statistics()
        results.update(stats or {})

    if collect_regime_diagnostics and hasattr(env, "get_statistics"):
        def _mean(values: list[float]) -> float:
            return float(fmean(values)) if values else 0.0

        results["regime_trade_actions"] = dict(trade_actions_by_regime)
        results["regime_buy_actions"] = dict(buy_actions_by_regime)
        results["regime_sell_actions"] = dict(sell_actions_by_regime)
        results["regime_step_counts"] = dict(step_counts_by_regime)
        results["regime_entry_counts"] = dict(entry_count_by_regime)
        results["regime_entry_size_mean"] = {k: _mean(v) for k, v in entry_sizes_by_regime.items()}

    return results


def run_backtest(
    *,
    model_path: str = "models/sac_v454_inverse_confidence.zip",
    config_path: str = "config/v454/sac_v454_config.json",
    data_path: str = "data/btc_jpy_1m_v454.csv",
    start: int = 0,
    end: int | None = None,
    entry_zscore_threshold: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    grid: bool = False,
    z_grid: Iterable[float] | None = None,
    sl_grid: Iterable[float] | None = None,
    tp_grid: Iterable[float] | None = None,
    deterministic: bool = True,
    report_path: str = "backtest_results/v454_hybrid_test_results.json",
) -> int:

    logger.info(f"🚀 Starting v454 Hybrid Strategy Backtest")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config: {config_path}")

    # Load Config
    try:
        unified_config = UnifiedConfig.from_file(config_path)
        config = unified_config.to_dict()
        logger.info("✅ Config loaded successfully")

        if any(v is not None for v in (entry_zscore_threshold, stop_loss_pct, take_profit_pct)):
            _apply_high_vol_ranging_params(
                config,
                entry_zscore_threshold=entry_zscore_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
        
        # Verify Hybrid Config is present
        env_config = config.get("environment", {})
        hybrid_config = env_config.get("hybrid_config", {})
        adaptive_mode = env_config.get("adaptive_threshold_mode", False)
        regime_filter = hybrid_config.get("regime_filter", {}) if isinstance(hybrid_config, dict) else {}
        
        logger.info(f"Adaptive Threshold Mode: {adaptive_mode}")
        logger.info(f"Hybrid Config Enabled: {hybrid_config.get('enabled', False)}")
        if hybrid_config.get('enabled'):
             logger.info(f"Regime Filter: {hybrid_config.get('regime_filter', {}).get('enabled')}")
             logger.info(f"Regime Filter Mode: {regime_filter.get('mode', 'hard')}")
             logger.info(f"Excluded Regimes: {hybrid_config.get('regime_filter', {}).get('excluded_regimes')}")
             constraints = regime_filter.get("regime_constraints", {}) if isinstance(regime_filter, dict) else {}
             if isinstance(constraints, dict) and constraints:
                 logger.info(f"Regime Constraints: {list(constraints.keys())}")

    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return 1

    # Load Data
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return 1
    
    df = load_csv_data(data_path, index_col=0, parse_dates=True)
    df = _slice_df(df, start=start, end=end)
    logger.info(f"✅ Data loaded: {len(df)} rows")

    # Create Environment
    env = HeavyTradingEnv(df, config)
    
    # Load Model
    try:
        model = SAC.load(model_path, env=env)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return 1

    if grid:
        if z_grid is None or sl_grid is None or tp_grid is None:
            raise ValueError("--grid requires --z-grid, --sl-grid, and --tp-grid")

        combos = [
            (float(z), float(sl), float(tp))
            for z in z_grid
            for sl in sl_grid
            for tp in tp_grid
        ]
        logger.info(f"Running grid search: {len(combos)} combinations")

        results: list[dict[str, Any]] = []
        for idx, (z, sl, tp) in enumerate(combos, start=1):
            _apply_high_vol_ranging_params(
                config,
                entry_zscore_threshold=z,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )
            # Update env config in-place (avoids full re-init)
            if isinstance(getattr(env.config, "hybrid_config", None), dict):
                env.config.hybrid_config = config.get("environment", {}).get("hybrid_config")

            metrics = _run_episode(env=env, model=model, deterministic=deterministic)
            row = {
                "entry_zscore_threshold": z,
                "stop_loss_pct": sl,
                "take_profit_pct": tp,
                "total_return_pct": metrics.get("total_return_pct"),
                "final_balance": metrics.get("final_balance"),
                "total_trades": metrics.get("total_trades"),
            }
            results.append(row)
            logger.info(
                f"[{idx:>3}/{len(combos)}] z={z:.3f} sl={sl:.4f} tp={tp:.4f} -> return={row['total_return_pct']:.2f}% trades={row.get('total_trades')}"
            )

        results_sorted = sorted(results, key=lambda r: float(r.get("total_return_pct") or -1e18), reverse=True)
        top = results_sorted[: min(10, len(results_sorted))]
        logger.info("Top results:")
        for r in top:
            logger.info(
                f"  return={r['total_return_pct']:.2f}% z={r['entry_zscore_threshold']:.3f} sl={r['stop_loss_pct']:.4f} tp={r['take_profit_pct']:.4f} trades={r.get('total_trades')}"
            )

        grid_report_path = str(Path(report_path).with_name("v454_hybrid_grid_search_results.json"))
        with open(grid_report_path, "w", encoding="utf-8") as f:
            json.dump(results_sorted, f, indent=2)
        logger.info(f"Grid results saved to {grid_report_path}")
        return 0

    # Single run (with full diagnostics)
    logger.info("Running backtest...")
    results = _run_episode(
        env=env,
        model=model,
        deterministic=deterministic,
        collect_regime_diagnostics=True,
    )

    logger.info("Backtest completed.")
    logger.info(f"Initial Balance: {results['initial_balance']:,.2f}")
    logger.info(f"Final Balance: {results['final_balance']:,.2f}")
    logger.info(f"Total Return: {results['total_return_pct']:.2f}%")

    # Keep previous behavior: write a detailed JSON report
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Results saved to {report_path}")

    if results:
        logger.info("Statistics:")
        for k, v in results.items():
            if k in {"regime_step_counts", "regime_trade_actions", "regime_entry_counts"}:
                logger.info(f"  {k}: {v}")
            elif k in {"initial_balance", "final_balance", "total_return_pct"}:
                continue
            else:
                logger.info(f"  {k}: {v}")

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAC v454 hybrid backtest (single or grid search).")
    parser.add_argument("--model-path", default="models/sac_v454_inverse_confidence.zip")
    parser.add_argument("--config-path", default="config/v454/sac_v454_config.json")
    parser.add_argument("--data-path", default="data/btc_jpy_1m_v454.csv")
    parser.add_argument("--start", type=int, default=0, help="Row start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="Row end index (0-based, exclusive)")
    parser.add_argument("--z", type=float, default=None, help="Override entry_zscore_threshold")
    parser.add_argument("--sl", type=float, default=None, help="Override stop_loss_pct (e.g., 0.015)")
    parser.add_argument("--tp", type=float, default=None, help="Override take_profit_pct (e.g., 0.01)")
    parser.add_argument("--grid", action="store_true", help="Run grid search over z/sl/tp")
    parser.add_argument("--z-grid", type=_parse_float_list, default=None, help="Comma-separated z thresholds")
    parser.add_argument("--sl-grid", type=_parse_float_list, default=None, help="Comma-separated stop-loss pcts")
    parser.add_argument("--tp-grid", type=_parse_float_list, default=None, help="Comma-separated take-profit pcts")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (non-deterministic) policy")
    parser.add_argument("--report-path", default="backtest_results/v454_hybrid_test_results.json")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    return run_backtest(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        start=args.start,
        end=args.end,
        entry_zscore_threshold=args.z,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        grid=args.grid,
        z_grid=args.z_grid,
        sl_grid=args.sl_grid,
        tp_grid=args.tp_grid,
        deterministic=not args.stochastic,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
