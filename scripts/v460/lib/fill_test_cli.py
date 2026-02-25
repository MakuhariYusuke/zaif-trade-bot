"""Fill Test CLI エントリポイント.

run_fill_test.py の main() を分離 (158# P2-4: god object 分割).
CLI 引数解析、adapter 構築、config 構築、実行、post-run 判定を担う。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import platform
import signal as signal_mod
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI 引数パーサ構築."""
    parser = argparse.ArgumentParser(
        description="G1.1-exec Fill Test Runner (009# §4.2)",
    )
    parser.add_argument("--hours", type=float, default=24.0,
                        help="実測時間 (時間). デフォルト: 24h")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run モード (実際に発注しない)")
    parser.add_argument("--config", default=None,
                        help="設定 YAML パス (デフォルト: configs/v460/fill_test.yaml)")
    parser.add_argument("--api-key", default=None,
                        help="[DEPRECATED] .env から読込を推奨")
    parser.add_argument("--api-secret", default=None,
                        help="[DEPRECATED] .env から読込を推奨")
    parser.add_argument("--results-dir", default=None,
                        help="結果保存ディレクトリ (CLI > YAML)")
    parser.add_argument("--results-only", action="store_true",
                        help="既存データからメトリクスのみ算出")
    parser.add_argument("--cycle-interval", type=float, default=None,
                        help="サイクル間隔 (秒) (CLI > YAML)")
    parser.add_argument("--output", default=None,
                        help="判定結果の JSON 出力先")
    parser.add_argument("--start-side", choices=["buy", "sell"], default=None,
                        help="開始サイド (CLI > YAML)")
    parser.add_argument("--spread-offset-ratio", type=float, default=None,
                        help="スプレッド比例オフセット率 (CLI > YAML)")
    parser.add_argument("--min-spread-jpy", type=float, default=None,
                        help="最小スプレッドフィルター (JPY) (CLI > YAML)")
    parser.add_argument("--enable-auto-adapt", action="store_true", default=False,
                        help="方策A: 自動パラメータ適応を有効化 (CLI > YAML)")
    parser.add_argument("--enable-dynamic-lot", action="store_true", default=False,
                        help="方策B: 動的ロットサイジングを有効化 (CLI > YAML)")
    parser.add_argument("--max-lot", type=float, default=None,
                        help="方策B: ロット上限 (BTC) (CLI > YAML)")
    parser.add_argument("--exchange", default="coincheck",
                        help="取引所名 (coincheck/bitflyer 等, デフォルト: coincheck)")
    return parser


def _create_adapter(args: argparse.Namespace) -> "IBroker":
    """取引所Adapterを生成."""
    from dotenv import load_dotenv
    from ztb.trading.live.registry.broker_registry import get_broker_registry

    load_dotenv(_PROJECT_ROOT / ".env")

    exchange_name = args.exchange.lower()
    registry = get_broker_registry()
    if not registry.has_broker(exchange_name):
        logger.error(
            f"Unknown exchange: {exchange_name!r}. "
            f"Available: {', '.join(registry.list_brokers())}"
        )
        sys.exit(1)

    cli_api_key: str | None = None
    cli_api_secret: str | None = None
    if args.api_key or args.api_secret:
        logger.warning(
            "WARNING: --api-key/--api-secret はプロセスリストや履歴に平文で残ります。"
            ".env ファイルからの読込を推奨します。"
        )
        cli_api_key = args.api_key
        cli_api_secret = args.api_secret

    try:
        adapter = registry.create_adapter(
            exchange_name,
            dry_run=args.dry_run,
            api_key=cli_api_key,
            api_secret=cli_api_secret,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Exchange adapter: {exchange_name} (dry_run={args.dry_run})")
    return adapter


def _build_config(args: argparse.Namespace) -> tuple["FillTestConfig", dict]:
    """YAML → CLI override で FillTestConfig を構築."""
    from scripts.v460.lib.config_loader import load_fill_test_config
    from scripts.v460.lib.fill_config import FillTestConfig

    yaml_cfg = load_fill_test_config(args.config)
    config = FillTestConfig.from_yaml(yaml_cfg)

    # CLI 引数が明示指定された場合のみ上書き
    if args.cycle_interval is not None:
        config.cycle_interval_sec = args.cycle_interval
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.start_side is not None:
        config.start_side = args.start_side
    if args.spread_offset_ratio is not None:
        config.spread_offset_ratio = args.spread_offset_ratio
    if args.min_spread_jpy is not None:
        config.min_spread_jpy = args.min_spread_jpy
    if args.enable_auto_adapt:
        config.enable_auto_adapt = True
    if args.enable_dynamic_lot:
        config.enable_dynamic_lot = True
    if args.max_lot is not None:
        config.max_lot = args.max_lot

    logger.info(
        f"Config loaded: YAML={args.config or 'default'}, "
        f"offset={config.spread_offset_ratio}, lot={config.order_quantity}, "
        f"adapt={config.enable_auto_adapt}, dynamic_lot={config.enable_dynamic_lot}, "
        f"regime={config.enable_regime}, "
        f"time_filter={config.enable_time_filter}, "
        f"loss_cap_auto={config.loss_cap_auto}"
    )
    return config, yaml_cfg


def _setup_file_logging(config: "FillTestConfig") -> None:
    """024# O3: ログファイル出力 (ローテーション付き)."""
    log_dir = Path(config.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "fill_test.log",
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, config.file_log_level, logging.DEBUG))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_dir / 'fill_test.log'}")


def _start_retrain_scheduler(
    yaml_cfg: dict,
    config: "FillTestConfig",
    cli_config_path: str | None,
) -> tuple[subprocess.Popen | None, object]:
    """126# retrain_scheduler を子プロセスとして自動起動."""
    retrain_proc = None
    retrain_stderr_fh = None
    retrain_cfg = yaml_cfg.get("retrain", {})
    if retrain_cfg.get("enabled", True):
        retrain_script = _PROJECT_ROOT / "scripts" / "v460" / "ml" / "retrain_scheduler.py"
        if retrain_script.exists():
            retrain_cmd = [
                sys.executable,
                str(retrain_script),
                "--config",
                str(cli_config_path or _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"),
            ]
            try:
                retrain_log_dir = Path(config.results_dir) / "logs"
                retrain_log_dir.mkdir(parents=True, exist_ok=True)
                retrain_stderr_path = retrain_log_dir / "retrain_scheduler_stderr.log"
                retrain_stderr_fh = open(retrain_stderr_path, "a", encoding="utf-8")
                retrain_proc = subprocess.Popen(
                    retrain_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=retrain_stderr_fh,
                )
                logger.info(
                    f"[126#] retrain_scheduler started (PID {retrain_proc.pid}), "
                    f"stderr → {retrain_stderr_path}"
                )
                time.sleep(10)
                if retrain_proc.poll() is not None:
                    logger.error(
                        f"[127#] retrain_scheduler DIED immediately "
                        f"(exit code {retrain_proc.returncode}). "
                        f"Check {retrain_stderr_path}"
                    )
                    retrain_proc = None
            except Exception as e:
                logger.warning(f"[126#] retrain_scheduler start failed: {e}")
    return retrain_proc, retrain_stderr_fh


def _compute_final_judgment(records: list) -> dict:
    """Post-run: metrics & judgment 算出."""
    from ztb.metrics.fill_quality import (
        compute_fill_metrics,
        filter_clean_records,
        g1_1_judgment,
        g1_1_quick_judgment,
        g1_2_full_judgment,
    )
    from scripts.v460.lib.config_loader import load_gate_thresholds
    from scripts.v460.lib.results_analyzer import (
        compute_event_contribution,
        compute_multi_track_analysis,
        compute_regime_breakdown,
        log_event_contribution,
        log_multi_track_summary,
        log_regime_breakdown,
    )

    clean_records, quarantine_records = filter_clean_records(records)
    if quarantine_records:
        logger.info(
            f"[main] quarantine {len(quarantine_records)}/{len(records)} "
            f"records excluded from final metrics"
        )
    metrics = compute_fill_metrics(clean_records)
    gate_cfg = load_gate_thresholds()
    thresholds = gate_cfg.get("g1_1_exec", {})
    judgment = g1_1_judgment(metrics, thresholds)

    # 116# 二段階判定
    quick_thresholds = gate_cfg.get("g1_1_quick_exec", {})
    full_thresholds = gate_cfg.get("g1_2_full_exec", {})
    quick_judgment = g1_1_quick_judgment(metrics, quick_thresholds)
    full_judgment = g1_2_full_judgment(metrics, full_thresholds)
    judgment["two_stage"] = {
        "g1_1_quick": quick_judgment,
        "g1_2_full": full_judgment,
    }

    n_total = len(records)
    judgment["data_quality"] = {
        "total_records": n_total,
        "clean_records": len(clean_records),
        "quarantine_records": len(quarantine_records),
        "clean_rate": len(clean_records) / n_total if n_total else 0.0,
        "quarantine_rate": len(quarantine_records) / n_total if n_total else 0.0,
        "as_coverage": metrics.as_coverage,
        "as_raw_coverage": metrics.as_raw_coverage,
    }
    del records, quarantine_records  # メモリ早期解放

    # 120# A2: run 別二系統分析
    multi_track = compute_multi_track_analysis(clean_records)
    log_multi_track_summary(multi_track)
    judgment["multi_track"] = multi_track

    event_contrib = compute_event_contribution(clean_records)
    log_event_contribution(event_contrib)
    judgment["event_contribution"] = event_contrib

    regime_breakdown = compute_regime_breakdown(clean_records)
    log_regime_breakdown(regime_breakdown)
    judgment["regime_breakdown"] = regime_breakdown

    return judgment


def fill_test_main() -> None:
    """Fill Test CLI メインエントリポイント."""
    from scripts.v460.lib.event_logger import log_event, setup_stderr_mirror
    from scripts.v460.lib.results_analyzer import run_results_only, save_judgment

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.results_only:
        rd = args.results_dir or "results/v460/fill_test"
        result = run_results_only(rd)
        if args.output:
            save_judgment(result, args.output)
            logger.info(f"Saved judgment to {args.output}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            sys.exit(0)
        elif gate == "PASS":
            logger.info(f"Gate PASS but judgment_type={jtype} (not FINAL), exit 2")
            sys.exit(2)
        else:
            sys.exit(1)

    # Adapter + Config 構築
    adapter = _create_adapter(args)
    config, yaml_cfg = _build_config(args)

    # ログ設定
    _setup_file_logging(config)
    setup_stderr_mirror(config.results_dir)

    # Runner 生成
    from scripts.v460.run_fill_test import FillTestRunner
    from scripts.v460.lib.lock_manager import LockConflictError

    runner = FillTestRunner(adapter, config, yaml_cfg=yaml_cfg)

    # 148# P0: start イベント記録
    log_event(
        "start",
        config.results_dir,
        run_id=runner._run_id,
        git_sha=runner._git_sha,
        details={
            "hours": args.hours,
            "config": args.config,
            "args": {
                "hours": args.hours,
                "config": args.config,
                "exchange": args.exchange,
                "dry_run": args.dry_run,
            },
        },
    )

    # Signal handler
    def _signal_handler(signum: int, frame: object) -> None:
        logger.info(f"Signal {signum} received — requesting shutdown")
        log_event(
            "signal",
            config.results_dir,
            run_id=runner._run_id,
            git_sha=runner._git_sha,
            reason=f"signal_{signum}",
        )
        runner._kill_switch.kill(f"signal_{signum}")

    signal_mod.signal(signal_mod.SIGINT, _signal_handler)
    if platform.system() == "Windows":
        try:
            signal_mod.signal(signal_mod.SIGBREAK, _signal_handler)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            logger.debug("SIGBREAK not available on this platform")
    else:
        signal_mod.signal(signal_mod.SIGTERM, _signal_handler)

    # retrain_scheduler 起動
    retrain_proc, retrain_stderr_fh = _start_retrain_scheduler(
        yaml_cfg, config, args.config,
    )

    # 実行
    stop_reason: str | None = None
    records: list = []
    try:
        records = asyncio.run(runner.run_continuous(args.hours))
        stop_reason = (
            runner._kill_switch.get_reason()
            if runner._kill_switch.is_killed()
            else "completed"
        )
    except LockConflictError as e:
        # 166# HF2: 別プロセス稼働中は crash 扱いせず正常終了
        logger.info(f"[lock] {e}")
        stop_reason = "lock_conflict"
        return
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        logger.info("KeyboardInterrupt — stopping gracefully")
    except Exception as e:
        import traceback
        stop_reason = f"crash:{type(e).__name__}"
        log_event(
            "crash",
            config.results_dir,
            run_id=runner._run_id,
            git_sha=runner._git_sha,
            reason=stop_reason,
            details={"traceback": traceback.format_exc()},
        )
        logger.error(f"[148#] Unhandled exception: {e}", exc_info=True)
        raise
    finally:
        if stop_reason and not stop_reason.startswith("crash:"):
            log_event(
                "stop",
                config.results_dir,
                run_id=runner._run_id,
                git_sha=runner._git_sha,
                reason=stop_reason,
            )
        if retrain_proc is not None and retrain_proc.poll() is None:
            retrain_proc.terminate()
            try:
                retrain_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                retrain_proc.kill()
            logger.info(f"[126#] retrain_scheduler stopped (PID {retrain_proc.pid})")
        if retrain_stderr_fh is not None:
            retrain_stderr_fh.close()

    # 最終判定
    if records:
        judgment = _compute_final_judgment(records)
        out_str = json.dumps(judgment, indent=2, ensure_ascii=False)
        print(out_str)
        if args.output:
            save_judgment(judgment, args.output)
            logger.info(f"Saved judgment to {args.output}")

        jtype = judgment.get("judgment_type", "PROVISIONAL")
        gate = judgment.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            sys.exit(0)
        elif gate == "PASS":
            logger.info(f"Gate PASS but judgment_type={jtype} (not FINAL), exit 2")
            sys.exit(2)
        else:
            sys.exit(1)
    else:
        logger.warning("No records collected")
        sys.exit(1)
