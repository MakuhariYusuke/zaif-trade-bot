"""Fill Test CLI エントリポイント.

run_fill_test.py の main() を分離 (158# P2-4: god object 分割).
CLI 引数解析、adapter 構築、config 構築、実行、post-run 判定を担う。
"""
from __future__ import annotations

import atexit
import argparse
import asyncio
import gc
import json
import logging
import logging.handlers
import math
import os
import platform
import signal as signal_mod
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import psutil  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class _ExitDiagnosticsRow(TypedDict, total=False):
    timestamp_utc: str
    trigger: str
    run_id: str
    stop_reason: str | None
    pid: int
    rss_mb: float | None
    vms_mb: float | None
    lock_heartbeat_age_sec: float | None
    gc_counts: list[int]
    gc_thresholds: list[int]
    ml_cache_stats: dict[str, int]
    sidecar_cache_stats: dict[str, int]
    runner_buffer_stats: dict[str, int | None]
    health_monitor: dict[str, int | float | list[int]]


def _safe_mb_from_bytes(raw_bytes: object) -> float | None:
    """byte 数を MB に変換する."""
    if isinstance(raw_bytes, bool):
        return None
    if not isinstance(raw_bytes, (int, float)):
        return None
    value = float(raw_bytes)
    if not math.isfinite(value):
        return None
    return value / (1024.0 * 1024.0)


def _safe_filename_token(value: str | None, *, default: str) -> str:
    """ファイル名へ使える安全な token へ正規化."""
    normalized = (value or "").strip() or default
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in normalized
    )[:96]


def _resolve_lock_path(results_dir: str | Path, lock_path: Path | None) -> Path:
    """診断対象の lock path を解決."""
    if lock_path is not None:
        return lock_path
    return Path(results_dir) / "fill_test.lock"


def _read_lock_heartbeat_age_sec(
    results_dir: str | Path,
    *,
    lock_path: Path | None = None,
    now_ts: float | None = None,
) -> float | None:
    """fill_test.lock の直近 heartbeat age を返す."""
    target = _resolve_lock_path(results_dir, lock_path)
    if not target.exists():
        return None
    try:
        content = target.read_text(encoding="utf-8").strip()
        parts = content.split("|")
        if len(parts) < 2:
            return None
        heartbeat_raw = parts[3] if len(parts) >= 4 else parts[1]
        heartbeat_ts = float(heartbeat_raw)
        reference_now = time.time() if now_ts is None else now_ts
        age_sec = reference_now - heartbeat_ts
        if not math.isfinite(age_sec):
            return None
        return max(age_sec, 0.0)
    except (OSError, TypeError, ValueError):
        return None


def _dump_exit_diagnostics(
    results_dir: str | Path,
    run_id: str,
    *,
    stop_reason: str | None,
    lock_path: Path | None = None,
    trigger: str,
    extra_payload: dict[str, object] | None = None,
) -> Path | None:
    """終了時メモリ診断を JSON へダンプする."""
    target_dir = Path(results_dir) / "diagnostics"
    now = datetime.now(timezone.utc)
    heartbeat_age_sec = _read_lock_heartbeat_age_sec(
        results_dir,
        lock_path=lock_path,
        now_ts=now.timestamp(),
    )

    rss_mb: float | None = None
    vms_mb: float | None = None
    pid = os.getpid()
    try:
        proc = psutil.Process()
        pid = int(proc.pid)
        mem = proc.memory_info()
        rss_mb = _safe_mb_from_bytes(getattr(mem, "rss", None))
        vms_mb = _safe_mb_from_bytes(getattr(mem, "vms", None))
    except Exception as exc:
        logger.debug("exit diagnostics psutil probe failed: %s", exc, exc_info=True)

    payload: _ExitDiagnosticsRow = {
        "timestamp_utc": now.isoformat(),
        "trigger": trigger,
        "run_id": run_id,
        "stop_reason": stop_reason,
        "pid": pid,
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "lock_heartbeat_age_sec": heartbeat_age_sec,
    }
    if extra_payload:
        payload.update(extra_payload)

    dump_name = (
        f"exit_dump_{_safe_filename_token(run_id, default='unknown')}"
        f"_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    dump_path = target_dir / dump_name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        heartbeat_label = (
            f"{heartbeat_age_sec:.1f}s" if heartbeat_age_sec is not None else "n/a"
        )
        rss_label = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
        vms_label = f"{vms_mb:.1f}MB" if vms_mb is not None else "n/a"
        logger.warning(
            "[diag] exit dump trigger=%s stop_reason=%s run_id=%s rss=%s vms=%s heartbeat_age=%s path=%s",
            trigger,
            stop_reason,
            run_id,
            rss_label,
            vms_label,
            heartbeat_label,
            dump_path,
        )
        return dump_path
    except OSError as exc:
        logger.warning("exit diagnostics dump failed: %s", exc, exc_info=True)
        return None


def _collect_runner_buffer_stats(runner: object) -> dict[str, int | None]:
    """fill_test runner が保持している主な in-memory バッファを要約する."""
    stats: dict[str, int | None] = {}

    recent_records = getattr(runner, "_recent_records", None)
    if recent_records is not None:
        try:
            stats["recent_records_count"] = len(recent_records)
        except TypeError:
            stats["recent_records_count"] = None
        stats["recent_records_maxlen"] = getattr(recent_records, "maxlen", None)

    batch_persistence = getattr(runner, "_batch_persistence", None)
    if batch_persistence is not None:
        unsaved_batch = getattr(batch_persistence, "unsaved_batch", None)
        if unsaved_batch is not None:
            try:
                stats["unsaved_batch_count"] = len(unsaved_batch)
            except TypeError:
                stats["unsaved_batch_count"] = None

    for attr_name, prefix in (
        ("_ob_recorder", "ob_recorder"),
        ("_trades_recorder", "trades_recorder"),
    ):
        recorder = getattr(runner, attr_name, None)
        snapshot = getattr(recorder, "snapshot_stats", None)
        if callable(snapshot):
            try:
                recorder_stats = snapshot()
            except Exception as exc:
                logger.debug("%s snapshot_stats failed: %s", attr_name, exc, exc_info=True)
            else:
                if isinstance(recorder_stats, dict):
                    for key, value in recorder_stats.items():
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, int):
                            stats[f"{prefix}_{key}"] = value

    return stats


def _collect_fill_test_memory_diagnostics(runner: object) -> dict[str, object]:
    """終了ダンプへ載せる fill_test/ML cache の軽量診断情報."""
    payload: dict[str, object] = {
        "gc_counts": list(gc.get_count()),
        "gc_thresholds": list(gc.get_threshold()),
    }

    runner_stats = _collect_runner_buffer_stats(runner)
    if runner_stats:
        payload["runner_buffer_stats"] = runner_stats

    health_monitor = getattr(runner, "_health_monitor", None)
    snapshot = getattr(health_monitor, "snapshot_memory_diagnostics", None)
    if callable(snapshot):
        try:
            health_stats = snapshot()
        except Exception as exc:
            logger.debug("health monitor diagnostics snapshot failed: %s", exc, exc_info=True)
        else:
            if isinstance(health_stats, dict) and health_stats:
                payload["health_monitor"] = health_stats

    try:
        from scripts.v460.ml.cache_cleanup import get_ml_data_cache_stats

        payload["ml_cache_stats"] = get_ml_data_cache_stats()
    except Exception as exc:
        logger.debug("ml cache diagnostics snapshot failed: %s", exc, exc_info=True)

    try:
        from scripts.v460.lib.sidecar_signal_io import get_sidecar_signal_cache_stats

        payload["sidecar_cache_stats"] = get_sidecar_signal_cache_stats()
    except Exception as exc:
        logger.debug("sidecar cache diagnostics snapshot failed: %s", exc, exc_info=True)

    return payload


def _build_memory_diagnostics_event_details(
    *,
    trigger: str,
    stop_reason: str | None,
    dump_path: Path | None,
    payload: dict[str, object],
) -> dict[str, object]:
    """fill_test_events.jsonl に流す軽量な診断サマリを構築する."""
    details: dict[str, object] = {
        "trigger": trigger,
        "stop_reason": stop_reason,
        "dump_path": str(dump_path) if dump_path is not None else None,
    }
    for key in (
        "gc_counts",
        "gc_thresholds",
        "ml_cache_stats",
        "sidecar_cache_stats",
        "runner_buffer_stats",
        "health_monitor",
    ):
        if key in payload:
            details[key] = payload[key]
    return details


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

    try:
        adapter = registry.create_adapter(
            exchange_name,
            dry_run=args.dry_run,
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
                from ztb.utils.system_utils import popen_no_window

                retrain_proc = subprocess.Popen(
                    retrain_cmd,
                    **popen_no_window(),
                    stdout=subprocess.DEVNULL,
                    stderr=retrain_stderr_fh,
                )
                logger.info(
                    f"[126#] retrain_scheduler started (PID {retrain_proc.pid}), "
                    f"stderr → {retrain_stderr_path}"
                )
                if not _wait_for_process_start(retrain_proc):
                    logger.error(
                        f"[127#] retrain_scheduler DIED immediately "
                        f"(exit code {retrain_proc.returncode}). "
                        f"Check {retrain_stderr_path}"
                    )
                    retrain_proc = None
            except Exception as e:
                # 327# セットアップ失敗時にファイルハンドルをリークさせない
                if retrain_stderr_fh is not None:
                    retrain_stderr_fh.close()
                    retrain_stderr_fh = None
                logger.warning(f"[126#] retrain_scheduler start failed: {e}")
    return retrain_proc, retrain_stderr_fh


def _wait_for_process_start(
    process: subprocess.Popen,
    *,
    max_wait_sec: float = 10.0,
    poll_interval_sec: float = 0.5,
    success_grace_sec: float = 1.0,
) -> bool:
    """Return True when process survives a short startup grace period."""
    waited_sec = 0.0
    while waited_sec < max_wait_sec:
        if process.poll() is not None:
            return False
        if waited_sec >= success_grace_sec:
            return True
        time.sleep(poll_interval_sec)
        waited_sec += poll_interval_sec
    return process.poll() is None


def _compute_final_judgment(records: list[FillRecord]) -> dict[str, object]:
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

    runner = FillTestRunner(
        adapter, config, yaml_cfg=yaml_cfg,
        config_yaml_path=args.config,  # 169# config hot-reload
    )
    diagnostics_state: dict[str, str | bool | None] = {
        "stop_reason": None,
        "dumped": False,
    }

    def _emit_exit_diagnostics(trigger: str, reason: str | None = None) -> None:
        if diagnostics_state["dumped"]:
            return
        active_stop_reason = reason or (
            diagnostics_state["stop_reason"]
            if isinstance(diagnostics_state["stop_reason"], str)
            else None
        )
        lock_manager = getattr(runner, "_lock_manager", None)
        diagnostics_payload = _collect_fill_test_memory_diagnostics(runner)
        dump_path = _dump_exit_diagnostics(
            config.results_dir,
            runner._run_id,
            stop_reason=active_stop_reason,
            lock_path=getattr(lock_manager, "lockfile_path", None),
            trigger=trigger,
            extra_payload=diagnostics_payload,
        )
        log_event(
            "memory_diagnostics",
            config.results_dir,
            run_id=runner._run_id,
            git_sha=runner._git_sha,
            reason=active_stop_reason,
            details=_build_memory_diagnostics_event_details(
                trigger=trigger,
                stop_reason=active_stop_reason,
                dump_path=dump_path,
                payload=diagnostics_payload,
            ),
        )
        if dump_path is not None:
            diagnostics_state["dumped"] = True

    def _atexit_hook() -> None:
        _emit_exit_diagnostics("atexit")

    atexit.register(_atexit_hook)

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

    # 168# Discord 通知: fill_test 開始
    try:
        from ztb.utils.notify import get_notifier
        _notifier = get_notifier()
        _notifier.notify_fill_test_status(
            runner._run_id, "start",
            {"hours": str(args.hours), "config": args.config,
             "git_sha": runner._git_sha, "dry_run": str(args.dry_run)},
        )
    except Exception as e:
        logger.debug("Discord notification skipped (start): %s", e, exc_info=True)

    # Signal handler
    def _signal_handler(signum: int, frame: object) -> None:
        signal_reason = f"signal_{signum}"
        logger.info(f"Signal {signum} received — requesting shutdown")
        log_event(
            "signal",
            config.results_dir,
            run_id=runner._run_id,
            git_sha=runner._git_sha,
            reason=signal_reason,
        )
        diagnostics_state["stop_reason"] = signal_reason
        _emit_exit_diagnostics("signal", signal_reason)
        runner._kill_switch.kill(signal_reason)

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
        diagnostics_state["stop_reason"] = stop_reason
        _emit_exit_diagnostics("finally", stop_reason)
        atexit.unregister(_atexit_hook)
        # 168# Discord 通知: fill_test 停止/クラッシュ
        try:
            from ztb.utils.notify import get_notifier
            _n = get_notifier()
            _details = {"stop_reason": stop_reason or "unknown",
                        "n_records": str(len(records)),
                        "git_sha": runner._git_sha}
            _n.notify_fill_test_status(
                runner._run_id,
                "crash" if (stop_reason or "").startswith("crash:") else "stop",
                _details,
            )
        except Exception as e:
            logger.debug("Discord notification skipped (stop): %s", e, exc_info=True)
        # 286# 283# P0-3: start/stop ペア保証 — crash 時も必ず stop イベントを記録。
        # crash イベントは except ブロックで既に記録済みだが、
        # stop イベントはセッション境界の確定に必要 (split-brain 事後検出の基盤)。
        if stop_reason:
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
