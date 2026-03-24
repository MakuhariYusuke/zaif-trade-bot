"""604# fill_test ログ診断スクリプト — 膠着パターン検出.

Usage:
    python -m scripts.v460.analysis.diagnose_deadlock
    python -m scripts.v460.analysis.diagnose_deadlock --log results/v460/fill_test/logs/fill_test.log.1
    python -m scripts.v460.analysis.diagnose_deadlock --tail 500

Purpose:
    602# 調査で判明した「滞留注文→両側膠着→SAFE_STOP」パターンを
    テキストログから自動検出し、診断レポートを出力する。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG = Path("results/v460/fill_test/logs/fill_test.log")
DEFAULT_LOG_PREV = DEFAULT_LOG.with_suffix(".log.1")

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_PREFLIGHT_SKIP_RE = re.compile(r"\[preflight_skip\] count=(\d+)/(\d+)")
_PREFLIGHT_PAUSE_RE = re.compile(r"\[preflight_pause\] .+ pause #(\d+)/(\d+)")
# 旧形式: [preflight_pause] 連続 preflight 失敗 5 回 / [balance_shrink] 連続 preflight 失敗 3 回
_PREFLIGHT_FAIL_LEGACY_RE = re.compile(
    r"\[(preflight_pause|balance_shrink)\] 連続 preflight 失敗 (\d+) 回"
)
_SAFE_STOP_RE = re.compile(r"SAFE_STOP: 連続 preflight スキップ (\d+)")
_ORDER_PLACE_RE = re.compile(r"Re-quote (\d+)/(\d+): new_price=(\S+)")
_AGE_CAP_RE = re.compile(
    r"sell_age_cap exceeded: elapsed=([\d.]+)s >= cap=([\d.]+)s.*?order_id=(\S+)"
)
_AGE_CAP_OLD_RE = re.compile(
    r"sell_age_cap exceeded: elapsed=([\d.]+)s >= cap=([\d.]+)s"
)
_CANCEL_STALE_RE = re.compile(
    r"\[startup\] Cancelled stale order: id=(\S+), side=(\S+)"
)
_RECOVERY_CANCEL_RE = re.compile(
    r"\[602# preflight_recovery\] Cancelled stale order: id=(\S+)"
)
_BALANCE_RE = re.compile(
    r"btc_free=([\d.?]+), btc_locked=([\d.?]+), jpy_free=([\d.?]+)"
)
_INSUFFICIENT_BTC_RE = re.compile(
    r"Insufficient BTC for sell: free=([\d.]+), locked=([\d.]+)"
)
_INSUFFICIENT_BTC_OLD_RE = re.compile(
    r"Insufficient BTC for sell: ([\d.]+) < "
)
_INSUFFICIENT_JPY_RE = re.compile(
    r"Insufficient JPY for buy: free=([\d.]+), locked=([\d.]+)"
)
_INSUFFICIENT_JPY_OLD_RE = re.compile(
    r"Insufficient JPY for buy: free=([\d.]+) <"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DeadlockEvent:
    """膠着パターンの検出結果."""
    first_skip_ts: str = ""
    last_skip_ts: str = ""
    max_skip_count: int = 0
    pause_count: int = 0
    safe_stopped: bool = False
    recovery_cancelled: list[str] = field(default_factory=list)
    age_cap_orders: list[str] = field(default_factory=list)
    startup_cancelled: list[str] = field(default_factory=list)
    btc_locked_max: float = 0.0
    jpy_free_min: float = float("inf")


def _extract_ts(line: str) -> str:
    m = _TS_RE.match(line)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_log(lines: list[str]) -> list[DeadlockEvent]:
    """ログ行列から膠着イベントを検出."""
    events: list[DeadlockEvent] = []
    current: DeadlockEvent | None = None

    for line in lines:
        ts = _extract_ts(line)

        # preflight skip 連続検出 (604# 新形式)
        m = _PREFLIGHT_SKIP_RE.search(line)
        if m:
            count = int(m.group(1))
            if current is None:
                current = DeadlockEvent(first_skip_ts=ts)
            current.last_skip_ts = ts
            current.max_skip_count = max(current.max_skip_count, count)

        # 旧形式: preflight_pause / balance_shrink × 連続失敗数
        m = _PREFLIGHT_FAIL_LEGACY_RE.search(line)
        if m:
            count = int(m.group(2))
            if current is None:
                current = DeadlockEvent(first_skip_ts=ts)
            current.last_skip_ts = ts
            current.max_skip_count = max(current.max_skip_count, count)

        # balance 詳細
        m = _BALANCE_RE.search(line)
        if m and current is not None:
            try:
                btc_locked = float(m.group(2))
                jpy_free = float(m.group(3))
                current.btc_locked_max = max(current.btc_locked_max, btc_locked)
                current.jpy_free_min = min(current.jpy_free_min, jpy_free)
            except ValueError:
                pass

        # Insufficient BTC with locked (604# 新形式)
        m = _INSUFFICIENT_BTC_RE.search(line)
        if m and current is not None:
            try:
                current.btc_locked_max = max(
                    current.btc_locked_max, float(m.group(2))
                )
            except ValueError:
                pass

        # Insufficient BTC (旧形式: locked なし)
        m = _INSUFFICIENT_BTC_OLD_RE.search(line)
        if m and current is not None:
            pass  # locked 情報なしだがイベントは追跡済み

        # Insufficient JPY (旧形式)
        m = _INSUFFICIENT_JPY_OLD_RE.search(line)
        if m and current is not None:
            try:
                current.jpy_free_min = min(
                    current.jpy_free_min, float(m.group(1))
                )
            except ValueError:
                pass

        # preflight pause
        m = _PREFLIGHT_PAUSE_RE.search(line)
        if m and current is not None:
            current.pause_count = int(m.group(1))

        # SAFE_STOP
        if _SAFE_STOP_RE.search(line):
            if current is None:
                current = DeadlockEvent(first_skip_ts=ts)
            current.safe_stopped = True
            current.last_skip_ts = ts
            events.append(current)
            current = None

        # 602# recovery cancel
        m = _RECOVERY_CANCEL_RE.search(line)
        if m:
            if current is not None:
                current.recovery_cancelled.append(m.group(1))
            # recovery 成功 → イベント終了
            events.append(current or DeadlockEvent(first_skip_ts=ts))
            current = None

        # age_cap exceeded (with order_id — 604#)
        m = _AGE_CAP_RE.search(line)
        if m:
            oid = m.group(3)
            if current is not None:
                current.age_cap_orders.append(oid)

        # age_cap exceeded (old format without order_id)
        elif _AGE_CAP_OLD_RE.search(line):
            if current is not None:
                current.age_cap_orders.append("unknown")

        # startup cancel
        m = _CANCEL_STALE_RE.search(line)
        if m:
            if current is not None:
                current.startup_cancelled.append(m.group(1))
                events.append(current)
                current = None

        # preflight 成功 (skip_count=0 リセット) → 膠着解消
        if current is not None and "preflight_skip_count" not in line:
            if "_preflight_skip_count = 0" in line or (
                "Resetting preflight counter" in line
            ):
                events.append(current)
                current = None

    if current is not None:
        events.append(current)

    return [e for e in events if e.max_skip_count >= 3 or e.safe_stopped]


def format_report(events: list[DeadlockEvent]) -> str:
    """診断レポートを文字列で返す."""
    if not events:
        return "膠着パターンは検出されませんでした。\n"

    lines: list[str] = []
    lines.append(f"=== 膠着診断レポート: {len(events)} 件検出 ===\n")

    for i, ev in enumerate(events, 1):
        lines.append(f"--- Event #{i} ---")
        lines.append(f"  期間: {ev.first_skip_ts} ~ {ev.last_skip_ts}")
        lines.append(f"  最大 skip count: {ev.max_skip_count}")
        lines.append(f"  pause 回数: {ev.pause_count}")
        lines.append(f"  SAFE_STOP: {'YES' if ev.safe_stopped else 'no'}")
        if ev.btc_locked_max > 0:
            lines.append(f"  BTC locked (max): {ev.btc_locked_max:.8f}")
        if ev.jpy_free_min < float("inf"):
            lines.append(f"  JPY free (min): {ev.jpy_free_min:.2f}")
        if ev.age_cap_orders:
            display = ev.age_cap_orders[:5]
            extra = len(ev.age_cap_orders) - 5
            oids = ", ".join(display)
            if extra > 0:
                oids += f" (+{extra} more)"
            lines.append(f"  age_cap 対象 order: {oids}")
        if ev.recovery_cancelled:
            lines.append(
                f"  602# recovery cancel: {', '.join(ev.recovery_cancelled)}"
            )
        if ev.startup_cancelled:
            lines.append(
                f"  startup cancel: {', '.join(ev.startup_cancelled)}"
            )

        # 原因推定
        if ev.btc_locked_max > 0 and ev.age_cap_orders:
            lines.append(
                "  >>> 推定原因: age_cap exceeded で注文が残留 → "
                "BTC locked → 両側膠着"
            )
        elif ev.btc_locked_max > 0:
            lines.append(
                "  >>> 推定原因: BTC locked (open order 残留) → 両側膠着"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="604# fill_test ログ膠着パターン診断",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="ログファイルパス (default: 直近 2 ファイルを自動検索)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="末尾 N 行のみ分析 (0=全行)",
    )
    args = parser.parse_args()

    log_files: list[Path] = []
    if args.log:
        log_files = [args.log]
    else:
        for p in [DEFAULT_LOG_PREV, DEFAULT_LOG]:
            if p.exists():
                log_files.append(p)
        if not log_files:
            print(f"ログファイルが見つかりません: {DEFAULT_LOG}", file=sys.stderr)
            sys.exit(1)

    all_lines: list[str] = []
    for lf in log_files:
        try:
            with open(lf, encoding="utf-8", errors="replace") as f:
                all_lines.extend(f.readlines())
        except OSError as e:
            print(f"ログ読込失敗: {lf}: {e}", file=sys.stderr)

    if args.tail > 0:
        all_lines = all_lines[-args.tail:]

    events = analyze_log(all_lines)
    report = format_report(events)
    print(report)

    # サマリー統計
    safe_stops = sum(1 for e in events if e.safe_stopped)
    recovered = sum(1 for e in events if e.recovery_cancelled)
    print(f"サマリー: 膠着={len(events)}, SAFE_STOP={safe_stops}, "
          f"602#回復={recovered}")


if __name__ == "__main__":
    main()
