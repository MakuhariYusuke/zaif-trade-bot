"""215# P0-C: alert_mode.json — オペレータ緊急介入 (DEFCON スイッチ).

ファイルタッチ型の外部シグナルフィード。
サイクル先頭で JSON ファイルの存在をチェックし、
halt / offset_mult / lot_mult / interval_mult をオーバーライドできる。

Usage (PowerShell):
    # 即座に halt
    echo '{"halt": true, "reason": "geopolitical risk"}' > results/v460/fill_test/alert_mode.json

    # 縮小運転 (offset 2x, lot 半減, interval 3x)
    echo '{"offset_mult": 2.0, "lot_mult": 0.5, "interval_mult": 3.0}' > results/v460/fill_test/alert_mode.json

    # 解除
    del results/v460/fill_test/alert_mode.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_ALERT_MODE_FILENAME = "alert_mode.json"


@dataclass(frozen=True, slots=True)
class AlertModeOverride:
    """alert_mode.json から読み込まれたオーバーライド設定."""

    halt: bool = False
    offset_mult: float = 1.0
    lot_mult: float = 1.0
    interval_mult: float = 1.0
    reason: str = ""

    @property
    def is_active(self) -> bool:
        """何らかのオーバーライドが有効か."""
        return (
            self.halt
            or self.offset_mult != 1.0
            or self.lot_mult != 1.0
            or self.interval_mult != 1.0
        )


# キャッシュ: 前回のログ状態 (同一内容の重複ログ抑制)
_last_logged_state: str | None = None

# 無効時の定数 (毎サイクル生成を避ける)
_INACTIVE = AlertModeOverride()


def load_alert_mode(results_dir: str | Path) -> AlertModeOverride:
    """alert_mode.json を読み込みオーバーライド設定を返す.

    ファイルが存在しない場合はデフォルト (無効) を返す。
    パースエラー時はログ出力してデフォルトを返す (fail-safe)。
    """
    global _last_logged_state

    path = Path(results_dir) / _ALERT_MODE_FILENAME
    if not path.exists():
        if _last_logged_state is not None:
            logger.info("[215# P0-C] alert_mode.json removed — overrides cleared")
            _last_logged_state = None
        return _INACTIVE

    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return _INACTIVE
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning(f"[215# P0-C] alert_mode.json: expected dict, got {type(data).__name__}")
            return _INACTIVE
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[215# P0-C] alert_mode.json parse error (fail-safe): {e}")
        return _INACTIVE

    override = AlertModeOverride(
        halt=bool(data.get("halt", False)),
        offset_mult=max(0.1, float(data.get("offset_mult", 1.0))),
        lot_mult=max(0.01, min(1.0, float(data.get("lot_mult", 1.0)))),
        interval_mult=max(1.0, float(data.get("interval_mult", 1.0))),
        reason=str(data.get("reason", "")),
    )

    # 変更時のみログ出力 (毎サイクルの冗長ログ回避)
    state_key = f"halt={override.halt},om={override.offset_mult},lm={override.lot_mult},im={override.interval_mult}"
    if state_key != _last_logged_state:
        _last_logged_state = state_key
        logger.warning(
            f"[215# P0-C] alert_mode ACTIVE: halt={override.halt}, "
            f"offset_mult={override.offset_mult}, lot_mult={override.lot_mult}, "
            f"interval_mult={override.interval_mult}"
            + (f", reason={override.reason}" if override.reason else "")
        )

    return override
