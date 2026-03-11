"""365# P4: sidecar_signal.json atomic I/O.

SAC sidecar ↔ fill_test 間のプロセス間通信ファイルを
atomic write/read で安全にやり取りする。

設計根拠 (365# §5.3):
  - SAC retrain scheduler が推論結果を JSON で書出す
  - fill_test の orchestrator が毎サイクル読み込む
  - 競合安全のため tmp → atomic rename パターン

構成:
  write_sidecar_signal()  — SidecarSignal → JSON (atomic write)
  read_sidecar_signal()   — JSON → SidecarSignal (safe read)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.v460.lib.sidecar_types import (
    DEFAULT_SIGNAL_TTL_SEC,
    SidecarSignal,
)

logger = logging.getLogger(__name__)

# デフォルトのシグナルファイルパス
DEFAULT_SIGNAL_PATH = Path("cache/sidecar_signal.json")


def write_sidecar_signal(
    signal: SidecarSignal,
    path: Path | str = DEFAULT_SIGNAL_PATH,
) -> Path:
    """SidecarSignal を JSON ファイルに atomic write.

    tmp ファイルに書き込み → rename で原子的に更新する。
    Windows では os.replace() を使用 (既存ファイルの上書き対応)。

    Args:
        signal: 書き込む sidecar シグナル
        path: 出力先パス

    Returns:
        書き込んだファイルのパス

    Raises:
        OSError: ファイル書き込み失敗
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _signal_to_dict(signal)

    # tmp → atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".sidecar_signal_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, str(path))
        logger.debug(f"Sidecar signal written: {path}")
    except Exception:
        # tmp ファイルが残らないようクリーンアップ
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


def read_sidecar_signal(
    path: Path | str = DEFAULT_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> SidecarSignal | None:
    """sidecar_signal.json を安全に読み込む.

    - ファイル不在 → None (初回起動時)
    - JSON パースエラー → None + warning ログ
    - TTL 超過 (stale) → None + info ログ
    - 正常 → SidecarSignal

    Args:
        path: シグナルファイルパス
        ttl_sec: シグナルの有効期限 (秒)。0 以下で TTL チェック無効。

    Returns:
        SidecarSignal or None (読込失敗/stale時)
    """
    path = Path(path)
    # 373# TOCTOU 修正: exists() + read_text() の間にファイルが消える
    # 可能性があるため、直接 try/except で処理する。
    try:
        raw = path.read_text(encoding="utf-8")
        data: dict = json.loads(raw)  # type: ignore[assignment]
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Sidecar signal read error: {e}")
        return None

    # TTL チェック
    if ttl_sec > 0:
        ts_str = data.get("timestamp", "")
        if ts_str and _is_stale(ts_str, ttl_sec):
            logger.info(
                f"Sidecar signal stale: timestamp={ts_str}, "
                f"ttl={ttl_sec}s exceeded"
            )
            return None

    try:
        return _dict_to_signal(data)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Sidecar signal parse error: {e}")
        return None


# ── 内部ヘルパー ──────────────────────────────────────────


def _signal_to_dict(signal: SidecarSignal) -> dict:
    """SidecarSignal → JSON-serializable dict."""
    return {
        "timestamp": signal.timestamp,
        "model_version": signal.model_version,
        "directional_bias": signal.directional_bias,
        "confidence": signal.confidence,
        "regime_hint": signal.regime_hint,
        "features_snapshot": dict(signal.features_snapshot),
        "training_metrics": dict(signal.training_metrics),
    }


def _dict_to_signal(data: dict) -> SidecarSignal:
    """JSON dict → SidecarSignal."""
    return SidecarSignal(
        timestamp=str(data.get("timestamp", "")),
        directional_bias=float(data["directional_bias"]),
        model_version=str(data.get("model_version", "")),
        confidence=float(data.get("confidence", 1.0)),
        regime_hint=str(data.get("regime_hint", "")),
        features_snapshot={
            str(k): float(v)
            for k, v in (data.get("features_snapshot") or {}).items()
        },
        training_metrics={
            str(k): float(v)
            for k, v in (data.get("training_metrics") or {}).items()
        },
    )


def _is_stale(timestamp_str: str, ttl_sec: float) -> bool:
    """ISO 8601 タイムスタンプが TTL を超過しているか判定."""
    try:
        ts = datetime.fromisoformat(timestamp_str)
        # timezone-naive → UTC として扱う
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_sec = (now - ts).total_seconds()
        return age_sec > ttl_sec
    except (ValueError, TypeError):
        # パース不可 → stale 扱い
        return True


def make_timestamp() -> str:
    """現在時刻の ISO 8601 タイムスタンプを生成 (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def create_neutral_signal(timestamp: str | None = None) -> SidecarSignal:
    """NEUTRAL な SidecarSignal を生成 (テスト用・初期値用).

    Args:
        timestamp: タイムスタンプ (省略時は現在時刻)

    Returns:
        directional_bias=0.0 の SidecarSignal
    """
    return SidecarSignal(
        timestamp=timestamp or make_timestamp(),
        directional_bias=0.0,
        confidence=0.0,
        model_version="neutral",
    )
