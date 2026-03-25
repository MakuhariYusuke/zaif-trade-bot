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

from collections import OrderedDict
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
from ztb.utils.time_utils import current_iso_timestamp

logger = logging.getLogger(__name__)

# デフォルトのシグナルファイルパス
DEFAULT_SIGNAL_PATH = Path("cache/sidecar_signal.json")
_SIDECAR_CACHE_MAX_ENTRIES = 8


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


_SIDECAR_CACHE: OrderedDict[str, tuple[float, SidecarSignal | None]] = OrderedDict()


def clear_sidecar_signal_cache() -> None:
    """mtime ベースの sidecar 読み込みキャッシュを空にする."""
    _SIDECAR_CACHE.clear()


def get_sidecar_signal_cache_stats() -> dict[str, int]:
    """sidecar signal キャッシュの軽量診断情報."""
    return {
        "entries": len(_SIDECAR_CACHE),
        "max_entries": _SIDECAR_CACHE_MAX_ENTRIES,
    }


def _store_sidecar_cache(
    abs_path: str,
    entry: tuple[float, SidecarSignal | None],
) -> None:
    _SIDECAR_CACHE[abs_path] = entry
    _SIDECAR_CACHE.move_to_end(abs_path)
    while len(_SIDECAR_CACHE) > _SIDECAR_CACHE_MAX_ENTRIES:
        _SIDECAR_CACHE.popitem(last=False)

def _read_sidecar_signal_core(
    path: Path | str = DEFAULT_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> tuple[SidecarSignal | None, str]:
    """sidecar signal 読み込みの共通コア.

    487# P0: signal + status を返す統合実装。
    read_sidecar_signal / read_sidecar_signal_with_status が共有する。

    Returns:
        (signal, status) — status は "fresh"/"stale"/"missing"/"error"
    """
    path = Path(path)
    abs_path = str(path.absolute())

    # mtime の取得 (I/O)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _SIDECAR_CACHE.pop(abs_path, None)
        return None, "missing"
    except OSError as e:
        logger.warning(f"Error reading sidecar signal stat: {e}")
        return None, "error"

    # キャッシュチェック
    if abs_path in _SIDECAR_CACHE:
        cached_mtime, cached_signal = _SIDECAR_CACHE[abs_path]
        _SIDECAR_CACHE.move_to_end(abs_path)
        if mtime == cached_mtime:
            # TTL は動的 (時間経過) なので、キャッシュヒットしても都度チェック
            if cached_signal is not None and ttl_sec > 0:
                if _is_stale(cached_signal.timestamp, ttl_sec):
                    logger.info(f"Cached sidecar signal is stale (TTL={ttl_sec}s exceeded)")
                    return None, "stale"
            if cached_signal is None:
                return None, "error"
            return cached_signal, "fresh"

    # キャッシュミス時の読み込み
    try:
        raw = path.read_text(encoding="utf-8")
        data: dict = json.loads(raw)  # type: ignore[assignment]
    except FileNotFoundError:
        _SIDECAR_CACHE.pop(abs_path, None)
        return None, "missing"
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Sidecar signal read error: {e}")
        _store_sidecar_cache(abs_path, (mtime, None))
        return None, "error"

    # 488#/489# P1: stat/read race condition 対策 (double-check pattern)
    # read_text() 中にファイルが書き換わった場合、mtime が変わっているはず
    try:
        mtime_after = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        mtime_after = mtime  # ファイル消失時はそのまま続行
    if mtime_after != mtime:
        # read 中にファイルが更新された → キャッシュせず次回再読み込み
        logger.debug("sidecar signal file changed during read, skipping cache")
        mtime = mtime_after

    try:
        signal = _dict_to_signal(data)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Sidecar signal parse error: {e}")
        _store_sidecar_cache(abs_path, (mtime, None))
        return None, "error"

    # 初回パース時の TTL チェック
    if ttl_sec > 0 and _is_stale(signal.timestamp, ttl_sec):
        logger.info(
            f"Sidecar signal stale: timestamp={signal.timestamp}, "
            f"ttl={ttl_sec}s exceeded"
        )
        # 629# fix: signal 実体を保持し、次回キャッシュヒット時に
        # _is_stale() で都度判定させる。(mtime, None) だと "error" に化ける。
        _store_sidecar_cache(abs_path, (mtime, signal))
        return None, "stale"

    # キャッシュ更新
    _store_sidecar_cache(abs_path, (mtime, signal))
    return signal, "fresh"


def read_sidecar_signal(
    path: Path | str = DEFAULT_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> SidecarSignal | None:
    """sidecar_signal.json を安全に読み込む.

    379# P3-C: mtime ベースのキャッシュを導入し、毎サイクルの同期 I/O を削減。
    
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
    signal, _status = _read_sidecar_signal_core(path, ttl_sec)
    return signal


def read_sidecar_signal_with_status(
    path: Path | str = DEFAULT_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> tuple[SidecarSignal | None, str]:
    """487# P0: sidecar signal を読み込み、状態文字列も返す.

    Returns:
        (signal, status) — status は "fresh"/"stale"/"missing"/"error"
    """
    return _read_sidecar_signal_core(path, ttl_sec)


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
    return current_iso_timestamp(utc=True)


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
