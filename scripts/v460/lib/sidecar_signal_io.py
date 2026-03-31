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

from collections.abc import Callable, Mapping
from collections import OrderedDict
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypeVar, cast

from scripts.v460.lib.sidecar_types import (
    DEFAULT_SIGNAL_TTL_SEC,
    PPOSidecarSignal,
    SidecarSignal,
)
from ztb.utils.safety import safe_to_float
from ztb.utils.time_utils import current_iso_timestamp

logger = logging.getLogger(__name__)

# デフォルトのシグナルファイルパス
DEFAULT_SIGNAL_PATH = Path("cache/sidecar_signal.json")
DEFAULT_PPO_SIGNAL_PATH = Path("cache/ppo_sidecar_signal.json")
_SIDECAR_CACHE_MAX_ENTRIES = 8
_PPO_SIDECAR_CACHE_MAX_ENTRIES = 8
SignalT = TypeVar("SignalT")


class TimestampedSignal(Protocol):
    timestamp: str


TimestampedSignalT = TypeVar("TimestampedSignalT", bound=TimestampedSignal)


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
    return _write_json_atomic(
        _signal_to_dict(signal),
        path=path,
        tmp_prefix=".sidecar_signal_",
    )


def write_ppo_sidecar_signal(
    signal: PPOSidecarSignal,
    path: Path | str = DEFAULT_PPO_SIGNAL_PATH,
) -> Path:
    """PPOSidecarSignal を JSON ファイルに atomic write."""
    return _write_json_atomic(
        _ppo_signal_to_dict(signal),
        path=path,
        tmp_prefix=".ppo_sidecar_signal_",
    )


_SIDECAR_CACHE: OrderedDict[str, tuple[float, TimestampedSignal | None]] = OrderedDict()
_PPO_SIDECAR_CACHE: OrderedDict[str, tuple[float, TimestampedSignal | None]] = (
    OrderedDict()
)


def clear_sidecar_signal_cache() -> None:
    """mtime ベースの sidecar 読み込みキャッシュを空にする."""
    _SIDECAR_CACHE.clear()


def clear_ppo_sidecar_signal_cache() -> None:
    """mtime ベースの ppo sidecar 読み込みキャッシュを空にする."""
    _PPO_SIDECAR_CACHE.clear()


def get_sidecar_signal_cache_stats() -> dict[str, int]:
    """sidecar signal キャッシュの軽量診断情報."""
    return {
        "entries": len(_SIDECAR_CACHE),
        "max_entries": _SIDECAR_CACHE_MAX_ENTRIES,
    }


def get_ppo_sidecar_signal_cache_stats() -> dict[str, int]:
    """ppo sidecar signal キャッシュの軽量診断情報."""
    return {
        "entries": len(_PPO_SIDECAR_CACHE),
        "max_entries": _PPO_SIDECAR_CACHE_MAX_ENTRIES,
    }


def _write_json_atomic(
    data: Mapping[str, object],
    *,
    path: Path | str,
    tmp_prefix: str,
) -> Path:
    """JSON を tmp → rename で atomic に書き込む."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=tmp_prefix,
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(data), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, str(path))
        logger.debug("Sidecar signal written: %s", path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


def _store_signal_cache(
    cache: OrderedDict[str, tuple[float, TimestampedSignalT | None]],
    abs_path: str,
    entry: tuple[float, TimestampedSignalT | None],
    *,
    max_entries: int,
) -> None:
    cache[abs_path] = entry
    cache.move_to_end(abs_path)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _read_signal_core(
    path: Path | str,
    ttl_sec: float,
    *,
    cache: OrderedDict[str, tuple[float, TimestampedSignalT | None]],
    parser: Callable[[dict[str, object]], TimestampedSignalT],
    max_entries: int,
) -> tuple[TimestampedSignalT | None, str]:
    """signal 読み込みの共通コア."""

    path = Path(path)
    abs_path = str(path.absolute())

    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        cache.pop(abs_path, None)
        return None, "missing"
    except OSError as e:
        logger.warning(f"Error reading sidecar signal stat: {e}")
        return None, "error"

    if abs_path in cache:
        cached_mtime, cached_signal = cache[abs_path]
        cache.move_to_end(abs_path)
        if mtime == cached_mtime:
            if cached_signal is not None and ttl_sec > 0:
                if _is_stale(cached_signal.timestamp, ttl_sec):
                    logger.info(
                        "Cached sidecar signal is stale (TTL=%ss exceeded)",
                        ttl_sec,
                    )
                    return None, "stale"
            if cached_signal is None:
                return None, "error"
            return cached_signal, "fresh"

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError("signal payload must be a JSON object")
        data = {str(k): v for k, v in parsed.items()}
    except FileNotFoundError:
        cache.pop(abs_path, None)
        return None, "missing"
    except (json.JSONDecodeError, OSError, TypeError) as e:
        logger.warning(f"Sidecar signal read error: {e}")
        _store_signal_cache(cache, abs_path, (mtime, None), max_entries=max_entries)
        return None, "error"

    try:
        mtime_after = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        mtime_after = mtime
    if mtime_after != mtime:
        logger.debug("sidecar signal file changed during read, skipping cache")
        mtime = mtime_after

    try:
        signal = parser(data)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Sidecar signal parse error: {e}")
        _store_signal_cache(cache, abs_path, (mtime, None), max_entries=max_entries)
        return None, "error"

    if ttl_sec > 0 and _is_stale(signal.timestamp, ttl_sec):
        logger.info(
            f"Sidecar signal stale: timestamp={signal.timestamp}, "
            f"ttl={ttl_sec}s exceeded"
        )
        _store_signal_cache(cache, abs_path, (mtime, signal), max_entries=max_entries)
        return None, "stale"

    _store_signal_cache(cache, abs_path, (mtime, signal), max_entries=max_entries)
    return signal, "fresh"


def _read_sidecar_signal_core(
    path: Path | str = DEFAULT_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> tuple[SidecarSignal | None, str]:
    """SAC sidecar signal 読み込みの共通コア."""
    return cast(
        tuple[SidecarSignal | None, str],
        _read_signal_core(
            path,
            ttl_sec,
            cache=_SIDECAR_CACHE,
            parser=cast(Callable[[dict[str, object]], TimestampedSignal], _dict_to_signal),
            max_entries=_SIDECAR_CACHE_MAX_ENTRIES,
        ),
    )


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


def read_ppo_sidecar_signal(
    path: Path | str = DEFAULT_PPO_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> PPOSidecarSignal | None:
    """ppo_sidecar_signal.json を安全に読み込む."""
    signal, _status = _read_ppo_sidecar_signal_core(path, ttl_sec)
    return signal


def read_ppo_sidecar_signal_with_status(
    path: Path | str = DEFAULT_PPO_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> tuple[PPOSidecarSignal | None, str]:
    """PPO sidecar signal を読み込み、状態文字列も返す."""
    return _read_ppo_sidecar_signal_core(path, ttl_sec)


# ── 内部ヘルパー ──────────────────────────────────────────


def _signal_to_dict(signal: SidecarSignal) -> dict[str, object]:
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


def _ppo_signal_to_dict(signal: PPOSidecarSignal) -> dict[str, object]:
    """PPOSidecarSignal → JSON-serializable dict."""
    return {
        "timestamp": signal.timestamp,
        "action": signal.action,
        "action_probabilities": dict(signal.action_probabilities),
        "model_version": signal.model_version,
        "confidence": signal.confidence,
        "regime_hint": signal.regime_hint,
        "features_snapshot": dict(signal.features_snapshot),
        "training_metrics": dict(signal.training_metrics),
    }


def _dict_to_signal(data: dict[str, object]) -> SidecarSignal:
    """JSON dict → SidecarSignal."""
    return SidecarSignal(
        timestamp=str(data.get("timestamp", "")),
        directional_bias=safe_to_float(data["directional_bias"]),
        model_version=str(data.get("model_version", "")),
        confidence=safe_to_float(data.get("confidence", 1.0), 1.0),
        regime_hint=str(data.get("regime_hint", "")),
        features_snapshot=_coerce_float_dict(data.get("features_snapshot")),
        training_metrics=_coerce_float_dict(data.get("training_metrics")),
    )


def _dict_to_ppo_signal(data: dict[str, object]) -> PPOSidecarSignal:
    """JSON dict → PPOSidecarSignal."""
    raw_probabilities = data.get("action_probabilities")
    if raw_probabilities is None:
        raise KeyError("action_probabilities")
    if not isinstance(raw_probabilities, Mapping):
        raise TypeError("action_probabilities must be a JSON object")

    return PPOSidecarSignal(
        timestamp=str(data.get("timestamp", "")),
        action=str(data["action"]),
        action_probabilities={
            str(k): safe_to_float(v) for k, v in raw_probabilities.items()
        },
        model_version=str(data.get("model_version", "")),
        confidence=safe_to_float(data.get("confidence", 1.0), 1.0),
        regime_hint=str(data.get("regime_hint", "")),
        features_snapshot=_coerce_float_dict(data.get("features_snapshot")),
        training_metrics=_coerce_float_dict(data.get("training_metrics")),
    )


def _read_ppo_sidecar_signal_core(
    path: Path | str = DEFAULT_PPO_SIGNAL_PATH,
    ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
) -> tuple[PPOSidecarSignal | None, str]:
    """PPO sidecar signal 読み込みの共通コア."""
    return cast(
        tuple[PPOSidecarSignal | None, str],
        _read_signal_core(
            path,
            ttl_sec,
            cache=_PPO_SIDECAR_CACHE,
            parser=cast(
                Callable[[dict[str, object]], TimestampedSignal],
                _dict_to_ppo_signal,
            ),
            max_entries=_PPO_SIDECAR_CACHE_MAX_ENTRIES,
        ),
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
    return str(current_iso_timestamp(utc=True))


def _coerce_float_dict(value: object) -> dict[str, float]:
    """JSON object を float dict に安全変換する."""
    if not isinstance(value, Mapping):
        return {}
    return {str(k): safe_to_float(v) for k, v in value.items()}


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


def create_neutral_ppo_signal(timestamp: str | None = None) -> PPOSidecarSignal:
    """SKIP な PPOSidecarSignal を生成 (テスト用・初期値用)."""
    return PPOSidecarSignal(
        timestamp=timestamp or make_timestamp(),
        action="skip",
        action_probabilities={
            "buy": 0.0,
            "sell": 0.0,
            "skip": 1.0,
        },
        confidence=0.0,
        model_version="neutral",
    )
