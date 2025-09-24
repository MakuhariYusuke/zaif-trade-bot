"""
Feature cache for backtesting optimization.
特徴量キャッシュ（バックテスト高速化用）
"""

from __future__ import annotations
import hashlib, json, pickle, zlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

class FeatureCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, data_path: str, params: Dict[str, Any]) -> str:
        payload = json.dumps({"data": data_path, "params": params}, sort_keys=True).encode()
        return hashlib.blake2b(payload, digest_size=20).hexdigest()

    def get(self, data_path: str, params: Dict[str, Any]) -> Optional[bytes]:
        key = self._key(data_path, params)
        f = self.cache_dir / f"{key}.pkl.z"
        if f.exists():
            return zlib.decompress(f.read_bytes())
        return None

    def put(self, data_path: str, params: Dict[str, Any], obj: Any) -> Path:
        key = self._key(data_path, params)
        f = self.cache_dir / f"{key}.pkl.z"
        blob = zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 9)
        f.write_bytes(blob)
        return f