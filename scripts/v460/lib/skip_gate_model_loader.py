"""
Skip-gate model-loader mixin.

Handles lazy loading and hot-reloading of skip-gate model files.
This mixin is consumed by SkipGateEvaluator; all attributes listed below
are initialised by SkipGateEvaluator.__init__.
"""
from __future__ import annotations

import hashlib
import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ztb.ml.skip_gate_contracts import FillTestConfig, _SkipGateLike

if TYPE_CHECKING:
    pass


class SkipGateModelLoaderMixin:
    """
    Mixin that handles loading and hot-reloading of skip-gate model files.

    Attribute declarations mirror SkipGateEvaluator.__init__ assignments so
    that mypy can resolve ``self.<attr>`` references without ``type: ignore``.
    """

    # ------------------------------------------------------------------
    # Attribute declarations (set by SkipGateEvaluator.__init__)
    # ------------------------------------------------------------------
    _config: FillTestConfig
    _project_root: Path
    _gate_path: Path | None
    _skip_gate: _SkipGateLike | None
    _model_file_hash: str
    _SIDE_MODEL_SLOTS: tuple[str, ...]
    _ALT_MODEL_SLOTS: tuple[str, ...]
    _last_reload_check: float | None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_skip_gate(self) -> None:
        """
        (Re-)load the primary skip-gate model from ``_gate_path``.

        Sets ``_skip_gate`` and ``_model_file_hash``.
        If ``_gate_path`` is None or the file does not exist the gate is
        cleared (set to None) and the hash is reset to the empty string.
        """
        path = self._resolve_gate_path(self._gate_path)
        if path is None or not path.exists():
            self._skip_gate = None
            self._model_file_hash = ""
            return

        self._skip_gate = self._load_model_from_path(path)
        self._model_file_hash = self._file_hash(path)
        self._last_reload_check = time.monotonic()

    def maybe_reload_skip_gate(self) -> bool:
        """
        Reload the model only if the file has changed since the last load.

        Returns
        -------
        bool
            True if the model was reloaded, False otherwise.
        """
        now = time.monotonic()
        last = self._last_reload_check
        if last is not None and (now - last) < self._config.reload_interval_seconds:
            return False

        self._last_reload_check = now

        path = self._resolve_gate_path(self._gate_path)
        if path is None or not path.exists():
            return False

        current_hash = self._file_hash(path)
        if current_hash == self._model_file_hash:
            return False

        self._skip_gate = self._load_model_from_path(path)
        self._model_file_hash = current_hash
        return True

    def get_slot_names(self) -> tuple[str, ...]:
        """Return all model slot names (side + alt)."""
        return self._SIDE_MODEL_SLOTS + self._ALT_MODEL_SLOTS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_gate_path(self, raw: Path | None) -> Path | None:
        """Resolve *raw* relative to ``_project_root`` if not absolute."""
        if raw is None:
            return None
        if raw.is_absolute():
            return raw
        return self._project_root / raw

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Return the SHA-256 hex-digest of *path* (used for change detection)."""
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _load_model_from_path(path: Path) -> _SkipGateLike:
        """
        Load a skip-gate model from *path*.

        The file is expected to be a Python module that exposes a
        ``build_model() -> _SkipGateLike`` factory, or a pickle/joblib
        file whose top-level object satisfies ``_SkipGateLike``.

        This implementation uses importlib for ``.py`` files and
        ``importlib.util`` / joblib for binary artefacts.
        """
        if path.suffix == ".py":
            return SkipGateModelLoaderMixin._load_py_module(path)
        return SkipGateModelLoaderMixin._load_binary(path)

    @staticmethod
    def _load_py_module(path: Path) -> _SkipGateLike:
        """Load a skip-gate model from a Python module file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("_skip_gate_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        factory = getattr(module, "build_model", None)
        if factory is None:
            raise AttributeError(f"Module {path} does not expose build_model()")
        model: _SkipGateLike = factory()
        return model

    @staticmethod
    def _load_binary(path: Path) -> _SkipGateLike:
        """Load a skip-gate model from a binary file (joblib / pickle)."""
        try:
            import joblib

            model: _SkipGateLike = joblib.load(path)
        except ImportError:
            import pickle

            with path.open("rb") as fh:
                model = pickle.load(fh)  # noqa: S301
        return model
