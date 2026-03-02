"""
PluginManager Component.

Provides plugin architecture for Action Signal Guide.
Enables easy addition of new pattern recognizers and signal processors.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from ztb.utils.logging_utils import get_logger

from ..pattern_recognition.base import PatternRecognizer

PluginType = Literal["pattern_recognizer", "signal_processor", "analyzer"]
PluginCallable = Callable[..., object]

class PluginMetadata(TypedDict):
    type: PluginType
    target: object
    description: str

class PluginManager:
    """
    Manages plugins for Action Signal Guide.

    Provides:
    - Dynamic plugin loading
    - Plugin registration and discovery
    - Plugin lifecycle management
    - Extension points for customization
    """

    def __init__(self) -> None:
        self.logger = get_logger("ztb.trading.strategies.plugin_manager")

        # Plugin registries
        self.pattern_recognizers: dict[str, type[PatternRecognizer]] = {}
        self.signal_processors: dict[str, PluginCallable] = {}
        self.analyzers: dict[str, PluginCallable] = {}

        # Plugin metadata
        self.plugin_metadata: dict[str, PluginMetadata] = {}

        # Plugin directories
        self.plugin_dirs = [
            Path(__file__).parent.parent / "pattern_recognition",
            Path(__file__).parent.parent / "components",
            Path(__file__).parent.parent / "analysis",
        ]

    def discover_plugins(self) -> None:
        """Automatically discover and load plugins."""
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self._scan_directory(plugin_dir)

    def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for plugins."""
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                self._load_plugin_from_file(file_path)
            except Exception as exc:
                self.logger.warning(f"Failed to load plugin from {file_path}: {exc}")

    def _load_plugin_from_file(self, file_path: Path) -> None:
        """Load plugins from a Python file."""
        # Convert path to module path
        module_path = self._path_to_module_path(file_path)

        try:
            module = importlib.import_module(module_path)

            # Find plugin classes/functions
            for name, obj in inspect.getmembers(module):
                if self._is_pattern_recognizer(obj):
                    self.register_pattern_recognizer(name, obj)
                elif self._is_signal_processor(obj):
                    self.register_signal_processor(name, obj)
                elif self._is_analyzer(obj):
                    self.register_analyzer(name, obj)

        except ImportError as exc:
            self.logger.debug(f"Could not import {module_path}: {exc}")

    def _path_to_module_path(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
        # This is a simplified conversion - in practice, you'd need more robust path handling
        parts = file_path.with_suffix("").parts
        # Find the ztb index
        try:
            ztb_index = parts.index("ztb")
            module_parts = parts[ztb_index:]
            return ".".join(module_parts)
        except ValueError:
            return str(file_path.stem)

    def _is_pattern_recognizer(self, obj: object) -> bool:
        """Check if object is a PatternRecognizer subclass."""
        return (
            inspect.isclass(obj)
            and issubclass(obj, PatternRecognizer)
            and obj != PatternRecognizer
        )

    def _is_signal_processor(self, obj: object) -> bool:
        """Check if object is a signal processor function."""
        if not callable(obj):
            return False
        name = getattr(obj, "__name__", "")
        return isinstance(name, str) and name.endswith("_processor")

    def _is_analyzer(self, obj: object) -> bool:
        """Check if object is an analyzer function."""
        if not callable(obj):
            return False
        name = getattr(obj, "__name__", "")
        return isinstance(name, str) and name.endswith("_analyzer")

    @staticmethod
    def _build_plugin_metadata(
        plugin_type: PluginType,
        target: object,
        description: str,
    ) -> PluginMetadata:
        """Create standardized plugin metadata payload."""
        return {
            "type": plugin_type,
            "target": target,
            "description": description,
        }

    def register_pattern_recognizer(
        self,
        name: str,
        recognizer_class: type[PatternRecognizer],
    ) -> None:
        """Register a pattern recognizer."""
        self.pattern_recognizers[name] = recognizer_class
        self.plugin_metadata[name] = self._build_plugin_metadata(
            "pattern_recognizer",
            recognizer_class,
            recognizer_class.__doc__ or "No description",
        )
        self.logger.info(f"Registered pattern recognizer: {name}")

    def register_signal_processor(self, name: str, processor_func: PluginCallable) -> None:
        """Register a signal processor."""
        self.signal_processors[name] = processor_func
        self.plugin_metadata[name] = self._build_plugin_metadata(
            "signal_processor",
            processor_func,
            str(getattr(processor_func, "__doc__", "No description")),
        )
        self.logger.info(f"Registered signal processor: {name}")

    def register_analyzer(self, name: str, analyzer_func: PluginCallable) -> None:
        """Register an analyzer."""
        self.analyzers[name] = analyzer_func
        self.plugin_metadata[name] = self._build_plugin_metadata(
            "analyzer",
            analyzer_func,
            str(getattr(analyzer_func, "__doc__", "No description")),
        )
        self.logger.info(f"Registered analyzer: {name}")

    def get_pattern_recognizer(self, name: str) -> type[PatternRecognizer] | None:
        """Get a registered pattern recognizer."""
        return self.pattern_recognizers.get(name)

    def get_signal_processor(self, name: str) -> PluginCallable | None:
        """Get a registered signal processor."""
        return self.signal_processors.get(name)

    def get_analyzer(self, name: str) -> PluginCallable | None:
        """Get a registered analyzer."""
        return self.analyzers.get(name)

    def list_plugins(
        self,
        plugin_type: PluginType | None = None,
    ) -> dict[str, PluginMetadata]:
        """list all registered plugins."""
        if plugin_type:
            return {
                name: metadata
                for name, metadata in self.plugin_metadata.items()
                if metadata.get("type") == plugin_type
            }
        return self.plugin_metadata.copy()

    def create_pattern_recognizer(
        self,
        name: str,
        config: dict[str, object] | None = None,
    ) -> PatternRecognizer | None:
        """Create an instance of a registered pattern recognizer."""
        recognizer_class = self.get_pattern_recognizer(name)
        if recognizer_class:
            try:
                return recognizer_class(config)
            except Exception as exc:
                self.logger.error(f"Failed to create pattern recognizer {name}: {exc}")
        return None

    def execute_signal_processor(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object | None:
        """Execute a registered signal processor."""
        processor = self.get_signal_processor(name)
        if processor:
            try:
                return processor(*args, **kwargs)
            except Exception as exc:
                self.logger.error(f"Failed to execute signal processor {name}: {exc}")
        return None

    def execute_analyzer(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object | None:
        """Execute a registered analyzer."""
        analyzer = self.get_analyzer(name)
        if analyzer:
            try:
                return analyzer(*args, **kwargs)
            except Exception as exc:
                self.logger.error(f"Failed to execute analyzer {name}: {exc}")
        return None

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        metadata = self.plugin_metadata.get(name)
        if not metadata:
            return False

        plugin_type = metadata["type"]

        if plugin_type == "pattern_recognizer":
            self.pattern_recognizers.pop(name, None)
        elif plugin_type == "signal_processor":
            self.signal_processors.pop(name, None)
        elif plugin_type == "analyzer":
            self.analyzers.pop(name, None)

        self.plugin_metadata.pop(name, None)
        self.logger.info(f"Unloaded plugin: {name}")
        return True

    def reload_plugins(self) -> None:
        """Reload all plugins."""
        # Clear existing plugins
        self.pattern_recognizers.clear()
        self.signal_processors.clear()
        self.analyzers.clear()
        self.plugin_metadata.clear()

        # Re-discover plugins
        self.discover_plugins()
        self.logger.info("Reloaded all plugins")
