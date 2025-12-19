"""
PluginManager Component.

Provides plugin architecture for Action Signal Guide.
Enables easy addition of new pattern recognizers and signal processors.
"""

from typing import Dict, Optional, Any, Type, Callable
import importlib
import inspect
from pathlib import Path
from ztb.utils.logging_utils import get_logger

from ..pattern_recognition.base import PatternRecognizer


class PluginManager:
    """
    Manages plugins for Action Signal Guide.

    Provides:
    - Dynamic plugin loading
    - Plugin registration and discovery
    - Plugin lifecycle management
    - Extension points for customization
    """

    def __init__(self):
        self.logger = get_logger("ztb.trading.strategies.plugin_manager")

        # Plugin registries
        self.pattern_recognizers: Dict[str, Type[PatternRecognizer]] = {}
        self.signal_processors: Dict[str, Callable] = {}
        self.analyzers: Dict[str, Callable] = {}

        # Plugin metadata
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}

        # Plugin directories
        self.plugin_dirs = [
            Path(__file__).parent.parent / "pattern_recognition",
            Path(__file__).parent.parent / "components",
            Path(__file__).parent.parent / "analysis"
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
            except Exception as e:
                self.logger.warning(f"Failed to load plugin from {file_path}: {e}")

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

        except ImportError as e:
            self.logger.debug(f"Could not import {module_path}: {e}")

    def _path_to_module_path(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
        # This is a simplified conversion - in practice, you'd need more robust path handling
        parts = file_path.with_suffix('').parts
        # Find the ztb index
        try:
            ztb_index = parts.index('ztb')
            module_parts = parts[ztb_index:]
            return '.'.join(module_parts)
        except ValueError:
            return str(file_path.stem)

    def _is_pattern_recognizer(self, obj: Any) -> bool:
        """Check if object is a PatternRecognizer subclass."""
        return (
            inspect.isclass(obj) and
            issubclass(obj, PatternRecognizer) and
            obj != PatternRecognizer
        )

    def _is_signal_processor(self, obj: Any) -> bool:
        """Check if object is a signal processor function."""
        return (
            callable(obj) and
            hasattr(obj, '__name__') and
            obj.__name__.endswith('_processor')
        )

    def _is_analyzer(self, obj: Any) -> bool:
        """Check if object is an analyzer function."""
        return (
            callable(obj) and
            hasattr(obj, '__name__') and
            obj.__name__.endswith('_analyzer')
        )

    def register_pattern_recognizer(self, name: str, recognizer_class: Type[PatternRecognizer]) -> None:
        """Register a pattern recognizer."""
        self.pattern_recognizers[name] = recognizer_class
        self.plugin_metadata[name] = {
            'type': 'pattern_recognizer',
            'class': recognizer_class,
            'description': recognizer_class.__doc__ or 'No description'
        }
        self.logger.info(f"Registered pattern recognizer: {name}")

    def register_signal_processor(self, name: str, processor_func: Callable) -> None:
        """Register a signal processor."""
        self.signal_processors[name] = processor_func
        self.plugin_metadata[name] = {
            'type': 'signal_processor',
            'function': processor_func,
            'description': getattr(processor_func, '__doc__', 'No description')
        }
        self.logger.info(f"Registered signal processor: {name}")

    def register_analyzer(self, name: str, analyzer_func: Callable) -> None:
        """Register an analyzer."""
        self.analyzers[name] = analyzer_func
        self.plugin_metadata[name] = {
            'type': 'analyzer',
            'function': analyzer_func,
            'description': getattr(analyzer_func, '__doc__', 'No description')
        }
        self.logger.info(f"Registered analyzer: {name}")

    def get_pattern_recognizer(self, name: str) -> Optional[Type[PatternRecognizer]]:
        """Get a registered pattern recognizer."""
        return self.pattern_recognizers.get(name)

    def get_signal_processor(self, name: str) -> Optional[Callable]:
        """Get a registered signal processor."""
        return self.signal_processors.get(name)

    def get_analyzer(self, name: str) -> Optional[Callable]:
        """Get a registered analyzer."""
        return self.analyzers.get(name)

    def list_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins."""
        if plugin_type:
            return {
                name: metadata for name, metadata in self.plugin_metadata.items()
                if metadata.get('type') == plugin_type
            }
        return self.plugin_metadata.copy()

    def create_pattern_recognizer(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[PatternRecognizer]:
        """Create an instance of a registered pattern recognizer."""
        recognizer_class = self.get_pattern_recognizer(name)
        if recognizer_class:
            try:
                return recognizer_class(config)
            except Exception as e:
                self.logger.error(f"Failed to create pattern recognizer {name}: {e}")
        return None

    def execute_signal_processor(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered signal processor."""
        processor = self.get_signal_processor(name)
        if processor:
            try:
                return processor(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to execute signal processor {name}: {e}")
        return None

    def execute_analyzer(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered analyzer."""
        analyzer = self.get_analyzer(name)
        if analyzer:
            try:
                return analyzer(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to execute analyzer {name}: {e}")
        return None

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name in self.plugin_metadata:
            plugin_type = self.plugin_metadata[name]['type']

            if plugin_type == 'pattern_recognizer':
                self.pattern_recognizers.pop(name, None)
            elif plugin_type == 'signal_processor':
                self.signal_processors.pop(name, None)
            elif plugin_type == 'analyzer':
                self.analyzers.pop(name, None)

            self.plugin_metadata.pop(name, None)
            self.logger.info(f"Unloaded plugin: {name}")
            return True

        return False

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
