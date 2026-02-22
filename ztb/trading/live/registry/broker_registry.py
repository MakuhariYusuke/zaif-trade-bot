"""
Broker Registry for Trading Operations

Provides a registry of available brokers and their implementations,
with type-safe factory pattern for multi-exchange support.

146# マルチ取引所対応: credential_env_map + create_adapter() ヘルパー追加。
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

from ..exchanges.base.broker_interfaces import IBroker
from ..exchanges.bitflyer.adapter import BitFlyerAdapter
from ..exchanges.coincheck.adapter import CoincheckAdapter

logger = logging.getLogger(__name__)

# 取引所名 → (API_KEY env var, API_SECRET env var) のマッピング
# 新しい取引所を追加する場合はここにも追記する
_CREDENTIAL_ENV_MAP: Dict[str, Tuple[str, str]] = {
    "coincheck": ("COINCHECK_API_KEY", "COINCHECK_API_SECRET"),
    "bitflyer": ("BITFLYER_API_KEY", "BITFLYER_API_SECRET"),
}


class BrokerRegistry:
    """Registry of available exchange adapters.

    All registered adapters must be subclasses of ``IBroker``.
    Use ``get_broker(name, **kwargs)`` to instantiate by name.
    Use ``create_adapter(name, dry_run=True)`` for env-var-aware creation.
    """

    def __init__(self) -> None:
        self._brokers: Dict[str, Type[IBroker]] = {}
        self._credential_env: Dict[str, Tuple[str, str]] = dict(_CREDENTIAL_ENV_MAP)
        self._register_default_brokers()

    def _register_default_brokers(self) -> None:
        """Register built-in exchange adapters."""
        self.register_broker("coincheck", CoincheckAdapter)
        self.register_broker("bitflyer", BitFlyerAdapter)

    def register_broker(
        self,
        name: str,
        broker_class: Type[IBroker],
        credential_env: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Register an exchange adapter class.

        Args:
            name: Short identifier (e.g. ``"coincheck"``, ``"bitflyer"``).
            broker_class: Must be a subclass of ``IBroker``.
            credential_env: Optional ``(API_KEY_VAR, API_SECRET_VAR)`` tuple.
                If provided, ``create_adapter`` will auto-resolve credentials.

        Raises:
            TypeError: If *broker_class* is not an ``IBroker`` subclass.
        """
        if not (isinstance(broker_class, type) and issubclass(broker_class, IBroker)):
            raise TypeError(
                f"{broker_class!r} is not a subclass of IBroker"
            )
        self._brokers[name] = broker_class
        if credential_env is not None:
            self._credential_env[name] = credential_env
        logger.info("Registered broker: %s -> %s", name, broker_class.__name__)

    def get_broker(self, name: str, **kwargs: Any) -> IBroker:
        """Create and return a broker instance.

        Args:
            name: Registered broker name.
            **kwargs: Forwarded to the adapter constructor.

        Raises:
            ValueError: If *name* is not registered.
        """
        if name not in self._brokers:
            available = ", ".join(sorted(self._brokers))
            raise ValueError(
                f"Unknown broker: {name!r}. Available: {available}"
            )
        return self._brokers[name](**kwargs)

    def get_credential_env_vars(self, name: str) -> Tuple[str, str]:
        """Return ``(API_KEY_VAR, API_SECRET_VAR)`` for a registered broker.

        Raises:
            ValueError: If *name* is not registered or has no credential mapping.
        """
        if name not in self._credential_env:
            raise ValueError(
                f"No credential env mapping for broker: {name!r}. "
                f"Known: {sorted(self._credential_env)}"
            )
        return self._credential_env[name]

    def resolve_credentials(
        self, name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve API credentials from environment variables.

        Returns:
            ``(api_key, api_secret)`` — both ``None`` if env vars are unset/empty.
        """
        key_var, secret_var = self.get_credential_env_vars(name)
        api_key = os.environ.get(key_var, "").strip() or None
        api_secret = os.environ.get(secret_var, "").strip() or None
        return api_key, api_secret

    def create_adapter(
        self,
        name: str,
        *,
        dry_run: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> IBroker:
        """High-level adapter factory with auto credential resolution.

        Args:
            name: Registered broker name (e.g. ``"coincheck"``).
            dry_run: Enable dry-run mode.
            api_key: Override env-var API key.
            api_secret: Override env-var API secret.
            **kwargs: Additional adapter constructor arguments.

        Returns:
            Configured broker adapter instance.

        Raises:
            ValueError: If *name* is unknown, or live mode without credentials.
        """
        if name not in self._brokers:
            available = ", ".join(sorted(self._brokers))
            raise ValueError(
                f"Unknown broker: {name!r}. Available: {available}"
            )

        # Credential resolution: explicit > env var
        if api_key is None or api_secret is None:
            env_key, env_secret = self.resolve_credentials(name)
            api_key = api_key or env_key
            api_secret = api_secret or env_secret

        if not dry_run and not (api_key and api_secret):
            key_var, secret_var = self.get_credential_env_vars(name)
            raise ValueError(
                f"API credentials required for live mode on {name}. "
                f"Set {key_var}/{secret_var} in .env or pass explicitly."
            )

        return self._brokers[name](
            api_key=api_key,
            api_secret=api_secret,
            dry_run=dry_run,
            **kwargs,
        )

    def list_brokers(self) -> list[str]:
        """Return sorted list of registered broker names."""
        return sorted(self._brokers.keys())

    def has_broker(self, name: str) -> bool:
        """Check if a broker is registered."""
        return name in self._brokers


# Global registry instance
_broker_registry: Optional[BrokerRegistry] = None


def get_broker_registry() -> BrokerRegistry:
    """Get global broker registry instance (singleton)."""
    global _broker_registry
    if _broker_registry is None:
        _broker_registry = BrokerRegistry()
    return _broker_registry
