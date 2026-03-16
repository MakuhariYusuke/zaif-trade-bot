"""Exchange adapter package.

Each sub-package (``coincheck``, ``bitflyer``, …) exposes an adapter class
that inherits from :class:`base.adapter.BaseExchangeAdapter`.  New exchanges
should follow the same pattern:

1. Create ``ztb/trading/live/exchanges/<name>/adapter.py`` inheriting
   ``BaseExchangeAdapter`` and implementing the 7 ``_*_real()`` methods.
2. Create ``ztb/trading/live/exchanges/<name>/config.py`` inheriting
   ``BaseExchangeConfig``.
3. Register in :mod:`ztb.trading.live.registry.broker_registry`.
"""
