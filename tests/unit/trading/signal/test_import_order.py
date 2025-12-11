import importlib
import sys


def safe_import(mod_name: str):
    """Import module safely after ensuring any previous import removed"""
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_import_signal_modules_varied_order():
    """Import modules in different orders to detect import-order side-effects"""
    modules = [
        "ztb.trading.signal.common.utilities",
        "ztb.trading.signal.regime.classifier",
        "ztb.trading.signal.quality_scorer",
        "backtest.signal_guidance_backtest",
    ]

    from itertools import permutations

    for order in permutations(modules, 3):
        # Attempt import in this order, ensure no exception
        for m in order:
            try:
                safe_import(m)
            except Exception as e:
                raise AssertionError(f"Failed to import {m} in order {order}: {e}")
