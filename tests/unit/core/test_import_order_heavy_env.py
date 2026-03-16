import importlib
import sys


def safe_import(mod_name: str):
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_import_heavy_env_and_signal_in_varied_order():
    modules = [
        "ztb.trading.environment.heavy_env.core",
        "ztb.trading.signal.regime.classifier",
        "ztb.trading.signal.common.utilities",
    ]
    from itertools import permutations

    for order in permutations(modules, 3):
        for m in order:
            try:
                safe_import(m)
            except Exception as e:
                raise AssertionError(f"Failed to import {m} in order {order}: {e}")
