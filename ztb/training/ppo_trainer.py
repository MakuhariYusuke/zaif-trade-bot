"""Compatibility shim: re-export items from core PPO trainer module.

Some tests and legacy code import `ztb.training.ppo_trainer` directly. The
actual implementation now lives under `ztb.training.core`. This shim keeps
backwards compatibility for imports during test collection.

This shim also exposes a few helper symbols (ActionMasker, MaskablePPO,
CompositeTrainingCallback, neutralize_policy_bias, load_csv_data_optimized)
at the module level to support older tests that patch these attributes on the
`ztb.training.ppo_trainer` module itself.
"""
from .core.ppo_trainer import *  # noqa: F401,F403

# Backwards-compatible re-exports for tests that patch module-level attributes
try:
	from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
except Exception:
	ActionMasker = None  # pragma: no cover - may be patched in tests

try:
	from sb3_contrib import MaskablePPO  # type: ignore
except Exception:
	MaskablePPO = None  # pragma: no cover

try:
	from ztb.training.callbacks.callbacks_legacy import CompositeTrainingCallback
except Exception:
	CompositeTrainingCallback = None  # pragma: no cover

try:
	from ztb.trading.environment.environment import HeavyTradingEnv
except Exception:
	HeavyTradingEnv = None  # pragma: no cover

try:
	from ztb.training.policies.policy_utils import neutralize_policy_bias
except Exception:
	neutralize_policy_bias = None  # pragma: no cover

try:
	from ztb.utils.data_utils import load_csv_data_optimized
except Exception:
	load_csv_data_optimized = None  # pragma: no cover

try:
	from ztb.training.models.custom_ppo import CustomPPO
except Exception:
	CustomPPO = None  # pragma: no cover

# Ensure the core module sees these names as well so tests that patch
# attributes on the compatibility shim affect the actual implementation in
# `ztb.training.core.ppo_trainer` (some tests patch `ztb.training.ppo_trainer.xxx`).
try:
	import importlib
	core = importlib.import_module("ztb.training.core.ppo_trainer")
	for name in (
		"ActionMasker",
		"MaskablePPO",
		"CompositeTrainingCallback",
		"neutralize_policy_bias",
		"load_csv_data_optimized",
		"HeavyTradingEnv",
	):
		# Install a small proxy on the core module that delegates to the
		# attribute present on this compatibility shim. This allows tests to
		# patch `ztb.training.ppo_trainer.<name>` and have the core
		# implementation call the (possibly mocked) value at runtime.
		def _make_proxy(n):
			def _proxy(*a, **kw):
				target = globals().get(n)
				if target is None:
					raise AttributeError(f"{n} not available on shim module")
				return target(*a, **kw)

			return _proxy

		# Only create proxies for names that exist on the shim
		if name in globals() and globals()[name] is not None:
			try:
				setattr(core, name, _make_proxy(name))
			except Exception:
				# If we can't set a proxy, fall back to copying the value
				try:
					setattr(core, name, globals().get(name))
				except Exception:
					pass
except Exception:
	pass

