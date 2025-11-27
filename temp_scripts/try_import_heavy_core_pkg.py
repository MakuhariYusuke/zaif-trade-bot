import traceback
import importlib.util

spec = importlib.util.spec_from_file_location(
    'ztb.trading.environment.heavy_env.core',
    r'c:\Users\Admin\dev\zaif-trade-bot\ztb\trading\environment\heavy_env\core.py'
)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('Loaded heavy_core OK')
    print('TORCH AVAILABLE:', getattr(mod, '_TORCH_AVAILABLE', None))
except Exception:
    traceback.print_exc()
