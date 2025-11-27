import traceback
import importlib.util

spec = importlib.util.spec_from_file_location('heavy_core', r'c:\Users\Admin\dev\zaif-trade-bot\ztb\trading\environment\heavy_env\core.py')
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('Loaded heavy_core OK')
except Exception:
    traceback.print_exc()
