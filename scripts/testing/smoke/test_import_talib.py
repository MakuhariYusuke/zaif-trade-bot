import importlib
import sys
import os
sys.path.insert(0, os.getcwd())
try:
    importlib.import_module('ztb.utils.talib_wrapper')
    print('Imported OK')
except Exception:
    import traceback
    traceback.print_exc()
