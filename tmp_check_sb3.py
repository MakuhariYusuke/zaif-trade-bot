import importlib
try:
    import stable_baselines3 as sb3
    print('OK', type(sb3), 'has_SAC=', hasattr(sb3, 'SAC'))
    print('module file:', getattr(sb3, '__file__', None))
    print('dir:', [k for k in dir(sb3) if k.lower().startswith('sa')][:20])
except Exception as e:
    print('IMPORT ERROR:', repr(e))
    raise
