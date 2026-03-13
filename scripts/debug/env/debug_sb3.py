import importlib, sys
try:
    import stable_baselines3 as sb3
    print('sb3 module:', type(sb3), 'has __file__:', hasattr(sb3, '__file__'))
    print('sb3.__spec__:', getattr(sb3, '__spec__', None))
    print('sb3.SAC exists:', hasattr(sb3, 'SAC'))
    try:
        import stable_baselines3.common.callbacks as cb
        print('callbacks module:', type(cb), 'has __file__:', hasattr(cb, '__file__'))
        print('CallbackList exists:', hasattr(cb, 'CallbackList'))
        print('EvalCallback exists:', hasattr(cb, 'EvalCallback'))
    except Exception as e:
        print('callbacks import failed:', e)
    try:
        import stable_baselines3.common.type_aliases as ta
        print('type_aliases module:', type(ta), 'has __file__:', hasattr(ta, '__file__'))
        print('GymEnv exists:', hasattr(ta, 'GymEnv'))
    except Exception as e:
        print('type_aliases import failed:', e)
except Exception as e:
    print('stable_baselines3 import failed:', e)

print('sys.path[0]:', sys.path[0])
print('stable_baselines3 in sys.modules:', 'stable_baselines3' in sys.modules)
print('stable_baselines3.common in sys.modules:', 'stable_baselines3.common' in sys.modules)
