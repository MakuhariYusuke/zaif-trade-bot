try:
    from stable_baselines3 import SAC
    print('Imported SAC:', SAC)
except Exception as e:
    print('ERROR importing SAC:', repr(e))
    raise
