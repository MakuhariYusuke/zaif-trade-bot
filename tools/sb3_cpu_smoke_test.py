import gymnasium as gym
try:
    from stable_baselines3 import SAC
except Exception:
    SAC = None

# Small smoke test to validate torch + SB3 training using CPU


def main():
    if SAC is None:
        print("stable_baselines3.SAC not available; skipping smoke test")
        return
    env = gym.make("Pendulum-v1")
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100)
    print("Training done")


if __name__ == "__main__":
    main()
