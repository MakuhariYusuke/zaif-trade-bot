import gymnasium as gym
from stable_baselines3 import SAC

# Small smoke test to validate torch + SB3 training using CPU


def main():
    env = gym.make("Pendulum-v1")
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100)
    print("Training done")


if __name__ == "__main__":
    main()
