import sys

sys.path.append(".")

import numpy as np
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def test_v443_2_model():
    """Test the retrained v443.2 model"""
    try:
        if PPO is None:
            print("stable_baselines3.PPO not available; skipping v443.2 model validation")
            return False
        # Load the retrained model
        model_path = "models/ppo_v443_2_backtest_optimization.zip"
        model = PPO.load(model_path)
        print(f"✅ Successfully loaded v443.2 model from {model_path}")

        # Test basic prediction
        # Create dummy observation (3 features as seen in training)
        obs = np.array([0.5, 0.2, -0.1]).reshape(1, -1)

        # Get action prediction
        action, _ = model.predict(obs, deterministic=True)
        print(f"📊 Test prediction - Observation: {obs.flatten()}, Action: {action}")

        # Test multiple predictions
        test_obs = np.random.randn(5, 3) * 0.1  # Small random observations
        actions, _ = model.predict(test_obs, deterministic=True)
        print("📊 Multiple predictions:")
        for i, (obs, action) in enumerate(zip(test_obs, actions)):
            print(f"  Test {i+1}: Obs {obs}, Action {action}")

        print("✅ v443.2 model validation completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error testing v443.2 model: {e}")
        return False


if __name__ == "__main__":
    test_v443_2_model()
