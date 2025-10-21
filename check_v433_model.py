import os
from stable_baselines3 import SAC

# Load the trained model
model_path = 'checkpoints/sac_v433_test_1000.zip'
if os.path.exists(model_path):
    print('✅ Model file exists:', model_path)
    try:
        model = SAC.load(model_path)
        print('✅ Model loaded successfully')
        print('Policy network architecture:', model.policy.net_arch)
        print('Learning rate:', model.learning_rate)
        print('Gamma:', model.gamma)
        if hasattr(model.replay_buffer, 'buffer_size'):
            print('Buffer size:', model.replay_buffer.buffer_size)
        else:
            print('Buffer size: N/A')
    except Exception as e:
        print('❌ Failed to load model:', e)
else:
    print('❌ Model file not found')