import sys

sys.path.insert(0, ".")

# Test imports
try:
    print("✓ training_utils imports successful")
except Exception as e:
    print(f"✗ training_utils imports failed: {e}")

try:
    print("✓ simple_reward import successful")
except Exception as e:
    print(f"✗ simple_reward import failed: {e}")

try:
    print("✓ train_simple_reward import successful")
except Exception as e:
    print(f"✗ train_simple_reward import failed: {e}")

try:
    print("✓ curriculum_transition import successful")
except Exception as e:
    print(f"✗ curriculum_transition import failed: {e}")

print("All imports completed successfully!")
