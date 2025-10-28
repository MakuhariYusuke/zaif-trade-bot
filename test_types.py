import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Add the action_signal_guide directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ztb', 'trading', 'strategies', 'action_signal_guide'))

try:
    from action_signal_guide import ActionSignalGuide
    print('ActionSignalGuide imported successfully')

    # Try basic instantiation
    guide = ActionSignalGuide()
    print('ActionSignalGuide instantiated successfully')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()