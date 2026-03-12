
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FixedTTLWrapper(gym.ActionWrapper):
    """
    ActionWrapper to fix the Time-To-Live (TTL) component of the action space.
    
    Transforms a 2D action space (Target Position, TTL) into a 1D action space (Target Position),
    fixedly setting the TTL component to a specified value (default 1.0 = Max TTL).
    
    This is used to debug "short TTL loop" issues or to train simplified agents.
    """
    
    def __init__(self, env: gym.Env, fixed_ttl: float = 1.0):
        super().__init__(env)
        self.fixed_ttl = fixed_ttl
        
        # Check if original action space is compatible (Box(2,))
        # We assume action[0] is Position, action[1] is TTL.
        
        # Create new 1D action space
        # Assuming the ranges for pos and ttl were [-1, 1] and [0, 1] respectively,
        # or similar. We copy the range of the FIRST component.
        low = env.action_space.low[0]
        high = env.action_space.high[0]
        
        self.action_space = spaces.Box(
            low=np.array([low], dtype=np.float32),
            high=np.array([high], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Takes the 1D action from the agent and appends the fixed TTL.
        """
        # Ensure action is scalar or 1D array
        if np.ndim(action) == 0:
             pos_action = float(action)
        else:
             pos_action = float(action[0])
             
        # Construct 2D action: [position_target, ttl]
        return np.array([pos_action, self.fixed_ttl], dtype=np.float32)
