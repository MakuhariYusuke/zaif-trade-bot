import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    GRU-based feature extractor for Recurrent RL.
    Expects input observation to be a flattened sequence of (N_STACK, N_FEATURES).
    Reshapes it to (BATCH, N_STACK, N_FEATURES), passes through GRU,
    and outputs the last hidden state or output.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_stack: int = 60,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        # The observation space is already flattened by VecFrameStack: (N_STACK * N_FEATURES,)
        super().__init__(observation_space, features_dim)

        self.n_stack = n_stack

        # Calculate input features per step
        # observation_space.shape[0] should be n_stack * n_features
        self.input_features = observation_space.shape[0] // n_stack

        if observation_space.shape[0] % n_stack != 0:
            raise ValueError(
                f"Observation shape {observation_space.shape[0]} is not divisible by n_stack {n_stack}"
            )

        self.gru = nn.GRU(
            input_size=self.input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Linear layer to map GRU output to features_dim
        self.linear = nn.Linear(hidden_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (Batch, N_Stack * Input_Features)
        batch_size = observations.shape[0]

        # Reshape to (Batch, N_Stack, Input_Features)
        reshaped = observations.view(batch_size, self.n_stack, self.input_features)

        # GRU forward
        # output: (Batch, Seq_Len, Hidden_Size)
        # hn: (Num_Layers, Batch, Hidden_Size)
        output, hn = self.gru(reshaped)

        # Take the output of the last time step
        last_output = output[:, -1, :]

        # Map to features_dim
        return self.relu(self.linear(last_output))
