from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_tiny_teacher_model(
    *,
    input_dim: int = 10,
    hidden_dim: int = 8,
    output_dim: int = 3,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def make_tiny_distillation_loader(
    *,
    rows: int = 8,
    input_dim: int = 10,
    output_dim: int = 3,
    batch_size: int = 4,
) -> DataLoader:
    x = torch.randn(rows, input_dim)
    y = torch.randint(0, output_dim, (rows,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)
