"""Utility shims."""
from __future__ import annotations

import random


def set_random_seed(seed: int) -> None:
    random.seed(seed)
