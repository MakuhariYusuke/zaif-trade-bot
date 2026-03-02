from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ActionParseResult:
    target_pos_fraction: float
    ttl_fraction: float
    action_value: float

@dataclass(frozen=True)
class TTLProcessResult:
    raw_target_position: float
    position_ttl: int
    cooldown_counter: int
    ttl_forced_exit: bool
    cooldown_triggered: bool

class FastIntradayActionProcessor:
    """Parse and gate actions for FastIntradayEnvV456."""

    def __init__(self, action_space_type: str, max_position: float, cooldown_steps: int) -> None:
        self.action_space_type = action_space_type
        self.max_position = float(max_position)
        self.cooldown_steps = int(cooldown_steps)
        self.ttl_enabled = action_space_type != "1d_position"

    def parse_action(self, action: np.ndarray | float | int | list | tuple) -> ActionParseResult:
        action_array = self._coerce_action(action)
        action_value = float(action_array[0]) if action_array.size > 0 else 0.0
        target_pos_fraction = float(np.clip(action_value, -1.0, 1.0))

        if self.action_space_type == "1d_position":
            ttl_fraction = 1.0
        else:
            ttl_raw = float(action_array[1]) if action_array.size > 1 else 1.0
            ttl_fraction = float(np.clip(ttl_raw, 0.0, 1.0))

        return ActionParseResult(
            target_pos_fraction=target_pos_fraction,
            ttl_fraction=ttl_fraction,
            action_value=action_value,
        )

    def apply_ttl_and_cooldown(
        self,
        target_pos_fraction: float,
        position: float,
        position_ttl: int,
        cooldown_counter: int,
    ) -> TTLProcessResult:
        raw_target_position = target_pos_fraction * self.max_position
        ttl_forced_exit = False
        cooldown_triggered = False

        next_position_ttl = int(position_ttl)
        next_cooldown_counter = int(cooldown_counter)

        if self.ttl_enabled and next_position_ttl <= 0 and abs(position) > 1e-6:
            raw_target_position = 0.0
            if next_position_ttl == 0:
                next_cooldown_counter = self.cooldown_steps
                next_position_ttl = -1
                ttl_forced_exit = True
                cooldown_triggered = True

        if next_cooldown_counter > 0:
            raw_target_position = 0.0
            next_cooldown_counter -= 1

        return TTLProcessResult(
            raw_target_position=raw_target_position,
            position_ttl=next_position_ttl,
            cooldown_counter=next_cooldown_counter,
            ttl_forced_exit=ttl_forced_exit,
            cooldown_triggered=cooldown_triggered,
        )

    @staticmethod
    def _coerce_action(action: np.ndarray | float | int | list | tuple) -> np.ndarray:
        if isinstance(action, np.ndarray):
            return action.astype(np.float32).reshape(-1)
        if isinstance(action, (list, tuple)):
            return np.asarray(action, dtype=np.float32).reshape(-1)
        try:
            return np.asarray([float(action)], dtype=np.float32)
        except (TypeError, ValueError):
            return np.asarray([0.0], dtype=np.float32)
