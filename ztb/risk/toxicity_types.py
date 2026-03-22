from __future__ import annotations

import enum
from dataclasses import dataclass


class ToxicityLevel(enum.Enum):
    """Adverse-selection risk tiers."""

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    KILL = "kill"


@dataclass(frozen=True, slots=True)
class ToxicityAssessment:
    """Pure toxicity assessment payload shared across risk modules."""

    level: ToxicityLevel
    score: float
    offset_mult: float
    participation_rate: float
    threshold_used: float
    rolling_mean: float | None


__all__ = ["ToxicityAssessment", "ToxicityLevel"]
