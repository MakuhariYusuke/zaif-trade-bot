"""
Shared interface types for Action Signal Guide interfaces.

Centralizes payload and history aliases used across interface modules to
reduce duplicated annotations and keep contracts consistent.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, TypeAlias

import pandas as pd

from ztb.types.common import ObjectMap, ObjectRecords

# Generic payload aliases used by interface contracts.
PayloadMap: TypeAlias = ObjectMap
PayloadRecords: TypeAlias = ObjectRecords
MetadataMap: TypeAlias = ObjectMap
MetricsMap: TypeAlias = ObjectMap
ConstraintMap: TypeAlias = ObjectMap

# Common data-shape aliases used by interface dataclasses.
FeatureData: TypeAlias = object
TargetData: TypeAlias = object
GenericData: TypeAlias = object
ObjectList: TypeAlias = list[object]
SeriesMap: TypeAlias = dict[str, pd.Series]

# Reusable callback/data-handler signatures.
DataHandler: TypeAlias = Callable[[object], None]


class IActionSignalGuideInterface(ABC):
    """Marker base class for Action Signal Guide interfaces."""

