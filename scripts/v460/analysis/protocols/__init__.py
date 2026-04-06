"""分析 protocol registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from scripts.v460.analysis.analysis_common import Record


@dataclass(slots=True)
class ProtocolResult:
    text_report: str
    json_payload: dict[str, object]
    warnings: list[str]


RecordList: TypeAlias = list[Record]


class AnalysisProtocol(ABC):
    protocol_name: str
    description: str

    @abstractmethod
    def execute(
        self,
        records: RecordList,
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        raise NotImplementedError


PROTOCOL_REGISTRY: dict[str, type[AnalysisProtocol]] = {}


def register_protocol(cls: type[AnalysisProtocol]) -> type[AnalysisProtocol]:
    PROTOCOL_REGISTRY[cls.protocol_name] = cls
    return cls


from scripts.v460.analysis.protocols.protocol_688 import Protocol688  # noqa: E402,F401
from scripts.v460.analysis.protocols.protocol_695_regime_as import (  # noqa: E402,F401
    Protocol695RegimeAS,
)
from scripts.v460.analysis.protocols.protocol_695_trend5s import (  # noqa: E402,F401
    Protocol695Trend5s,
)
from scripts.v460.analysis.protocols.protocol_708_skip_gate_quality import (  # noqa: E402,F401
    Protocol708SkipGateQuality,
)
