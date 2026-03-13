"""
Report generators package.
"""

from ztb.reporting.generators.analysis import ReportGenerator as SimpleReportGenerator
from ztb.reporting.generators.analysis_rich import (
    ReportGenerator as AnalysisReportGenerator,
)
from ztb.reporting.generators.backtest import (
    ReportGenerator as BacktestReportGenerator,
)
from ztb.reporting.generators.monitoring import (
    ReportGenerator as MonitoringReportGenerator,
)

__all__ = [
    "AnalysisReportGenerator",
    "BacktestReportGenerator",
    "MonitoringReportGenerator",
    "SimpleReportGenerator",
]
