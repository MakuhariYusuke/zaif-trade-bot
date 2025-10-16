"""
Ichimoku Cloud Features Module

This module contains all Ichimoku-related feature computations including:
- Basic Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)
- Extended features (cloud thickness, price-cloud distance, lagging span analysis)
- Cloud expansion and slope analysis
- Momentum confirmation signals
- Sanyaku Kouten (Three Role Reversal Points)
- Time theory extensions
- Value measurement theory
- Wave theory applications
"""

from .ichimoku import Ichimoku
from .ichimoku_ext import (
    calculate_ichimoku_extended,
    compute_ichimoku_chikou,
    compute_ichimoku_cloud_thickness,
    compute_ichimoku_cloud_thickness_d1,
    compute_ichimoku_cloud_thickness_h1,
    compute_ichimoku_cloud_thickness_h4,
    compute_ichimoku_cloud_thickness_m1,
    compute_ichimoku_cloud_thickness_m15,
    compute_ichimoku_cloud_thickness_m5,
    compute_ichimoku_composite_signal,
    compute_ichimoku_composite_signal_d1,
    compute_ichimoku_composite_signal_h1,
    compute_ichimoku_composite_signal_h4,
    compute_ichimoku_composite_signal_m1,
    compute_ichimoku_composite_signal_m15,
    compute_ichimoku_composite_signal_m5,
    compute_ichimoku_kijun,
    compute_ichimoku_price_cloud_distance,
    compute_ichimoku_price_cloud_distance_d1,
    compute_ichimoku_price_cloud_distance_h1,
    compute_ichimoku_price_cloud_distance_h4,
    compute_ichimoku_price_cloud_distance_m1,
    compute_ichimoku_price_cloud_distance_m15,
    compute_ichimoku_price_cloud_distance_m5,
    compute_ichimoku_senkou_a,
    compute_ichimoku_senkou_b,
    compute_ichimoku_tenkan,
    compute_ichimoku_trend,
    compute_ichimoku_trend_d1,
    compute_ichimoku_trend_h1,
    compute_ichimoku_trend_h4,
    compute_ichimoku_trend_m1,
    compute_ichimoku_trend_m15,
    compute_ichimoku_trend_m5,
)
from .ichimoku_cloud_expansion import (
    IchimokuCloudExpansion,
    compute_ichimoku_cloud_expansion,
)
from .ichimoku_cloud_slope import IchimokuCloudSlope, compute_ichimoku_cloud_slope
from .ichimoku_momentum_confirmation import (
    IchimokuMomentumConfirmation,
    compute_ichimoku_momentum_confirmation,
)
from .ichimoku_sanyaku_kouten import (
    IchimokuSanyakuKouten,
    compute_ichimoku_sanyaku_kouten,
)
from .ichimoku_time_theory import IchimokuTimeTheory, compute_ichimoku_time_theory
from .ichimoku_value_measurement import (
    IchimokuValueMeasurement,
    compute_ichimoku_value_measurement,
)
from .ichimoku_wave_theory import IchimokuWaveTheory, compute_ichimoku_wave_theory

__all__ = [
    "Ichimoku",
    "calculate_ichimoku_extended",
    "compute_ichimoku_chikou",
    "compute_ichimoku_cloud_thickness",
    "compute_ichimoku_cloud_thickness_d1",
    "compute_ichimoku_cloud_thickness_h1",
    "compute_ichimoku_cloud_thickness_h4",
    "compute_ichimoku_cloud_thickness_m1",
    "compute_ichimoku_cloud_thickness_m15",
    "compute_ichimoku_cloud_thickness_m5",
    "compute_ichimoku_composite_signal",
    "compute_ichimoku_composite_signal_d1",
    "compute_ichimoku_composite_signal_h1",
    "compute_ichimoku_composite_signal_h4",
    "compute_ichimoku_composite_signal_m1",
    "compute_ichimoku_composite_signal_m15",
    "compute_ichimoku_composite_signal_m5",
    "compute_ichimoku_kijun",
    "compute_ichimoku_price_cloud_distance",
    "compute_ichimoku_price_cloud_distance_d1",
    "compute_ichimoku_price_cloud_distance_h1",
    "compute_ichimoku_price_cloud_distance_h4",
    "compute_ichimoku_price_cloud_distance_m1",
    "compute_ichimoku_price_cloud_distance_m15",
    "compute_ichimoku_price_cloud_distance_m5",
    "compute_ichimoku_senkou_a",
    "compute_ichimoku_senkou_b",
    "compute_ichimoku_tenkan",
    "compute_ichimoku_trend",
    "compute_ichimoku_trend_d1",
    "compute_ichimoku_trend_h1",
    "compute_ichimoku_trend_h4",
    "compute_ichimoku_trend_m1",
    "compute_ichimoku_trend_m15",
    "compute_ichimoku_trend_m5",
    "IchimokuCloudExpansion",
    "compute_ichimoku_cloud_expansion",
    "IchimokuCloudSlope",
    "compute_ichimoku_cloud_slope",
    "IchimokuMomentumConfirmation",
    "compute_ichimoku_momentum_confirmation",
    "IchimokuSanyakuKouten",
    "compute_ichimoku_sanyaku_kouten",
    "IchimokuTimeTheory",
    "compute_ichimoku_time_theory",
    "IchimokuValueMeasurement",
    "compute_ichimoku_value_measurement",
    "IchimokuWaveTheory",
    "compute_ichimoku_wave_theory",
]