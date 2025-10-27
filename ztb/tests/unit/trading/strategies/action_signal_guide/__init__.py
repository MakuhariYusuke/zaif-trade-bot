#!/usr/bin/env python3
"""
最適化されたAction Signal Guide設定

強度分析の結果に基づいて最適化された設定を提供します。
"""

import sys
import os
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import ActionSignalGuideConfig


def get_optimized_config() -> ActionSignalGuideConfig:
    """
    強度分析結果に基づいて最適化された設定を返します。

    分析結果:
    - Fibonacci: 平均強度0.59、高い一貫性
    - Gann: 平均強度0.59、高い一貫性
    - Wave: 平均強度0.63、高い一貫性
    - Oscillator: 平均強度0.72、強いシグナル
    - Bollinger: 平均強度0.40、高い一貫性
    - ADX: 平均強度0.54、高い一貫性、利益相関最高
    - Granville: 平均強度0.72、強いシグナル

    推奨戦略:
    1. 高信頼性パターン (Fibonacci, Gann, Wave, Bollinger, ADX) を優先
    2. 強いシグナルパターン (Oscillator, Granville) を補助的に使用
    3. シグナル生成数の多いパターン (ADX, Wave, Fibonacci) を重視
    """

    # 最適化されたパターン強度設定
    pattern_strengths = {
        'fibonacci': 0.59,  # 高信頼性、一貫性が高い
        'gann': 0.59,       # 高信頼性、一貫性が高い
        'wave': 0.63,       # 高信頼性、一貫性が高く、利益相関良好
        'oscillator': 0.72, # 中信頼性、強いシグナル
        'bollinger': 0.40,  # 高信頼性、一貫性が高い
        'adx': 0.54,        # 高信頼性、利益相関最高、シグナル数最多
        'granville': 0.72,  # 中信頼性、強いシグナル
        # 他のパターンはデフォルト値を使用（シグナル生成が少ないため）
        'candlestick': 1.0,
        'harmonic': 1.0,
        'volume': 1.0,
        'heikin_ashi': 1.0,
        'dow_theory': 1.0,
    }

    # 最適化された設定を作成
    config = ActionSignalGuideConfig(
        # パフォーマンス設定
        max_signals_per_bar=5,  # シグナル数を適度に制限
        enable_parallel_processing=True,  # 並列処理を有効化
        enable_caching=True,     # キャッシュを有効化

        # パターン有効化設定（分析結果に基づく）
        enable_fibonacci_patterns=True,   # 推奨: 良い強度分布
        enable_gann_patterns=True,        # 推奨: 良い強度分布
        enable_wave_patterns=True,        # 推奨: 最も安定したパフォーマンス
        enable_oscillator_patterns=True,  # 推奨: 高い平均強度
        enable_adx_patterns=True,         # 推奨: 利益相関最高
        enable_granville_patterns=True,   # 推奨: 高い平均強度
        enable_bollinger_patterns=True,   # 推奨: 安定した強度

        # 低パフォーマンスパターンは無効化
        enable_candlestick_patterns=False,  # シグナル生成なし
        enable_harmonic_patterns=False,     # シグナル生成なし
        enable_volume_patterns=False,       # シグナル生成なし
        enable_heikin_ashi_patterns=False,  # シグナル生成なし
        enable_dow_theory_patterns=False,   # シグナル生成なし
    )

    # パターンごとの詳細設定を適用
    _apply_pattern_strengths(config, pattern_strengths)

    return config


def _apply_pattern_strengths(config: ActionSignalGuideConfig, strengths: Dict[str, float]) -> None:
    """設定にパターン強度を適用します。"""

    # Fibonacciパターン
    if config.fibonacci_patterns:
        for pattern in config.fibonacci_patterns:
            pattern.weight = strengths.get('fibonacci', 1.0)

    # Gannパターン
    if config.gann_patterns:
        for pattern in config.gann_patterns:
            pattern.weight = strengths.get('gann', 1.0)

    # Waveパターン
    if config.wave_patterns:
        for pattern in config.wave_patterns:
            pattern.weight = strengths.get('wave', 1.0)

    # Oscillatorパターン
    if config.oscillator_patterns:
        for pattern in config.oscillator_patterns:
            pattern.weight = strengths.get('oscillator', 1.0)

    # Bollingerパターン
    if config.bollinger_patterns:
        for pattern in config.bollinger_patterns:
            pattern.weight = strengths.get('bollinger', 1.0)

    # ADXパターン
    if config.adx_patterns:
        for pattern in config.adx_patterns:
            pattern.weight = strengths.get('adx', 1.0)

    # Granvilleパターン
    if config.granville_patterns:
        for pattern in config.granville_patterns:
            pattern.weight = strengths.get('granville', 1.0)


def get_performance_summary() -> Dict[str, Any]:
    """
    強度分析の性能要約を返します。

    主要な発見:
    1. ADX: 最も多くのシグナル (430) と最高の利益相関 (0.106)
    2. Wave: 最も安定した強度分布と良い勝率相関 (0.083)
    3. Oscillator/Granville: 最高の平均強度 (0.72)
    4. Fibonacci/Gann: 良好な強度分布と一貫性
    5. Bollinger: 最も低い強度だが非常に安定
    """

    return {
        'top_performers': {
            'signal_volume': ['adx', 'wave', 'fibonacci'],
            'strength_consistency': ['wave', 'fibonacci', 'gann'],
            'profit_correlation': ['adx', 'wave', 'granville'],
            'win_rate_correlation': ['wave', 'adx', 'oscillator'],
        },
        'recommended_weights': {
            'adx': 0.54,        # 利益相関最高
            'wave': 0.63,       # 最も安定
            'fibonacci': 0.59,  # 良好な一貫性
            'gann': 0.59,       # 良好な一貫性
            'oscillator': 0.72, # 高い強度
            'granville': 0.72,  # 高い強度
            'bollinger': 0.40,  # 安定性重視
        },
        'disabled_patterns': [
            'candlestick',  # シグナルなし
            'harmonic',     # シグナルなし
            'volume',       # シグナルなし
            'heikin_ashi',  # シグナルなし
            'dow_theory',   # シグナルなし
        ],
        'key_insights': [
            'ADXが最も信頼性の高いパフォーマンス指標',
            'Waveパターンが最も安定した強度分布を示す',
            'OscillatorとGranvilleが最も強いシグナルを提供',
            'FibonacciとGannが良好なバランスを提供',
            'Bollingerは保守的な設定に適する',
        ]
    }

