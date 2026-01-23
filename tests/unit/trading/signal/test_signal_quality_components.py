"""
Phase 3-1: シグナル品質向上 - 構造化単体テスト

各コンポーネントの単体テストを構造化して実装します。
共通基盤、テストデータファクトリ、ユーティリティを提供します。
"""

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from ztb.analysis.signal_quality.signal_quality_analyzer import (
    SignalQualityAnalyzer,
    SignalQualityMetrics,
)
from ztb.trading.backtest.frameworks.multitimeframe_validator import (
    MultiTimeFrameSignal,
    MultiTimeFrameValidator,
    TimeFrame,
)
from ztb.trading.risk.optimizers.integrated_signal_filter import (
    IntegratedSignalFilter,
    SignalQuality,
)
from ztb.trading.risk.optimizers.price_action_filter import (
    PriceActionAnalysisResult,
    PriceActionFilter,
    PriceActionPattern,
)
from ztb.trading.risk.optimizers.volume_filter import (
    VolumeAnalysisResult,
    VolumeFilter,
    VolumePattern,
)
from ztb.trading.strategies.risk_management.confidence_scoring_engine import (
    ConfidenceScore,
    ConfidenceScoringEngine,
)

# ===== テスト基盤クラス =====


class TestDataFactory:
    """テストデータ生成ファクトリ - 構造化されたテストデータ生成"""

    @staticmethod
    def create_sample_signals(
        count: int = 3, base_time: pd.Timestamp = None
    ) -> List[Dict[str, Any]]:
        """サンプルシグナルデータを生成"""
        if base_time is None:
            base_time = pd.Timestamp("2023-01-01 10:00:00")

        signals = []
        actions = ["buy", "sell", "hold"]
        signal_types = [
            "bullish_pattern",
            "bearish_pattern",
            "oscillator_signal",
            "harmonic_pattern",
        ]

        np.random.seed(42)  # 再現性のため

        for i in range(count):
            signals.append(
                {
                    "action": np.random.choice(actions),
                    "confidence": np.random.uniform(0.5, 0.95),
                    "timestamp": base_time + timedelta(hours=i),
                    "signal_type": np.random.choice(signal_types),
                    "strength": np.random.uniform(0.6, 0.95),
                    "direction": np.random.uniform(-1, 1),
                }
            )

        return signals

    @staticmethod
    def create_market_data(
        periods: int = 100, freq: str = "1h", start_price: float = 100000
    ) -> pd.DataFrame:
        """市場データを生成（ランダムウォーク）"""
        dates = pd.date_range("2023-01-01", periods=periods, freq=freq)

        np.random.seed(42)
        # 幾何ブラウン運動で価格生成
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = start_price * np.exp(np.cumsum(returns))

        # 高値・安値・始値の生成
        high_noise = np.random.uniform(0.001, 0.005, periods)
        low_noise = np.random.uniform(0.001, 0.005, periods)

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.002, periods)),
                "high": prices * (1 + high_noise),
                "low": prices * (1 - low_noise),
                "close": prices,
                "volume": np.random.lognormal(10, 1, periods),
            },
            index=dates,
        )

    @staticmethod
    def create_timeframe_data(timeframe: TimeFrame, periods: int = 50) -> pd.DataFrame:
        """特定の時間軸のデータを生成"""
        freq_map = {
            TimeFrame.M1: "1min",
            TimeFrame.M5: "5min",
            TimeFrame.M15: "15min",
            TimeFrame.M30: "30min",
            TimeFrame.H1: "1h",
            TimeFrame.H4: "4h",
            TimeFrame.D1: "1d",
            TimeFrame.W1: "1w",
        }

        return TestDataFactory.create_market_data(periods, freq_map[timeframe])

    @staticmethod
    def create_invalid_signals() -> List[Dict[str, Any]]:
        """無効なシグナルデータを生成（エラーハンドリングテスト用）"""
        return [
            {},  # 空のシグナル
            {"action": "invalid_action"},  # 無効なアクション
            {"action": "buy", "confidence": 1.5},  # 範囲外のコンフィデンス
            {"action": "buy", "confidence": -0.1},  # 負のコンフィデンス
            {"action": "buy", "timestamp": "invalid_date"},  # 無効なタイムスタンプ
            {"action": "buy", "strength": "invalid"},  # 無効な強度
        ]

    @staticmethod
    def create_edge_case_signals() -> List[Dict[str, Any]]:
        """エッジケースのシグナルデータを生成"""
        return [
            {"action": "buy", "confidence": 0.0},  # 最小コンフィデンス
            {"action": "buy", "confidence": 1.0},  # 最大コンフィデンス
            {"action": "hold", "confidence": 0.5},  # ホールドシグナル
            {"action": "sell", "confidence": 0.99, "strength": 1.0},  # 高品質シグナル
        ]


class TestUtilities:
    """テストユーティリティ - 共通の検証ロジック"""

    @staticmethod
    def assert_signal_quality_metrics(metrics: SignalQualityMetrics):
        """シグナル品質メトリクスの検証"""
        assert isinstance(metrics, SignalQualityMetrics)
        assert hasattr(metrics, "total_signals")
        assert hasattr(metrics, "precision")
        assert hasattr(metrics, "recall")
        assert hasattr(metrics, "f1_score")
        assert metrics.total_signals >= 0
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1

    @staticmethod
    def assert_confidence_score(score: ConfidenceScore):
        """コンフィデンススコアの検証"""
        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.base_score <= 1
        assert 0 <= score.market_alignment <= 1
        assert 0 <= score.volume_confirmation <= 1
        assert 0 <= score.timeframe_consistency <= 1
        assert 0 <= score.volatility_adaptation <= 1
        assert 0 <= score.total_score <= 1

    @staticmethod
    def assert_multitimeframe_signal(signal: MultiTimeFrameSignal):
        """マルチタイムフレームシグナルの検証"""
        assert isinstance(signal, MultiTimeFrameSignal)
        assert isinstance(signal.primary_timeframe, TimeFrame)
        assert isinstance(signal.aligned_timeframes, list)
        assert 0 <= signal.consistency_score <= 1
        assert 0 <= signal.alignment_strength <= 1
        assert signal.timestamp is not None

    @staticmethod
    def assert_volume_analysis(result: VolumeAnalysisResult):
        """出来高分析結果の検証"""
        assert isinstance(result, VolumeAnalysisResult)
        assert isinstance(result.pattern, VolumePattern)
        assert isinstance(result.confirmation_strength, (int, float))
        assert 0 <= result.confirmation_strength <= 1

    @staticmethod
    def assert_price_action_analysis(result: PriceActionAnalysisResult):
        """価格アクション分析結果の検証"""
        assert isinstance(result, PriceActionAnalysisResult)
        assert isinstance(result.pattern, PriceActionPattern)
        assert isinstance(result.strength, (int, float))
        assert 0 <= result.strength <= 1

    @staticmethod
    def assert_signal_quality(quality):
        """シグナル品質の検証 - IntegratedFilterResultに対応"""
        # IntegratedFilterResultの場合
        if hasattr(quality, "overall_quality"):
            assert hasattr(quality, "quality_score")
            assert hasattr(quality, "recommended_action")
            assert 0 <= quality.quality_score <= 1
            assert quality.recommended_action in ["accept", "reject", "review"]
        # SignalQuality Enumの場合
        else:
            assert isinstance(quality, SignalQuality)

    @staticmethod
    def assert_component_has_memory_management(component):
        """コンポーネントのメモリ管理機能を検証"""
        if hasattr(component, "max_history_size"):
            assert component.max_history_size > 0
            assert isinstance(component.max_history_size, int)

    @staticmethod
    def assert_component_has_profiler(component):
        """コンポーネントのプロファイラ機能を検証"""
        assert hasattr(component, "profiler")
        assert component.profiler is not None


class BaseSignalQualityTest(ABC):
    """シグナル品質テストの基底クラス - 共通のテスト構造を提供"""

    @abstractmethod
    def get_component_class(self) -> type:
        """テスト対象のコンポーネントクラスを返す"""
        pass

    @pytest.fixture
    def component(self):
        """テスト対象コンポーネントのフィクスチャ"""
        return self.get_component_class()()

    @pytest.fixture
    def sample_signals(self):
        """サンプルシグナルデータのフィクスチャ"""
        return TestDataFactory.create_sample_signals()

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データのフィクスチャ"""
        return TestDataFactory.create_market_data()

    @pytest.fixture
    def invalid_signals(self):
        """無効なシグナルデータのフィクスチャ"""
        return TestDataFactory.create_invalid_signals()

    @pytest.fixture
    def edge_case_signals(self):
        """エッジケースのシグナルデータのフィクスチャ"""
        return TestDataFactory.create_edge_case_signals()

    def test_initialization(self, component):
        """初期化テスト - 全コンポーネント共通"""
        assert component is not None
        TestUtilities.assert_component_has_profiler(component)
        TestUtilities.assert_component_has_memory_management(component)

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト - 全コンポーネント共通"""
        # 各サブクラスで実装
        pass

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト - 全コンポーネント共通"""
        # 各サブクラスで実装
        pass

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト - 全コンポーネント共通"""
        # 各サブクラスで実装（sample_market_dataを使用可能）
        pass


# ===== 個別コンポーネントテスト =====


class TestSignalQualityAnalyzer(BaseSignalQualityTest):
    """SignalQualityAnalyzerの構造化テスト"""

    def get_component_class(self) -> type:
        return SignalQualityAnalyzer

    def test_evaluate_signal_quality(
        self, component, sample_signals, sample_market_data
    ):
        """シグナル品質評価テスト"""
        result = component.evaluate_signal_quality(sample_signals, sample_market_data)
        TestUtilities.assert_signal_quality_metrics(result)

    def test_empty_signals(self, component, sample_market_data):
        """空のシグナルリストテスト"""
        result = component.evaluate_signal_quality([], sample_market_data)
        TestUtilities.assert_signal_quality_metrics(result)
        assert result.total_signals == 0

    def test_insufficient_data(self, component, sample_signals):
        """不十分なデータテスト"""
        insufficient_data = pd.DataFrame({"close": [100, 101, 102]})
        result = component.evaluate_signal_quality(sample_signals, insufficient_data)
        TestUtilities.assert_signal_quality_metrics(result)

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        result = component.evaluate_signal_quality([], pd.DataFrame())
        TestUtilities.assert_signal_quality_metrics(result)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        for invalid_signal in invalid_signals[:2]:  # 最初の2つの無効シグナルを使用
            result = component.evaluate_signal_quality([invalid_signal], pd.DataFrame())
            TestUtilities.assert_signal_quality_metrics(result)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        for edge_signal in edge_case_signals:
            result = component.evaluate_signal_quality(
                [edge_signal], sample_market_data
            )
            TestUtilities.assert_signal_quality_metrics(result)


class TestConfidenceScoringEngine(BaseSignalQualityTest):
    """ConfidenceScoringEngineの構造化テスト"""

    def get_component_class(self) -> type:
        return ConfidenceScoringEngine

    def test_calculate_confidence_score(
        self, component, sample_signals, sample_market_data
    ):
        """コンフィデンススコア計算テスト"""
        for signal in sample_signals:
            score = component.calculate_confidence_score(signal, sample_market_data)
            TestUtilities.assert_confidence_score(score)

    def test_get_quality_statistics(
        self, component, sample_signals, sample_market_data
    ):
        """品質統計取得テスト"""
        # いくつかのスコアを生成
        for _ in range(3):
            component.calculate_confidence_score(sample_signals[0], sample_market_data)

        stats = component.get_quality_statistics()
        assert isinstance(stats, dict)
        assert "average_confidence" in stats
        assert "total_signals" in stats

    def test_should_accept_signal(self, component, sample_signals, sample_market_data):
        """シグナル受け入れ判定テスト"""
        for signal in sample_signals:
            accepted, reason, score = component.should_accept_signal(
                signal, sample_market_data
            )
            assert isinstance(accepted, bool)
            assert isinstance(reason, str)
            TestUtilities.assert_confidence_score(score)

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        score = component.calculate_confidence_score({}, pd.DataFrame())
        TestUtilities.assert_confidence_score(score)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        for invalid_signal in invalid_signals:
            score = component.calculate_confidence_score(invalid_signal, pd.DataFrame())
            TestUtilities.assert_confidence_score(score)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        for edge_signal in edge_case_signals:
            score = component.calculate_confidence_score(
                edge_signal, sample_market_data
            )
            TestUtilities.assert_confidence_score(score)


class TestMultiTimeFrameValidator(BaseSignalQualityTest):
    """MultiTimeFrameValidatorの構造化テスト"""

    def get_component_class(self) -> type:
        return MultiTimeFrameValidator

    def test_validate_signal_consistency(self, component, sample_signals):
        """シグナル整合性検証テスト"""
        # モックデータを使用
        mock_data = {
            TimeFrame.H1: TestDataFactory.create_timeframe_data(TimeFrame.H1),
            TimeFrame.H4: TestDataFactory.create_timeframe_data(TimeFrame.H4),
        }

        for signal in sample_signals:
            result = component.validate_signal_consistency(signal, mock_data)
            TestUtilities.assert_multitimeframe_signal(result)

    def test_timeframe_hierarchy(self, component):
        """時間軸階層テスト"""
        hierarchy = component.timeframe_hierarchy
        assert isinstance(hierarchy, dict)
        assert TimeFrame.H1 in hierarchy
        assert TimeFrame.H4 in hierarchy

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        result = component.validate_signal_consistency({}, {})
        TestUtilities.assert_multitimeframe_signal(result)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        for invalid_signal in invalid_signals:
            result = component.validate_signal_consistency(invalid_signal, {})
            TestUtilities.assert_multitimeframe_signal(result)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        mock_data = {TimeFrame.H1: TestDataFactory.create_timeframe_data(TimeFrame.H1)}
        for edge_signal in edge_case_signals:
            result = component.validate_signal_consistency(edge_signal, mock_data)
            TestUtilities.assert_multitimeframe_signal(result)


class TestVolumeFilter(BaseSignalQualityTest):
    """VolumeFilterの構造化テスト"""

    def get_component_class(self) -> type:
        return VolumeFilter

    def test_analyze_volume_pattern(
        self, component, sample_signals, sample_market_data
    ):
        """出来高パターン分析テスト"""
        for signal in sample_signals:
            timestamp = signal.get("timestamp", pd.Timestamp("2023-01-01 10:00:00"))
            result = component.analyze_volume_pattern(sample_market_data, timestamp)
            TestUtilities.assert_volume_analysis(result)

    def test_get_volume_statistics(self, component, sample_signals, sample_market_data):
        """出来高統計取得テスト"""
        # いくつかの分析を実行
        for signal in sample_signals:
            timestamp = signal.get("timestamp", pd.Timestamp("2023-01-01 10:00:00"))
            component.analyze_volume_pattern(sample_market_data, timestamp)

        stats = component.get_volume_statistics()
        assert isinstance(stats, dict)

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        empty_data = pd.DataFrame()
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        result = component.analyze_volume_pattern(empty_data, timestamp)
        TestUtilities.assert_volume_analysis(result)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        # VolumeFilterはsignalを直接使わないので、timestampのみテスト
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        result = component.analyze_volume_pattern(pd.DataFrame(), timestamp)
        TestUtilities.assert_volume_analysis(result)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        for edge_signal in edge_case_signals:
            timestamp = edge_signal.get(
                "timestamp", pd.Timestamp("2023-01-01 10:00:00")
            )
            result = component.analyze_volume_pattern(sample_market_data, timestamp)
            TestUtilities.assert_volume_analysis(result)


class TestPriceActionFilter(BaseSignalQualityTest):
    """PriceActionFilterの構造化テスト"""

    def get_component_class(self) -> type:
        return PriceActionFilter

    def test_analyze_price_action(self, component, sample_signals, sample_market_data):
        """価格アクション分析テスト"""
        for signal in sample_signals:
            timestamp = signal.get("timestamp", pd.Timestamp("2023-01-01 10:00:00"))
            result = component.analyze_price_action(sample_market_data, timestamp)
            TestUtilities.assert_price_action_analysis(result)

    def test_get_pattern_statistics(
        self, component, sample_signals, sample_market_data
    ):
        """パターン統計取得テスト"""
        # いくつかの分析を実行
        for signal in sample_signals:
            timestamp = signal.get("timestamp", pd.Timestamp("2023-01-01 10:00:00"))
            component.analyze_price_action(sample_market_data, timestamp)

        stats = component.get_pattern_statistics()
        assert isinstance(stats, dict)

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        empty_data = pd.DataFrame()
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        result = component.analyze_price_action(empty_data, timestamp)
        TestUtilities.assert_price_action_analysis(result)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        # PriceActionFilterはsignalを直接使わないので、timestampのみテスト
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        result = component.analyze_price_action(pd.DataFrame(), timestamp)
        TestUtilities.assert_price_action_analysis(result)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        for edge_signal in edge_case_signals:
            timestamp = edge_signal.get(
                "timestamp", pd.Timestamp("2023-01-01 10:00:00")
            )
            result = component.analyze_price_action(sample_market_data, timestamp)
            TestUtilities.assert_price_action_analysis(result)


class TestIntegratedSignalFilter(BaseSignalQualityTest):
    """IntegratedSignalFilterの構造化テスト"""

    def get_component_class(self) -> type:
        return IntegratedSignalFilter

    def test_evaluate_signal_quality(
        self, component, sample_signals, sample_market_data
    ):
        """シグナル品質評価テスト"""
        for signal in sample_signals:
            quality = component.evaluate_signal_quality(signal, sample_market_data)
            TestUtilities.assert_signal_quality(quality)

    def test_batch_evaluate_signals(
        self, component, sample_signals, sample_market_data
    ):
        """バッチシグナル評価テスト"""
        qualities = component.batch_evaluate_signals(sample_signals, sample_market_data)
        assert isinstance(qualities, list)
        assert len(qualities) == len(sample_signals)
        for quality in qualities:
            TestUtilities.assert_signal_quality(quality)

    def test_get_filter_statistics(self, component, sample_signals, sample_market_data):
        """フィルタ統計取得テスト"""
        # いくつかの評価を実行
        for signal in sample_signals:
            component.evaluate_signal_quality(signal, sample_market_data)

        stats = component.get_filter_statistics()
        assert isinstance(stats, dict)

    def test_update_market_regime(self, component, sample_market_data):
        """市場レジーム更新テスト"""
        market_conditions = {
            "volatility": 0.02,
            "trend_strength": 0.1,
            "volume_trend": "increasing",
        }

        component.update_market_regime(market_conditions)
        # 更新が正常に完了することを確認（例外が発生しない）

    def test_empty_input_handling(self, component):
        """空入力のハンドリングテスト"""
        quality = component.evaluate_signal_quality({}, pd.DataFrame())
        TestUtilities.assert_signal_quality(quality)

    def test_invalid_input_handling(self, component, invalid_signals):
        """無効入力のハンドリングテスト"""
        for invalid_signal in invalid_signals:
            quality = component.evaluate_signal_quality(invalid_signal, pd.DataFrame())
            TestUtilities.assert_signal_quality(quality)

    def test_edge_case_handling(self, component, edge_case_signals, sample_market_data):
        """エッジケースのハンドリングテスト"""
        for edge_signal in edge_case_signals:
            quality = component.evaluate_signal_quality(edge_signal, sample_market_data)
            TestUtilities.assert_signal_quality(quality)


# ===== テスト実行支援 =====

if __name__ == "__main__":
    """テスト実行のエントリーポイント"""
    pytest.main([__file__, "-v", "--tb=short"])
