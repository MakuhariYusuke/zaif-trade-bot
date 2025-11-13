"""
Unit tests for signal common base classes

Tests the shared base classes and interfaces used across
all signal processing components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ztb.trading.signal.common.base_classes import (
    BaseSignalProcessor, SignalContext, SignalResult
)


class TestSignalContext:
    """Test SignalContext dataclass"""

    def test_signal_context_creation(self):
        """Test SignalContext creation with valid data"""
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        context = SignalContext(
            market_data=market_data,
            position_context={'size': 1.0, 'entry_price': 100.0},
            portfolio_state={'cash': 5000.0, 'total_value': 15000.0},
            timestamp=pd.Timestamp('2024-01-01 12:00:00')
        )

        assert context.market_data.equals(market_data)
        assert context.position_context['size'] == 1.0
        assert context.portfolio_state['cash'] == 5000.0
        assert context.timestamp == pd.Timestamp('2024-01-01 12:00:00')

    def test_signal_context_immutable(self):
        """Test SignalContext is immutable (dataclass frozen behavior)"""
        context = SignalContext(
            market_data=pd.DataFrame({'close': [100]}),
            position_context={},
            portfolio_state={},
            timestamp=pd.Timestamp.now()
        )

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            context.market_data = pd.DataFrame({'close': [200]})


class TestSignalResult:
    """Test SignalResult dataclass"""

    def test_signal_result_creation(self):
        """Test SignalResult creation"""
        result = SignalResult(
            discrete_action=1,
            quality_score=75.5,
            confidence=0.85,
            metadata={'regime': 'bull_trend', 'indicators': {'rsi': 65}}
        )

        assert result.discrete_action == 1
        assert result.quality_score == 75.5
        assert result.confidence == 0.85
        assert result.metadata['regime'] == 'bull_trend'
        assert result.metadata['indicators']['rsi'] == 65

    def test_signal_result_default_metadata(self):
        """Test SignalResult with default empty metadata"""
        result = SignalResult(
            discrete_action=0,
            quality_score=50.0,
            confidence=0.5
        )

        assert result.metadata == {}


class TestBaseSignalProcessor:
    """Test BaseSignalProcessor abstract base class"""

    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = {'test_param': 'value', 'threshold': 0.8}

        processor = BaseSignalProcessor(config)

        assert processor.config == config
        assert hasattr(processor, 'logger')

    def test_initialization_without_config(self):
        """Test initialization without config uses defaults"""
        processor = BaseSignalProcessor()

        assert isinstance(processor.config, dict)
        assert hasattr(processor, 'logger')

    def test_validate_input_success(self):
        """Test validate_input with valid context"""
        processor = BaseSignalProcessor()

        context = SignalContext(
            market_data=pd.DataFrame({'close': [100]}),
            position_context={},
            portfolio_state={},
            timestamp=pd.Timestamp.now()
        )

        assert processor.validate_input(context) == True

    def test_validate_input_missing_market_data(self):
        """Test validate_input with missing market_data"""
        processor = BaseSignalProcessor()

        context = Mock()
        del context.market_data  # Remove required attribute

        assert processor.validate_input(context) == False

    def test_validate_input_missing_position_context(self):
        """Test validate_input with missing position_context"""
        processor = BaseSignalProcessor()

        context = Mock()
        context.market_data = pd.DataFrame({'close': [100]})
        del context.position_context

        assert processor.validate_input(context) == False

    def test_validate_input_missing_portfolio_state(self):
        """Test validate_input with missing portfolio_state"""
        processor = BaseSignalProcessor()

        context = Mock()
        context.market_data = pd.DataFrame({'close': [100]})
        context.position_context = {}
        del context.portfolio_state

        assert processor.validate_input(context) == False

    def test_log_processing_result(self):
        """Test log_processing_result method"""
        processor = BaseSignalProcessor()

        context = SignalContext(
            market_data=pd.DataFrame({'close': [100]}),
            position_context={},
            portfolio_state={},
            timestamp=pd.Timestamp.now()
        )

        result = SignalResult(
            discrete_action=1,
            quality_score=75.0,
            confidence=0.8,
            metadata={'test': 'value'}
        )

        # Should not raise exception
        processor.log_processing_result(result, context)

    @pytest.mark.parametrize("config_input,expected_config", [
        (None, {}),
        ({}, {}),
        ({'param': 'value'}, {'param': 'value'}),
        ({'nested': {'key': 'value'}}, {'nested': {'key': 'value'}})
    ])
    def test_config_handling(self, config_input, expected_config):
        """Test various config input scenarios"""
        processor = BaseSignalProcessor(config_input)

        if config_input is None:
            # Should get default empty config
            assert processor.config == {}
        else:
            assert processor.config == expected_config

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        processor = BaseSignalProcessor()

        context = SignalContext(
            market_data=pd.DataFrame({'close': [100]}),
            position_context={},
            portfolio_state={},
            timestamp=pd.Timestamp.now()
        )

        with pytest.raises(NotImplementedError):
            processor._get_default_config()

        with pytest.raises(NotImplementedError):
            processor.process_signal(context)

    def test_logger_initialization(self):
        """Test logger is properly initialized"""
        processor = BaseSignalProcessor()

        assert hasattr(processor, 'logger')
        assert processor.logger is not None

        # Logger name should be based on class name
        assert 'BaseSignalProcessor' in str(processor.logger)


class MockSignalProcessor(BaseSignalProcessor):
    """Mock implementation for testing"""

    def _get_default_config(self):
        return {'default_param': 'default_value'}

    def process_signal(self, context):
        return SignalResult(
            discrete_action=0,
            quality_score=50.0,
            confidence=0.5,
            metadata={'processed': True}
        )


class TestMockSignalProcessor:
    """Test the mock implementation"""

    def test_mock_implementation(self):
        """Test mock processor works correctly"""
        processor = MockSignalProcessor()

        assert processor.config['default_param'] == 'default_value'

        context = SignalContext(
            market_data=pd.DataFrame({'close': [100]}),
            position_context={},
            portfolio_state={},
            timestamp=pd.Timestamp.now()
        )

        result = processor.process_signal(context)

        assert result.discrete_action == 0
        assert result.quality_score == 50.0
        assert result.confidence == 0.5
        assert result.metadata['processed'] == True

    def test_mock_with_custom_config(self):
        """Test mock processor with custom config"""
        config = {'custom_param': 'custom_value', 'default_param': 'overridden'}
        processor = MockSignalProcessor(config)

        assert processor.config['custom_param'] == 'custom_value'
        assert processor.config['default_param'] == 'overridden'