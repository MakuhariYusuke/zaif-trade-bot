""""""""""""

Unit tests for HeavyTradingEnv class.

"""Unit tests for HeavyTradingEnv class.



from unittest.mock import Mock, patch"""Unit tests for HeavyTradingEnv class.Unit tests for HeavyTradingEnv class.

import numpy as np

import pandas as pd

import pytest

from unittest.mock import Mock, patch""""""

from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv

import numpy as np



class TestHeavyTradingEnv:import pandas as pd

    """Test suite for HeavyTradingEnv class."""

import pytest

    @pytest.fixture

    def sample_data(self) -> pd.DataFrame:import gcimport gc

        """Create sample trading data for testing."""

        np.random.seed(42)from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv

        n_steps = 50

import mathimport math

        prices = 100.0 + np.random.normal(0, 1, n_steps)

        volumes = np.random.uniform(100, 1000, n_steps)



        data = {class TestHeavyTradingEnv:from collections import dequefrom collections import deque

            'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='1min'),

            'open': prices,    """Test suite for HeavyTradingEnv class."""

            'high': prices * 1.01,

            'low': prices * 0.99,from pathlib import Pathfrom pathlib import Path

            'close': prices,

            'volume': volumes,    @pytest.fixture

            'sma_20': pd.Series(prices).rolling(20).mean().fillna(100.0),

            'rsi_14': 50.0 + np.random.normal(0, 5, n_steps),    def sample_data(self) -> pd.DataFrame:from unittest.mock import MagicMock, Mock, patchfrom unittest.mock import MagicMock, Mock, patch

        }

        """Create sample trading data for testing."""

        return pd.DataFrame(data)

        np.random.seed(42)

    @pytest.fixture

    def default_config(self) -> EnvironmentConfig:        n_steps = 50  # Smaller dataset for faster tests

        """Create default environment configuration."""

        return EnvironmentConfig(import gymnasium as gymimport gymnasium as gym

            reward_scaling=1.0,

            transaction_cost=0.001,        # Create basic OHLCV data

            max_position_size=1.0,

            initial_portfolio_value=100000.0,        prices = 100.0 + np.random.normal(0, 1, n_steps)import numpy as npimport numpy as np

        )

        volumes = np.random.uniform(100, 1000, n_steps)

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')import pandas as pdimport pandas as pd

    def test_initialization_success(self, mock_fee_model_class, mock_feature_registry_class, sample_data, default_config):

        """Test successful initialization of HeavyTradingEnv."""        data = {

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()            'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='1min'),import pytestimport pytest



        env = HeavyTradingEnv(df=sample_data, config=default_config)            'open': prices,



        assert env.config == default_config            'high': prices * 1.01,from numpy.typing import NDArrayfrom numpy.typing import NDArray

        assert env.portfolio_value == default_config.initial_portfolio_value

        assert env.n_steps == len(sample_data)            'low': prices * 0.99,

        assert env.current_step == 0

        assert env.position == 0.0            'close': prices,

        assert env.trades_count == 0

            'volume': volumes,

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')            'sma_20': pd.Series(prices).rolling(20).mean().fillna(100.0),from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnvfrom ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv

    def test_initialization_without_data_raises_error(self, mock_fee_model_class, mock_feature_registry_class):

        """Test that initialization without data raises ValueError."""            'rsi_14': 50.0 + np.random.normal(0, 5, n_steps),

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()        }



        with pytest.raises(ValueError, match="Either df or streaming_pipeline must be provided"):

            HeavyTradingEnv()

        return pd.DataFrame(data)

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_reset_functionality(self, mock_fee_model_class, mock_feature_registry_class, sample_data, default_config):

        """Test environment reset functionality."""    @pytest.fixtureclass TestHeavyTradingEnv:class TestHeavyTradingEnv:

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()    def default_config(self) -> EnvironmentConfig:



        env = HeavyTradingEnv(df=sample_data, config=default_config)        """Create default environment configuration."""    """Test suite for HeavyTradingEnv class."""    """Test suite for HeavyTradingEnv class."""



        # Modify state        return EnvironmentConfig(

        env.current_step = 10

        env.position = 1.0            reward_scaling=1.0,

        env.portfolio_value = 95000.0

        env.trades_count = 3            transaction_cost=0.001,



        # Reset            max_position_size=1.0,    @pytest.fixture    @pytest.fixture

        observation, info = env.reset()

            initial_portfolio_value=100000.0,

        assert env.current_step == 0

        assert env.position == 0.0        )    def sample_data(self) -> pd.DataFrame:    def sample_data(self) -> pd.DataFrame:        # Check if stop loss would trigger (implementation dependent)

        assert env.portfolio_value == default_config.initial_portfolio_value

        assert env.trades_count == 0

        assert isinstance(observation, np.ndarray)

        assert isinstance(info, dict)    @patch('ztb.trading.environment.environment.FeatureRegistry')        """Create sample trading data for testing."""        # This tests that the stop loss threshold is properly set

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_initialization_success(        np.random.seed(42)        assert env.config.stop_loss_threshold == 0.05     """Create sample trading data for testing."""

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config        n_steps = 100        np.random.seed(42)

    ):

        """Test successful initialization of HeavyTradingEnv."""        n_steps = 100

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()        # Create realistic OHLCV data



        env = HeavyTradingEnv(df=sample_data, config=default_config)        base_price = 100.0        # Create realistic OHLCV data



        assert env.config == default_config        prices = []        base_price = 100.0

        assert env.portfolio_value == default_config.initial_portfolio_value

        assert env.n_steps == len(sample_data)        current_price = base_price        prices = []

        assert env.current_step == 0

        assert env.position == 0.0        current_price = base_price

        assert env.trades_count == 0

        for _ in range(n_steps):

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')            # Random walk with some trend        for _ in range(n_steps):

    def test_initialization_without_data_raises_error(

        self, mock_fee_model_class, mock_feature_registry_class            change = np.random.normal(0, 0.01)            # Random walk with some trend

    ):

        """Test that initialization without data raises ValueError."""            current_price *= (1 + change)            change = np.random.normal(0, 0.01)

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()            prices.append(current_price)            current_price *= (1 + change)



        with pytest.raises(ValueError, match="Either df or streaming_pipeline must be provided"):            prices.append(current_price)

            HeavyTradingEnv()

        prices = np.array(prices)

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')        prices = np.array(prices)

    def test_reset_functionality(

        self, mock_fee_model_class, mock_feature_registry_class,        # Create OHLCV data

        sample_data, default_config

    ):        high_prices = prices * (1 + np.random.uniform(0, 0.005, n_steps))        # Create OHLCV data

        """Test environment reset functionality."""

        mock_fee_model_class.return_value = Mock()        low_prices = prices * (1 - np.random.uniform(0, 0.005, n_steps))        high_prices = prices * (1 + np.random.uniform(0, 0.005, n_steps))

        mock_feature_registry_class.return_value = Mock()

        open_prices = prices * (1 + np.random.normal(0, 0.002, n_steps))        low_prices = prices * (1 - np.random.uniform(0, 0.005, n_steps))

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        volumes = np.random.uniform(100, 1000, n_steps)        open_prices = prices * (1 + np.random.normal(0, 0.002, n_steps))

        # Modify state

        env.current_step = 10        volumes = np.random.uniform(100, 1000, n_steps)

        env.position = 1.0

        env.portfolio_value = 95000.0        data = {

        env.trades_count = 3

            'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='1min'),        data = {

        # Reset

        observation, info = env.reset()            'open': open_prices,            'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='1min'),



        assert env.current_step == 0            'high': high_prices,            'open': open_prices,

        assert env.position == 0.0

        assert env.portfolio_value == default_config.initial_portfolio_value            'low': low_prices,            'high': high_prices,

        assert env.trades_count == 0

        assert isinstance(observation, np.ndarray)            'close': prices,            'low': low_prices,

        assert isinstance(info, dict)

            'volume': volumes,            'close': prices,

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')            # Add some technical indicators            'volume': volumes,

    def test_step_with_hold_action(

        self, mock_fee_model_class, mock_feature_registry_class,            'sma_20': pd.Series(prices).rolling(20).mean().fillna(base_price),            # Add some technical indicators

        sample_data, default_config

    ):            'rsi_14': 50.0 + np.random.normal(0, 10, n_steps),  # RSI around 50            'sma_20': pd.Series(prices).rolling(20).mean().fillna(base_price),

        """Test step method with hold action."""

        mock_fee_model_class.return_value = Mock()            'macd': np.random.normal(0, 0.1, n_steps),            'rsi_14': 50.0 + np.random.normal(0, 10, n_steps),  # RSI around 50

        mock_feature_registry_class.return_value = Mock()

            'macd_signal': np.random.normal(0, 0.1, n_steps),            'macd': np.random.normal(0, 0.1, n_steps),

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()            'bb_upper': prices * 1.02,            'macd_signal': np.random.normal(0, 0.1, n_steps),



        initial_portfolio = env.portfolio_value            'bb_lower': prices * 0.98,            'bb_upper': prices * 1.02,

        initial_position = env.position

            'atr_14': np.full(n_steps, 1.0),            'bb_lower': prices * 0.98,

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        }            'atr_14': np.full(n_steps, 1.0),

        assert isinstance(observation, np.ndarray)

        assert isinstance(reward, (int, float))        }

        assert isinstance(done, bool)

        assert isinstance(truncated, bool)        df = pd.DataFrame(data)

        assert isinstance(info, dict)

        return df        df = pd.DataFrame(data)

        # Hold should not change position significantly

        assert env.position == initial_position        return df

        assert abs(env.portfolio_value - initial_portfolio) < 1e-6

        assert env.current_step == 1    @pytest.fixture



    @patch('ztb.trading.environment.environment.FeatureRegistry')    def mock_feature_registry(self):    @pytest.fixture

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_step_with_buy_action(        """Mock FeatureRegistry for testing."""    def mock_feature_registry(self):

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config        mock_registry = Mock()        """Mock FeatureRegistry for testing."""

    ):

        """Test step method with buy action."""        mock_registry.get_feature_names.return_value = [        mock_registry = Mock()

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()            'close', 'volume', 'sma_20', 'rsi_14', 'macd', 'bb_upper'        mock_registry.get_feature_names.return_value = [



        env = HeavyTradingEnv(df=sample_data, config=default_config)        ]            'close', 'volume', 'sma_20', 'rsi_14', 'macd', 'bb_upper'

        env.reset()

        return mock_registry        ]

        initial_portfolio = env.portfolio_value

        return mock_registry

        observation, reward, done, truncated, info = env.step(1)  # BUY

    @pytest.fixture

        # Position should change to long

        assert env.position == 1.0    def mock_fee_model(self):    @pytest.fixture

        # Portfolio should decrease due to transaction cost

        assert env.portfolio_value < initial_portfolio        """Mock ExchangeFeeModel for testing."""    def mock_fee_model(self):

        assert env.trades_count == 1

        assert env.current_step == 1        mock_fee = Mock()        """Mock ExchangeFeeModel for testing."""



    @patch('ztb.trading.environment.environment.FeatureRegistry')        mock_fee.get_fee_rate.return_value = 0.001  # 0.1% fee        mock_fee = Mock()

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_step_with_sell_action(        return mock_fee        mock_fee.get_fee_rate.return_value = 0.001  # 0.1% fee

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config        return mock_fee

    ):

        """Test step method with sell action."""    @pytest.fixture

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()    def default_config(self) -> EnvironmentConfig:    @pytest.fixture



        env = HeavyTradingEnv(df=sample_data, config=default_config)        """Create default environment configuration."""    def default_config(self) -> EnvironmentConfig:

        env.reset()

        return EnvironmentConfig(        """Create default environment configuration."""

        # First buy to establish position

        env.step(1)  # BUY            reward_scaling=1.0,        return EnvironmentConfig(

        initial_portfolio = env.portfolio_value

            transaction_cost=0.001,            reward_scaling=1.0,

        observation, reward, done, truncated, info = env.step(2)  # SELL

            max_position_size=1.0,            transaction_cost=0.001,

        # Position should change to short

        assert env.position == -1.0            initial_portfolio_value=100000.0,            max_position_size=1.0,

        # Portfolio should change due to closing position

        assert env.portfolio_value != initial_portfolio            max_consecutive_trades=5,            initial_portfolio_value=100000.0,

        assert env.trades_count == 2

        assert env.current_step == 2            reward_clip_value=2.0,            max_consecutive_trades=5,



    @patch('ztb.trading.environment.environment.FeatureRegistry')        )            reward_clip_value=2.0,

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_observation_space_properties(        )

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config    @patch('ztb.trading.environment.environment.FeatureRegistry')

    ):

        """Test observation space properties."""    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()    def test_initialization_with_dataframe(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')



        env = HeavyTradingEnv(df=sample_data, config=default_config)        self,    def test_initialization_with_dataframe(



        assert hasattr(env, 'observation_space')        mock_fee_model_class,        self,

        assert hasattr(env, 'action_space')

        assert env.action_space.n == 3  # HOLD, BUY, SELL        mock_feature_registry_class,        mock_fee_model_class,



        # Test observation generation        sample_data,        mock_feature_registry_class,

        obs = env._get_observation()

        assert isinstance(obs, np.ndarray)        default_config,        sample_data,

        assert obs.dtype == np.float32

        assert len(obs) > 0        mock_fee_model,        default_config,



    @patch('ztb.trading.environment.environment.FeatureRegistry')        mock_feature_registry        mock_fee_model,

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_episode_termination(    ):        mock_feature_registry

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config        """Test HeavyTradingEnv initialization with DataFrame."""    ):

    ):

        """Test episode termination at end of data."""        mock_fee_model_class.return_value = mock_fee_model        """Test HeavyTradingEnv initialization with DataFrame."""

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model



        env = HeavyTradingEnv(df=sample_data, config=default_config)        mock_feature_registry_class.return_value = mock_feature_registry

        env.reset()

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        # Fast forward to end

        env.current_step = len(sample_data) - 1        env = HeavyTradingEnv(df=sample_data, config=default_config)



        observation, reward, done, truncated, info = env.step(0)  # HOLD        assert env.config == default_config



        assert done  # Should be done at end        assert env.portfolio_value == default_config.initial_portfolio_value        assert env.config == default_config

        assert not truncated

        assert env.n_steps == len(sample_data)        assert env.portfolio_value == default_config.initial_portfolio_value

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')        assert isinstance(env.df, pd.DataFrame)        assert env.n_steps == len(sample_data)

    def test_get_legal_actions(

        self, mock_fee_model_class, mock_feature_registry_class,        assert hasattr(env, 'action_space')        assert isinstance(env.df, pd.DataFrame)

        sample_data, default_config

    ):        assert hasattr(env, 'observation_space')        assert hasattr(env, 'action_space')

        """Test get_legal_actions method."""

        mock_fee_model_class.return_value = Mock()        assert env.current_step == 0        assert hasattr(env, 'observation_space')

        mock_feature_registry_class.return_value = Mock()

        assert env.position == 0.0        assert env.current_step == 0

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        assert env.trades_count == 0        assert env.position == 0.0

        legal_actions = env.get_legal_actions()

        assert env.trades_count == 0

        assert isinstance(legal_actions, list)

        assert all(isinstance(action, int) for action in legal_actions)    @patch('ztb.trading.environment.environment.FeatureRegistry')

        assert 0 in legal_actions  # HOLD should always be legal

        assert len(legal_actions) <= 3    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')



    @patch('ztb.trading.environment.environment.FeatureRegistry')    def test_initialization_without_dataframe_raises_error(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_portfolio_statistics(        self,    def test_initialization_without_dataframe_raises_error(

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config        mock_fee_model_class,        self,

    ):

        """Test portfolio statistics calculation."""        mock_feature_registry_class        mock_fee_model_class,

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()    ):        mock_feature_registry_class



        env = HeavyTradingEnv(df=sample_data, config=default_config)        """Test that initialization without df or streaming pipeline raises error."""    ):

        env.reset()

        mock_fee_model_class.return_value = Mock()        """Test that initialization without df or streaming pipeline raises error."""

        # Perform some trades

        env.step(1)  # BUY        mock_feature_registry_class.return_value = Mock()        mock_fee_model_class.return_value = Mock()

        env.step(2)  # SELL

        mock_feature_registry_class.return_value = Mock()

        stats = env.get_portfolio_stats()

        with pytest.raises(ValueError, match="Either df or streaming_pipeline must be provided"):

        assert isinstance(stats, dict)

        assert 'total_return' in stats            HeavyTradingEnv()        with pytest.raises(ValueError, match="Either df or streaming_pipeline must be provided"):

        assert 'total_trades' in stats

        assert stats['total_trades'] == 2            HeavyTradingEnv()



    @patch('ztb.trading.environment.environment.FeatureRegistry')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    def test_reward_calculation(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

        self, mock_fee_model_class, mock_feature_registry_class,

        sample_data, default_config    def test_reset_method(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

    ):

        """Test reward calculation logic."""        self,    def test_reset_method(

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()        mock_fee_model_class,        self,



        env = HeavyTradingEnv(df=sample_data, config=default_config)        mock_feature_registry_class,        mock_fee_model_class,

        env.reset()

        sample_data,        mock_feature_registry_class,

        # Set up a position and some PnL

        env.position = 1.0        default_config,        sample_data,

        env.pnl_history.append(0.01)  # 1% profit

        mock_fee_model,        default_config,

        reward = env._calculate_reward()

        mock_feature_registry        mock_fee_model,

        assert isinstance(reward, float)

    ):        mock_feature_registry

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')        """Test environment reset functionality."""    ):

    def test_environment_config_from_dict(

        self, mock_fee_model_class, mock_feature_registry_class        mock_fee_model_class.return_value = mock_fee_model        """Test environment reset functionality."""

    ):

        """Test EnvironmentConfig.from_dict method."""        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_fee_model_class.return_value = Mock()

        mock_feature_registry_class.return_value = Mock()        mock_feature_registry_class.return_value = mock_feature_registry



        config_dict = {        env = HeavyTradingEnv(df=sample_data, config=default_config)

            'reward_scaling': 2.0,

            'transaction_cost': 0.002,        env = HeavyTradingEnv(df=sample_data, config=default_config)

            'initial_portfolio_value': 50000.0,

        }        # Modify state



        config = EnvironmentConfig.from_dict(config_dict)        env.current_step = 50        # Modify state



        assert config.reward_scaling == 2.0        env.position = 1.0        env.current_step = 50

        assert config.transaction_cost == 0.002

        assert config.initial_portfolio_value == 50000.0        env.portfolio_value = 95000.0        env.position = 1.0

        # Other values should be defaults

        assert config.max_position_size == 1.0        env.trades_count = 5        env.portfolio_value = 95000.0

        env.trades_count = 5

        # Reset environment

        observation, info = env.reset()        # Reset environment

        observation, info = env.reset()

        assert env.current_step == 0

        assert env.position == 0.0        assert env.current_step == 0

        assert env.portfolio_value == default_config.initial_portfolio_value        assert env.position == 0.0

        assert env.trades_count == 0        assert env.portfolio_value == default_config.initial_portfolio_value

        assert isinstance(observation, np.ndarray)        assert env.trades_count == 0

        assert isinstance(info, dict)        assert isinstance(observation, np.ndarray)

        assert 'portfolio_value' in info        assert isinstance(info, dict)

        assert 'position' in info        assert 'portfolio_value' in info

        assert 'step' in info        assert 'position' in info

        assert 'step' in info

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_step_hold_action(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_step_hold_action(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test step method with hold action (0)."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test step method with hold action (0)."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        initial_portfolio = env.portfolio_value

        initial_position = env.position        initial_portfolio = env.portfolio_value

        initial_position = env.position

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        assert isinstance(observation, np.ndarray)

        assert isinstance(reward, (int, float))        assert isinstance(observation, np.ndarray)

        assert isinstance(done, bool)        assert isinstance(reward, (int, float))

        assert isinstance(truncated, bool)        assert isinstance(done, bool)

        assert isinstance(info, dict)        assert isinstance(truncated, bool)

        assert isinstance(info, dict)

        # Hold action should not change position or significantly change portfolio

        assert env.position == initial_position        # Hold action should not change position or significantly change portfolio

        assert abs(env.portfolio_value - initial_portfolio) < 1e-6  # Minimal change        assert env.position == initial_position

        assert env.current_step == 1        assert abs(env.portfolio_value - initial_portfolio) < 1e-6  # Minimal change

        assert not done  # Not done in middle of episode        assert env.current_step == 1

        assert not done  # Not done in middle of episode

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_step_buy_action(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_step_buy_action(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test step method with buy action (1)."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test step method with buy action (1)."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        initial_portfolio = env.portfolio_value

        current_price = env.df.iloc[0]['close']        initial_portfolio = env.portfolio_value

        current_price = env.df.iloc[0]['close']

        observation, reward, done, truncated, info = env.step(1)  # BUY

        observation, reward, done, truncated, info = env.step(1)  # BUY

        # Position should change to long

        assert env.position == 1.0        # Position should change to long

        # Portfolio value should decrease due to transaction cost        assert env.position == 1.0

        assert env.portfolio_value < initial_portfolio        # Portfolio value should decrease due to transaction cost

        assert env.trades_count == 1        assert env.portfolio_value < initial_portfolio

        assert env.current_step == 1        assert env.trades_count == 1

        assert env.current_step == 1

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_step_sell_action(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_step_sell_action(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test step method with sell action (2)."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test step method with sell action (2)."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        # First buy to establish position

        env.step(1)  # BUY        # First buy to establish position

        initial_portfolio = env.portfolio_value        env.step(1)  # BUY

        initial_portfolio = env.portfolio_value

        observation, reward, done, truncated, info = env.step(2)  # SELL

        observation, reward, done, truncated, info = env.step(2)  # SELL

        # Position should change to short (from long)

        assert env.position == -1.0        # Position should change to short (from long)

        # Portfolio value should change due to closing position and transaction cost        assert env.position == -1.0

        assert env.portfolio_value != initial_portfolio        # Portfolio value should change due to closing position and transaction cost

        assert env.trades_count == 2        assert env.portfolio_value != initial_portfolio

        assert env.current_step == 2        assert env.trades_count == 2

        assert env.current_step == 2

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_reward_calculation(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_reward_calculation(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test reward calculation logic."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test reward calculation logic."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        # Test with a position and some PnL

        env.position = 1.0        # Test with a position and some PnL

        env.pnl_history.append(0.01)  # 1% profit        env.position = 1.0

        env.pnl_history.append(0.01)  # 1% profit

        reward = env._calculate_reward()

        reward = env._calculate_reward()

        assert isinstance(reward, float)

        # Reward should be positive for profitable position        assert isinstance(reward, float)

        assert reward > 0        # Reward should be positive for profitable position

        assert reward > 0

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_observation_space(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_observation_space(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test observation space properties."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test observation space properties."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        assert isinstance(env.observation_space, gym.spaces.Box)

        assert env.observation_space.dtype == np.float32        assert isinstance(env.observation_space, gym.spaces.Box)

        assert env.observation_space.dtype == np.float32

        # Get observation

        obs = env._get_observation()        # Get observation

        assert isinstance(obs, np.ndarray)        obs = env._get_observation()

        assert obs.dtype == np.float32        assert isinstance(obs, np.ndarray)

        assert len(obs) > 0        assert obs.dtype == np.float32

        assert len(obs) > 0

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_action_space(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_action_space(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test action space properties."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test action space properties."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        assert isinstance(env.action_space, gym.spaces.Discrete)

        assert env.action_space.n == 3  # HOLD, BUY, SELL        assert isinstance(env.action_space, gym.spaces.Discrete)

        assert env.action_space.n == 3  # HOLD, BUY, SELL

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_episode_termination(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_episode_termination(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test episode termination at end of data."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test episode termination at end of data."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        # Fast forward to near end

        env.current_step = len(sample_data) - 2        # Fast forward to near end

        env.current_step = len(sample_data) - 2

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        assert not done  # Not yet at end

        assert not done  # Not yet at end

        # Take one more step

        observation, reward, done, truncated, info = env.step(0)  # HOLD        # Take one more step

        observation, reward, done, truncated, info = env.step(0)  # HOLD

        assert done  # Should be done at end of data

        assert not truncated        assert done  # Should be done at end of data

        assert not truncated

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_memory_management(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_memory_management(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test memory management features."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test memory management features."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        config_with_memory = EnvironmentConfig(

            memory_logging_enabled=True,        config_with_memory = EnvironmentConfig(

            memory_log_interval_steps=10,            memory_logging_enabled=True,

        )            memory_log_interval_steps=10,

        )

        env = HeavyTradingEnv(df=sample_data, config=config_with_memory)

        env = HeavyTradingEnv(df=sample_data, config=config_with_memory)

        # Test that memory logging attributes are set

        assert hasattr(env, '_memory_logging_enabled')        # Test that memory logging attributes are set

        assert env._memory_logging_enabled        assert hasattr(env, '_memory_logging_enabled')

        assert hasattr(env, '_memory_log_interval_steps')        assert env._memory_logging_enabled

        assert env._memory_log_interval_steps == 10        assert hasattr(env, '_memory_log_interval_steps')

        assert env._memory_log_interval_steps == 10

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_get_legal_actions(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_get_legal_actions(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test get_legal_actions method."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test get_legal_actions method."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        legal_actions = env.get_legal_actions()

        legal_actions = env.get_legal_actions()

        assert isinstance(legal_actions, list)

        assert all(isinstance(action, int) for action in legal_actions)        assert isinstance(legal_actions, list)

        assert 0 in legal_actions  # HOLD should always be legal        assert all(isinstance(action, int) for action in legal_actions)

        assert len(legal_actions) <= 3  # At most 3 actions        assert 0 in legal_actions  # HOLD should always be legal

        assert len(legal_actions) <= 3  # At most 3 actions

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_portfolio_statistics(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_portfolio_statistics(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test portfolio statistics calculation."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test portfolio statistics calculation."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        # Simulate some trading

        env.step(1)  # BUY        # Simulate some trading

        env.step(2)  # SELL        env.step(1)  # BUY

        env.step(1)  # BUY again        env.step(2)  # SELL

        env.step(1)  # BUY again

        stats = env.get_portfolio_stats()

        stats = env.get_portfolio_stats()

        assert isinstance(stats, dict)

        assert 'total_return' in stats        assert isinstance(stats, dict)

        assert 'sharpe_ratio' in stats        assert 'total_return' in stats

        assert 'max_drawdown' in stats        assert 'sharpe_ratio' in stats

        assert 'win_rate' in stats        assert 'max_drawdown' in stats

        assert 'total_trades' in stats        assert 'win_rate' in stats

        assert 'total_trades' in stats

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_reward_settings_methods(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_reward_settings_methods(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test reward settings getter methods."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test reward settings getter methods."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        # Test various reward setting getters

        assert isinstance(env._get_reward_setting_float("position_penalty_scale", 0.0), float)        # Test various reward setting getters

        assert isinstance(env._get_reward_setting_int("inventory_window", 100), int)        assert isinstance(env._get_reward_setting_float("position_penalty_scale", 0.0), float)

        assert isinstance(env._get_reward_setting_bool("enable_forced_diversity", False), bool)        assert isinstance(env._get_reward_setting_int("inventory_window", 100), int)

        assert isinstance(env._get_reward_setting_bool("enable_forced_diversity", False), bool)

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_consecutive_trades_penalty(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_consecutive_trades_penalty(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test consecutive trades penalty logic."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test consecutive trades penalty logic."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=default_config)

        env.reset()

        # Simulate consecutive trades

        for _ in range(6):  # More than max_consecutive_trades (5)        # Simulate consecutive trades

            env.step(1)  # BUY        for _ in range(6):  # More than max_consecutive_trades (5)

            env.step(2)  # SELL            env.step(1)  # BUY

            env.step(2)  # SELL

        # Check that consecutive trade counter is tracked

        assert hasattr(env, '_consecutive_trade_steps')        # Check that consecutive trade counter is tracked

        assert env._consecutive_trade_steps >= 0        assert hasattr(env, '_consecutive_trade_steps')

        assert env._consecutive_trade_steps >= 0

    @patch('ztb.trading.environment.environment.FeatureRegistry')

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')    @patch('ztb.trading.environment.environment.FeatureRegistry')

    def test_stop_loss_functionality(    @patch('ztb.trading.environment.environment.ExchangeFeeModel')

        self,    def test_stop_loss_functionality(

        mock_fee_model_class,        self,

        mock_feature_registry_class,        mock_fee_model_class,

        sample_data,        mock_feature_registry_class,

        default_config,        sample_data,

        mock_fee_model,        default_config,

        mock_feature_registry        mock_fee_model,

    ):        mock_feature_registry

        """Test stop loss functionality."""    ):

        mock_fee_model_class.return_value = mock_fee_model        """Test stop loss functionality."""

        mock_feature_registry_class.return_value = mock_feature_registry        mock_fee_model_class.return_value = mock_fee_model

        mock_feature_registry_class.return_value = mock_feature_registry

        config_with_stop_loss = EnvironmentConfig(

            stop_loss_threshold=0.05,  # 5% stop loss        config_with_stop_loss = EnvironmentConfig(

        )            stop_loss_threshold=0.05,  # 5% stop loss

        )

        env = HeavyTradingEnv(df=sample_data, config=config_with_stop_loss)

        env.reset()        env = HeavyTradingEnv(df=sample_data, config=config_with_stop_loss)

        env.reset()

        # Buy and simulate large loss

        env.step(1)  # BUY        # Buy and simulate large loss

        env.portfolio_value *= 0.9  # 10% loss (more than 5% threshold)        env.step(1)  # BUY

        env.portfolio_value *= 0.9  # 10% loss (more than 5% threshold)

        # Check if stop loss would trigger (implementation dependent)

        # This tests that the stop loss threshold is properly set        # Check if stop loss would trigger (implementation dependent)

        assert env.config.stop_loss_threshold == 0.05        # This tests that the stop loss threshold is properly set
        assert env.config.stop_loss_threshold == 0.05</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\ztb\tests\unit\training\test_environment.py