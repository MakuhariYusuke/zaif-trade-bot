Configuration Guide
===================

This section covers all configuration options available in Zaif Trade Bot.

Configuration Overview
-----------------------

Zaif Trade Bot uses YAML configuration files to control all aspects of trading, training, and evaluation. Configurations are organized into logical sections for easy management.

Configuration File Structure
-----------------------------

.. code-block:: yaml

   # Main configuration structure
   trading:        # Trading parameters
   environment:    # Environment settings
   training:       # Training parameters
   evaluation:     # Evaluation settings
   logging:        # Logging configuration
   data:          # Data settings

Trading Configuration
---------------------

Basic Trading Settings
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   trading:
     symbol: "BTC/JPY"          # Trading pair symbol
     initial_balance: 1000000   # Starting balance in base currency
     position_size: 0.1         # Default position size (0.1 = 10%)
     max_position_size: 0.5     # Maximum position size limit
     min_position_size: 0.01    # Minimum position size
     leverage: 1.0             # Leverage multiplier (1.0 = no leverage)

Risk Management
~~~~~~~~~~~~~~~

.. code-block:: yaml

   trading:
     stop_loss: 0.05           # Stop loss percentage (5%)
     take_profit: 0.10         # Take profit percentage (10%)
     max_drawdown: 0.20        # Maximum drawdown limit (20%)
     risk_per_trade: 0.02      # Risk per trade (2% of balance)
     max_open_positions: 5     # Maximum number of open positions

Order Settings
~~~~~~~~~~~~~~

.. code-block:: yaml

   trading:
     order_type: "market"      # Order type: market, limit, stop
     slippage_tolerance: 0.001 # Maximum slippage tolerance (0.1%)
     partial_fill_allowed: true # Allow partial order fills

Environment Configuration
-------------------------

Market Environment
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   environment:
     transaction_cost: 0.0005  # Trading fee (0.05%)
     slippage: 0.0001          # Market slippage
     spread: 0.0002           # Bid-ask spread
     market_impact: 0.0001    # Market impact for large orders

Time Settings
~~~~~~~~~~~~~

.. code-block:: yaml

   environment:
     timezone: "Asia/Tokyo"    # Timezone for trading
     trading_hours:           # Trading hours (optional)
       start: "09:00"
       end: "15:00"
     max_steps: 1000          # Maximum steps per episode

Reward Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   environment:
     reward:
       profit_multiplier: 1.0     # Profit reward scaling
       loss_penalty: 1.0          # Loss penalty scaling
       holding_penalty: 0.01      # Penalty for holding positions
       transaction_penalty: 0.001 # Penalty for transactions
       time_penalty: 0.0001       # Time-based penalty

Training Configuration
----------------------

Algorithm Selection
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     algorithm: "PPO"         # Algorithm: PPO, SAC, A2C, DDPG
     total_timesteps: 1000000 # Total training timesteps
     learning_rate: 0.0003    # Learning rate

PPO Specific Settings
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     # PPO parameters
     n_epochs: 10             # Number of epochs per update
     gamma: 0.99              # Discount factor
     gae_lambda: 0.95         # GAE lambda parameter
     clip_range: 0.2          # Clipping parameter
     ent_coef: 0.01           # Entropy coefficient
     vf_coef: 0.5             # Value function coefficient
     max_grad_norm: 0.5       # Maximum gradient norm

SAC Specific Settings
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     # SAC parameters
     buffer_size: 1000000     # Replay buffer size
     learning_starts: 1000    # Steps before learning starts
     batch_size: 256          # Batch size
     tau: 0.005               # Target smoothing coefficient
     ent_coef: "auto"         # Entropy coefficient (auto or float)
     target_entropy: "auto"   # Target entropy

Network Architecture
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     policy_kwargs:
       net_arch: [256, 256]   # Hidden layer sizes
       activation_fn: "relu"  # Activation function

     # Feature extraction
     features_extractor_kwargs:
       net_arch: [128, 64]    # Feature extractor layers

Optimization Settings
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     optimizer: "adam"        # Optimizer: adam, sgd, rmsprop
     weight_decay: 0.0        # Weight decay (L2 regularization)
     use_scheduler: false     # Use learning rate scheduler

     scheduler_kwargs:
       scheduler: "linear"    # Scheduler type
       total_steps: 1000000   # Total scheduler steps

Data Configuration
------------------

Data Sources
~~~~~~~~~~~~

.. code-block:: yaml

   data:
     source: "csv"           # Data source: csv, database, api
     path: "data/market_data.csv"  # Data file path
     format: "ohlcv"         # Data format: ohlcv, tick

Data Preprocessing
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     preprocessing:
       normalize: true        # Normalize price data
       add_features: true     # Add technical indicators
       handle_missing: "drop" # Missing data handling: drop, fill, interpolate
       outlier_removal: true  # Remove outliers

     features:
       - "rsi"               # Relative Strength Index
       - "macd"              # MACD
       - "bbands"            # Bollinger Bands
       - "sma"               # Simple Moving Average
       - "ema"               # Exponential Moving Average

Training/Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     split:
       train_ratio: 0.7      # Training data ratio
       val_ratio: 0.2        # Validation data ratio
       test_ratio: 0.1       # Test data ratio
       time_series: true     # Time series split (no random shuffle)

Evaluation Configuration
-------------------------

Evaluation Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   evaluation:
     metrics:
       - "sharpe_ratio"      # Sharpe ratio
       - "max_drawdown"      # Maximum drawdown
       - "total_return"      # Total return
       - "win_rate"          # Win rate
       - "profit_factor"     # Profit factor
       - "calmar_ratio"      # Calmar ratio

Backtest Settings
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   evaluation:
     backtest:
       episodes: 100         # Number of backtest episodes
       deterministic: false  # Use deterministic policy
       render: false         # Render environment during evaluation

Benchmark Comparison
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   evaluation:
     benchmark:
       compare_to_buy_hold: true  # Compare to buy-and-hold strategy
       compare_to_baseline: true  # Compare to baseline strategy
       baseline_strategy: "random"  # Baseline strategy type

Logging Configuration
---------------------

Logging Levels
~~~~~~~~~~~~~~

.. code-block:: yaml

   logging:
     level: "INFO"           # Logging level: DEBUG, INFO, WARNING, ERROR
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     file: "logs/trading.log"  # Log file path
     max_file_size: "10MB"   # Maximum log file size
     backup_count: 5         # Number of backup files

Advanced Logging
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   logging:
     handlers:
       - type: "file"        # Handler type: file, console, syslog
         level: "DEBUG"
         format: "%(asctime)s - %(levelname)s - %(message)s"
       - type: "console"
         level: "INFO"

     loggers:
       ztb.trading: "DEBUG"  # Specific logger levels
       ztb.training: "INFO"

Configuration Validation
-------------------------

Validating Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.config import ConfigManager, ConfigValidator

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')

   # Validate configuration
   validator = ConfigValidator()
   errors = validator.validate(config)

   if errors:
       print("Configuration errors:")
       for error in errors:
           print(f"- {error}")
   else:
       print("Configuration is valid!")

Configuration Schema
~~~~~~~~~~~~~~~~~~~~

Zaif Trade Bot includes a configuration schema for validation:

.. code-block:: python

   from ztb.config import get_config_schema

   # Get configuration schema
   schema = get_config_schema()
   print(schema)

Environment Variables
---------------------

Configuration can also be overridden using environment variables:

.. code-block:: bash

   # Override configuration values
   export ZTB_TRADING_SYMBOL="ETH/JPY"
   export ZTB_TRAINING_TOTAL_TIMESTEPS="2000000"
   export ZTB_LOGGING_LEVEL="DEBUG"

.. code-block:: yaml

   # Configuration with environment variable support
   trading:
     symbol: ${ZTB_TRADING_SYMBOL:BTC/JPY}  # Default to BTC/JPY if not set

Configuration Templates
-----------------------

Pre-built Configuration Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zaif Trade Bot provides several pre-built configuration templates:

**Beginner Template**

.. code-block:: yaml

   # config/beginner.yaml
   trading:
     symbol: "BTC/JPY"
     initial_balance: 1000000
     position_size: 0.05
     max_position_size: 0.2

   training:
     algorithm: "PPO"
     total_timesteps: 50000
     learning_rate: 0.0003

**Advanced Template**

.. code-block:: yaml

   # config/advanced.yaml
   trading:
     symbol: "BTC/JPY"
     initial_balance: 5000000
     position_size: 0.02
     max_position_size: 0.1
     stop_loss: 0.03
     take_profit: 0.08

   training:
     algorithm: "SAC"
     total_timesteps: 2000000
     learning_rate: 0.0001
     ent_coef: "auto"

Configuration Management
-------------------------

Loading Multiple Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.config import ConfigManager

   # Load base configuration
   config = ConfigManager.load_config('config/base.yaml')

   # Override with environment-specific settings
   env_config = ConfigManager.load_config('config/production.yaml')
   config = ConfigManager.merge_configs(config, env_config)

Saving Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save current configuration
   ConfigManager.save_config(config, 'config/my_config.yaml')

Configuration Profiles
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.config import ConfigProfile

   # Create configuration profile
   profile = ConfigProfile(name="conservative")
   profile.set_trading_params(
       position_size=0.02,
       max_drawdown=0.1
   )

   # Save profile
   profile.save('profiles/conservative.yaml')

Best Practices
--------------

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Validate Always**: Always validate your configuration before training
3. **Version Control**: Keep your configurations under version control
4. **Document Changes**: Comment your configuration changes
5. **Test Thoroughly**: Test configurations in simulation before live trading
6. **Backup Regularly**: Keep backups of working configurations

Troubleshooting
---------------

Common Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Invalid Algorithm Name**
* Check spelling: Use "PPO", "SAC", "A2C", or "DDPG"
* Verify algorithm is installed

**Memory Issues**
* Reduce batch_size
* Decrease network size (net_arch)
* Use smaller buffer_size

**Training Not Converging**
* Adjust learning_rate (try 0.0001 to 0.001)
* Modify gamma (0.95 to 0.999)
* Change ent_coef (0.001 to 0.1)

**Poor Performance**
* Check reward function scaling
* Verify data quality
* Adjust exploration parameters

Next Steps
----------

* :doc:`../quickstart/index` - Get started with basic usage
* :doc:`../training/index` - Learn about training configurations
* :doc:`../evaluation/index` - Understand evaluation settings
