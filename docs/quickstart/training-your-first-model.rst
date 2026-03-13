Training Your First Model
========================

This tutorial will guide you through training your first reinforcement learning model for trading with Zaif Trade Bot.

What is Reinforcement Learning?
-------------------------------

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. In trading, the agent learns optimal trading strategies through trial and error.

Zaif Trade Bot supports two main RL algorithms:

* **PPO (Proximal Policy Optimization)**: Stable and reliable, good for beginners
* **SAC (Soft Actor-Critic)**: More advanced, can achieve higher performance

Prerequisites
-------------

* Zaif Trade Bot installed (:doc:`installation`)
* Basic configuration created (:doc:`basic-usage`)
* Training data prepared (:doc:`first-backtest`)

Understanding the Training Process
-----------------------------------

1. **Environment**: Simulated trading environment
2. **Agent**: The RL algorithm that makes trading decisions
3. **Rewards**: Feedback signals (profit/loss) that guide learning
4. **Training Loop**: Agent interacts with environment, learns from rewards

Creating Training Configuration
-------------------------------

Create a training configuration file:

.. code-block:: yaml

   # config/training.yaml
   training:
     algorithm: "PPO"           # PPO or SAC
     total_timesteps: 100000    # Total training steps (increase for better results)
     learning_rate: 0.0003      # Learning rate
     batch_size: 256           # Batch size for training
     n_epochs: 10              # Number of epochs per update
     gamma: 0.99               # Discount factor
     gae_lambda: 0.95          # GAE lambda
     clip_range: 0.2           # PPO clip range
     ent_coef: 0.01            # Entropy coefficient

   environment:
     symbol: "BTC/JPY"
     initial_balance: 1000000
     position_size: 0.1
     transaction_cost: 0.0005
     max_steps: 1000

   data:
     path: "data/training_data.csv"
     validation_split: 0.2

   logging:
     log_dir: "logs/"
     save_freq: 10000          # Save model every 10k steps
     eval_freq: 5000           # Evaluate every 5k steps

Advanced PPO Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     algorithm: "PPO"
     total_timesteps: 500000

     # PPO specific parameters
     learning_rate: 0.0003
     n_epochs: 10
     gamma: 0.99
     gae_lambda: 0.95
     clip_range: 0.2
     clip_range_vf: None
     ent_coef: 0.0
     vf_coef: 0.5
     max_grad_norm: 0.5
     target_kl: None

     # Network architecture
     policy_kwargs:
       net_arch: [256, 256]    # Hidden layers

SAC Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     algorithm: "SAC"
     total_timesteps: 500000

     # SAC specific parameters
     learning_rate: 0.0003
     buffer_size: 1000000
     learning_starts: 1000
     batch_size: 256
     tau: 0.005
     gamma: 0.99
     train_freq: 1

     # Entropy tuning
     ent_coef: "auto"          # Auto-tune entropy
     target_entropy: "auto"

Preparing Training Data
------------------------

Training data should be high-quality historical market data. Ensure:

* Sufficient data points (minimum 10,000)
* Realistic market conditions
* Proper OHLCV format
* No missing values

.. code-block:: python

   import pandas as pd
   from ztb.data import DataValidator

   # Load and validate data
   data = pd.read_csv('data/training_data.csv')
   validator = DataValidator()

   # Check data quality
   issues = validator.validate(data)
   if issues:
       print("Data issues found:")
       for issue in issues:
           print(f"- {issue}")
   else:
       print("Data validation passed!")

   # Preprocess data
   processed_data = validator.preprocess(data)
   print(f"Processed {len(processed_data)} data points")

Starting Training
-----------------

Method 1: Using Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.training import PPOTrainer
   from ztb.config import ConfigManager
   from ztb.callbacks import TrainingCallbacks
   import os

   # Load configuration
   config = ConfigManager.load_config('config/training.yaml')

   # Create trainer
   trainer = PPOTrainer(config)

   # Set up callbacks for monitoring
   callbacks = TrainingCallbacks(
       save_freq=config.training.save_freq,
       log_dir=config.logging.log_dir,
       eval_freq=config.logging.eval_freq
   )

   # Start training
   print("Starting training... This may take several hours.")
   print(f"Training for {config.training.total_timesteps} timesteps")

   trainer.train(callbacks=callbacks)

   # Save final model
   model_path = "models/trained_model.zip"
   trainer.save(model_path)
   print(f"Model saved to {model_path}")

Method 2: Using Command Line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train model from command line
   ztb train --config config/training.yaml --output models/my_trained_model.zip

   # Monitor training progress
   tail -f logs/training.log

Method 3: Using Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # train_model.py
   import argparse
   from ztb.training import create_trainer
   from ztb.config import ConfigManager
   from ztb.utils import setup_logging

   def main():
       parser = argparse.ArgumentParser(description='Train trading model')
       parser.add_argument('--config', required=True, help='Configuration file')
       parser.add_argument('--output', required=True, help='Output model path')
       args = parser.parse_args()

       # Setup logging
       setup_logging()

       # Load config
       config = ConfigManager.load_config(args.config)

       # Create trainer based on algorithm
       trainer = create_trainer(config)

       # Train model
       trainer.train()

       # Save model
       trainer.save(args.output)
       print(f"Model saved to {args.output}")

   if __name__ == "__main__":
       main()

   # Run the script
   # python train_model.py --config config/training.yaml --output models/my_model.zip

Monitoring Training Progress
-----------------------------

Training progress can be monitored through:

1. **Console Output**: Real-time updates on training progress
2. **TensorBoard**: Visual monitoring of training metrics
3. **Log Files**: Detailed logs in the logs directory

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir logs/

   # Open browser to http://localhost:6006

Key Metrics to Monitor
~~~~~~~~~~~~~~~~~~~~~~~

* **Episode Reward**: Average reward per episode (should increase)
* **Episode Length**: Average episode duration
* **Value Loss**: Critic network loss (should decrease)
* **Policy Loss**: Actor network loss (should decrease)
* **Entropy**: Exploration level (higher = more exploration)

Evaluating Training Results
---------------------------

After training, evaluate your model:

.. code-block:: python

   from ztb.evaluation import ModelEvaluator
   from ztb.backtesting import BacktestRunner

   # Load trained model
   model = PPOTrainer.load("models/trained_model.zip")

   # Create evaluator
   evaluator = ModelEvaluator(model)

   # Evaluate on validation data
   validation_data = pd.read_csv('data/validation_data.csv')
   metrics = evaluator.evaluate(validation_data)

   print("Model Evaluation Results:")
   print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
   print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
   print(f"Total Return: {metrics.total_return:.2f}%")
   print(f"Win Rate: {metrics.win_rate:.2f}%")

   # Run backtest with trained model
   backtest_config = ConfigManager.load_config('config/backtest.yaml')
   backtest_runner = BacktestRunner(backtest_config)

   backtest_results = backtest_runner.run_with_model(model, validation_data)
   print(f"Backtest Return: {backtest_results.total_return:.2f}%")

Common Training Issues
----------------------

**Training Not Converging**
* Reduce learning rate
* Increase batch size
* Check reward function
* Verify data quality

**Poor Performance**
* Start with simpler environments
* Adjust reward scaling
* Increase exploration (higher entropy)
* Use more training data

**Memory Issues**
* Reduce batch size
* Use smaller networks
* Enable gradient checkpointing
* Use mixed precision training

**Slow Training**
* Use GPU acceleration
* Reduce model complexity
* Use distributed training
* Optimize data pipeline

Best Practices
--------------

1. **Start Small**: Begin with short training runs to validate setup
2. **Monitor Closely**: Watch key metrics during training
3. **Use Validation**: Regularly evaluate on held-out data
4. **Experiment**: Try different hyperparameters
5. **Save Frequently**: Keep checkpoints of good models
6. **Document**: Record your training configurations and results

Advanced Training Techniques
-----------------------------

Curriculum Learning
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   training:
     curriculum:
       enabled: true
       phases:
         - name: "simple"
           timesteps: 100000
           difficulty: 0.1
         - name: "medium"
           timesteps: 200000
           difficulty: 0.5
         - name: "hard"
           timesteps: 200000
           difficulty: 1.0

Multi-Environment Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.training import MultiEnvTrainer

   # Train on multiple market conditions
   environments = [
       create_env('bull_market'),
       create_env('bear_market'),
       create_env('sideways_market')
   ]

   trainer = MultiEnvTrainer(environments, config)
   trainer.train()

Next Steps
----------

* :doc:`../evaluation/index` - Learn about model evaluation techniques
* :doc:`../deployment/index` - Deploy your trained models to production
* :doc:`../guides/index` - Explore advanced training techniques
* :doc:`../optimization/index` - Optimize your models for better performance
