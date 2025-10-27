#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Quick SAC v438.1 Backtest"""

"""

Quick SAC v438.1 Backtest - Bull/Bear Balanced Features Test""""""

import sys

from pathlib import Path



project_root = Path(__file__).parentQuick backtest to validate SAC v438.1 balanced bull/bear features and adjusted rewards.Quick SAC v438.1 Backtest - Bull/Bear Balanced Features TestQuick SAC v438.1 Backtest - Bull/Bear Balanced Features Test

sys.path.insert(0, str(project_root))

"""

import pandas as pd

from stable_baselines3 import SAC



from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineerimport argparse

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

from ztb.utils.logging_utils import get_loggerimport osQuick backtest to validate SAC v438.1 balanced bull/bear features and adjusted rewards.Quick backtest to validate SAC v438.1 balanced bull/bear features and adjusted rewards.



logger = get_logger(__name__)import sys



from datetime import datetime"""Includes bull market features complementary to bear market features for improved trading frequency.

def run_quick_backtest():

    """Run quick backtest for SAC v438.1"""from pathlib import Path

    logger.info("🔍 Running SAC v438.1 quick backtest")

from typing import Optional"""

    model_path = "checkpoints/sac_v438_production_150000_steps.zip"

    data_path = "data/btc_jpy_real_dataset.csv"



    # Load data# Add project root to pathimport argparse

    df = pd.read_csv(data_path)

    logger.info(f"📊 Loaded {len(df)} data points")project_root = Path(__file__).parent



    # Generate featuressys.path.insert(0, str(project_root))import osimport sys

    feature_engineer = SACv427FeatureEngineer()

    features_df = feature_engineer.generate_v427_features(df.head(100))  # First 100 points

    logger.info(f"🔧 Generated {len(features_df.columns)} features")

import pandas as pdimport sysfrom pathlib import Path

    # Create environment

    env_config = {from stable_baselines3 import SAC

        "initial_balance": 200000,

        "transaction_cost": 0.001,from datetime import datetime

        "max_position_size": 0.1,

        "curriculum_stage": "profit_optimized",from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

    }

    env = HeavyTradingEnv(df=features_df, config=env_config, random_start=False)from ztb.trading.environment.heavy_env.core import HeavyTradingEnvfrom pathlib import Pathimport pandas as pd



    # Load modelfrom ztb.utils.logging_utils import get_logger

    model = SAC.load(model_path)

    logger.info("🤖 Model loaded")from typing import Optional



    # Run backtestlogger = get_logger(__name__)

    obs, info = env.reset()

    done = False# Add project root to path

    total_reward = 0

    total_trades = 0

    steps = 0

def backtest_sac_v438_quick(# Add project root to pathproject_root = Path(__file__).parent

    while not done and steps < 100:

        action, _ = model.predict(obs, deterministic=True)    model_path: str,

        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward    data_path: Optional[str] = None,project_root = Path(__file__).parentsys.path.insert(0, str(project_root))

        steps += 1

        if info.get("trade_executed", False):    output_dir: str = "backtest_experiments/v438.1",

            total_trades += 1

    n_episodes: int = 3,sys.path.insert(0, str(project_root))

    logger.info("✅ Backtest completed!")

    logger.info(f"📊 Total Reward: {total_reward:.2f}")    deterministic: bool = True,

    logger.info(f"📊 Total Trades: {total_trades}")

    logger.info(f"📊 Trades per Step: {total_trades / steps:.3f}")):from ztb.trading.backtest.adapters import RLPolicyAdapter



    return {    """

        "total_reward": total_reward,

        "total_trades": total_trades,    Quick backtest for SAC v438.1 model with balanced features.import pandas as pdfrom ztb.trading.backtest.runner import BacktestEngine

        "trades_per_step": total_trades / steps,

    }



    Args:from stable_baselines3 import SACfrom ztb.utils.logging_utils import get_logger

if __name__ == "__main__":

    result = run_quick_backtest()        model_path: Path to trained model

    print("\n" + "="*50)

    print("SAC v438.1 BACKTEST RESULTS")        data_path: Path to test data

    print("="*50)

    print(f"Total Reward: {result['total_reward']:.2f}")        output_dir: Output directory for results

    print(f"Total Trades: {result['total_trades']}")

    print(f"Trades per Step: {result['trades_per_step']:.3f}")        n_episodes: Number of backtest episodesfrom ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineerlogger = get_logger(__name__)

    print("="*50)
        deterministic: Whether to use deterministic actions

    """from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

    logger.info(f"Starting SAC v438.1 quick backtest with model: {model_path}")

from ztb.utils.logging_utils import get_logger

    # Create output directory

    os.makedirs(output_dir, exist_ok=True)def run_v438_quick_backtest():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(output_dir, f"backtest_sac_model_{timestamp}")logger = get_logger(__name__)    """Run quick backtest for SAC v438.1 balanced bull/bear features."""

    os.makedirs(run_dir, exist_ok=True)



    # Load and preprocess data

    if data_path is None:    logger.info("🔍 Running SAC v438.1 quick backtest - Bull/Bear Balance Validation")

        data_path = "data/btc_jpy_real_dataset.csv"

def backtest_sac_v438_quick(

    logger.info(f"Loading test data from {data_path}")

    df = pd.read_csv(data_path)    model_path: str,    # Model path - use the trained model



    # Generate v438.1 features (includes both bear and bull market features)    data_path: Optional[str] = None,    model_path = "checkpoints/sac_session/sac_model_final.zip"

    logger.info("Generating SAC v427 features with bull/bear balance")

    feature_engineer = SACv427FeatureEngineer()    output_dir: str = "backtest_experiments/v438.1",

    features_df = feature_engineer.generate_v427_features(df)

    n_episodes: int = 3,    if not Path(model_path).exists():

    logger.info(f"Generated {len(features_df.columns)} features")

    deterministic: bool = True,        logger.error(f"❌ Model not found: {model_path}")

    # Create environment with v438.1 settings

    env_config = {):        logger.info("💡 Please run training first: python train_sac_v438_quick.py")

        "initial_balance": 200000,

        "transaction_cost": 0.001,    """        return None

        "max_position_size": 0.1,

        "feature_set": "full",    Quick backtest for SAC v438.1 model with balanced features.

        "reward_scaling": 1.0,

        "risk_free_rate": 0.02,    # Load test data

        "timeframe": "1m",

        "exchange": "coincheck",    Args:    data_path = project_root / "data" / "btc_jpy_real_dataset.csv"

        "stop_loss_threshold": 0.05,

        "max_consecutive_trades": 10,        model_path: Path to trained model    if not data_path.exists():

        "min_holding_period": 1,

        "curriculum_stage": "profit_optimized",  # Use profit_optimized for v438.1        data_path: Path to test data        logger.error(f"❌ Test data not found: {data_path}")

    }

        output_dir: Output directory for results        return None

    env = HeavyTradingEnv(

        df=features_df, config=env_config, random_start=False        n_episodes: Number of backtest episodes

    )

        deterministic: Whether to use deterministic actions    df = pd.read_csv(data_path)

    # Load model

    logger.info(f"Loading model from {model_path}")    """    logger.info(f"� Loaded {len(df)} data points for backtest")

    model = SAC.load(model_path)

    logger.info(f"Starting SAC v438.1 quick backtest with model: {model_path}")

    # Run backtest

    results = []    # Create RL policy adapter with SAC model

    portfolio_values = []

    trades_history = []    # Create output directory    adapter = RLPolicyAdapter(model_path=str(model_path))



    logger.info(f"Running {n_episodes} backtest episodes")    os.makedirs(output_dir, exist_ok=True)



    for episode in range(n_episodes):    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # Create backtest engine

        logger.info(f"Episode {episode + 1}/{n_episodes}")

    run_dir = os.path.join(output_dir, f"backtest_sac_model_{timestamp}")    engine = BacktestEngine(

        obs, info = env.reset()

        done = False    os.makedirs(run_dir, exist_ok=True)        initial_capital=1000000.0,  # Increased capital to avoid insufficient funds

        episode_reward = 0

        episode_trades = 0        commission_bps=15.0,  # 0.15%

        step_count = 0

    # Load and preprocess data        slippage_bps=5.0,     # 0.05%

        episode_portfolio = []

        episode_trades_data = []    if data_path is None:    )



        while not done:        data_path = "data/btc_jpy_real_dataset.csv"

            action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, done, truncated, info = env.step(action)    # Run quick backtest (first 100 data points for speed)



            episode_reward += reward    logger.info(f"Loading test data from {data_path}")    test_df = df.head(100).copy()

            step_count += 1

    df = pd.read_csv(data_path)    logger.info("📈 Running quick backtest on first 100 data points...")

            # Record portfolio value

            portfolio_value = info.get("portfolio_value", env_config["initial_balance"])

            episode_portfolio.append(

                {    # Generate v438.1 features (includes both bear and bull market features)    equity_curve, orders_df, adaptation_history = engine.run_backtest(adapter, test_df)

                    "step": step_count,

                    "portfolio_value": portfolio_value,    logger.info("Generating SAC v427 features with bull/bear balance")

                    "reward": reward,

                }    feature_engineer = SACv427FeatureEngineer()    # Calculate basic metrics

            )

    features_df = feature_engineer.generate_v427_features(df)    total_return = 0

            # Record trades

            if info.get("trade_executed", False):    win_rate = 0

                episode_trades += 1

                trade_info = {    logger.info(f"Generated {len(features_df.columns)} features")    total_trades = 0

                    "episode": episode + 1,

                    "step": step_count,

                    "action": action,

                    "portfolio_value": portfolio_value,    # Create environment with v438.1 settings    if len(equity_curve) > 0:

                    "reward": reward,

                    **info,    env_config = {        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

                }

                episode_trades_data.append(trade_info)        "initial_balance": 200000,        logger.info(f"📊 Total Return: {total_return:.2f}%")



        # Store episode results        "transaction_cost": 0.001,

        results.append(

            {        "max_position_size": 0.1,    if len(orders_df) > 0:

                "episode": episode + 1,

                "total_reward": episode_reward,        "feature_set": "full",        profitable_trades = len(orders_df[orders_df['pnl'] > 0])

                "total_trades": episode_trades,

                "final_portfolio_value": portfolio_value,        "reward_scaling": 1.0,        total_trades = len(orders_df)

                "total_steps": step_count,

                "avg_reward_per_step": episode_reward / step_count if step_count > 0 else 0,        "risk_free_rate": 0.02,        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

                "trades_per_step": episode_trades / step_count if step_count > 0 else 0,

            }        "timeframe": "1m",        logger.info(f"📊 Win Rate: {win_rate:.2f}%")

        )

        "exchange": "coincheck",        logger.info(f"📊 Total Trades: {total_trades}")

        portfolio_values.extend(episode_portfolio)

        trades_history.extend(episode_trades_data)        "stop_loss_threshold": 0.05,



        logger.info(        "max_consecutive_trades": 10,        # Analyze bull/bear market performance balance

            f"Episode {episode + 1}: Reward={episode_reward:.2f}, "

            f"Trades={episode_trades}, Final Value={portfolio_value:.2f}, "        "min_holding_period": 1,        if 'bear_market_signal' in orders_df.columns:

            f"Trades/Step={episode_trades / step_count:.3f}"

        )        "curriculum_stage": "profit_optimized",  # Use profit_optimized for v438.1            bear_trades = orders_df[orders_df['bear_market_signal'] == True]



    # Save results    }            bull_trades = orders_df[orders_df['bear_market_signal'] == False]  # Assuming bull when not bear

    results_df = pd.DataFrame(results)

    portfolio_df = pd.DataFrame(portfolio_values)            

    trades_df = pd.DataFrame(trades_history)

    env = HeavyTradingEnv(            if len(bear_trades) > 0:

    results_file = os.path.join(run_dir, "backtest_results.json")

    portfolio_file = os.path.join(run_dir, "portfolio_values.csv")        df=features_df, config=env_config, random_start=False                bear_profitable = len(bear_trades[bear_trades['pnl'] > 0])

    trades_file = os.path.join(run_dir, "trades_history.csv")

    )                bear_win_rate = (bear_profitable / len(bear_trades) * 100) if len(bear_trades) > 0 else 0

    results_df.to_json(results_file, orient="records", indent=2)

    portfolio_df.to_csv(portfolio_file, index=False)                logger.info(f"🐻 Bear Market Win Rate: {bear_win_rate:.2f}% ({bear_profitable}/{len(bear_trades)} trades)")

    trades_df.to_csv(trades_file, index=False)

    # Load model            else:

    # Calculate summary statistics

    summary = calculate_backtest_summary(results_df, portfolio_df, trades_df)    logger.info(f"Loading model from {model_path}")                logger.info("🐻 No bear market trades detected")



    summary_file = os.path.join(run_dir, "backtest_summary.json")    model = SAC.load(model_path)                

    with open(summary_file, "w") as f:

        import json            if len(bull_trades) > 0:

        json.dump(summary, f, indent=2, default=str)

    # Run backtest                bull_profitable = len(bull_trades[bull_trades['pnl'] > 0])

    logger.info(f"Backtest completed. Results saved to {run_dir}")

    logger.info(f"Summary: {summary}")    results = []                bull_win_rate = (bull_profitable / len(bull_trades) * 100) if len(bull_trades) > 0 else 0



    return summary    portfolio_values = []                logger.info(f"🐂 Bull Market Win Rate: {bull_win_rate:.2f}% ({bull_profitable}/{len(bull_trades)} trades)")



    trades_history = []            else:

def calculate_backtest_summary(results_df, portfolio_df, trades_df):

    """Calculate backtest summary statistics."""                logger.info("🐂 No bull market trades detected")

    summary = {

        "total_episodes": len(results_df),    logger.info(f"Running {n_episodes} backtest episodes")

        "avg_total_reward": results_df["total_reward"].mean(),

        "std_total_reward": results_df["total_reward"].std(),    logger.info("✅ Backtest completed!")

        "avg_final_portfolio_value": results_df["final_portfolio_value"].mean(),

        "std_final_portfolio_value": results_df["final_portfolio_value"].std(),    for episode in range(n_episodes):

        "avg_total_trades": results_df["total_trades"].mean(),

        "avg_trades_per_step": results_df["trades_per_step"].mean(),        logger.info(f"Episode {episode + 1}/{n_episodes}")    # Save results to file

        "total_trades_all_episodes": trades_df.shape[0],

        "best_episode_reward": results_df["total_reward"].max(),    import json

        "worst_episode_reward": results_df["total_reward"].min(),

        "reward_positive_ratio": (results_df["total_reward"] > 0).mean(),        obs, info = env.reset()    from datetime import datetime

        "portfolio_value_positive_ratio": (

            results_df["final_portfolio_value"] > 200000        done = False    

        ).mean(),

    }        episode_reward = 0    result_data = {



    # Calculate Sharpe-like ratio        episode_trades = 0        'timestamp': datetime.now().isoformat(),

    if len(results_df) > 1:

        returns = results_df["total_reward"]        step_count = 0        'model': 'sac_v438.1',

        summary["sharpe_ratio"] = returns.mean() / (returns.std() + 1e-8)

        'total_return': total_return,

    # Calculate max drawdown from portfolio values

    if not portfolio_df.empty:        episode_portfolio = []        'win_rate': win_rate,

        portfolio_values = portfolio_df.groupby("step")["portfolio_value"].mean()

        peak = portfolio_values.expanding().max()        episode_trades_data = []        'total_trades': total_trades,

        drawdown = (portfolio_values - peak) / peak

        summary["max_drawdown"] = drawdown.min()        'equity_curve': equity_curve.tolist() if hasattr(equity_curve, 'tolist') else list(equity_curve),



    return summary        while not done:        'orders': orders_df.to_dict('records') if len(orders_df) > 0 else []



            action, _ = model.predict(obs, deterministic=deterministic)    }

def main():

    parser = argparse.ArgumentParser(description="Quick backtest SAC v438.1 model")            obs, reward, done, truncated, info = env.step(action)    

    parser.add_argument(

        "--model-path", type=str, required=True, help="Path to trained model"    result_file = f"backtest_results_sac_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    )

    parser.add_argument("--data-path", type=str, default=None, help="Path to test data")            episode_reward += reward    with open(result_file, 'w') as f:

    parser.add_argument(

        "--output-dir",            step_count += 1        json.dump(result_data, f, indent=2, default=str)

        type=str,

        default="backtest_experiments/v438.1",    logger.info(f"💾 Results saved to {result_file}")

        help="Output directory for results",

    )            # Record portfolio value

    parser.add_argument(

        "--episodes", type=int, default=3, help="Number of backtest episodes"            portfolio_value = info.get("portfolio_value", env_config["initial_balance"])    return {

    )

    parser.add_argument(            episode_portfolio.append(        'equity_curve': equity_curve,

        "--deterministic", action="store_true", help="Use deterministic actions"

    )                {        'orders': orders_df,



    args = parser.parse_args()                    "step": step_count,        'total_return': total_return,



    # Run backtest                    "portfolio_value": portfolio_value,        'win_rate': win_rate,

    summary = backtest_sac_v438_quick(

        model_path=args.model_path,                    "reward": reward,        'total_trades': total_trades,

        data_path=args.data_path,

        output_dir=args.output_dir,                }        'result_file': result_file

        n_episodes=args.episodes,

        deterministic=args.deterministic,            )    }

    )



    print("Backtest Summary:")

    for key, value in summary.items():            # Record trades

        print(f"  {key}: {value}")

            if info.get("trade_executed", False):if __name__ == "__main__":



if __name__ == "__main__":                episode_trades += 1    result = run_v438_quick_backtest()

    main()
                trade_info = {    if result:

                    "episode": episode + 1,        print("\n" + "="*50)

                    "step": step_count,        print("SAC v438.1 BACKTEST RESULTS")

                    "action": action,        print("="*50)

                    "portfolio_value": portfolio_value,        print(f"Total Return: {result['total_return']:.2f}%")

                    "reward": reward,        print(f"Win Rate: {result['win_rate']:.2f}%")

                    **info,        print(f"Total Trades: {result['total_trades']}")

                }        print("="*50)

                episode_trades_data.append(trade_info)    else:

        print("❌ Backtest failed!")

        # Store episode results
        results.append(
            {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "total_trades": episode_trades,
                "final_portfolio_value": portfolio_value,
                "total_steps": step_count,
                "avg_reward_per_step": episode_reward / step_count if step_count > 0 else 0,
                "trades_per_step": episode_trades / step_count if step_count > 0 else 0,
            }
        )

        portfolio_values.extend(episode_portfolio)
        trades_history.extend(episode_trades_data)

        logger.info(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
            f"Trades={episode_trades}, Final Value={portfolio_value:.2f}, "
            f"Trades/Step={episode_trades / step_count:.3f}"
        )

    # Save results
    results_df = pd.DataFrame(results)
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades_history)

    results_file = os.path.join(run_dir, "backtest_results.json")
    portfolio_file = os.path.join(run_dir, "portfolio_values.csv")
    trades_file = os.path.join(run_dir, "trades_history.csv")

    results_df.to_json(results_file, orient="records", indent=2)
    portfolio_df.to_csv(portfolio_file, index=False)
    trades_df.to_csv(trades_file, index=False)

    # Calculate summary statistics
    summary = calculate_backtest_summary(results_df, portfolio_df, trades_df)

    summary_file = os.path.join(run_dir, "backtest_summary.json")
    with open(summary_file, "w") as f:
        import json
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Backtest completed. Results saved to {run_dir}")
    logger.info(f"Summary: {summary}")

    return summary


def calculate_backtest_summary(results_df, portfolio_df, trades_df):
    """Calculate backtest summary statistics."""
    summary = {
        "total_episodes": len(results_df),
        "avg_total_reward": results_df["total_reward"].mean(),
        "std_total_reward": results_df["total_reward"].std(),
        "avg_final_portfolio_value": results_df["final_portfolio_value"].mean(),
        "std_final_portfolio_value": results_df["final_portfolio_value"].std(),
        "avg_total_trades": results_df["total_trades"].mean(),
        "avg_trades_per_step": results_df["trades_per_step"].mean(),
        "total_trades_all_episodes": trades_df.shape[0],
        "best_episode_reward": results_df["total_reward"].max(),
        "worst_episode_reward": results_df["total_reward"].min(),
        "reward_positive_ratio": (results_df["total_reward"] > 0).mean(),
        "portfolio_value_positive_ratio": (
            results_df["final_portfolio_value"] > 200000
        ).mean(),
    }

    # Calculate Sharpe-like ratio
    if len(results_df) > 1:
        returns = results_df["total_reward"]
        summary["sharpe_ratio"] = returns.mean() / (returns.std() + 1e-8)

    # Calculate max drawdown from portfolio values
    if not portfolio_df.empty:
        portfolio_values = portfolio_df.groupby("step")["portfolio_value"].mean()
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        summary["max_drawdown"] = drawdown.min()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Quick backtest SAC v438.1 model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_experiments/v438.1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of backtest episodes"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )

    args = parser.parse_args()

    # Run backtest
    summary = backtest_sac_v438_quick(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    print("Backtest Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()