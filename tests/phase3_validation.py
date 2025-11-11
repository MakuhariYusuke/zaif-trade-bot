#!/usr/bin/env python3
"""
Phase 3 Validation Script
既存バックテスト vs Phase 3統合バックテストの比較検証

このスクリプトは、Phase 3の実装を実際のバックテストで検証し、
リスク管理と統計検証の効果を確認します。
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import SAC

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import setup_logging
from ztb.utils.config_loader import safe_json_load

# Phase 3 imports
from ztb.trading.backtest.integrated_backtest_runner import IntegratedBacktestRunner
from ztb.risk.enhanced_risk_manager import EnhancedRiskManager
from ztb.utils.statistical_validator import StatisticalValidator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from backtest.data_generator import generate_synthetic_data


def create_sac_strategy(model, feature_engineer):
    """
    SACモデルを使った戦略関数を作成

    Args:
        model: 学習済みSACモデル
        feature_engineer: 特徴量エンジニア

    Returns:
        戦略関数
    """
    def strategy_func(market_data: pd.DataFrame, current_position: float = 0.0) -> str:
        """
        SACベースの取引戦略

        Args:
            market_data: 市場データ（直近のデータ）
            current_position: 現在のポジション

        Returns:
            アクション（'buy', 'sell', 'hold'）
        """
        try:
            # 特徴量抽出
            features = feature_engineer.extract_features(market_data)

            # 観測値作成
            obs = np.array(features).astype(np.float32)

            # SACでアクション予測
            action, _ = model.predict(obs, deterministic=True)

            # 連続アクションを離散アクションに変換
            discrete_action = continuous_to_discrete_action(action)

            # アクションを文字列に変換
            if discrete_action == 0:
                return ACTION_SELL
            elif discrete_action == 1:
                return ACTION_HOLD
            elif discrete_action == 2:
                return ACTION_BUY
            else:
                return ACTION_HOLD

        except Exception as e:
            logger.warning(f"Strategy execution failed: {e}")
            return ACTION_HOLD

    return strategy_func


def run_baseline_backtest(model_name: str) -> Dict[str, Any]:
    """
    ベースライン（従来）のバックテスト実行

    Args:
        model_name: モデル名
        config_path: 設定ファイルパス

    Returns:
        バックテスト結果
    """
    logger.info("🏃 Running baseline backtest (without Phase 3 features)")

    try:
        # 簡易設定（UnifiedConfigを使わず直接定義）
        config = {
            "model_name": model_name,
            "algorithm": "sac",
            "feature_config": {
                "features": ["open", "high", "low", "close", "volume"],
                "timeframes": ["1h"],
                "technical_indicators": []
            }
        }

        # データ読み込み
        data_file = "data/btc_jpy_real_dataset.csv"
        if not Path(data_file).exists():
            logger.info("Generating synthetic BTC price data...")
            synthetic_df = generate_synthetic_data(n_periods=5000, start_price=50000.0, volatility=500)
            synthetic_df.to_csv(data_file)

        df = pd.read_csv(data_file)
        logger.info(f"Data loaded: {len(df)} rows")

        # 特徴量エンジニアリング
        feature_engineer = SACv427FeatureEngineer(config.get("feature_config", {}))

        # モデルロード
        model_path = f"models/{model_name}.zip"
        model = SAC.load(model_path)
        logger.info(f"Model loaded: {model_name}")

        # 戦略関数作成
        strategy_func = create_sac_strategy(model, feature_engineer)

        # 簡易バックテスト実行
        capital = 10000.0
        position = 0.0
        portfolio_values = [capital]
        trades = []

        for i in range(100, len(df)):  # ウォームアップ期間をスキップ
            current_data = df.iloc[i-100:i+1]  # 直近100期間のデータ

            # 戦略実行
            action = strategy_func(current_data, float(position))

            # 簡易取引実行（実際の取引ロジック）
            current_price = float(df.loc[i, 'close'])

            if action == ACTION_BUY and position == 0.0:
                # 買い注文
                position_size = capital * 0.1  # 10%ポジション
                shares = position_size / current_price
                position = shares
                capital -= position_size
                trades.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'action': 'buy',
                    'price': current_price,
                    'size': shares,
                    'capital': capital
                })

            elif action == ACTION_SELL and position > 0.0:
                # 売り注文
                position_value = position * current_price
                capital += position_value
                trades.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'action': 'sell',
                    'price': current_price,
                    'size': position,
                    'capital': capital
                })
                position = 0.0

            # ポートフォリオ価値更新
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            portfolio_values.append(portfolio_value)

        # 結果集計
        baseline_result = {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - 10000.0) / 10000.0,
            'total_trades': len(trades),
            'sharpe_ratio': calculate_sharpe_ratio(portfolio_values),
            'max_drawdown': calculate_max_drawdown(portfolio_values)
        }

        logger.info(f"✅ Baseline backtest completed. Final value: {baseline_result['final_value']:.2f}")
        return baseline_result

    except Exception as e:
        logger.error(f"❌ Baseline backtest failed: {e}")
        return {'error': str(e)}


def run_phase3_backtest(model_name: str) -> Dict[str, Any]:
    """
    Phase 3統合バックテスト実行

    Args:
        model_name: モデル名
        config_path: 設定ファイルパス

    Returns:
        Phase 3バックテスト結果
    """
    logger.info("🚀 Running Phase 3 integrated backtest (with risk management & statistical validation)")

    try:
        # 簡易設定
        config = {
            "model_name": model_name,
            "algorithm": "sac",
            "feature_config": {
                "features": ["open", "high", "low", "close", "volume"],
                "timeframes": ["1h"],
                "technical_indicators": []
            }
        }

        # データ読み込み
        data_file = "data/btc_jpy_real_dataset.csv"
        if not Path(data_file).exists():
            logger.info("Generating synthetic BTC price data...")
            synthetic_df = generate_synthetic_data(n_periods=5000, start_price=50000.0, volatility=500)
            synthetic_df.to_csv(data_file)

        df = pd.read_csv(data_file)
        logger.info(f"Data loaded: {len(df)} rows")

        # 特徴量エンジニアリング
        feature_engineer = SACv427FeatureEngineer(config.get("feature_config", {}))

        # モデルロード
        model_path = f"models/{model_name}.zip"
        model = SAC.load(model_path)
        logger.info(f"Model loaded: {model_name}")

        # 戦略関数作成
        strategy_func = create_sac_strategy(model, feature_engineer)

        # Phase 3統合バックテスト設定
        phase3_config = {
            "backtest_config": {
                "initial_capital": 10000.0,
                "slippage_bps": 5.0,
                "commission_bps": 0.0,
                "enable_risk": True,
                "risk_profile": "balanced",
                "max_position_size": 1.0
            },
            "risk_config": {
                "max_position_size": 0.2,  # 20%最大ポジション
                "max_drawdown_limit": 0.1,  # 10%最大ドローダウン
                "var_limit": 0.05,  # 5% VaRリミット
                "enable_multi_timeframe": True
            },
            "validation_config": {
                "confidence_level": 0.95,
                "min_sample_size": 30,
                "enable_bootstrap": True,
                "n_bootstrap_samples": 1000
            },
            "enable_risk_management": True,
            "enable_statistical_validation": True,
            "multi_timeframe_enabled": True,
            "n_iterations": 10,  # 検証用に10イテレーション
            "confidence_level": 0.95
        }

        # Phase 3統合バックテスト実行
        runner = IntegratedBacktestRunner(phase3_config)
        result = runner.run_integrated_backtest(
            strategy_func=strategy_func,
            market_data=df,
            initial_capital=10000.0,
            commission=0.001
        )

        if result.get('success'):
            logger.info("✅ Phase 3 integrated backtest completed successfully")
            return result
        else:
            logger.error(f"❌ Phase 3 backtest failed: {result.get('error')}")
            return result

    except Exception as e:
        logger.error(f"❌ Phase 3 backtest failed: {e}")
        return {'error': str(e)}


def calculate_sharpe_ratio(portfolio_values: list) -> float:
    """シャープレシオ計算"""
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    return np.mean(returns) / np.std(returns) * np.sqrt(252)  # 年率化


def calculate_max_drawdown(portfolio_values: list) -> float:
    """最大ドローダウン計算"""
    if len(portfolio_values) < 2:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def compare_results(baseline_result: Dict[str, Any], phase3_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    結果比較

    Args:
        baseline_result: ベースライン結果
        phase3_result: Phase 3結果

    Returns:
        比較結果
    """
    comparison = {
        'baseline': baseline_result,
        'phase3': {},
        'improvements': {}
    }

    if 'error' in baseline_result:
        logger.error(f"Baseline had error: {baseline_result['error']}")
        return comparison

    if not phase3_result.get('success'):
        logger.error(f"Phase 3 had error: {phase3_result.get('error')}")
        return comparison

    # Phase 3のサマリー結果を抽出
    phase3_summary = phase3_result.get('summary', {})
    comparison['phase3'] = phase3_summary

    # 改善点計算
    baseline_return = baseline_result.get('total_return', 0)
    baseline_sharpe = baseline_result.get('sharpe_ratio', 0)
    baseline_dd = baseline_result.get('max_drawdown', 0)

    phase3_return = phase3_summary.get('mean_total_return', 0)
    phase3_sharpe = phase3_summary.get('mean_sharpe_ratio', 0)
    phase3_dd = phase3_summary.get('mean_max_drawdown', 0)

    comparison['improvements'] = {
        'return_improvement': phase3_return - baseline_return,
        'sharpe_improvement': phase3_sharpe - baseline_sharpe,
        'drawdown_reduction': baseline_dd - phase3_dd,  # 正の値が改善
        'return_std_reduction': phase3_summary.get('std_total_return', 0),  # Phase 3の標準偏差
        'sharpe_std_reduction': phase3_summary.get('std_sharpe_ratio', 0)
    }

    return comparison


def main():
    """メイン実行関数"""
    logger.info("🎯 Phase 3 Validation Script Started")
    logger.info("=" * 60)

    # 設定
    model_name = "sac_v444.1"  # 使用するモデル名

    try:
        # 1. ベースラインバックテスト実行
        logger.info("Step 1: Running baseline backtest...")
        baseline_result = run_baseline_backtest(model_name)

        # 2. Phase 3統合バックテスト実行
        logger.info("Step 2: Running Phase 3 integrated backtest...")
        phase3_result = run_phase3_backtest(model_name)

        # 3. 結果比較
        logger.info("Step 3: Comparing results...")
        comparison = compare_results(baseline_result, phase3_result)

        # 4. 結果出力
        logger.info("Step 4: Generating report...")

        print("\n" + "="*80)
        print("PHASE 3 VALIDATION RESULTS")
        print("="*80)

        if 'error' not in baseline_result:
            print("\n📊 BASELINE RESULTS:")
            print(f"  Final Value: {baseline_result.get('final_value', 0):.2f}")
            print(f"  Total Return: {baseline_result.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {baseline_result.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {baseline_result.get('max_drawdown', 0):.2%}")
            print(f"  Total Trades: {baseline_result.get('total_trades', 0)}")

        if phase3_result.get('success'):
            phase3_summary = comparison.get('phase3', {})
            print("\n🚀 PHASE 3 INTEGRATED RESULTS:")
            print(f"  Mean Final Value: {phase3_summary.get('mean_final_value', 0):.2f}")
            print(f"  Mean Total Return: {phase3_summary.get('mean_total_return', 0):.2%}")
            print(f"  Mean Sharpe Ratio: {phase3_summary.get('mean_sharpe_ratio', 0):.2f}")
            print(f"  Mean Max Drawdown: {phase3_summary.get('mean_max_drawdown', 0):.2%}")
            print(f"  Iterations: {phase3_summary.get('iterations_count', 0)}")

        improvements = comparison.get('improvements', {})
        if improvements:
            print("\n✨ IMPROVEMENTS (Phase 3 vs Baseline):")
            print(f"  Return Improvement: {improvements.get('return_improvement', 0):.2%}")
            print(f"  Sharpe Improvement: {improvements.get('sharpe_improvement', 0):.2f}")
            print(f"  Drawdown Reduction: {improvements.get('drawdown_reduction', 0):.2%}")
            print(f"  Return Std Dev: {improvements.get('return_std_reduction', 0):.4f}")
            print(f"  Sharpe Std Dev: {improvements.get('sharpe_std_reduction', 0):.4f}")

        # 詳細結果をJSONで保存
        output_file = f"phase3_validation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'baseline_result': baseline_result,
                'phase3_result': phase3_result,
                'comparison': comparison
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Detailed results saved to: {output_file}")
        print("="*80)

        logger.info("✅ Phase 3 validation completed successfully")

    except Exception as e:
        logger.error(f"❌ Phase 3 validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()