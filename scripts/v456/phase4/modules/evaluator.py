"""Walk-Forward Model Evaluator: 各ウィンドウでの訓練・評価"""

import logging
from typing import Dict, List

import numpy as np
from stable_baselines3 import SAC

from ztb.config.environment_config import TrainingConfig
from .result import WindowPerformance

logger = logging.getLogger(__name__)


class WalkForwardModelEvaluator:
    """各ウィンドウでのSAC訓練・評価"""

    def __init__(self):
        self.models: Dict[int, SAC] = {}
        self.results: Dict[int, WindowPerformance] = {}

    def train_and_evaluate_window(
        self,
        df,
        window,
        timesteps: int = 10000,
    ) -> WindowPerformance:
        """ウィンドウ内で訓練・評価"""
        # 遅延インポート（循環依存回避）
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from train_and_evaluate_v456_phase3 import (
            create_environment_wrapper,
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Window {window.window_id}: Training & Evaluation")
        logger.info(f"{'='*70}")
        
        # データ分割
        train_df = df.iloc[window.train_start:window.train_end]
        val_df = df.iloc[window.val_start:window.val_end]
        test_df = df.iloc[window.test_start:window.test_end]
        
        logger.info(f"Train: {len(train_df)} bars")
        logger.info(f"Val:   {len(val_df)} bars")
        logger.info(f"Test:  {len(test_df)} bars")
        
        # 訓練環境作成
        train_env = create_environment_wrapper(train_df, None)
        
        # SAC訓練
        logger.info(f"\n[Training]")
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=TrainingConfig.LEARNING_RATE,
            batch_size=256,
            buffer_size=TrainingConfig.BUFFER_SIZE,
            tau=0.005,
            gamma=0.99,
            verbose=0,
        )
        
        model.learn(total_timesteps=timesteps)
        logger.info(f"✓ Training completed ({timesteps} timesteps)")
        
        # 検証評価
        logger.info(f"\n[Validation Evaluation]")
        val_result = self._evaluate_on_df(model, val_df)
        logger.info(f"  Val ROI: {val_result['roi']:.4f}")
        logger.info(f"  Val Balance: {val_result['final_balance']:,.0f} JPY")
        
        # テスト評価
        logger.info(f"\n[Test Evaluation (Out-of-Sample)]")
        test_result = self._evaluate_on_df(model, test_df)
        logger.info(f"  Test ROI: {test_result['roi']:.4f}")
        logger.info(f"  Test Balance: {test_result['final_balance']:,.0f} JPY")
        
        # 性能オブジェクト作成
        performance = WindowPerformance(
            window_id=window.window_id,
            val_roi=val_result["roi"],
            test_roi=test_result["roi"],
            val_final_balance=val_result["final_balance"],
            test_final_balance=test_result["final_balance"],
            sharpe_ratio=test_result.get("sharpe", 0.0),
            max_drawdown=test_result.get("max_drawdown", 0.0),
            win_rate=test_result.get("win_rate", 0.0),
            trades=test_result.get("trades", 0),
        )
        
        # モデル保存
        self.models[window.window_id] = model
        self.results[window.window_id] = performance
        
        return performance

    def _evaluate_on_df(self, model: SAC, df) -> Dict:
        """データフレーム上で評価"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from train_and_evaluate_v456_phase3 import (
            create_environment_wrapper,
        )
        
        eval_env = create_environment_wrapper(df, None)
        
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        actions = []
        trades = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += float(reward)
            actions.append(float(action[0]))
            
            if info.get("trade_executed"):
                trades += 1
        
        final_balance = eval_env.balance
        total_pnl = final_balance - eval_env.initial_balance
        roi = total_pnl / eval_env.initial_balance
        
        return {
            "episode_reward": episode_reward,
            "final_balance": final_balance,
            "total_pnl": total_pnl,
            "roi": roi,
            "trades": trades,
            "sharpe": self._calculate_sharpe(actions),
            "max_drawdown": self._calculate_max_drawdown([final_balance]),
            "win_rate": 0.5,  # 簡略化
        }

    @staticmethod
    def _calculate_sharpe(returns: List[float], rf_rate: float = 0.0) -> float:
        """Sharpe比計算"""
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = np.mean(returns) - rf_rate
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        return excess_returns / volatility

    @staticmethod
    def _calculate_max_drawdown(balances: List[float]) -> float:
        """最大ドローダウン計算"""
        if not balances:
            return 0.0
        
        balances = np.array(balances)
        cummax = np.maximum.accumulate(balances)
        drawdown = (balances - cummax) / cummax
        return float(np.min(drawdown))
