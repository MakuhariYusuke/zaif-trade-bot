#!/usr/bin/env python3
"""
ActionConverterV456: アクション変換の統一化

目的: train / eval / live で同じアクション変換ロジックを使用する
・ランダムな差異を排除
・パフォーマンス検証の有効性を確保
・本番適用時の予測可能性を向上

統一仕様:
- Continuous Action Range: [-1.0, 1.0]
- Buy Threshold: >= 0.3333
- Sell Threshold: <= -0.3333
- Neutral Zone: -0.3333 < action < 0.3333 → HOLD
"""

import numpy as np
from typing import Literal
import logging

logger = logging.getLogger(__name__)

# アクション定義
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

class ActionConverterV456:
    """
    統一的なアクション変換クラス
    
    SAC が出力する連続値 [-1, 1] を
    実際の売買シグナル {HOLD, BUY, SELL} に変換する
    """
    
    # 閾値（最終仕様）
    CONTINUOUS_BUY_THRESHOLD = 1.0 / 3.0   # 0.3333...
    CONTINUOUS_SELL_THRESHOLD = -1.0 / 3.0  # -0.3333...
    
    # アクション定義
    ACTION_MAPPING = {
        ACTION_HOLD: "HOLD",
        ACTION_BUY: "BUY",
        ACTION_SELL: "SELL",
    }
    
    @staticmethod
    def clip_action(action: np.ndarray | float) -> float:
        """アクションを [-1, 1] にクリップ"""
        action = float(action) if isinstance(action, np.ndarray) else float(action)
        return float(np.clip(action, -1.0, 1.0))
    
    @staticmethod
    def continuous_to_discrete(action: np.ndarray | float) -> int:
        """
        連続値アクション → 離散的な売買シグナル
        
        Args:
            action: [-1, 1] の連続値
        
        Returns:
            ACTION_HOLD, ACTION_BUY, ACTION_SELL のいずれか
        """
        action = ActionConverterV456.clip_action(action)
        
        if action >= ActionConverterV456.CONTINUOUS_BUY_THRESHOLD:
            return ACTION_BUY
        elif action <= ActionConverterV456.CONTINUOUS_SELL_THRESHOLD:
            return ACTION_SELL
        else:
            return ACTION_HOLD
    
    @staticmethod
    def continuous_to_position_size(
        action: np.ndarray | float,
        max_position_size: float = 0.01,
    ) -> float:
        """
        連続値アクション → ポジションサイズ (BTC単位)
        
        アクションの絶対値をポジションサイズにマッピング
        符号は方向（買い/売り）を表す
        
        Args:
            action: [-1, 1] の連続値
            max_position_size: 最大ポジションサイズ (BTC)
        
        Returns:
            ポジションサイズ (正 = 買い, 負 = 売り)
        
        例：
            action=0.5, max=0.01 → position=0.005 BTC (買い)
            action=-0.8, max=0.01 → position=-0.008 BTC (売り)
        """
        action = ActionConverterV456.clip_action(action)
        
        # 絶対値をポジションサイズに変換
        position_intensity = abs(action)  # [0, 1]
        position_size = position_intensity * max_position_size
        
        # 符号を適用
        if action < 0:
            position_size = -position_size
        
        return position_size
    
    @staticmethod
    def action_to_confidence(action: np.ndarray | float) -> float:
        """
        アクションの確実度（信頼度）を計算
        
        絶対値が大きいほど確実度が高い
        
        Args:
            action: [-1, 1] の連続値
        
        Returns:
            [0, 1] の確実度スコア
        """
        action = ActionConverterV456.clip_action(action)
        return abs(action)
    
    @staticmethod
    def get_action_name(discrete_action: int) -> str:
        """離散的アクションの名前を取得"""
        return ActionConverterV456.ACTION_MAPPING.get(
            discrete_action, f"UNKNOWN({discrete_action})"
        )
    
    @staticmethod
    def validate_action(action: np.ndarray | float) -> bool:
        """アクション値が有効か確認"""
        if isinstance(action, np.ndarray):
            action = float(action.item() if action.size == 1 else action[0])
        else:
            action = float(action)
        
        return -1.0 <= action <= 1.0

class ActionAnalyzer:
    """アクション分布の分析"""
    
    def __init__(self):
        self.action_history = []
        self.discrete_action_history = []
    
    def record_action(self, action: float):
        """アクションを記録"""
        action = ActionConverterV456.clip_action(action)
        self.action_history.append(action)
        
        discrete = ActionConverterV456.continuous_to_discrete(action)
        self.discrete_action_history.append(discrete)
    
    def get_statistics(self) -> dict:
        """統計情報を取得"""
        if not self.action_history:
            return {}
        
        actions = np.array(self.action_history)
        discrete = np.array(self.discrete_action_history)
        
        return {
            'action_mean': float(np.mean(actions)),
            'action_std': float(np.std(actions)),
            'action_min': float(np.min(actions)),
            'action_max': float(np.max(actions)),
            'hold_ratio': float(np.sum(discrete == ACTION_HOLD) / len(discrete)),
            'buy_ratio': float(np.sum(discrete == ACTION_BUY) / len(discrete)),
            'sell_ratio': float(np.sum(discrete == ACTION_SELL) / len(discrete)),
            'total_actions': len(self.action_history),
        }
    
    def reset(self):
        """統計をリセット"""
        self.action_history = []
        self.discrete_action_history = []

if __name__ == '__main__':
    # テスト
    print("=" * 70)
    print("ActionConverterV456 テスト")
    print("=" * 70)
    
    test_actions = [-1.0, -0.5, -0.33, 0.0, 0.33, 0.5, 1.0]
    
    print("\n[1] 連続値 → 離散的アクション")
    print("-" * 70)
    print(f"{'Action':>10} {'Threshold':>20} {'Discrete':>15} {'Name':>10}")
    print("-" * 70)
    
    for action in test_actions:
        discrete = ActionConverterV456.continuous_to_discrete(action)
        name = ActionConverterV456.get_action_name(discrete)
        
        threshold_info = ""
        if action >= ActionConverterV456.CONTINUOUS_BUY_THRESHOLD:
            threshold_info = f"≥ {ActionConverterV456.CONTINUOUS_BUY_THRESHOLD:.4f}"
        elif action <= ActionConverterV456.CONTINUOUS_SELL_THRESHOLD:
            threshold_info = f"≤ {ActionConverterV456.CONTINUOUS_SELL_THRESHOLD:.4f}"
        else:
            threshold_info = "Neutral Zone"
        
        print(f"{action:>10.4f} {threshold_info:>20} {discrete:>15} {name:>10}")
    
    print("\n[2] アクション → ポジションサイズ")
    print("-" * 70)
    max_pos = 0.01
    print(f"{'Action':>10} {'Position (BTC)':>20} {'Direction':>15}")
    print("-" * 70)
    
    for action in test_actions:
        position = ActionConverterV456.continuous_to_position_size(action, max_pos)
        direction = "買い" if position > 0 else "売り" if position < 0 else "ノーポジ"
        print(f"{action:>10.4f} {position:>20.6f} {direction:>15}")
    
    print("\n[3] アクション分析")
    print("-" * 70)
    
    analyzer = ActionAnalyzer()
    
    # ランダムアクションを記録
    np.random.seed(42)
    random_actions = np.random.uniform(-1, 1, 1000)
    
    for action in random_actions:
        analyzer.record_action(action)
    
    stats = analyzer.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:>20}: {value:>10.4f}")
        else:
            print(f"  {key:>20}: {value:>10}")
    
    print("\n" + "=" * 70)
    print("✓ テスト完了")
    print("=" * 70)
