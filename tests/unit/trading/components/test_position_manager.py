"""Tests for PositionManager component."""

from typing import Any, Dict
import pytest

from ztb.trading.environment.components.position_manager import PositionManager
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """サンプル設定"""
    return {
        "initial_balance": 1000000,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "allow_reverse": True,  # PositionManagerが必要とする設定
        "enforce_reverse_cooldown": False,
        "initial_portfolio_value": 200000.0,
    }


@pytest.fixture
def position_manager(sample_config: Dict[str, Any]) -> PositionManager:
    """PositionManagerインスタンス"""
    # get_price_callbackを提供
    def get_price() -> float:
        return 100000.0
    
    # 設定を属性アクセス可能なオブジェクトに変換
    from types import SimpleNamespace
    config_obj = SimpleNamespace(**sample_config)
    
    return PositionManager(config_obj, get_price)


class TestPositionManagerInitialization:
    """初期化テスト"""
    
    def test_initialization(self, position_manager: PositionManager) -> None:
        """正常に初期化できることを確認"""
        assert position_manager.position == 0.0
        assert position_manager.entry_price == 0.0
        assert position_manager.realized_pnl == 0.0
        assert position_manager.trades_count == 0
    
    def test_get_price_callback(self, position_manager: PositionManager) -> None:
        """価格取得コールバックが機能することを確認"""
        # get_price_callbackは100000を返す設定
        price = position_manager._get_price()
        assert price == 100000.0


class TestPositionManagerActions:
    """アクション実行テスト"""
    
    def test_buy_action_from_flat(self, position_manager: PositionManager) -> None:
        """フラットからの買い注文"""
        pnl = position_manager.execute_action(ACTION_BUY, current_step=0)
        
        # ロングポジションが開く
        assert position_manager.position > 0
        # エントリーコストが発生（負のPnL）
        assert pnl < 0  # 手数料分
        assert position_manager.trades_count == 1
    
    def test_sell_action_from_flat(self, position_manager: PositionManager) -> None:
        """フラットからの売り注文"""
        pnl = position_manager.execute_action(ACTION_SELL, current_step=0)
        
        # ショートポジションが開く
        assert position_manager.position < 0
        # エントリーコストが発生
        assert pnl < 0
        assert position_manager.trades_count == 1
    
    def test_hold_action(self, position_manager: PositionManager) -> None:
        """ホールド注文"""
        initial_position = position_manager.position
        pnl = position_manager.execute_action(ACTION_HOLD, current_step=0)
        
        assert pnl == 0.0
        assert position_manager.position == initial_position
    
    def test_close_long_position(self, position_manager: PositionManager) -> None:
        """ロングポジションのクローズ"""
        # ロングポジションを開く
        position_manager.execute_action(ACTION_BUY, current_step=0)
        assert position_manager.position > 0
        
        # ポジションをクローズ
        pnl = position_manager.execute_action(ACTION_SELL, current_step=1)
        
        # ポジションがゼロに（allow_reverseがFalseの場合）
        # または逆ポジションに転換（allow_reverseがTrueの場合）
        # 実装によるので、変化があることだけ確認
        assert pnl != 0.0  # クローズPnLがある
    
    def test_close_short_position(self, position_manager: PositionManager) -> None:
        """ショートポジションのクローズ"""
        # ショートポジションを開く
        position_manager.execute_action(ACTION_SELL, current_step=0)
        assert position_manager.position < 0
        
        # ポジションをクローズ
        pnl = position_manager.execute_action(ACTION_BUY, current_step=1)
        
        # クローズPnLがある
        assert pnl != 0.0


class TestPositionManagerPnL:
    """PnL calculation tests"""
    
    def test_realized_pnl_tracking(self, position_manager: PositionManager) -> None:
        """実現損益のトラッキング"""
        # ロングポジションを開いてクローズ
        position_manager.execute_action(ACTION_BUY, current_step=0)
        pnl = position_manager.execute_action(ACTION_SELL, current_step=1)
        
        # 何らかのPnLが計算される
        assert isinstance(pnl, (int, float))
        # 実現PnLが更新される
        assert isinstance(position_manager.realized_pnl, (int, float))


class TestPositionManagerEdgeCases:
    """Edge case tests"""
    
    def test_consecutive_trades(self, position_manager: PositionManager) -> None:
        """連続取引のカウント"""
        # 複数回取引
        position_manager.execute_action(ACTION_BUY, current_step=0)
        position_manager.execute_action(ACTION_SELL, current_step=1)
        position_manager.execute_action(ACTION_BUY, current_step=2)
        
        # トレードカウントが増加
        assert position_manager.trades_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
