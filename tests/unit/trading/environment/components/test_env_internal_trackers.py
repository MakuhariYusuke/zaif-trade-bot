import pytest
from ztb.trading.environment.components.env_internal_trackers import EnvInternalTracker

def test_initial_state():
    tracker = EnvInternalTracker()
    features = tracker.get_feature_vector()
    
    assert features["inventory_pressure"] == 0.0
    assert features["loss_risk"] == 0.0
    assert features["time_in_market"] == 0.0

def test_update_step_time_in_market():
    tracker = EnvInternalTracker()
    
    # 状態なし
    tracker.update_step(has_position=False)
    assert tracker.get_feature_vector()["time_in_market"] == 0.0
    
    # ポジション保持
    for _ in range(50):
        tracker.update_step(has_position=True)
        
    # time_in_marketは 50 / 100 = 0.5 のはず
    assert tracker.get_feature_vector()["time_in_market"] == pytest.approx(0.5)
    
    # max clip = 5.0 を超えるかテスト
    for _ in range(500):
        tracker.update_step(has_position=True)
    assert tracker.get_feature_vector()["time_in_market"] == 5.0  # clipped
    
    # resetの確認
    tracker.reset()
    assert tracker.get_feature_vector()["time_in_market"] == 0.0

def test_inventory_decay_and_pressure():
    tracker = EnvInternalTracker(inventory_decay=0.9)
    
    # Buy 0.01 BTC -> pressure = 1 * 0.01 * 50.0 = 0.5
    tracker.on_trade(direction=1, amount=0.01)
    assert tracker.get_feature_vector()["inventory_pressure"] == pytest.approx(0.5)
    
    # Decay applied
    tracker.update_step(has_position=True)
    assert tracker.get_feature_vector()["inventory_pressure"] == pytest.approx(0.45) # 0.5 * 0.9
    
    # Sell 0.02 BTC -> pressure = 0.45 + (-1 * 0.02 * 50.0) = -0.55
    tracker.on_trade(direction=-1, amount=0.02)
    assert tracker.get_feature_vector()["inventory_pressure"] == pytest.approx(-0.55)

def test_loss_risk_decay():
    tracker = EnvInternalTracker(loss_decay=0.8)
    
    # No loss
    tracker.on_trade_close(realized_pnl_pct=0.01)  # Win 1%
    assert tracker.get_feature_vector()["loss_risk"] == 0.0
    
    # Loss event! 2% loss -> risk = 0 + 0.02 * 100.0 = 2.0
    tracker.on_trade_close(realized_pnl_pct=-0.02)
    assert tracker.get_feature_vector()["loss_risk"] == pytest.approx(2.0)
    
    # Decay applied
    tracker.update_step(has_position=False)
    assert tracker.get_feature_vector()["loss_risk"] == pytest.approx(1.6) # 2.0 * 0.8
    
    # Reset
    tracker.reset()
    assert tracker.get_feature_vector()["loss_risk"] == 0.0
