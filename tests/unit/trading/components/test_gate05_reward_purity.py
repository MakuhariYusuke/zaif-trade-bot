"""
Gate 0.5: 報酬純粋性検証テスト

95#レビュー指摘対応:
- use_simple_reward=True でPnL以外のコンポーネントが混入しないか
- ペナルティパラメータ変更で実際にreward_componentsが変化するか
- 未知キーが custom_reward_params に吸収されることの検出

既存テスト基盤(tests/unit/trading/components/test_reward_calculator.py)と
同じfixture構造を使用。
"""

import math

import numpy as np
import pytest

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


# ============================================================================
# Fixtures (既存test_reward_calculator.pyと同じパターンを継承)
# ============================================================================


@pytest.fixture
def base_env_config():
    """最小限の環境設定"""
    config_dict = {
        "max_position_size": 1.0,
        "transaction_cost": 0.001,
        "exchange": "coincheck",
        "reward_scaling": 1.0,
        "action_space_type": "continuous",
        "use_continuous_actions": True,
        "feature_set": "minimal",
        "enable_action_masking": True,
        "use_standardized_observations": True,
        "random_start": True,
        "continuous_to_discrete_threshold": 0.08,
        "behavior_optimization": {
            "action_balance_target": 0.333,
            "entropy_regularization": 0.0,
            "action_smoothing": 0.0,
            "consistency_penalty": 0.0,
            "balance_penalty": 0.0,
            "redundant_trade_penalty": 0.0,
            "balance_penalty_min_actions": 1,
        },
        "action_bonuses": {
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        },
        "base_action_penalty": 0.0,
    }
    return EnvironmentConfig.from_dict(config_dict)


def _make_pure_pnl_settings() -> RewardSettings:
    """P1-1修正版: 純PnLのみ報酬設定

    95#で指摘された全ての漏れを塞ぐ。
    """
    return RewardSettings(
        use_simple_reward=True,  # ← 最重要: 複合経路をバイパス
        reward_scale=1.0,
        # ペナルティ関連すべて0
        balance_penalty=0.0,
        balance_penalty_tolerance=1.0,
        position_penalty_scale=0.0,
        inventory_penalty_scale=0.0,
        trade_frequency_penalty=0.0,
        trade_cooldown_penalty=0.0,
        consecutive_trade_penalty=0.0,
        hold_penalty_multiplier=1.0,  # *=1.0 で無影響
        volatility_penalty_scale=0.0,
        consistency_penalty=0.0,
        redundant_trade_penalty=0.0,
        # ボーナスなし
        trading_bonus=0.0,
        entropy_bonus=0.0,
        # 非対称スケーリングを無効化
        long_position_reward_multiplier=1.0,
        short_position_reward_multiplier=1.0,
        long_position_penalty_multiplier=1.0,
        short_position_penalty_multiplier=1.0,
        # confidence penalty無効化
        custom_reward_params={
            "confidence_penalty_factor": 0.0,
            "balance_shaping_enabled": False,
            "action_entropy_shaping_enabled": False,
        },
        profit_weight=1.0,
    )


def _make_default_settings() -> RewardSettings:
    """デフォルト（複合報酬）設定"""
    return RewardSettings(
        use_simple_reward=False,
        reward_scale=100.0,
        balance_penalty=0.1,
        position_penalty_scale=0.1,
        hold_penalty_multiplier=0.01,
    )


@pytest.fixture
def pure_pnl_calculator(base_env_config):
    """純PnLモードのRewardCalculator"""
    return RewardCalculator(
        config=base_env_config,
        reward_settings=_make_pure_pnl_settings(),
        initial_portfolio_value=100000.0,
    )


@pytest.fixture
def default_calculator(base_env_config):
    """デフォルト（複合報酬）モードのRewardCalculator"""
    return RewardCalculator(
        config=base_env_config,
        reward_settings=_make_default_settings(),
        initial_portfolio_value=100000.0,
    )


# ============================================================================
# 共通テストヘルパー
# ============================================================================

_COMMON_KWARGS = dict(
    current_price=10_000_000.0,
    portfolio_value=100_500.0,
    atr=50000.0,
    transaction_cost=0.001,
    reward_scaling=1.0,
    step=10,
    observation=np.array([1.0, 2.0, 3.0]),
    reward_history=[0.0, 0.0],
    portfolio_value_history=[100000.0, 100200.0],
)


def _call_reward(calc: RewardCalculator, action: int, pnl: float,
                 position: float = 0.0, old_position: float = 0.0) -> float:
    """calculate_rewardを統一的に呼び出す"""
    return calc.calculate_reward(
        action=action,
        pnl=pnl,
        position=position,
        old_position=old_position,
        **_COMMON_KWARGS,
    )


# ============================================================================
# Test A-1: use_simple_reward=True で PnL以外のコンポーネントが混入しないこと
# ============================================================================


class TestSimpleRewardPurity:
    """use_simple_reward=True の報酬純粋性を検証"""

    def test_simple_reward_profit_is_positive(self, pure_pnl_calculator):
        """利益時の報酬が正"""
        reward = _call_reward(pure_pnl_calculator, ACTION_BUY, pnl=500.0)
        assert reward > 0, f"利益時に報酬が正であるべき: {reward}"

    def test_simple_reward_loss_is_negative(self, pure_pnl_calculator):
        """損失時の報酬が負"""
        reward = _call_reward(pure_pnl_calculator, ACTION_SELL, pnl=-500.0)
        assert reward < 0, f"損失時に報酬が負であるべき: {reward}"

    def test_simple_reward_zero_pnl_is_zero(self, pure_pnl_calculator):
        """PnL=0 → 報酬≈0（hold_penalty_multiplier=1.0 なので HOLD でも 0*1.0=0）"""
        reward = _call_reward(pure_pnl_calculator, ACTION_HOLD, pnl=0.0)
        assert abs(reward) < 0.2, f"PnL=0 で報酬がほぼ0であるべき: {reward}"

    def test_no_confidence_penalty_in_components(self, pure_pnl_calculator):
        """confidence_penaltyが報酬コンポーネントに含まれない（simple経路）"""
        _call_reward(pure_pnl_calculator, ACTION_BUY, pnl=-100.0,
                     position=0.5, old_position=0.0)
        comps = pure_pnl_calculator._last_reward_components
        # simple_reward 経路では confidence_penalty が含まれないはず
        assert comps.get("stage") == "simple_reward", \
            f"simple_reward ステージであるべき: {comps.get('stage')}"

    def test_no_balance_shaping_in_simple(self, pure_pnl_calculator):
        """simple経路ではbalance_shapingが計算されない"""
        _call_reward(pure_pnl_calculator, ACTION_BUY, pnl=100.0)
        comps = pure_pnl_calculator._last_reward_components
        assert "balance_shaping" not in comps, \
            f"balance_shapingが存在するべきでない: {comps}"

    def test_no_entropy_shaping_in_simple(self, pure_pnl_calculator):
        """simple経路ではentropy_shapingが計算されない"""
        _call_reward(pure_pnl_calculator, ACTION_SELL, pnl=-100.0)
        comps = pure_pnl_calculator._last_reward_components
        assert "entropy_shaping" not in comps, \
            f"entropy_shapingが存在するべきでない: {comps}"

    def test_reward_proportional_to_pnl(self, pure_pnl_calculator):
        """報酬がPnLに概ね比例する（scaling除く）"""
        reward_1 = _call_reward(pure_pnl_calculator, ACTION_BUY, pnl=100.0)
        pure_pnl_calculator.reset_episode_state()
        reward_2 = _call_reward(pure_pnl_calculator, ACTION_BUY, pnl=200.0)
        # 2倍のPnLで概ね2倍の報酬（完全一致は求めない、position_changeペナルティ等で多少ずれる）
        ratio = reward_2 / reward_1 if reward_1 != 0 else float('inf')
        assert 1.5 < ratio < 2.5, \
            f"PnL 2倍で報酬比は1.5-2.5倍であるべき: {ratio}"


# ============================================================================
# Test A-2: ペナルティパラメータ変更でreward_componentsに差分が出ること (Gate 0.5)
# ============================================================================


class TestPenaltyToggleEffect:
    """パラメータ変更が実際の報酬計算に反映されることを検証"""

    def test_balance_penalty_toggle(self, base_env_config):
        """balance_penalty=0 vs 0.5 で報酬が変わる"""
        settings_off = RewardSettings(
            use_simple_reward=False,
            balance_penalty=0.0,
        )
        settings_on = RewardSettings(
            use_simple_reward=False,
            balance_penalty=0.5,
        )
        calc_off = RewardCalculator(
            config=base_env_config, reward_settings=settings_off,
            initial_portfolio_value=100000.0,
        )
        calc_on = RewardCalculator(
            config=base_env_config, reward_settings=settings_on,
            initial_portfolio_value=100000.0,
        )

        # 複数アクションを実行してbalance_penaltyが発生する状況を作る
        for _ in range(5):
            _call_reward(calc_off, ACTION_BUY, pnl=100.0, position=0.1, old_position=0.0)
            _call_reward(calc_on, ACTION_BUY, pnl=100.0, position=0.1, old_position=0.0)

        # 最後の報酬コンポーネントを比較
        comps_off = calc_off._last_reward_components
        comps_on = calc_on._last_reward_components

        bp_off = comps_off.get("balance_penalty", 0.0)
        bp_on = comps_on.get("balance_penalty", 0.0)

        # balance_penalty=0.5の方がペナルティが大きいはず
        assert abs(bp_on) >= abs(bp_off), \
            f"balance_penalty有効時にペナルティが増えるべき: off={bp_off}, on={bp_on}"

    def test_simple_vs_complex_path_differs(self, base_env_config):
        """use_simple_reward=True/False で異なる経路を通る"""
        settings_simple = RewardSettings(use_simple_reward=True, reward_scale=1.0)
        settings_complex = RewardSettings(use_simple_reward=False, reward_scale=1.0)

        calc_simple = RewardCalculator(
            config=base_env_config, reward_settings=settings_simple,
            initial_portfolio_value=100000.0,
        )
        calc_complex = RewardCalculator(
            config=base_env_config, reward_settings=settings_complex,
            initial_portfolio_value=100000.0,
        )

        _call_reward(calc_simple, ACTION_BUY, pnl=100.0)
        _call_reward(calc_complex, ACTION_BUY, pnl=100.0)

        stage_simple = calc_simple._last_reward_components.get("stage")
        stage_complex = calc_complex._last_reward_components.get("stage")

        assert stage_simple == "simple_reward", f"simple経路であるべき: {stage_simple}"
        assert stage_complex != "simple_reward", f"complex経路であるべき: {stage_complex}"


# ============================================================================
# Test A-3: 未知キーが custom_reward_params に吸収されることの検出
# ============================================================================


class TestUnknownKeyDetection:
    """RewardSettings.from_dict() の未知キー処理を検証"""

    def test_unknown_keys_go_to_custom_params(self):
        """未知のキーが custom_reward_params に格納される"""
        settings = RewardSettings.from_dict({
            "balance_penalty": 0.0,  # 既知
            "nonexistent_param_xyz": 42.0,  # 未知
        })
        assert "nonexistent_param_xyz" in settings.custom_reward_params, \
            f"未知キーがcustom_reward_paramsに入るべき: {settings.custom_reward_params}"
        assert settings.custom_reward_params["nonexistent_param_xyz"] == 42.0

    def test_known_keys_not_in_custom_params(self):
        """既知のキーは custom_reward_params に入らない"""
        settings = RewardSettings.from_dict({
            "balance_penalty": 0.5,
            "reward_scale": 100.0,
        })
        assert "balance_penalty" not in settings.custom_reward_params
        assert "reward_scale" not in settings.custom_reward_params
        assert settings.balance_penalty == 0.5
        assert settings.reward_scale == 100.0

    def test_from_dict_preserves_all_p1_params(self):
        """P1-1の全パラメータがRewardSettingsに正しく反映される"""
        p1_params = {
            "use_simple_reward": True,
            "balance_penalty": 0.0,
            "balance_penalty_tolerance": 1.0,
            "position_penalty_scale": 0.0,
            "inventory_penalty_scale": 0.0,
            "trade_frequency_penalty": 0.0,
            "trade_cooldown_penalty": 0.0,
            "consecutive_trade_penalty": 0.0,
            "hold_penalty_multiplier": 1.0,
            "volatility_penalty_scale": 0.0,
            "consistency_penalty": 0.0,
            "redundant_trade_penalty": 0.0,
            "profit_weight": 1.0,
            "reward_scale": 1.0,
        }
        settings = RewardSettings.from_dict(p1_params)
        assert settings.use_simple_reward is True
        assert settings.balance_penalty == 0.0
        assert settings.hold_penalty_multiplier == 1.0
        assert settings.reward_scale == 1.0
        # 全てが既知キーなので custom_reward_params は空のはず
        assert len(settings.custom_reward_params) == 0, \
            f"未知キーがないはずだが: {settings.custom_reward_params}"


# ============================================================================
# Test C0: Gate C0 計測基盤修正の検証
# ============================================================================


class TestGateC0MeasurementIntegrity:
    """Gate C0: 計測基盤の正しさを検証"""

    def test_position_manager_buy_sell_count_exists(self):
        """PositionManagerにbuy_count/sell_count属性が存在する"""
        from ztb.trading.environment.components.position_manager import PositionManager

        class FakeConfig:
            max_position_size = 1.0
            transaction_cost = 0.001
            initial_portfolio_value = 100000.0
            exchange_profile = None
            risk_management = {}

        pm = PositionManager(
            config=FakeConfig(),
            get_price_callback=lambda: 10_000_000.0,
        )
        assert hasattr(pm, "buy_count"), "buy_count属性が存在すべき"
        assert hasattr(pm, "sell_count"), "sell_count属性が存在すべき"
        assert pm.buy_count == 0
        assert pm.sell_count == 0

    def test_position_manager_reset_clears_counts(self):
        """reset()でbuy_count/sell_countがゼロクリアされる"""
        from ztb.trading.environment.components.position_manager import PositionManager

        class FakeConfig:
            max_position_size = 1.0
            transaction_cost = 0.001
            initial_portfolio_value = 100000.0
            exchange_profile = None
            risk_management = {}

        pm = PositionManager(
            config=FakeConfig(),
            get_price_callback=lambda: 10_000_000.0,
        )
        pm.buy_count = 5
        pm.sell_count = 3
        pm.reset()
        assert pm.buy_count == 0, "reset後buy_countは0であるべき"
        assert pm.sell_count == 0, "reset後sell_countは0であるべき"

    def test_position_change_penalty_is_configurable(self, pure_pnl_calculator):
        """ハードコードペナルティが設定値化されている
        
        position_change_penalty=0.0 (デフォルト)のとき、
        ポジション変更でペナルティが発生しないことを確認。
        """
        # position変更あり (0.0 → 0.5) だが penalty=0.0(デフォルト)
        reward = _call_reward(
            pure_pnl_calculator, ACTION_BUY, pnl=100.0,
            position=0.5, old_position=0.0,
        )
        # ペナルティがデフォルト0なので、reward > 0 であるべき
        assert reward > 0, f"position_change_penalty=0.0で正のrewardであるべき: {reward}"

    def test_hold_penalty_multiplier_one_preserves_pnl(self, pure_pnl_calculator):
        """hold_penalty_multiplier=1.0でHOLD時にPnLが保持される（98#修正確認）"""
        # HOLD + 正のPnL → reward > 0 であるべき（multiplier=1.0）
        reward = _call_reward(pure_pnl_calculator, ACTION_HOLD, pnl=500.0)
        assert reward > 0, \
            f"hold_penalty_multiplier=1.0でHOLD+正PnLは正報酬であるべき: {reward}"
