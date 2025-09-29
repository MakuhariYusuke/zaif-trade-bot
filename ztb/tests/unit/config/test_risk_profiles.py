"""Unit tests for risk profile management."""

from ztb.config.schema import RiskProfileConfig
from ztb.live.risk_profiles import RiskProfileManager


class TestRiskProfileManager:
    """Test risk profile management functionality."""

    def test_add_and_get_profile(self):
        """Test adding and retrieving risk profiles."""
        manager = RiskProfileManager()

        profile = RiskProfileConfig(
            name="test_profile",
            max_position_size=0.15,
            max_daily_loss=0.03,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            max_open_positions=7,
            risk_per_trade=0.015,
            max_leverage=1.5,
            cooldown_period=90,
        )

        manager.add_profile(profile)
        retrieved = manager.get_profile("test_profile")

        assert retrieved is not None
        assert retrieved.name == "test_profile"
        assert retrieved.max_position_size == 0.15

    def test_list_profiles(self):
        """Test listing all risk profiles."""
        manager = RiskProfileManager()

        profile1 = RiskProfileConfig(name="profile1")
        profile2 = RiskProfileConfig(name="profile2")

        manager.add_profile(profile1)
        manager.add_profile(profile2)

        profiles = manager.list_profiles()
        assert len(profiles) == 2
        assert "profile1" in profiles
        assert "profile2" in profiles

    def test_validate_position_size(self):
        """Test position size validation."""
        manager = RiskProfileManager()

        profile = RiskProfileConfig(
            name="test",
            max_position_size=0.1,  # 10% of portfolio
        )
        manager.add_profile(profile)

        # Valid position size
        assert manager.validate_position_size("test", 1000, 10000)  # 1000 / 10000 = 0.1

        # Invalid position size
        assert not manager.validate_position_size(
            "test", 1500, 10000
        )  # 1500 / 10000 = 0.15

    def test_validate_daily_loss(self):
        """Test daily loss validation."""
        manager = RiskProfileManager()

        profile = RiskProfileConfig(
            name="test",
            max_daily_loss=0.05,  # 5% of portfolio
        )
        manager.add_profile(profile)

        # Valid daily loss
        assert manager.validate_daily_loss("test", -400, 10000)  # -400 / 10000 = -0.04

        # Invalid daily loss
        assert not manager.validate_daily_loss(
            "test", -600, 10000
        )  # -600 / 10000 = -0.06

    def test_calculate_position_size(self):
        """Test position size calculation."""
        manager = RiskProfileManager()

        profile = RiskProfileConfig(
            name="test",
            max_position_size=0.2,
            risk_per_trade=0.01,  # 1% risk per trade
            stop_loss_pct=0.02,  # 2% stop loss
        )
        manager.add_profile(profile)

        portfolio_value = 10000
        volatility = 0.02  # 2% volatility

        position_size = manager.calculate_position_size(
            "test", portfolio_value, volatility
        )

        # Expected: risk_amount = 0.01 * 10000 = 100
        # stop_loss_amount = 100 / 0.02 = 5000
        # volatility_adjustment = min(1.0, 0.02 / 0.02) = 1.0
        # position_size = 5000 * 1.0 = 5000
        # But max_size = 0.2 * 10000 = 2000, so should be 2000

        assert position_size == 2000

    def test_unknown_profile(self):
        """Test handling of unknown profiles."""
        manager = RiskProfileManager()

        assert manager.get_profile("unknown") is None
        assert not manager.validate_position_size("unknown", 1000, 10000)
        assert manager.calculate_position_size("unknown", 10000, 0.02) == 0.0
