"""
Unit tests for fee model implementations
手数料モデル実装の単体テスト
"""

from ztb.utils.fee_model import (
    ExchangeFeeModel,
    FeeModelFactory,
    FixedFeeModel,
    TieredFeeModel,
    load_fee_model_from_config,
)


class TestFixedFeeModel:
    """Test FixedFeeModel"""

    def test_calculate_fee_buy(self):
        """Test fee calculation for buy trades"""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)
        fee = model.calculate_fee(10000.0, "buy")
        assert fee == 10.0  # 10000 * 0.001

    def test_calculate_fee_sell(self):
        """Test fee calculation for sell trades"""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)
        fee = model.calculate_fee(10000.0, "sell")
        assert fee == 20.0  # 10000 * 0.002

    def test_get_fee_rate_buy(self):
        """Test getting buy fee rate"""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)
        rate = model.get_fee_rate("buy")
        assert rate == 0.001

    def test_get_fee_rate_sell(self):
        """Test getting sell fee rate"""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)
        rate = model.get_fee_rate("sell")
        assert rate == 0.002

    def test_default_rates(self):
        """Test default fee rates"""
        model = FixedFeeModel()
        assert model.get_fee_rate("buy") == 0.001
        assert model.get_fee_rate("sell") == 0.001


class TestTieredFeeModel:
    """Test TieredFeeModel"""

    def test_calculate_fee(self):
        """Test fee calculation with tiered rates"""
        model = TieredFeeModel()
        fee = model.calculate_fee(10000.0, "buy")
        assert fee == 10.0  # 10000 * 0.001 (base rate)

    def test_get_fee_rate_buy(self):
        """Test getting buy fee rate"""
        model = TieredFeeModel()
        rate = model.get_fee_rate("buy")
        assert rate == 0.001  # Base rate

    def test_get_fee_rate_sell(self):
        """Test getting sell fee rate"""
        model = TieredFeeModel()
        rate = model.get_fee_rate("sell")
        assert rate == 0.001  # Base rate

    def test_custom_tiers(self):
        """Test custom tier configuration"""
        custom_tiers = {
            "buy_tiers": [[0, 0.002], [5000, 0.001]],
            "sell_tiers": [[0, 0.003], [5000, 0.002]],
        }
        model = TieredFeeModel(tiers=custom_tiers)
        assert model.get_fee_rate("buy") == 0.002
        assert model.get_fee_rate("sell") == 0.003


class TestExchangeFeeModel:
    """Test ExchangeFeeModel"""

    def test_initialization_default(self):
        """Test initialization with default exchange fees"""
        model = ExchangeFeeModel()
        assert model.current_exchange == "binance"
        assert model.exchange_fees["coincheck"]["buy"] == 0.0
        assert model.exchange_fees["bitflyer"]["buy"] == 0.001
        assert model.exchange_fees["binance"]["buy"] == 0.001

    def test_initialization_custom(self):
        """Test initialization with custom exchange fees"""
        custom_fees = {"test_exchange": {"buy": 0.002, "sell": 0.003}}
        model = ExchangeFeeModel(exchange_fees=custom_fees)
        assert model.exchange_fees["test_exchange"]["buy"] == 0.002
        assert model.exchange_fees["test_exchange"]["sell"] == 0.003

    def test_set_exchange_valid(self):
        """Test setting exchange to valid value"""
        model = ExchangeFeeModel()
        model.set_exchange("bitflyer")
        assert model.current_exchange == "bitflyer"

    def test_set_exchange_invalid(self):
        """Test setting exchange to invalid value (falls back to binance)"""
        model = ExchangeFeeModel()
        model.set_exchange("invalid_exchange")  # Should fallback to binance
        assert model.current_exchange == "binance"  # Fallback behavior

    def test_calculate_fee_coincheck(self):
        """Test fee calculation for coincheck (0% fees)"""
        model = ExchangeFeeModel()
        model.set_exchange("coincheck")
        fee = model.calculate_fee(10000.0, "buy")
        assert fee == 0.0

    def test_calculate_fee_bitflyer(self):
        """Test fee calculation for bitflyer (0.1% fees)"""
        model = ExchangeFeeModel()
        model.set_exchange("bitflyer")
        fee = model.calculate_fee(10000.0, "buy")
        assert fee == 10.0  # 10000 * 0.001

    def test_calculate_fee_binance(self):
        """Test fee calculation for binance (0.1% fees)"""
        model = ExchangeFeeModel()
        model.set_exchange("binance")
        fee = model.calculate_fee(10000.0, "sell")
        assert fee == 10.0  # 10000 * 0.001

    def test_get_fee_rate_buy_coincheck(self):
        """Test getting buy fee rate for coincheck"""
        model = ExchangeFeeModel()
        model.set_exchange("coincheck")
        rate = model.get_fee_rate("buy")
        assert rate == 0.0

    def test_get_fee_rate_sell_bitflyer(self):
        """Test getting sell fee rate for bitflyer"""
        model = ExchangeFeeModel()
        model.set_exchange("bitflyer")
        rate = model.get_fee_rate("sell")
        assert rate == 0.001

    def test_get_fee_rate_unknown_exchange(self):
        """Test getting fee rate for unknown exchange (fallback to binance)"""
        model = ExchangeFeeModel()
        model.set_exchange("unknown_exchange")
        rate = model.get_fee_rate("buy")
        assert rate == 0.001  # Default binance rate

    def test_case_insensitive_trade_type(self):
        """Test that trade type is case insensitive"""
        model = ExchangeFeeModel()
        model.set_exchange("bitflyer")
        assert model.get_fee_rate("BUY") == 0.001
        assert model.get_fee_rate("Sell") == 0.001


class TestFeeModelFactory:
    """Test FeeModelFactory"""

    def test_create_fixed_model(self):
        """Test creating fixed fee model"""
        config = {"buy_fee_rate": 0.002, "sell_fee_rate": 0.003}
        model = FeeModelFactory.create_fee_model("fixed", config)
        assert isinstance(model, FixedFeeModel)
        assert model.get_fee_rate("buy") == 0.002
        assert model.get_fee_rate("sell") == 0.003

    def test_create_tiered_model(self):
        """Test creating tiered fee model"""
        config = {"tiers": {"buy_tiers": [[0, 0.002]], "sell_tiers": [[0, 0.003]]}}
        model = FeeModelFactory.create_fee_model("tiered", config)
        assert isinstance(model, TieredFeeModel)
        assert model.get_fee_rate("buy") == 0.002
        assert model.get_fee_rate("sell") == 0.003

    def test_create_exchange_model(self):
        """Test creating exchange fee model"""
        config = {"exchange_fees": {"test": {"buy": 0.002, "sell": 0.003}}}
        model = FeeModelFactory.create_fee_model("exchange", config)
        assert isinstance(model, ExchangeFeeModel)
        model.set_exchange("test")
        assert model.get_fee_rate("buy") == 0.002
        assert model.get_fee_rate("sell") == 0.003

    def test_create_unknown_model(self):
        """Test creating unknown model type (fallback to fixed)"""
        model = FeeModelFactory.create_fee_model("unknown")
        assert isinstance(model, FixedFeeModel)

    def test_create_model_no_config(self):
        """Test creating model without config"""
        model = FeeModelFactory.create_fee_model("fixed")
        assert isinstance(model, FixedFeeModel)

    def test_load_fee_model_from_config(self, tmp_path):
        """Test loading fee model from config file"""
        config = {
            "fee_model": {
                "type": "exchange",
                "exchange_fees": {"bitflyer": {"buy": 0.001, "sell": 0.001}},
            }
        }

        config_file = tmp_path / "fee_config.json"
        import json

        with open(config_file, "w") as f:
            json.dump(config, f)

        model = load_fee_model_from_config(str(config_file))
        assert isinstance(model, ExchangeFeeModel)
        model.set_exchange("bitflyer")
        assert model.get_fee_rate("buy") == 0.001

    def test_load_fee_model_invalid_file(self):
        """Test loading fee model from invalid file"""
        model = load_fee_model_from_config("nonexistent.json")
        assert model is None
