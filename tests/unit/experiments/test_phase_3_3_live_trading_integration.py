# test_phase_3_3_live_trading_integration.py

"""
Phase 3-3 ライブトレーディング統合テスト

ライブトレーディング統合モジュールの機能を検証します。
"""

import pytest
import unittest.mock as mock
from datetime import datetime
from ztb.live_trading.trading_api import TradingAPI
from ztb.live_trading.live_trader import LiveTrader, TradingSignal


class TestLiveTradingIntegration:
    """ライブトレーディング統合テスト"""

    def setup_method(self):
        """テスト前準備"""
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.trading_api = TradingAPI(self.api_key, self.api_secret)

    def test_trading_api_initialization(self):
        """TradingAPI初期化テスト"""
        assert self.trading_api.api_key == self.api_key
        assert self.trading_api.api_secret == self.api_secret
        assert not self.trading_api.is_live_trading

    @mock.patch('ztb.live_trading.trading_api.time.sleep')
    def test_get_ticker_mock(self, mock_sleep):
        """ティッカー取得テスト（モック）"""
        # モックレスポンス
        mock_response = {
            'last': 100000.0,
            'bid': 99950.0,
            'ask': 100050.0,
            'volume': 100.5
        }

        with mock.patch.object(self.trading_api, '_make_request', return_value=mock_response):
            ticker = self.trading_api.get_ticker('btc_jpy')

            assert ticker['last'] == 100000.0
            assert ticker['bid'] == 99950.0
            assert ticker['ask'] == 100050.0
            assert ticker['volume'] == 100.5

    @mock.patch('ztb.live_trading.trading_api.time.sleep')
    def test_get_balance_mock(self, mock_sleep):
        """残高取得テスト（モック）"""
        mock_response = {
            'btc': 0.5,
            'jpy': 50000.0
        }

        with mock.patch.object(self.trading_api, '_make_request', return_value=mock_response):
            balance = self.trading_api.get_balance()

            assert balance['btc'] == 0.5
            assert balance['jpy'] == 50000.0

    @mock.patch('ztb.live_trading.trading_api.time.sleep')
    def test_create_order_mock(self, mock_sleep):
        """注文作成テスト（モック）"""
        mock_response = {
            'order_id': '12345',
            'status': 'open'
        }

        with mock.patch.object(self.trading_api, '_make_request', return_value=mock_response):
            order = self.trading_api.create_order(
                symbol='btc_jpy',
                side='buy',
                order_type='limit',
                amount=0.01,
                price=100000.0
            )

            assert order['order_id'] == '12345'
            assert order['status'] == 'open'

    def test_rate_limiting(self):
        """レート制限テスト"""
        # 連続リクエストでレート制限がかかることを確認
        start_time = datetime.now()

        # 複数回のリクエスト（実際のAPI呼び出しでレート制限がかかる）
        for i in range(5):
            self.trading_api.get_ticker('btc_jpy')

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # レート制限により時間がかかっているはず
        assert duration >= 0.1  # 最低レート制限時間

    def test_live_trader_initialization(self):
        """LiveTrader初期化テスト"""
        signal_generator = mock.Mock()
        risk_manager = mock.Mock()

        trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=signal_generator,
            risk_manager=risk_manager
        )

        assert trader.trading_api == self.trading_api
        assert trader.signal_generator == signal_generator
        assert trader.risk_manager == risk_manager
        assert not trader.is_running

    def test_position_management(self):
        """ポジション管理テスト"""
        signal_generator = mock.Mock()
        risk_manager = mock.Mock()

        trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=signal_generator,
            risk_manager=risk_manager
        )

        # ポジション追加
        trader._add_position('btc_jpy', 0.01, 100000.0, 'buy')
        assert len(trader.positions) == 1
        assert trader.positions[0].symbol == 'btc_jpy'
        assert trader.positions[0].amount == 0.01

        # ポジション削除
        trader._remove_position('btc_jpy')
        assert len(trader.positions) == 0

    @mock.patch('ztb.live_trading.live_trader.time.sleep')
    def test_trading_loop_execution(self, mock_sleep):
        """トレーディングループ実行テスト"""
        signal_generator = mock.Mock()
        risk_manager = mock.Mock()

        # モック設定
        signal_generator.generate_signal.return_value = [
            TradingSignal(
                symbol='btc_jpy',
                action='buy',
                confidence=0.8,
                amount=0.01,
                price=100000.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        risk_manager.validate_trade.return_value = True

        trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=signal_generator,
            risk_manager=risk_manager
        )

        # 1回のループ実行
        with mock.patch.object(trader, '_execute_buy') as mock_execute:
            trader._trading_loop_iteration()

            # シグナル生成と実行が呼ばれたことを確認
            signal_generator.generate_signal.assert_called_once()
            mock_execute.assert_called_once()

    def test_risk_management_integration(self):
        """リスク管理統合テスト"""
        signal_generator = mock.Mock()
        risk_manager = mock.Mock()

        trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=signal_generator,
            risk_manager=risk_manager
        )

        # リスクチェック
        trade_signal = {
            'action': 'buy',
            'symbol': 'btc_jpy',
            'amount': 0.01,
            'price': 100000.0
        }

        risk_manager.validate_trade.return_value = True
        is_valid = trader._check_risk_management(trade_signal)

        assert is_valid
        risk_manager.validate_trade.assert_called_once_with(trade_signal)

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        signal_generator = mock.Mock()
        risk_manager = mock.Mock()

        trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=signal_generator,
            risk_manager=risk_manager
        )

        # APIエラーのシミュレーション
        with mock.patch.object(self.trading_api, 'get_ticker', side_effect=Exception("API Error")):
            # エラーが発生してもクラッシュしないことを確認
            try:
                trader._trading_loop_iteration()
                # 正常に処理が継続された
                assert True
            except Exception:
                # エラーが伝播していないはず
                assert False, "Error should be handled internally"


if __name__ == '__main__':
    pytest.main([__file__])