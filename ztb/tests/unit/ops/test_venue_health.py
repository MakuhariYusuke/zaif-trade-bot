"""
Unit tests for venue health checker
取引所ヘルスチェックの単体テスト
"""

import socket
from unittest.mock import patch
from ztb.ops.health.check_venue_health import VenueHealthChecker

class TestVenueHealthChecker:
    def test_initialization_coincheck(self):
        checker = VenueHealthChecker('coincheck', 'BTC_JPY')
        assert checker.venue == 'coincheck'
        assert checker.symbol == 'BTC_JPY'
        assert checker.rest_base == 'https://coincheck.com'
        assert checker.ws_url == 'wss://ws-api.coincheck.com/'

    @patch('socket.gethostbyname')
    def test_check_internet_connectivity_failure(self, mock_gethostbyname):
        mock_gethostbyname.side_effect = OSError('Network unreachable')
        checker = VenueHealthChecker('coincheck', 'BTC_JPY')
        result = checker.check_internet_connectivity()
        assert result is False
        assert checker.results['connectivity']['internet'] is False
