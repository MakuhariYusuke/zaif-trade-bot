#!/usr/bin/env python3
"""
Week 4: 実残高情報取得スクリプト

Zaif取引所から実際のアカウント情報を取得し、
トレーニング初期化用のパラメータを表示します。

Usage:
    python scripts/v456/get_account_info.py
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.live_trading.trading_api import TradingAPI

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """メイン実行"""
    
    print("=" * 70)
    print("Week 4: Zaif アカウント情報取得")
    print("=" * 70)
    print()
    
    # 環境変数読み込み
    env_file = PROJECT_ROOT / '.env'
    if not env_file.exists():
        print("❌ .env ファイルが見つかりません")
        print("   Zaif APIキーが設定されていません")
        print()
        print("デフォルト値（テストモード）:")
        print("  Initial Balance: 0 JPY")
        print("  Position Size: 100% (uncapped)")
        print()
        return
    
    load_dotenv(env_file)
    
    api_key = os.getenv('ZAIF_API_KEY', '')
    api_secret = os.getenv('ZAIF_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("❌ ZAIF_API_KEY または ZAIF_API_SECRET が設定されていません")
        print()
        print("デフォルト値（テストモード）:")
        print("  Initial Balance: 0 JPY")
        print("  Position Size: 100% (uncapped)")
        print()
        return
    
    try:
        # Zaif API接続
        api = TradingAPI(
            api_key=api_key,
            api_secret=api_secret,
            test_mode=False
        )
        
        print("✓ Zaif API接続中...")
        balance = api.get_balance()
        
        print("✓ 残高取得成功")
        print()
        print("=" * 70)
        print("アカウント情報")
        print("=" * 70)
        print(f"  BTC残高:  {balance['btc']:.8f} BTC")
        print(f"  JPY残高:  {balance['jpy']:,.2f} JPY")
        print()
        print("=" * 70)
        print("訓練初期化パラメータ")
        print("=" * 70)
        print(f"  initial_balance:  {balance['jpy']:,.2f} JPY")
        print(f"  max_position:     None (100%, uncapped)")
        print(f"  position_size:    max_balance / BTC_price")
        print()
        
        # 設定値をJSONで保存
        config_file = PROJECT_ROOT / 'scripts' / 'v456' / 'account_config.json'
        config = {
            'initial_balance_jpy': balance['jpy'],
            'initial_btc': balance['btc'],
            'max_position': None,
            'position_sizing': '100% (uncapped)',
            'status': 'ready_for_training'
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 設定を保存: {config_file}")
        print()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        print()
        print("デフォルト値（テストモード）:")
        print("  Initial Balance: 0 JPY")
        print("  Position Size: 100% (uncapped)")
        print()

if __name__ == '__main__':
    main()
