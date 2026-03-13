"""
データソース検証スクリプト
各データソースの実装状況とアクセス可能性を確認
"""

import sys
from pathlib import Path
from datetime import datetime

try:
    from ztb.utils.path_utils import get_project_root
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))


def check_coincheck_availability():
    """CoinCheck API アクセス可能性を確認"""
    print("\n" + "="*70)
    print("CoinCheck API チェック")
    print("="*70)
    
    try:
        import requests
        
        url = "https://api.coincheck.com/api/ticker?pair=btc_jpy"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ CoinCheck API: 利用可能")
            print(f"  最新 BTC/JPY レート: ¥{data.get('last', 'N/A'):,}")
            return True
        else:
            print(f"✗ CoinCheck API: ステータス {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ CoinCheck API: エラー - {e}")
        return False


def check_bitflyer_availability():
    """BitFlyer API アクセス可能性を確認"""
    print("\n" + "="*70)
    print("BitFlyer API チェック")
    print("="*70)
    
    try:
        import requests
        
        url = "https://api.bitflyer.jp/v1/ticker?product_code=BTC_JPY"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ BitFlyer API: 利用可能 (ティッカーのみ)")
            print(f"  最新 BTC/JPY レート: ¥{data.get('ltp', 'N/A'):,}")
            print(f"  note: OHLC エンドポイントなし（WebSocket 推奨）")
            return True
        else:
            print(f"✗ BitFlyer API: ステータス {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ BitFlyer API: エラー - {e}")
        return False


def check_yfinance_availability():
    """YahooFinance アクセス可能性を確認"""
    print("\n" + "="*70)
    print("YahooFinance チェック")
    print("="*70)
    
    try:
        import yfinance as yf
        
        print("✓ yfinance モジュール: インストール済み")
        
        # サンプルダウンロード（1 データポイント）
        df = yf.download("BTC-JPY", interval="1m", period="1d", progress=False)
        
        if not df.empty:
            latest = df.iloc[-1]
            print(f"✓ YahooFinance: データ取得可能")
            print(f"  最新 BTC/JPY レート: ¥{latest['Close']:,.0f}")
            print(f"  note: 直近 7日のみ利用可能")
            return True
        else:
            print(f"✗ YahooFinance: データ取得失敗")
            return False
            
    except ImportError:
        print(f"✗ yfinance モジュール: インストール未済")
        print(f"  インストール: pip install yfinance")
        return False
    except Exception as e:
        print(f"✗ YahooFinance: エラー - {e}")
        return False


def check_existing_data():
    """既存データファイルをチェック"""
    print("\n" + "="*70)
    print("既存データ ファイルチェック")
    print("="*70)
    
    try:
        import pandas as pd
        
        candidates = [
            project_root / "data" / "btc_jpy_real_dataset.csv",
            project_root / "data" / "btc_jpy_1m_v456.csv",
            project_root / "data" / "btc_jpy_1m_v455.csv",
            project_root / "data" / "btc_jpy_1m_v454.csv",
        ]
        
        found = False
        for candidate in candidates:
            if candidate.exists():
                df = pd.read_csv(candidate, index_col=0, parse_dates=True, nrows=1000)
                
                if not df.empty:
                    print(f"✓ ファイル: {candidate.name}")
                    print(f"  範囲: {df.index.min()} ～ {df.index.max()}")
                    print(f"  レコード数: {len(df):,}")
                    print(f"  列: {', '.join(df.columns.tolist())}")
                    found = True
                    break
        
        if not found:
            print(f"✗ BTC/JPY データファイルが見つかりません")
            print(f"  確認パス:")
            for candidate in candidates:
                print(f"    - {candidate}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False


def check_script_availability():
    """更新スクリプトの利用可能性を確認"""
    print("\n" + "="*70)
    print("更新スクリプト チェック")
    print("="*70)
    
    scripts = [
        "update_data_comprehensive.py",
        "update_data_coincheck.py",
        "update_data_bitflyer.py",
        "update_data_simple.py",
    ]
    
    scripts_dir = Path(__file__).parent
    
    all_found = True
    for script in scripts:
        script_path = scripts_dir / script
        status = "✓" if script_path.exists() else "✗"
        print(f"{status} {script}")
        if not script_path.exists():
            all_found = False
    
    return all_found


def main():
    print("\n" + "="*70)
    print("BTC/JPY データ更新 - 環境チェック")
    print("="*70)
    print(f"実行時刻: {datetime.now()}")
    
    results = {
        'CoinCheck': check_coincheck_availability(),
        'BitFlyer': check_bitflyer_availability(),
        'YahooFinance': check_yfinance_availability(),
        '既存データ': check_existing_data(),
        'スクリプト': check_script_availability(),
    }
    
    # サマリー
    print("\n" + "="*70)
    print("サマリー")
    print("="*70)
    
    available_sources = [k for k, v in results.items() if v and k != 'スクリプト']
    
    if not results['既存データ']:
        print("\n⚠ 既存データファイルが見つかりません")
        print("  推奨: data/btc_jpy_real_dataset.csv を用意してから更新\n")
    
    if available_sources:
        print(f"\n✓ 利用可能なデータソース: {', '.join(available_sources)}")
        print(f"\n実行コマンド:")
        print(f"  python scripts/v456/update_data_comprehensive.py")
    else:
        print(f"\n✗ 利用可能なデータソースがありません")
        print(f"  対策:")
        print(f"    1. インターネット接続を確認")
        print(f"    2. API サーバーのステータスを確認")
        print(f"    3. 必要なモジュールをインストール: pip install yfinance requests")
    
    if results['スクリプト']:
        print(f"\n✓ すべての更新スクリプトが利用可能です")
    else:
        print(f"\n✗ 一部のスクリプトが見つかりません")
    
    # 推奨アクション
    print("\n" + "="*70)
    print("推奨アクション")
    print("="*70)
    
    if not results['既存データ'] and results['CoinCheck']:
        print("\n1. 初期データを CoinCheck から取得:")
        print("   python scripts/v456/update_data_coincheck.py")
    
    elif results['既存データ'] and (results['CoinCheck'] or results['BitFlyer'] or results['YahooFinance']):
        print("\n1. データを最新に更新:")
        print("   python scripts/v456/update_data_comprehensive.py")
        print("\n2. 定期更新を設定:")
        print("   - Linux/macOS: cron で毎日実行")
        print("   - Windows: Task Scheduler で毎日実行")
    
    print("\n詳細: scripts/v456/DATA_UPDATE_README.md\n")


if __name__ == "__main__":
    main()
