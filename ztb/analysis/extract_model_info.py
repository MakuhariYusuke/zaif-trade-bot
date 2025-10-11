"""
モデルファイルから環境設定とメタデータを抽出するツール

スキーマファイルが存在しない古いモデル（v378, v379等）から
訓練時の設定情報を可能な限り復元します。
"""
import zipfile
import json
import sys
from pathlib import Path

def extract_model_info(model_path: str):
    """モデルZIPファイルから情報を抽出"""
    print(f"\n{'='*80}")
    print(f"Model: {Path(model_path).name}")
    print(f"{'='*80}\n")
    
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            # ZIPファイル内のファイル一覧
            print("📦 Files in model ZIP:")
            for name in zip_ref.namelist():
                info = zip_ref.getinfo(name)
                print(f"  - {name} ({info.file_size:,} bytes)")
            
            print("\n" + "="*80)
            
            # data ファイルを探す
            data_file = None
            for name in zip_ref.namelist():
                if name == 'data' or name.endswith('/data'):
                    data_file = name
                    break
            
            if data_file:
                print(f"\n📊 Attempting to read {data_file}...")
                try:
                    with zip_ref.open(data_file) as f:
                        # バイナリデータの最初の1KB を表示（デバッグ用）
                        data = f.read(1024)
                        print(f"  First 200 bytes (hex): {data[:200].hex()}")
                        print(f"  First 200 bytes (ascii): {data[:200]}")
                except Exception as e:
                    print(f"  ❌ Error reading data: {e}")
            
            # pytorch_variables.pkl を探す
            pkl_file = None
            for name in zip_ref.namelist():
                if 'pytorch_variables.pkl' in name:
                    pkl_file = name
                    break
            
            if pkl_file:
                print(f"\n🐍 Found {pkl_file}")
                print(f"  Size: {zip_ref.getinfo(pkl_file).file_size:,} bytes")
                print(f"  Note: This contains model weights (not easily readable)")
            
            # .json ファイルを探す
            json_files = [name for name in zip_ref.namelist() if name.endswith('.json')]
            if json_files:
                print(f"\n📝 JSON files found:")
                for json_file in json_files:
                    print(f"\n  File: {json_file}")
                    try:
                        with zip_ref.open(json_file) as f:
                            data = json.load(f)
                            print(f"  Content:\n{json.dumps(data, indent=4)}")
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
            else:
                print(f"\n⚠️  No JSON files found")
            
            # .txt ファイルを探す
            txt_files = [name for name in zip_ref.namelist() if name.endswith('.txt')]
            if txt_files:
                print(f"\n📄 Text files found:")
                for txt_file in txt_files:
                    print(f"\n  File: {txt_file}")
                    try:
                        with zip_ref.open(txt_file) as f:
                            content = f.read().decode('utf-8')
                            print(f"  Content:\n{content[:500]}")  # 最初の500文字
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
            
            print("\n" + "="*80)
            print("\n💡 Summary:")
            print("  - Stable-Baselines3モデルは訓練時の環境設定を含まない")
            print("  - 環境設定を復元するには訓練スクリプトやログを確認する必要がある")
            print("  - または、逆算: モデルを実行してobservation spaceから特徴量数を推定")
            
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return False
    except zipfile.BadZipFile:
        print(f"❌ Invalid ZIP file: {model_path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_model_info.py <model_path>")
        print("\nExamples:")
        print("  python extract_model_info.py models/ppo_reward_v378_scale.zip")
        print("  python extract_model_info.py models/ppo_reward_v379_dynamic_short.zip")
        sys.exit(1)
    
    model_path = sys.argv[1]
    extract_model_info(model_path)

if __name__ == "__main__":
    main()
