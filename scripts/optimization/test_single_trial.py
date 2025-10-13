#!/usr/bin/env python3
"""
最適化フレームワーク 単一試行テスト
訓練が正しく実行され、Critic Loss が抽出できるか確認
"""
import sys
import os
import time
import subprocess
import json
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def extract_critic_loss(stdout: str) -> float:
    """標準出力から最終Critic Lossを抽出"""
    lines = stdout.split('\n')
    
    # 最後の方から検索
    for line in reversed(lines[-200:]):  # 最後の200行
        if 'critic_loss' in line.lower():
            try:
                # "critic_loss: 0.123" や "critic_loss=0.123" に対応
                for sep in [':', '=']:
                    if sep in line:
                        parts = line.split(sep)
                        if len(parts) >= 2:
                            loss_str = parts[-1].strip()
                            # カンマや括弧を除去
                            loss_str = loss_str.split(',')[0].split(')')[0].split()[0]
                            return float(loss_str)
            except (ValueError, IndexError):
                continue
    
    print("⚠️ Critic Lossを抽出できませんでした")
    print(f"stdout最後の10行:")
    for line in lines[-10:]:
        print(f"  {line}")
    return 1e6


def main():
    print("="*80)
    print("  最適化フレームワーク 単一試行テスト")
    print("  - 100ステップ訓練を1回実行")
    print("  - Critic Loss抽出テスト")
    print("="*80)
    
    # テスト設定
    root = get_project_root()
    config_path = root / 'configs' / 'sac_test_100steps.json'
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    print(f"\n📄 設定: {config_path}")
    print(f"🚀 訓練開始...")
    
    start_time = time.time()
    
    try:
        # 環境変数を設定
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        result = subprocess.run(
            [
                'python',
                'scripts/optimization/train_with_config.py',
                '--config', str(config_path)
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5分タイムアウト
            cwd=str(root),
            env=env
        )
        
        duration = time.time() - start_time
        print(f"✅ 訓練完了: {duration:.1f}秒")
        print(f"   Return code: {result.returncode}")
        
        # Critic Loss抽出
        print(f"\n🔍 Critic Loss 抽出中...")
        critic_loss = extract_critic_loss(result.stdout)
        
        print(f"\n結果:")
        print(f"  Critic Loss: {critic_loss}")
        print(f"  Duration: {duration:.1f}秒")
        print(f"  Success: {critic_loss < 1e6}")
        
        # stdout/stderrの長さ表示
        print(f"\n出力:")
        print(f"  stdout: {len(result.stdout)} 文字")
        print(f"  stderr: {len(result.stderr)} 文字")
        
        if result.returncode != 0:
            print(f"\n⚠️ エラー終了 (code {result.returncode})")
            print(f"stderr:")
            print(result.stderr[-1000:])  # 最後の1000文字
        
        # 結果をファイルに保存
        result_data = {
            'success': critic_loss < 1e6,
            'critic_loss': critic_loss,
            'duration_seconds': duration,
            'returncode': result.returncode,
            'stdout_length': len(result.stdout),
            'stderr_length': len(result.stderr),
            'stdout_tail': result.stdout[-1000:],
            'stderr_tail': result.stderr[-1000:]
        }
        
        result_path = root / 'scripts' / 'optimization' / 'single_trial_test_result.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 結果保存: {result_path}")
        
    except subprocess.TimeoutExpired:
        print("❌ タイムアウト")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
