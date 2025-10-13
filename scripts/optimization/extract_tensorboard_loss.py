#!/usr/bin/env python3
"""
TensorBoardイベントファイルからCritic Lossを抽出
"""
import sys
from pathlib import Path
from typing import Optional

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ tensorboardがインストールされていません")
    print("  pip install tensorboard")
    sys.exit(1)


def extract_critic_loss_from_tensorboard(log_dir: Path, verbose: bool = False) -> Optional[float]:
    """
    TensorBoardログディレクトリからCritic Lossを抽出
    
    Args:
        log_dir: TensorBoardログディレクトリ
        verbose: 詳細出力を行うか
        
    Returns:
        最後のCritic Loss値（見つからない場合はNone）
    """
    # イベントファイルを検索
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    
    if not event_files:
        if verbose:
            print(f"⚠️ イベントファイルが見つかりません: {log_dir}")
        return None
    
    if verbose:
        print(f"📂 イベントファイル: {len(event_files)}個")
    
    critic_losses = []
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            # 利用可能なタグを表示
            scalars = ea.Tags().get('scalars', [])
            
            if verbose:
                print(f"  利用可能なスカラー ({len(scalars)}個):")
                for tag in scalars[:10]:  # 最初の10個のみ表示
                    print(f"    - {tag}")
            
            # Critic Loss関連のタグを検索
            critic_tags = [tag for tag in scalars if 'critic' in tag.lower() or 'loss' in tag.lower()]
            
            if verbose:
                print(f"  Critic/Loss関連タグ ({len(critic_tags)}個):")
            
            for tag in critic_tags:
                if verbose:
                    print(f"    - {tag}")
                
                # 値を取得
                events = ea.Scalars(tag)
                if events:
                    latest_value = events[-1].value
                    critic_losses.append((tag, latest_value))
                    if verbose:
                        print(f"      最新値: {latest_value}")
        
        except Exception as e:
            if verbose:
                print(f"⚠️ イベントファイル読み込みエラー: {e}")
            continue
    
    if not critic_losses:
        if verbose:
            print("⚠️ Critic Loss が見つかりませんでした")
        return None
    
    # 最も妥当なCritic Lossを選択（'train/critic_loss' など）
    for tag, value in critic_losses:
        if 'train' in tag.lower() and 'critic' in tag.lower():
            if verbose:
                print(f"✅ Critic Loss 抽出成功: {value} (tag: {tag})")
            return value
    
    # 見つからない場合は最初のCritic Loss関連値を返す
    tag, value = critic_losses[0]
    if verbose:
        print(f"✅ Critic Loss 抽出: {value} (tag: {tag})")
    return value


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorBoardからCritic Loss抽出')
    parser.add_argument('log_dir', type=str, help='TensorBoardログディレクトリ')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"❌ ログディレクトリが見つかりません: {log_dir}")
        sys.exit(1)
    
    critic_loss = extract_critic_loss_from_tensorboard(log_dir)
    
    if critic_loss is not None:
        print(f"\n最終Critic Loss: {critic_loss}")
    else:
        print(f"\n❌ Critic Loss を抽出できませんでした")
        sys.exit(1)


if __name__ == "__main__":
    main()
