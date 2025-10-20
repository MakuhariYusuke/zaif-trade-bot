"""TensorBoardログのデバッグ"""
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

project_root = Path(__file__).parent.parent.parent
log_dir = project_root / "checkpoints" / "sac_session" / "SAC_28"

print(f"ログディレクトリ: {log_dir}")
print(f"存在: {log_dir.exists()}")
print()

ea = event_accumulator.EventAccumulator(str(log_dir))
ea.Reload()

print("利用可能なタグ:")
print()

print("スカラー:")
for tag in ea.Tags()["scalars"]:
    print(f"  - {tag}")
    events = ea.Scalars(tag)
    print(f"    データポイント数: {len(events)}")
    if len(events) > 0:
        print(f"    最初の値: step={events[0].step}, value={events[0].value:.6f}")
        print(f"    最後の値: step={events[-1].step}, value={events[-1].value:.6f}")
    print()
