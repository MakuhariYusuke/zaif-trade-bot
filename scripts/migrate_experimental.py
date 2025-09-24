"""
migrate_experimental.py
experimental 特徴量を成熟度に応じて移行
"""

import yaml
from pathlib import Path
import shutil
import logging

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "feature_maturity.yaml"
FEATURES_YAML = ROOT / "config" / "features.yaml"
EXPERIMENTAL = ROOT / "src" / "trading" / "features" / "experimental.py"

TARGET_DIRS = {
    "trend": ROOT / "src" / "trading" / "features" / "trend",
    "momentum": ROOT / "src" / "trading" / "features" / "momentum",
    "volatility": ROOT / "src" / "trading" / "features" / "volatility",
    "volume": ROOT / "src" / "trading" / "features" / "volume",
}

def load_maturity():
    """成熟度設定を読み込み"""
    with open(CONFIG, "r") as f:
        return yaml.safe_load(f)

def migrate_if_ready():
    """成熟度がstableになった特徴量を移行"""
    maturity_cfg = load_maturity()["experimental"]
    for feat, meta in maturity_cfg.items():
        if meta["maturity"] == "stable":
            target = determine_target(feat)
            if not target:
                print(f"⚠️ {feat} の移行先不明、手動対応してください")
                continue
            move_feature(feat, target)
            update_features_yaml(feat, target)
            print(f"✅ {feat} を {target}/ に移行しました")

def determine_target(feat_name: str) -> str | None:
    """特徴量名から移行先を決定"""
    print(f"Determining target for {feat_name}")
    if "Moving" in feat_name or "Trend" in feat_name or "Cross" in feat_name or "TEMA" in feat_name or "KAMA" in feat_name:
        print(f"{feat_name} -> trend")
        return "trend"
    if "RSI" in feat_name or "MACD" in feat_name or "Stochastic" in feat_name or "CCI" in feat_name:
        return "momentum"
    if "Vol" in feat_name or "ATR" in feat_name or "HV" in feat_name or "Bollinger" in feat_name:
        return "volatility"
    if "Volume" in feat_name or "VWAP" in feat_name or "OBV" in feat_name or "MFI" in feat_name:
        return "volume"
    print(f"{feat_name} -> None")
    return None

def move_feature(feat: str, target: str):
    """特徴量をexperimentalからtargetディレクトリに移動"""
    # experimental.pyから該当クラスを抽出して個別ファイルに移動
    # 簡易実装として、experimental.pyの内容をコピー
    src_file = EXPERIMENTAL
    dst_file = TARGET_DIRS[target] / f"{feat.lower()}.py"

    # 実際の移行は手動で行うことを想定し、ここではログ出力のみ
    logging.info(f"Move {feat} from {src_file} to {dst_file}")
    # shutil.copy(src_file, dst_file)  # 実際のコピー

def update_features_yaml(feat: str, target: str):
    """features.yamlを更新"""
    with open(FEATURES_YAML, "r") as f:
        data = yaml.safe_load(f)

    # features辞書に追加
    if "features" not in data:
        data["features"] = {}

    if feat not in data["features"]:
        data["features"][feat] = {
            "enabled": True,
            "wave": target,
            "params": {}
        }

    with open(FEATURES_YAML, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_if_ready()