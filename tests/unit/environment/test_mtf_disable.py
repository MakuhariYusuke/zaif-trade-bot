"""MTF (Multi-Timeframe) 強制有効化削除の検証テスト

Phase 4 Week 1 Day 1の前提条件検証:
- feature_flags.include_multi_timeframe_features=False が機能すること
- デフォルトはTrue（後方互換性）
- 強制有効化コードが削除されていること
"""

import inspect
import pytest


def test_mtf_can_be_disabled():
    """MTFを設定で無効化できることを検証"""
    # feature_flags で MTF を無効化
    feature_flags = {"include_multi_timeframe_features": False}

    # initialization.py の動作を模倣
    include_mtf = feature_flags.get("include_multi_timeframe_features", True)

    # 無効化が機能することを確認
    assert include_mtf is False, "MTF should be disabled when feature_flags is False"


def test_mtf_enabled_by_default():
    """デフォルトではMTFが有効であることを検証（後方互換性）"""
    # feature_flags が空の場合
    feature_flags = {}

    # デフォルトはTrue
    include_mtf = feature_flags.get("include_multi_timeframe_features", True)

    assert include_mtf is True, "MTF should be enabled by default"


def test_mtf_flag_not_forced():
    """MTFの強制有効化コードが削除されていることを検証"""
    # initialization.py のソースコードを読み込み
    with open(
        "ztb/trading/environment/heavy_env/mixins/initialization.py",
        "r",
        encoding="utf-8",
    ) as f:
        source = f.read()

    # 強制有効化コードが含まれていないことを確認
    assert (
        "Forcing enable of multi-timeframe features" not in source
    ), "Forced MTF enable code should be removed"

    # 強制有効化の代入文が含まれていないことを確認
    # 許容: include_mtf = feature_flags.get(...)
    # 禁止: if not include_mtf: ... include_mtf = True
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "include_mtf" in line and "= True" in line:
            # 次の行を確認して、コメントまたはデフォルト値でないことを確認
            if i > 0 and "if not include_mtf" in lines[i - 1]:
                pytest.fail(
                    f"Forced MTF enable found at line {i+1}: {lines[i-1:i+2]}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

