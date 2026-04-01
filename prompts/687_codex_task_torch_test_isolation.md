# Codex Task: test_ab_param_search torch DLL 依存の隔離 (687#)

## 目的
`tests/unit/tools/test_ab_param_search.py` が torch DLL ロード失敗で FAIL する環境依存問題を解消する。

## 背景

### 現象
```
OSError: [WinError 1114] ダイナミック リンク ライブラリ (DLL) 初期化ルーチンの実行に失敗しました。
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

- `tools/ab_param_search.py` が `ztb.training.unified_optimizer` → `system_optimizer` → `import torch` の依存チェーンで torch を直接インポート
- `.venv` 内の torch が DLL 問題を持つ環境（Windows GPU なし等）でテスト FAIL
- テスト自体は torch の機能を使わない（config 生成とCLI 引数テストのみ）

### 根本原因
`test_ab_param_search_generates_configs` が `subprocess.run` で `tools/ab_param_search.py` を丸ごと実行するため、スクリプトのトップレベル import が走る。

## タスク

### Task 1: テストの環境依存解消

以下のいずれかの方法で修正:

**方法 A（推奨）**: torch import を skipCondition に
```python
import pytest

torch_available = False
try:
    import torch
    torch_available = True
except (ImportError, OSError):
    pass

@pytest.mark.skipif(not torch_available, reason="torch not available or DLL load failed")
def test_ab_param_search_generates_configs(tmp_path):
    ...
```

**方法 B**: `tools/ab_param_search.py` の lazy import 化
- トップレベルの `from ztb.training.unified_optimizer import ...` を関数内に移動
- テストでは config 生成パスのみテスト（optimizer 不要）

### Task 2: 検証

1. `python -m pytest tests/unit/tools/test_ab_param_search.py -x --tb=short` で skip or pass 確認
2. `python -m pytest tests/ -x --tb=short --ignore=tests/unit/tools/test_ab_param_search.py` で既存全テスト pass 確認

## 受け入れ基準

- [ ] torch DLL 不在環境でテストが skip（FAIL ではない）
- [ ] torch 利用可能環境では従来通り実行
- [ ] 全テスト suite で exit code 0
