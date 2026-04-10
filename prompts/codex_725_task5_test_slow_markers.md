# 725# Task 5: テストスイート高速化 — @pytest.mark.slow マーカー追加

## 背景
フルユニットテストスイート (`tests/unit/`) が約 7-8 分かかる。
ML モデル訓練を含むテストに `@pytest.mark.slow` を追加し、
`pytest -m "not slow"` で高速実行（推定 2-3 分）を可能にする。

## 対象クラス/テスト

### 5A. `tests/unit/v460/test_retrain_hot_reload.py`
- `TestRetrainModel` クラス (line 708〜) 全体に `@pytest.mark.slow` を追加
  - LGBM 訓練を含む統合テスト群
  - `test_retrain_deploy_and_hot_reload` 等が重い

### 5B. 他のスローテスト候補の調査
以下のパターンでスローテストを特定:
```bash
python -m pytest tests/unit/ --durations=30 -q 2>&1 | head -40
```
30秒以上かかるテストに `@pytest.mark.slow` を追加。

### 注意: 既存マーカーとの整合
- `test_ml_pipeline.py::Test057Integration` は **すでに `@pytest.mark.slow` + `@pytest.mark.integration` 付き**
- 新規追加分も同パターンに従う（`slow` のみ、`integration` は統合テストのみ）

## 要件

### Step 1: `--durations` で遅いテストを特定
```bash
python -m pytest tests/unit/ --durations=50 -q 2>&1 | Select-String "^\d" | Select-Object -First 50
```

### Step 2: 10秒以上のテストに `@pytest.mark.slow` を追加
- クラス単位のマーカーが望ましい（個別テストより管理しやすい）
- fixture レベルで重い場合はクラス全体にマーカー

### Step 3: pytest.ini に `slow` マーカー定義があることを確認
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
```

### Step 4: 高速実行の確認
```bash
# 高速モード（slow 除外）
python -m pytest tests/unit/ -m "not slow" -x --tb=short -q
# フルモード（変更なし）
python -m pytest tests/unit/ -x --tb=short -q
```

## テスト対象ファイル
- `tests/unit/v460/test_retrain_hot_reload.py` (確定)
- `--durations` で特定した追加ファイル群

## 検証
- `pytest -m "not slow"` で全パス + 実行時間 < 4分
- `pytest` (フルスイート) で全パス（既存と互換）
- マーカー未登録による PytestUnknownMarkWarning が出ないこと

## 制約
- テストロジックの変更は不可（マーカー追加のみ）
- 既存の `@pytest.mark.integration` マーカーを削除しないこと
- `conftest.py` の fixture 変更は不可
