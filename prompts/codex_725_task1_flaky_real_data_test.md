# 725# Task 1: test_load_real_data flaky テスト修正

## 背景
`tests/unit/v460/test_ml_pipeline.py::Test057Integration::test_load_real_data` が
fill レコード数がハードコード閾値 20 未満のとき失敗する。当日のレコード蓄積が
少ない時間帯（朝〜昼）で CI が恒常的に赤になるフレキーテスト。

## 現状コード (line 425)
```python
def test_load_real_data(self, real_data_available: bool, real_fill_df: pd.DataFrame) -> None:
    if not real_data_available:
        pytest.skip("No real fill records")
    assert len(real_fill_df) >= 20   # ← ハードコード: 本日8件で失敗
    X, y = build_as_features(real_fill_df)
    assert len(X) >= 10
```

## 要件

### 修正方針
1. `>= 20` のハードコード閾値を**実データ件数に応じた条件付きスキップ**に変更
2. 閾値未満（例: 5件未満）は `pytest.skip()` で飛ばす
3. 閾値以上の場合は `build_as_features` が正常動作することの検証に集中
4. `assert len(X) >= 10` も同様にフレキーなので、`len(X) >= 1` または `len(X) > 0` に緩和

### 推奨実装
```python
MIN_RECORDS_FOR_INTEGRATION = 5  # 特徴量構築に最低限必要な件数

def test_load_real_data(self, real_data_available: bool, real_fill_df: pd.DataFrame) -> None:
    if not real_data_available:
        pytest.skip("No real fill records")
    if len(real_fill_df) < MIN_RECORDS_FOR_INTEGRATION:
        pytest.skip(f"Insufficient fill records: {len(real_fill_df)} < {MIN_RECORDS_FOR_INTEGRATION}")
    X, y = build_as_features(real_fill_df)
    assert len(X) >= 1  # 特徴量構築が正常に動作すること
```

## テスト対象ファイル
- `tests/unit/v460/test_ml_pipeline.py`

## 検証
- `python -m pytest tests/unit/v460/test_ml_pipeline.py::Test057Integration -x --tb=short`
- 実データ 0〜4件: skip
- 実データ 5件以上: pass (build_as_features が正常動作)
- 既存の他テスト (`Test057DataLoaderCache`, `TestASFeatures` 等) が引き続きパスすること

## 制約
- `@pytest.mark.slow` / `@pytest.mark.integration` マーカーは維持
- fixture `real_data_available`, `real_fill_df` の構造は変更不可
