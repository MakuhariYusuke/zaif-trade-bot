# 725# Task 2: sac_retrain_scheduler type: ignore 除去（型安全向上）

## 背景
`scripts/v460/ml/sac_retrain_scheduler.py` の `SACRetrainConfig.from_yaml_dict()` に
8箇所の `# type: ignore` がある。`dict.get()` の戻り値型が `object` のため
`dict` 型変数への代入で mypy が警告するパターン。

## 現状コード (lines 157-172)
```python
@classmethod
def from_yaml_dict(cls, cfg: dict) -> SACRetrainConfig:
    data_cfg: dict = cfg.get("data", {})  # type: ignore[assignment]
    sac_cfg: dict = cfg.get("sac_hyperparameters", {})  # type: ignore[assignment]
    training_cfg: dict = cfg.get("training", {})  # type: ignore[assignment]
    env_cfg: dict = cfg.get("environment", {})  # type: ignore[assignment]
    feat_cfg: dict = cfg.get("features", {})  # type: ignore[assignment]
    output_cfg: dict = cfg.get("output", {})  # type: ignore[assignment]
    retrain_cfg: dict = cfg.get("sac_retrain", {})  # type: ignore[assignment]
    ...
    ohlcv_path=str(
        data_cfg.get("ohlcv_path", cls.ohlcv_path)  # type: ignore[arg-type]
    ),
```

## 要件

### 修正方針
**方法A (推奨): 引数型を `dict[str, object]` に明示し、`cast()` または `ensure_dict()` ヘルパーを使用**

```python
from typing import cast

@classmethod
def from_yaml_dict(cls, cfg: dict[str, object]) -> SACRetrainConfig:
    data_cfg = cast(dict[str, object], cfg.get("data", {}))
    sac_cfg = cast(dict[str, object], cfg.get("sac_hyperparameters", {}))
    ...
```

**方法B: ztb.utils に既存の `ensure_dict` がある場合はそれを使用**

```python
from ztb.utils.type_utils import ensure_dict  # 既存ヘルパーがあれば

data_cfg = ensure_dict(cfg.get("data", {}))
```

### line 172 の修正
```python
# Before
ohlcv_path=str(data_cfg.get("ohlcv_path", cls.ohlcv_path))  # type: ignore[arg-type]
# After
ohlcv_path=str(data_cfg.get("ohlcv_path", str(cls.ohlcv_path)))
```
`cls.ohlcv_path` が `Path` 型のため `str()` で明示的に変換すれば `arg-type` 抑制不要。

## テスト対象ファイル
- `scripts/v460/ml/sac_retrain_scheduler.py`

## 検証
```bash
mypy scripts/v460/ml/sac_retrain_scheduler.py --config-file mypy.ini
python -m pytest tests/unit/v460/test_sac_retrain*.py -x --tb=short
```

## 制約
- ランタイム動作の変更は不可（pure 型リファクタ）
- `SACRetrainConfig` のフィールド型は変更不可
- 既存テスト全パス必須
- 新規ヘルパー作成は最小限（既存ユーティリティ優先）
