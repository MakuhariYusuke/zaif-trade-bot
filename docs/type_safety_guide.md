# 型安全性ガイド

## 概要

Zaif Trade Botプロジェクトでは、型安全性の向上を重要な保守性目標として掲げています。このガイドでは、プロジェクトにおける型安全性のベストプラクティスと実装方針を説明します。

## 基本方針

### 1. Any型の使用制限
- `Any`型の使用は最小限に抑え、具体的な型を使用する
- ユーティリティ関数やProtocolでの使用は許容される
- ビジネスロジックでは`Any`を避ける

### 2. 型アノテーションの完全性
- すべての関数とメソッドに型アノテーションを付与
- クラス属性にも適切な型を指定
- Optional型の適切な使用

### 3. Protocolと抽象基底クラスの活用
- インターフェース定義にはProtocolを使用
- 共通機能の実装にはABCを使用
- 型チェック時の柔軟性を確保

## 実装パターン

### ConfigDictの使用
```python
from ztb.types.common import ConfigDict

def process_config(config: ConfigDict) -> Dict[str, Any]:
    # 設定処理
    pass
```

### Optional型の適切な使用
```python
from typing import Optional

class Trainer:
    def __init__(self, config: ConfigDict, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
```

### Union型の活用
```python
from typing import Union

def handle_action(action: Union[np.ndarray, float, tuple]) -> float:
    # アクション処理
    pass
```

## mypy設定

プロジェクトでは厳格なmypy設定を採用しています：

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
show_error_codes = true
```

## 型安全性のチェックポイント

### 1. 新規コード作成時
- すべての関数/メソッドに型アノテーションを付与
- Any型の使用理由をコメントで説明
- mypyで型チェックを実行

### 2. リファクタリング時
- 既存のAny型を具体的な型に置き換え
- 型安全性が向上することを確認
- テストで機能が維持されていることを確認

### 3. 外部ライブラリ使用時
- スタブファイルの活用
- ignore_missing_importsの適切な使用
- 型安全性を損なわないラッパーの作成

## よくある問題と解決法

### 1. 循環インポート
```python
# 問題のあるコード
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ztb.training.trainer import Trainer
```

### 2. 複雑なUnion型
```python
# 推奨される解決法
ActionType = Union[np.ndarray, float, Tuple[float, ...]]
```

### 3. 動的属性アクセス
```python
# 避けるべきコード
config: Any = load_config()
value = config.some_dynamic_attr

# 推奨されるコード
config: ConfigDict = load_config()
value = config.get("some_dynamic_attr", default_value)
```

## 型安全性の測定

### メトリクス
- Any型の使用数
- 型アノテーションの完全性率
- mypyエラーの数

### 継続的な改善
- 新規コードでのAny型使用禁止
- 定期的な型チェック実行
- コードレビューの型安全性確認

## 参考資料

- [mypy公式ドキュメント](https://mypy.readthedocs.io/)
- [Python Typing Guide](https://docs.python.org/3/library/typing.html)
- [Real Python Type Checking](https://realpython.com/python-type-checking/)</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\type_safety_guide.md
