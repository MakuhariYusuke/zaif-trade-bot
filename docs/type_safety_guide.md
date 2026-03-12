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

## 設定(ConfigValue)の取り扱い

ConfigValueは再帰的なUnion型で、辞書やリスト、プリミティブ型すべてを含むため、値を直接使うとmypyでエラーになることがあります。以下のパターンを推奨します。

1. TypeGuardを用いて型を絞る
```python
from ztb.types.common import is_config_dict, is_numeric_config_value

val = config.get("training")
if is_config_dict(val):
    # mypyはここで val が dict[str, ConfigValue] として扱う
    total_timesteps = val.get("total_timesteps")
    if is_numeric_config_value(total_timesteps):
        # numeric として安全に扱える
        pass
```

2. ユーティリティを作る
`ztb.utils.config_helpers.get_numeric` や `get_dict` を使って、ネストした値の取得時に共通のバリデーションと型狭めを行ってください。

例: logging_utils.pyでの使用
```python
from ztb.utils.config_helpers import get_dict, get_string, get_int

def setup_logging_from_config(config: ConfigDict) -> None:
    logging_config = get_dict(config, "logging")  # 安全にdict取得
    level_str = get_string(logging_config, "level", "INFO")  # 安全にstr取得
    max_bytes = get_int(logging_config, "max_bytes", 10 * BYTES_PER_MB)  # 安全にint取得
```

例: parallel_experiments.pyでの使用
```python
from ztb.utils.config_helpers import get_string

def get_priority(config: ConfigDict) -> int:
    model_type = get_string(config, "model_type", "generalization")  # 安全にstr取得
    # ...
```

3. 型変換は明示的に
`int()` や `float()` で明示的に変換して値の型を揃えるか、Typed dataclass / pydanticモデルに変換して扱う。


## よくある問題と解決法

### 1. 循環インポート
```python
# 問題のあるコード
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ztb.training.trainer import Trainer
```

---

## v448: プロジェクト固有の型安全 & 実行ガイド ✅

この節では、v448で導入した重要なコンポーネント（BehavioralPenaltyCalculatorの語義修正、MTFWeightManagerの追加、RewardCalculator への統合）と、ローカルでのテスト実行・トラブルシューティング手順をまとめます。

### 変更点ハイライト
- BehavioralPenaltyCalculator: lookbackの構成は「直近 (lookback + 1)」を含むように修正されています。HOLD(0)は一部のwhipsaw検出で無視され、`consistency_min_actions`に満たない場合はペナルティ対象外になります。ペナルティ値は内部的に負値で保存されます。
- RewardCalculator: `MTFWeightManager` を統合し、`last_reward_components` に `mtf_weights` テレメトリを出力するようになりました。
- MTFWeightManager: 安全な保守的更新ルール(α smoothing, min/max bounds, renormalize) を実装済みです。
- 設定・テスト: `config/v448/mtf_mini_test.json` と `tools/run_child_trainer_wrapper.py` の `--diagnostics-only` フラグ等で、CI / ローカル検証を行えるようになっています。

### ローカルでのテスト実行（推奨）
下記の最小セットは、レイヤー4〜5で重要なロジックをテストします。Windows + PyTorch 環境に依存する重いテストは分離しておくことを推奨します。

1) BehavioralPenaltyCalculator のユニットテスト:
```cmd
pytest -q tests/unit/trading/environment/components/test_behavioral_penalty_calculator.py
```

2) MTFWeightManager ユニットテスト:
```cmd
pytest -q tests/unit/training/mtf
```

3) RewardCalculator 実装の統合テスト（mtf telemetryなど）:
```cmd
pytest -q tests/unit/trading/components/test_reward_calculator.py
```

4) Quick AB-run テスト（スクリプト存在確認, 軽量 smoke test）:
```cmd
pytest -q tests/unit/tools/test_run_quick_mtf_ab.py
```

5) 子プロセスでの import/トレーニング診断（Windows の import / DLL エラー検出）:
```cmd
python tools/run_child_trainer_wrapper.py --config config/v448/sac_v448_emergency_fix.json --diagnostics-only
```

### よくある実行時エラーと対処法
- DLL/WinError 1114 (c10.dll load failure)：CPU-onlyビルドの `torch` をインストールするか、Torch import をガードする実装を利用します。CIでは `tools/run_child_trainer_wrapper.py --diagnostics-only` を追加して、環境ごとの import 問題を早期に検出してください。
- ImportError: cannot import name 'AlertCondition' / 'AlertStatus' など `ztb.adaptation.monitoring` の型定義に関連するエラー：
    - 起点となるモジュールを特定し、`from ... import` の記述ミスや循環参照を検証してください（例: `ztb/adaptation/monitoring/types.py` に該当シンボルが未定義 or 名前が変わっているケース）。
    - `pytest -q tests/unit/adaptation` で関連ユニットテストを実行して、問題箇所を切り分けてください。
- TypeError: non-default argument 'data_config' follows default argument：dataclass の引数の順序で必ずデフォルトのない引数を先に書きます。`ztb/training/config/configuration_manager.py`の dataclass を見直してください。

### CI に入れるチェック（推奨）
- `tools/run_child_trainer_wrapper.py --diagnostics-only` を smoke-tests ジョブに追加して import/DLL問題を早期検出
- `pytest -q tests/unit/training/mtf` を unit-tests ジョブで実行
- quick AB-run（3 seeds × 1000 steps）を integration smoke ジョブで実行して、`mtf.weight_optimizer.enabled` の ON/OFF で BIAs Z 分析を行う

### Tips & Notes 💡
- 既に v448ドキュメント (`docs/SAC_v448_DEVELOPMENT_PLAN.md` / `docs/SAC_v448_LAYER5_DESIGN_SPEC.md`) に実装ロードマップと acceptance criteria を記載しています。CI に quick AB-run を追加することで、MTFの挙動が再現可能か素早く検証できます。
- Windowsで開発する際は、PyTorchのCUDA依存を避けるため `pip install torch --index-url https://download.pytorch.org/whl/cpu` のような CPU-only wheel を利用してください。

---
小さな変更ですが、`type_safety_guide.md` へv448導入に関する実行/デバッグ手順を追記しました。CI の修正 (smoke-tests) も検討してください。

### 実施済み・推奨修正（開発ログ）
- 既に適用: `ztb/adaptation/monitoring/monitor.py` に `PerformanceMonitor = AdaptationPerformanceMonitor` の互換エイリアスを追加。これにより、`from ztb.adaptation.monitoring.monitor import PerformanceMonitor` を期待する既存のコード / テストが安定しました。
- 推奨: `ztb/training/config/configuration_manager.py` の dataclass は `non-default` 引数を先に定義する必要があるため、`data_config` の位置を見直してください（TypeError: non-default argument 'data_config' follows default argument）。
- 推奨: `ztb/training/gradient_accumulation.py` 等の typing 注釈で `Tuple` が未定義で NameError が出る場合は、`from typing import Tuple` を追加してください。
- 推奨: import の循環/タイプミス（`AlertCondition` などの名前が期待通りに定義されない）に備えて、モジュールの export / re-export 設計を確認して `__all__` を整備してください。

これらの修正は、`tools/run_child_trainer_wrapper.py --diagnostics-only` が CI の quick smoke-test で精度よく import の問題を検出できるようにするために非常に有効です。

### 手元での検証結果（簡易サマリ）
- `tests/unit/trading/environment/components/test_behavioral_penalty_calculator.py` — PASS ✅
- `tests/unit/training/mtf` — PASS ✅
- `tests/unit/trading/components/test_reward_calculator.py` — PASS ✅
- `tests/unit/tools/test_run_quick_mtf_ab.py` — PASS ✅
- `tools/run_child_trainer_wrapper.py --diagnostics-only` — **部分的に失敗**（`ztb.adaptation.monitoring`の `AlertCondition`、`AlertStatus` の import エラー、dataclass の引数順による TypeError など。対応が必要です）

上の結果は、ローカルの Windows + CPU-only torch 環境で実行した際のサマリです。 child wrapper の診断は、CI の OS 行列で早期検知できるため、ワークフローに含めることを強く推奨します。

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
