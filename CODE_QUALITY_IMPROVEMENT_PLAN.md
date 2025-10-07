# コードベース検査・改善指示書

**日付**: 2025年10月7日  
**目的**: 型安全性向上、保守性強化、不整合検出・修正  
**対象**: zaif-trade-bot全体（特にztb/training/配下）

---

## 🎯 検査・改善の目的

1. **型安全性向上**: TypeHintの追加、型エラーの解消
2. **保守性強化**: インターフェース統一、命名規則の一貫性
3. **不整合解消**: 設定ファイル構造、パラメータ命名、デフォルト値の統一
4. **ドキュメント整備**: docstring追加、型情報の明示化

---

## 📋 Phase 1: 型安全性検査（優先度: 🔴 高）

### 1.1 型ヒント欠落の検出

**対象ファイル**:
- `ztb/training/*.py`（全トレーナー）
- `ztb/env/*.py`（環境クラス）
- `ztb/utils/*.py`（ユーティリティ）

**検査項目**:
```python
# ❌ 型ヒントなし（改善対象）
def train(self, session_id):
    return model

# ✅ 型ヒントあり（目標）
def train(self, session_id: str) -> Optional[CustomPPO]:
    return model
```

**検査コマンド**:
```bash
# mypy strict mode
mypy --strict ztb/training/ > mypy_strict_report.txt

# 型ヒント欠落カウント
grep -c "error: Missing type parameters" mypy_strict_report.txt
```

**改善作業**:
1. 全関数に引数・戻り値の型ヒント追加
2. `Optional`, `Union`, `Dict[str, Any]`の明示化
3. ジェネリクス（`List[T]`, `Dict[K, V]`）の活用

---

### 1.2 型エラーの解消

**既知の問題**:
```python
# 問題1: type: ignore の乱用
trainer = trainer_class(
    data_path=self.config.get("data_path"),  # type: ignore[arg-type]
)

# 改善案: 適切な型キャストまたはアサーション
data_path = self.config.get("data_path")
assert isinstance(data_path, str), "data_path must be a string"
trainer = trainer_class(data_path=data_path)
```

**検査コマンド**:
```bash
# type: ignore の使用箇所を検索
grep -rn "type: ignore" ztb/ > type_ignore_usage.txt

# カウント
grep -c "type: ignore" ztb/**/*.py
```

**改善作業**:
1. `type: ignore`を使わずに型エラーを解消
2. 必要な場合は具体的なエラーコード指定（例: `# type: ignore[arg-type]`）
3. コメントで理由を説明

---

### 1.3 Protocol/ABC の活用

**現状の問題**:
```python
# 暗黙的なインターフェース（duck typing）
class PPOTrainer:
    def train(self, session_id: str):
        pass

class CustomTrainer:
    def train(self, session_id: str):
        pass
```

**改善案**:
```python
from typing import Protocol

class Trainer(Protocol):
    """Trainer interface."""
    def train(self, session_id: str) -> Optional[Any]:
        ...

class PPOTrainer:
    def train(self, session_id: str) -> Optional[CustomPPO]:
        # Implementation
        pass
```

**改善作業**:
1. `ztb/training/protocols.py`作成
2. `Trainer`, `Environment`, `Callback`のProtocol定義
3. 既存クラスをProtocolに準拠させる

---

## 📋 Phase 2: 設定ファイル不整合検査（優先度: 🟠 中）

### 2.1 パラメータ命名の統一

**検出すべき不整合**:

| 問題 | 例 | 統一案 |
|------|-----|--------|
| スネークケース/キャメルケース混在 | `checkpoint_dir` vs `checkpointDir` | `checkpoint_dir`（スネークケース統一） |
| 省略形の不統一 | `ckpt_dir` vs `checkpoint_dir` | `checkpoint_dir`（省略しない） |
| 階層の不統一 | `output.checkpoint_dir` vs `checkpoint_dir` | `checkpoint_dir`（トップレベル統一） |

**検査スクリプト**:
```python
# scripts/check_config_consistency.py
import json
from pathlib import Path
from typing import Dict, Set

def check_config_consistency(config_dir: Path) -> Dict[str, Set[str]]:
    """設定ファイルのキー名の一貫性をチェック."""
    all_keys: Dict[str, Set[str]] = {}
    
    for config_file in config_dir.glob("**/*.json"):
        with open(config_file) as f:
            config = json.load(f)
            keys = set(config.keys())
            all_keys[config_file.name] = keys
    
    # キー名の差分検出
    common_keys = set.intersection(*all_keys.values())
    unique_keys = {
        name: keys - common_keys
        for name, keys in all_keys.items()
    }
    
    return unique_keys

if __name__ == "__main__":
    inconsistencies = check_config_consistency(Path("configs/train"))
    for file, unique in inconsistencies.items():
        if unique:
            print(f"{file}: {unique}")
```

**改善作業**:
1. スクリプト実行で不整合を検出
2. 統一ルールを決定（スネークケース、省略なし）
3. 全設定ファイルを一括修正
4. `config_schema.json`でバリデーション

---

### 2.2 デフォルト値の統一

**検出すべき不整合**:
```python
# unified_trainer.py
checkpoint_interval = self.config.get("checkpoint_interval", 25000)

# run_1m.py
checkpoint_interval = args.checkpoint_interval  # default=10000

# 問題: 同じパラメータで異なるデフォルト値
```

**検査スクリプト**:
```bash
# デフォルト値の検索
grep -rn "\.get\(\"checkpoint_interval\"" ztb/ | grep -oP "default\s*=\s*\K[0-9]+"
```

**改善作業**:
1. デフォルト値を`constants.py`に集約
2. 全ファイルで同じ定数を参照
3. アルゴリズム別に異なる場合はコメントで明記

---

### 2.3 必須パラメータの明示化

**現状の問題**:
```python
# 暗黙的な必須パラメータ
data_path = self.config.get("data_path")  # Noneの可能性
```

**改善案**:
```python
from typing import TypedDict, Required

class TrainingConfig(TypedDict):
    algorithm: Required[str]
    data_path: Required[str]
    total_timesteps: Required[int]
    checkpoint_interval: int  # Optional (has default)
    
# 使用
config: TrainingConfig = load_config("config.json")
```

**改善作業**:
1. `ztb/config/schemas.py`作成
2. `TypedDict`で設定スキーマ定義
3. `Required`で必須パラメータを明示
4. バリデーション関数追加

---

## 📋 Phase 3: インターフェース統一（優先度: 🟠 中）

### 3.1 Trainerインターフェースの統一

**現状の問題**:
```python
# PPOTrainer
def __init__(self, data_path: str, config: PPOConfig, checkpoint_dir: str, checkpoint_interval: int = 10000)

# SELLBiasMitigationPPOTrainer
def __init__(self, data_path: str, config: PPOConfig, checkpoint_dir: str, checkpoint_interval: int = 10000, enable_lagrange: bool = True, ...)

# 問題: シグネチャが異なる
```

**改善案**:
```python
@dataclass
class TrainerParams:
    """共通トレーナーパラメータ."""
    data_path: str
    checkpoint_dir: str
    checkpoint_interval: int = 10000

@dataclass
class SELLMitigationParams(TrainerParams):
    """SELL緩和用追加パラメータ."""
    enable_lagrange: bool = True
    enable_probes: bool = True
    enable_weights: bool = True
    probe_csv_path: Optional[str] = None

class PPOTrainer:
    def __init__(self, config: PPOConfig, params: TrainerParams):
        self.config = config
        self.params = params
```

**改善作業**:
1. `TrainerParams`データクラス定義
2. 全トレーナーで統一インターフェース採用
3. 拡張時は継承を活用

---

### 3.2 Callbackインターフェースの統一

**検査項目**:
- `on_step()`, `on_training_end()`のシグネチャ統一
- 戻り値の型統一（`bool` vs `None`）
- エラーハンドリングの統一

**改善作業**:
1. `BaseCallback`のProtocol定義
2. 全カスタムCallbackをProtocolに準拠
3. 型チェックでシグネチャ違反を検出

---

## 📋 Phase 4: ドキュメント整備（優先度: 🟡 低）

### 4.1 Docstring追加

**対象**:
- 全public関数・クラス
- 複雑なprivate関数

**フォーマット**:
```python
def train(
    self,
    session_id: str,
    total_timesteps: int = 100000,
) -> Optional[CustomPPO]:
    """
    PPOアルゴリズムで学習を実行.

    Args:
        session_id: セッション識別子（ログ・チェックポイント用）
        total_timesteps: 総学習ステップ数（デフォルト: 100000）

    Returns:
        学習済みCustomPPOモデル。学習失敗時はNone

    Raises:
        ValueError: 設定が不正な場合
        FileNotFoundError: データファイルが見つからない場合

    Examples:
        >>> trainer = PPOTrainer(config, params)
        >>> model = trainer.train("test_session", total_timesteps=10000)
    """
    pass
```

**検査コマンド**:
```bash
# docstring欠落の検出
pydocstyle ztb/ > pydocstyle_report.txt
```

**改善作業**:
1. 全public APIにdocstring追加
2. Google styleまたはNumPy styleで統一
3. 型情報とdocstringの一致を確認

---

### 4.2 型スタブファイル（.pyi）の作成

**対象**:
- 外部ライブラリで型情報がない場合
- 複雑な型定義を分離したい場合

**例**:
```python
# ztb/training/ppo_trainer.pyi
from typing import Optional
from .custom_ppo import CustomPPO
from .config import PPOConfig

class PPOTrainer:
    def __init__(
        self,
        data_path: str,
        config: PPOConfig,
        checkpoint_dir: str,
        checkpoint_interval: int = ...,
    ) -> None: ...
    
    def train(self, session_id: str) -> Optional[CustomPPO]: ...
```

---

## 📋 Phase 5: テストカバレッジ向上（優先度: 🟢 推奨）

### 5.1 型安全性のテスト

**テストケース**:
```python
# tests/training/test_type_safety.py
from typing import get_type_hints
from ztb.training.ppo_trainer import PPOTrainer

def test_ppo_trainer_type_hints():
    """PPOTrainerの型ヒントが正しく定義されているか."""
    hints = get_type_hints(PPOTrainer.train)
    assert "session_id" in hints
    assert hints["session_id"] == str
    assert hints["return"].__origin__ == Union  # Optional[CustomPPO]
```

**改善作業**:
1. 各トレーナーの型ヒント検証テスト追加
2. CI/CDで型チェック自動化
3. カバレッジ目標: 90%以上

---

### 5.2 設定ファイルバリデーションテスト

**テストケース**:
```python
def test_config_schema_validation():
    """設定ファイルがスキーマに準拠しているか."""
    from ztb.config.schemas import TrainingConfig
    import json
    
    config = json.load(open("configs/train/ensemble_A_1M.json"))
    # TypedDictバリデーション
    assert "algorithm" in config
    assert "data_path" in config
    assert isinstance(config["total_timesteps"], int)
```

---

## 🚀 実行プラン

### Step 1: 現状分析（1-2日）

```bash
# 1. 型エラーレポート生成
mypy --strict ztb/ > mypy_strict_report.txt

# 2. 設定ファイル不整合検出
python scripts/check_config_consistency.py > config_inconsistencies.txt

# 3. type: ignore 使用箇所カウント
grep -rn "type: ignore" ztb/ | wc -l

# 4. docstring欠落検出
pydocstyle ztb/ > pydocstyle_report.txt
```

**成果物**: 
- `mypy_strict_report.txt`
- `config_inconsistencies.txt`
- `type_ignore_usage.txt`
- `pydocstyle_report.txt`

---

### Step 2: 優先度付け（0.5日）

**基準**:
1. 🔴 高: 型安全性、必須パラメータの明示化
2. 🟠 中: 設定ファイル統一、インターフェース統一
3. 🟡 低: ドキュメント整備

**成果物**: 
- `IMPROVEMENT_PRIORITY_LIST.md`（改善優先度リスト）

---

### Step 3: 段階的改善（1-2週間）

**Week 1: 型安全性（Phase 1）**
- Day 1-2: 型ヒント追加（ztb/training/）
- Day 3-4: type: ignore 削減
- Day 5: Protocol/ABC導入

**Week 2: 設定・インターフェース（Phase 2-3）**
- Day 1-2: 設定ファイル統一
- Day 3-4: Trainerインターフェース統一
- Day 5: テスト追加

---

### Step 4: 検証（1日）

```bash
# 型チェック
mypy --strict ztb/

# テスト実行
pytest tests/ --cov=ztb --cov-report=html

# 設定ファイルバリデーション
python scripts/validate_all_configs.py
```

**合格基準**:
- mypy エラー: 0件
- テストカバレッジ: >80%
- 設定ファイルバリデーション: 全パス

---

## 📊 成果物テンプレート

### IMPROVEMENT_REPORT.md

```markdown
# コードベース改善レポート

## 実施日: YYYY-MM-DD

### Phase 1: 型安全性向上
- ✅ 型ヒント追加: 150関数
- ✅ type: ignore削減: 45 → 12
- ✅ Protocol導入: 3インターフェース

### Phase 2: 設定ファイル統一
- ✅ キー名統一: 25ファイル
- ✅ デフォルト値統一: 10パラメータ
- ✅ スキーマ定義: TrainingConfig, EnvironmentConfig

### Phase 3: インターフェース統一
- ✅ TrainerParams導入
- ✅ Callbackシグネチャ統一

### 検証結果
- mypy エラー: 0件（改善前: 237件）
- テストカバレッジ: 85%（改善前: 62%）
- 設定ファイルバリデーション: 100%パス

### 今後の課題
- [ ] ドキュメント整備（Phase 4）
- [ ] 追加のProtocol定義
- [ ] パフォーマンステスト追加
```

---

## 🛠️ 便利スクリプト

### scripts/check_config_consistency.py

上記のスクリプトを実装

### scripts/validate_all_configs.py

```python
"""全設定ファイルのバリデーション."""
import json
from pathlib import Path
from typing import List, Dict, Any

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """設定ファイルをスキーマに対してバリデーション."""
    errors = []
    
    # 必須キーチェック
    required_keys = schema.get("required", [])
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    
    # 型チェック
    # ... (実装)
    
    return errors

if __name__ == "__main__":
    # 全設定ファイルをバリデーション
    configs_dir = Path("configs/train")
    all_passed = True
    
    for config_file in configs_dir.glob("*.json"):
        with open(config_file) as f:
            config = json.load(f)
            errors = validate_config(config, TRAINING_SCHEMA)
            
            if errors:
                print(f"❌ {config_file.name}:")
                for error in errors:
                    print(f"  - {error}")
                all_passed = False
            else:
                print(f"✅ {config_file.name}")
    
    exit(0 if all_passed else 1)
```

---

## 📚 参考資料

1. **型ヒント**: [PEP 484](https://peps.python.org/pep-0484/)
2. **Protocol**: [PEP 544](https://peps.python.org/pep-0544/)
3. **TypedDict**: [PEP 589](https://peps.python.org/pep-0589/)
4. **mypy**: [mypy Documentation](https://mypy.readthedocs.io/)
5. **pydocstyle**: [pydocstyle Docs](http://www.pydocstyle.org/)

---

## ✅ チェックリスト

### 開始前
- [ ] mypyインストール（`pip install mypy`）
- [ ] pydocstyleインストール（`pip install pydocstyle`）
- [ ] ブランチ作成（`git checkout -b feature/type-safety-improvements`）

### Phase 1完了時
- [ ] mypy --strict でエラー50%削減
- [ ] type: ignore を50%削減
- [ ] Protocol定義3つ以上

### Phase 2完了時
- [ ] 設定ファイルキー名統一
- [ ] デフォルト値を constants.py に集約
- [ ] スキーマ定義完了

### Phase 3完了時
- [ ] TrainerParams導入
- [ ] 全トレーナーでインターフェース統一
- [ ] テストカバレッジ80%以上

### 最終確認
- [ ] mypy --strict 全パス
- [ ] pytest 全パス
- [ ] 設定ファイルバリデーション全パス
- [ ] ドキュメント更新（IMPROVEMENT_REPORT.md）

---

**このドキュメントをCopilotに渡して、段階的に改善を進めてください！**
