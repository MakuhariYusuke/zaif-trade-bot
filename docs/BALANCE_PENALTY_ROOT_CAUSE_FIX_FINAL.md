# SAC v444 Balance Penalty Fix - 最終的な根本原因と解決策

## 問題の概要

トレーニング中のアクション分布が極度に偏っていた：
- **SELL**: 93.34%
- **BUY**: 3.36%
- **HOLD**: 3.30%

予想される分布：
- 各アクション: 33% 程度

## 根本原因の追跡

### Phase 1: 表面的な原因（前回のセッション）
- `reward_calculator.py` が複数の `curriculum_stage` 値をサポートしていないと思われた
- 4つのステージ値 (`forced_balance`, `balanced_penalty`, `balance_optimization`, `balance_penalty`) を明示的に列挙するコード修正を実施
- **結果**: 改善なし - 構成上の問題が根深いことが判明

### Phase 2: 深層原因の特定（現在のセッション）

#### 発見1: 設定値の流れが途絶している
```
v444 config: training.curriculum_learning.curriculum_stage = "balanced_penalty"
    ↓
V4XXConfigConverter.convert_v444_to_unified()
    ↓
training.environment.curriculum_stage = ? (設定されていない)
    ↓
RewardCalculator が environment から curriculum_stage を読み込み
    ↓
??? デフォルト値に置き換わっている
```

#### 発見2: デフォルト値が `"forced_balance"` に硬く設定されていた
- ファイル: `ztb/trading/environment/utils/config.py` (Line 107)
- コード: `curriculum_stage: str = "forced_balance"`
- 問題: V4XXConfigConverter が environment に設定した `curriculum_stage` 値が、後で `EnvironmentConfig` インスタンス化時にデフォルト値で上書きされていた

## 実装された解決策

### 修正 1: デフォルト値を None に変更
**ファイル**: `ztb/trading/environment/utils/config.py` Line 107

**変更前**:
```python
curriculum_stage: str = "forced_balance"
```

**変更後**:
```python
curriculum_stage: Optional[str] = None  # Set from training.curriculum_learning
```

**効果**:
- デフォルト値がないため、V4XXConfigConverter から渡された値が保持される
- `training.curriculum_learning.curriculum_stage` の設定値が、`training.environment.curriculum_stage` を通じて RewardCalculator に正しく伝播される

### 修正 2: RewardCalculator は既に複数ステージをサポート（前回実装）
`reward_calculator.py` の複数ステージサポートはそのまま有効：
```python
balance_penalty_enabled_stages = (
    "forced_balance",
    "balanced_penalty",     # ← 設定値から指定される
    "balance_optimization",
    "balance_penalty",
)
```

### 修正 3: Pydantic スキーマで curriculum_learning を保持（前回実装済み）
`ztb/config/schema.py` で curriculum_learning が明示的にスキーマに定義済み

## 修正の検証

### 実行した検証スクリプト: `verify_curriculum_fix.py`
```
✅ SUCCESS: curriculum_stage flows correctly!
   Config value 'balanced_penalty' correctly propagates to environment

1. Original config curriculum_stage: balanced_penalty
2. Converted to training.environment.curriculum_stage: balanced_penalty
3. EnvironmentConfig default curriculum_stage: None (after fix)
4. EnvironmentConfig instance curriculum_stage: balanced_penalty
```

## 期待される動作改善

### トレーニング実行時のログ
**修正前**:
```
BALANCE_PENALTY (forced_balance): total_actions=20, buy=0.000, sell=1.000, hold=0.000
↑ forced_balance が使用されていた（デフォルト値）
```

**修正後**:
```
BALANCE_PENALTY (balanced_penalty): total_actions=20, buy=..., sell=..., hold=...
↑ balanced_penalty が使用される（設定値）
```

### 期待される結果
- アクション分布が理想値（各33%）に近づく
- RewardCalculator が SELL バイアス時に正しくペナルティを適用
- 継続的なトレーニングで動作が改善

## 実装の責任感

このバグは以下の理由で見落とされやすかった：
1. **デフォルト値の暗黙的なオーバーライド**: 設定値を渡しても、Dataclass のデフォルト値で置き換わる
2. **ログの一貫性**: `curriculum_stage` の値が正確に出力されていたが、「forced_balance」という不正な値が当たり前のように見えた
3. **アーキテクチャの複雑性**: 設定値が複数のレイヤーを通じて流れるが、各レイヤーで検証が不足していた

## 学習ポイント

✅ **構成システムの堅牢性**: デフォルト値は明示的かつ無意味な値（`None`）にすべき
✅ **E2E 検証**: 設定値が実際の実行時に正しく使用されているか、テストで確認する必要がある
✅ **SOLID 原則**: 単一責任が重要 - V4XXConfigConverter はその責任を果たしているが、デフォルト値がそれを無効化していた

## ファイル変更一覧

1. `ztb/trading/environment/utils/config.py` (Line 107)
   - `curriculum_stage: str = "forced_balance"` → `curriculum_stage: Optional[str] = None`

2. `ztb/utils/v4xx_config_converter.py` (Lines 200-212) - 前回実装済み
   - curriculum_stage を training.environment に明示的にマッピング

3. `ztb/config/schema.py` - 前回実装済み
   - CurriculumLearningConfig の明示的定義
   - TrainingConfig に curriculum_learning フィールド追加
   - extra="allow" で将来の拡張性確保

## 次のステップ

1. ✅ `verify_curriculum_fix.py` で E2E 検証完了
2. 📊 実際のトレーニング実行で改善を確認
3. 📈 アクション分布が理想値に改善したことを統計で確認
4. 🧪 回帰テストを追加して、デフォルト値の問題が再発しないようにする
