# コードレビュー・デバッグ依頼

## 📋 プロジェクト概要

**プロジェクト名:** zaif-trade-bot
**言語:** Python 3.11+
**フレームワーク:** Stable-Baselines3 (MaskablePPO), Gymnasium
**目的:** 仮想通貨(BTC/JPY)取引のための強化学習エージェント

---

## 🎯 依頼内容

**検証スクリプトの重大なバグを発見・修正しましたが、追加のレビューとデバッグをお願いします。**

特に以下の点を重点的に確認してください:

1. **トレーニングスクリプトの整合性チェック**
2. **MaskablePPO関連のAPI使用の正確性**
3. **型安全性とエラーハンドリング**
4. **メモリリークやパフォーマンス問題**
5. **その他の潜在的なバグ**

---

## 🐛 これまでに発見・修正したバグ

### バグ#1: 評価スクリプトでaction_masks未使用 (致命的)

**影響ファイル:**
- `validate_model_behavior.py`
- `backtest_model.py`

**問題:**
```python
# ❌ 間違い (MaskablePPOでaction_masksを渡していなかった)
action, _ = model.predict(obs, deterministic=True)
```

**修正:**
```python
# ✅ 正解
if model_type == "MaskablePPO":
    action_masks = env.get_action_masks()
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
else:
    action, _ = model.predict(obs, deterministic=False)
```

**影響:**
- モデルが96% HOLD、0% SELLと誤検出されていた
- 実際は73% HOLD、13.5% BUY、13.4% SELL

### バグ#2: トレード検出ロジックの不備

**影響ファイル:**
- `backtest_model.py`

**問題:**
- ポジション反転(Long→Short)を検出できていなかった
- `allow_reverse=True`の仕様を考慮していなかった

**修正:**
- ポジション反転時のトレード記録ロジックを追加

**影響:**
- 247トレードが0と誤認されていた

---

## 📂 重点レビュー対象ファイル

### 優先度: 高 🔴

1. **ztb/training/ppo_trainer.py**
   - PPOTrainer, PPOTrainerAutoHaltの実装
   - ユーザーが手動で編集済み
   - MaskablePPOの使用方法を確認

2. **ztb/training/sell_mitigation_ppo_trainer.py**
   - SELL bias緩和機能の実装
   - ユーザーが手動で編集済み
   - Lagrange制約、Gradient Probe、PAN、Entropy Controller等

3. **ztb/trading/environment/environment.py** (960行)
   - HeavyTradingEnv実装
   - action masking, position management
   - reward calculation

4. **ztb/trading/environment/components/position_manager.py**
   - ポジション管理ロジック
   - `allow_reverse=True`時の動作

### 優先度: 中 🟡

5. **validate_model_behavior.py** (修正済み)
   - 紙上検証スクリプト
   - action_masks対応を確認

6. **backtest_model.py** (修正済み)
   - バックテストスクリプト
   - トレード検出ロジックを確認

7. **run_training.py**
   - トレーニング実行スクリプト

8. **ztb/training/callbacks_legacy.py**
   - カスタムコールバック実装

### 優先度: 低 🟢

9. **ztb/training/custom_ppo.py**
   - CustomPPO実装

10. **ztb/training/lagrange_constraint.py**
    - Lagrange制約実装

---

## 🔍 具体的な確認ポイント

### 1. MaskablePPO API使用の正確性

**確認内容:**
```python
# トレーニング時
model.learn(
    total_timesteps=...,
    callback=...  # ここでaction_masksが自動適用されるか?
)

# 予測時
model.predict(
    observation=obs,
    action_masks=masks,  # ← 必須!
    deterministic=False   # ← 学習時と同じ
)
```

**質問:**
- トレーニングスクリプト内でaction_masksの扱いは正しいか?
- ActionMaskerラッパーの使用方法は適切か?
- カスタムコールバック内でaction_masksを正しく取得しているか?

### 2. ポジション管理ロジック

**確認内容:**
```python
# environment.py & position_manager.py
class PositionManager:
    def execute_action(self, action: int, current_step: int):
        if action == 1:  # BUY
            if self.position < 0:  # Short position
                self.close_position()
                if self.config.allow_reverse:
                    self.open_position(1, current_step)  # ← 0を経由せず直接Long
```

**質問:**
- `allow_reverse=True`時の動作は正しく実装されているか?
- ポジションが0.5 → -0.5と変化する場合の処理は適切か?
- トレード記録のタイミングは正しいか?

### 3. Reward計算の整合性

**確認内容:**
```python
# components/reward_calculator.py
def calculate_reward(self, ...):
    reward = (
        pnl_component +
        diversity_bonus +
        lagrange_penalty +
        ...
    )
```

**質問:**
- 報酬スケーリングは適切か?
- Lagrange制約のペナルティは正しく適用されているか?
- NaN/Infチェックは十分か?

### 4. メモリ管理

**確認内容:**
```python
# モデル保存/読込時
model.save(...)
gc.collect()  # ← ガベージコレクションは適切に呼ばれているか?
```

**質問:**
- メモリリークの可能性はないか?
- 大きな配列の不要な複製はないか?
- DataFrameの肥大化は防げているか?

### 5. エラーハンドリング

**確認内容:**
```python
try:
    model.learn(...)
except Exception as e:
    logger.error(...)
    # ← エラー後の復旧処理は適切か?
```

**質問:**
- 例外処理は十分か?
- ログ出力は適切か?
- クリティカルセクションでのエラーハンドリングは?

---

## 📊 現在のモデル性能

### ppo_memory_optimized.zip (修正後の評価)

**アクション分布:**
- HOLD: 73.1%
- BUY: 13.5%
- SELL: 13.4%

**バックテスト結果:**
- トータルリターン: +0.13%
- 勝率: 55.47%
- トレード数: 247
- シャープレシオ: 25.30
- 最大ドローダウン: -6.15%

**問題点:**
- HOLD bias (73%) が依然として高い
- リターンが実用レベルではない

---

## 🎯 期待する改善

### 短期目標
1. ✅ 検証スクリプトのバグ修正 (完了)
2. トレーニングスクリプトの潜在的バグ発見
3. コード品質の向上

### 中期目標
1. HOLD biasの軽減 (目標: 50%以下)
2. アクション多様性の向上 (Balance Score > 0.5)
3. 実用的なリターン達成 (>1%)

---

## 🛠️ 環境情報

**Python:** 3.11+
**主要ライブラリ:**
- stable-baselines3 (MaskablePPO)
- sb3-contrib
- gymnasium
- pandas, numpy

**開発環境:** Windows (cmd.exe/PowerShell)

---

## 📁 プロジェクト構造

```
zaif-trade-bot/
├── ztb/
│   ├── trading/
│   │   └── environment/
│   │       ├── environment.py (960行)
│   │       └── components/
│   │           ├── position_manager.py
│   │           └── reward_calculator.py
│   └── training/
│       ├── ppo_trainer.py (436行, 手動編集済み)
│       ├── sell_mitigation_ppo_trainer.py (489行, 手動編集済み)
│       ├── custom_ppo.py
│       ├── callbacks_legacy.py
│       ├── lagrange_constraint.py
│       └── ...
├── validate_model_behavior.py (修正済み)
├── backtest_model.py (修正済み)
├── run_training.py
└── configs/training/ppo_memory_optimized.json
```

---

## 🚨 特に注意してほしいポイント

### 1. 型安全性の問題

**現状:**
- mypy で unused-ignore 警告あり
- 型ヒントが不完全な箇所がある

**依頼:**
- 型エラーの洗い出し
- 修正提案

### 2. MaskablePPO特有の問題

**現状:**
- 評価スクリプトでaction_masks漏れがあった

**依頼:**
- トレーニングスクリプトでも同様の問題がないか?
- ActionMaskerの使用方法は正しいか?

### 3. Lagrange制約の効果

**現状:**
- 学習中: SELL 19% (Lagrange強制)
- 評価時: SELL 13.4% (制約なし)
- → 本質的な方策改善になっていない?

**依頼:**
- Lagrange制約の実装を確認
- より効果的な実装を提案

### 4. パフォーマンス問題

**現状:**
- メモリ使用量が高い
- 学習が途中で止まることがあった

**依頼:**
- メモリリークの調査
- 最適化提案

---

## 📝 レビュー方法

### Step 1: 静的解析

```bash
# 型チェック
python -m mypy ztb/training/ --show-error-codes

# Linting
python -m pylint ztb/training/

# Import構造チェック
python -m importlinter
```

### Step 2: コードレビュー

以下のファイルを重点的にレビュー:
1. `ztb/training/ppo_trainer.py`
2. `ztb/training/sell_mitigation_ppo_trainer.py`
3. `ztb/trading/environment/environment.py`
4. `ztb/trading/environment/components/position_manager.py`

### Step 3: 動作確認

```bash
# 修正後のバックテスト再実行
python backtest_model.py --model-path models/ppo_memory_optimized.zip

# 修正後の紙上検証再実行
python validate_model_behavior.py --model-path models/ppo_memory_optimized.zip --episodes 10
```

---

## 📤 成果物

以下の形式でフィードバックをお願いします:

### 1. バグレポート
```markdown
## バグ: [タイトル]
**ファイル:** [ファイル名:行番号]
**深刻度:** [Critical/High/Medium/Low]
**問題:** [説明]
**修正案:** [コード例]
```

### 2. 改善提案
```markdown
## 改善提案: [タイトル]
**ファイル:** [ファイル名]
**現状:** [問題点]
**提案:** [改善方法]
**期待効果:** [効果]
```

### 3. 質問事項
```markdown
## 質問: [タイトル]
**箇所:** [ファイル名:行番号]
**質問内容:** [質問]
**背景:** [なぜこの質問が重要か]
```

---

## 🎯 最優先事項

1. **MaskablePPOのaction_masks漏れチェック** (最重要)
2. **ポジション管理ロジックの検証**
3. **メモリリーク調査**
4. **型安全性の改善**
5. **Lagrange制約の効果検証**

---

## 📚 参考資料

### 発見済みバグ詳細
- `BUG_FIX_REPORT.md` - バグ修正レポート
- `VALIDATION_FINAL_REPORT.md` - 最終検証レポート

### MaskablePPO API
```python
# 正しい使用方法
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 環境ラップ
env = ActionMasker(env, mask_fn)

# 学習
model = MaskablePPO("MlpPolicy", env)
model.learn(total_timesteps=10000)

# 予測
action_masks = env.get_action_masks()
action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
```

---

## ✅ チェックリスト

レビュー時に以下を確認してください:

- [ ] すべての`model.predict()`でaction_masksが渡されているか
- [ ] ActionMaskerの使用方法は正しいか
- [ ] ポジション反転時のトレード記録は正しいか
- [ ] 型ヒントは完全か
- [ ] エラーハンドリングは十分か
- [ ] メモリリークの可能性はないか
- [ ] ログ出力は適切か
- [ ] テストカバレッジは十分か
- [ ] ドキュメントは正確か
- [ ] パフォーマンスボトルネックはないか

---

## 💬 連絡事項

- 不明点があれば遠慮なく質問してください
- 部分的なレビューでも構いません
- コード例を含む具体的なフィードバックを希望します

---

**レビュー担当者:** [エージェント名]
**依頼日:** 2025年10月7日
**期限:** なし(できる範囲で)
**優先度:** 高

よろしくお願いします! 🙏
