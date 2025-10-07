# 🔴 統合失敗診断レポート - 10k Smoke Test

## エグゼクティブサマリー

**結論**: 4つの改修機能は **実装されたが、学習ループに統合されていない**

- ✅ コード実装: 完了
- ✅ ユニットテスト: 全パス
- ❌ **統合**: 失敗 ← **現在地**
- ⏸️ 効果検証: 未実施（統合が前提条件）

---

## 🔍 根本原因の特定

### 問題の本質

**SB3 (Stable-Baselines3) のPPOアルゴリズムは、advantage計算・エントロピー計算・バッチサンプリングを内部で実行している**

我々の実装:
```python
# ztb/training/sell_mitigation_ppo_trainer.py
class SELLBiasMitigationPPOTrainer(PPOTrainer):
    def __init__(self):
        self.pan_normalizer = PerActionAdvantageNormalizer()  # ✅ 作成
        self.entropy_controller = TargetEntropyController()   # ✅ 作成
        self.stratified_sampler = StratifiedSampler()        # ✅ 作成
        # しかし... これらは使われていない!
```

**なぜ使われないのか**:

SB3のPPOは以下を内部で実行:
1. **Advantage計算**: `sb3_contrib/ppo/ppo.py` 内で `compute_advantages()`
2. **エントロピー計算**: Policy networkの `forward()` 内で
3. **バッチサンプリング**: `RolloutBuffer.get()` 内で

→ 我々のカスタムコンポーネントは **呼ばれる場所がない**

---

## 📊 証拠データ

### 統計がすべて0

```
pan/action_counts: [0, 0, 0]           # データ処理なし
pan/total_samples: 0                   # サンプル0件

entropy/num_updates: 0                 # 更新なし
entropy/current_alpha: 0.01           # 初期値のまま

stratified/bucket_r0_a0~r2_a2: ALL 0  # 全バケット0
```

### Probe CSVが空

```bash
$ cat checkpoints/smoke_test_10k/sell_probe.csv
step,grad_norm_sell,grad_norm_ma,advantage_sell,advantage_ma,is_healthy,consecutive_unhealthy
# データ行: 0件
```

### Lagrangeが動作せず

```
lagrange/r_sell_mean: 0     # SELL率0%
lagrange/lambda_dual: 0     # 制約未適用
```

---

## 🏗️ アーキテクチャ上の問題

### 現在の構造 (失敗)

```
┌─────────────────────────────────────┐
│   SELLBiasMitigationPPOTrainer      │
├─────────────────────────────────────┤
│ ✓ PAN作成                           │
│ ✓ Entropy Controller作成            │
│ ✓ Stratified Sampler作成            │
│                                     │
│ ✗ これらを使う場所なし!              │
└─────────────────────────────────────┘
           ↓ 呼び出し
┌─────────────────────────────────────┐
│   Stable-Baselines3 PPO             │
├─────────────────────────────────────┤
│ • 独自のAdvantage計算                │
│ • 独自のエントロピー計算              │
│ • 独自のバッチサンプリング            │
│                                     │
│ → カスタムコンポーネントを無視       │
└─────────────────────────────────────┘
```

### 必要な構造

```
┌─────────────────────────────────────┐
│   Custom PPO Algorithm              │
├─────────────────────────────────────┤
│ 1. Rollout収集                      │
│ 2. ★ PAN でAdvantage正規化          │
│ 3. ★ Stratified でバッチ作成        │
│ 4. ★ Entropy Controller で温度更新  │
│ 5. Policy/Value更新                 │
└─────────────────────────────────────┘
```

---

## 🛠️ 修正オプション

### Option A: SB3 PPOアルゴリズムのオーバーライド ⭐ 推奨

**難易度**: 🔴🔴🔴 高い  
**効果**: 🟢🟢🟢 完全統合  
**実装時間**: 2〜4時間

```python
class CustomPPO(MaskablePPO):
    """SB3のPPOを継承し、学習ループを上書き"""
    
    def train(self) -> None:
        # 1. Rollout取得
        rollout_data = self.rollout_buffer.get(self.batch_size)
        
        # 2. ★ PANでAdvantage正規化
        rollout_data.advantages = self.pan_normalizer.normalize(
            rollout_data.advantages,
            rollout_data.actions
        )
        
        # 3. ★ Stratified Samplerでバッチ再構築
        indices = self.stratified_sampler.sample_batch(...)
        rollout_data = rollout_data[indices]
        
        # 4. Policy更新
        # ...
        
        # 5. ★ Entropy Controllerで温度更新
        entropy = compute_entropy(action_logits)
        self.ent_coef = self.entropy_controller.update(entropy)
```

**メリット**:
- 完全な制御
- すべての改修が確実に適用される

**デメリット**:
- SB3内部を深く理解する必要
- メンテナンスコスト高

---

### Option B: Wrapperパターン（部分的統合）

**難易度**: 🟡🟡 中程度  
**効果**: 🟡🟡 部分的  
**実装時間**: 1〜2時間

```python
class AdvantageNormalizationWrapper:
    """Rollout Bufferをラップしてadvantageを変換"""
    
    def get(self, batch_size):
        data = self.buffer.get(batch_size)
        # PANを適用
        data.advantages = self.pan_normalizer.normalize(...)
        return data
```

**メリット**:
- SB3のコア変更不要
- 比較的実装が簡単

**デメリット**:
- すべての改修を統合できない可能性
- Stratified Samplingの統合が難しい

---

### Option C: PyTorch Lightning風カスタムTrainer

**難易度**: 🔴🔴🔴🔴 非常に高い  
**効果**: 🟢🟢🟢🟢 最高  
**実装時間**: 1週間以上

SB3を使わず、フルスクラッチでPPOを実装。

**メリット**:
- 完全な自由度
- すべての改修を理想的に統合

**デメリット**:
- 工数が膨大
- バグリスク高
- SB3の最適化を失う

---

### Option D: 環境側での疑似統合（緊急回避）

**難易度**: 🟢 低い  
**効果**: 🔴 限定的  
**実装時間**: 30分

```python
class BiasAwareEnv(HeavyTradingEnv):
    """環境側でSELLをブーストする報酬設計"""
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # SELL時に報酬ブースト（擬似的なバイアス緩和）
        if action == 2:  # SELL
            reward *= 1.5  # ブースト
        
        return obs, reward, done, info
```

**メリット**:
- 実装が簡単
- 即座にテスト可能

**デメリット**:
- 根本解決ではない
- PAN/Entropy/Stratifiedの効果を検証できない
- 報酬ハッキングのリスク

---

## 🎯 推奨アクション

### 即座に実施 (1時間以内)

1. **Option Dを実装** - 緊急回避として環境側でSELLをブースト
   ```python
   # quick_fix_env.py を作成
   # 10k再実行して最低限の動作確認
   ```

2. **統合失敗の事実をステークホルダーに報告**
   - 4つの機能は実装済み
   - ユニットテスト全パス
   - しかし学習ループへの統合が未完了

### 短期対応 (1〜2日)

3. **Option Bを実装** - Wrapperパターンで部分統合
   - 最低限PANとTarget Entropyを適用
   - Stratifiedは一旦保留

4. **10k再実行**
   - 部分統合版で効果を測定
   - SELL率が改善するか確認

### 中期対応 (1週間)

5. **Option Aを実装** - Custom PPOで完全統合
   - SB3のPPOを継承
   - 学習ループを完全にカスタマイズ
   - すべての改修を適用

6. **50k validation**
   - 完全統合版で長時間トレーニング
   - 当初の目標を検証

---

## 📝 教訓

### 何がうまくいったか

1. ✅ **モジュール設計**: PAN/Entropy/Stratifiedは独立してテスト可能
2. ✅ **ユニットテスト**: 各コンポーネントは正しく動作
3. ✅ **ドキュメント**: 実装意図が明確

### 何が失敗したか

1. ❌ **統合計画の不足**: 「どこで使うか」の設計が不十分
2. ❌ **SB3の理解不足**: 内部構造を十分に調査せず
3. ❌ **段階的検証の欠如**: 統合前にE2Eテストすべきだった

### 次回への改善

1. 🔧 **統合テストを先に作る**: 「実際に呼ばれるか」を最初に確認
2. 🔧 **ライブラリ内部を先に調査**: フレームワークの制約を理解してから設計
3. 🔧 **最小実装で検証**: 1機能ずつ統合して動作確認

---

## 🚨 緊急判断が必要な事項

**質問1**: Option D（環境側の疑似統合）で一旦妥協しますか?
- YES → 30分で実装、今日中に10k再実行
- NO → Option B/Aの実装に着手（1日〜1週間）

**質問2**: 完全統合(Option A)に投資する価値はありますか?
- YES → 1週間かけて完全版を実装
- NO → Option Dで十分と判断し、次の課題へ

**質問3**: 外部レビューを求めますか?
- YES → SB3の詳しい人にアーキテクチャレビューを依頼
- NO → 自力で試行錯誤を継続

---

## 📎 関連ファイル

- 実装: `ztb/training/sell_mitigation_ppo_trainer.py`
- テスト: `tests/unit/training/test_*.py` (すべてパス)
- 結果: `checkpoints/smoke_test_10k/sell_probe.csv` (空)
- ログ: `logs/smoke_test_10k/` (未確認)

