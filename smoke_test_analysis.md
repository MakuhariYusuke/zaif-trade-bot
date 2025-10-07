# 10k Smoke Test - 結果分析レポート

## ⚠️ 重大な問題: プローブデータが記録されていない

### 🔴 問題の概要

**プローブCSVが空**: ヘッダーのみでデータ行が0件
- ファイルサイズ: 95バイト（ヘッダーのみ）
- 期待: 数千行のステップごとのデータ
- 実際: データ行0件

### 📊 トレーニングログからの観測

#### ✅ 正常に動作した部分

1. **トレーニング完了**: 10,240 timesteps (目標10,000達成)
2. **FPS**: 76 (正常)
3. **Iterations**: 5回
4. **実行時間**: 134秒 (約2分)

#### 🟡 統計情報（最終イテレーション）

```
entropy/
  current_alpha: 0.01          # Target Entropy: 初期値のまま
  target_entropy: 0.769        # ✓ 正しく設定されている

lagrange/
  lambda_dual: 0               # ⚠️ 全く動作していない
  r_sell_mean: 0               # ⚠️ SELL率0%

pan/
  action_counts: [0, 0, 0]     # ⚠️ 統計が記録されていない
  action_means: [0.0, 0.0, 0.0]
  action_stds: [0.0, 0.0, 0.0]
  total_samples: 0

probe/
  is_healthy: False            # ⚠️ 不健全状態
  grad_norm_mean: 0            # ⚠️ データなし
  advantage_mean: 0

stratified/
  bucket_r0_a0~r2_a2: ALL 0    # ⚠️ 全く動作していない

train/
  approx_kl: 0.000948          # 正常
  entropy_loss: -0.616         # 正常
  policy_gradient_loss: -0.000984
  value_loss: 709
```

#### 🔴 警告メッセージ

```
⚠️ SELL bias mitigation targets not fully achieved
   - SELL rate below 15%: 0.0%
```

### 🔍 根本原因分析

#### 問題1: forced_balanceモード

ログに `curriculum_stage: forced_balance` が出力されていました。
これは **強制バランスモード** で、環境が action を強制的にバランスさせている可能性があります。

**推測**:
- `forced_balance`モードでは、環境が内部的にアクションを調整している
- そのため、プローブやPAN/Stratified Samplerが正しくフックされていない
- 実際の学習ループと統計収集が切り離されている

#### 問題2: コールバック統合の不備

以下の統計がすべて0:
- PAN: `total_samples: 0`
- Stratified: すべてのバケットが0
- Probe: データなし

**推測**:
- コールバックは呼ばれているが、データが渡されていない
- または、データ収集のタイミングが学習ループとずれている
- SB3のコールバック機構との統合が不完全

#### 問題3: Lagrange制約が動作していない

`lambda_dual: 0`, `r_sell_mean: 0` は異常です。

**推測**:
- Lagrange制約の更新ロジックが呼ばれていない
- または、データソースが接続されていない

### 🐛 バグの可能性

1. **SELLBiasMitigationCallback の統合不備**
   - `_on_step()` は呼ばれているが、データが空
   - 統計取得メソッドが正しく呼ばれていない可能性

2. **PPOTrainer との互換性問題**
   - 親クラスの `train()` メソッドが、新しいコンポーネントを認識していない
   - カスタムコールバックが適切にフックされていない

3. **forced_balance モードとの競合**
   - 強制バランスモードが、カスタムロジックをバイパスしている
   - 環境レベルでアクションが上書きされている

### 📋 次のアクション候補

#### Option A: デバッグモードで再実行
```python
# 詳細ログを有効にして再実行
# ztb/training/sell_mitigation_ppo_trainer.py にログ追加
```

#### Option B: 統合テストの作成
```python
# コールバックが正しくデータを受け取るかテスト
# PAN/Entropy/Stratifiedが実際に呼ばれるかテスト
```

#### Option C: forced_balance を無効化
```json
// smoke_test_10k_config.json
"environment": {
  "curriculum_stage": "full"  // forced_balanceを避ける
}
```

#### Option D: 簡易版トレーナーで検証
```python
# SB3の標準PPOを直接使用
# カスタムコールバックなしでベースラインを確立
```

### 🎯 推奨アクション

1. **即座に実行**: コールバックのデバッグログを追加
2. **検証**: 統合テストで各コンポーネントが呼ばれることを確認
3. **再実行**: デバッグモードで10k再実行

### 📝 現時点での結論

**判定**: ❌ テスト失敗（データ収集不備により評価不可）

**原因**: 
- 4つの改修機能が実装されているが、トレーニングループに統合されていない
- コールバックは動作しているが、データが空
- 統計収集メカニズムが機能していない

**次の優先事項**: 統合デバッグ
