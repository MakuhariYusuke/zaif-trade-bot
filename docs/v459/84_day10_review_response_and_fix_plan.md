# 84# Day 10 Codexレビュー対応 - 評価基盤修正と再実験計画

**日付**: 2026-02-01  
**対象**: 83# Codexレビューの「追加で見落とされがちな観点」への対応  
**関連**: 45#, 78#, 79#, 82#, 83#

---

## 1. 83# 「追加で見落とされがちな観点」の妥当性評価

### 1.1 reward_scale / reward_scaling / reward_clip_min/max の実効値ログ

| 評価 | **✅ 妥当** |
|------|------------|

**現状調査結果**:
- `REWARD_SIMPLE`で`reward_scale=100.0`を指定
- `fast_intraday_env_v456.py`で`reward_scale`属性あり
- しかし、実効値がログに出力されていない

**問題点**:
- 設定値と実動作の乖離を検証不可能
- 過去にも設定と実動作のズレが報告あり

**対応**: 環境初期化時にreward_scale/clip値をログ出力する機能追加

---

### 1.2 walk-forward無効化による評価の歪み

| 評価 | **✅ 妥当** |
|------|------------|

**現状調査結果**:
- 45# Day5: `walk_forward.enabled=True`, `n_splits=4`
- Day10: `walk_forward.enabled=False`

**影響**:
- walk-forward有効時は Train/Val/Test分割で過学習を検知
- walk-forward無効時は全データで学習 → 過適合リスク増大

**45# Day5との差分原因としての可能性**: **高**
- 45# Day5の-5%ROIはwalk-forward評価後の結果
- Day10の-36%ROIはwalk-forward無しの結果
- 同一データでの過学習により50kで悪化した可能性

**対応**: walk-forward有効での再実験

---

### 1.3 reward構成要素の相殺

| 評価 | **✅ 妥当** |
|------|------------|

**現状調査結果 (D2_stage2)**:
- `use_simple_reward=False`
- `sharpe_bonus_scale=0.03`, `sortino_bonus_scale=0.025`
- `trade_frequency_penalty=0.002`
- `asymmetric_reward_scaling`: `positive_mult=1.0`, `negative_mult=1.5`

**ROI≈0%の原因仮説**:
1. **相殺効果**: profit報酬 + Sharpe bonus - frequency penalty - 非対称負scaling ≈ 0
2. **探索崩壊**: 複雑な報酬で学習が困難、HOLD傾向に収束
3. **reward計算バグ**: 報酬が誤ってゼロに近い値を返している

**検証方法**: reward_components履歴の詳細分解

---

### 1.4 行動の質の低下（1トレード当たり損失増大）

| 評価 | **✅ 妥当** |
|------|------------|

**現状データ**:
- 行動分布: HOLD≈40%, BUY≈30%, SELL≈30% (ほぼ均一)
- ROI: -5%〜-36% (設定により大幅変動)

**仮説**:
- 行動比率が同じでも、**取引タイミングが悪化**している可能性
- 50kステップで「誤った確信」を学習 → 損失が拡大

**検証方法**: 1トレード当たりPnLの時系列分析

---

## 2. 追加の見落とし確認

### 2.1 ROI計算の根本的バグ（最優先）

**調査結果**:

```python
# run_day10_comprehensive.py (line 375-395)
# 問題: trainer.model へのアクセスが失敗

if hasattr(trainer, 'model') and hasattr(trainer.model, 'env'):
    env = trainer.model.env  # ← UnifiedTrainerにはmodelがない
```

**実際の構造**:
```python
# UnifiedTrainer
trainer.algorithm_trainer.model.env  # ← 正しいパス
```

**さらに、環境の属性名が異なる**:
```python
# fast_intraday_env_v456.py
unwrapped_env.balance  # 実際の属性名
unwrapped_env.portfolio_value  # ← この属性は存在しない
```

**修正必要箇所**:
1. `trainer.algorithm_trainer.model.env`へのアクセス
2. `unwrapped_env.balance`の使用

---

### 2.2 collect_training_statsにfinal_balanceが含まれていない

**調査結果**:
```python
# base_trainer.py (line 401-425)
def collect_training_stats(self, ...):
    stats = {
        "total_timesteps": total_timesteps,
        "training_time": training_time,
        "model_path": model_path,
        "status": "completed",
    }
    # final_balance, ROIは含まれていない
```

**対応**: 
- `collect_training_stats`にfinal_balance取得ロジックを追加
- または呼び出し側で環境から取得

---

### 2.3 45# run_ab_feature_test.pyとの差分

| 項目 | 45# run_ab_feature_test.py | Day10 run_day10_comprehensive.py |
|------|---------------------------|----------------------------------|
| トレーナー | `SACTrainer`直接使用 | `UnifiedTrainer`経由 |
| 環境アクセス | `trainer.model.env` | `trainer.model.env`（失敗） |
| walk-forward | 有効 | **無効** |
| メトリクス取得 | 成功 | **失敗** |

**結論**: 45# run_ab_feature_test.pyのロジックを流用すべき

---

## 3. 修正計画

### Phase 1: 評価基盤の修正（最優先）

#### 3.1 run_day10_comprehensive.pyの環境アクセス修正

```python
# 修正前
if hasattr(trainer, 'model') and hasattr(trainer.model, 'env'):
    env = trainer.model.env

# 修正後
if hasattr(trainer, 'algorithm_trainer'):
    alg_trainer = trainer.algorithm_trainer
    if hasattr(alg_trainer, 'model') and hasattr(alg_trainer.model, 'env'):
        env = alg_trainer.model.env
```

#### 3.2 環境属性名の修正

```python
# 修正前
if hasattr(unwrapped_env, 'portfolio_value'):
    final_balance = float(unwrapped_env.portfolio_value)

# 修正後
if hasattr(unwrapped_env, 'balance'):
    final_balance = float(unwrapped_env.balance)
elif hasattr(unwrapped_env, 'portfolio_value'):
    final_balance = float(unwrapped_env.portfolio_value)
```

#### 3.3 初期残高の取得修正

```python
# 修正後
if hasattr(unwrapped_env, 'initial_balance'):
    initial_balance = float(unwrapped_env.initial_balance)
elif hasattr(unwrapped_env, 'initial_portfolio_value'):
    initial_balance = float(unwrapped_env.initial_portfolio_value)
```

---

### Phase 2: 再実験（Day 11）

#### 3.4 45# run_ab_feature_test.py流用での50k実験

- 既存スクリプトそのままで実行
- SAC_DEFAULT設定、seed=42, 123
- walk-forward有効
- 目的: ROI=-5%の再現確認

#### 3.5 修正版run_day10_comprehensive.pyでの再実験

- Phase 1の修正を適用
- walk-forward有効化オプション追加
- Aカテゴリ（ベースライン）のみ再実験

---

### Phase 3: 詳細分析（Day 12以降）

#### 3.6 reward_scale実効値ログ追加

```python
# 環境初期化後にログ出力
logger.info(f"Effective reward_scale: {env.reward_scale}")
logger.info(f"Effective reward_clip: [{env.reward_clip_min}, {env.reward_clip_max}]")
```

#### 3.7 stage2報酬の構成要素分解

- reward_components履歴を詳細分析
- profit/sharpe/sortino/frequency_penaltyの寄与度計算

#### 3.8 1トレード当たりPnL分析

- 取引履歴からステップごとのPnLを抽出
- 25k vs 50kでの1取引当たり損失比較

---

## 4. 実装タスク一覧

### 即座対応（Day 11）

| ID | タスク | 優先度 | 見積時間 |
|----|--------|--------|----------|
| T1 | run_day10_comprehensive.py 環境アクセス修正 | **最高** | 30分 |
| T2 | 45# run_ab_feature_test.py で50k再実験 | **最高** | 3時間 |
| T3 | 修正版での再実験（Aカテゴリのみ） | 高 | 2時間 |

### 翌日以降（Day 12〜）

| ID | タスク | 優先度 | 見積時間 |
|----|--------|--------|----------|
| T4 | reward_scale実効値ログ追加 | 中 | 1時間 |
| T5 | stage2報酬構成要素分解 | 中 | 2時間 |
| T6 | 1トレード当たりPnL分析 | 中 | 2時間 |
| T7 | walk-forward有効化オプション追加 | 中 | 1時間 |

---

## 5. 期待される成果

### 5.1 評価基盤修正後

- **正確なROI計算**: final_balanceベースでの資産ROI
- **45# Day5との比較可能性**: 同一評価基準での結論

### 5.2 45# Day5再現成功の場合

- 「同一設定で31%乖離」問題は**評価方式の差**で説明可能
- Day10のROI値は信頼性なし → 再実験で正確な値を取得

### 5.3 45# Day5再現失敗の場合

- 別の原因追求が必要（環境変化、ライブラリバージョン等）
- より詳細なデバッグが必要

---

## 6. 83# Codexレビューへの総合回答

### レビュー指摘の妥当性

| 指摘 | 妥当性 | 対応 |
|------|--------|------|
| ROI算出の不整合 | **✅ 正確** | Phase 1で修正 |
| 設定交絡の多さ | ✅ 妥当 | 段階的ablation継続 |
| 更新強度過多と探索減衰 | ✅ 妥当 | Cカテゴリで検証済み |
| reward_scale実効値ログ | ✅ 妥当 | Phase 3で追加 |
| walk-forward無効化の影響 | **✅ 重要** | 再実験で有効化 |
| reward構成要素の相殺 | ✅ 妥当 | Phase 3で分析 |
| 行動の質の低下 | ✅ 妥当 | Phase 3で分析 |

### 追加発見

| 発見 | 深刻度 | 対応 |
|------|--------|------|
| trainer.model パスの誤り | **最高** | Phase 1で修正 |
| balance vs portfolio_value | **最高** | Phase 1で修正 |
| collect_training_stats不完全 | 高 | Phase 1で対応 |

---

## 7. 次のアクション

### 即座（今日中）

1. **T1実行**: run_day10_comprehensive.py の環境アクセス修正
2. **T2準備**: 45# run_ab_feature_test.py の実行準備

### 明日（Day 11）

3. **T2実行**: 45# run_ab_feature_test.py で50k再実験
4. **T3実行**: 修正版での再実験

### 来週（Day 12〜）

5. **T4-T7**: 詳細分析タスク

---

**作成日**: 2026-02-01  
**作成者**: GitHub Copilot  
**状態**: 修正実装中
