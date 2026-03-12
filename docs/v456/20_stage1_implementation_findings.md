# Week 4 Stage 1 実装発見ログ

**作成日**: 2026-01-13 23:50  
**タイトル**: 改善版訓練スクリプト実行結果と環境診断

---

## 概要

段階1実装 (train_mlp_v456_improved.py) を実行した結果、新たな問題が発見されました。根本的には環境の報酬計算ロジック問題であることが確認されました。

---

## 実装状況

### ✓ 完了したもの

1. **train_mlp_v456_improved.py** 作成
   - drawdown_limit: 0.1 → 0.3 (改善)
   - initial_balance: 124 → 50,000 JPY (改善)
   - max_steps: None → 500 (改善)
   - ImprovedRewardCalculator 実装
   - ThousandStepStatisticsCallback 実装
   - 実特徴量計算 (SMA, ROC, Volatility, Trend)

2. **validate_week4_improved.py** 作成
   - 30エピソード評価フレームワーク
   - ベースライン比較機能実装

### ❌ 問題発見

実行テストで以下の問題が発生：

```
訓練開始... (10,000 ステップ)
2026-01-13 23:45:05,309 - __main__ - INFO - 訓練中断
```

**実行時間**: わずか 1.36秒 → 即座に終了

---

## 根本原因分析

### 診断実行結果

```python
環境診断: FastIntradayEnvV456
✓ データロード: 27,012 rows
✓ 環境作成成功: obs=Box(88,), action=Box(2,)
✓ リセット成功

単一ステップテスト:
  Step 1: reward=-0.250002, done=True, pos=-100.000
  ⚠️  エピソード終了 (ステップ 1)
```

### 重要な発見：既存スクリプトでも同じ症状

既存スクリプト (train_mlp_v456.py) と同じパラメータで再テスト：

```python
既存スクリプト (train_mlp_v456.py) のパラメータ:
  initial_balance: 124.01 JPY
  max_position: 1.0 (max(None, 1.0))
  drawdown_limit: 0.1 (デフォルト)
  max_steps: None

実行結果:
  Step 1: reward=-0.179990, balance=-3089.29, done=True
  エピソード終了 (ステップ 1)
```

**結論**: 改善版スクリプトも既存スクリプトも **同じ根本問題を持っている**

ただし、5000-step訓練では何らかの方法でこれを回避していた可能性がある。

### 原因の特定

```python
# FastIntradayEnvV456 の step() メソッド (Line 316-317)
if self.balance < self.initial_balance * (1 - self.drawdown_limit):
    done = True
```

**問題の流れ**:

1. **初期化**: balance = 50,000 JPY
2. **第1アクション**: ランダムなので空売り (-100%)
3. **報酬計算**: raw_reward = -0.25 (fee/slippage 含む)
4. **balance 更新**: 50,000 - (50,000 × 0.25) = 37,500 JPY
5. **drawdown チェック**: 37,500 < 50,000 × (1 - 0.3) = 35,000 ❌
6. **エピソード終了**: done = True

### 根因の根底

**環境の design 問題**:

```
balance の更新タイミング
  ↓
報酬 が balance に直接反映されている
  ↓
最初のランダムアクションが大きなマイナス報酬
  ↓
即座に drawdown_limit を超過
  ↓
エピソード起動と同時に終了
```

---

## 改善案の評価

### 案1: drawdown_limit をさらに大きくする ❌

**考案内容**:
```python
drawdown_limit = 0.5  # 50%
```

**評価**:
- 一時的にはステップ数が増える可能性
- しかし根本的解決にならない
- 環境の報酬計算ロジックの問題は残存

### 案2: 環境の報酬計算を修正 ⚠️

**必要な修正**:
1. balance への報酬反映を段階化
2. 初期数ステップの報酬を安定化
3. または報酬計算関数 compute_hft_reward() の改良

**評価**:
- 根本解決だが実装コスト高
- 環境自体の変更が必要

### 案3: 既存の成功パターンを活用 ✓ (推奨)

**考案内容**:
- 既存の train_mlp_v456.py (5000-step 版) がなぜ成功していたのかを分析
- その設定をベースに改善を加える
- 環境パラメータの互換性を確認

**評価**:
- 既知の成功パターン
- 段階的改善
- リスク最小化

---

## 次のアクション

### 推奨ステップ

1. **既存スクリプト分析**
   - scripts/v456/train_mlp_v456.py を確認
   - 成功した設定パラメータを抽出
   - 何が 1.2-step 問題を回避していたのか明確化

2. **ハイブリッドアプローチ**
   - 既存スクリプトの環境設定をベース
   - 改善版の報酬関数と特徴量は有効かつ保持
   - 段階的な testing

3. **環境診断の拡張**
   - データセットの品質確認
   - 初期価格が NaN/ゼロでないか確認
   - fee/slippage の計算ロジック確認

---

## 判定

| 項目 | 判定 |
|-----|------|
| 改善案の有効性 | △ (部分的) |
| 環境問題の特定 | ✓ (完了) |
| 根本原因の理解 | ✓ (完了) |
| 次の方向性 | 既存パターン + 段階的改善 |

**ステータス**: Stage 1 の改善案は環境の設計問題により即座に失敗
**決定**: 既存スクリプト分析に基づくハイブリッドアプローチへ pivot

---

## 技術メモ

### 環境の 5値返却について

Gym 0.26+ では step() が 5値を返します:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated`: エピソード終了 (drawdown/bankruptcy)
- `truncated`: タイムアウト (max_steps)

改善版で適切に処理されていることを確認済み。

### 診断スクリプト活用

作成した diagnose_env.py は環境の健全性チェックに有効:
- リセット確認
- 単一ステップの実行
- reward/done の状態確認
- 初期化の問題検出

---

## ドキュメント履歴

- 16_root_cause_analysis.md: 初期問題分析 (5000-step 検証)
- 17_fix_implementation_roadmap.md: 修正計画
- **20_stage1_implementation_findings.md**: Stage 1 実装時の発見 (本文書)

**次フェーズ**: 21_hybrid_approach.md (予定)
