# 80# Day 10: 79# Codexレビュー対応 - 包括実験計画

**日付**: 2026-01-31  
**目的**: 78# 50k悪化の原因特定と高収益化への道筋

---

## 1. 79# Codex指摘の妥当性評価

### 指摘1: ROIの根拠が薄い（estimated_roi_pct = final_reward×100）

| 評価 | 判定 |
|------|------|
| 妥当性 | ✅ **完全に妥当** |

**現状**:
```python
estimated_roi = final_reward * 100  # ← 単純スケーリング
```

**問題点**:
- `final_reward` は累積報酬であり、ポートフォリオROIではない
- reward_scale=100 の影響で数値が歪む
- 実際のポートフォリオ履歴から計算していない

**対応**: 
- `trainer.get_training_stats()` から `final_balance` を取得
- ROI = (final_balance - initial_balance) / initial_balance * 100

---

### 指摘2: 更新強度過多による方策崩壊

| 評価 | 判定 |
|------|------|
| 妥当性 | ✅ **妥当** |

**現状（78# Day9b）**:
- lr=0.0005, batch=128, grad_steps=2, buffer=25k
- 50k steps で 100,000回の更新（steps × grad_steps）
- buffer=25k だと同じデータを4回以上再利用

**45# Day5**:
- lr=0.0003, batch=256, grad_steps=1, buffer=100k
- 50k steps で 50,000回の更新
- buffer=100k だとデータ再利用は少ない

**対応**: 45# Day5設定の再現実験

---

### 指摘3: 行動比率が同じなのに損失拡大（行動の質悪化）

| 評価 | 判定 |
|------|------|
| 妥当性 | ✅ **妥当** |

**観察**:
- 25k: HOLD=41.6%, BUY=29.4%, SELL=28.9%
- 50k: HOLD=37.4%, BUY=31.4%, SELL=31.2%
- 行動比率は微差だが、損失は -6% → -37%

**解釈**:
- 「いつ取引するか」の判断が劣化
- 取引タイミングの質が悪化

**対応**: per-trade PnLの可視化（将来課題）

---

## 2. 実験設計

### 実験構成（全16実験）

#### A. ベースライン再現（2実験）
| ID | 設定 | 目的 |
|----|------|------|
| A1 | 45# Day5 SAC_DEFAULT (50k) | ベースライン確立 |
| A2 | 78# Day9b 設定 (50k) | 悪化再現確認 |

#### B. gamma×ent_coef 2×2（4実験）
| ID | gamma | ent_coef | 目的 |
|----|-------|----------|------|
| B1 | 0.95 | 0.01 | 72# Day8相当 |
| B2 | 0.95 | auto | 交互作用検証 |
| B3 | 0.99 | 0.01 | 75# Day9相当 |
| B4 | 0.99 | auto | 45# Day5相当 |

#### C. batch×grad_steps 2×2（4実験）
| ID | batch | grad_steps | 目的 |
|----|-------|------------|------|
| C1 | 128 | 1 | 更新頻度低 |
| C2 | 128 | 2 | 78# Day9b相当 |
| C3 | 256 | 1 | 45# Day5相当 |
| C4 | 256 | 2 | 更新頻度高 |

#### D. 報酬構造比較（4実験）
| ID | 報酬設定 | 目的 |
|----|----------|------|
| D1 | simple (scale=100) | 現行設定 |
| D2 | simple (scale=1) | スケール影響 |
| D3 | stage2_extended | リスク調整報酬 |
| D4 | デフォルト（指定なし） | 45# Day5相当 |

### 固定パラメータ（B, C, D実験共通）
- steps: 50,000
- seeds: [42]（時間短縮のため1 seed）
- データ: btc_jpy_1m_v451_optimized_features.parquet

---

## 3. 時間見積もり

| カテゴリ | 実験数 | 時間/実験 | 合計 |
|----------|--------|----------|------|
| A (ベースライン) | 2 | 60min | 2h |
| B (gamma×ent_coef) | 4 | 60min | 4h |
| C (batch×grad_steps) | 4 | 60min | 4h |
| D (報酬構造) | 4 | 60min | 4h |
| **合計** | **14** | - | **約14時間** |

---

## 4. 成功基準

### 最低基準
- A1 (45# Day5再現) で ROI > -10%
- 設定間で統計的に有意な差が観察できる

### 目標基準
- いずれかの設定で ROI > -5%
- 高収益化への道筋が見える

### 理想基準
- ROI > 0% の設定を発見
- 再現性のある改善パターンを特定

---

## 5. 実行方法

### スクリプト

**ファイル**: `scripts/v459/run_day10_comprehensive.py`

**実行コマンド**:
```powershell
# 無人実行（外出時向け）
cd c:\Users\Admin\dev\zaif-trade-bot
python scripts/v459/run_day10_comprehensive.py
```

**機能**:
- 24実験を優先順に自動実行（A→B→C→D）
- 各実験後に中間結果を `results/phase4_day10_comprehensive/day10_comprehensive_interim.json` に保存
- 環境から `final_balance` を取得して正確なROI計算
- カテゴリ別の集計と解釈を自動生成

**出力**:
- `day10_comprehensive_interim.json` - 中間結果（随時更新）
- `day10_comprehensive_analysis_YYYYMMDD_HHMMSS.json` - 最終分析
- `day10_comprehensive_results_YYYYMMDD_HHMMSS.json` - 全結果

---

## 6. 実験後の分析項目

1. **ROI正確算出**: final_balance から計算
2. **設定間比較**: 統計的有意差の確認
3. **交互作用**: gamma×ent_coef, batch×grad_stepsの相互効果
4. **報酬構造効果**: simple vs stage2 の比較
5. **行動分布**: HOLD/BUY/SELL比率の設定依存性

---

**作成日**: 2026-01-31  
**作成者**: GitHub Copilot  
**状態**: ✅ スクリプト作成完了

**関連ドキュメント**:
- 78# Day9b 50k検証結果
- 79# Codexレビュー
- 81# レビュー対応
