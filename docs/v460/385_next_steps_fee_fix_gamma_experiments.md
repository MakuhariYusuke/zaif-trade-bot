# 385# 次ステップ計画: 手数料修正 + Gamma 比較実験

## 1. 概要

384# で修正したパイプライン (CRITICAL-1/2, HIGH-1/2) の上に、
更なる改善として **手数料設定の矛盾修正** と **gamma 比較実験** を実施する。

## 2. 発見事項

### 2.1 手数料矛盾 (CRITICAL)

| ソース | Coincheck 手数料 | 備考 |
|--------|-----------------|------|
| **000# proposal** | **0%** (maker) | "全取引はmaker注文（手数料0%）で執行" |
| g2_sac_train.yaml (旧) | 0.1% | `transaction_cost: 0.001 # Coincheck Maker` |
| ExchangeFeeModel | 0% | `coincheck: {"buy": 0.0, "sell": 0.0}` |
| TradeExecutionEngine | maker=0%, taker=0.08% | 正しい二重構造 |
| DEFAULT_FEE_RATE | 0.1% | constants.py のデフォルト |

**矛盾の原因**: YAML 作成時に `DEFAULT_FEE_RATE` (0.1%) を参照し、
000# の maker 0% 前提を反映し忘れた。

**インパクト**: 
- 片道手数料: `0.01 BTC × 15M JPY × 0.001 = 150 JPY/trade`
- 往復: 300 JPY × ~1000 trades (seed42) = **300,000 JPY** 
- ポートフォリオ 10M JPY に対して **-3% ROI ドラグ**
- 前回の OOS ROI=-0.25% は、手数料なしなら **+2.75%** の可能性

### 2.2 報酬関数での手数料影響パス

```
position_manager.open_position()
  → entry_cost = trade_value × transaction_cost   ← ここで差し引き
  → realized_pnl -= entry_cost

position_manager.close_position()
  → exit_cost = trade_value × transaction_cost     ← ここで差し引き
  → trade_pnl -= exit_cost

core.py
  → step_pnl = trade_pnl + (unrealized_pnl - prev_unrealized)
  → reward = reward_calculator(pnl=step_pnl)      ← fee差引済PnLが報酬に
                                                     reward_calculator内での
                                                     二重差引はコメントアウト済
```

**結論**: `transaction_cost` は `position_manager` で PnL から直接控除される。
reward_calculator では二重課金されない設計。transaction_cost=0.0 にすれば
fee ペナルティは完全消滅する。

### 2.3 OOS 全走査の所要時間

- 384# で `max_steps_per_episode=None` (全走査) に変更
- val データ: 1,216,930 × 0.2 = **243,386 steps**
- `inspect.signature` は 379# でキャッシュ済み (per-step ns オーダー)
- 推定: 243K steps × ~7.5ms/step ≈ **30分/seed**
- 4 seed 合計: ~120分 (前回の訓練時間とほぼ同等)

## 3. 実験計画

### 3.1 実験 A: 385# Baseline (実行中)

| パラメータ | 値 | 変更点 |
|-----------|-----|--------|
| transaction_cost | **0.0** | 旧 0.001 → 0.0 |
| gamma | 0.80 | 変更なし |
| total_timesteps | 50K | 変更なし |
| パイプライン | 384# fixes | CRITICAL-1/2 + HIGH-1/2 |

**期待**: ROI が +2~3% 改善し、G2 E1 (positive_seed_ratio ≥ 0.75) が PASS する可能性。

### 3.2 実験 B: gamma=0.95

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| gamma | 0.95 | step+10 で 40% discount (0.80 では 89%) |
| total_timesteps | 100K | gamma 大 → 収束遅 → 2× steps |
| transaction_cost | 0.0 | 385# fix |

**YAML**: `configs/v460/experiments/g2_sac_gamma095.yaml`

### 3.3 実験 C: gamma=0.99

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| gamma | 0.99 | RL 標準値。step+100 で 37% 残存 |
| total_timesteps | 100K | gamma 大 → 2× steps |
| transaction_cost | 0.0 | 385# fix |

**YAML**: `configs/v460/experiments/g2_sac_gamma099.yaml`

### 3.4 実験優先度

1. **実験 A** (baseline): 手数料修正のインパクト測定 — **最重要**
2. **実験 B** (gamma=0.95): A の結果を見てから実行
3. **実験 C** (gamma=0.99): B の結果を見てから実行

## 4. gamma 選択の理論的考察

| gamma | step+10 残存 | step+100 残存 | 適用場面 |
|-------|-------------|---------------|---------|
| 0.80 | 10.7% | 0.0% | 超短期 (1-2 step) スキャルピング |
| 0.95 | 59.9% | 0.6% | 中期 (10-30 step) トレンドフォロー |
| 0.99 | 90.4% | 36.6% | 長期 (100+ step) ポジション保持 |

**1分足 × スキャルピング**: gamma=0.80 は「次の1-2分のPnLのみ最適化」。
10分後の結果を全く考慮しない。これはトレンドが続く場合にポジションを
早期にクローズしてしまうリスクがある。

**推奨**: gamma=0.95 を第一候補。理由:
- 10 step (10分) 後でも 60% の報酬を考慮 → トレンド継続に対応
- 100 step (100分) 以降は無視 → 日中ノイズに引きずられない
- 1分足スキャルピングの典型的なホールド時間 (5-30分) と整合

## 5. 今後の B シリーズ課題

| ID | 課題 | 状態 | 次アクション |
|----|------|------|------------|
| B4 | gamma 比較実験 | 🔄 YAML 準備完了 | A 結果後に実行 |
| B5 | transaction_cost 検証 | ✅ 修正済み | A で実測確認 |
| B6 | curriculum_learning 設計 | ⬜ | A/B 結果後に検討 |
| B1-B3 | conftest/_sb3_test_stub cleanup | ⬜ | 低優先 |

## 6. 既存実装の活用ポイント

### 6.1 参照すべき vXXX シリーズ

| 文書 | 内容 | 活用場面 |
|------|------|---------|
| 365# §3.5 | CurriculumManager + replay buffer flush | B6 curriculum_learning |
| 365# §4 | Warm-start incremental training | 実験 B/C で使用可能 |
| 363# A3 | Time-series train/val split | 既に実装済み |
| 363# A4 | IC→ROI seed std 変更 | G2 gate E2 で使用中 |
| 361# F1 | In-sample 評価の過学習問題 | 384# で解決済み |
| 372# audit | evaluate_model_oos 集約 | 384# で改善済み |

### 6.2 活用中の既存実装

- `ExchangeFeeModel`: coincheck 手数料 0% が既に定義済みだった (fee_model.py)
- `EnvironmentConfig.scaler_mean/std`: train→val scaler 転送 (384# で活用)
- `BalanceCurriculumManager`: balance_curriculum.py (506行) — B6 で活用候補
- `inspect.signature` キャッシュ: reward_calculator.py 379# で実装済み

## 7. 成功基準

### 実験 A (385# Baseline)
- **G2 E1 PASS**: positive_seed_ratio ≥ 0.75 (3/4 seeds が gross ROI > 0)
- **G2 E2**: roi_seed_std ≤ 0.03
- **G2 E4**: worst_seed_roi > -0.02

### 全体目標
- G2 gate PASS → ph4 (microstructure features) へ進行
- transaction_cost=0.0 で ROI 正転 → maker-only 戦略の妥当性を実証
