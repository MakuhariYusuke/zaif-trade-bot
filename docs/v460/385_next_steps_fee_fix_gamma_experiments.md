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

### 2.3 OOS 全走査の所要時間 (実測値更新)

- 384# で `max_steps_per_episode=None` (全走査) に変更
- val データ: 1,216,930 × 0.2 = **243,386 steps**
- `inspect.signature` は 379# でキャッシュ済み (per-step ns オーダー)
- **実測**: 243K steps × ~16ms/step ≈ **65分/seed** (初期推定30分の2倍)
- 訓練20分 + OOS65分 = **~85分/seed**, 4 seed 合計: ~340分 (5.7時間)

### 2.4 reward_scaling デッドコード (HIGH — 386# で対応)

**発見**: `_calculate_default_reward()` の関数シグネチャに `reward_scaling` パラメータが
含まれておらず、`inspect.signature()` フィルタにより除外される。

```
calculate_reward() 内:
  scale_adjustment = 1.0 / max(0.01, max_position_size=0.01) = 100.0
  reward_scaling = config.reward_scaling(=6.0) × 100.0 = 600.0
  → method_args["reward_scaling"] = 600.0
  → _calculate_default_reward に渡されない（シグネチャに無い）
  → pnl_reward = pnl × 1.0 × 1.0 = 生 JPY PnL がそのまま報酬
```

**影響**:
- 報酬スケールは O(1〜50) JPY/step — SAC には大きい (理想は O(1) 以下)
- `ent_coef="auto"` がある程度自動調整するため、致命的ではないが最適ではない
- `config.reward_scaling` を YAML で変更しても **一切効果がない**
- `asymmetric_reward_scaler` も全乗数=1.0 で実質ノーオプ
- クリップ [-80, 80] は典型的な報酬 (1-50) に対して実質無効

**386# 対応案**:
1. `_calculate_default_reward` に `reward_scaling` パラメータを追加
2. または `portfolio_return` (既に計算・渡し済み) を使用してPnLを正規化
3. SAC 用の適切なスケール値を YAML に明示設定

## 3. 実験計画

### 3.1 実験 A: 385# Baseline (実行中)

| パラメータ | 値 | 変更点 |
|-----------|-----|--------|
| transaction_cost | **0.0** | 旧 0.001 → 0.0 |
| gamma | 0.80 | 変更なし |
| total_timesteps | 50K | 変更なし |
| パイプライン | 384# fixes | CRITICAL-1/2 + HIGH-1/2 |

**期待**: ROI が +2~3% 改善し、G2 E1 (positive_seed_ratio ≥ 0.75) が PASS する可能性。

### 3.1.1 Seed42 チェックポイント結果 (in-sample, 5K step eval)

| Steps | 385# ROI | Pre-385# ROI | Delta |
|-------|----------|--------------|-------|
| 5K | **+0.16%** | -0.28% | +0.44% |
| 10K | **+0.11%** | -0.20% | +0.31% |
| 15K | **+0.13%** | -0.13% | +0.26% |
| 20K | **+0.08%** | -0.08% | +0.16% |
| 25K | **+0.17%** | -0.12% | +0.29% |
| 30K | **+0.11%** | -0.23% | +0.34% |
| 35K | **+0.23%** | -0.18% | +0.41% |
| 40K | **+0.10%** | -0.18% | +0.28% |
| 45K | **+0.21%** | -0.18% | +0.39% |
| 50K | **+0.13%** | -0.19% | +0.32% |

**全10チェックポイントで正ROI** ✅ — fee fix の効果は明確。
Pre-385# baseline (全て負) との差は一貫して +0.16% 〜 +0.44%。

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
| B4 | gamma 比較実験 | 🔄 YAML 準備完了、model_dir分離済み | A 結果後に実行 |
| B5 | transaction_cost 検証 | ✅ 修正済み | A で実測確認中 |
| B6 | curriculum_learning 設計 | ⬜ | A/B 結果後に検討 |
| B7 | **reward_scaling デッドコード修正** | ⬜ HIGH | 386# で対応 |
| B8 | **報酬正規化 (portfolio_return 活用)** | ⬜ HIGH | 386# で対応 |
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
