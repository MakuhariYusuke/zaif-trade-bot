# 102# Phase C 実験ログ — C0+C1 結果と根本原因分析

**Date**: 2026-02-09
**前提**: 100# Phase 4.5 完了報告、101# Codexレビュー反映済み
**目的**: Phase C0+C1 全14実験の詳細ログ、根本原因の深層分析、次ステップへの接続

---

## 0. エグゼクティブサマリ

**14実験、349分、全NO-GO。** Phase C0+C1（γ感度・閾値・保有期間・v451復元）のいずれもSACの根本的な学習失敗を解消できなかった。最も重大な発見は「**学習後のdeterministic policyが50Kステップ全区間で1回も取引しない**」ことであり、学習中に観測された~1,000回の取引は全てSACのエントロピー駆動の探索ノイズによる偶発的行動だった。

---

## 1. 実験設計

### 1.1 共通条件

| 項目 | 値 |
|------|-----|
| アルゴリズム | SAC (Soft Actor-Critic) |
| 環境 | HeavyTradingEnv |
| データ | `btc_jpy_1m_v451_optimized_features.parquet` (1.2M行, 8特徴) |
| 特徴量 | RSI×7 + ReturnStdDev (`feature_set="v451"`) |
| 学習ステップ | 50,000 |
| 初期資金 | 100,000 JPY |
| 手数料 | 0.1% (Zaif maker rate) |
| 評価方法 | 学習後に`deterministic=True`で50Kステップ1エピソード評価 |
| seed | 42（スクリーニング） |
| メモリリーク対策 | subprocess分離（各実験を独立プロセスで実行） |

### 1.2 実験一覧

| # | 実験名 | 操作変数 | 仮説根拠 |
|---|--------|---------|---------|
| 1 | c0_baseline_p1 | — (P1-1再現) | Gate2 KPI計測基盤の検証 |
| 2 | c1_gamma_080 | γ=0.80 | 91# H1: v451 Golden Era |
| 3 | c1_gamma_090 | γ=0.90 | 91# H1: 中間値 |
| 4 | c1_gamma_095 | γ=0.95 | 91# H1: 中間値 |
| 5 | c1_threshold_50 | threshold=0.50 | 100# §11 H2: 過剰取引抑制 |
| 6 | c1_threshold_60 | threshold=0.60 | 同上 |
| 7 | c1_threshold_70 | threshold=0.70 | 同上 |
| 8 | c1_g080_thr50 | γ=0.80 + threshold=0.50 | H1×H2組合せ |
| 9 | c1_g080_thr60 | γ=0.80 + threshold=0.60 | 同上 |
| 10 | c1_g080_thr70 | γ=0.80 + threshold=0.70 | 同上 |
| 11 | c1_holding_5 | min_holding_period=5 | 101#修正: 現行=3 |
| 12 | c1_holding_10 | min_holding_period=10 | 同上 |
| 13 | c1_holding_15 | min_holding_period=15 | 同上 |
| 14 | c1_v451_golden | γ=0.80 + V457RewardCalc + scale=1.0 | 91# v451完全復元 |

### 1.3 スクリプト

| ファイル | 役割 |
|---------|------|
| `scripts/v459/run_phase_c.py` | 単一実験実行 + Gate2 KPI計算 + deterministic eval |
| `scripts/v459/run_phase_c_batch.py` | subprocess分離バッチ実行 + 中間保存 + 再開機能 |
| `tests/unit/scripts/test_run_phase_c.py` | 20テスト全PASS |

---

## 2. 全実験結果

### 2.1 学習時メトリクス（探索行動を含む）

| # | Experiment | γ | Thr | MHP | Net ROI% | Trades | Fees | Gross PnL | BUY:SELL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | c0_baseline_p1 | 0.99 | 0.33 | 3 | -15.02 | 1,036 | 16,127 | **+1,111** | 518:518 |
| 2 | c1_gamma_080 | 0.80 | 0.33 | 3 | -15.02 | 870 | 13,856 | -1,162 | 435:435 |
| 3 | c1_gamma_090 | 0.90 | 0.33 | 3 | **-14.96** | 946 | 15,224 | +261 | 473:473 |
| 4 | c1_gamma_095 | 0.95 | 0.33 | 3 | -15.01 | 962 | 15,287 | +276 | 481:481 |
| 5 | c1_threshold_50 | 0.99 | 0.50 | 3 | -15.02 | 862 | 14,098 | -925 | 431:431 |
| 6 | c1_threshold_60 | 0.99 | 0.60 | 3 | -15.00 | 780 | 12,745 | -2,259 | 390:390 |
| 7 | c1_threshold_70 | 0.99 | 0.70 | 3 | -15.01 | **742** | **11,837** | -3,176 | 371:371 |
| 8 | c1_g080_thr50 | 0.80 | 0.50 | 3 | **-15.27** | 808 | 13,157 | -2,109 | 404:404 |
| 9 | c1_g080_thr60 | 0.80 | 0.60 | 3 | -15.01 | 822 | 13,301 | -1,713 | 411:411 |
| 10 | c1_g080_thr70 | 0.80 | 0.70 | 3 | -15.00 | 860 | 13,751 | -1,254 | 430:430 |
| 11 | c1_holding_5 | 0.99 | 0.33 | 5 | -15.04 | **1,080** | 17,013 | **+1,977** | 540:540 |
| 12 | c1_holding_10 | 0.99 | 0.33 | 10 | -15.01 | 950 | 15,267 | +256 | 475:475 |
| 13 | c1_holding_15 | 0.99 | 0.33 | 15 | -14.97 | 888 | 14,094 | -875 | 444:444 |
| 14 | c1_v451_golden | 0.80 | 0.33 | 3 | -15.03 | 880 | 13,905 | -1,124 | 440:440 |

### 2.2 Gate 2 Deterministic評価（全実験共通）

| 指標 | 0番 Gate2基準 | 14実験の結果 |
|------|-------------|-------------|
| Eval Trades | — | **0**（全実験） |
| MTM ROI | > 5% | **0.0%** |
| Profit Factor | > 1.20 | **1.0**（取引なしのため未定義） |
| Sharpe | > 1.0 | **0.0** |
| Max Drawdown | < 15% | **0.0%**（balance変動なし） |
| Win Rate | > 35% | **0.0%** |
| **Gate2 判定** | | **全14実験 FAIL** |

### 2.3 実行時間

| 統計 | 値 |
|------|-----|
| 総実行時間 | 349.5分（5時間50分） |
| 平均/実験 | 25.0分 |
| 最短 | 22.5分 (c1_g080_thr50) |
| 最長 | 25.9分 (c1_holding_10) |
| メモリ/プロセス | ~910MB (peak)、subprocess終了時に完全解放 |

---

## 3. 観測事実の整理

### 3.1 BUY:SELL 完全対称

14実験すべてで `BUY:SELL = N:N`（完全1:1）。SAC出力が方向に偏りを持たず、actionの符号が対称的に分布していることを意味する。

### 3.2 Net ROIの不変性

14実験のNet ROI範囲: **-14.96% ~ -15.27%**（幅わずか0.31ポイント）。

この値は次の式で近似できる:

$$\text{Net ROI} \approx -\frac{\text{Trades} \times \text{avg\_position\_size} \times \text{tx\_cost}}{100{,}000}$$

実際 `Trades × avg_fee ≈ Trades × 15.6 ≈ 15,000 ≈ 15%`。つまりNet ROI ≈ -15%は**ランダム取引時の手数料理論値**であり、パラメータによらず安定する。

### 3.3 γとGross PnLの関係

| γ | Gross PnL | Trades | 解釈 |
|---|-----------|--------|------|
| 0.99 | **+1,111** | 1,036 | 長期視野→探索が広範→多少の方向偏りが利益に |
| 0.95 | +276 | 962 | |
| 0.90 | +261 | 946 | |
| 0.80 | **-1,162** | 870 | 短期視野→報酬が小さく見える→探索が保守化→方向偏りなし |

γ=0.80で取引が-16%減少する一方、Gross PnLは**負に転落**。91# H1仮説の「γ=0.80がv451成功の鍵」は、HeavyTradingEnv + 8特徴量の条件下では**棄却**。

ただし注意: v451成功時はFastIntradayEnv + 88特徴量 + 100K+ステップだった可能性が高い。γ単体の問題ではなく、環境全体の差異が支配的。

### 3.4 Threshold効果の限界

| Threshold | Trades | Trade削減率 | Fees削減 | Gross PnL変化 |
|-----------|--------|-------------|---------|---------------|
| 0.33 (base) | 1,036 | — | — | +1,111 |
| 0.50 | 862 | -17% | -2,029 | -925 (-2,036) |
| 0.60 | 780 | -25% | -3,382 | -2,259 (-3,370) |
| 0.70 | 742 | -28% | -4,290 | -3,176 (-4,287) |

Fees削減とGross PnL悪化がほぼ**等量**で相殺。threshold↑は「ランダムactionのうち弱いものを除外」するが、残るactionも方向性がないため、取引単価の損失が増加。

### 3.5 min_holding_period逆効果

| MHP | Trades | 変化 | 解釈 |
|-----|--------|------|------|
| 3 (default) | 1,036 | — | |
| 5 | **1,080** | **+4%増** | 保有強制→解放直後の反動取引が増発 |
| 10 | 950 | -8% | 物理的に取引頻度を制限 |
| 15 | 888 | -14% | 同上 |

MHP=5での取引増加は、SACが「強制保有が解除された直後に取引する」行動を探索で発見したことを示唆。hold→release→immediate tradeのパターン。

---

## 4. 根本原因の深層分析

### 4.1 なぜDeterministic Policyが取引しないのか

SACの学習目標は:

$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

エントロピー項 $\alpha \mathcal{H}(\pi)$ が学習初期に支配的で、報酬 $r_t$ が相対的に小さい場合、最適方策は「高エントロピー（全action均等確率）」に収束する。deterministic policyはこのエントロピー分布の**平均値**（≈ 0.0, 閾値未満）を出力するため、取引しない。

**検証**: 全14実験でBUY:SELL=N:Nの完全対称は、action分布がゼロ中心対称であることの直接的証拠。

### 4.2 報酬信号の強度問題

`use_simple_reward=True` での報酬:

```
reward = (現在portfolio_value - 前ステップportfolio_value) / initial_balance × reward_scale
```

- 1ステップの典型的なprice変動: ±0.01%（1分足BTC/JPY）
- 1取引のGross PnL: ±100,000 × 0.01% = ±10 JPY
- 手数料: 100,000 × 0.1% = 100 JPY
- **報酬: (10 - 100) / 100,000 × 100 = -0.09**

→ 取引すると即座に-0.09のペナルティ。HOLDすれば0.0。SACにとって**HOLD=最適解は合理的**。

### 4.3 v451 Golden Era設定が効かない理由

c1_v451_golden（γ=0.80 + V457RewardCalculator + scale=1.0）が機能しない原因:
- `reward_scale=1.0` → 報酬がさらに100分の1に縮小（base=100.0）
- V457RewardCalculatorの`loss_multiplier=1.2`は報酬の非対称性を加えるが、報酬自体が微小ではその差も微小
- v451成功時は**異なる環境**（おそらくFastIntradayEnv）であり、報酬計算パスが根本的に異なる

### 4.4 構造的問題の整理

```
根本原因チェーン:
  1分足BTC/JPY変動 ~0.01%/step
    ↓
  1取引のEdge ≈ 0 (方向予測なし)
    ↓
  取引コスト 0.1% >> Edge
    ↓
  reward(HOLD) = 0.0 > reward(TRADE) = -0.09
    ↓
  SAC最適方策 = HOLD
    ↓
  Deterministic Trades = 0
```

---

## 5. 仮説の棄却・保留・維持

| 仮説 | 出典 | 状態 | 根拠 |
|------|------|------|------|
| H1: γ=0.80でv451再現 | 91# | **棄却** | Gross PnL悪化(-1,162)、取引減少のみ |
| H2: threshold↑でコスト削減 | 100# §11 | **保留** | Fees削減はするが等量のGross PnL悪化で相殺 |
| H3: MHP↑で過剰取引抑制 | 101# | **棄却** | MHP=5は逆効果、MHP>10は効果あるがpolicy学習に寄与せず |
| H4: v451設定完全復元 | 91# | **棄却** | c1_v451_golden効果なし |
| 新仮説: 報酬構造がHOLD選好 | 本実験 | **要検証** | §4.2の報酬分析。取引インセンティブの付与で検証 |
| 新仮説: 50Kステップは探索のみ | 本実験 | **要検証** | 200K+で学習が始まる可能性 |

---

## 6. 計算基盤の検証結果

### 6.1 正常動作を確認した項目
- Gate2 KPI計算パイプライン: `ztb/metrics/metrics.py` → `compute_gate2_metrics_from_balances()`
- Deterministic evaluation loop: 50Kステップ上限キャップ、env再利用（MTFメモリ爆発回避）
- Subprocess分離: 各実験プロセスのメモリ ~910MB peak → 終了時完全解放
- 中間結果保存: クラッシュ耐性あり（`--resume`で再開可能）

### 6.2 判明した問題と対処

| 問題 | 対処 | ファイル |
|------|------|---------|
| MTF特徴量がOOMを起こす | `feature_set="v451"` + `correlation_reduction=False` | run_phase_c.py |
| VecEnvリセットでbalance履歴消失 | 学習後にdeterministic evalを別途実行 | run_phase_c.py |
| eval_envのn_steps=1.2M行で長時間ハング | `max_eval_steps = min(n_steps, TOTAL_TIMESTEPS)` でキャップ | run_phase_c.py |
| `min_holding_period` 既定=3, not 0 | 101# 検証済み、A-2実験の基準値修正 | 100# §12 |

---

## 7. 次ステップ候補

100# §13.6で提言した4オプションを再掲し、優先順位を付記する。

| 優先 | オプション | 内容 | 狙い | リスク |
|------|-----------|------|------|--------|
| ★★★ | **C3-A** | 報酬設計変更: trade incentive / hold penalty追加 | HOLDが最適解である構造を崩す | 過学習的な取引増、報酬ハック |
| ★★ | **C3-B** | discrete行動空間 (BUY/SELL/HOLD 3択) | continuous→discrete変換での情報損失を排除 | 実装変更が大きい |
| ★ | **C3-C** | ステップ数拡大 (50K→200K) | 探索フェーズが長いだけの可能性 | 構造問題なら時間の浪費 |
| ★★ | **C-alt** | FastIntradayEnvV456切替 | v451成功環境に回帰 | 88特徴量のメモリ問題再発 |
| — | **C-D** | ent_coef手動制御（auto→小値） | エントロピー探索を抑え、policyを決定的方向へ誘導 | 早期収束リスク |

### C3-A 報酬設計変更の具体案

**案1: 取引報酬 (trade_reward)**
```python
if action_taken (not HOLD):
    reward += trade_bonus  # e.g., 0.01
```
→ 取引すること自体に正のインセンティブ。ただし取引回数の爆発に注意。

**案2: hold_penalty > 0 復活**
```python
if action == HOLD and has_position:
    reward -= hold_penalty * holding_duration
```
→ 含み益の放置にペナルティ。利確学習を促進。

**案3: edge_penalty (HFT流)**
```python
if abs(predicted_edge) < transaction_cost:
    reward -= edge_penalty  # エッジがないのに取引したらペナルティ
```
→ `compute_hft_reward` のロジック。ただしHeavyEnv未接続のため要アダプタ。

**案4: reward_scale大幅引上げ**
```python
reward_scale: 1000.0  # 現行100.0の10倍
```
→ 報酬信号を強化して、エントロピー項に対する相対的重要度を上げる。最も低コストな試行。

---

## 8. 次ステップ実験結果（続報として追記予定）

> **以下、C3-A等の実験実施後に追記する。**

### 8.1 (TBD)

### 8.2 (TBD)
