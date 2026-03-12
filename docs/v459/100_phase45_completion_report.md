# 100# Phase 4.5 完了レポート — Gate C0-C3 結果と Phase 5 準備

**Date**: 2026-02-08
**Phase**: 4.5 (Gate C0-C3)
**大義**: 短期間での高収益性システム — 「計器を直し、正しく測り、方針を決める」

---

## 0. Executive Summary

### 実施内容
Phase B (97#) → 外部レビュー (98#) → コード検証 (99#) → 計器修正 (Gate C0-C1) → 再実験・ベースライン比較 (Gate C2-C3)

### 重要な発見

| 項目 | Phase B (修正前) | Phase 2 (C1修正後) | 変化 |
|------|-----------------|-------------------|------|
| BUY:SELL | 518:518 (推定値) | 512:512 (実測値) | 計測精度向上 |
| hold_penalty_multiplier | 0.0 (報酬消去) | 1.0 (PnL保持) | HOLD報酬正常化 |
| position_change_penalty | -0.1 (ハードコード) | 0.0 (設定値) | 不要ペナルティ除去 |
| Net ROI | -15% ± 1% | -14.99% ± 0.04% | 変化なし（手数料支配） |

### ベースライン比較結果

| 戦略 | Net ROI | Gross PnL | Fees | Trades | BUY:SELL |
|------|---------|-----------|------|--------|----------|
| Random | -15.02% ± 0.02% | ≈0 | 15,023 | 950 | 475:475 |
| BuyAndHold | -0.29% | 0 | 20 | 1 | 1:0 |
| Momentum_RSI | -5.59% | +7,051 | 12,455 | 729 | 365:364 |
| **SAC (P1-1)** | **-14.99% ± 0.04%** | **+1,171** | **16,161** | **1,024** | **512:512** |

### Go/No-Go判定
> **NO-GO（条件付き継続）** — SAC (P1-1) の Net ROI = -14.99% は Random (-15.02%) と統計的に同等。BuyAndHold (-0.29%) 及び Momentum_RSI (-5.59%) を大幅に下回る。Gross PnL = +1,171 は正値だが手数料 16,161 に対し 7.2% に過ぎない。**アルゴリズムレベルの改善が必要。**

---

## 1. Gate C0: 計測インフラ修正

### 1.1 BUY/SELLカウント実測化

**問題**: `core.py L1618-1631` の `buy_count`/`sell_count` プロパティが常に `int(trades_count * 0.5)` を返していた。PositionManager に `buy_count`/`sell_count` 属性が存在しなかったため、hasattr チェックが常に False。

**修正**: `PositionManager.__init__()` に `self.buy_count: int = 0`, `self.sell_count: int = 0` を追加。 `open_position()`, `close_position()` で正確にインクリメント。`get_state()`, `reset()` にも反映。

**検証**: `test_position_manager_buy_sell_count_exists` + `test_position_manager_reset_clears_counts` (PASS)

### 1.2 ハードコードペナルティ設定値化

**問題**: `reward_calculator.py L1289` に `if position_change > 0.1: reward -= 0.1` がハードコードされ、設定で無効化できなかった。

**修正**: `position_change_penalty` と `position_change_threshold` を RewardSettings から読み取るように変更。P1-1 では `position_change_penalty=0.0` で無効化。

**検証**: `test_position_change_penalty_is_configurable` (PASS)

### 1.3 サブプロセスログ保存

ログ保存機能を `run_phase45_p1_subprocess.py` に追加。各seed実行の stdout/stderr を `logs/{category}_seed{seed}.{stdout,stderr}.log` に保存。

---

## 2. Gate C1: hold_penalty_multiplier 修正

### 2.1 問題の発見と修正

**問題**: `hold_penalty_multiplier=0.0` 設定により、HOLD アクション時の報酬が `reward *= 0.0 = 0` に消去されていた。50,000 ステップ中 ~49,050 ステップが HOLD（no-op BUY/SELL も含む）であり、全学習データの 98% が報酬ゼロ。

**修正**: P1-1 設定で `hold_penalty_multiplier=1.0` に変更。PnLベースの報酬がHOLD時も保持される。

**検証**: `test_hold_penalty_multiplier_one_preserves_pnl` (PASS)

### 2.2 Gate テスト結果

```
tests/unit/trading/components/test_gate05_reward_purity.py: 16/16 PASSED
```

| テストカテゴリ | テスト数 | 結果 |
|---------------|---------|------|
| SimpleRewardPurity (Gate 0.5) | 6 | ✅ ALL PASS |
| PenaltyToggleEffect (Gate 0.5) | 2 | ✅ ALL PASS |
| UnknownKeyDetection (Gate 0.5) | 4 | ✅ ALL PASS |
| MeasurementIntegrity (Gate C0) | 4 | ✅ ALL PASS |

---

## 3. Phase 3: ベースライン実験結果

### 3.1 実験設定

- **環境**: HeavyTradingEnv（SAC実験と同一）
- **ステップ数**: 50,000（SAC実験と同一）
- **シード**: [42, 123, 456, 789]
- **reward_settings**: SAC実験と同一（`use_simple_reward=True`, `hold_penalty_multiplier=1.0`）
- **取引コスト**: 0.1%（片道）

### 3.2 Random Baseline

均等確率で `[-1, 1]` の一様乱数アクションを生成。

| Seed | Net ROI | Gross PnL | Fees | Trades | BUY:SELL |
|------|---------|-----------|------|--------|----------|
| 42 | -15.05% | +164 | 15,210 | 952 | 476:476 |
| 123 | -15.03% | -597 | 14,431 | 902 | 451:451 |
| 456 | -15.00% | -1,025 | 13,975 | 880 | 440:440 |
| 789 | -15.00% | +1,480 | 16,477 | 1,068 | 534:534 |
| **平均** | **-15.02%** | **+6** | **15,023** | **950** | **475:475** |

**考察**: Gross PnL ≈ 0（ランダムなので期待通り）。手数料が Net ROI を支配。

### 3.3 Buy & Hold Baseline

最初にBUYアクション、以後全てHOLD。

| Seed | Net ROI | Gross PnL | Fees | Trades |
|------|---------|-----------|------|--------|
| All | -0.29% | 0 | 20 | 1 |

**考察**: seed不変（deterministic）。50,000分（≈35日）の相場変動は -0.29%。取引1回のみ。

### 3.4 Momentum RSI Baseline

RSI<30→BUY(0.8), RSI>70→SELL(-0.8), else→HOLD(0.0)

| Seed | Net ROI | Gross PnL | Fees | Trades | BUY:SELL |
|------|---------|-----------|------|--------|----------|
| All | -5.59% | **+7,051** | 12,455 | 729 | 365:364 |

**考察**: seed不変（RSI信号が決定的）。**Gross PnL は +7,051 で正**。手数料 12,455 が利益を食い尽くしている。1取引あたり粗利 +9.67 JPY vs 手数料 17.09 JPY。

### 3.5 ベースライン比較の意義

| 指標 | Random | BuyHold | Momentum | Phase 5目標 |
|------|--------|---------|----------|-------------|
| Net ROI | -15.02% | -0.29% | -5.59% | > 5% |
| Gross PnL | ≈0 | 0 | +7,051 | — |
| Trades | 950 | 1 | 729 | — |
| 粗利/取引 | ≈0 | — | 9.67 | — |
| 手数料/取引 | 15.81 | 20.00 | 17.09 | — |

**核心的洞察**: 
1. **Random = 手数料相当の損失** (-15%) → SAC がこれを超えなければ「学習していない」
2. **Momentum_RSI = 手数料前では利益** → 取引コスト削減が収益化の鍵
3. **BuyHold = 最小損失** → ベンチマークとして最も安定

---

## 4. Phase 2: SAC 再実験結果（Gate C1修正後）

### 4.1 P1-1（純PnL + C0/C1修正）

`use_simple_reward=True`, `hold_penalty_multiplier=1.0`, 全penalty=0.0

| Seed | Net ROI | Gross PnL | Fees | Trades | BUY:SELL |
|------|---------|-----------|------|--------|----------|
| 42 | -15.00% | -523 | 14,478 | 908 | 454:454 |
| 123 | -14.93% | +2,173 | 17,102 | 1,078 | 539:539 |
| 456 | -14.99% | +1,606 | 16,596 | 1,064 | 532:532 |
| 789 | -15.04% | +1,429 | 16,470 | 1,048 | 524:524 |
| **平均** | **-14.99%** | **+1,171** | **16,161** | **1,024** | **512:512** |

**考察**:
- **Net ROI ≈ -15%** で Random (-15.02%) とほぼ同一。Gate C1修正（hold_penalty=1.0）は Net ROI に有意な影響を与えなかった。
- **Gross PnL = +1,171** は平均して正。SACは粗利レベルでは若干のエッジを学習している可能性がある。
- **手数料 16,161 >> Gross PnL 1,171** がNet損失の主因。手数料率 1,380% → 取引コスト削減が唯一の改善経路。
- **BUY:SELL が完全対称** (454:454, 539:539, etc.) — SACはロング/ショートの方向性優位を見出せていない。

### 4.2 P1-3（デフォルト報酬 + C0/C1修正）

`use_simple_reward=False`（complex reward path）

| Seed | Net ROI | Gross PnL | Fees | Trades | BUY:SELL |
|------|---------|-----------|------|--------|----------|
| 42 | -15.01% | -81 | 14,927 | 928 | 464:464 |
| 123 | — | — | — | — | — |
| 456 | — | — | — | — | — |
| 789 | — | — | — | — | — |

※ P1-3はseed=42のみ完了。残りはバックグラウンド実行中。

**考察**: P1-3 seed=42 も -15.01% で P1-1 と同等。complex reward path も改善効果なし。

---

## 5. Gate C3: 統計検定結果（0番§5.6）

SAC P1-1 (n=4) vs ベースライン各 (n=4)

### 5.1 Mann-Whitney U 検定

| 比較 | U統計量 | p値 | Holm補正後 |
|------|---------|-----|-----------|
| SAC(P1-1) vs Random | 11.0 | 0.6422 | 非有意 |
| SAC(P1-1) vs BuyHold | 0.0 | 0.0314 | 非有意* |
| SAC(P1-1) vs Momentum | 0.0 | 0.0314 | 非有意* |

*Holm-Bonferroni補正後（α=0.05/3=0.0167）で閾値を超えず非有意。

### 5.2 Cliff's Delta（効果量）

| 比較 | Delta | 効果量 | 0番§5.6基準(>0.33) | 方向 |
|------|-------|--------|-------------------|------|
| SAC(P1-1) vs Random | +0.375 | medium | ✅ PASS | SAC微優位(+0.03%) |
| SAC(P1-1) vs BuyHold | -1.000 | large | ✅ PASS | BuyHold優位(-14.70%) |
| SAC(P1-1) vs Momentum | -1.000 | large | ✅ PASS | Momentum優位(-9.40%) |

### 5.3 統計的結論

- **SAC ≈ Random**: +0.03% の差は統計的に非有意 (p=0.64)。SACの学習効果は Random とノイズレベルで区別不能。
- **SAC << BuyHold**: -14.70% の差 (p=0.03, Cliff's d=-1.0)。Holm補正後は非有意だが、効果量は large。n=4 の検出力限界。
- **SAC << Momentum_RSI**: -9.40% の差。単純RSIルールベースにも大幅に劣後。

---

## 6. Go/No-Go 判定

### 最終判定: **NO-GO（条件付き継続）**

| 基準 | 値 | 判定 |
|------|------|------|
| Net ROI > 5% | -14.99% | ❌ FAIL |
| Net ROI > 0% | -14.99% | ❌ FAIL |
| Gross PnL > 0 | +1,171 | ✅ PASS |
| SAC > Random (有意) | +0.03% (p=0.64) | ❌ FAIL |
| SAC > BuyHold | -14.70% | ❌ FAIL |
| SAC > Momentum_RSI | -9.40% | ❌ FAIL |

### 判定ロジック

| 条件 | 判定 | 次アクション |
|------|------|-------------|
| Net ROI > 5% かつ有意差あり | **GO → Phase 5** | Walk-forward 4split実施 |
| 0% < Net ROI < 5% | **CONDITIONAL GO → Phase C** | コスト圧縮実験 |
| -5% < Net ROI < 0% | **CONDITIONAL** | コスト圧縮で0%到達可能性評価 |
| Net ROI < -5% | **NO-GO** | アーキテクチャ再設計 |

### 追加判定: ベースライン超過

| 条件 | 意味 |
|------|------|
| SAC > Momentum_RSI | SACが単純RSI戦略を超える → RL学習に価値あり |
| Random < SAC ≤ Momentum_RSI | 学習はしているがRSIルールに劣る |
| SAC ≈ Random | 学習効果なし |

---

## 7. Phase 5 準備状況（66#/0番§5.2）

### 7.1 メトリクス測定体制

| メトリクス | 測定可否 | 備考 |
|-----------|---------|------|
| Net ROI | ✅ | PositionManager + portfolio_value |
| Profit Factor | ⚠️ | 個別取引のwin/loss集計が必要 |
| Sharpe Ratio | ⚠️ | エピソードリターンの年率化が必要 |
| Max Drawdown | ⚠️ | ポートフォリオ時系列の追跡が必要 |
| Win Rate | ⚠️ | 個別取引の勝敗集計が必要 |
| buy_count / sell_count | ✅ | Gate C0で実装済み |

### 7.2 統計検定体制（66#/0番§5.6）

| 要件 | 状態 | 備考 |
|------|------|------|
| n ≥ 16 (4seed × 4split) | ⚠️ 4seed実施（4splitは Phase 5） | Phase 5で walk-forward |
| Mann-Whitney U | ✅ | gate_c3_comparison.py に実装 |
| Holm-Bonferroni | ✅ | gate_c3_comparison.py に実装 |
| Cliff's Delta > 0.33 | ✅ | gate_c3_comparison.py に実装 |

### 7.3 Phase 5 に向けた残課題

1. **Profit Factor / Sharpe / MaxDD / WinRate** の計測インフラ追加
2. **Walk-forward split** (4分割OOS) の実装
3. **コスト圧縮** (continuous_threshold × min_holding_period) の最適化
4. **ステップ数拡大** (50K → 200K+) での学習効果検証

---

## 8. コミット履歴

| Hash | 説明 |
|------|------|
| 3cb71469d | Gate C0/C1修正 + ベースライン + テスト (2026-02-08) |
| TBD | Gate C3分析 + 100#完了レポート (本コミット) |

---

## 9. ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `ztb/trading/environment/components/position_manager.py` | buy_count/sell_count追加 |
| `ztb/trading/environment/components/calculators/reward_calculator.py` | position_change_penalty設定値化 |
| `scripts/v459/run_phase45_p1.py` | hold_penalty_multiplier=1.0 |
| `scripts/v459/run_phase45_p1_subprocess.py` | seed別ログ保存 |
| `scripts/v459/run_baselines.py` | ベースライン3戦略実験 |
| `scripts/v459/gate_c3_comparison.py` | 統計検定比較分析 |
| `tests/unit/trading/components/test_gate05_reward_purity.py` | 16テスト(Gate 0.5 + C0) |

---

---

## 10. 根本原因分析と次フェーズへの提言

### 10.1 なぜ SAC ≈ Random なのか

1. **手数料が利益を完全に吸収**: SAC Gross PnL = +1,171 vs Fees = 16,161 → 手数料率 1,380%。取引コスト 0.1%/片道でも、1,024回取引で原資の 16% が消失。
2. **BUY:SELL完全対称**: SACが方向性エッジを学習していない証拠。50K ステップでは特徴量とリターンの相関を学習するには不十分な可能性。
3. **1分足の低S/N比**: BTC/JPY 1分足のRSI系特徴量だけでは、取引コストを超える予測力を持つシグナルの抽出が困難。

### 10.2 次フェーズへの提言

| 優先度 | 施策 | 期待効果 | 根拠 |
|--------|------|---------|------|
| 1 | **取引コスト削減** (continuous_threshold↑) | Net ROI +10%以上 | 手数料16,161 → trades半減で8,000削減 |
| 2 | **ステップ数拡大** (50K → 200K+) | 学習効果の発現 | 50Kでは方向性学習に不十分 |
| 3 | **特徴量拡充** (ボリュームプロファイル, OBV等) | 予測力向上 | RSI系のみでは限界 |
| 4 | **報酬関数のShaping** (取引頻度ペナルティ) | 無駄取引の抑制 | BUY:SELL対称問題の解消 |

*Phase 4.5 完了。Gate C3判定 = NO-GO。Phase C (コスト圧縮) → 特徴量/報酬見直し → ステップ数拡大の順で改善を進める。*

---

## 11. Phase C 実行計画 — 既存実装活用による改善ロードマップ

> **方針**: 新規実装を極力避け、既存の豊富な実装資産を組み合わせる。
> **優先順位**: ① 取引コスト削減 → ② 報酬関数見直し → ③ 特徴量拡充 → ④ ステップ数拡大（後回し）

---

### 11.1 施策 A: 取引コスト削減（最優先）

**問題**: SAC Gross PnL = +1,171 に対し手数料 = 16,161（1,380%）。取引回数 1,024 回が全損失の原因。

#### A-1. continuous_threshold の引き上げ

| パラメータ | 現行値 | 既存実装 | 調整案 |
|-----------|--------|---------|--------|
| `SAC_CONTINUOUS_THRESHOLD` | 0.3333 | [ztb/trading/constants.py L29](ztb/trading/constants.py#L29) | 0.5 / 0.6 / 0.7 を比較 |
| `EnvironmentConfig.continuous_to_discrete_threshold` | 0.3333 | [config.py L286](ztb/trading/environment/utils/config.py#L286) | 同上 |
| `min_action_threshold` (FastIntraday) | 0.001 | [fast_intraday_env_v456.py L288](ztb/trading/environment/fast_intraday_env_v456.py#L288) | 0.01〜0.05 |

**既存の動的閾値システム**（即利用可能）:
- `ThresholdManager` ([threshold_manager.py](ztb/trading/environment/components/threshold_manager.py)): adaptive_mode, z_score_mode, ボラティリティ乗算, 市場レジーム検出
- v456設定: `dynamic_threshold_mode: 'z_score'`, `z_score_threshold: 3.0`, `max_action_threshold: 0.08`

**期待効果**: threshold 0.3→0.6 で取引回数約50%削減 → 手数料 -8,000 → Net ROI: -15% → -7%

#### A-2. min_holding_period の導入

| パラメータ | 現行値 | 既存実装 | 調整案 |
|-----------|--------|---------|--------|
| `min_holding_period` | 0（HeavyEnv） | [config.py L222](ztb/trading/environment/utils/config.py#L222) | 3 / 5 / 10 |
| `RECOMMENDED_MIN_HOLDING_PERIOD` | 3 | [environment/constants.py L126](ztb/trading/environment/constants.py#L126) | — |
| `enforce_reverse_cooldown` | False | [config.py L321](ztb/trading/environment/utils/config.py#L321) | True |

**LiveTrader既定値**: `min_holding_period=5`（[live_trader.py L268](ztb/trading/live_trader/live_trader.py#L268)）→ 訓練時もこれを反映すべき。

#### A-3. 取引コスト段階的低減実験

| 設定 | transaction_cost | 根拠 |
|------|-----------------|------|
| Zaif Maker | **0.001 (0.1%)** | 現行 |
| v450-v454方式 | **0.0005 (0.05%)** | 過去に使用実績あり |
| ゼロコスト訓練 | **0.0** | [run_ab_minimal.py L49](scripts/v459/run_ab_minimal.py#L49) に実験済み |

**注意**: ゼロコスト訓練→実コスト評価のギャップを交差検証で制御する必要あり。

#### A 実験設計

```
A-1: threshold_sweep = [0.3, 0.5, 0.6, 0.7]  × 4seeds
A-2: holding_period  = [0, 3, 5, 10]         × 4seeds
A-3: tx_cost         = [0.001, 0.0005, 0.0]  × 4seeds
→ 計 44 実験 (A-1×16 + A-2×16 + A-3×12)
→ 各50Kステップ ≈ 40分 → 直列で約30時間、並列2で15時間
→ 先に A-1 の4値を単独seed=42で試行→最良値を確定→フルseed展開
```

---

### 11.2 施策 B: 報酬関数見直し

**問題**: BUY:SELL 完全対称 → 方向性学習ができていない。取引回数過多。

#### B-1. 既存の取引頻度ペナルティを有効化

現在 P1-1 は全ペナルティ=0.0。以下の既存パラメータを段階的に導入:

| パラメータ | 現行 | 既存YAML | 調整案 |
|-----------|------|---------|--------|
| `trade_frequency_penalty` | 0.0 | [stage1_trade_reduced.yaml](configs/rewards/stage1_trade_reduced.yaml): **0.01** | 0.005 / 0.01 |
| `trade_cooldown_steps` | 5 | 同上: **10** | 5 / 10 |
| `trade_cooldown_penalty` | 0.01 | [config.py L67](ztb/trading/environment/utils/config.py#L67) | 0.01 / 0.05 |
| `consecutive_trade_penalty` | 0.05 | [config.py L68](ztb/trading/environment/utils/config.py#L68) | 0.05 / 0.1 |
| `action_smoothing` | 0.0 | [stage1_exploration_tuned.yaml](configs/rewards/stage1_exploration_tuned.yaml): **0.01** | 0.01 |

#### B-2. V457RewardCalculator（v451 Golden Era復元）

[v457_reward_calculator.py](ztb/trading/environment/components/calculators/v457_reward_calculator.py) — 85行のシンプルな純PnL計算器:
- 非対称損失回避 (loss_multiplier=1.2)
- 報酬クリッピング [-5, 5]
- **これが歴史的に最良の結果を出した報酬設計** (91#参照)

**v451設定の復元候補**:
| パラメータ | v451値 | 現行P1-1値 | 差分 |
|-----------|--------|------------|------|
| gamma | **0.80** | 0.99 | 短期→長期 |
| reward_scale | **1.0** | 100.0 | 100倍の差 |
| loss_multiplier | **1.2** | 1.0 | 非対称なし |

#### B-3. HFT報酬関数の活用

[ztb/trading/rewards/fast_intraday.py](ztb/trading/rewards/fast_intraday.py) の `compute_hft_reward()`:
- `edge_penalty_rate`: ATRが取引コスト×倍率を下回るときペナルティ（**低ボラ時の無駄取引を抑制**）
- `vol_floor_penalty`: 低ボラティリティ時の保有ペナルティ
- `hold_ramp`: 保有時間による時間減衰ペナルティ

→ HeavyTradingEnv の simple_reward にこの「エッジがないなら取引するな」ロジックを統合可能。

#### B 実験設計

```
B-1: stage1_trade_reduced.yaml ベース × 4seeds
B-2: V457RewardCalculator（gamma=0.80, reward_scale=1.0） × 4seeds
B-3: B-1 + edge_penalty_rate × 4seeds
B-mix: A最良 + B最良の組合せ × 4seeds
→ 計 16 実験
```

---

### 11.3 施策 C: 特徴量拡充

**問題**: 現行8特徴（RSI×7 + ReturnStdDev）は多様性ゼロ。RSI変種のみ。

#### C-0. 8特徴への削減経緯と問題点（要Codexレビュー）

**経緯**: 35# で特徴生成が訓練時間の 63.7% を占有する速度ボトルネックが発見され、**収益性ではなく速度を動機**として `FeatureRegistry.get_optimized_feature_set(correlation_threshold=0.95)` による相関ベース削減が実施された (39#)。結果として 88 → 8 特徴に削減。

**しかし「相関フィルタが実際に機能したか」は極めて疑わしい:**

1. **分析ファイル不在**: `reports/feature_analysis_*.json` が現在ワークスペースに存在しない。[registry.py](ztb/features/core/registry.py) の `select_features_by_correlation()` は分析ファイルが無い場合 **`cls.list()`（登録済み全特徴をそのまま返す）** 設計。38# レビューでもこの点が警告されていた。
2. **遅延インポート問題**: `precompute_optimized_features.py` は `from ztb.features.core.registry import FeatureRegistry` のみインポート。generators 配下の各モジュール（ADX, OBV, MACD 等）は遅延ロードで**未登録のまま**だった可能性が高い。
3. **結論**: 88→8 の「相関ベース削減」は**実際には起きておらず、たまたまインポート時に `_registry` に登録されていた RSI 系 + ReturnStdDev がそのまま採用された実装ミス**の可能性が高い。

**A/Bテスト (45#) の統計的信頼性も低い:**

| 問題 | 詳細 |
|------|------|
| **n=2** | seed 42, 123 のみ。統計的検出力ほぼゼロ |
| **両方損失** | -5.07% vs -5.25% — 「どちらがマシか」の消極的比較 |
| **差分 +0.17%** | 2サンプルで有意と言えない |
| **「5.2倍安定」** | 2サンプルの分散比に統計的信頼性なし |

**速度恩恵（589倍高速化）は本物**だが、収益性への影響は未検証のまま RSI 変種のみの特徴セットが Phase 4.5 全体にわたって固定化された。74# で 22 特徴拡張が計画されたが gamma ablation 優先で保留され、現在に至る。

> **Codex レビュー依頼**: `FeatureRegistry.get_optimized_feature_set()` の呼出経路で実際に相関フィルタが機能していたか、遅延インポートの影響を含めて検証してほしい。該当箇所: [registry.py L760-831](ztb/features/core/registry.py#L760), [precompute_optimized_features.py L69-72](scripts/v459/precompute_optimized_features.py#L69)

#### C-1. FeatureRegistry 未使用資産の棚卸し

**実装済み・未使用の高価値特徴量**（即利用可能）:

| カテゴリ | 指標 | ファイル | 期待効果 |
|---------|------|--------|---------|
| **ボリューム** | OBV, CMF, MFI, VWAP | [volume/](ztb/features/generators/technical/volume/) | 出来高→価格先行シグナル |
| **ボラティリティ** | ATR(MTF), BB_Width/Position, Normalized_ATR | [volatility/](ztb/features/generators/technical/volatility/) | 取引タイミング最適化 |
| **トレンド** | ADX, EMA_Cross, Supertrend | [trend/](ztb/features/generators/technical/trend/) | 方向性学習の支援 |
| **モメンタム** | MACD, Stochastic, Williams_R | [momentum/](ztb/features/generators/technical/momentum/) | RSI補完 |
| **スキャルピング** | price_velocity, micro_trend, volume_surge | [scalping.py](ztb/features/scalping.py) | 1分足特化シグナル |
| **時間** | hour_sin/cos, dow_sin/cos | [time/cyclical_v456.py](ztb/features/time/cyclical_v456.py) | 時間帯効果 |

#### C-2. v459拡張Parquetの活用

[scripts/v459/generate_v459_features.py](scripts/v459/generate_v459_features.py) が22特徴量セットを生成済み:
- `log_return`, `close_change_pct`, `RSI_M1/H1/D1`, `SMA20/50_ratio`, `EMA_slope`
- `Stochastic_K`, `Williams_R`, `ROC`, `ATR_norm`, `BB_position`
- `volume_ratio`, `OBV_slope`, `hour/day sin/cos`, `vol_ratio/rank`

**課題**: 現在24,374行のサブセットのみ → 1.2M行フルデータで再生成が必要。

#### C-3. v456 FastIntradayEnv の88次元観測空間

[fast_intraday_env_v456.py L113](ztb/trading/environment/fast_intraday_env_v456.py#L113) の固定88次元構成が既に最適化済み:
- Base 30D (OHLCV + SMA/EMA/RSI/ATR/BB/MACD/ADX/OBV)
- MTF 27D (3TF × 9指標)
- Cyclical 6D + Global 6D + Regime 13D + Account 6D

**選択肢**: HeavyTradingEnv の Parquet 特徴量を拡充するか、FastIntradayEnvV456 に切り替えるか。

#### C-4. 特徴量セット実験（74#計画の復活）

74#で保留された8→22特徴拡張計画を復活:

```
C-base: 現行8特徴（RSI×7 + ReturnStdDev）      × seed=42
C-mid:  14特徴（+ATR, BB, MACD, OBV, Volume, EMA）× seed=42
C-full: 22特徴（v459拡張セット）                  × seed=42
C-v456: FastIntradayEnv 88次元                   × seed=42
→ 計 4 実験（seed=42での速攻比較）
→ 最良セットでフル4seed展開
```

**過去の知見** (45# A/Bテスト): 8特徴 vs フル特徴で ROI 差 +0.17% → 8特徴採用。**ただしこのテストは n=2 で統計的に無意味であり、かつ「フル特徴」自体も RSI 変種過多で多様性が低かった (§C-0 参照)**。今回は ATR/BB/OBV/Volume 等の**異質な次元**を追加するため、45# とは本質的に異なる比較になる。

---

### 11.4 施策 D: ステップ数拡大（後回し）

50K → 200K+ は最も時間コストが高い（現状40分/50K → 160分/200K × 4seed = 10時間超）。
施策 A-C の効果が確認されてから実施。

---

### 11.5 統合実行タイムライン

| フェーズ | 施策 | 実験数 | 推定時間 | 判定基準 |
|---------|------|--------|---------|---------|
| **C-α** | A-1 threshold_sweep (seed=42のみ) | 4 | **2.5h** | 取引回数50%削減 |
| **C-β** | A-1最良 + A-2 holding_period (seed=42のみ) | 4 | 2.5h | Net ROI > -10% |
| **C-γ** | B-1/B-2/B-3 報酬見直し (seed=42のみ) | 3 | 2h | Gross PnL向上 + 取引削減 |
| **C-δ** | C-1/C-2 特徴量比較 (seed=42のみ) | 4 | 2.5h | ROI差 > 1σ |
| **C-ε** | 最良の A+B+C 組合せ × 4seeds | 4 | 2.5h | Gate C3再判定 |
| **C-ζ** | (optional) ステップ数拡大 200K | 4 | 10h+ | — |
| | **合計 (C-ζ除く)** | **19** | **≈12h** | |

#### Gate C-ε（フェーズC最終判定）

| 判定 | 条件 | 次ステップ |
|------|------|-----------|
| **GO** | Net ROI > 0% && SAC > Momentum_RSI (有意) | → Phase 5 Walk-forward |
| **CONDITIONAL** | Net ROI > -5% && Gross PnL > 0 | → ステップ数拡大 (C-ζ) |
| **NO-GO** | Net ROI < -5% | → アーキテクチャ根本見直し |

---

### 11.6 既存実装資産マップ（Codexレビュー用）

以下は本計画で活用する既存コードの一覧。**新規実装は原則不要。**

#### 取引コスト制御

| コンポーネント | パス | 状態 |
|---------------|------|------|
| ThresholdManager | `ztb/trading/environment/components/threshold_manager.py` | Active |
| FeeModel (Fixed/Tiered/Exchange) | `ztb/utils/fee_model.py` | Active |
| VenueTransactionCostManager | `ztb/trading/cost/venue_transaction_cost_manager.py` | Active |
| stage1_trade_reduced.yaml | `configs/rewards/stage1_trade_reduced.yaml` | Active |

#### 報酬関数

| コンポーネント | パス | 状態 |
|---------------|------|------|
| RewardCalculator (2131行) | `ztb/trading/environment/components/calculators/reward_calculator.py` | Active |
| V457RewardCalculator (v451復元) | `ztb/trading/environment/components/calculators/v457_reward_calculator.py` | Active |
| compute_hft_reward | `ztb/trading/rewards/fast_intraday.py` | Active |
| DynamicRewardShaper | `ztb/trading/environment/components/dynamic_reward_shaper.py` | Active |
| RewardConfigSchema (YAML→Settings) | `ztb/training/reward_config_schema.py` | Active |

#### 特徴量エンジニアリング

| コンポーネント | パス | 状態 |
|---------------|------|------|
| FeatureRegistry (100+特徴量) | `ztb/features/core/registry.py` | Active |
| base_features_v456 (30特徴量) | `ztb/features/base_features_v456.py` | Active |
| scalping features (12指標) | `ztb/features/scalping.py` | Active |
| MultiTimeframeFeatureSystem | `ztb/features/generators/multi_timeframe/engine.py` | Active |
| generate_v459_features | `scripts/v459/generate_v459_features.py` | Active |
| FeatureCorrelationProcessor | `ztb/preprocessing/feature_correlation_filter.py` | Active |
| Feature ablation framework | `ztb/benchmarks/ablate_features.py` | Active |

#### 分析・評価

| コンポーネント | パス | 状態 |
|---------------|------|------|
| gate_c3_comparison.py | `scripts/v459/gate_c3_comparison.py` | Active |
| Metrics (PF/Sharpe/MDD/WinRate) | `ztb/metrics/metrics.py` | Active |
| RewardFunctionEvaluator | `ztb/training/reward_function_evaluator.py` | Active |

---

### 11.7 歴史的教訓の整理

| Doc# | 教訓 | 本計画への反映 |
|------|------|---------------|
| 45# | 8特徴≧フル特徴（RSI変種のみの場合） | C-3で**異質な特徴量**を追加 |
| 67# | 報酬設計が主因、SACハイパラは副因 | B-1/B-2で報酬を先に最適化 |
| 72# | ent_coef小→HOLD回避最適化（取引回避） | A-1でthreshold調整を優先 |
| 75# | gamma=0.99>>0.95（圧倒的差） | B-2でgamma=0.80も検証(v451方式) |
| 91# | v451 Golden Eraはgamma=0.80, scale=1.0 | B-2で復元実験 |
| 98# | hold_penalty=0.0は致命的バグ | 全実験で1.0固定 |
| 100# | Gross PnL=+1,171>0 → エッジは存在する | コスト削減で顕在化を狙う |
---

## 12. 101#レビュー反映 — 修正事項と改訂Phase C計画

> **参照**: `101_phase45_followup_reuse_recommendations.md` (Codexレビュー, 2026-02-08)
> **検証方法**: 全指摘事項をソースコード読解により独立検証。6件全て正当性を確認。

### 12.1 §11の誤り訂正（コード検証済み）

#### 訂正1: `min_holding_period` の現行値 — 0 ではなく **3**

| 項目 | 100# §11記載 | 実際の値 | 根拠 |
|------|-------------|----------|------|
| `EnvironmentConfig.min_holding_period` | 0 | **3** | `ppo_config.py` L119 → `config.py` L222 |
| `environment/constants.py DEFAULT_MIN_HOLDING_PERIOD` | — | 0 (dead code, 未参照) | |
| `core.py getattr(self.config, "min_holding_period", 0)` | — | フォールバック到達不能 | 属性は常に存在 |
| P1-1実験の実効値 | 0と暗黙推定 | **3** | スクリプトに明示設定なし→デフォルト3 |

**影響**: A-2実験の基準値を 0→**3** に修正。調整案も 3/5/10 → **5/10/15** に変更。

#### 訂正2: `dynamic_threshold_mode` — HeavyEnvに配線不在

`ThresholdManager` は `getattr(config, "dynamic_threshold_mode", "fixed")` で読み取るが、`EnvironmentConfig` に以下のフィールドが**存在しない**:
- `dynamic_threshold_mode`
- `z_score_window`
- `z_score_threshold`
- `z_score_method`

→ 常に `"fixed"` にフォールバックし、z_score動的閾値モードは**設定経由では有効化不可能**。
→ §11.1 の「即利用可能」は**誤り**。`EnvironmentConfig` と `from_dict()` へのフィールド追加が前提条件。

#### 訂正3: `stage1_trade_reduced.yaml` は `use_simple_reward: false`

3つの既存stage1 YAMLは全て `use_simple_reward: false`（complex reward path）:

| YAML | use_simple_reward |
|------|------------------|
| `stage1_trade_reduced.yaml` | **false** |
| `stage1_basic.yaml` | **false** |
| `stage1_exploration_tuned.yaml` | **false** |

→ §11.2 B-1のYAML流用は**PnL-onlyアプローチとは別トラック**。因果分離のため2系統に分離する。

#### 訂正4: `V457RewardCalculator` の有効化パス

有効化には `reward_settings.custom_reward_params` に `type: "pnl_centered"` を**明示指定**が必要:

```python
# initialization.py L654-679
reward_type = self.reward_settings_obj.custom_reward_params.get("type")
if reward_type == "pnl_centered":
    # V457RewardCalculator を使用
else:
    # デフォルト RewardCalculator (2131行の複雑ロジック)
```

→ §11.2 B-2の実験設計に設定スイッチを明記。

#### 訂正5: `compute_hft_reward` — HeavyEnv未接続

`compute_hft_reward` は `FastIntradayEnv` 専用。`HeavyTradingEnv` の `core.py` にimport/参照は**ゼロ**。
→ §11.2 B-3の「統合可能」を「アダプタ実装が必要」に修正。**Phase C初期では見送り**。

#### 訂正6: 決定的ベースラインの統計的独立性

`BuyAndHold` と `Momentum_RSI` は決定的 — seed変更で同一結果。n=4の独立サンプルとして扱うのは統計的に不正確。
→ ベースライン比較は**window分割（時期差）**で標本を作る方式に変更。

### 12.2 追加の批判的観点（101# §1 検証済み）

| 批判点 | 妥当性 | 対策 |
|--------|--------|------|
| §1.1 決定的baseline seed independence | **妥当** | window分割方式に変更 |
| §1.2 n=4 Mann-Whitney近似誤差 | **妥当** | window増加 or exact検定 |
| §1.3 Gross/Net指標の曖昧さ | **妥当** | realized/mark-to-market を明示分離 |
| §1.4 OOS未確定 | **妥当** | C4で即walk-forward実施 |

### 12.3 資産再利用判定（101# §2 検証済み）

#### A. 即採用 — 検証で問題なし

| 資産 | 検証状態 |
|------|---------|
| `ztb/metrics/metrics.py` | PF/Sharpe/MaxDD/WinRate 実装確認 |
| `ztb/evaluation/walk_forward/splitter.py` | 機能的。embargo付き時系列分割 |
| `ztb/evaluation/walk_forward/evaluator.py` | 機能的だがDeprecated → `UnifiedEvaluator`推奨 |
| `tests/unit/evaluation/test_walk_forward_*` | **64テスト全PASS** (1.55秒) |
| `ztb/trading/environment/components/threshold_manager.py` | 機能的（ただしz_scoreモードは配線修正要） |
| `ztb/trading/execution/realistic.py` | 機能的。ATR連動スリッページ、50ms±20msレイテンシ |
| `ztb/trading/execution/pseudo_hft.py` | 機能的。3成分スリッページモデル |
| `ztb/trading/cost/venue_transaction_cost_manager.py` | 機能的。Zaif maker/taker 0.1% |

#### B. 条件付き採用

| 資産 | 条件 |
|------|------|
| `config/v451/sac_v451_optimized.json` | gamma=0.80, reward_scaling=1.0 を独立A/B検証 |
| `V457RewardCalculator` | `custom_reward_params.type="pnl_centered"` 指定で有効化 |
| `FeatureRegistry` + 相関フィルタ | analysis JSON生成 + 明示的generator import が前提 |

#### C. 非推奨（現状のまま不可）

| 資産 | 理由 |
|------|------|
| `ztb/analysis/baseline_comparison.py` | `BaselineComparisonEngine` 重複定義で1回目の機能消失 |
| `scripts/v459/check_data_leakage.py` | MTF因果性チェックが`pass`のみ（プレースホルダ） |
| `scripts/v458/run_walk_forward_v458.py` | script import依存が脆い |

### 12.4 改訂Phase C実行計画

101#の指摘を反映し、§11の実行順を改訂。**C0（計測統一）を最優先**に追加。

#### C0: 計測統一（最優先 — 新規追加）

| 作業 | 詳細 | 既存資産 |
|------|------|---------|
| KPI統一 | `ztb/metrics/metrics.py` でPF/Sharpe/MaxDD/WinRateを全実験に適用 | 即利用可 |
| realized/MTM分離 | `position_manager.py` の `realized_pnl` / `unrealized_pnl` を明示記録 | 即利用可 |
| 決定的baseline | window分割（4期間）で標本作成。seed複製廃止 | splitter.py活用 |
| Mann-Whitney改善 | window増加で近似依存を低減、または bootstrap confidence interval | gate_c3改修 |

#### C1: コスト圧縮（§11.1改訂版）

| 実験 | 変更点 |
|------|--------|
| A-1: threshold_sweep [0.3, 0.5, 0.6, 0.7] | 変更なし |
| A-2: min_holding_period [**3**, 5, 10, 15] | 基準値 0→**3** に修正 |
| A-3: tx_cost [0.001, 0.0005, 0.0] | 変更なし |
| ~~動的閾値(z_score)~~ | **延期** — `EnvironmentConfig` 配線修正が先 |

#### C2: 約定モデル・コスト現実化（101# §4準拠、新規追加）

| 実験 | 既存資産 |
|------|---------|
| `realistic.py` スリッページ感度 | ATR連動、base_slippage=0.05% |
| `pseudo_hft.py` 高頻度頑健性 | 3成分スリッページ（spread/vol/impact） |
| `venue_transaction_cost_manager.py` venue別感度 | Zaif 0.1% vs bitflyer -0.02%(maker) |

#### C3: 報酬・ハイパラ再現実験（§11.2改訂版）

2系統に分離（§訂正3対応）:

**系統A: PnL-only微調整（因果分離優先）**
| 実験 | 設定 |
|------|------|
| base | 現行 `use_simple_reward=True` |
| +penalty | `trade_frequency_penalty=0.01`, `trade_cooldown_steps=10` |
| v451 | `custom_reward_params.type="pnl_centered"`, gamma=0.80, reward_scale=1.0 |

**系統B: Complex reward比較（別トラック）**
| 実験 | 設定 |
|------|------|
| stage1_reduced | `stage1_trade_reduced.yaml` (`use_simple_reward=false`) |
| stage1_basic | `stage1_basic.yaml` (`use_simple_reward=false`) |

> `compute_hft_reward` のHeavyEnv統合は**Phase C後半またはPhase 5**に延期。

#### C4: OOS最終判定（101# §4準拠）

| 作業 | 既存資産 |
|------|---------|
| 4-split walk-forward | `splitter.py` (embargo=7日) |
| 4seeds × 4splits 評価 | `evaluator.py` or `UnifiedEvaluator` |
| Random/BuyHold/Momentum超過判定 | `gate_c3_comparison.py` 改修版 |
| Phase 5可否確定 | PF>1.0, Sharpe>0, Net ROI>0% |

### 12.5 改訂タイムライン

| フェーズ | 施策 | 推定時間 | Gate |
|---------|------|---------|------|
| **C0** | 計測統一 + baseline再構築 | 4h (実装) | — |
| **C1** | threshold + holding_period sweep (seed=42) | 5h | ROI > -10% |
| **C2** | 約定モデル感度 (seed=42) | 3h | — |
| **C3** | 報酬2系統 (seed=42) | 4h | Gross PnL↑ + 取引↓ |
| **C-ε** | 最良組合せ × 4seeds | 3h | Gate C3再判定 |
| **C4** | 4-split OOS | 6h | PF>1.0, Net ROI>0% |
| | **合計** | **≈25h** | |

### 12.6 101#への回答

101#の指摘は**6件全て正当**。§11は方向性として概ね正しいが、実装前提に5つの誤りがあった。
本§12で全修正を反映し、C0（計測統一）を追加した改訂版で Phase C を実行する。

---

## 13. Phase C0+C1 実験結果（2026-02-09）

### 13.1 実行環境

- **スクリプト**: `scripts/v459/run_phase_c.py` + `run_phase_c_batch.py`（subprocess分離）
- **総実験数**: 14（seed=42スクリーニング）
- **総実行時間**: 349分（5.8時間）
- **各実験**: 50Kステップ学習 + 50Kステップdeterministic評価
- **結果JSON**: `results/phase_c/c0_c1_20260209_114855_final.json`
- **メモリリーク防止**: subprocess分離、各実験プロセス終了時にメモリ完全解放

### 13.2 全実験結果テーブル

| # | Experiment | γ | Thr | MHP | Net ROI | Trades | Fees | Gross PnL | G2 Det. Trades |
|---|---|---|---|---|---|---|---|---|---|
| 1 | c0_baseline_p1 | 0.99 | 0.33 | 3 | -15.02% | 1,036 | 16,127 | **+1,111** | 0 |
| 2 | c1_gamma_080 | 0.80 | 0.33 | 3 | -15.02% | 870 | 13,856 | -1,162 | 0 |
| 3 | c1_gamma_090 | 0.90 | 0.33 | 3 | **-14.96%** | 946 | 15,224 | +261 | 0 |
| 4 | c1_gamma_095 | 0.95 | 0.33 | 3 | -15.01% | 962 | 15,287 | +276 | 0 |
| 5 | c1_threshold_50 | 0.99 | 0.50 | 3 | -15.02% | 862 | 14,098 | -925 | 0 |
| 6 | c1_threshold_60 | 0.99 | 0.60 | 3 | -15.00% | 780 | 12,745 | -2,259 | 0 |
| 7 | c1_threshold_70 | 0.99 | 0.70 | 3 | -15.01% | **742** | **11,837** | -3,176 | 0 |
| 8 | c1_g080_thr50 | 0.80 | 0.50 | 3 | -15.27% | 808 | 13,157 | -2,109 | 0 |
| 9 | c1_g080_thr60 | 0.80 | 0.60 | 3 | -15.01% | 822 | 13,301 | -1,713 | 0 |
| 10 | c1_g080_thr70 | 0.80 | 0.70 | 3 | -15.00% | 860 | 13,751 | -1,254 | 0 |
| 11 | c1_holding_5 | 0.99 | 0.33 | 5 | -15.04% | 1,080 | 17,013 | **+1,977** | 0 |
| 12 | c1_holding_10 | 0.99 | 0.33 | 10 | -15.01% | 950 | 15,267 | +256 | 0 |
| 13 | c1_holding_15 | 0.99 | 0.33 | 15 | -14.97% | 888 | 14,094 | -875 | 0 |
| 14 | c1_v451_golden | 0.80 | 0.33 | 3 | -15.03% | 880 | 13,905 | -1,124 | 0 |

### 13.3 Gate 2 KPI — 全実験 FAIL

**全14実験でdeterministic policy評価時のTrades=0。**

Gate 2 (0番§5.2) 基準に対し全指標が未達:

| 指標 | 基準 | 14実験の結果 |
|------|------|-------------|
| Net ROI | > 5% | **-14.96% ~ -15.27%**（全て負） |
| PF | > 1.20 | **1.0**（取引ゼロのため未定義） |
| Sharpe | > 1.0 | **0.0** |
| MaxDD | < 15% | **0.0%**（balance変動なし） |
| WinRate | > 35% | **0.0%** |

### 13.4 根本原因分析（C0+C1から判明した事実）

#### 発見1: Deterministic policyは「何もしない」を学習

50Kステップの学習後、`model.predict(obs, deterministic=True)` はすべての観測に対して**action valueが閾値未満**。つまり学習済みpolicyの最適解は「HOLD」（取引しない）。

**含意**: 学習中の1,000件前後の取引は全て**SACのエントロピー駆動の探索的行動**であり、学習された方策ではない。P1-1で「Gross PnL > 0」だったのは探索ノイズの偶然が方向に偏る場合に限定。

#### 発見2: ハイパラ調整の効果は「探索行動のフィルタリング」に過ぎない

| パラメータ変更 | Trades変化 | 実質効果 |
|---|---|---|
| γ: 0.99→0.80 | 1036→870 (-16%) | 短期視野で探索が保守化 |
| threshold: 0.33→0.70 | 1036→742 (-28%) | ランダムactionの閾値フィルタ |
| MHP: 3→15 | 1036→888 (-14%) | 連続取引の物理的ブロック |
| v451_golden (γ=0.80+V457) | 1036→880 (-15%) | 報酬スケール変更、効果なし |

→ いずれもNet ROIは-15%±0.3%で変動なし。**探索行動をフィルタしているだけでpolicy自体は学習されていない。**

#### 発見3: γ=0.80はGross PnLを悪化させる

91# H1仮説「γ=0.80がv451成功の鍵」は**現行環境では不適合**:
- γ=0.80: Gross PnL = **-1,162** (baseline +1,111 から2,273悪化)
- γ=0.90: Gross PnL = +261
- γ=0.95: Gross PnL = +276
- γ=0.99: Gross PnL = **+1,111**

→ 長期γの方がGross PnLが高い。v451時代のγ=0.80成功は環境条件（データ、報酬設計、特徴量）が異なった可能性が高い。

#### 発見4: MHP=5は逆効果

min_holding_period=5では取引が**増加**（1,036→1,080）。強制保有後の反動取引が増える。MHP=10/15で減少するが、方向精度（Gross PnL）も劣化。

### 13.5 Phase C 判定

| 判定 | 条件 | 結果 |
|------|------|------|
| ~~GO~~ | Net ROI > 0% && SAC > Momentum | — |
| ~~CONDITIONAL~~ | Net ROI > -5% && Gross PnL > 0 | — |
| **NO-GO** | Net ROI < -5% | **全14実験が-15%で該当** |

**C0+C1の結果は全てNO-GO。** ハイパラ/閾値/保有期間の調整では根本的な解決に至らない。

### 13.6 次ステップへの提言

SACが50Kステップで学習しない根本原因は**ハイパラではなく環境/報酬設計**にある:

1. **Deterministic policyが取引を学ばない** → 報酬信号が弱すぎる（`reward_scale=100.0` でも無効）
2. **Gross PnL>0はエントロピー探索の偶然** → policyが方向性を獲得していない
3. **C2(約定モデル)/C3(報酬2系統)に進む前に、学習可能性自体を確保する必要がある**

**Phase C 後半への推奨オプション**:

| オプション | 内容 | 根拠 |
|-----------|------|------|
| **C3-A** (推奨) | 報酬設計変更: `hold_penalty > 0` + `trade_reward > 0` で取引に正のインセンティブ | 現状「何もしない」が最適→取引へのインセンティブが不足 |
| **C3-B** | 行動空間変更: continuous→discrete 3択(BUY/SELL/HOLD) | 連続actionの閾値変換で情報損失 |
| **C3-C** | ステップ数拡大: 50K→200K | 学習時間が不足の可能性（ただし他の条件が同じなら効果薄） |
| **C-alt** | FastIntradayEnvV456への切替 | HeavyTradingEnvの設計問題を回避 |