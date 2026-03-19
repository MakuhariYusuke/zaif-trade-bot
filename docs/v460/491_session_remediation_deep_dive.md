# 491# Session: Fill Pipeline Remediation & Defense Layer Deep Dive

**日付**: 2026-03-19  
**セッション**: 488#→491# レビュー修正対応・深堀り・sidecar 復旧

---

## §1 セッション概要

本セッションでは 488#/489#/490# レビュー指摘に基づく修正を実施し、
fill pipeline の防御レイヤー多重共線性問題を定量的に解析した。

### 実施内容

| # | 項目 | 状態 | 対象 |
|---|------|------|------|
| 1 | sidecar retrain_scheduler 復旧 | ✅ 完了 | ops/windows/retrain_scheduler.ps1, sac_retrain_scheduler.py |
| 2 | P0-1 VPIN boost 勾配有効化 | ✅ 完了 | configs/v460/fill_test.yaml |
| 3 | P1-6 Config 相互参照バリデーション | ✅ 完了 | fill_config_validation.py + test |
| 4 | P1-7 Offset clamping 統一 | ✅ 非バグ確認 | 421# で完了済み (resolve_offset_ceiling) |
| 5 | P1-4/P1-5 Reanchor/Cooldown | ✅ 設計確認 | 設計通り動作、チューニング対象 |
| 6 | 防御レイヤー多重共線性分析 | ✅ 完了 | 46+ gate の定量分析 |
| 7 | retrain_scheduler Timestamp 修正 | ✅ 完了 | sac_retrain_scheduler.py L296 |
| 8 | retrain_scheduler PYTHONPATH 修正 | ✅ 完了 | retrain_scheduler.ps1 |
| 9 | P0 訓練例外→信号stale防止 | ✅ 完了 | sac_retrain_scheduler.py (neutral fallback) |
| 10 | ゾンビ検出 venv限定フィルタ | ✅ 完了 | retrain_scheduler.ps1 |
| 11 | hot_swap config 不整合修正 | ✅ 完了 | hot_swap_restart.ps1 |
| 12 | hot_swap graceful shutdown | ✅ 完了 | hot_swap_restart.ps1 |
| 13 | 防御レイヤー実運用定量分析 | ✅ 完了 | ログベース fill rate / blocking 分析 |
| 14 | Composite Risk Score 実装 | ✅ 完了 | 490# Level 2: Soft Gate→weight加算集約 |

---

## §2 sidecar retrain_scheduler 復旧

### 問題

- 2026-03-11 以降 `cache/sidecar_signal.json` が 8 日間未更新 (stale)
- 6 個のゾンビ retrain_scheduler プロセスが `retrain_scheduler.py` (旧スクリプト) + `fill_test.yaml` (誤 config) で起動されていた
- DLL エラー: システム Python (3.11) の torch c10.dll ロード失敗

### 原因分析

1. **PYTHONPATH 欠落**: `Start-Process` で起動すると `sys.path[0]` がスクリプトディレクトリになり、プロジェクトルートの `ztb` パッケージが見つからない
2. **Timestamp 型不整合**: `_build_training_debug_details()` で `float(pd.Timestamp)` が TypeError

### 修正

- `retrain_scheduler.ps1`: `$env:PYTHONPATH = $ProjectRoot` を `Start-Process` の前に追加
- `sac_retrain_scheduler.py` L296-303: `float(timestamp)` → `ts.timestamp() if hasattr(ts, "timestamp") else float(ts)`

### 結果

- PID 85352 で正常起動、warm-start で `models/v460/sac_sidecar.zip` をロード
- 15,000 step 増分訓練が進行中
- signal file は訓練完了後に更新される

---

## §3 P0-1: VPIN Boost 勾配有効化

### 問題の定量分析

```
VPIN 統計 (直近200サイクル):
  平均: 0.69    最小: 0.42    最大: 0.84

旧設定: vpin_threshold=0.60, vpin_continuous_min=0.40
  → ランプ範囲: 0.40-0.60 (0.20幅)
  → VPIN 0.60+ で _norm=1.0 (飽和) → boost=2.0x 常時適用
  → 全サンプルが cont=1.00 を記録

新設定: vpin_threshold=0.80, vpin_continuous_min=0.40
  → ランプ範囲: 0.40-0.80 (0.40幅)
  → VPIN 分布全体をカバーする二次曲線
```

### boost 計算の変化

| VPIN | 旧 _norm | 旧 boost | 新 _norm | 新 boost | 勾配改善 |
|------|----------|----------|----------|----------|---------|
| 0.50 | 0.50 | 1.25x | 0.25 | 1.06x | ↓19% |
| 0.60 | 1.00 | 2.00x | 0.50 | 1.25x | ↓75% |
| 0.65 | 1.00 | 2.00x | 0.63 | 1.39x | ↓61% |
| 0.70 | 1.00 | 2.00x | 0.75 | 1.56x | ↓44% |
| 0.80 | 1.00 | 2.00x | 1.00 | 2.00x | 同等 |

### 即時効果 (ログ確認)

```
変更前: vpin=0.70(cont=1.00), vpin=0.80(cont=1.00)  ← 100%飽和
変更後: vpin=0.50(cont=0.25), vpin=0.72(cont=0.79)  ← 段階的勾配
```

### sell_dynamic_kill への期待効果

- 変更前: 12.5% ブロック率 (272 sell サイクル中 34 ブロック)
- offset 膨張が抑制されるため quote freshness 改善 → rolling PnL 改善 → kill 閾値到達頻度低下
- 24-48h 後の再測定が必要

---

## §4 防御レイヤー多重共線性分析 (490# Deep Dive)

### 定量結果

```
総サイクル要求: 4,120
実際の fill: 997
fill_rate: 24.2%
失敗: 3,123 (75.8%)

主要ブロッカー:
  sell_dynamic_kill:  979件 (最大)
  one_sided_*:        500+件
  ranging_low_vol:    ~400件
  unknown_regime:     ~300件
  spread_too_narrow:  ~250件
```

### 5つの多重共線性クラスタ

| クラスタ | 潜在因子 | 関連ゲート数 | 代表ゲート |
|---------|---------|-------------|----------|
| A: レジーム不確実性 | regime=unknown | 3-4 | unknown_regime_buy/sell, rule_skip_unknown |
| B: 流動性枯渇 | spread_bps | 5-6 | narrow_spread, SAD, no_feasible_quote, postonly |
| C: 市場ストレス | velocity_bps | 4-5 | ranging_low_vol, MCB, velocity_skip |
| D: 在庫偏重 | inv_imbalance | 3 | one_sided_freeze/cooldown, degraded_liquidation |
| E: 逆選択リスク | rolling_pnl | 3-4 | buy/sell_dynamic_kill, toxicity, cross_venue |

### 既存の部分統合

| 統合手法 | 対象 | 状態 | 範囲 |
|---------|------|------|------|
| Toxicity Budget (240#) | Gate 4/5 | 段階応答 | cycle_gate 内のみ |
| Alert Aggregation (211#) | MCB/SAD | 乗算型 | pre-cycle 内のみ |
| EV-Weighted (188#) | Skip Gate | 加算型 | skip_gate 内のみ |
| Soft Mode (195#/196#) | 各ゲート | 個別 offset_mult | 分散・統一なし |

### 改善方向性 (490# 推奨)

| Level | アプローチ | 期間 | 効果 |
|-------|----------|------|------|
| 1. 対症療法 | 閾値個別調整 | 即時 | 低 (多重共線性残存) |
| **2. 構造改善** | **Composite Risk Score + 単一閾値** | **2-4週** | **中** |
| 3. アーキテクチャ | SAC に gate 移行 | 1-3月 | 高 |

---

## §5 再学習装置 堅牢性強化 (Turn 3)

### 5.1 P0: 訓練例外時の信号stale防止

**問題**: `retrain_once()` が `ImportError` や汎用 `Exception` で失敗すると、
`_push_neutral_fallback()` が呼ばれず `sidecar_signal.json` が stale 状態のまま放置される。

**修正**: `sac_retrain_scheduler.py` — 両方の例外ハンドラに `_push_neutral_fallback()` 呼出しを追加。

```python
except ImportError as e:
    _push_neutral_fallback()  # ← 追加
    ...
except Exception as e:
    _push_neutral_fallback()  # ← 追加
    ...
```

### 5.2 ゾンビプロセス検出の精緻化

**問題**: `retrain_scheduler.ps1` の `Get-RetrainProcess` がシステム Python も検出し、
venv 外のプロセスをゾンビ誤判定。実際に PID 78540 (system Python) が検出されていた。

**修正**: `retrain_scheduler.ps1` — venv パス限定フィルタ。

```powershell
Where-Object {
    $_.CommandLine -like "*retrain_scheduler*" -and
    $_.CommandLine -like "*$($ProjectRoot.Replace('\\', '\\\\'))\\.venv*"
}
```

### 5.3 hot_swap_restart.ps1 config 不整合

**問題**: retrain_scheduler 再起動時に `$Config` (= fill_test.yaml) を渡していた。
本来は `g2_sac_train.yaml` が必要。

**修正**: ハードコード `configs/v460/experiments/g2_sac_train.yaml` に変更。

### 5.4 hot_swap graceful shutdown

**問題**: retrain_scheduler を `-Force` で即座に kill しており、
訓練中のモデル書き込みやチェックポイント保存が中断される可能性。

**修正**: 3段階停止シーケンス。

```
1. Stop-Process (通常) → _shutdown_event をトリガー
2. 15秒間隔で 3秒ごとにプロセス生存チェック
3. 残存時 → Stop-Process -Force
```

---

## §6 防御レイヤー実運用定量分析 (Turn 3)

### 6.1 直近ログ (613サイクル) の blocking 内訳

| カテゴリ | 件数 | 割合 | 性質 |
|---------|------|------|------|
| 残高不足 (JPY) | 409 | 最大 | 資金制約 (非バグ) |
| 残高不足 (BTC) | 266 | | 資金制約 (非バグ) |
| no_feasible_quote | 48 | 7.8% | cross_venue_lead_lag_veto burst |
| sell_dynamic_kill | 34 | 5.5% | VPIN 起因 offset 膨張 |
| unknown_regime | 19 | 3.1% | レジーム判定不確実 |
| buy_dynamic_kill | 19 | 3.1% | gate level blocker |
| ranging_low_vol | 14 | 2.3% | 低 vol 時の保護 |
| SAD/MCB | 0 | 0% | 正常動作 (非トリガー) |

### 6.2 Fill Rate

```
総サイクル:   613
発注済:       345 (56.5%)
約定:         222 (36.3% of total, 64.3% of placed)
```

### 6.3 重要な発見

1. **残高不足が最大のブロッカー** — JPY 255円 / BTC 0.000000 で繰り返しスキップ。
   資金追加無しではこれ以上の fill rate 改善は限定的。
2. **no_feasible_quote (48件)** — 100% が `cross_venue_lead_lag_veto` 起因。
   7回連続 burst パターンで NO_FEASIBLE_QUOTE (閾値3回) をトリガー。設計通りだが burst 頻度は注視対象。
3. **SAD/MCB は 0 トリガー** — スプレッド異常検知は正常域。過剰ブロックなし。
4. **sell_dynamic_kill (34件)** — VPIN threshold 修正 (0.60→0.80) により改善見込み。24-48h 後に再測定。

## §5 P1 課題ステータス

| ID | 項目 | 判定 | 理由 |
|----|------|------|------|
| P1-1 | cycle exception 分類 | ✅ 修正済 (18a7d66a8) | ConnectionError/TimeoutError → WARNING |
| P1-2 | サイクル結果ログ enrichment | 🟡 保留 | 技術的負債、優先度低 |
| P1-3 | プログレスログ欠落 | 🟡 保留 | 技術的負債、優先度低 |
| P1-4 | Reanchor budget 25bps | ✅ 設計確認 | 設計通り動作、チューニング課題 |
| P1-5 | Cooldown 段階的回復 | ✅ 設計確認 | rearm 機序 (249#) で対応済み |
| P1-6 | Config 相互参照 | ✅ 修正済 (本セッション) | VPIN min < threshold チェック追加 |
| P1-7 | Offset clamping 3重 | ✅ 非バグ確認 | 421# で resolve_offset_ceiling 統一済み |

---

## §6 コミット一覧

### 前セッション (489#-490# 対応)

| SHA | 内容 |
|-----|------|
| 5608149c6 | sidecar stat/read race + cross-venue 例外分類 |
| cc6c9466f | re-quote place_order kwargs 修正 |
| eddca6dc9 | hot_swap_restart.ps1 taskkill ErrorActionPreference |
| 17eb54d05 | 488# ドキュメント 489#/490# 補正反映 |
| 18a7d66a8 | P0-3 sigma_floor + P1-1 cycle exception + P0-2 非バグ確認 |

### 本セッション (491#)

| 対象ファイル | 変更内容 |
|-------------|---------|
| `configs/v460/fill_test.yaml` | vpin_threshold: 0.60→0.80 |
| `scripts/v460/ml/sac_retrain_scheduler.py` | Timestamp→float 変換修正 |
| `scripts/v460/lib/fill_config_validation.py` | VPIN min < threshold バリデーション追加 |
| `tests/unit/v460/test_346_fill_config_validation.py` | VPIN バリデーションテスト 3 件追加 |
| `ops/windows/retrain_scheduler.ps1` | PYTHONPATH 設定追加 |

---

## §7 Composite Risk Score 実装 (Turn 4: 490# Level 2)

### 7.1 設計概要

490# 指摘「Veto/Skip を個別 Boolean で判定するのではなく、連続値の複合リスクスコアに
合成し、一元的な閾値で判断する構造へ転換」を実装。

```
アーキテクチャ:
  Soft Gate (1,2,2b,3,6,7) → 連続値 risk_weight [0.0, 1.0] を返す
  Hard Gate (4,5,8,9)     → Boolean 短絡維持 (安全性ゲート)
  集約: Σ(risk_weight) >= threshold → block
```

### 7.2 ゲート分類と重み設定

| Gate | 分類 | Risk Weight | 理論的根拠 |
|------|------|-------------|-----------|
| 1 (unknown_buy) | Soft | 0.6 | Regime uncertainty → conditional EV negative |
| 2/2b (ranging_lv) | Soft | 0.5 | Low vol → spread capture insufficient |
| 3 (trending_sell) | Soft | 0.7 | Directional alpha forfeit (Glosten-Milgrom) |
| 4 (buy_kill) | **Hard** | N/A | Inverse selection cost → EV definitely negative |
| 5 (sell_kill) | **Hard** | N/A | Inverse selection cost → EV definitely negative |
| 6 (velocity) | Soft | 0.4 | Kyle λ imprecision → advisory level |
| 7 (unknown_sell) | Soft | 0.6 | Regime uncertainty (buy と対称) |
| 8 (narrow_spread) | **Hard** | N/A | Maker advantage vanishes (Roll 1984) |
| 9 (maker_price_pre) | **Hard** | N/A | Execution quality hard constraint |

### 7.3 動作モード

- `composite_risk_enabled=False` (既定): 従来 AND-chain を完全維持 (後方互換)
- `composite_risk_enabled=True`: Soft Gate の weight 加算、閾値 (1.5) 超過で block
- Hard Gate は composite mode でも即座に block (安全性担保)
- `halt_recovery_active=True` 時は Soft Gate の weight を蓄積しない

### 7.4 期待効果

```
現状 (AND-chain):
  Gate 1 blocked → 即 return (Gate 2-7 未評価)
  → 個別カウントでは blocking 少でも cascade rejection の可能性

Composite mode:
  Gate 1 weight=0.6 + Gate 7 weight=0.6 = 1.2 < 1.5 → 通過
  Gate 1 + Gate 6 + Gate 2 = 0.6+0.4+0.5 = 1.5 → block (複合リスク)
  → 「1つだけなら通す、3つ重なったら止める」の政策的判断が可能に
```

### 7.5 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| fill_config.py | composite_risk_* 設定 5 フィールド追加 |
| cycle_gate_aggregator.py | evaluate() リファクタ: Soft Gate→weight加算、Hard Gate→即block維持 |
| cycle_gate_aggregator.py | CycleGateResult に composite_risk_score/details 追加 |
| test_491_composite_risk_score.py | 15 テスト (disabled/enabled/hard/soft/combined) |
| test_336_yaml_code_drift_prevention.py | field_count 上限 450→470 |

### 7.6 テスト結果

```
57 passed (38 既存cycle_gate + 15 composite + 4 drift)
```

---

## §8 次回アクション

### 短期 (次セッション)

1. **Composite Risk Score 有効化**: fill_test.yaml に `composite_risk_enabled: true` 追加し A/B テスト
2. **sidecar signal 更新確認**: retrain_scheduler 訓練完了後の signal freshness を検証
3. **VPIN threshold 効果測定**: 24-48h 後の sell_dynamic_kill ブロック率変化
4. **fill_rate 回復度測定**: 36.3% → 目標 45%+ への改善幅

### 中期 (2-4週)

5. **Composite Risk Score チューニング**: weight/threshold のグリッドサーチ
6. **Gate 統合 PoC**: クラスタ B (spread 5-6 重) の統一 → 単一 liquidity_score

### 長期 (1-3月)

7. **SAC gate 移行**: 観測化 + 学習最適化 (490# Level 3)
8. **マルチシード ensemble**: seed42/123 の prediction averaging

---

## §9 ログ分析に基づく改善 (491# Turn 5)

**日付**: 2026-03-20
**対象ログ**: fill_test.log (cycle 12842–13559, ~717 cycles)

### §9.1 定量統計

| 指標 | 値 | 深刻度 |
|------|-----|--------|
| 総サイクル | 717 | — |
| 約定 (filled) | 269 (37.5%) | ⚠ 低 fill rate |
| 未約定 (unfilled) | 137 | — |
| PnL 平均 | **-2.728 bps** (600 fills) | 🔴 致命的 |
| PnL 勝率 | **22.8%** (137/600) | 🔴 致命的 |
| PnL 合計 | -1636.72 bps | 🔴 |
| Offset ceiling clamp | **478/717 = 67%** | 🟠 シグナル圧縮 |
| Pre-clamp offset avg | 0.2982 (min=0.201, max=0.45) | — |
| HARD SKIP | 34 | ⚠ 極端な市場 |
| Skip buy (JPY不足) | 461 | 🟠 残高危機 |
| Skip sell (BTC不足) | 285 | 🟠 残高危機 |
| route-to-kill deadlock | 28 | ⚠ |
| sell_dynamic_kill blocking | 多数 (trending_down regime) | — |
| Sidecar stale | 121 (TTL=7800s 超過) | 🟠 |
| Macro boost applied | 151 → 全て ceiling で clamp | 🟠 無効化 |
| Micro-timeout re-quote | 286 | ⚠ |
| Order not found | 75 | ⚠ API信頼性 |
| Post-only reject | 12 | — |
| SDK block | 24 | — |
| BDK block | 45 | — |

### §9.2 重大発見: 二重クランプ構造欠陥

**問題**: maker_price (306#) と offset_pipeline (421#) が同一の `resolve_offset_ceiling()` (0.20) を使用。

```
maker_price → base=0.05 → 9-stage boosts → 0.30 → [306# CLAMP → 0.20]
                                                         ↓
offset_pipeline → 0.20 × macro_boost(1.6) = 0.32 → [421# CLAMP → 0.20]
```

**影響**: offset_pipeline の全 multiplier (EV, velocity, macro_boost, toxicity, etc.) が **構造的に無効化**。
macro_boost が trending_down で 1.6× を適用しても、入力が既に ceiling で丸められているため
出力も ceiling に再クランプされる。67% のサイクルで信号が圧縮。

**PnL への影響**: 市場が trending_down 時、buy に対する保護 (= 広い offset) が ceiling で制限され、
不利な約定が増加。avg PnL = -2.728 bps という深刻な損失の主因の一つ。

### §9.3 改善措置

| # | 改善 | 変更 | 根拠 |
|----|------|------|------|
| 1 | Buy ceiling 引上げ | `offset_ceiling_ratio_buy: 0.20 → 0.25` | Pre-clamp avg=0.2982。0.25 で約 40% の clamp 解除。macro_boost(1.6×) → 0.40 → 0.25 clamp (元: 0.32 → 0.20) |
| 2 | Retrain scheduler 再起動 | PID 80936 (dead) → PID 81112 | WorkingSet=16KB = ゾンビ。6h 超 stale signal 解消 |
| 3 | Composite Risk Score 有効化 | `composite_risk_enabled: true` (threshold=1.5) | 490# Level 2。保守的閾値で段階的導入 |

### §9.4 二重クランプ問題: 将来的解決策

**現行**: 306# と 421# が同一 ceiling → multiplier chain が no-op。

**将来案 (P1)**:
- **案 A**: 306# ceiling を廃止し 421# final clamp のみに統一 → multiplier が unclamped 値で動作
- **案 B**: 306# を `max_offset_ratio` (0.30) に引上げ → intermediate exploration を許容
- **案 C**: regime-aware ceiling multiplier を追加 → trending 時のみ ceiling 緩和

いずれも A/B テスト + PnL 回帰分析が必要。current PnL = -2.73 bps は ceiling 以外にも
sidecar stale、残高不足 (JPY 306 / BTC 0.00232) が寄与。

### §9.5 残高危機による影響

- JPY=306, BTC=0.00232 → buy は実質不可 (461 skip)、sell は trade size 制約
- route-to-kill deadlock: buy 不可 + sell が kill-gated → 両サイド完全停止 (28 events)
- **残高補充が PnL 改善の前提条件**。offset ceiling 改善だけでは構造的問題は解決しない。

---

## §10 Deep-Dive 定量分析 (Turn 6)

3/19 fill_test ログ (717 cycles, 271 fills) の多角的分析結果。

### §10.1 時間帯別 PnL

| UTC | JST | fills | avg_bps | worst_single | 評価 |
|-----|-----|-------|---------|-------------|------|
| 00h | 09h | 27 | **-8.35** | -80.57 | 最悪帯。朝の流動性薄い時間に被弾 |
| 22h | 07h | 18 | **-6.71** | -42.97 | 早朝、同様に薄い |
| 03h | 12h | 20 | -4.17 | -28.74 | 昼前後 |
| 06h | 15h | 1 | -21.48 | -21.48 | 単発大損 |
| 11h | 20h | 1 | -21.14 | -21.14 | 単発大損 |
| 18h | 03h | 12 | **+0.44** | -10.14 | 唯一プラス圏 |
| 16h | 01h | 24 | -0.06 | -18.04 | ほぼ損益分岐 |
| 19h | 04h | 15 | -0.10 | -10.02 | ほぼ損益分岐 |

**観察**: 00h (JST 09h) と 22h (JST 07h) が突出して悪い。東京市場オープン前後の薄い板に
逆選択される構図。`hour_ceiling_mult` による deep-night 保護は JST 01-04h をカバーするが
07-09h は未保護。

### §10.2 Buy/Sell 別 PnL

| side | fills | avg_bps | win | win_rate |
|------|-------|---------|-----|----------|
| buy  | 138   | -0.578  | 61  | 44.2%    |
| sell | 133   | -0.790  | 77  | 57.9%    |

sell は win_rate 57.9% にも関わらず avg PnL が悪い → 負けたときの損失が大きい (fat tail)。
buy は win_rate 44.2% と低い → 方向バイアス or ceiling 制約の影響。

### §10.3 逆選択分析 (Adverse Selection)

約定までの待ち時間と PnL の相関を分析。

| 区間 | fills | avg_bps | win | win_rate |
|------|-------|---------|-----|----------|
| **quick** (<10s) | 131 | **+0.273** | 70 | 53.4% |
| mid (10-30s) | 110 | -0.983 | 53 | 48.2% |
| **slow** (>=30s) | 30 | **-3.750** | 15 | 50.0% |

**重要発見**: 約定待ち時間と PnL に明確な負の相関。

- **10秒未満の即時約定はプラス圏** (+0.273 bps) — 市場が自分方向に動いている
- **30秒以上待つ約定は大損** (-3.750 bps) — 典型的な逆選択パターン
- マーケットメイクの教科書的結論: 長時間板に晒される注文は informed trader に狩られる

**対策案**:
- order TTL (time-to-live) を 20-30s に制限し、stale order を自動キャンセル
- 30s 以上経過でスプレッドを自動拡大 (offset 引き上げ)
- velocity multiplier と組み合わせ、急変時は即座にキャンセル

### §10.4 Offset Pipeline ボトルネック分析

fill_records_20260319.jsonl からパイプラインステージを抽出。

```
base=0.05 → regime_adj=0.09 → spread_adapt=0.18 → vol_guard=0.30
  ↓ (max_offset_ratio=0.30 で天井到達!)
306# ceiling_clamp=0.20 → velocity×2.25 → ExPre=0.449
  ↓ (421# final_clamp)
final_offset=0.20
```

**構造的問題**:
1. vol_guard (0.18→0.30) で max_offset_ratio=0.30 に到達する確率が高い
2. 306# ceiling (0.20) で 0.30→0.20 にクランプ → regime/spread/vol 情報消失
3. velocity multiplier (avg 1.73×, 82 events) が ceiling 後に適用 → ExPre=0.449
4. 421# final_clamp が ExPre→0.20 に再クランプ → velocity 情報も消失
5. 結果: 5段階の signal → 全て 0.20 に圧縮 = **情報理論的に 0 bit**

### §10.5 Ceiling 0.25 影響定量化

ceiling_ratio_buy を 0.20→0.25 に引上げた効果を事前定量。

| pre-clamp 範囲 | 件数 | 割合 |
|---------------|------|------|
| 0.20 - 0.22 | 12 | 2.6% |
| 0.22 - 0.25 | 23 | 5.1% |
| 0.25 - 0.30 | 29 | 6.4% |
| **0.30+** | **391** | **85.9%** |

- 新 ceiling 0.25 で clamp 解除: **35/455 = 7.7%** のみ
- 依然 **420/455 = 92.3%** が ceiling でクランプ
- 85.9% が max_offset_ratio=0.30 で先に飽和 → ceiling 問題の上流に vol_guard 問題あり

**結論**: ceiling 0.25 の効果は限定的。根本は vol_guard→max_offset_ratio=0.30 飽和。

### §10.6 3日間クランプ vs 非クランプ PnL 比較

**驚くべき発見**: ceiling はむしろ保護的に作用。

| 状態 | fills | avg_pnl_bps |
|------|-------|-------------|
| **Clamped** (ceiling 適用) | 702 | **-0.189** |
| Unclamped (ceiling 未到達) | 814 | -0.338 |

Clamped fills の方が損失が小さい。これは ceiling が「スプレッドを適正範囲内に保つ」
ことで、広すぎる offset による機会損失 (fill されない → トレンド追随不能) を防いでいるため。

ただし velocity multiplier (82 events, avg 1.73×) は二重 clamp で完全に無効化されており、
「急変時の保護情報」が 0 bit になっている問題は残存。

### §10.7 Composite Risk Hot-Reload 修正

`composite_risk_enabled` を含む 6 フィールドが `_HOT_RELOADABLE_FIELDS` に
未登録であったため、YAML 変更が実行時に反映されなかった。

**修正**: `config_hot_reload.py` に以下を追加:
```python
# --- 491# Composite Risk Score ---
"composite_risk_enabled",
"composite_risk_threshold",
"composite_risk_weight_unknown_regime",
"composite_risk_weight_ranging_low_vol",
"composite_risk_weight_trending_sell",
"composite_risk_weight_velocity",
```

### §10.8 次段階の改善提案 (Priority Queue)

| 優先 | 項目 | 根拠 | 効果期待 |
|------|------|------|---------|
| **P0** | Order TTL 導入 (20-30s) | §10.3 逆選択: slow fill = -3.75bps | 逆選択損失の大幅削減 |
| P1 | 306# ceiling 廃止 or max_offset に統一 | §10.4-10.5 二重 clamp = 0 bit | velocity/macro_boost 情報復元 |
| P1 | 07-09h (JST) hour_ceiling_mult 拡張 | §10.1 00h=-8.35bps | 朝方の逆選択保護 |
| P2 | vol_guard 飽和対策 | §10.5 85.9% が 0.30 到達 | offset resolution 改善 |
| P2 | sell fat-tail 損失制限 | §10.2 sell win=57.9% but avg=-0.79 | 負けトレード幅の制限 |
