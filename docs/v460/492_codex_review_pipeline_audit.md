# 492# Codex レビュー: Fill Pipeline 総合監査

**日付**: 2026-03-20
**対象**: 491# セッション全修正 + 実運用ログ定量分析  
**目的**: Codex による多角的レビュー・批判的検証

---

## §1 エグゼクティブサマリー

### 1.1 現行システム状態

| 指標 | 値 | 評価 |
|------|-----|------|
| 累計 cycle | 13,614 | — |
| clean records (quarantine除外) | 12,469 | — |
| quarantine (blank git_sha) | 1,145 (8.4%) | ⚠ |
| 残高 | JPY ≈269 / BTC ≈0.00232 | 🔴 危機的 |
| Regime (最新) | trending_down, stability=2 | — |
| Sidecar signal | stale (10:35 UTC, ~6h超) | 🟠 |
| Retrain scheduler | PID 86000 起動済、子PID 72904 訓練中 (WS=438MB) | ✅ |
| Fill test | PID 23072 (01:40 restart), hard_skip UTC 16h | ✅ |
| soft_loss_cap | triggered (cumPnL -1,211 JPY) | 🔴 |
| Daily PnL (3/20) | -24.30 bps (buy +27.22, sell -51.52) | 🟠 |

### 1.2 本セッション (491#) コミット一覧

| SHA | 内容 | 影響範囲 |
|-----|------|---------|
| `07487339a` | VPIN threshold 0.60→0.80 + sidecar scheduler修正 + config validation | gate tuning + retrain infra |
| `5641e6730` | retrain堅牢性4件 + 防御レイヤー定量分析 | retrain lifecycle |
| `e179fbd01` | Composite Risk Score 実装 (490# Level 2) | gate architecture |
| `5498d1385` | buy ceiling 0.20→0.25 + composite risk YAML有効化 | offset tuning |
| `cfea47214` | §10 deep-dive分析 + composite_risk hot-reload修正 | hot-reload fix |

### 1.3 主要な発見 (重大度順)

| # | 発見 | 重大度 | 根拠 |
|---|------|--------|------|
| 1 | **逆選択**: quick fill +0.273bps vs slow fill -3.75bps | 🔴 P0 | Glosten-Milgrom逆選択。30s+の注文は10倍以上の損失 |
| 2 | **二重クランプ = 0 bit**: 5段階のoffset signal が全て 0.20 に圧縮 | 🔴 P0 | 306# + 421# 同一ceiling → velocity/macro/toxicity情報が消失 |
| 3 | **Ceiling 0.25 は 7.7% しか解放しない**: 85.9% が vol_guard で先に飽和 | 🟠 P1 | 上流の vol_guard→max_offset_ratio=0.30 が真のボトルネック |
| 4 | **Ceiling はむしろ保護的**: clamped=-0.189bps vs unclamped=-0.338bps | 🟡 意外 | 広すぎる offset → fill されず → 機会損失 |
| 5 | **Composite Risk がまだ未稼働**: hot-reload未登録 → 修正済だが再起動必要 | 🟠 P1 | config_hot_reload.py 修正は cfea47214 でコミット済 |
| 6 | **Sell fat-tail**: win 57.9% but avg -0.79bps → 負け幅が大きい | 🟠 P1 | 勝率高いが一敗の損失が大きすぎる |
| 7 | **07-09h JST 未保護**: -8.35bps (UTC 00h), -6.71bps (UTC 22h) | 🟠 P1 | hour_ceiling_mult は JST 01-04h のみ |

---

## §2 Offset Pipeline アーキテクチャ監査

### 2.1 パイプライン全体像

```
[maker_price.compute()] ─ 13 stages ─┐
                                      ├→ 306# ceiling clamp
                                      │   ceil = resolve_offset_ceiling(side, utc_hour)
                                      │   offset_ceiling_ratio_buy: 0.25 (was 0.20)
                                      ▼
                      MakerPriceResult(price, spread, effective_offset_ratio)
                                      │
[offset_pipeline._apply()] ─ 9 stages ┤
  1. EV offset (193#)                 │
  2. Velocity offset (195#)           │
  3. Trending sell (196#)             │
  4. Toxicity (240#)                  │
  5. VG supplement (202# C)           │
  6. Macro trend boost (458# F)       │
  7. Alert mode (215#)                │
  8. Sidecar offset (372# F1)         │
  9. → 421# FINAL CLAMP              │
                                      ▼
                      OffsetPipelineResult(order_price, effective_offset_ratio)
```

### 2.2 二重クランプの実例 (fill_records_20260319.jsonl)

```
入力:
  base=0.05 → regime_adj=0.09 → spread_adapt=0.18 → vol_guard=0.30
  ↓ (max_offset_ratio=0.30 天井到達!)

306# ceiling_clamp:
  0.30 → clamp to 0.20 (旧) / 0.25 (新)
  ↓

Executor multipliers:
  velocity × 2.25 → ExPre=0.449
  ↓

421# final_clamp:
  0.449 → clamp to 0.20 (旧) / 0.25 (新)
  ↓

出力: 0.20 (旧) / 0.25 (新)

結論: 5段階の市場情報 → 単一値に圧縮 = 情報理論的 0 bit
```

### 2.3 Offset 分布 (3/19, n=72 fills)

| offset 範囲 | 件数 | 割合 |
|-------------|------|------|
| 0.13 - 0.15 | 1 | 1.4% |
| 0.15 - 0.195 | 23 | 31.9% |
| **0.195 - 0.205** | **48** | **66.7%** |
| 0.205+ | 0 | 0% |

**66.7% が ceiling 0.20 に張り付き** — ceiling 0.25 への変更は 3/20 01:01 hot-reload で反映済。
効果測定は 24h 後に実施。

### 2.4 Ceiling 0.25 効果シミュレーション (§10.5 再掲)

| pre-clamp 範囲 | 件数 | 割合 |
|---------------|------|------|
| 0.20 - 0.22 | 12 | 2.6% |
| 0.22 - 0.25 | 23 | 5.1% |
| 0.25 - 0.30 | 29 | 6.4% |
| **0.30+** | **391** | **85.9%** |

新 ceiling 0.25 での clamp 解除: **7.7% のみ**。根本は vol_guard 飽和。

### 2.5 レビュー指摘事項

| # | 指摘 | 重大度 | 対象ファイル | 行 |
|---|------|--------|-------------|-----|
| A1 | 306# と 421# が `resolve_offset_ceiling()` を共有 → 二重クランプが設計上不可避 | 🔴 | maker_price.py L1020, offset_pipeline.py L314 | |
| A2 | `_effective_max_ratio()` (405#) が `max(base, ceiling)` を返す → intermediate stage が ceiling 超過を許容 → final clamp 依存 | 🟡 | fill_config.py ~L405 | |
| A3 | Executor の `_recalc_price_with_new_offset()` が mid-cross を生む可能性 | 🔴 | offset_pipeline.py L346 | 要ユニットテスト |
| A4 | `early_return_record` の caller 側チェックの有無が未確認 | 🔴 | offset_pipeline.py L361 | |
| A5 | `spread_at_order=None` 時に offset ratio と price が不整合 | 🟡 | offset_pipeline.py L348 | fail-open vs skip 判断 |
| A6 | Maker `_stages` と Executor `executor_offset_stages_json` が別々 → 監査証跡が分断 | 🟡 | cross-file | FillRecord 統合案 |

### 2.6 改善案

| 案 | 内容 | リスク | ROI |
|----|------|--------|-----|
| **A**: 306# ceiling 廃止 | 421# final clamp のみに統一。maker は `max_offset_ratio` まで自由 | 中 (intermediate 暴走) | 高 |
| **B**: 306# ceiling を max_offset_ratio に引上げ | 0.20→0.30。intermediate exploration 許容 | 低 | 中 |
| **C**: regime-aware ceiling | trending 時のみ ceiling 緩和 (e.g., ×1.5) | 低 | 中 |
| **D**: 監査証跡統一 | `offset_audit_log: List[Tuple]` を FillRecord に追加 | 低 | 低 |

---

## §3 Cycle Gate Aggregator 監査

### 3.1 ゲート分類

| Gate | 名称 | 分類 | Composite Weight | 理論根拠 |
|------|------|------|-----------------|---------|
| 1 | unknown_regime_buy | **Soft** | 0.6 | Regime uncertainty → conditional EV negative |
| 2 | ranging_buy_low_vol | **Soft** | 0.5 | Low vol → spread capture insufficient |
| 2b | ranging_sell_low_vol | **Soft** | 0.5 | Gate 2 の sell 対称 |
| 3 | trending_sell | **Soft** | 0.7 | Glosten-Milgrom 方向 alpha 放棄 |
| 4 | buy_dynamic_kill | **Hard** | N/A | 逆選択コスト → EV 確実にマイナス |
| 5 | sell_dynamic_kill | **Hard** | N/A | 逆選択コスト → EV 確実にマイナス |
| 6 | velocity_skip | **Soft** | 0.4 | Kyle λ 精度不足 → advisory level |
| 7 | unknown_regime_sell | **Soft** | 0.6 | Gate 1 の sell 対称 |
| 8 | narrow_spread_pause | **Hard** | N/A | Roll (1984) 有効スプレッド消失 |
| 9 | maker_price_precheck | **Hard** | N/A | 実行品質ハード制約 |

### 3.2 累計 Gate Fire Counts (13,614 cycles)

| Gate | 累計 | 全 cycle 比 |
|------|------|------------|
| gate_sell_dynamic_kill | **948** | 7.0% |
| gate_ranging_low_vol_skip | **723** | 5.3% |
| gate_buy_dynamic_kill | **512** | 3.8% |
| toxic_veto_set | 446 | 3.3% |
| per_side_halt_switch | 375 | 2.8% |
| balance_forced_halt_block | 335 | 2.5% |
| forced_buy_delay | 300 | 2.2% |
| degraded_liquidation_duty_skip | 150 | 1.1% |
| degraded_liquidation_active | 145 | 1.1% |
| route_to_kill_deadlock | **140** | 1.0% |
| preflight_insufficient | 100 | 0.7% |

**sell_dynamic_kill が最大のブロッカー** (948件)。VPIN threshold 0.60→0.80 修正の効果測定が必要。

### 3.3 Composite Risk Score 設計レビュー

#### 現状

- `composite_risk_enabled=True` を YAML 設定済 (5498d1385)
- `_HOT_RELOADABLE_FIELDS` に 6 フィールド追加済 (cfea47214)
- ただし **botの実行コードが cfea47214 時点のコードを使用していない**
  - 01:40 restart 時の git_sha=cfea47214 → コード反映済み ✅
  - hot-reload で `composite_risk_enabled` が True に反映されているか → 要確認

#### Composite モード動作フロー

```
evaluate() 開始
  ├─ Gate 1 (soft): weight=0.6 → score+=0.6 (AND: early return)
  ├─ Gate 2 (soft): weight=0.5 → score+=0.5 (AND: early return)
  ├─ Gate 3 (soft): weight=0.7 → score+=0.7 (AND: early return)
  ├─ Gate 4 (hard): → blocked=True, return (composite でも即 block)
  ├─ Gate 5 (hard): → blocked=True, return (同上)
  ├─ Gate 6 (soft): weight=0.4 → score+=0.4 (AND: early return)
  ├─ Gate 7 (soft): weight=0.6 → score+=0.6 (AND: early return)
  ├─ Gate 8 (hard): → blocked=True, return
  └─ Gate 9 (hard): → blocked=True, return
  
  score >= 1.5 → blocked=True, reason="composite_risk_exceeded"
```

**例**:
- Gate 1(0.6) + Gate 6(0.4) = 1.0 < 1.5 → **通過** (AND-chain では block)
- Gate 1(0.6) + Gate 2(0.5) + Gate 6(0.4) = 1.5 → **block** (複合リスク)
- Gate 3(0.7) + Gate 7(0.6) = 1.3 < 1.5 → **通過** (AND-chain では block)

### 3.4 レビュー指摘事項

| # | 指摘 | 重大度 | 詳細 |
|---|------|--------|------|
| B1 | **Weight 較正にエビデンスなし** | 🔴 | [0.4, 0.5, 0.6, 0.7] は経験的。バックテスト/A/Bテスト結果の記載なし |
| B2 | **Threshold=1.5 で `>=` 判定** | 🟡 | ちょうど 1.5 で block。意図的か？ `>` の方が適切な場合もある |
| B3 | **hot-reload 中のセマンティクス変更** | 🔴 | `composite_risk_enabled` をサイクル実行中に変更すると、同一サイクル内でゲート評価の意味論が変わる |
| B4 | **halt_recovery_active 時は soft gate 蓄積を完全スキップ** | 🟡 | 在庫回復中は composite scoring を完全無視 → 寛容すぎる可能性 |
| B5 | **Weight の合計バリデーションなし** | 🟡 | 全weight発火: 0.6+0.5+0.7+0.4=2.2。threshold を 0.5 に誤設定すると過剰 block |
| B6 | **composite_risk_score が executor に伝播しない** | 🟡 | FillRecord に score/details はあるが、executor は未参照 |

---

## §4 PnL 定量分析

### 4.1 日別 PnL トレンド (3日間)

| 日付 | records | fills | avg_pnl_bps | win_rate | adv_sel | avg_wait_s |
|------|---------|-------|-------------|----------|---------|-----------|
| 3/17 | 635 | 109 | **-0.840** | 50% | 30 (28%) | 12.1 |
| 3/18 | 453 | 108 | **-0.125** | 52% | 31 (29%) | 10.2 |
| 3/19 | 206 | 72 | **-0.783** | 56% | 24 (33%) | 22.1 |

**傾向**:
- 3/18 は比較的良好 (-0.125bps)、3/17・3/19 は悪化
- 3/19 は wait time が倍増 (22.1s vs 10.2s) → 逆選択リスク増大
- 逆選択率は微増傾向 (28%→29%→33%)

### 4.2 逆選択分析 (3/19, n=72)

| 区間 | fills | avg_bps | win_rate |
|------|-------|---------|----------|
| **quick** (<10s) | 27 | **+2.522** | 56% |
| mid (10-30s) | 18 | -0.613 | 61% |
| **slow** (>=30s) | 27 | **-4.202** | 52% |

**3/19 は quick fill で +2.522bps と大幅改善** (全期間の +0.273bps より良好)。
slow fill は -4.202bps と悪化。逆選択パターンがより鮮明に。

### 4.3 サイド別 PnL (3/19, n=72)

| side | fills | avg_bps | win | win_rate |
|------|-------|---------|-----|----------|
| buy | 36 | **+0.227** | 20 | 55.6% |
| sell | 36 | **-1.793** | 20 | 55.6% |

**buy 側が初めてプラス圏** (+0.227bps)。sell の fat-tail 問題は継続。

### 4.4 時間帯別 PnL (3/19)

| UTC | JST | fills | avg_bps | worst |
|-----|-----|-------|---------|-------|
| 09h | 18h | 9 | -0.081 | -9.15 |
| 10h | 19h | 14 | -0.177 | -6.54 |
| 11h | 20h | 14 | **+2.477** | -7.28 |
| 12h | 21h | 13 | **+3.684** | -13.77 |
| 13h | 22h | 10 | **-9.345** | -72.65 |
| 14h | 23h | 6 | -3.001 | -19.42 |
| 15h | 00h | 6 | -4.050 | -23.81 |

**UTC 13h (JST 22h) で -72.65bps の極端な外れ値** → 1件で全体 PnL を大幅悪化。
UTC 11-12h (JST 20-21h) は安定してプラス圏。

### 4.5 3/19 のキーイベント

| 時刻 (JST) | イベント |
|------------|--------|
| ~14:24 | hot-reload #1: `loss_cap_jpy` 更新 |
| ~19:20 (04:20 JST) | hot-reload #2: `vpin_threshold 0.6→0.8` 反映 |
| ~01:01 (10:01 JST) | hot-reload #3: `offset_ceiling_ratio_buy 0.2→0.25` 反映 |
| ~01:40 (10:40 JST) | watchdog restart → git_sha=cfea47214 反映 |

---

## §5 Hot-Reload / Config 監査

### 5.1 `_HOT_RELOADABLE_FIELDS` の網羅性

300+ フィールドが `_HOT_RELOADABLE_FIELDS` に登録済み。

**最近追加 (491#):**
```python
"composite_risk_enabled",
"composite_risk_threshold",
"composite_risk_weight_unknown_regime",
"composite_risk_weight_ranging_low_vol",
"composite_risk_weight_trending_sell",
"composite_risk_weight_velocity",
```

### 5.2 Hot-reload で変更不可なフィールド (意図的に除外)

| フィールド | 理由 |
|-----------|------|
| `skip_gate_model_path` | モデルロード → メモリ再配置必要 |
| `skip_utc_hours` / `skip_utc_hours_buy/sell` | リスト型 → パース安全性 |
| `regime_lot_multipliers` | Dict 型 → パース安全性 |
| `regime_timeout_multipliers` | Dict 型 → 同上 |

### 5.3 レビュー指摘事項

| # | 指摘 | 重大度 | 詳細 |
|---|------|--------|------|
| C1 | **composite_risk_enabled toggle がサイクル中に変更されうる** | 🔴 | `maybe_reload()` は cycle 間で呼ばれるが、タイミングによっては評価途中に変更 |
| C2 | **Weight バリデーションなし** | 🟡 | `composite_risk_weight_*` の範囲チェックなし。0→10.0 に誤設定可能 |
| C3 | **`hour_ceiling_mult` の key バリデーションなし** | 🟡 | `{25: 2.0}` (不正な UTC hour) を silently 無視 |
| C4 | **Config update 失敗時のフォールバック** | 🟡 | 旧値を維持するが、警告ログがオペレータに伝搬する保証なし |

---

## §6 Retrain Scheduler 監査

### 6.1 現行状態

| 項目 | 値 |
|------|-----|
| Scheduler PID | 86000 (01:33 起動) |
| 訓練子プロセス PID | 72904 (WS=438MB, CPU=2919s) |
| 最終 signal | 2026-03-19 10:35 UTC (stale ~6h) |
| Model version | sac_sidecar_20260319_1015 |
| 前回訓練時間 | 1156.5s (~19min) |
| Warm-start | ✅ sac_sidecar.zip + buffer.pkl |

### 6.2 セッション中の修正事項

| # | 修正 | コミット | 影響 |
|---|------|---------|------|
| 1 | Timestamp→float TypeError 修正 | 07487339a | retrain_scheduler 起動失敗を防止 |
| 2 | PYTHONPATH 設定追加 | 07487339a | `ztb` パッケージ import 失敗を防止 |
| 3 | 訓練例外時 neutral fallback | 5641e6730 | signal stale 防止 |
| 4 | ゾンビ検出 venv 限定フィルタ | 5641e6730 | 誤検出防止 |

### 6.3 懸念事項

| # | 指摘 | 重大度 |
|---|------|--------|
| D1 | 訓練完了後の signal deploy がログに出ていない → 訓練が正常終了するか要監視 | 🟠 |
| D2 | 前回 PID 81112 は ~10min で死亡。原因は OOME か例外か不明 | 🟡 |
| D3 | signal stale 中のボット動作 — sidecar offset が 0 or neutral → fallback 動作は正常だが警告ログ多発 | 🟡 |

---

## §7 残高・損益管理

### 7.1 残高状態

```
JPY:  ≈269 (buy 実質不可、min_order=500円以上)
BTC:  ≈0.00232 (≈$200相当)
soft_loss_cap: triggered (cumPnL -1,211 JPY <= cap -516 JPY)
```

### 7.2 残高制約によるブロック

| ブロック種別 | 累計 |
|------------|------|
| balance_forced_halt_block | 335 |
| preflight_insufficient | 100 |
| forced_buy_delay | 300 |
| route_to_kill_deadlock | 140 |

**140 件の route_to_kill deadlock** — buy 不可 + sell が kill-gated → 両サイド完全停止。
残高補充なしではこの問題は解決不可能。

### 7.3 Daily PnL (3/20, warmup from fill records)

```
Total: -24.30 bps
  Buy:  +27.22 bps (数少ない buy が好結果)
  Sell: -51.52 bps (fat-tail 問題)
halted: False
```

---

## §8 改善提案 (Priority Queue)

### P0 (即時対応)

| # | 項目 | 根拠 | 実装難易度 | 期待効果 |
|---|------|------|----------|---------|
| P0-1 | **Order TTL 導入** (20-30s) | §4.2: slow fill = -4.2bps, quick = +2.5bps | 中 | 逆選択損失の大幅削減 |
| P0-2 | **Composite Risk 動作確認** | §3.3: git_sha反映済だが実動作未確認 | 低 | gate semantics 改善 |

### P1 (短期)

| # | 項目 | 根拠 | 実装難易度 | 期待効果 |
|---|------|------|----------|---------|
| P1-1 | **306# ceiling 統一** (廃止 or max_offset) | §2.2: 二重 clamp = 0 bit | 中 | velocity/macro_boost 情報復元 |
| P1-2 | **07-09h JST hour_ceiling_mult 拡張** | §4.4: UTC 00h = -8.35bps | 低 | 朝方の逆選択保護 |
| P1-3 | **Sell fat-tail 損失制限** | §4.3: sell avg=-1.79bps | 中 | 負けトレードの損失幅制限 |
| P1-4 | **Weight 較正の A/B テスト** | §3.4 B1: エビデンスなし | 中 | Composite Risk の信頼性向上 |
| P1-5 | **`_recalc_price_with_new_offset()` ユニットテスト** | §2.5 A3: mid-cross 可能性 | 低 | 安全性担保 |

### P2 (中期)

| # | 項目 | 根拠 |
|---|------|------|
| P2-1 | vol_guard 飽和対策 | §2.4: 85.9% が 0.30 到達 |
| P2-2 | Offset 監査証跡統一 | §2.5 A6: maker + executor が分断 |
| P2-3 | Config validation 強化 | §5.3 C2-C3: weight/hour バリデーション |
| P2-4 | Composite risk hot-reload race condition 対策 | §5.3 C1 |

---

## §9 コードレビュー対象ファイル索引

Codex レビュー時の参照先。

| ファイル | 行範囲 | レビューポイント |
|---------|--------|----------------|
| `scripts/v460/lib/cycle_gate_aggregator.py` | L187-400 | evaluate() — composite vs AND-chain 分岐 |
| `scripts/v460/lib/cycle_gate_aggregator.py` | L220-245 | Soft gate weight 蓄積ロジック |
| `scripts/v460/lib/cycle_gate_aggregator.py` | L401-414 | Composite threshold 判定 (`>=`) |
| `scripts/v460/lib/maker_price.py` | L1020-1042 | 306# ceiling clamp |
| `scripts/v460/lib/offset_pipeline.py` | L314-365 | 421# final clamp + hard skip |
| `scripts/v460/lib/offset_pipeline.py` | L346 | `_recalc_price_with_new_offset()` — mid-cross 可能性 |
| `scripts/v460/lib/fill_config.py` | L835-849 | composite_risk_* フィールド定義 |
| `scripts/v460/lib/fill_config.py` | L430-450 | `resolve_offset_ceiling()` 共通ヘルパー |
| `scripts/v460/lib/config_hot_reload.py` | L39-449 | `_HOT_RELOADABLE_FIELDS` 許可リスト |
| `scripts/v460/lib/config_hot_reload.py` | L501-510 | composite_risk 追加箇所 (491#) |
| `tests/unit/v460/test_491_composite_risk_score.py` | 全体 | 15 テスト: disabled/enabled/hard/soft/combined |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | 全体 | ドリフト防止テスト |

---

## §10 テスト状態

### 10.1 491# テスト結果

```
57 passed (cycle_gate 38 + composite_risk 15 + drift 4)
19 passed (drift 4 + composite_risk 15) — 直近実行
```

### 10.2 テストカバレッジ懸念

| # | 未テスト領域 | リスク |
|---|-------------|--------|
| T1 | `_recalc_price_with_new_offset()` edge cases (tiny spread, squeezed mid) | 🔴 |
| T2 | `early_return_record` が caller で正しくチェックされるか | 🔴 |
| T3 | Composite risk + halt_recovery_active の組み合わせ | 🟡 |
| T4 | hot-reload mid-cycle race condition | 🟡 |
| T5 | `hour_ceiling_mult` 不正キー (e.g., 25) の挙動 | 🟡 |

---

## §11 Codex へのレビュー依頼事項

以下の観点で批判的レビューを依頼:

1. **アーキテクチャ**: 二重クランプ (306# + 421#) の設計妥当性。廃止/統一の判断
2. **Composite Risk Score**: weight 較正、threshold 値、`>=` vs `>` 判定の妥当性
3. **逆選択対策**: Order TTL 導入の実装方針 (cancel loop vs expiry timer)
4. **Hot-reload 安全性**: composite_risk_enabled の mid-cycle toggle リスクと対策
5. **Sell fat-tail**: 損失制限の具体的手法 (stop-loss offset? dynamic kill threshold 引下げ?)
6. **残高制約下での最適戦略**: JPY 269 / BTC 0.00232 で取れる戦略的選択肢
7. **テスト不足領域**: §10.2 の未テスト項目の優先度判定
8. **ログ分析**: 3日間の PnL トレンドから読み取れる市場構造変化の兆候

---

**コミット対象**: 本ドキュメント
**次セッション**: Codex レビュー結果に基づく改善実装
