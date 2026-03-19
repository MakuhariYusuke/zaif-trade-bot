# 496# 493/494 レビュー実装 & ブラインドスポット分析

| 項目 | 内容 |
|------|------|
| 日付 | 2026-03-20 |
| 対象 | 493# PHG Profit-First Fill Test Review, 494# PHG Verify & Action Plan |
| 種別 | 実装 + 分析 |
| 前提 | 495# retrain scheduler crash resilience 完了後 |

---

## 1. レビュー妥当性評価

### 493# PHG レビュー

**総合評価: 妥当（高品質）**

493# は 490#–492# のドキュメントを横断的にレビューし、fill_records データに基づく定量的根拠で問題を指摘した。

| 指摘 | 妥当性 | 理由 |
|------|--------|------|
| 参加率崩壊 (fill_rate=24.7%, PF=0.891) | ✅ 妥当 | 4328行中1070fillは定量的に正しい |
| Slow Fill Adverse Selection | ✅ 妥当 | quick<10s +0.238, mid 10-30s -0.684, slow≥30s -3.750 は明確な傾向 |
| Route-to-Kill Deadlock | ✅ 妥当 | 140回のskipループは実際にHaltリスク |
| Runtime Drift (P0認定) | ✅ 妥当 | 495# で crash resilience + auto-restart 実装済みだが根本は健在 |
| SAC・Composite Risk は時期尚早 | ⚠️ 部分的 | サイドカーは既に稼働中。ただし「まず既存Fix」の方針は正しい |

### 494# PHG アクションプラン

**総合評価: 妥当（即実装可能）**

3つの戦術的修正を Ho & Stoll (1981)、Avellaneda-Stoikov (2008) 等の理論的根拠とともに提示。

| 修正案 | 妥当性 | 実装判断 |
|--------|--------|----------|
| §2.1 Micro-Timeout TTL Cut | ✅ 採用 | wait 30→15s, sell 20→10s, requote 2→4 |
| §2.2 Inventory Recovery Skew | ✅ 採用 | kill gate bypass + wide offset (ceiling×2.0) |
| §2.3 Runtime Drift Cold Restart | ✅ 済 | 495# で auto-restart 実装完了 |

---

## 2. 実装内容

### 2.1 Micro-Timeout TTL Cut

**変更ファイル:** `configs/v460/fill_test.yaml`

```yaml
# Before → After
micro_timeout_wait_sec:      30 → 15   # buy TTL 半減
micro_timeout_wait_sec_sell: 20 → 10   # sell TTL 半減
micro_timeout_max_requote:    2 →  4   # requote試行倍増
```

**根拠:** slow fill (≥30s) の平均損益 -3.750 JPY/fill。TTL短縮で adverse selection を回避し、requote 増加で fill機会を確保。

### 2.2 Inventory Recovery Skew

**変更ファイル:** 6ファイル

| ファイル | 変更内容 |
|----------|----------|
| `fill_config.py` | `recovery_skew_enabled`, `recovery_skew_offset_mult` 追加 |
| `fill_test.yaml` | `recovery_skew_enabled: true`, `recovery_skew_offset_mult: 2.0` |
| `orchestrator_pre_cycle.py` | `CycleContext.recovery_skew` フィールド追加 |
| `orchestrator_balance.py` | deadlock skip → recovery skew bypass + wide offset |
| `orchestrator_mid_cycle.py` | `recovery_skew` をゲートアグリゲーターに伝播 |
| `cycle_gate_aggregator.py` | kill gate bypass + `toxicity_offset_mult` 最低保証 |

**動作フロー:**
1. `orchestrator_balance`: buy残高不足 + sell kill-gated (or 逆) を検出
2. 従来: `ROUTE_TO_KILL_DEADLOCK` → skip (参加率ゼロ)
3. 新規: `recovery_skew=True` → kill gate bypass + `offset_mult ≥ 2.0×`
4. `cycle_gate_aggregator`: Gate 4/5 (buy/sell dynamic kill) を recovery_skew で貫通
5. `toxicity_offset_mult` を最低 2.0× に強制 → 超ワイドスプレッドで安全に在庫清算

**フォールバック:** `recovery_skew_enabled: false` で従来の deadlock skip に戻る。

### 2.3 Drift Prevention Test 更新

`tests/unit/v460/test_336_yaml_code_drift_prevention.py` の allowlist に新規 config フィールドを追加。

---

## 3. ブラインドスポット分析

493#/494# が見落としている、または深掘り不足の領域を6点検出。

### BS-1: Sidecar Signal 信頼性 (中程度)

**状況:**
- Fill records: fresh=27, stale=28, error=17 → **失敗率 62.5%**
- `orchestrator_mid_cycle.py` で `missing` と `error` が同一カウンタに集約
- stale 信号時の confidence 自動減衰ロジックなし

**影響:** sidecar_offset_bps が stale データに基づく不正確な offset を生成しうる。

**推奨:** error/stale分離カウンタ + stale時 confidence 半減。ただし sidecar 自体が 487# P0 で導入されたばかりであり、データ蓄積を優先。

### BS-2: NO_FEASIBLE_QUOTE 初回原因未記録 (低)

**状況:**
- 3連続失敗で初めて `NO_FEASIBLE_QUOTE` を記録 (112回)
- 初回・2回目の制約崩壊原因がログから消失

**影響:** 分析時に「なぜ infeasible だったか」の初回トリガーが不明。

**推奨:** 直近3回の reason を circular buffer で保持。ただし 112 回は全体の 2.6% であり優先度低。

### BS-3: offset_stages 二重命名 (問題なし)

`offset_stages` (maker_price pipeline) と `executor_offset_stages` (executor 実行時) は意図的な二重記録。テスト済み (`test_421`).

### BS-4: spread_too_narrow ソフト閾値全無効 (高)

**状況:**
- `spread_too_narrow` = 393回 (全体の 9.1%)
- `skip_gate_narrow_spread_threshold_jpy = 0.0` (デフォルト無効)
- `skip_gate_narrow_spread_offset = 0.0` (デフォルト無効)
- ソフト閾値が YAML 未設定のため全てハード拒否

**影響:** spread が min_spread_jpy 境界付近の「通せたかもしれない」ケースが一律 skip。

**推奨:** 将来的に `skip_gate_narrow_spread_threshold_jpy` を有効化し、境界 spread をソフトゲート化すれば参加率向上の余地あり。fill_records の spread 分布分析が前提。

### BS-5: ranging_low_vol_skip Hard Mode 全開 (高)

**状況:**
- `ranging_low_vol_skip` = **718回 (全体の 16.6%)** — 最大の単一 skip 原因
- `skip_ranging_buy_low_vol: true` + `ranging_buy_low_vol_as_offset: false`
- ソフトモード (`as_offset=true`) の offset boost 実装は存在するが、ハードブロックが前段で通さない

**影響:** ranging + low vol 環境で全サイクル skip → 安定環境での参加機会を大量喪失。

**推奨:**
- `ranging_buy_low_vol_as_offset: true` に変更して試験
- ハードブロック → ソフトゲート (offset boost) への段階的移行
- **効果測定が容易** (Before/After の fill_rate 比較)

### BS-6: Requote × Recovery Skew 相互作用 (問題なし)

**検証結果:** `effective_offset_ratio` は offset pipeline 適用後の値（`toxicity_offset_mult` 含む）がそのまま requote ループで再利用される。recovery_skew の offset 強制は正しく requote に伝播している。

---

## 4. サマリー

### 実装済み (本セッション)

| # | 施策 | 期待効果 |
|---|------|----------|
| 2.1 | Micro-Timeout TTL Cut | slow fill adverse selection 回避 |
| 2.2 | Recovery Skew | deadlock skip 140回のうち相当数が fill 化 |
| 2.3 | Drift Prevention Test | CI gate 通過保証 |

### 未着手 (将来候補)

| # | 施策 | 優先度 | 備考 |
|---|------|--------|------|
| BS-1 | Sidecar stale/error 分離 | P2 | データ蓄積後に判断 |
| BS-2 | NO_FEASIBLE_QUOTE 原因保持 | P3 | 全体 2.6%、費用対効果低 |
| BS-4 | spread_too_narrow ソフト化 | P1 | spread 分布分析が前提 |
| BS-5 | ranging_low_vol ソフトモード | P1 | **16.6% の参加率回復** が期待大 |

### 次回アクション候補

1. **BS-5 ソフトモード移行**: `ranging_buy_low_vol_as_offset: true` → A/B テスト
2. **BS-4 spread 分析**: fill_records から spread 分布を抽出、ソフト閾値の適正値を算出
3. **BS-1 sidecar 品質改善**: stale 率 62.5% → signal TTL や refresh 頻度の見直し
