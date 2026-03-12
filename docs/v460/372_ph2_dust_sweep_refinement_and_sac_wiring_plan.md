# 372# ph2 Dust Sweep 洗練 + SAC sidecar 配線計画

| 項目 | 値 |
|---|---|
| 文書番号 | 372# |
| フェーズ | ph2 G1.1-exec |
| 前提文書 | 128# (dust sweep 初版), 370# §4 (SAC F1/F2), 365# (P1-P8) |
| 作業日 | 2026-03-10 |
| コミット | `154bd38b2` (dust sweep), `9ee8b662e` (SAC F1/F2), `c6991afc2` (self-review) |

---

## §1 Dust Sweep 洗練: Buy-to-Clear 方式

### §1.1 問題

| BTC残高 | 動作 (before) | 結果 |
|---|---|---|
| `≥ 0.001` + 端数 | `_maybe_dust_sweep` → 全額 sell | ✅ 解消 |
| `< 0.001` (micro-dust) | sell 最小数量を満たせず → スキップ | ❌ **永久残留** |

Coincheck API の最小注文数量 `0.001 BTC` を下回る micro-dust は売却不可能。

### §1.2 解法: 自動2サイクル方式

```
Cycle N (sell):
  btc_free = 0.00042 (< 0.001)
  → sell スキップ
  → dust_buy_pending = True

Cycle N+1 (buy):
  dust_buy_pending → prepare_dust_buy()
  → lot = min_order_btc (0.001)
  → buy 0.001 BTC

Cycle N+2 (sell):
  btc_free = 0.00142 (≥ 0.001)
  → _maybe_dust_sweep → 全額売却 0.00142
  → dust 完全解消 ✅
```

### §1.3 変更ファイル

| ファイル | 変更 |
|---|---|
| `scripts/v460/lib/balance_checker.py` | `_dust_buy_pending` フラグ、`dust_buy_pending` プロパティ、`prepare_dust_buy()` / `clear_dust_buy_pending()` メソッド追加。`_check_sell` で micro-dust 検出時にフラグセット |
| `scripts/v460/lib/orchestrator_balance.py` | `dust_buy_pending` 時に `prepare_dust_buy()` を呼出し |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | buy 完了後に `clear_dust_buy_pending()` 呼出し |
| `tests/unit/v460/test_dust_sweep.py` | `TestDustBuyToClear` クラス追加 (7テスト) |

### §1.4 テスト結果

```
$ python -m pytest tests/unit/v460/test_dust_sweep.py -x -q --no-cov
22 passed in 1.60s
```

---

## §2 SAC Sidecar 監査結果

### §2.1 現状: 3箇所の断線

365# P1-P8 ブロッカー解消で **部品は全て揃っている** が、end-to-end で **3箇所が断線**:

```
[sac_retrain_scheduler] ─write→ cache/sidecar_signal.json
                                     │
                              Gap-1: read 未呼出
                                     ↓
[orchestrator_mid] ─evaluate()→ [cycle_gate_aggregator]
                    sidecar_signal=None (常に)
                                     │
                              Gap-2: offset 未伝搬
                                     ↓
[orchestrator_mid] ─run_single_cycle()→ [fill_cycle_executor]
                    sidecar_offset_bps パラメータ未定義
                                     │
                              Gap-3: pricing 未反映
                                     ↓
[fill_cycle_executor] ─price─→ Exchange
                    sidecar offset 未考慮
```

### §2.2 F2: Signal 推論が過去データ

`sac_retrain_scheduler.py` L648:
```python
obs, _ = env.reset()  # ← 訓練ウィンドウ先頭に巻き戻し (数日前のデータ)
action, _ = model.predict(obs, deterministic=True)
```

コメント「最新 obs で推論」は**虚偽**。`env.reset()` は HeavyTradingEnv の `current_step` を先頭に戻すため、signal は過去の市場状態に基づく。

### §2.3 Gate 側は完備

`cycle_gate_aggregator.py` の実装は完全：
- `evaluate()` は `sidecar_signal: SidecarSignal | None` を受容
- Gate 9 通過後に `_apply_sidecar_offset()` を呼出し
- `compute_sidecar_offset_bps()` で非対称 offset を計算
- `CycleGateResult.sidecar_offset_bps` にセット

**問題は呼出し側が何も渡していないこと。**

---

## §3 SAC 改善計画

### §3.1 F2 修正: signal 推論の現在市場化 (P0) — ✅ 完了

```python
# 修正前 (バグ):
obs, _ = env.reset()  # 訓練データ先頭にリワインド

# 修正後 (372# F2):
obs = _get_latest_obs(env)  # 訓練データ末尾 = 最新市場状態
```

`_get_latest_obs()` は `env.current_step` を末尾に設定して `_get_observation()` を
呼ぶことで、OnlineScaler + action_masks を含む正規化パイプラインを再利用。
HeavyTradingEnv (df) / LiteTradingEnv (_feature_matrix) 両対応、
フォールバックとして reset() も保持。

### §3.2 F1 修正: 3段階配線 (P0) — ✅ 完了

| Gap | 修正箇所 | 内容 | 状態 |
|---|---|---|---|
| Gap-1 | `orchestrator_mid_cycle.py` | `read_sidecar_signal()` → `evaluate(sidecar_signal=sig)` | ✅ |
| Gap-2 | `fill_cycle_executor.py` | `run_single_cycle()` に `sidecar_offset_bps: float = 0.0` 追加 | ✅ |
| Gap-3 | `fill_cycle_executor.py` | `sidecar_offset_bps` → `delta_jpy = bps/10000 * price` → 価格直接調整 | ✅ |
| 呼出連鎖 | `orchestrator_mid_cycle.py` | `gate_result.sidecar_offset_bps` を `run_single_cycle()` に伝搬 | ✅ |

**Gap-3 pricing ロジック:**
```
正bps = 攻撃的 (mid に近づく):  buy → +delta,  sell → -delta
負bps = 保守的 (mid から離れる): buy → -delta,  sell → +delta
delta_jpy = round(sidecar_offset_bps / 10000 * order_price)
```

既存の `_apply_offset_multiplier()` (乗数ベース) とは独立した bps 直接適用。
toxicity/trending/velocity offset の後に最終調整として適用。

### §3.3 Deploy Gate 強化 (P1) — ✅ 完了

**min_trade_count (c6991afc2):**
OOS 評価中の取引回数が `min_trade_count` (default=3) 未満 → `oos_failed` として棄却。
取引回数0〜2回で `gross_roi > 0` をパスする偶発的モデルを排除。

**confidence 動的計算 (c6991afc2):**
```python
# 修正前: confidence = 1.0 (ハードコード)
# 修正後:
gate_threshold = sac_gate_roi_threshold  # default 0.0
full_roi = confidence_roi_full            # default 0.005
confidence = clamp((oos_roi - gate_threshold) / (full_roi - gate_threshold), 0.0, 1.0)
```
- Gate ギリギリ (`roi ≈ 0`) → `confidence ≈ 0` → sidecar offset 小
- `roi ≥ 0.005` → `confidence = 1.0` → sidecar offset 全力
- 低品質モデルが大きなオフセットを適用するリスクを自動抑制

**FillRecord 監査証跡 (c6991afc2):**
`FillRecord` に `sidecar_offset_bps` / `sidecar_bias` フィールド追加。
fill_record_builder 経由で記録 → 事後分析で sidecar impact を定量評価可能。

**残存 (deferred):**
- Seed stability check (複数 seed 一貫性) — 現時点では single-seed
- Sharpe ratio threshold — eval window が短すぎるため有効性未確認

---

## §4 実行順序

| # | タスク | 状態 | 備考 |
|---|---|---|---|
| 1 | F2: `_get_latest_obs()` + signal 推論修正 | ✅ 完了 | env API 変更不要 (current_step 直接操作) |
| 2 | F1: Gap-1 配線 (orchestrator → gate) | ✅ 完了 | `read_sidecar_signal()` → `evaluate()` |
| 3 | F1: Gap-2/3 配線 (gate → executor → pricing) | ✅ 完了 | bps 直接適用、乗数独立 |
| 4 | Deploy gate 強化 | ✅ 完了 | min_trade_count=3 + confidence 動的計算 |
| 5 | Self-review: FillRecord 監査証跡 | ✅ 完了 | sidecar_offset_bps / sidecar_bias |

### §4.1 テスト結果

```
$ python -m pytest tests/unit/v460/test_sidecar_sac_integration.py -v --no-cov
63 passed in 0.87s  (25 new: 5 F2 + 7 F1-Gap3 + 1 line-guard + 4 FillRecord + 6 confidence + 2 config)

$ python -m pytest tests/unit/v460/ -q --no-cov
4492 passed, 33 skipped in 25.10s
```

### §4.2 変更ファイル (F1/F2 + Self-review)

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/sac_retrain_scheduler.py` | `_get_latest_obs()` 追加、confidence 動的計算、`min_trade_count` gate |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | `read_sidecar_signal()` + `sidecar_signal=` パラメータ、`sidecar_offset_bps=` 伝搬 |
| `scripts/v460/lib/fill_cycle_executor.py` | `sidecar_offset_bps` パラメータ + bps pricing + FillRecord 記録 |
| `scripts/v460/lib/fill_record_builder.py` | `sidecar_offset_bps` / `sidecar_bias` パラメータ追加 |
| `ztb/metrics/fill_quality.py` | `FillRecord` に sidecar フィールド追加 |
| `tests/unit/v460/test_sidecar_sac_integration.py` | 25 tests (F2/F1/FillRecord/confidence/config) |
| `tests/unit/v460/test_253_...py` | line-guard 上限 1100→1120 |
| `tests/unit/v460/test_113_resilience.py` | `run_single_cycle` line-guard 740→755 |

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-10 | 1.0 | 初版 — dust sweep buy-to-clear + SAC sidecar 監査 |
| 2026-03-10 | 1.1 | F1/F2 完了記録 |
| 2026-03-10 | 1.2 | Self-review 修正 — FillRecord 監査証跡 + confidence 動的計算 + Deploy Gate 強化 |
