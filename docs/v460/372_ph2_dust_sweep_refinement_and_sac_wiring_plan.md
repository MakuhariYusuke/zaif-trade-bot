# 372# ph2 Dust Sweep 洗練 + SAC sidecar 配線計画

| 項目 | 値 |
|---|---|
| 文書番号 | 372# |
| フェーズ | ph2 G1.1-exec |
| 前提文書 | 128# (dust sweep 初版), 370# §4 (SAC F1/F2), 365# (P1-P8) |
| 作業日 | 2026-03-10 |
| コミット | `154bd38b2` |

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

### §3.1 F2 修正: signal 推論の現在市場化 (P0)

```python
# 現状 (バグ):
obs, _ = env.reset()

# 修正案:
# 最新の feature row から observation を構築
latest_obs = env.build_latest_observation(feature_source)
action, _ = model.predict(latest_obs, deterministic=True)
```

**依存**: `LiteTradingEnv` / `HeavyTradingEnv` に `build_latest_observation()` API 追加が必要。

### §3.2 F1 修正: 3段階配線 (P0)

| Gap | 修正箇所 | 内容 |
|---|---|---|
| Gap-1 | `orchestrator_mid_cycle.py` | `read_sidecar_signal()` → `evaluate(sidecar_signal=sig)` |
| Gap-2 | `fill_cycle_executor.py` | `run_single_cycle()` に `sidecar_offset_bps: float = 0.0` パラメータ追加 |
| Gap-3 | `fill_cycle_executor.py` | `sidecar_offset_bps` を price に反映 (bps → JPY → 加算) |

### §3.3 Deploy Gate 強化 (P1)

現在: `gross_roi > 0` のみ。
改善案:
- Seed stability check (複数 seed で一貫した方向性)
- Worst-window 検証 (最悪期間の損失が許容範囲内)
- Sharpe ratio threshold

---

## §4 実行順序

| # | タスク | 依存 | 工数 |
|---|---|---|---|
| 1 | F2: `build_latest_observation()` 実装 + signal 推論修正 | なし | 2-3h |
| 2 | F1: Gap-1 配線 (orchestrator → gate) | F2 | 0.5h |
| 3 | F1: Gap-2/3 配線 (gate → executor → pricing) | Gap-1 | 1-2h |
| 4 | Deploy gate 強化 | F1/F2 | 2-4h |

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-10 | 1.0 | 初版 — dust sweep buy-to-clear + SAC sidecar 監査 |
