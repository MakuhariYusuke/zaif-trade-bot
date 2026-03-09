# 364# SHA別分析レポート + TUNE-3 SDK閾値緩和

| 項目 | 値 |
|---|---|
| 文書番号 | 364# |
| フェーズ | ph2 (G1.1-exec) |
| 前提 | 360#, 361# F7, 363# B3 |
| 作成 | Copilot 038 |
| ステータス | **ACTIVE** |

---

## §1 SHA 819ec73b2081 再集計結果 (2026-03-09 最新 Bot)

`tools/reaggregate_by_sha.py --sha 819ec73b2081` による current-SHA 限定集計。

> **Note**: レコードは継続蓄積中。以下は 2026-03-10 時点のスナップショット。

### §1.1 総合指標

| 指標 | 値 | Gate判定 |
|---|---|---|
| Total records | 238 | — |
| Filled | 91 | — |
| Skipped (skip_gate) | 46 | — |
| Attempted | 192 (=238-46) | — |
| **K1 attempted_fill_rate** | **47.4%** | **FAIL** (≥60%) |
| **K2 cancel_ratio** | **52.6%** | **FAIL** (≤40%) |
| PnL mean | -0.108 bps | — |
| PnL median | -0.183 bps | — |

### §1.2 Side別内訳

| Side | Total | Filled | Skip | Attempted | Fill Rate (att) |
|---|---|---|---|---|---|
| Buy | 99 | 45 | 23 | 76 | **59.2%** |
| Sell | 134 | 46 | 23 | 111 | **41.4%** |

**Sell側がK1低下のボトルネック。** Buy は 59.2% で Gate 閾値 (60%) に近接。

### §1.3 PnL分布

| Side | PnL mean | PnL median |
|---|---|---|
| Buy | +0.233 bps | — |
| Sell | -0.072 bps | — |

PnL percentiles (全 filled, n=87):

| p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|
| -8.64 | -6.57 | -4.27 | -0.18 | +2.94 | +8.26 | +13.45 |

mean: -0.029 bps, std: 7.190 bps

### §1.4 Cancel理由 (Side別)

**Buy cancel (attempted 中非 fill = 31件):**

| Reason | 件数 |
|---|---|
| stale_adverse_drift | 13 |
| spread_too_narrow | 12 |
| postonly_crossing_skip | 2 |
| status_unknown_fast | 2 |
| post_only_reject | 2 |

**Sell cancel (attempted 中非 fill = 65件):**

| Reason | 件数 |
|---|---|
| **sell_dynamic_kill** | **39** |
| spread_too_narrow | 17 |
| stale_adverse_drift | 5 |
| status_unknown_fast | 2 |
| post_only_reject | 1 |
| postonly_crossing_skip | 1 |

> skip_gate (buy: 23, sell: 23) は attempted から除外済み。上記は attempted-but-not-filled のみ。

### §1.5 SDK Cancel の Regime 分布

| Regime | SDK Cancel 件数 | 現行閾値 | Effective (max inv_relax) |
|---|---|---|---|
| **ranging** | **23** | -0.5 bps | -1.0 bps |
| **trending_up** | **16** | -0.3 bps | -0.8 bps |
| trending_down | 0 | -1.0 bps | -1.5 bps |

### §1.6 Regime別パフォーマンス

| Regime | Total | Filled | Fill% | Avg PnL |
|---|---|---|---|---|
| ranging | 199 | 77 | 39% | -0.22 bps |
| trending_up | 30 | 5 | 17% | +3.40 bps |
| trending_down | 5 | 5 | 100% | -0.56 bps |

**特記**: trending_up は Fill率が極端に低い (17%) が、filled 時の PnL は +3.40 bps と最良。
SDK kill (16件) が trending_up 利益機会を大量にブロックしている。

---

## §2 比較SHA分析

| SHA | K1 | PnL mean | 特徴 |
|---|---|---|---|
| 819ec73b2081 (最新) | 46.15% | +0.077 bps | TUNE-1解消後、K1改善 |
| 0d22298c5e7e (旧) | 33.51% | +1.215 bps | forced_buy_delay 残存、低fill高PnL |

K1 は旧SHAから +12.6pp 改善 (TUNE-1 = forced_buy_delay 撤廃の効果)。
PnL は低下したが、これは fill 数増加による分母効果。

---

## §3 TUNE-3 実装: SDK 閾値緩和

### §3.1 根拠

1. **SDK が sell 側最大キャンセル要因** (39件 / sell attempted 111件 = 35.1%)
2. **ranging で 23 SDK kill**: 現行 -0.5 bps (effective -1.0 at max inv_relax) でも過剰発動
3. **trending_up で 16 SDK kill**: 利益機会 (+3.40 bps/fill) をブロック
4. **360# TUNE-3 推定**: K1 寄与 +7.0pp (K1 47% → ~54%)

### §3.2 変更内容

F7 制約 (361#): `ewma_alpha`, `ewma_time_decay_tau_sec` は **変更しない** (time decay と threshold 緩和の同時投入禁止)。

閾値のみ変更：

| パラメータ | Before | After | Effective (max relax) | 根拠 |
|---|---|---|---|---|
| threshold_bps (default) | -0.3 | **-0.5** | -1.0 | 360# TUNE-3 提案値 |
| regime: trending_up | -0.3 | **-0.5** | -1.0 | default と同期。16 SDK kill 解消狙い |
| regime: ranging | -0.5 | **-0.7** | -1.2 | 23 SDK kill (最大バケット)。-0.2 bps 緩和 |
| regime: trending_down | -1.0 | -1.0 (据置) | -1.5 | 0件 SDK kill。変更不要 |

### §3.3 リスク評価

- **逆選択リスク**: 閾値緩和により toxic fill が増加する可能性
  - 緩和幅は -0.2 bps/regime と保守的
  - inv_relaxation は非線形 (imbalance=0 → offset=0)。max_relax は高在庫偏重時のみ
  - ewma_alpha=0.05 (window≈20) の感度は維持 → 持続的損失は依然捕捉
- **想定効果**: K1 +5～+10pp (SDK 39件の一部解消)
- **検証方法**: デプロイ後 168h → `reaggregate_by_sha.py` で再計測

### §3.4 変更ファイル

| ファイル | 変更 |
|---|---|
| `configs/v460/fill_test.yaml` | threshold_bps + regime_thresholds 更新 |
| `scripts/v460/lib/fill_config.py` | デフォルト値更新 (一貫性) |
| `tests/unit/v460/test_169_c1_c3_c4_config.py` | アサーション値更新 |
| `tests/unit/v460/test_336_fill_config_parser.py` | パーサーテスト値更新 |

---

## §4 残存課題ステータス (360-363# 統合)

### P0 (Week1 / 即時)
| ID | 説明 | ステータス | 担当 |
|---|---|---|---|
| OPS-5/A1 | 本番 Task Scheduler IgnoreNew 確認 | ⏳ pending | ops/user |
| OPS-1/A2 | atexit hook RSS/状態ダンプ | ✅ 完了 | codex 037-074 (`4da98356a`) |
| OPS-2 | health_monitor 300→60s | ✅ 完了 | codex 037-075 (`2fba1020d`) |
| OPS-4 | restart.lock stale 30s→120s | ✅ 完了 | codex 037-075 (`2fba1020d`) |
| OPS-6 | Start-Process 後 lock 確認 | ✅ 完了 | codex 037-075 (`2fba1020d`) |

### P1 (Week2)
| ID | 説明 | ステータス | 担当 |
|---|---|---|---|
| **TUNE-3/B3** | **SDK threshold 緩和** | **✅ 本文書で実装** | copilot 038 |
| TUNE-2 | per_side_dd_halt -30→-50 bps | ⏳ pending | codex 038 (C-4) |
| TUNE-4 | BDK threshold 緩和 | ⏳ skip (0件 BDK) | codex 038 (C-5 note) |
| B2 | current-SHA 再集計→K1/K2 判定 | ✅ §1 で実施 | copilot 038 |
| P1-1 | buy_ranging deep dive | ⏳ pending | codex |
| B4/P1-3 | ph3 sidecar 設計文書 | ✅ 365# で作成 | copilot 038 |

### P2 (Week3+)
| ID | 説明 | ステータス | 担当 |
|---|---|---|---|
| GATE-1/C3 | K1 gate 60%→40% + K4 PnL≥0 | ⏳ pending | user判断 |
| C1 | G2 holdout 評価→ph3 SAC 実行 | ⏳ pending | codex |
| C2/G3 | Micro-price 実装 | ⏳ pending | codex |
| C4/F3 | env microstructure proxy | ⏳ pending | codex |

---

## §5 次ステップ

1. **即時**: OPS-5 (本番 Task Scheduler 確認) — user 作業
2. **Codex 038**: C-4 (TUNE-2), C-5 (TUNE-4 skip note), C-6 (uncommitted commit) — `prompts/codex_038_tune2_tune4.md`
3. **168h 後**: TUNE-3 デプロイ SHA で `reaggregate_by_sha.py` 再計測
4. **K1 再判定**: TUNE-3 効果確認後、GATE-1 (閾値改訂) の要否を決定
