# 163# audit: 姑息的手段 (Stopgap/Palliative Measures) 総カタログ

**作成日**: 2026-02-25  
**目的**: Fill Test システム内の「姑息的手段」(band-aid fix) を網羅的に洗い出し、根本対策 vs 対症療法を分類する。

---

## サマリー

| カテゴリ | 件数 | 姑息度 | 影響度 |
|----------|------|--------|--------|
| 1. 時間帯フィルター / デッドロック防止 | 4 | ★★★ | 高 (49%アイドル→改善済だが構造問題残存) |
| 2. Sell 側防御ガード | 5 | ★★☆〜★★★ | 高 (sell fill_rate 36.7% vs buy 58.7%) |
| 3. Balance forced skip | 3 | ★★★ | 最高 (21.9% = 377件がキャンセル) |
| 4. Spread ガード | 3 | ★☆☆ | 低〜中 |
| 5. Stale データガード | 2 | ★☆☆ | 低 (27件/1.6%) |
| 6. その他防御ゲート | 3 | ★★☆ | 中 |

---

## 1. 時間帯フィルター / デッドロック防止

### 1-A. TimeFilter — 静的時間帯遮断

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/lib/time_filter.py](../../scripts/v460/lib/time_filter.py) (全体) |
| **設定** | [configs/v460/fill_test.yaml](../../configs/v460/fill_test.yaml) `time_filter:` セクション |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L102) `enable_time_filter`, `skip_utc_hours`, `skip_utc_hours_buy`, `skip_utc_hours_sell` |
| **呼び出し** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L623) `_is_time_filtered()` → L1454-L1547 の main loop |
| **概要** | 特定 UTC 時間帯を静的にブロック。buy: UTC 8,16,18 / sell: UTC 4,8,14 / グローバル: UTC 16 |
| **判定** | **姑息的** ★★★ — 統計平均 PnL が悪い時間帯を一律遮断。市場状況に応じた動的判断ではなく、過去データの静的ルール。107# でも「動的ゲーティングへの移行」が提案済み。 |

### 1-B. 086# デッドロック防止 (max_086_consecutive_wait)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1500-L1538) |
| **設定** | `time_filter.max_086_consecutive_wait: 3` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L218) |
| **Cancel Reason** | `CR.TIME_FILTER_086_DEADLOCK` |
| **概要** | time_filter で片側のみ filtered の場合、alt_side == last_side で連続待機が蓄積し 49% アイドルとなるデッドロック。3回(6分)超過で代替 side を強制許可。 |
| **判定** | **姑息的** ★★★ — time_filter 自体が静的ルールのため、その副作用を別の閾値パラメータで対処。デッドロック→カウンタ制限→強制解除は二重の対症療法。 |

### 1-C. TIME_FILTER_BOTH_SIDES (両側遮断)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1454-L1493) |
| **Cancel Reason** | `CR.TIME_FILTER_BOTH_SIDES` |
| **概要** | buy/sell 両方がフィルタ対象の場合、heartbeat を出力しつつスリープ。 |
| **判定** | **姑息的** ★★☆ — 両側遮断時に完全停止する設計。VolatilityGuard 等の動的判断で代替すべき。 |

### 1-D. Heartbeat during time_filter

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1473-L1493) |
| **設定** | `tuning.heartbeat_interval_sec: 900` |
| **概要** | time_filter 抑制中に 15 分間隔で heartbeat ログ出力。プロセス生存確認。 |
| **判定** | やや姑息（time_filter 自体の必要性に依存）だが、監視としては妥当。 |

---

## 2. Sell 側防御ガード

### 2-A. trending_sell_skip (トレンド sell スキップ)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1775-L1833) |
| **設定** | `loss_control.skip_sell_trending: true`, `skip_sell_trending_up_only: true` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L232-L234) |
| **Cancel Reason** | `CR.TRENDING_SELL_SKIP` |
| **統計** | 10 日間: 233 件 (13.5%) |
| **概要** | trending_up レジーム時に sell を一律スキップ。trending_down は開放 (156# D-4)。 |
| **判定** | **姑息的** ★★★ — 「上昇トレンド中は sell が損」という統計に基づくが、一律遮断は機会損失を生む。offset 調整やスプレッド拡大で対処すべき。159# レビューでも指摘済み。 |

### 2-B. max_consecutive_trending_sell_skip (安全弁)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1798-L1813) |
| **設定** | `止血.max_consecutive_trending_sell_skip: 30` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L236) |
| **概要** | trending_sell_skip の連続スキップが 30 回に達したら売りを強制許可する安全弁。 |
| **判定** | **姑息的** ★★★ — 姑息的手段 (2-A) のさらなる安全弁 = 二重の対症療法。trending_sell_skip 自体が不要になれば消える。 |

### 2-C. sell_dynamic_kill (sell 動的停止)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1860-L1880) |
| **モジュール** | [ztb/risk/sell_dynamic_kill.py](../../ztb/risk/sell_dynamic_kill.py) (`SellDynamicKillManager`) |
| **設定** | `loss_control.sell_dynamic_kill.enabled: true`, `threshold_bps: -0.5`, `resume_window: 10` |
| **Cancel Reason** | `CR.SELL_DYNAMIC_KILL` |
| **統計** | 10 日間: 92 件 (5.3%) |
| **概要** | 直近 50 sell fill の rolling PnL が -0.5bps 以下で sell を自動停止。レジーム別閾値あり（trending_up: -0.3, trending_down: -1.0）。cooldown 10 cycle 後に再評価。 |
| **判定** | **準根本対策** ★★☆ — rolling PnL によるフィードバック制御は方向性として正しいが、sell が構造的に不利な原因（AS リスク、offset 設計）を解決していない。症状抑制型。 |

### 2-D. sell_guard (スプレッド/offset 下限ガード)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/lib/maker_price.py](../../scripts/v460/lib/maker_price.py#L290-L304) (sell_offset_floor), L296-L304 (max_spread_jpy) |
| **設定** | `sell_guard.max_spread_jpy: 4000.0`, `sell_guard.offset_floor: 0.10` |
| **Cancel Reason** | `CR.SELL_GUARD_REJECT` |
| **統計** | 10 日間: 16 件 (orderbook_error 内に 161 件混入、多くが sell_guard 関連) |
| **概要** | sell 側に max_spread_jpy (4000 JPY 超スキップ) と offset_floor (0.10 最低保証) を設定。 |
| **判定** | **姑息的** ★★☆ — ハードコード閾値でのスプレッド遮断。なぜ sell で spread が広い時に損になるかの根因（在庫偏り、AS メカニズム）を対処していない。 |

### 2-E. buy_dynamic_kill (buy 動的停止 — sell との対称版)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1836-L1855) |
| **モジュール** | [ztb/risk/sell_dynamic_kill.py](../../ztb/risk/sell_dynamic_kill.py) (`BuyDynamicKillManager`) |
| **設定** | `loss_control.buy_dynamic_kill.enabled: true`, `threshold_bps: -0.8`, `resume_window: 10` |
| **Cancel Reason** | `CR.BUY_DYNAMIC_KILL` |
| **概要** | sell_dynamic_kill と対称的に buy 側 rolling PnL を監視。buy は構造的 AS リスクが低いため閾値は sell より寛容 (-0.8bps)。 |
| **判定** | **準根本対策** ★★☆ — 対称設計は良いが、同じく症状抑制。 |

---

## 3. Balance Forced Skip (残高強制切替スキップ)

### 3-A. skip_balance_forced (残高強制切替時の発注抑制)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1683-L1743) |
| **設定** | `loss_control.skip_balance_forced: true` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L223) |
| **Cancel Reason** | `CR.BALANCE_FORCED_SKIP` |
| **統計** | 10 日間: **377 件 (21.9%)** — **最大の損失源** |
| **概要** | 残高不足で side が強制切替された場合、そのサイクルをスキップ（平均 -1.98bps の損失回避）。 |
| **判定** | **姑息的** ★★★ — 在庫偏りが発生してから事後的にスキップする設計。根本は Inventory Skewing（事前的な在庫バランス管理）であるべき。159# レビューで「P0 格上げを強く推奨」と指摘済み。 |

### 3-B. balance_forced_deadlock_limit (強制スキップ連続上限)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1694-L1711) |
| **設定** | `loss_control.balance_forced_deadlock_limit: 3` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L225) |
| **テスト** | [tests/unit/v460/test_154_deadlock_prevention.py](../../tests/unit/v460/test_154_deadlock_prevention.py) |
| **概要** | 連続 3 回 forced skip に達したら、forced side でも強制実行（デッドロック防止）。 |
| **判定** | **姑息的** ★★★ — 姑息的手段 (3-A) のデッドロック回避策 = 二重の対症療法。P0-08 デッドロック（9.5h 空転）の再発防止が目的。 |

### 3-C. balance_forced_rescue (救済モード)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1713-L1721), [L926-L944](../../scripts/v460/run_fill_test.py#L926-L944) |
| **設定** | `loss_control.balance_forced_rescue_enabled: false` (未有効化), `balance_forced_rescue_offset_mult: 2.0` |
| **Config** | [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py#L227-L228) |
| **概要** | forced skip の代わりに offset を 2 倍にして安全にポジション解消を試みる救済モード。現在は **未有効化**。 |
| **判定** | **姑息的** ★★★ — 3-A の代替策としてやや改善しているが、やはり在庫偏りの事後対応。Inventory Skewing の劣化版。 |

---

## 4. Spread ガード

### 4-A. spread_too_narrow 分類

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L883-L884) |
| **Cancel Reason** | `CR.SPREAD_TOO_NARROW` |
| **統計** | 10 日間: 50 件 (2.9%) |
| **概要** | maker_price 計算でスプレッドがマイナス/ゼロ（spread too narrow）の場合を ERROR→INFO に降格し、専用分類として記録。 |
| **判定** | **正常な分類改善** ★☆☆ — 分類・ログレベル修正であり、姑息ではない。市場の自然状態の適切な記録。 |

### 4-B. narrow_spread_pause (スプレッド狭小時の一時停止)

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L951-L979) |
| **設定** | `loss_control.narrow_spread_pause.enabled: false` (未有効化), `threshold_bps: 3.0`, `pause_sec: 5.0`, `max_consecutive: 3` |
| **Cancel Reason** | `CR.NARROW_SPREAD_PAUSE` |
| **概要** | スプレッドが 3bps 未満の場合、5 秒スリープ。連続 3 回超過で強行。 |
| **判定** | **姑息的** ★★☆ — 未有効化だが、狭スプレッド時にスリープで待機する設計は市場状態の変化を待つ受動的アプローチ。spread_adaptive.narrow_spread_boost との二重防御。 |

### 4-C. spread_adaptive (スプレッド適応型 Offset)

| 項目 | 内容 |
|------|------|
| **設定** | `spread_adaptive.enabled: true`, `narrow_spread_bps: 2.5`, `narrow_spread_boost: 2.0`, `wide_spread_bps: 4.5`, `wide_spread_ratio: 0.5` |
| **ファイル** | [scripts/v460/lib/maker_price.py](../../scripts/v460/lib/maker_price.py) (MakerPriceCalculator 内) |
| **概要** | スプレッド幅に応じて offset を動的調整。narrow → boost 2x、wide → 割引 0.5x。buy/sell 非対称化あり。 |
| **判定** | **準根本対策** ★☆☆ — スプレッド条件への適応的対応。静的閾値だが方向性は正しい。 |

---

## 5. Stale データガード

### 5-A. stale_skip_gate_blocked

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/lib/order_monitor.py](../../scripts/v460/lib/order_monitor.py#L335) |
| **Cancel Reason** | `CR.STALE_SKIP_GATE_BLOCKED` |
| **統計** | 10 日間: 27 件 (1.6%) |
| **概要** | stale order 再発注時に SkipGate が再評価して block した場合の分類。 |
| **判定** | **やや姑息** ★★☆ — stale order + SkipGate の二重チェックで安全側に倒すが、stale order が発生する根因（order_timeout_sec 設計、市場ミスマッチ）の対処ではない。 |

### 5-B. ob_age_ms (Orderbook 鮮度追跡)

| 項目 | 内容 |
|------|------|
| **ファイル** | [ztb/metrics/fill_quality.py](../../ztb/metrics/fill_quality.py#L87) `FillRecord.ob_age_ms` |
| **概要** | OB 取得からの経過ミリ秒を FillRecord に記録。鮮度判定のテレメトリ。 |
| **判定** | **正常な計測** — テレメトリ記録であり、姑息ではない。 |

---

## 6. その他の防御ゲート

### 6-A. unknown_regime_buy_skip

| 項目 | 内容 |
|------|------|
| **ファイル** | [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py#L1748-L1772) |
| **設定** | `loss_control.skip_buy_unknown_regime: true` |
| **Cancel Reason** | `CR.UNKNOWN_REGIME_BUY_SKIP` |
| **概要** | レジームが "unknown" の場合に buy をスキップ（unknown-buy 平均 -1.384bps）。 |
| **判定** | **姑息的** ★★☆ — unknown レジーム＝判定不能な状態での一律回避。レジーム検知精度向上が根本対策。 |

### 6-B. fast_fill_defense (即約定防御)

| 項目 | 内容 |
|------|------|
| **設定** | `fast_fill_defense.enabled: true`, buy: 10s, sell: 15s, `offset_boost: 2.0` |
| **概要** | 約定が早すぎる（queue_wait <= 閾値）場合、次サイクルで offset を boost。 |
| **判定** | **準根本対策** ★★☆ — AS の兆候（即約定 = 情報優位トレーダーに狙われた）を検出して対処。方向性は正しいが、閾値が静的。 |

### 6-C. volatility_guard (急変検知)

| 項目 | 内容 |
|------|------|
| **設定** | `volatility_guard.enabled: true`, `velocity_threshold_bps: 15.0`, `vpin_threshold: 0.63`, `offset_boost_factor: 2.0` |
| **概要** | 短期 price velocity と VPIN で急変を検知し、offset を boost。 |
| **判定** | **根本対策に近い** ★☆☆ — リアルタイム市場状態に基づく動的ガード。time_filter の静的遮断を補完。 |

---

## 姑息度ランキング（影響度 × 姑息度）

| 順位 | 項目 | 姑息度 | 10日間件数 | 根本対策 |
|------|------|--------|-----------|----------|
| **1** | balance_forced_skip (3-A) | ★★★ | 377 (21.9%) | Inventory Skewing (159# P0 推奨) |
| **2** | trending_sell_skip (2-A) | ★★★ | 233 (13.5%) | offset 非対称化 + 動的レジーム適応 |
| **3** | TimeFilter 静的遮断 (1-A) | ★★★ | N/A (時間帯全停止) | 動的ゲーティング (107# 提案済) |
| **4** | sell_dynamic_kill (2-C) | ★★☆ | 92 (5.3%) | sell AS 根因分析 + offset 最適化 |
| **5** | 086 deadlock 防止 (1-B) | ★★★ | 14 | time_filter 廃止で消滅 |
| **6** | balance_forced_deadlock_limit (3-B) | ★★★ | (3-A の subset) | Inventory Skewing で 3-A ごと消滅 |
| **7** | max_consecutive_trending_sell (2-B) | ★★★ | (2-A の subset) | 2-A 解消で消滅 |
| **8** | sell_guard (2-D) | ★★☆ | 16 + 161混入 | スプレッド構造分析 |
| **9** | stale_skip_gate_blocked (5-A) | ★★☆ | 27 (1.6%) | order lifecycle 改善 |
| **10** | unknown_regime_buy_skip (6-A) | ★★☆ | N/A | レジーム検知精度向上 |

---

## YAML 設定ファイル内の姑息的閾値一覧

| YAML パス | 値 | 姑息度 | 備考 |
|-----------|-----|--------|------|
| `time_filter.enabled` | `true` | ★★★ | 静的時間帯遮断の有効化 |
| `time_filter.skip_utc_hours` | `[16]` | ★★★ | グローバル遮断時間 |
| `time_filter.skip_utc_hours_buy` | `[8, 16, 18]` | ★★★ | buy 遮断時間 |
| `time_filter.skip_utc_hours_sell` | `[4, 8, 14]` | ★★★ | sell 遮断時間 |
| `time_filter.max_086_consecutive_wait` | `3` | ★★★ | deadlock 防止カウンタ |
| `loss_control.skip_balance_forced` | `true` | ★★★ | forced 時スキップ |
| `loss_control.balance_forced_deadlock_limit` | `3` | ★★★ | forced deadlock 限界 |
| `loss_control.balance_forced_rescue_enabled` | `false` | ★★★ | (未有効化) |
| `loss_control.balance_forced_rescue_offset_mult` | `2.0` | ★★★ | (未有効化) |
| `loss_control.skip_sell_trending` | `true` | ★★★ | trending sell スキップ |
| `loss_control.skip_sell_trending_up_only` | `true` | ★★★ | trending_up のみ |
| `loss_control.max_consecutive_trending_sell_skip` | (未指定=30) | ★★★ | 安全弁 |
| `loss_control.skip_buy_unknown_regime` | `true` | ★★☆ | unknown buy スキップ |
| `loss_control.sell_dynamic_kill.enabled` | `true` | ★★☆ | sell 動的停止 |
| `loss_control.sell_dynamic_kill.threshold_bps` | `-0.5` | ★★☆ | sell 停止閾値 |
| `loss_control.sell_dynamic_kill.resume_window` | `10` | ★★☆ | 凍結解除サイクル |
| `loss_control.buy_dynamic_kill.enabled` | `true` | ★★☆ | buy 動的停止 |
| `loss_control.buy_dynamic_kill.threshold_bps` | `-0.8` | ★★☆ | buy 停止閾値 |
| `sell_guard.max_spread_jpy` | `4000.0` | ★★☆ | sell スプレッド上限 |
| `sell_guard.offset_floor` | `0.10` | ★★☆ | sell offset 最低保証 |
| `loss_control.narrow_spread_pause.enabled` | `false` | ★★☆ | (未有効化) |
| `side_offset.sell` | `0.18` | ★★☆ | sell 非対称 offset |

---

## 構造的問題の整理

### 問題 A: 「止血のための止血」パターン
```
balance_forced_skip → balance_forced_deadlock_limit → balance_forced_rescue
trending_sell_skip → max_consecutive_trending_sell_skip
time_filter → max_086_consecutive_wait (deadlock break)
```
姑息的手段が副作用を生み、その副作用を別の姑息的手段で対処する **カスケード構造**。

### 問題 B: Sell 側の過剰防御レイヤ
Sell は以下 **6 層** のガードを通過する必要がある:
1. `time_filter` (静的時間帯遮断)
2. `skip_sell_trending` (レジーム遮断)
3. `sell_dynamic_kill` (rolling PnL 停止)
4. `sell_guard` (max_spread / offset_floor)
5. `skip_gate` (ML 分類器)
6. `balance_forced_skip` (残高遮断)

結果: sell fill_rate が buy の 63% (36.7% vs 58.7%)。

### 問題 C: Inventory Skewing の不在 → **162# で実装済み (enabled=false)**
159# レビューで最も強く指摘された根本問題。~~現在は `balance_forced_skip` で事後対応しているが~~、マーケットメイカー本来の設計は **事前的** な在庫バランス管理 (Inventory Skewing)。

> **162# 対応**: `maker_price.py` に `update_inventory()` + `compute()` 内の非対称 offset 補正を実装。
> `fill_test.yaml` に `inventory_skewing:` セクション追加。`enabled: false` でデプロイ済み、ステージング後に ON 予定。
> commit: `42a06d8e9`

---

## 推奨: 根本対策への移行ロードマップ

| 優先度 | 対策 | 消滅する姑息的手段 | 期待効果 |
|--------|------|-------------------|----------|
| **P0** | ~~Inventory Skewing 実装~~ **162# 実装済み** (`42a06d8e9`, enabled=false) | 3-A, 3-B, 3-C (balance_forced 関連全体) | fill_rate +15-20pt |
| **P0** | 動的ゲーティング (time_filter 代替) | 1-A, 1-B, 1-C | アイドル時間 -30%+ |
| **P1** | sell offset 動的最適化 | 2-A, 2-B (trending_sell_skip) | sell fill_rate 改善 |
| **P1** | AS 根因分析 + モデル改善 | 2-C (sell_dynamic_kill), 6-A (unknown_regime skip) | PnL 改善 |
| **P2** | sell_guard 閾値動的化 | 2-D | 16+件のキャンセル削減 |

---

## Stopgap 退出基準表 (162# §7 P0)

> 各 stopgap を OFF にする際の **前提条件・監視指標・ロールバック条件** を定義する。
> 上位の stopgap ほど依存関係が深いため、退出順序を厳守すること。

### 退出順序: IS → per-regime判定 → time_filter縮退 → sell_guard緩和

| ID | Stopgap | 退出前提条件 | 監視指標 (24h) | OFF判定基準 | ロールバック条件 |
|----|---------|-------------|---------------|------------|----------------|
| 3-A | balance_forced_skip | IS enabled=true で 24h 稼働 | buy/sell fill_count 比率, inventory imbalance | imbalance 標準偏差 < 0.3, forced_skip 発生率 < 5% | forced_skip 率 > 15% 即時 ON |
| 3-B | balance_forced_deadlock_limit | 3-A 退出後 24h 観察 | deadlock_count, max_consecutive_forced | deadlock_count = 0 (24h) | 1 回でも deadlock 発生で即時 ON |
| 3-C | balance_forced_rescue | 3-B 退出後 24h 観察 | rescue_triggered_count | rescue 0 回 (48h) | rescue 発生で即時 ON |
| 2-A | trending_sell_skip | IS + 107# dynamic gating 安定 | sell AS_rate (per-regime), sell PnL/trade | sell AS_rate < 35% (trending regime), total PnL > 0 | AS_rate > 50% で即時 ON |
| 2-B | max_consecutive_trending_sell_skip | 2-A 退出後 | consecutive_skip_count | 2-A OFF で不要 | 2-A 復活時に自動 ON |
| 1-A | time_filter (静的) | 107# Step 2 regime-adaptive で 48h 安定 | skip_utc_hours 該当時間帯の fill PnL | regime-adaptive PnL > baseline (p < 0.1) | 該当時間帯で AS_rate > 50% |
| 1-C | TIME_FILTER_BOTH_SIDES | sell 側 time_filter 縮退後 72h | 両側同時遮断の発生頻度 | 発生 0 回 (72h) | 1 回発生で検討 |
| 2-C | sell_dynamic_kill | 164# SHAP 知見で SkipGate 改善後 | sell rolling PnL, sell_dynamic_kill 発動回数 | kill 発動 < 1回/day (7日平均) | kill 発動 > 3回/day |
| 2-D | sell_guard | sell offset 動的化 (P1) 完了後 | sell cancel 率, sell PnL | cancel 率 < 10%, PnL > 0 | cancel 率 > 20% |
| 6-A | unknown_regime_buy_skip | regime 検出精度向上後 | unknown_regime 出現率 | unknown < 5% (7日) | unknown > 15% |

### 退出プロセス

1. **ステージング (24h)**: 対象 stopgap を `enabled: false` に変更し dry-run 相当で観察
2. **判定**: 上記監視指標が OFF 基準を満たすか確認
3. **本番 OFF**: 基準満たした場合のみ OFF。`git_sha` と日時を記録
4. **観察期間 (48h)**: ロールバック条件を継続監視
5. **確定**: 48h 問題なければ当該 stopgap をコード削除候補に移行

---

## 関連ドキュメント
- [107_ph2_analysis_time_filter_dynamic_gating.md](107_ph2_analysis_time_filter_dynamic_gating.md) — Time Filter 動的ゲーティング提案
- [110_ph2_fix_086_time_filter_deadlock.md](110_ph2_fix_086_time_filter_deadlock.md) — 086# deadlock 修正
- [154_ph2_dryrun_10h_analysis.md](154_ph2_dryrun_10h_analysis.md) — P0-08 deadlock 発見
- [156_ph2_rpt_sell_root_cause_and_phase_d_plan.md](156_ph2_rpt_sell_root_cause_and_phase_d_plan.md) — Sell 根因分析
- [157_ph2_fix_regime_deadlock_and_cancel.md](157_ph2_fix_regime_deadlock_and_cancel.md) — レジームデッドロック修正
- [159_phg_rev_158_phase_d_backlog_review.md](159_phg_rev_158_phase_d_backlog_review.md) — Inventory Skewing P0 推奨
- [162_phg_rpt_fill_test_10day_log_analysis.md](162_phg_rpt_fill_test_10day_log_analysis.md) — 10 日間ログ分析
- [164_phg_rpt_skip_gate_shap_analysis.md](164_phg_rpt_skip_gate_shap_analysis.md) — SkipGate SHAP 特徴量分析

---

## 更新履歴
| 日付 | 内容 |
|------|------|
| 2026-02-25 | 初版作成 (17 件カタログ) |
| 2026-02-25 | Inventory Skewing 実装反映 (`42a06d8e9`) — P0 項目消化 |
| 2026-02-25 | God Object 分割完了 (`6b766caf9`): run_fill_test 2,231→378行 3 Mixin, maker_price compute() 306→143行, fill_config from_yaml() 479→139行 |
| 2026-02-26 | IS YAML enabled=true, 107# Phase 3 Step 2 動的ゲーティング実装 (regime-adaptive time_filter) |
| 2026-02-26 | 162# §7 P0: Stopgap 退出基準表追記, 164# SHAP 分析との相互参照追加 |
