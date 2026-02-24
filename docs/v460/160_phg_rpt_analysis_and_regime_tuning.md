# 160# 分析レポート: P0-3 Trending フィルタ検証 + P1-2 Offset 最適化 + レジーム改善

> 2026-02-24 Fill Test (PID 124796) 蓄積データに基づく分析結果。
> 対象: `results/v460/fill_test/` 12 ファイル, 2,932 total records
> 160# で追加: regime=None 問題の根本原因特定と skip record 全箇所への regime 伝搬

---

## §1 P0-3: skip_sell_trending 方向別分解の効果測定

### 1.1 背景

| 項目 | 値 |
|---|---|
| 施策 | 156# D-4 `skip_sell_trending_up_only: true` |
| 目的 | trending_up 時のみ sell をスキップ、trending_down sell は開放 |
| 成果基準 | trending_down sell の avg PnL30 > -0.3 bps |

### 1.2 分析手法

- **trending_sell_skip**: 221 件 — 約定前キャンセルのため PnL データなし
- **代替手法**: filled sell/buy レコードを regime 別に集計し、フィルタ妥当性を検証
- 分析対象: 681 filled sell + 692 filled buy

### 1.3 Regime × Side PnL30 テーブル

| regime | sell PnL30 (bps) | buy PnL30 (bps) | n(sell) | n(buy) | sell 判定 |
|---|---|---|---|---|---|
| ranging | -0.32 | -0.03 | 377 | 375 | 許容範囲 (主力) |
| trending | **-0.66** | **+0.57** | 118 | 118 | ❌ skip 正解 |
| trending_down | **+1.18** | +4.25 | 11 | 13 | ✅ 開放成功 |
| unknown | -0.69 | -0.46 | 174 | 186 | 別ゲートで対応済 |
| **全体** | **-0.45** | **+0.04** | 680 | 692 | — |

### 1.4 カウンターファクチュアル分析

| 指標 | 値 |
|---|---|
| trending sell skip 件数 | 221 件 |
| trending sell 実績 PnL30 | -0.66 bps (仮に約定していた場合の期待値) |
| **累積損失回避** | **-0.66 × 221 = -145.9 bps** |
| trending_down sell 累積利益 | +1.18 × 11 = **+12.97 bps** |
| D-4 (trending_down 開放) の純効果 | +12.97 bps (損失回避に追加) |

### 1.5 安全弁の確認

- `max_consecutive_trending_sell_skip: 30` — 機能確認済み
- trending buy は +0.57 bps で正の期待値を維持 (sell skip 後も buy は継続)

### 1.6 最終判定

| 基準 | 結果 |
|---|---|
| trending_down sell PnL30 > -0.3 bps | **+1.18 bps → ✅ PASS** (基準を 1.48 bps 超過) |
| trending sell skip の妥当性 | **✅ 正解** (-0.66 bps は負の期待値) |
| 方向別分解 (D-4) の有効性 | **✅ 有効** (down +1.18, up -0.66 の方向性一致) |
| **総合判定** | **✅ フィルタ設計は妥当** |

### 1.7 今後の運用推奨

1. **現行設定を維持**: `skip_sell_trending: true`, `skip_sell_trending_up_only: true`
2. **trending_down sell の継続監視**: n=11 はサンプル不足。n≥30 到達後に再評価
3. **安全弁は据え置き**: `max_consecutive_trending_sell_skip: 30` は適切

---

## §2 P1-2: Fill Rate 向上のための Offset 最適化

### 2.1 背景

| 項目 | 値 |
|---|---|
| 問題 | timeout 268 件 (buy 161, sell 107) — offset が保守的すぎて fill されない |
| Fill rate | 全体 46.9% (BUY 59.0%, SELL 38.7%) |
| 分析対象 | 直近 2 日間 filled records (n=102) |

### 2.2 実効 Offset 分布

| 指標 | BUY | SELL |
|---|---|---|
| Mean offset | 0.097 | 0.289 |
| Base config | 0.05 | 0.18 |
| Boost 倍率 | ×1.0 (trending), ×0.90 (ranging) | ×1.5 (trending), ×0.90 (ranging) |
| 実効範囲 | 0.045–0.15 | 0.162–0.30 |

### 2.3 Offset × PnL30 バケット分析

| side | offset range | n | mean PnL30 (bps) | 判定 |
|---|---|---|---|---|
| BUY | ≤0.1 | 39 | +0.16 | 低 offset → 利益薄 |
| BUY | 0.1–0.3 | 12 | **+5.86** | ⭐ **スイートスポット** |
| BUY | >0.3 | 1 | -4.17 | 過剰 → 逆選択のみ fill |
| SELL | ≤0.1 | 0 | — | — |
| SELL | 0.1–0.3 | 44 | +0.66 | 現行レンジ、妥当 |
| SELL | >0.3 | 6 | -0.50 | 過剰 |

### 2.4 Key Insights

1. **Buy offset スイートスポット**: 0.1–0.3 帯の PnL30 は ≤0.1 帯の **36.6 倍** (+5.86 vs +0.16)
2. **Buy base offset 0.05 は低すぎる**: 実効 offset の大半が ≤0.1 帯に集中し、最適帯に到達していない
3. **Sell offset は概ね妥当**: 0.1–0.3 帯 (n=44) で +0.66 bps、現行 0.18 base は適切
4. **>0.3 帯は両 side とも負の期待値**: offset 上限 0.30 は正しいキャップ

### 2.5 推奨アクション

| アクション | 現行値 | 推奨値 | 根拠 | リスク |
|---|---|---|---|---|
| Buy base offset 引き上げ | 0.05 | **0.12–0.15** | 0.1–0.3 帯が最適 | n=12 はサンプル不足 |
| Sell base offset | 0.18 | **据え置き** | 既に 0.1–0.3 帯内 | P0-2 A/B で段階縮小予定 |
| adaptive offset range | 0.01–0.30 | **据え置き** | 上限 0.30 は妥当 | — |

### 2.6 実施判断

| 項目 | 判断 |
|---|---|
| 即時変更の可否 | **❌ 保留** — n=102 (2 日間) はサンプル不足 |
| 必要サンプル数 | n≥200 (4 日分) で統計的有意性を確認後に実施 |
| 検証方法 | P0-2 A/B テスト基盤 (`ab_test.variant`) を活用 |
| 暫定対応 | `spread_offset_ratio` を YAML 上で **コメント付き推奨値を記載** (変更は次回 restart 時) |

---

## §3 YAML 外部化候補パラメータ

### 3.1 ハードコード値監査結果

`run_fill_test.py`, `scripts/v460/lib/` 全ファイルを精査し、
YAML 未定義のマジックナンバーを特定。

#### Priority A: YAML 化すべき（運用チューニング直結）

| # | ファイル | ハードコード値 | 用途 | 推奨 |
|---|---|---|---|---|
| A1 | `resilience.py` L47-50 | `failure_threshold=5`, `recovery_timeout=120.0`, `success_threshold=2`, `timeout=30.0` | CircuitBreaker デフォルト | **即時外部化** — API 安定性で要チューニング |
| A2 | `resilience.py` L73-77 | `rss_warn_mb=1500.0`, `rss_critical_mb=2500.0`, `disk_free_warn_gb=2.0`, `gc_interval_cycles=100`, `check_interval_sec=300.0` | HealthMonitor 閾値 | **即時外部化** — 環境依存 |
| A3 | `skip_gate_evaluator.py` L88 | `_HOT_RELOAD_CHECK_INTERVAL_SEC = 120.0` | モデル hot-reload 間隔 | **外部化** — retrain frequency との連動 |
| A4 | `adaptation_engine.py` L42 | `_RECORDS_CACHE_TTL_SEC = 10.0` | 適応キャッシュ TTL | **外部化** — I/O vs stale のバランス |
| A5 | `run_fill_test.py` L819 | `limit=100` | trades 取得件数 | **外部化** — VPIN 計算に影響 |
| A6 | `side_selector.py` L139 | `cycles=3` (freeze_side) | 残高不足 side 凍結 | **外部化** — cycle_interval × 3 = 6 分凍結 |

#### Priority B: 検討（安定性向上）

| # | ファイル | ハードコード値 | 用途 |
|---|---|---|---|
| B1 | `regime_detector.py` L245/253/261 | confidence 計算定数 (0.6+excess×0.4 等) | 分類器感度 — 頻繁に変更しない |
| B2 | `order_monitor.py` L196/200 | `poll_interval×3`, `min_spread×2` | postonly_reject 判定 — 分析に影響 |

### 3.2 YAML 追加設計

```yaml
# ---- resilience (CircuitBreaker / HealthMonitor) ----
resilience:
  circuit_breaker:
    failure_threshold: 5          # API 連続失敗→OPEN
    recovery_timeout: 120.0       # OPEN→HALF_OPEN 待機 (秒)
    success_threshold: 2          # HALF_OPEN→CLOSE 成功回数
    timeout: 30.0                 # API タイムアウト (秒)
  health_monitor:
    rss_warn_mb: 1500.0           # RSS 警告閾値 (MB)
    rss_critical_mb: 2500.0       # RSS 緊急閾値 (MB)
    disk_free_warn_gb: 2.0        # ディスク空き警告 (GB)
    gc_interval_cycles: 100       # GC 実行間隔 (サイクル数)
    check_interval_sec: 300.0     # ヘルスチェック間隔 (秒)

# ---- tuning (追加) ----
tuning:
  # ...既存値...
  records_cache_ttl_sec: 10.0             # 適応エンジン キャッシュ TTL
  hot_reload_check_interval_sec: 120.0    # SkipGate モデル差替チェック間隔
  trades_recorder_fetch_limit: 100        # TradesRecorder 取得件数
  balance_freeze_cycles: 3                # 残高不足 side の凍結サイクル数
```

---

## §4 変更履歴

| 日付 | 内容 |
|---|---|
| 2026-02-24 | 初版: P0-3 trending フィルタ検証結果 + P1-2 offset 最適化分析 + YAML 外部化候補 |
| 2026-02-24 | 160# 改番: regime=None 根本原因分析 + skip record regime 伝搬実装 |

---

## §5 160# レジーム改善: regime=None 問題の解消

### 5.1 問題分析

直近 5 ファイル (1,753 records) の regime フィールド分布:

| 区分 | 件数 | 割合 | 説明 |
|---|---|---|---|
| `regime=None` | 770 | **43.9%** | ❌ 早期キャンセルで regime 未記録 |
| `regime="ranging"` | 600 | 34.2% | ✅ 正常 |
| `regime="trending"` | 321 | 18.3% | ✅ 正常 |
| `regime="trending_down"` | 35 | 2.0% | ✅ 正常 |
| `regime="trending_up"` | 27 | 1.5% | ✅ 正常 |
| `regime="unknown"` | 0 | **0.0%** | ✅ 分類器は正常動作 (min_confidence=0.2 効果) |

#### 根本原因: regime=None は「分類器の問題」ではなく「記録漏れ」

`_make_skip_record()` による早期キャンセルレコードが `regime` 引数なしで生成されていた。
分類器自体は近直データで unknown=0% と極めて良好に動作。

cancel_reason 別内訳 (regime=None 770件):

| cancel_reason | 件数 | 説明 |
|---|---|---|
| balance_forced_skip | 377 | 残高強制切替スキップ |
| skip_gate | 169 | SkipGate ML 判定 |
| orderbook_error | 140 | OB 取得失敗 |
| spread_too_narrow | 37 | スプレッド狭小 |
| sell_dynamic_kill | 16 | sell 動的停止 |
| sell_guard_reject | 15 | sell ガード拒否 |
| time_filter_086_deadlock | 14 | 時間帯フィルタ |
| time_filter_both_sides | 2 | 両 side フィルタ |

### 5.2 修正内容

`_current_regime_value()` ヘルパーメソッドを追加し、全 `_make_skip_record` 呼び出しに `regime=` を伝搬:

| ファイル | 修正箇所 | cancel_reason |
|---|---|---|
| `run_fill_test.py` | L818 | circuit_breaker_open |
| `run_fill_test.py` | L912 | orderbook_error / sell_guard_reject |
| `run_fill_test.py` | L969 | narrow_spread_pause |
| `run_fill_test.py` | L1459 | time_filter_both_sides |
| `run_fill_test.py` | L1521 | time_filter_086_deadlock |
| `run_fill_test.py` | L1594 | preflight_insufficient |
| `run_fill_test.py` | L1646 | preflight_pause |
| `run_fill_test.py` | L1109 | api_error (direct FillRecord) |
| `run_fill_test.py` | L1728 | balance_forced_skip |
| `run_fill_test.py` | L1842 | buy_dynamic_kill |
| `run_fill_test.py` | L1867 | sell_dynamic_kill |
| `skip_gate_evaluator.py` | L478 | skip_gate_rule_unknown_sell |
| `skip_gate_evaluator.py` | L629 | skip_gate (ML判定) |

既に `regime=` を持っていた箇所 (変更不要):
- L1755: `unknown_regime_buy_skip` → `regime="unknown"` (明示的)
- L1814: `trending_sell_skip` → `regime=_current_regime.value` (明示的)

### 5.3 期待効果

| 指標 | Before | After |
|---|---|---|
| regime=None 率 | 43.9% (770/1753) | **~0%** (warmup 期間のみ) |
| regime=unknown 率 | 0.0% (recent) | 0.0% (変化なし) |
| 分析可能レコード率 | 56.1% | **~100%** |

これにより:
1. `compute_regime_metrics()` の精度向上 (43.9% のデータが分析対象に復帰)
2. `skip_sell_unknown_regime` ゲートの改善 (regime_value=None を unknown 扱いしていた問題が解消)
3. regime 別 PnL 分析の信頼性向上
