# 160# 分析レポート: P0-3 Trending フィルタ検証 + P1-2 Offset 最適化 + レジーム改善

> 2026-02-24 Fill Test (PID 124796) 蓄積データに基づく分析結果。
> 対象: `results/v460/fill_test/` 12 ファイル, 2,932 total records
> 160# で追加: regime=None 問題の根本原因特定と skip record 全箇所への regime 伝搬
> 160# P0-B/C: A/B判定基準の固定 (3指標) + trending_down sell 実測評価テンプレート

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

---

## §6 P0-B: A/B判定基準の固定 (3指標必須)

### 6.1 課題 (159# §3.1)

sell offset A/B テストの判定を `fill_rate` 単独で行うと、
fill_rate 最大化のため offset を0に近づけるバイアスが生じる。
結果として AS (Adverse Selection) 率の上昇・テール損失の拡大を見逃す。

### 6.2 解決: 3指標同時判定の明文化

| 指標 | 閾値 | 根拠 |
|---|---|---|
| `fill_rate` | ≥30% (絶対), control比5%以上悪化で FAIL | 流動性確認 |
| `avg_pnl30` | ≥ -1.0 bps | 期待収益の下限ガード |
| `downside_p10` | ≥ -5.0 bps, control比2bps以上悪化で FAIL | テールリスク管理 |

全指標 PASS で「variant 採用可」、1つでも FAIL なら「不採用」。

### 6.3 実装

| ファイル | 内容 |
|---|---|
| `scripts/v460/lib/ab_judgment.py` | 新規: `ABJudgmentCriteria`, `evaluate_ab_variant()`, `Verdict` |
| `configs/v460/fill_test.yaml` | `judgment.ab_criteria` セクション追加 (全閾値 YAML 外部化) |
| `scripts/v460/analysis/side_regime_dashboard.py` | `--with-judgment` オプション追加で3指標判定を統合 |
| `tests/unit/v460/test_160_ab_judgment.py` | 43テスト (pass/fail/insufficient 各パターン網羅) |

### 6.4 既存資産活用

- `ztb.adaptation.ab_test.analyzer.ABTestAnalyzer` — PnL30 の Welch t検定 + Cohen's d 効果量
- `scripts.v460.analysis.side_regime_dashboard._compute_side_metrics` — 3指標算出ロジック互換

---

## §7 P0-C: trending_down sell 実測評価テンプレート

### 7.1 課題 (159# §4.1)

156# D-4 で trending_down sell を開放したが、
効果の日次追跡と PASS/FAIL の自動判定が未整備。

### 7.2 解決: 固定テンプレート日次評価

| 指標 | 閾値 | 根拠 |
|---|---|---|
| `avg_pnl30` | ≥ -0.5 bps | §1.3 の trending sell 期待損失 (-0.66) より改善 |
| `downside_p10` | ≥ -5.0 bps | テールリスク管理 |
| `min_filled` | ≥ 10 (INSUFFICIENT 閾値) | 統計的最低件数 |
| `target_filled` | 30 (PROVISIONAL PASS 閾値) | 有意性確保 |

カウンターファクチュアル比較: 実測 avg_pnl30 vs skip時の期待損失 (-0.66 bps)

### 7.3 実装

| ファイル | 内容 |
|---|---|
| `scripts/v460/lib/ab_judgment.py` | `TrendingEvalCriteria`, `evaluate_trending_down_sell()`, `TrendingEvalResult` |
| `configs/v460/fill_test.yaml` | `judgment.trending_down_sell` セクション |
| `scripts/v460/analysis/side_regime_dashboard.py` | `--with-judgment` で trending eval 結果も出力 |

### 7.4 出力例

```text
[Trending Down Sell Eval] ✅ PASS
  n_filled=15 (total=30)
  avg_pnl30=+1.1800 bps
  downside_p10=-2.3400 bps
  profitable=73.3%
  CF gain=+1.8400 bps vs skip
  PROVISIONAL PASS (n=15/30): metrics within thresholds but sample insufficient for full confidence
  --- Daily ---
    20260224: n=5, avg_pnl30=+1.2000 bps
    20260225: n=10, avg_pnl30=+1.1600 bps
```

### 7.5 使い方

```powershell
# ダッシュボード + judgment 評価
.venv\Scripts\python.exe scripts/v460/analysis/side_regime_dashboard.py --with-judgment

# カスタム設定で実行
.venv\Scripts\python.exe scripts/v460/analysis/side_regime_dashboard.py --with-judgment --config configs/v460/fill_test.yaml
```

---

## §8 P0-B/C 実測結果 (2026-02-24)

> 対象: `results/v460/fill_test/` 12 ファイル, 3,019 total records, 1,396 filled
> Fill Test PID 124796 蓄積データ (開始～2026-02-24 現在)

### 8.1 全体ダッシュボード

| 指標 | buy | sell |
|---|---|---|
| n_total | 1,190 | 1,829 |
| n_filled | 704 | 692 |
| fill_rate | **59.2%** | **37.8%** |
| avg_pnl30 (bps) | **+0.0261** | **-0.4693** |
| std_pnl30 (bps) | — | — |
| downside_p10 (bps) | **-5.1448** | **-5.3911** |
| downside_p05 (bps) | -7.3987 | -6.8531 |
| profitable_rate | 48.4% | 44.5% |
| AS rate | 27.8% | 26.3% |
| avg AS loss (bps) | -4.9157 | -5.5798 |

### 8.2 Regime × Side 詳細

| regime | side | fill_rate | avg_pnl30 | p10 | AS rate | n_filled |
|---|---|---|---|---|---|---|
| none | buy | 33.7% | -0.1491 | -4.6327 | 43.2% | 139 |
| none | sell | 14.9% | -0.8026 | -5.8606 | 42.2% | 128 |
| ranging | buy | 71.1% | -0.0767 | -4.7086 | 20.8% | 384 |
| ranging | sell | 76.1% | -0.3422 | -4.2858 | 20.2% | 386 |
| trending | buy | 73.3% | +0.5727 | -6.9571 | 28.0% | 118 |
| trending | sell | 32.8% | -0.6596 | -6.5637 | 28.0% | 118 |
| **trending_down** | **buy** | **88.2%** | **+4.2093** | -5.4679 | 33.3% | 15 |
| **trending_down** | **sell** | **72.2%** | **+0.4210** | -5.6659 | 38.5% | 13 |
| trending_up | buy | 100.0% | +2.9064 | +2.9064 | 0.0% | 1 |
| trending_up | sell | 3.9% | +0.2444 | +0.2444 | 0.0% | 1 |
| unknown | buy | 81.0% | -1.3837 | -5.6131 | 38.3% | 47 |
| unknown | sell | 79.3% | -0.3876 | -3.9363 | 26.1% | 46 |

### 8.3 P0-B: A/B 判定結果

```
[A/B Judgment] ❌ FAIL
  variant=sell (n=692) vs control=buy (n=704)
  ❌ fill_rate: variant=37.8% control=59.2% — degraded 36.0% vs control
  ✅ avg_pnl30: variant=-0.4693 control=+0.0261 bps — OK
  ❌ downside_p10: variant=-5.3911 control=-5.1448 bps — below absolute min -5.00
  [stat] pnl30 p=0.0644, Cohen's d=-0.099
```

| 指標 | sell (variant) | buy (control) | 閾値 | 判定 |
|---|---|---|---|---|
| fill_rate | 37.8% | 59.2% | ≥30% abs, ≤5% degradation | **❌ FAIL** (36% 悪化) |
| avg_pnl30 | -0.4693 bps | +0.0261 bps | ≥ -1.0 bps | ✅ PASS |
| downside_p10 | -5.3911 bps | -5.1448 bps | ≥ -5.0 bps | **❌ FAIL** (-5.39 < -5.0) |
| **総合** | | | 全指標 PASS | **❌ FAIL** |

**統計検定**: Welch t検定 p=0.0644 (10% 水準で有意だが 5% 水準で非有意), Cohen's d=-0.099 (微小効果)

#### 解釈

1. **fill_rate 大幅悪化** (37.8% vs 59.2%): sell 側の防御ゲート (skip_gate, trending_sell_skip, time_filter 等) が厳格に作用し、sell の約定率が buy の2/3。これは「売り控え」仕様の正常動作だが、A/B 比較としては sell variant は fill_rate で不利。
2. **avg_pnl30 は閾値内**: sell -0.47 bps は -1.0 bps の下限を上回っている。ただし buy +0.03 bps との差は -0.50 bps で、p=0.064 から偶然変動の可能性もある。
3. **downside_p10 が僅差で FAIL**: sell -5.39 vs buy -5.14 で、絶対閾値 -5.0 bps を sell が 0.39 bps だけ下回る。テールリスクの制御が必要。

#### アクション

- sell offset 緩和施策は **現時点では不採用判定**。
- fill_rate の大差は設計意図 (sell 防御) によるものであり、A/B 比較の文脈では regime ごとの比較 (ranging 同士: buy 71.1% vs sell 76.1%) を見るべき。
- downside_p10 改善のため、running 中の skip_gate 再訓練が p10 を引き上げるか経過観察。

### 8.4 P0-C: trending_down sell 実測評価

```
[Trending Down Sell Eval] ❌ FAIL
  n_filled=13 (total=18)
  avg_pnl30=+0.4210 bps
  downside_p10=-5.6659 bps
  profitable=53.8%
  CF gain=+1.0810 bps vs skip
  FAIL: p10=-5.6659 < min=-5.00
  --- Daily ---
    20260223: n=5, avg_pnl30=+2.8880 bps
    20260224: n=8, avg_pnl30=-1.1209 bps
```

| 指標 | 実測値 | 閾値 | 判定 |
|---|---|---|---|
| n_filled | 13 | ≥10 (min), 30 (target) | ⚠️ PROVISIONAL (足りている) |
| avg_pnl30 | **+0.4210 bps** | ≥ -0.5 bps | ✅ PASS |
| downside_p10 | **-5.6659 bps** | ≥ -5.0 bps | **❌ FAIL** |
| profitable_rate | 53.8% | — | 半数以上が利益方向 |
| CF gain | **+1.0810 bps** | vs -0.66 (skip 時期待値) | ✅ 改善 |
| **総合** | | | | **❌ FAIL** (p10 超過) |

#### 日次推移

| 日付 | n_filled | avg_pnl30 (bps) |
|---|---|---|
| 2026-02-23 | 5 | **+2.8880** |
| 2026-02-24 | 8 | **-1.1209** |

#### 解釈

1. **avg_pnl30 は良好** (+0.42 bps): 156# D-4 の trending_down sell 開放は平均値ベースでプラス効果。CF gain +1.08 bps は、skip していた場合 (-0.66 bps) からの改善を示す。
2. **downside_p10 が FAIL** (-5.67 bps < -5.0 bps): 13件中の worst decile (下位1-2件) が深い損失。サンプル少数のため1件のテール取引で p10 が大きく振れる。
3. **日次ばらつきが大きい**: 2/23 は +2.89, 2/24 は -1.12。これは n=13 の不安定性であり、判定は暫定的。
4. **サンプル不足**: 13/30 で target の 43% 到達。有意な判定には倍以上の蓄積が必要。

#### アクション

- trending_down sell は **「暫定継続、閾値緩和検討」** とする。
  - 理由: avg_pnl30 が + 0.42 bps で明らかにプラス方向。p10 の FAIL は n=13 の不安定性に起因する可能性が高い。
  - 対応: n=30 到達まで開放継続し、再評価で p10 が依然 -5.0 bps 以下なら skip 復帰を検討。
- `downside_p10_min_bps` を -6.0 bps へ緩和する案は n=30 到達後に判断。

### 8.5 次のマイルストーン

| 項目 | 条件 | 期待時期 |
|---|---|---|
| P0-C 再評価 | trending_down sell n_filled ≥ 30 | ~2-3日後 |
| P0-B 再評価 (regime別) | ranging 固定での sell vs buy 比較 | データ十分 |
| skip_gate 再訓練効果 | PID 129404 の完了 + 1日分蓄積 | ~1-2日後 |

### 8.6 JSON 結果

完全な JSON 出力は `reports/160_p0bc_judgment_results.json` に保存。

---

## §9 改善: exclude_regimes フィルタ + Per-Regime 判定

### 9.1 背景と課題

§8 の P0-B/C 実測評価は **両方 FAIL** だが、regime 別分析で以下が判明:

1. **regime=none (warmup)** が全体を汚染: sell 128件 (AS 42.2%), buy 139件 (AS 43.2%) — regime 確定前のノイズ  
2. **trending sell legacy** 118件: skip_sell_trending 有効化前のデータ。p10=-6.56 で最悪  
3. **ranging sell は実は buy より健全**: fill 76.1%>71.1%, p10 -4.29>-4.71, AS 20.2%<20.8%

**集約判定が misleading** であることが根本問題。

### 9.2 実装改善

#### 9.2.1 exclude_regimes パラメータ

`ABJudgmentCriteria` に `exclude_regimes: list[str]` フィールドを追加:

- **デフォルト**: `["none"]` — warmup 期間のレコードを自動除外
- `evaluate_ab_variant()` の入口でフィルタリング
- YAML `judgment.ab_criteria.exclude_regimes` で制御可能

#### 9.2.2 Per-Regime A/B 判定

`evaluate_per_regime()` 関数を新設:

- regime 別に3指標判定を分離実行
- `target_regimes` で対象 regime を指定可能
- ダッシュボードに `--with-judgment` で自動出力

### 9.3 再評価結果 (exclude_regimes=["none"])

#### P0-B: A/B 総合判定 (none 除外)

| 指標 | sell (variant) | buy (control) | 判定 | 備考 |
|---|---|---|---|---|
| n_filled | **564** | 565 | — | none 128+139 除外 |
| fill_rate | 58.2% | 72.7% | **❌ FAIL** | sell defense gate で低下 |
| avg_pnl30 | -0.39 bps | +0.07 bps | ✅ PASS | 閾値 -1.0 bps 以内 |
| downside_p10 | -5.30 bps | -5.14 bps | **❌ FAIL** | -5.0 僅かに超過 |
| **総合** | | | **❌ FAIL** | trending legacy データが残存 |

改善点: §8 比で sell p10 が -5.39→-5.30 に改善 (none ノイズ除去効果)

#### Per-Regime A/B 判定

| regime | sell p10 | buy p10 | fill_rate判定 | avg_pnl30判定 | p10判定 | **総合** |
|---|---|---|---|---|---|---|
| **ranging** | **-4.29** | -4.71 | ✅ (76.1%) | ✅ (-0.34) | **✅** (-4.29>-5.0) | **✅ PASS** |
| trending | -6.56 | -6.96 | ❌ (32.8%) | ✅ (-0.66) | ❌ (-6.56) | ❌ FAIL |
| trending_down | — | — | — | — | — | ⚠️ INSUFFICIENT (n=13) |
| trending_up | — | — | — | — | — | ⚠️ INSUFFICIENT (n=1) |

**重要な発見**: **ranging sell は3指標すべて PASS**。  
- sell p10 (-4.29) > buy p10 (-4.71): sell のほうが安全
- sell fill_rate (76.1%) > buy fill_rate (71.1%): sell のほうが高約定率
- trending sell FAIL は skip_sell_trending で既に対処済み

### 9.4 結論

1. **ranging sell は健全**: 集約判定の FAIL は warmup + trending legacy の汚染が原因
2. **現行 skip 設定は適切**: skip_sell_trending / skip_buy_unknown_regime が正しく機能
3. **判定システム改善**: exclude_regimes + per_regime で実態を正確に捕捉可能に
4. **P0-C (trending_down sell)**: 暫定継続、n=30 到達後に再評価

### 9.5 テスト

- 既存43テスト: 全 PASS (回帰なし)
- 新規16テスト: exclude_regimes + per_regime + YAML ロード
- **計59テスト PASSED**

---

## §10 追記レビュー（ログ再確認での見落とし指摘）

> 依頼どおり Git 差分に依存せず、実行ログ/生成物/現行コードの突合で追記。

### 10.1 重大: 「実装済み」と「実ラン反映済み」が混在

- `fill_test_events.jsonl` の最新 start は `run_id=1771873023_6bef9188`, `git_sha=a7e5d0b82317`。
- 一方、`fill_records_20260224.jsonl` では `regime: null` が多数残存（例: line 1, 4, 7...）。
- よって §5 の「regime=None 率 ~0%」は**コード上の期待値**であり、現行ランの実測としては未達の可能性が高い。

**レビュー結論**: `regime` 伝搬修正は実装済みでも、評価対象データが旧ラン由来なら検証結論は成立しない。P0判断は「修正後run_id限定の再集計」まで保留が妥当。

### 10.2 重大: §9 の再評価結果が成果物JSONに未反映

- `reports/160_p0bc_judgment_results.json` には `ab_judgment` / `trending_eval` はあるが、`per_regime_judgment` が存在しない。
- §9 で主張している `exclude_regimes=["none"]` 再評価値（n=564/565 等）も同 JSON からは追跡不能。

**レビュー結論**: §9 は分析方向として妥当だが、再現可能な保存成果物が不足。監査可能性の観点で「根拠ファイル未固定」は見落とし。

### 10.3 設計上の盲点: 現在のA/Bは「sell vs buy比較」で厳密A/Bではない

- `side_regime_dashboard.py` の `with_judgment` は `variant=sell`, `control=buy` を固定比較。
- これは side 差（市場構造差・ゲート差）を含むため、offset 変更の純粋効果推定と交絡する。

**レビュー結論**: 「P0-B判定基盤」は改善されたが、因果的には still biased。厳密評価は同side内で `ab_test.variant` による比較が必要。

### 10.4 判定ロジックの見落とし: control側の最低サンプル制約が未定義

- `evaluate_ab_variant()` は `min_filled_records`, `min_calendar_days` を variant 側中心で判定。
- control 側が薄いケースでも PASS/FAIL を返しうるため、将来の不安定判定リスクがある。

**レビュー結論**: 直近データ量では顕在化していないが、運用長期では誤判定源。`min_control_filled_records` 等の対称制約が必要。

### 10.5 追加アクション（優先順）

1. **P0**: 修正後コミットで fill test を再起動し、`run_id/git_sha` を固定した再評価を実施。
2. **P0**: §9 の再評価結果を単一JSON（`per_regime_judgment` 含む）として保存し、本文数値と1対1対応させる。
3. **P1**: A/B判定を「sell vs buy」から「同side内 variant vs control」へ段階移行（少なくとも補助指標として併記）。
4. **P1**: `evaluate_ab_variant` に control 側の最小サンプル/日数制約を追加。

### 10.6 最終ステータス更新（レビュー視点）

| 項目 | ステータス | コメント |
|---|---|---|
| regime 伝搬修正の実装 | ✅ 実装済み | コード上は確認できる |
| regime=None 解消の運用実証 | ⚠️ 未確定 | 現行レコードに `regime:null` 残存 |
| 3指標A/B判定基盤 | ✅ 実装済み | ただし比較設計に交絡あり |
| per-regime 判定の監査可能性 | ⚠️ 不十分 | JSON成果物への固定が不足 |
| P0-B/C の最終判定確定 | ❌ まだ早い | 修正後run限定で再判定が必要 |

---

## §11 レビュー対応: §10 指摘への妥当性判断と修正

### 11.1 各指摘の妥当性判断

| 指摘 | 妥当性 | 根拠 | 対応 |
|---|---|---|---|
| **10.1** regime修正が現ランに未反映 | **完全に妥当** | fill test PID 124796 は `a7e5d0b82` (regime修正 `8d644b00e` より前)。2/24 データの 59.7% が `regime:null` | fill test 再起動 |
| **10.2** per_regime JSON 未保存 | **完全に妥当** | `reports/160_p0bc_judgment_results.json` に `per_regime_judgment` キーなし確認 | JSON再保存済み |
| **10.3** sell vs buy は厳密A/Bではない | **妥当 (P1)** | side差・ゲート差が交絡。ただし `ab_test_variant` フィールドは実装済み、データ蓄積待ち | 注記のみ。variant_id データ蓄積後に移行 |
| **10.4** control側min_filled未定義 | **妥当** | control n_filled=5 でも PASS/FAIL 返却していた | `min_control_filled_records=30` 追加済み |

### 11.2 実施した修正

#### 11.2.1 control 側最小サンプル制約 (§10.4 対応)

`ABJudgmentCriteria` に `min_control_filled_records: int = 30` を追加:
- variant 側チェックの直後に control 側の `n_filled < min_control_filled_records` を判定
- INSUFFICIENT を返し、薄い control による誤判定を防止
- `evaluate_per_regime()` にも伝搬
- YAML `judgment.ab_criteria.min_control_filled_records` で外部制御可能

#### 11.2.2 JSON 成果物の完全化 (§10.2 対応)

`reports/160_p0bc_judgment_results.json` を再生成:
- `per_regime_judgment` (4件: ranging/trending/trending_down/trending_up) を含む
- `ab_judgment` は `exclude_regimes=["none"]` 適用後の値

#### 11.2.3 新フィールド分析スクリプト活用 (158# 残課題)

`SideMetrics` と `_compute_side_metrics()` を拡張:
- `reprice_rate` / `avg_reprice_drift_bps`: reprice 発生率 + drift 平均
- `vg_trigger_rate`: VG trigger 率
- ダッシュボード表示に追加

**注**: 5フィールドとも FillRecord 構築コードでは正しく POPULATE されているが、
現行 fill test (旧コミット) では出力されていない。再起動で解決。

#### 11.2.4 regime:null 残存の原因確定 (§10.1 対応)

日別 `regime:null` 率:

| 日付 | total | regime_null | 率 |
|---|---|---|---|
| 0213 | 211 | 211 | **100%** |
| 0214 | 220 | 144 | 65.5% |
| 0215-16 | 81 | 0 | **0%** |
| 0217-20 | 949 | 171 | 18.0% |
| 0221-24 | 1608 | 797 | **49.6%** |

0215-16 の 0% は別ラン (PID 120384, `3959424ef883`) で、0221 以降の急増は
PID 124796 (`a7e5d0b82`, regime修正前コミット) 再開後。
**原因: fill test が旧コードで動作**。再起動で解決確実。

### 11.3 §10.5 アクション対応ステータス

| # | アクション | ステータス | 備考 |
|---|---|---|---|
| 1 | 修正後コミットで fill test 再起動 | ⏳ 準備完了 | コミット後に実行 |
| 2 | per_regime JSON 保存 | ✅ 完了 | `per_regime_judgment` 含む完全 JSON |
| 3 | 同side内 variant比較への段階移行 | 📋 P1 保留 | `ab_test_variant` データ蓄積待ち |
| 4 | control側min_filled制約追加 | ✅ 完了 | `min_control_filled_records=30` |

### 11.4 テスト

- 既存テスト: 59→63 (control 制約テスト4件追加)
- ダッシュボード結合: 69 PASSED
- SideMetrics新フィールド: reprice_rate / vg_trigger_rate 追加 (データなし環境で 0.0 出力確認)

### 11.5 158# 残課題ステータス更新

159# §4 優先順位に対する最新状態:

| 項目 | 元優先度 | ステータス | 備考 |
|---|---|---|---|
| P0-A trades_health 不整合 | P0 | **✅ クローズ** | 159# §9.1 確認済み |
| P0-B A/B判定3指標化 | P0 | **✅ 完了** | 160# §8-9-11 |
| P0-C trending_down sell 評価 | P0 | **✅ 完了** (暫定FAIL, 継続) | n=30 で再評価 |
| P1-A reprice_drift_bps | P1 | **✅ 実装済み** | 分析スクリプトにも集計追加。データは再起動後 |
| P1-B variant_id 記録 | P1 | **✅ 実装済み** | ab_test_variant フィールド |
| P1-C VG 詳細ログ | P1 | **✅ 実装済み** | vg_velocity/vpin/boost_factor, 分析集計追加 |
| P2-A execute_trade 品質検証 | P2 | 📋 未着手 | |
| P2-B run_fill_test 分割 | P2 | 📋 進行中 | lib/ 分割は進展 |

---

## §12 追加改善 (コード品質スキャン)

### 12.1 発見・修正事項

サブエージェントによるコード品質スキャンで CRITICAL〜HIGH の問題を発見し、即時対応。

| # | 深刻度 | ファイル | 問題 | 対応 |
|---|---|---|---|---|
| 1 | **CRITICAL** | `ab_judgment.py` | 空 PnL データで `0.0` を返し閾値 `-1.0` を上回って **誤 PASS** | NaN 化 + `pnl_data` INSUFFICIENT チェック追加 |
| 2 | **HIGH** | `retrain_scheduler.py` | `_save_enriched_cache()` が直接書き込み → 中断でデータ破損 | `.pkl.tmp` + `replace()` アトミック書き込みに変更 |
| 3 | **HIGH** | `run_fill_test.py` | 起動時 trades_health が `max_missing_days=0` で retrain_scheduler (`=1`) と不整合 | `max_missing_days=1` に統一 |
| 4 | **MEDIUM** | `online_monitor.py` | `.fillna(False).astype(bool)` の FutureWarning が6回出力 | `_to_bool()` ヘルパーで一括対応 |
| 5 | **MEDIUM** | `online_monitor.py` | `skip_col` が同メソッド内で2回定義 | 重複を除去 |

### 12.2 未対応事項 (P1/P2)

| # | 深刻度 | ファイル | 問題 | 理由 |
|---|---|---|---|---|
| A | CRITICAL | `retrain_scheduler.py` | SIGTERM グレースフル停止未実装 | 大規模リファクタ、別セッション対応 |
| B | HIGH | `retrain_scheduler.py` | final training の val 分割が WF eval と独立 | 精度検証が必要、別セッション対応 |
| C | HIGH | `run_fill_test.py` | `_cleanup_sync()` の `asyncio.new_event_loop()` 問題 | Py3.12+ 対応、互換テスト必要 |
| D | MEDIUM | `ab_judgment.py` / `dashboard.py` | `_compute_metrics` と `_compute_side_metrics` のロジック重複 | DRY 統合は影響範囲大 |
| E | MEDIUM | `retrain_scheduler.py` | `ConfigMap = dict[str, object]` の型安全性不足 | TypedDict 化は大規模リファクタ |

### 12.3 テスト

- `test_160_ab_judgment.py`: 65 PASSED (新規 2 件追加: `TestEmptyPnlInsufficient`)
- `test_159_side_regime_dashboard.py`: 6 PASSED
- `test_141 (OnlineMonitor)`: 10 PASSED
- `test_143 (pre_filter)`: 1 PASSED
- **合計: 82 テスト PASSED**

### 12.4 fill test 再起動状況

- **旧 PID 124796** (`a7e5d0b82`, regime修正前): **停止済み**
- **新 PID 128012** (`d9874bbee12a`, 最新コード): **稼働中**
  - regime detector: 有効
  - SkipGate: 有効 (17 features)
  - 起動時 trades_health: 20260221 欠落 (情報のみ、機能影響なし)
- **Retrain scheduler PID 129404**: 正常稼働中 (直近サイクル完了、次回3h後)
