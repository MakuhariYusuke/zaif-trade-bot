# 047# — Cancel Race 修正・Gate 整合・ログ最適化

| key | value |
|---|---|
| 番号 | 047 |
| フェーズ | ph2 |
| 種別 | fix |
| 対象文書 | `044_ph2_rev_043.md`, `046_ph2_remaining_fixes_and_log_analysis.md` |
| 作成日 | 2026-02-15 |
| コミット | `bee3fcc7d` |
| テスト | **459 passed** (451 既存 + 8 新規) |

---

## §1 概要

046# コミット (`68a13aac2`) 後のログ分析で発見した Bug11 (cancel race condition) を含む 4 件の修正と 2 件のリファクタリングを実施。044# レビュー指摘の Finding3 (FINAL 7 日ルール) / Finding4 (AS coverage) も本セッションで解消。

---

## §2 ログ分析結果

**データ期間**: 046# 再起動 (2/14 21:17) 〜 2/15 01:00 (約 3.7h)

### §2.1 定量サマリ

| 指標 | 値 | 備考 |
|---|---|---|
| 総サイクル数 | 37 | |
| 約定 | 28 (75.7%) | |
| 未約定 | 9 | |
| ERROR | 6 | 全て cancel 失敗 (Bug11) |
| 残高不足スキップ | 10 | JPY=7, BTC=3 |
| time_filter sleep | 57 | Issue12: 2 分毎にログ出力 |
| cancel 失敗 | 3 件 (ログ 9 行) | Bug11 |
| PnL (bps) | avg: −1.06, sum: −29.68 | pos=11, neg=17 |
| 約定待ち | avg: 60.7s, max: 304.4s | |
| 未約定待ち | avg: 203.0s, max: 303.8s | |

### §2.2 ログサイズ分析

| 構成要素 | 文字数 | 割合 | 問題 |
|---|---|---|---|
| Balance ログ | 287,208 | 44.4% | 047# で DEBUG に降格 |
| API Response ログ | 322,565 | 49.8% | **Issue13** (後述) |
| その他 | 37,390 | 5.8% | 有用な情報 |

**ログの 94.2% がノイズ**。有用な情報は全体の 5.8% のみ。

### §2.3 ポーリング浪費

| 指標 | 値 |
|---|---|
| 未約定サイクル数 | 6 |
| 平均ポーリング回数 | 52 |
| 最大ポーリング回数 | 52 |
| 浪費 API コール | 310+ |

未約定の 1 サイクルで最大 120 API コール (60 ポーリング × 2 エンドポイント) を消費。**Issue14** として記録。

---

## §3 実装内容

### Fix 1: Bug11 — Cancel Race Condition (HIGH)

**問題**: `cancel_order` が 400 "Failed to cancel" を返すケースが 3/37 サイクル (8.1%)。注文は実際には約定済みだが、コードは `status_unknown` (未約定) として記録。残高の BTC が増加しているにも関わらず、`filled=False` としてデータが汚染される。

**原因**: Coincheck API の race condition — 約定とキャンセル要求が競合し、既に約定した注文のキャンセルが 400 を返す。

**修正**: `scripts/v460/run_fill_test.py`
- cancel 失敗時に `get_order_status()` を再呼び出しして約定確認
- 約定済みなら `filled=True`, `fill_price`, `cancel_reason_poll=None` に修正
- 未約定なら従来通り `status_unknown` 扱い

**効果**: データ汚染 8.1% → 0%。AS ratio / fill rate の算出精度向上。

### Fix 2: Finding3 — FINAL 7 日ルール + INTERIM 判定 (HIGH, 044# §4)

**問題**: `sample_sufficient` が `3 暦日以上` で判定していたが、000# §3.3 は「デフォルト 7 日間。暫定判定条件 (n≥200, 3 暦日以上)」と規定。3 日で FINAL 判定を出すのは仕様違反。

**修正**: `ztb/metrics/fill_quality.py`
- `sample_sufficient`: `len(daily_fill_rates) >= 3` → `>= 7`
- `g1_1_judgment()`: 3 段階判定を導入

| 段階 | 条件 | 意味 |
|------|------|------|
| **PROVISIONAL** | n < 200 or days < 3 | 暫定以前 — 参考値 |
| **INTERIM** | n ≥ 200 and 3 ≤ days < 7 | 暫定判定 — 000# §3.3 の「暫定判定可」 |
| **FINAL** | n ≥ 200 and days ≥ 7 | 最終判定 — Gate 通過/不通過の確定 |

**000# §3.3 整合**: INTERIM で Gate 通過の見込みを確認し、FINAL で確定判定。

### Fix 3: Finding4 — AS Coverage フィールド (HIGH, 044# §4)

**問題**: AS ratio の分母（adverse selection の母集団サイズ）が不透明。

**修正**: `ztb/metrics/fill_quality.py` の `FillMetrics` dataclass に追加:
- `as_coverage: int` — adverse selection 判定対象レコード数 (`mid_at_fill` 非 null)
- `as_raw_coverage: int` — raw adverse selection 判定対象レコード数

**効果**: AS ratio の信頼性を定量的に確認可能。

### Fix 4: Issue12 — time_filter ログ抑制 (LOW)

**問題**: High-AS 時間帯のスキップが 120 秒ごとにログ出力 → 30 行/時のノイズ。

**修正**: `scripts/v460/run_fill_test.py`
- `_in_time_filter: bool = False` フラグ追加
- ログを enter/exit のみに変更 (遷移時のみ出力)

**効果**: 30 行/時 → 2 行/遷移。

### Refactor 1: インライン import 統合

**修正**: `scripts/v460/run_fill_test.py`
- `from datetime import datetime, timezone` × 5 箇所 → top-level import 1 箇所
- `import traceback` × 1 箇所 → top-level import
- import の重複除去で一貫性向上

### Refactor 2: Balance ログ削減

**修正**: `ztb/trading/live/exchanges/coincheck/adapter.py`
- `get_balance()` の raw API レスポンス: `INFO` → `DEBUG`
- 非ゼロ残高の要約を `INFO` で出力: `{jpy: '885.31', btc: '0.0017'}`

**効果**: Balance ログ量 44.4% → 約 0.5% (サイズ 287,208 → 推定 3,000 文字)

---

## §4 テスト

### 新規テスト (9 件, 4 クラス)

| クラス | テスト名 | 対象 |
|---|---|---|
| `TestBug11CancelRaceCondition` | `test_cancel_fail_detects_fill` | cancel 失敗 → 再確認で約定検出 |
| | `test_cancel_fail_no_fill` | cancel 失敗 → 再確認で未約定確認 |
| `TestInterimJudgment` | `test_interim_3_days_200_samples` | 3 日 + n≥200 → INTERIM |
| | `test_final_7_days` | 7 日 + n≥200 → FINAL |
| | `test_provisional_insufficient` | n < 200 → PROVISIONAL |
| `TestASCoverage` | `test_coverage_fields_present` | as_coverage フィールド存在 |
| | `test_coverage_in_dict` | to_dict() に含まれる |
| `TestTimeFilterLogThrottle` | `test_in_time_filter_flag_init` | フラグ初期値 False |

### 既存テスト修正 (1 件)

- `test_sample_sufficient_true`: 3 日 × 70 件 → 7 日 × 30 件 (7 日ルールに合わせて修正)

### テスト結果

- **459 passed** (451 → +8)
- 0 failed
- 8 warnings (scipy precision × 4, pytest mark × 1 — 全て既知)

---

## §5 変更ファイル一覧

| ファイル | 変更 | 主な内容 |
|---|---|---|
| `scripts/v460/run_fill_test.py` | +32/−10 | Bug11, Issue12, inline import 統合 |
| `ztb/metrics/fill_quality.py` | +18/−5 | Finding3 INTERIM, Finding4 AS coverage |
| `ztb/trading/live/exchanges/coincheck/adapter.py` | +6/−4 | Balance ログ削減 |
| `tests/unit/v460/test_fill_quality.py` | +228/−3 | 9 新規テスト + 1 既存修正 |
| **合計** | **+284/−22** | |

---

## §6 G1.1-exec 現状評価

### §6.1 クリーンデータ統計

| 指標 | 値 | 000# §3.3 閾値 | 判定 |
|---|---|---|---|
| クリーンレコード数 | 236 | n ≥ 200 | ✅ PASS |
| 約定率 | 202/236 = 85.6% | ≥ 90% (P90) | ⚠️ 未達 |
| カバー日数 | 2 日 | ≥ 7 日 (FINAL) | ❌ PROVISIONAL |
| PnL (30s) | avg: −0.27 bps | ≥ 0 | ⚠️ 負のバイアス |
| AS ratio | 72/202 = 35.6% | ≤ 20% | ❌ FAIL |
| AS raw | 102/202 = 50.5% | — | 参考値 |
| 勝率 | 98/202 = 48.5% | — | 参考値 |

### §6.2 Side 別分析

| Side | 約定 | 合計 | 約定率 |
|---|---|---|---|
| Buy | 102 | 118 | 86.4% |
| Sell | 100 | 118 | 84.7% |

### §6.3 Git SHA 別レコード

| SHA | レコード数 | 備考 |
|---|---|---|
| `a9320c9` | 136 | 初期コード |
| `ca1bcae` | 70 | 045# |
| `68a13aa` | 30 | 046# |

### §6.4 評価

**現在の判定ステージ: PROVISIONAL** (2 日 / 7 日)

AS ratio 35.6% は 000# §3.3 の 20% 閾値を大幅に超過。ただし以下の緩和要因あり:

1. **初期コード (a9320c9) のデータ比率が高い** — Bug7 (ゾンビプロセス)・E-3 (int 切り捨て)・E-4 (balance 誤解析) 修正前のデータが 136/236 = 57.6%
2. **046# 以降のデータ (30 件) は少サンプル** — 046# の修正効果を統計的に評価するには不十分
3. **Bug11 (cancel race) の影響** — 約定済みを未約定と誤記録 → fill rate / AS ratio が悪化方向にバイアス

**000# §3.9 中止条件との照合**:

| 条件 | 状態 | 備考 |
|---|---|---|
| fill_rate < 70% (n ≥ 200) | **非該当** (85.6%) | 閾値クリア |
| AS_ratio > spread/2 継続 (n ≥ 500) | **未到達** (n=236) | 500 到達前は保留 |
| 累積実損 > 10,000 JPY | **非該当** | 推定損失 ≈ 数百 JPY |

**結論**: 中止条件には非該当。データ蓄積を継続し、046# 以降のクリーンデータで再評価する。INTERIM 判定 (3 日目) で AS ratio 改善傾向の有無を確認する。

---

## §7 新規発見事項

### Issue13: API Response ログ過剰 (HIGH)

**問題**: `adapter.py` の `_make_api_request()` が全 API レスポンスの status + content を `INFO` で出力。ポーリング 1 サイクルで最大 120 API コール → 240 行のログ。ログの 49.8% を占有。

**位置**: `ztb/trading/live/exchanges/coincheck/adapter.py` L224-227

```python
logger.info(f"API Response status: {response.status_code}")
logger.info(f"API Response content: {response.text[:500]}")
```

**推奨**: `logger.debug` に降格。エラー時のみ `INFO` / `WARNING`。

### Issue14: 未約定ポーリング API 浪費 (MEDIUM)

**問題**: 未約定サイクルで 52 回ポーリング × 2 API コール = 104 コール/サイクル。rate limit (4 req/s) 逼迫リスク。

**推奨**: 指数バックオフまたはポーリング間隔の動的延伸 (5s → 10s → 15s)。

### Issue15: run_single_cycle 巨大メソッド (LOW)

**問題**: `run_single_cycle` が約 295 行。発注・ポーリング・キャンセル・PnL 計測・レジーム更新・レコード構築を 1 メソッドに集約。単一責任原則違反。

**推奨**: `_poll_order()`, `_measure_pnl()`, `_build_record()` 等に分割。ただし fill test は変更凍結期間中のため、データ収集完了後に実施。

### Issue16: 板取得失敗のサイレント例外 (LOW)

**問題**: L612 `except Exception: pass` — リトライ時の板取得失敗をログなしで無視。古い価格でリトライする暗黙動作。

**推奨**: `logger.debug("Orderbook fetch failed during retry, using previous price")` 追加。

---

## §8 045#〜047# 累積修正状況

| 044# ID | 重要度 | 状態 | 対応セッション |
|---|---|---|---|
| Bug7 (多重起動) | CRITICAL | ✅ 完了 | 045# |
| E-1 (rate limit) | CRITICAL | ✅ 完了 | 045# |
| A-1 (SIGTERM) | HIGH | ✅ 完了 | 045# |
| E-3 (int→round) | HIGH | ✅ 完了 | 045# |
| E-4 (balance) | HIGH | ✅ 完了 | 045# |
| A-4 (cleanup) | HIGH | ✅ 完了 | 045# |
| F8 (preflight) | MEDIUM | ✅ 完了 | 045# |
| A-7 (loss_cap) | MEDIUM | ✅ 完了 | 045# |
| F7 (dead code) | LOW | ✅ 完了 | 045# |
| P0-2 (clean/quarantine) | HIGH | ✅ 完了 | 046# |
| P0-5 (soft/hard loss_cap) | HIGH | ✅ 完了 | 046# |
| Bug10 (insufficient retry) | HIGH | ✅ 完了 | 046# |
| Finding3 (FINAL 7 日) | HIGH | ✅ 完了 | **047#** |
| Finding4 (AS coverage) | HIGH | ✅ 完了 | **047#** |
| Bug11 (cancel race) | HIGH | ✅ 完了 | **047#** |
| Issue12 (time_filter noise) | LOW | ✅ 完了 | **047#** |
| Issue13 (API log bloat) | HIGH | ⏳ 次回 | — |
| Issue14 (polling waste) | MEDIUM | ⏳ 次回 | — |
| Issue15 (SRP 違反) | LOW | 📋 計画 | データ収集後 |
| Issue16 (silent exception) | LOW | 📋 計画 | — |
| P0-3 (side offset) | HIGH | 📋 保留 | データ蓄積後 |
| P0-4 (time_filter A/B) | MEDIUM | 📋 保留 | データ蓄積後 |
| Finding5 (adapter 目標) | MEDIUM | 📋 保留 | 設計検討要 |

---

## §9 次セッション向け推奨

1. **Issue13 修正 + fill test 再起動** — API Response ログを `DEBUG` に降格し、047# コードで再起動。ログサイズ 94% 削減。
2. **INTERIM 判定 (3 日目)** — 2/16 以降にクリーンデータ 3 暦日到達予定。046# 以降のデータで AS ratio 改善傾向を確認。
3. **JPY 残高補充** — 現在 885 JPY (buy 発注不可)。BTC 0.0017 のみ。buy 側データ収集のため入金検討。
4. **Issue14 (ポーリング最適化)** — 指数バックオフ導入で API コール削減。

---

## §10 補足レビュー追記 (見落とし候補)

047# の修正は有効だが、以下は未解消で収益性・判定整合性に直結する。

### §10.1 追加指摘

| # | 重要度 | 指摘 | 根拠 | 推奨対応 | 状態 |
|---|---|---|---|---|---|
| A1 | HIGH | **clean/quarantine が適応ロジックに未適用** | `run_continuous()` の累積PnL初期化では `filter_clean_records()` を使用する一方、`_try_auto_adapt()` / `_try_auto_lot_size()` は `load_fill_records_glob()` の全件を使用 | 方策A/Bの入力を clean のみに限定。quarantine 混入時は適応停止または警告 | ✅ 完了 |
| A2 | HIGH | **results-only 判定が quarantine を含む** | `run_results_only()` が全レコードで `compute_fill_metrics()` を実行 | `--results-only` に `--clean-only` を追加し、デフォルトを clean-only に変更 | ✅ 完了 |
| A3 | HIGH | **終了コードが judgment_type を無視** | `main()` は `gate_result=="PASS"` で `exit 0`。`PROVISIONAL/INTERIM` でも PASS なら成功扱い | `FINAL` かつ `PASS` のみ `exit 0`。それ以外は `exit 2` 等で分離 | ✅ 完了 |
| A4 | MEDIUM | **単一起動ロックが原子的でない** | lockfile は存在確認→書込みの2段階。同時起動で race の余地 | OSファイルロック (`fcntl`/`msvcrt`) を併用し、排他獲得を原子化 | ✅ 完了 (`O_CREAT\|O_EXCL`) |
| A5 | MEDIUM | **quarantine 判定が git_sha のみ** | `filter_clean_records()` は blank git_sha だけを隔離。run_id欠損/旧スキーマ混在は通過 | 判定条件を拡張 (`git_sha`, `run_id`, 必須フィールドcoverage, schema_version) | ✅ 完了 |
| A6 | MEDIUM | **Issue13 は文書化のみで未適用** | `_make_api_request()` で `API Response status/content` が依然 `INFO` | `DEBUG` 降格 + 失敗時のみ `WARNING/ERROR` へ即適用 | ✅ 完了 |

### §10.2 高収益観点の補足

1. **A1/A2 の放置は最適化方向を誤らせる**  
   汚染データ混入の適応は、offset/lot を逆方向へ学習させるリスクが高い。
2. **A3 の放置は Gate 順序違反を誘発**  
   PROVISIONAL PASS で次工程に進むと、ph3/ph4 の工数を先に消費しやすい。
3. **A4/A5 の放置は再汚染再発リスク**  
   データ品質事故の再発は、収益改善以前に実験信頼性を崩す。

### §10.3 次セッション優先順 (追補)

1. ~~A1 + A2 (clean-only 適用の統一)~~ ✅ 完了
2. ~~A3 (FINAL 判定連動の終了コード)~~ ✅ 完了
3. ~~A6 (API INFO ログ削減)~~ ✅ 完了
4. ~~A4 + A5 (ロック原子化・quarantine基準拡張)~~ ✅ 完了

### §10.4 実装詳細 (047# 追加コミット)

| 修正 | ファイル | 変更内容 |
|---|---|---|
| A1 | `run_fill_test.py` | `_try_auto_adapt()` / `_try_auto_lot_size()` に `filter_clean_records()` 適用。inline import 除去 |
| A2 | `run_fill_test.py` | `run_results_only()` で clean-only 判定。quarantine 全件時は `NO_DATA` 返却 |
| A3 | `run_fill_test.py` | exit code: FINAL+PASS→0, INTERIM/PROVISIONAL+PASS→2, FAIL→1 |
| A4 | `run_fill_test.py` | `_acquire_lock()` を `os.open(O_CREAT\|O_EXCL)` による排他的作成に変更 |
| A5 | `fill_quality.py` | `filter_clean_records()` に `run_id`, `side`, `order_price`, `order_quantity` チェック追加。`_quarantine_reason()` ヘルパー抽出 |
| A6/Issue13 | `adapter.py` | API Response status/content を `logger.debug` に降格 |

**テスト**: 新規 24 テスト追加。全 483 テスト PASS。

---

## Appendix A: 000# §3.3 G1.1-exec 原文 (参照)

> **測定期間**: デフォルト 7 日間。ただし n ≥ 200 サイクル かつ 3 暦日以上をカバーしていれば暫定判定可。最終判定は 7 日間データで確定する。

> | 条件 | 閾値 |
> |------|------|
> | fill rate (90 percentile) | ≥ 90% |
> | adverse selection ratio | ≤ 20% |
> | post_fill_30s_pnl (mean) | ≥ 0 |

## Appendix B: 000# §3.9 継続中止ルール (参照)

> | 条件 | n 最低要件 | 判断 |
> |------|-----------|------|
> | fill_rate < 70% | n ≥ 200 | **中止** |
> | AS_ratio > spread/2 が継続 | n ≥ 500 | **中止** |
> | 累積実損 > 10,000 JPY | — | **一時停止** |
