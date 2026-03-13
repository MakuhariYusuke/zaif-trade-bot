# 050# 実装完了報告 + ログ分析

| 項目 | 内容 |
|---|---|
| 番号 | 050 |
| フェーズ | ph2 |
| 種別 | impl (実装) + log_analysis |
| 参照 | `049_ph2_rev_048.md`, `048_ph2_exit_timing_analysis_and_e3_data_collection.md` |
| 作成日 | 2026-02-15 |
| コミット | `51c02be69` (P0+P1 実装), YAML `side_offset.sell=0.07` 有効化後に再起動 |

---

## §1 実装済み項目 (P0+P1 全7件)

### P0 (即時修正, 4件)

| # | 内容 | 対応箇所 |
|---|---|---|
| P0-1 | 048 日付メタ修正 (2025→2026) | `048_ph2_exit_timing_analysis_and_e3_data_collection.md` |
| P0-2 | `main()` clean-only metrics | `run_fill_test.py` L1643: `filter_clean_records()` → `compute_fill_metrics()` |
| P0-3 | exit code FINAL整合 | `main()`: FINAL+PASS→0, INTERIM/PROVISIONAL+PASS→2, FAIL→1 |
| P0-4 | data_quality セクション | judgment出力に `clean_records`, `quarantine_records`, `clean_rate`, `as_coverage` 追加 |

### P1 (7日スパン, 3件)

| # | 内容 | YAML設定 |
|---|---|---|
| P1-1 | E3 サンプリング | `e3.sampling_ratio: 0.33` (1/3のみ60s/120s計測) |
| P1-2 | Side別 offset | `side_offset.sell: 0.07` (buy は共通値 0.05 継承) |
| P1-3 | Fast fill defense | `fast_fill_defense: enabled: true, threshold_sec: 5.0, offset_boost: 2.0` |

テスト: 497 tests passing (+14 new)

---

## §2 ログ分析結果 (385 records, 291 fills)

### 2.1 PnL 統計

| 指標 | 値 | 備考 |
|---|---|---|
| Overall Mean PnL | **-0.53 bps** | 赤字 |
| Buy Mean PnL | -0.26 bps (n=151) | |
| Sell Mean PnL | **-0.83 bps** (n=140) | Buy の 3.2x 悪い |
| Fast fill (<10s) PnL | **-0.88 bps** (n=122) | 全 fill の 42% |
| Slow fill (>=10s) PnL | -0.28 bps (n=169) | Fast の 3.1x 改善 |

### 2.2 AS 率

| 指標 | 値 | 目標 |
|---|---|---|
| AS(deadzone ±2bps) | **41.6%** | <40% |
| AS(raw) | 35.1% | |

AS(dz) > AS(raw) → deadzone が AS を増幅している可能性あり。

### 2.3 Fast Fill パターン

25件の fast fill + negative PnL (0214のみ):
- 最悪: **-31.40 bps** (sell), -11.82 bps (sell), -6.15 bps (sell)
- Fast fill は全 fill の 42%, PnL は -0.88 bps vs slow -0.28 bps (3x 悪化)
- **sell 側の fast fill に大損が集中** → `fast_fill_defense` + `side_offset.sell=0.07` で対策

### 2.4 運用問題

| 問題 | 詳細 | 影響 |
|---|---|---|
| 残高枯渇 | JPY: 885 → buy不可, BTC: 0.0007 → sell不可 | 12回の insufficient balance |
| Cancel race error | 271件のERROR (大半: "Failed to cancel the order") | 実害低 (既fill済みの注文cancel) |
| time_filter 長時間 | 79回の sleep (各120s) | 最長2h連続sleep |

### 2.5 Sell PnL悪化の根拠

| 推定原因 | 証拠 |
|---|---|
| offset 不足 | 共通 offset 0.05 では sell 側の AS を吸収不十分 |
| Fast fill 集中 | -31.4, -11.8, -6.1 bps の大損は全て sell |
| 構造的非対称 | BTC/JPY市場で sell maker は buy taker に狙われやすい |

→ `side_offset.sell: 0.07` (共通比 +40%) で保守化を適用済み。

---

## §3 再起動情報

| 項目 | 値 |
|---|---|
| PID | 53560 |
| run_id | `1771095285_383ebf85` |
| git_sha | `51c02be69` |
| 開始時刻 | 2026-02-15 03:54:45 JST |
| 期間 | 168h (7日間) |
| 既存データ | 385 records (clean 236 + quarantine 149) 引き継ぎ |
| 新機能 | E3 sampling (0.33), fast_fill_defense (on), side_offset.sell (0.07) |

### 次回モニタリングチェックポイント

1. **24h後**: sell PnL が改善しているか (target: -0.5 bps 以内)
2. **48h後**: fast_fill_defense の発動頻度と offset_boost 効果
3. **72h後**: E3 sampling によるサイクル時間改善の定量評価
4. **7日後**: FINAL 判定 → 全指標の最終評価

---

## §4 残課題 (P2)

| # | 内容 | 優先度 |
|---|---|---|
| P2-1 | tick path TP/SL 検証 (049# §6.3) | MEDIUM |
| P2-2 | round-trip 対応評価 (buy→sell 対応) | MEDIUM |
| P2-3 | 残高管理の自動化 (insufficient balance 防止) | LOW |
| P2-4 | レジーム独立計算 (fill cycle非依存) | LOW |
