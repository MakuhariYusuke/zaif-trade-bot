# 040# ph2: レジーム検知統合と fill test 中間報告

| key | value |
|---|---|
| 番号 | 040 |
| フェーズ | ph2 |
| 種別 | rpt (実装報告 + 中間計測) |
| 作成日 | 2026-02-14 |
| 前提文書 | 034#, 035# |
| コミット | `9cb484e0e` (037# コミットメッセージ — 番号重複注意※) |

> ※ コミットメッセージで `037#` を使用したが、別 AI が 037–039 を先に採番済みのため、  
> 文書番号は 040# に繰り上げ。今後の採番は `040#` 以降を使用する。

---

## 1. 実装概要

035# レビュー指摘 (§7 Week 1) を受け、以下を実装した。

### 1.1 新規ファイル

| ファイル | 内容 |
|---------|------|
| `scripts/v460/lib/regime_detector.py` | 軽量レジーム検知器 |
| `tests/unit/v460/test_regime_detector.py` | 検知器テスト (18 件) |

### 1.2 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ztb/metrics/fill_quality.py` | FillRecord に `regime`, `regime_confidence`, `regime_stability` 追加 |
| `configs/v460/fill_test.yaml` | `regime:` セクション追加 |
| `scripts/v460/run_fill_test.py` | FillTestConfig レジーム設定 + Runner 統合 + ログ強化 |
| `docs/v460/034_phg_rpt_action_space_analysis.md` | タイトル番号 036→034 修正 |

---

## 2. レジーム検知器の設計

### 2.1 状態空間 (035# §4.2 #1)

| 状態 | 意味 | 判定条件 |
|------|------|---------|
| `trending` | トレンド進行中 | `abs(price_change%) ≥ trend_threshold_pct` (default 0.5%) |
| `ranging` | レンジ・横ばい | トレンドでも高ボラでもない |
| `high_vol` | 急激なボラティリティ | `current_vol / baseline_vol ≥ high_vol_multiplier` (default 2.0x) |
| `unknown` | 判定不能 / サンプル不足 | データ不足 or 信頼度 < `min_confidence` |

### 2.2 ヒステリシス (035# §4.2 #2)

- 連続 `hysteresis_count` 回 (default 3) の一致で状態遷移確定
- 単発のノイズで状態が振動しない

### 2.3 信頼度ゲート (035# §4.2 #3)

- `confidence < min_confidence` (default 0.4) → 強制的に `unknown`
- `unknown` 時は方策 A/B の適応を停止する起点（Week 2 で実装予定）

### 2.4 計測方式

- 入力: `run_single_cycle()` で取得する `mid_at_fill` (約定時) or `order_price` (未約定時)
- バッファ: `window × 3` 観測まで保持 (baseline 算出用)
- 算出: `window` 区間の price change % + returns の std ratio

---

## 3. fill test 中間報告 (2026-02-14 17:00 JST 時点)

### 3.1 稼働状況

| 項目 | 値 |
|------|-----|
| プロセス PID | 30024 |
| 開始時刻 | 2026-02-13 19:40 JST |
| 稼働時間 | 約 21 時間 |
| 最新レコード | 2026-02-14 17:09 JST (08:09 UTC) |
| レジーム検知 | **未反映** (コード更新前のプロセスが継続中) |

### 3.2 計測データ

| 指標 | 値 | 備考 |
|------|-----|------|
| Total records | 335 | |
| Filled | 255 (76.1%) | |
| Cancelled | 80 (23.9%) | |
| 計測日数 | 2 日 | 3 日未到達 → PROVISIONAL |
| Buy/Sell | 168 / 167 | ほぼ均等 ✓ |

### 3.3 キャンセル理由

| 理由 | 件数 | 比率 |
|------|------|------|
| api_error | 30 | 37.5% |
| timeout | 24 | 30.0% |
| (その他/不明) | 26 | 32.5% |

### 3.4 G1.1 Gate 判定

```
Result: FAIL (PROVISIONAL — 3暦日&200件未達)
```

| Check | 値 | 閾値 | 判定 |
|-------|-----|------|------|
| E1 fill_rate_p90 | 0.7450 | ≥ 0.90 | **FAIL** |
| E2 cancel_ratio | 0.2388 | ≤ 0.30 | PASS |
| E3 queue_wait_median | 12.0s | ≤ 60s | PASS |
| E4 post_fill_30s_pnl | -0.45 bps | ≥ 0.0 (p ≥ 0.05) | PASS (p=0.105) |
| E5 adverse_selection | 0.4314 | ≤ 0.20 | **FAIL** |
| E5-raw (参考) | 0.4819 | ≤ 0.20 | FAIL |

### 3.5 PnL

| 項目 | 値 |
|------|-----|
| 平均 PnL | -0.45 bps |
| 中央値 PnL | -0.01 bps |
| 標準偏差 | 5.67 bps |
| 累積 PnL | -119.61 JPY |
| 損失キャップ消費率 | 1.2% (10,000 JPY 中) |

### 3.6 評価

| 項目 | 評価 |
|------|------|
| **E1 fill_rate** | 0.745 は 0.90 閾値に大きく不足。offset 調整の効果が限定的 |
| **E5 AS_ratio** | 0.43 は 0.20 閾値の 2 倍以上。逆選択が深刻 |
| **E4 PnL** | 平均 -0.45bps だが p=0.105 で統計的に有意でない → 暫定 PASS |
| **累積損失** | -120 JPY はキャップの 1.2% で安全圏内 |
| **サンプル** | 335 件/2 日。あと 1 日で PROVISIONAL → FINAL 昇格 |

### 3.7 懸念事項

1. **E5 AS_ratio 43%** — 約定の約半数が逆選択。offset 0.05 は板の薄い BTC/JPY には小さすぎか
2. **E1 fill_rate** — 日別 P90 が 0.745。cancel が多いのは `api_error` (30 件) が主因
3. **api_error 30 件** — Coincheck API の一時障害か、リクエスト頻度制限か要調査
4. レジーム検知は次回再起動後に反映される

---

## 4. 035# 対応状況チェックリスト

| 035# 項目 | 状態 | 対応内容 |
|-----------|------|---------|
| §2 #1 レジーム接続 (HIGH) | ✅ 実装済 | regime_detector.py + Runner 統合 |
| §2 #2 委任境界 (HIGH) | 📐 設計済 | 035# §3.1 統治モデルを YAML で体現 |
| §2 #3 再利用基準 (MED) | ⏳ 一部 | 軽量4状態で新設。既存 `MarketRegimeDetector` は ph3 以降 |
| §2 #4 番号不整合 (LOW) | ✅ 修正済 | 034# タイトル修正 |
| §7 Week 1 #1 fill record タグ | ✅ 実装済 | regime/confidence/stability 3 項目 |
| §7 Week 1 #2 日次集計 | ⏳ 未着手 | regime×side×offset_bin 集計 (データ蓄積待ち) |
| §7 Week 1 #3 unknown フォールバック | ⏳ 未着手 | unknown 比率閾値で適応停止 |
| §7 Week 2 レジーム別 A/B | ⏳ 未着手 | レジーム別閾値テーブル + 分岐ロジック |

---

## 5. fill test 確認方法

### 5.1 状況確認コマンド

```powershell
# 1. プロセス稼働確認
Get-Process python* | Format-Table Id, ProcessName, StartTime, @{N='CPU_s';E={[math]::Round($_.CPU,1)}}, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}} -AutoSize

# 2. レコード件数確認
Get-ChildItem results/v460/fill_test/fill_records_*.jsonl | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName | Measure-Object -Line).Lines) lines" }

# 3. メトリクス＋Gate判定
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --results-only --results-dir results/v460/fill_test

# 4. 結果JSON出力 (保存)
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --results-only --results-dir results/v460/fill_test --output results/v460/fill_test/g1_1_judgment.json

# 5. 詳細確認 (一時スクリプト)
.venv\Scripts\python.exe temp/check_fill_test.py
```

### 5.2 ログ確認

```powershell
# 最新ログの末尾
Get-Content results/v460/fill_test/logs/fill_test.log -Tail 20

# 方策A/B の適応ログを検索
Select-String -Path results/v460/fill_test/logs/fill_test.log -Pattern "方策" | Select-Object -Last 10
```

### 5.3 再起動手順 (レジーム検知有効化)

```powershell
# 現在のプロセスを停止 (Ctrl+C or kill)
Stop-Process -Id <PID>

# 再起動 (レジーム検知が enabled: true で自動有効化)
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 168
# or dry-run 確認
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 1 --dry-run
```

---

## 6. 次ステップ

| 優先度 | 項目 | 条件 |
|--------|------|------|
| P0 | fill test 継続 → 3 暦日到達で FINAL 判定 | 2/15 19:40 以降 |
| P0 | api_error 30 件の原因調査 | ログ分析 |
| P1 | fill test 再起動でレジーム検知有効化 | 3 暦日計測完了後 |
| P1 | unknown 比率フォールバック実装 (035# Week 1 #3) | レジームデータ蓄積後 |
| P2 | regime×side×offset_bin 集計 (035# Week 1 #2) | 同上 |
| P2 | レジーム別方策 A/B 分岐 (035# Week 2) | 同上 |
