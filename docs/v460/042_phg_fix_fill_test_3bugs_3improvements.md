# 042# fill test 3バグ修正 + 3追加改善 + ゾンビプロセス発見

**日時**: 2026-02-14  
**種別**: hotfix + improvement + critical discovery  
**コミット**: `d38fe0653` (3バグ修正), `04b5f4dd1` (3追加改善), `0c8f71367` (doc)  
**前セッション**: 041# (5施策コミット 33a78e742)

---

## 1. 再起動前の状況

| 指標 | 値 | 補足 |
|------|-----|------|
| 総レコード | 345 | 211 (2/13) + 134 (2/14) |
| fill_rate | 75.7% | G1.1 PASS 基準: ≥60% |
| AS_ratio | 42.5% | G1.1 PASS 基準: ≤50% |
| AS_raw | 31.4% | deadzone 非適用 |
| 累積PnL | **-116.64 JPY** | 損失方向 |
| cancel reasons | api_error:32, unknown:26, timeout:26 | |
| 稼働時間 | 23.1h | PID 30024 |
| regime | 全件 None | 037# 以前のプロセス |

---

## 2. バグ発見経緯

041# コードで再起動 → ログ解析で **3件の重大バグ** を発見:

### Bug 1: time_filter がジャンク FillRecord を生成

- **症状**: スキップされたサイクルが `FillRecord(cancelled=True, cancel_reason="time_filter")` を生成
- **影響**: fill_rate が人為的に低下（分母が膨張）
- **原因**: `run_single_cycle()` 内で time_filter チェック → FillRecord return
- **修正**: `_is_time_filtered()` を `run_continuous()` ループ先頭で呼び、True の場合はレコードを生成せず sleep only

### Bug 2: BTC 残高不足で sell 注文が全失敗

- **症状**: BTC=0.0007 < 0.001 (Coincheck 最小ロット) → sell 注文エラー
- **影響**: sell 側全件 api_error（本来は insufficient_funds だがエラー分類も不備）
- **原因**: 残高不足の事前チェックなし
- **修正**: `_check_balance_for_side()` pre-flight check を追加。不足時はサイドを反転して次サイクルで反対方向を試行

### Bug 3: 動的 loss_cap が reserved 残高を無視

- **症状**: loss_cap = 426 JPY（実際の口座価値 ≈ 19,174 JPY の 2.2%）
- **影響**: 本来 959 JPY であるべき loss_cap が過小 → 不必要に早い停止リスク
- **原因**: `_update_dynamic_loss_cap()` が `JPY_RESERVED`, `BTC_RESERVED` を集計対象外
- **修正**: Coincheck API の `jpy_reserved`, `btc_reserved` を total に加算

---

## 3. 追加改善 (3件)

### Fix 4: Adapter エラー本文を例外メッセージに含有

- **問題**: `_make_api_request` の HTTPError ハンドラが `raise NetworkError(f"...{e}")` — response body を含まない
- **影響**: Coincheck の日本語エラー文 (e.g., "所持金額が足りません") が呼び出し元で解析不能
- **修正**: `raise NetworkError(f"...{e} | body={body}")` に変更

### Fix 5: 日本語エラーメッセージの正しい分類

- **問題**: エラー分類が英語キーワード ("insufficient", "balance") のみ
- **影響**: "Amount BTC の所持金額が足りません" → `api_error` に誤分類（正しくは `insufficient_funds`）
- **修正**: `"所持金額"`, `"足りません"` を insufficient_funds 判定条件に追加

### Fix 6: 起動時の滞留注文自動クリア

- **問題**: 旧プロセスが残した未約定注文 (order 8665514703) のポーリングに 303秒浪費
- **影響**: 新プロセス起動後 ~5分間の非生産的待機
- **修正**: `_cancel_stale_orders()` メソッドを追加、`run_continuous()` 開始直後に `get_open_orders()` → 全件キャンセル

---

## 4. 口座残高 (再起動時)

| 通貨 | free | reserved | 備考 |
|------|------|----------|------|
| JPY | 1,011.77 | 0.0 | 旧注文キャンセル後 |
| BTC | 0.0017 | 0.0 | 旧注文約定分を含む |

**推定口座総額**: ~19,150 JPY (BTC×10.67M + JPY)  
**動的 loss_cap**: 959 JPY (5%)

---

## 5. 修正後の動作確認

```
18:14:31 Config loaded: adapt=True, regime=True, time_filter=True, loss_cap_auto=True
18:14:32 [loss_cap] 動的キャップ算出: 残高=19173 JPY × 5% = 959 JPY (旧: 10000 JPY)  ✅
18:14:32 Resumed: n=345, last_side=buy, cycle_count=345
18:14:32 [time_filter] High-AS hour — sleeping 120.0s  ✅ (レコード不生成)
```

| 修正 | 検証結果 | 詳細 |
|------|---------|------|
| Fix 1 (time_filter) | ✅ PASS | FillRecord 不生成、sleep のみ |
| Fix 2 (balance pre-flight) | ✅ Ready | BTC=0.0017 で現在は充足 |
| Fix 3 (reserved loss_cap) | ✅ PASS | 959 JPY (旧426) |
| Fix 4 (adapter body) | ✅ Committed | 次回エラー時に本文付き例外 |
| Fix 5 (JP error) | ✅ Test PASS | 6パターン全正分類 |
| Fix 6 (stale cleanup) | ✅ Committed | 次回再起動時に自動実行 |

---

## 6. テスト

- **既存**: 56 passed (v460 unit tests)
- **新規追加** (6件):
  - `TestTimeFilterNoRecord` × 3: disabled / no_hours / empty_hours
  - `TestDynamicLossCapReserved` × 1: reserved キー認識
  - `TestJapaneseErrorClassification` × 1: 6パターン分類
  - `TestStaleOrderCleanup` × 1: メソッド存在 + async 確認

---

## 7. 変更ファイル

| ファイル | 変更概要 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | +6 修正 (time_filter, balance, loss_cap, JP error, stale cleanup) |
| `ztb/trading/live/exchanges/coincheck/adapter.py` | Fix 4: body in exception |
| `tests/unit/v460/test_regime_detector.py` | +6 テストケース |

---

## 8. Bug 7: ゾンビプロセスによるデータ汚染 (CRITICAL)

### 発見経緯
- Fix 1-3 適用後のログに **time_filter が効いていないサイクル** (Cycle 352-360) が出現
- 初期仮説: time_filter バイパスのバグ → 実際: **別プロセスの並行書き込み**
- `Get-CimInstance Win32_Process | Where CommandLine -match fill_test` で発覚

### 原因
- PID 50848 / 52444 が **13:24 から起動中** (約5時間ゾンビ化)
- CLI args のみ起動 (`--spread-offset-ratio 0.05 --start-side buy`) — **YAML config 未使用**
- 旧コード: time_filter なし、regime なし、adaptation なし、dynamic loss_cap なし、git_sha 未記録
- `Stop-Process -Id 30024` は PID 30024 のみを停止 → ゾンビは見逃し

### データ汚染の範囲

| git_sha | 件数 | 起源 |
|---------|------|------|
| (空) | **149** | ゾンビプロセス (git_sha 未記録 = 旧コード) |
| a9320c9a5 | 136 | PID 30024 (037# コード) |
| ca1bcaed1 | 70 | 更に以前のプロセス |

**149/355 件 (42%) がゾンビ由来の汚染データ**。

### 対応
1. PID 50848, 52444 を停止
2. PID 41812 (Fix 1-3 のみ) も停止
3. Fix 1-6 全適用コードで再起動 → PID 35232/10236

### 再起動後の確認 (18:38)

```
[loss_cap] 残高=19144 JPY × 5% = 957 JPY
[startup] Cancelled stale order: id=8665582567    ← Fix 6 動作確認
[startup] Stale order cleanup complete: 1/1 cancelled
Resumed from n=355
[time_filter] High-AS hour — sleeping 120.0s      ← Fix 1 動作確認
```

### 今後の対策案
- PID ロックファイルによるプロセス排他制御
- 起動時に既存 fill_test プロセスの自動検出 + kill
- ログフォーマットに PID 追加 (多重プロセス診断)
- git_sha=空のレコードデータクレンジング

---

## 9. 最終統計 (クリーンプロセス開始時)

| 指標 | 値 | 前回 (§1) | 差分 |
|------|-----|-----------|------|
| 総レコード | 355 | 345 | +10 (ゾンビ由来) |
| fill_rate | 75.2% | 75.7% | -0.5pp |
| AS_ratio | 42.7% | 42.5% | +0.2pp |
| 累積PnL | -129.83 JPY | -116.64 JPY | **-13.19 JPY** (悪化) |
| avg_pnl | -0.46 bps | — | — |
| avg_wait | 42.2s | — | — |

---

## 10. 現在の fill test ステータス

- **PID**: 35232/10236 (18:38:22 開始)
- **設定**: 168h, 120s interval, **Fix 1-6 全適用**
- **Terminal**: d931bc04-9df4-405d-b8d9-82eeec1e68b0
- **ゾンビ**: すべて停止済み

---

## 11. 残課題

- [ ] Bug 7 対策: プロセス排他制御の実装
- [ ] git_sha=空 の 149件のデータクレンジング
- [ ] "unknown" cancel_reason 26件の原因調査 (ゾンビ由来の可能性)
- [ ] fill_rate / AS_ratio の推移をモニタリング (Fix 1-6 全適用後のクリーンデータ)
- [ ] 累積 PnL の黒字転換トラッキング
- [ ] Codex パッケージによる外部レビュー (→ `042_codex_review_package.md`)

