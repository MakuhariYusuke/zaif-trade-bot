# 043# Codex レビュー用情報パッケージ

**目的**: 外部 AI エージェント (Codex) による fill test 改善策レビュー  
**日時**: 2026-02-14 18:40 JST  
**プロジェクト**: v460 "Microstructure Edge" BTC/JPY maker-only 自動売買  
**取引所**: Coincheck (日本国内、現物取引)

---

## 1. システム概要

BTC/JPY のスプレッド内に maker limit 注文を交互 (buy/sell) に配置し、fill quality (約定率、逆選択率) を実測する fill test ランナー。

### 戦略ロジック
1. orderbook の best_bid/best_ask を取得
2. スプレッド × `spread_offset_ratio` (0.05) だけ内側に指値注文 (`time_in_force=post_only`)
3. 5秒間隔でポーリング → 約定 or 5分タイムアウトでキャンセル
4. 約定後 30 秒の mid price 変化で adverse selection (AS) を判定
5. buy/sell を交互実行、120秒サイクル

### 判定基準 (G1.1-exec Gate)
- fill_rate ≥ 60%: PASS
- AS_ratio ≤ 50%: PASS
- 両方満たせばゲート通過 → 本番取引移行

---

## 2. 現在の統計 (355 records, ~27h)

### 全体メトリクス

| 指標 | 値 | ターゲット |
|------|-----|-----------|
| fill_rate | 75.2% (267/355) | ≥60% ✅ |
| AS_ratio (w/deadzone 2.0bps) | 42.7% (114/267) | ≤50% ✅ |
| AS_raw (deadzone なし) | 32.2% (86/267) | — |
| avg_pnl | -0.46 bps | — |
| avg_wait | 42.2s | — |
| cumulative PnL | **-129.83 JPY** | — |
| cancel reasons | api_error:34, timeout:27, unknown:26, status_unknown:1 | — |

### サイド別メトリクス

| サイド | 件数 | AS率 | avg PnL |
|--------|------|------|---------|
| buy | 139 | 43.2% | -0.15 bps |
| sell | 128 | 42.2% | -0.80 bps |

### ⚠️ データ品質問題 (Bug 7: 多重プロセス)

| git_sha | 件数 | 起源 |
|---------|------|------|
| (空) | **149** | **ゾンビプロセス** (13:24開始, CLI args, YAML未使用) |
| a9320c9a5 | 136 | PID 30024 (037# コード) |
| ca1bcaed1 | 70 | 更に以前のプロセス |

**149件 (42%) はゾンビプロセス由来のデータ汚染**。ゾンビプロセスは:
- YAML config なし (CLI 引数のみ)
- time_filter なし、regime なし、adaptation なし、dynamic loss_cap なし
- 旧バージョンのコード (git_sha 未設定 = 033# 以前)
- 同一の fill_records JSONL ファイルに並行書き込み

---

## 3. 口座残高 (18:38 JST)

| 通貨 | free | reserved | 備考 |
|------|------|----------|------|
| JPY | 1,006.46 | 10,665.82 | ゾンビプロセスの注文ロック |
| BTC | 0.0007 | 0.0 | 最小ロット (0.001) 未満 |

**推定口座総額**: ~19,144 JPY  
**動的 loss_cap**: 957 JPY (5%)

---

## 4. 設定ファイル (configs/v460/fill_test.yaml)

```yaml
symbol: btc_jpy
order_quantity: 0.001         # Coincheck 最小ロット
cycle_interval_sec: 120.0     # 2分間隔
order_timeout_sec: 300.0      # 5分タイムアウト
poll_interval_sec: 5.0        # 5秒ポーリング
post_fill_wait_sec: 30.0      # 約定後PnL計測
spread_offset_ratio: 0.05     # スプレッドの5%内側
min_offset_jpy: 1.0
max_order_retries: 2          # リトライ2回
as_deadzone_bps: 2.0          # ±2bps以内はAS判定しない

# 方策 A: パラメータ適応
adaptation:
  enabled: true
  interval_cycles: 50
  min_fill_rate: 0.80
  max_as_ratio: 0.15
  step_ratio: 0.01

# 時間帯フィルター
time_filter:
  enabled: true
  skip_utc_hours: [8, 9, 14, 16, 17, 18, 19]  # JST 17-04 の高AS時間帯

# 安全設計
safety:
  loss_cap_auto: true
  loss_cap_ratio: 0.05
```

---

## 5. 修正済みバグ (042# セッション)

### Fix 1: time_filter レコード汚染 (CRITICAL)
- 旧: `run_single_cycle()` 内で FillRecord(cancel_reason="time_filter") 生成
- 新: `run_continuous()` ルート先頭で `_is_time_filtered()` → sleep only, レコード不生成

### Fix 2: 残高 pre-flight check (CRITICAL)
- 旧: BTC < 0.001 で sell 注文失敗 → api_error 分類
- 新: `_check_balance_for_side()` で事前チェック、不足時サイド反転

### Fix 3: reserved 残高 loss_cap (HIGH)
- 旧: JPY + BTC のみ → cap=426 (実際の 2.2%)
- 新: JPY_RESERVED + BTC_RESERVED も加算 → cap=957 (正しい 5%)

### Fix 4: adapter body in exception (MEDIUM)
- 旧: `NetworkError("Coincheck API error: {e}")` — body なし
- 新: `NetworkError("...{e} | body={body}")` — Coincheck の日本語エラー文含有

### Fix 5: 日本語エラー分類 (MEDIUM)
- 旧: "所持金額が足りません" → `api_error`
- 新: "所持金額" / "足りません" → `insufficient_funds`

### Fix 6: 滞留注文自動クリア (HIGH)
- 旧: 前プロセスの注文をポーリング (303秒浪費)
- 新: 起動時に `get_open_orders()` → 全キャンセル → log

### Bug 7: 多重プロセス書き込み (CRITICAL — 未修正)
- 旧プロセス (PID 50848/52444, 13:24 開始) がゾンビ化
- 同一 JSONL ファイルに並行書き込み → 149件のデータ汚染
- `Stop-Process` で別 PID を停止しただけで、本体プロセスを見落とし
- **対策案**: PID ロックファイル、起動時に既存 fill_test プロセス検出 + kill

---

## 6. コードアーキテクチャ (scripts/v460/run_fill_test.py)

```
1371 行
├── FillTestConfig (dataclass, 30+ fields)
│   └── from_yaml(cls, yaml_cfg) → FillTestConfig
├── FillTestRunner
│   ├── __init__(adapter, config, yaml_cfg)
│   ├── resume_from_existing() → list[FillRecord]
│   ├── _next_side() → str  (buy/sell 交互)
│   ├── _get_mid_price() → float
│   ├── _compute_maker_price(side) → (price, spread)
│   ├── _is_time_filtered() → bool
│   ├── _check_balance_for_side(side) → bool
│   ├── _cancel_stale_orders() → int
│   ├── run_single_cycle() → FillRecord       # 1サイクル実行
│   ├── run_continuous(hours) → list[FillRecord]  # メインループ
│   ├── _try_save_batch(batch) → bool
│   ├── _save_batch_by_date(batch) → None
│   ├── _emergency_dump(batch, reason)
│   ├── _update_dynamic_loss_cap()
│   ├── _try_auto_adapt(total, filled)         # 方策A
│   ├── _try_auto_lot_size()                    # 方策B
│   └── _cleanup_sync()                         # atexit handler
├── run_results_only(results_dir) → dict
└── main() → None
```

---

## 7. 関連コンポーネント

| ファイル | 役割 |
|----------|------|
| `ztb/trading/live/exchanges/coincheck/adapter.py` | Coincheck REST API ラッパー (825行) |
| `ztb/metrics/fill_quality.py` | FillRecord dataclass, compute_fill_metrics, g1_1_judgment |
| `scripts/v460/lib/param_adapter.py` | 方策A: offset 自動適応 |
| `scripts/v460/lib/lot_sizer.py` | 方策B: ロット自動適応 |
| `scripts/v460/lib/regime_detector.py` | レジーム検知 (ranging/trending/high_vol) |
| `configs/v460/fill_test.yaml` | 全設定の一元管理 |

---

## 8. 起動ログ（最新クリーンプロセス 18:38:28）

```
Config loaded: YAML=configs/v460/fill_test.yaml, offset=0.05, lot=0.001,
  adapt=True, dynamic_lot=False, regime=True, time_filter=True, loss_cap_auto=True
[Regime] detector enabled: window=20, hysteresis=3
JPY free: 1006.46, BTC free: 0.0007, JPY_RESERVED: 10665.817
[loss_cap] 動的キャップ算出: 残高=19144 JPY × 5% = 957 JPY (旧: 10000 JPY)
[startup] Cancelled stale order: id=8665582567, side=buy, price=10665817.0, qty=0.001  ← Fix 6 動作確認
[startup] Stale order cleanup complete: 1/1 cancelled.
Resumed from existing records: n=355, last_side=buy, cycle_count=355
Starting fill test: 168.0h, interval=120.0s
[time_filter] High-AS hour — sleeping 120.0s  ← Fix 1 動作確認 (UTC 09)
```

---

## 9. ログファイルのサイクルイベント時系列 (抜粋)

```
# PID 30024 (旧プロセス、037# コード、YAML config あり)
17:43:49 === Cycle 345 (buy) ===
17:46:28 === Cycle 346 (sell) ===  ← PID 30024 continuing
17:49:07 === Cycle 347 (buy) ===
...
18:01:09 === Cycle 351 (buy) ===

# PID 15876 (041# コード + 042# Fix 1-3 適用前、18:04 起動)
18:04:59 [loss_cap] 残高=8513 JPY × 5% = 426 JPY (Bug 3 — reserved 未計上)
18:04:59 Resumed from n=345
18:04:59 === Cycle 346 (sell) SKIPPED (UTC 09)  ← time_filter 旧版 (レコード生成)
18:06:59 === Cycle 347 (buy) SKIPPED

# PID 50848/52444 (ゾンビ、13:24 起動、CLI args のみ、旧コード)
# ↓ time_filter なし — UTC 09 でも注文実行
18:08:14 === Cycle 352 (sell) ===
18:10:20 === Cycle 353 (buy) ===
18:13:05 === Cycle 354 (sell) ===

# PID 41812 (042# Fix 1-3 適用、18:14 起動)
18:14:32 [loss_cap] 残高=19173 JPY × 5% = 959 JPY (Fix 3 OK)
18:14:32 Resumed from n=345
18:14:32 [time_filter] sleeping  ← Fix 1 OK (レコード不生成)

# PID 50848/52444 (まだ生きている！)
18:15:17 === Cycle 355 (buy) ===
18:17:22 === Cycle 356 (sell) ===
18:21:47 === Cycle 357 (buy) ===
18:24:43 === Cycle 358 (sell) ===
18:27:46 === Cycle 359 (buy) ===
18:34:48 === Cycle 360 (sell) ===

# PID 10236 (042# Fix 1-6 全適用、18:38 起動 — 現在の唯一のプロセス)
18:38:30 [loss_cap] 残高=19144 JPY × 5% = 957 JPY
18:38:32 [startup] Cancelled stale order (Fix 6)
18:38:32 Resumed from n=355
18:38:32 [time_filter] sleeping
```

---

## 10. 未解決の課題・改善候補

### P0: CRITICAL — プロセス排他制御
- **問題**: 複数プロセスが同時に fill_records に書き込み → データ汚染
- **原因**: PID ベースの停止では見落としが発生
- **対策案**:
  - PID ロックファイル (`results/v460/fill_test/fill_test.pid`)
  - 起動時に `Get-CimInstance Win32_Process | Where CommandLine -match fill_test` で既存プロセス検出
  - ファイルロック (fcntl / msvcrt) で JSONL 排他書き込み

### P1: データクレンジング
- 149件のゾンビレコード (git_sha=空) をどう扱うか:
  - 削除して clean state で再集計
  - git_sha ベースでフィルタして有効データのみ分析
  - 全データ維持するが `data_quality` フラグを追加

### P2: 方策 A (適応) の効果検証
- `adaptation.enabled=true` since 041# (335+ records)
- まだ `adapt_interval_cycles=50` に未到達 (min_samples=50)
- `min_fill_rate=0.80` vs 実績 75.2% → offset 増加方向
- `max_as_ratio=0.15` vs 実績 42.7% → offset 減少方向
- **fill_rate と AS_ratio が相反** → 適応の方向が ambiguous

### P3: PnL の改善
- cumPnL = -129.83 JPY (赤字)
- avg_pnl = -0.46 bps
- sell 側が特に悪い (-0.80 bps vs buy -0.15 bps)
- **仮説**: sell 限は BTC 売却後に価格上昇する傾向 (逆トレンド about)
- **改善案**: トレンド検知による片側スキップ、spread_offset の非対称化

### P4: cancel_reason "unknown" 26件
- git_sha が空 → ゾンビプロセス由来の可能性大
- 本来は timeout or api_error であるべきだが、旧コードに cancel_reason デフォルト値なし

### P5: ログ改善
- PID をログフォーマットに追加 → 多重プロセス問題の診断
- rotation で旧プロセスの entries が紛失 → JSONL 形式のイベントログ検討
- API response の 500 char truncation → 重要な error body の欠損

### P6: BTC 残高管理
- BTC = 0.0007 (< 0.001 最小ロット) → sell 不可
- buy 約定でBTC増加 → sell 可能に → しかしゾンビプロセスが売却
- **改善案**: 残高不足検知 → buy 連続 or 成行買い補充

### P7: レジーム検知の活用
- 現時点で regime データが 0 件 (ゾンビプロセスは regime なし)
- 042# プロセスから regime 付きレコード生成開始
- **活用案**: regime=trending 時は offset 拡大、regime=ranging 時は offset 縮小

### P8: time_filter の妥当性
- skip_utc_hours: [8,9,14,16,17,18,19] → 1日の 7/24 = 29% を棄却
- 実データでの検証: UTC 8-9 / 14 / 16-19 の AS 率はどの程度高いか?
- time_filter 効果の事前事後比較が必要

---

## 11. Git コミット履歴 (最新)

```
0c8f71367 042# doc: fill test 3 bugs + 3 improvements report
04b5f4dd1 042# fix: 3 additional improvements (adapter body, JP error, stale cleanup)
d38fe0653 042# fix: 3 critical fill test bugs (time_filter, balance, loss_cap)
33a78e742 041# ph2: profitability improvements + dynamic loss_cap + AS optimization
0d29e65e2 040# ph2: regime integration report + fill test interim (335 records)
9cb484e0e 037# ph2: regime detection integration (035# Week 1)
```

---

## 12. Codex への質問リスト

Codex への主要な質問ポイント:

1. **データ汚染 (149件)**: 除外して再計測すべきか、全データで判断すべきか
2. **PnL マイナスの根因**: avg=-0.46bps は構造的問題か、統計的揺らぎか
3. **方策 A 適応パラメータ**: fill_rate 0.80 と AS_ratio 0.15 の閾値は適切か
4. **プロセス排他制御**: PID ロックファイル vs ファイルロック vs プロセス検出、最適なアプローチは
5. **時間帯フィルター**: UTC 7時間のスキップは攻撃的すぎないか
6. **spread_offset_ratio 0.05**: BTC/JPY スプレッド (~14bps) に対して最適か
7. **AS deadzone 2.0bps**: ノイズ除去には効果的だが、真の AS を見逃していないか
8. **リスク管理**: loss_cap 5% (957 JPY) は適切か、口座規模 (19K JPY) に対して

---

## 13. 主要ファイルの場所

| ファイル | パス |
|----------|------|
| fill test ランナー | `scripts/v460/run_fill_test.py` (1371行) |
| Coincheck adapter | `ztb/trading/live/exchanges/coincheck/adapter.py` (825行) |
| fill quality metrics | `ztb/metrics/fill_quality.py` |
| param adapter (方策A) | `scripts/v460/lib/param_adapter.py` |
| regime detector | `scripts/v460/lib/regime_detector.py` |
| YAML 設定 | `configs/v460/fill_test.yaml` (87行) |
| テスト | `tests/unit/v460/test_regime_detector.py` |
| fill records (Day 1) | `results/v460/fill_test/fill_records_20260213.jsonl` (211行) |
| fill records (Day 2) | `results/v460/fill_test/fill_records_20260214.jsonl` (144行) |
| ログ | `results/v460/fill_test/logs/fill_test.log` |
