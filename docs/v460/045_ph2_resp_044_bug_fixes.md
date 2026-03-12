# 045# 044レビュー対応: 9件のバグ修正

| key | value |
|---|---|
| 番号 | 045 |
| フェーズ | ph2 |
| 種別 | resp (044 対応) |
| 対象文書 | `044_ph2_rev_043.md` |
| 作成日 | 2026-02-14 |
| テスト | 440 passed (428既存 + 12新規) |

---

## 1. 修正一覧

| # | ID | 重大度 | 修正内容 | ファイル |
|---|---|---|---|---|
| 1 | Bug7 | CRITICAL | 単一起動ロック (lockfile + PID + stale回収) | `run_fill_test.py` |
| 2 | E-1 | CRITICAL | `get_order_status` の rate limit 二重チェック | `adapter.py` |
| 3 | A-1 | HIGH | Windows SIGTERM → SIGBREAK 切替 | `run_fill_test.py` |
| 4 | E-3 | HIGH | 注文価格 `int()` → `round()` (sell側偏り解消) | `adapter.py` |
| 5 | E-4 | HIGH | `get_balance` reserved → locked 解析 | `adapter.py` |
| 6 | A-4 | HIGH | `_cleanup_sync` 残存注文キャンセルの確実な実行 | `run_fill_test.py` |
| 7 | F8 | MEDIUM | 連続 preflight 失敗上限 + SAFE_STOP | `run_fill_test.py` |
| 8 | A-7 | MEDIUM | loss_cap 定期更新 (50サイクル毎) | `run_fill_test.py` |
| 9 | F7 | LOW | dead code `max_consecutive_same_side` 削除 | `run_fill_test.py` |

---

## 2. 修正詳細

### Fix 1: 単一起動ロック (Bug7 恒久対策) — CRITICAL

**問題**: 複数 fill_test プロセスが同一 JSONL に並行書き込み → 42% データ汚染  
**修正**: `_acquire_lock()` / `_release_lock()` メソッドを追加。

- 起動時に `results/v460/fill_test/fill_test.lock` を作成 (PID|timestamp|run_id)
- 既存ロック検出時: `psutil` で PID 生存チェック + コマンドライン `fill_test` 照合
- 既存プロセスが生存 → `RuntimeError` で起動拒否
- stale lock (プロセス死亡) → 警告付きで回収
- `atexit` → `_cleanup_sync` → `_release_lock()` で解放

### Fix 2: rate limit 二重チェック (E-1) — CRITICAL

**問題**: `get_order_status()` が 2 API 呼び出し (opens + transactions) なのに `_check_rate_limit()` は冒頭の 1 回のみ  
**影響**: 5秒ポーリング×60回 → 1サイクル最大120リクエスト → Coincheck 4req/s 超過リスク  
**修正**: transactions API 呼び出し前に 2 回目の `_check_rate_limit()` を追加

### Fix 3: Windows SIGTERM (A-1) — HIGH

**問題**: `signal.signal(signal.SIGTERM, handler)` は Windows で `OSError` → graceful shutdown が機能しない  
**修正**: `platform.system() == "Windows"` で分岐:
- Windows: `signal.SIGBREAK` (Ctrl+Break)
- Other: `signal.SIGTERM`

### Fix 4: 注文価格 int()→round() (E-3) — HIGH

**問題**: `int(price)` は floor 切り捨て → sell 側が体系的に 1 JPY 低い (aggressive) → AS リスク上昇  
**影響**: 043# データの sell avg_pnl = -0.80 bps vs buy -0.15 bps の一因  
**修正**: `str(int(price))` → `str(round(price))` (2箇所: 指値注文 + フォールバック)

### Fix 5: get_balance reserved 解析 (E-4) — HIGH

**問題**: Coincheck API `{btc: "0.1", btc_reserved: "0.05"}` に対し `locked=0.0` ハードコード  
**影響**: pre-flight balance check がロック残高を利用可能と誤認 → 残高不足の sell 発注  
**修正**: `*_reserved` キーを `locked` として解析、`total = free + locked` で正しく算出

### Fix 6: _cleanup_sync 確実実行 (A-4) — HIGH

**問題**: `loop.is_running()` 時に `create_task()` で `await` せず → プロセス終了で消失  
**修正**: `asyncio.new_event_loop()` で専用ループを作成し `run_until_complete()` で確実に実行

### Fix 7: 連続 preflight 上限 (F8) — MEDIUM

**問題**: buy/sell 両方で残高不足 → `cycle_interval_sec` 毎の無限 sleep ループ  
**修正**: `_preflight_skip_count` カウンタ + `max_preflight_skip=10` (設定可能)
- 連続 10 回 preflight スキップ → `SAFE_STOP` でログ出力 + `_shutdown_requested = True`

### Fix 8: loss_cap 定期更新 (A-7) — MEDIUM

**問題**: `_update_dynamic_loss_cap()` が起動時 1 回のみ → 168h テスト中の残高変動を反映しない  
**修正**: 50 サイクル毎に `_update_dynamic_loss_cap()` を再呼び出し

### Fix 9: dead code 整理 (F7) — LOW

**問題**: `max_consecutive_same_side` と `_same_side_count` が定義のみで使用されていない  
**修正**: 設定フィールドを `max_preflight_skip` に置換、`_same_side_count` を `_preflight_skip_count` に改名

---

## 3. param_adapter 方向性の再検証

前回セッションのサブエージェントが「offset 増減方向が意図と逆の可能性」と指摘したが、**コード検証の結果、param_adapter は正しい**ことを確認。

| 操作 | offset 変化 | 効果 | 意図 |
|------|------------|------|------|
| AS 高 → offset 減少 | 小さくなる | best_bid/ask に近い = less aggressive | AS 回避 ✅ |
| fill 低 → offset 増加 | 大きくなる | mid に近い = more aggressive | fill 向上 ✅ |

`buy: best_bid + offset` (offset 大 → mid 寄り = 攻撃的)  
`sell: best_ask - offset` (offset 大 → mid 寄り = 攻撃的)

サブエージェントは「offset 大 = mid から遠い」と誤解。実際は逆。

---

## 4. 新規テスト (12件)

| クラス | テスト数 | 概要 |
|--------|---------|------|
| `TestSingleInstanceLock` | 3 | lockfile 生成/解放/stale回収 |
| `TestPreflightSkipLimit` | 3 | max_preflight_skip 設定・初期化・旧設定削除 |
| `TestCleanupSyncImproved` | 1 | _cleanup_sync がロック解放する |
| `TestLossCapPeriodicUpdate` | 1 | _loss_cap_update_interval 存在 |
| `TestWindowsSignalHandler` | 1 | platform モジュール import |
| `TestRateLimitDoubleCheck` | 1 | get_order_status に 2回の rate limit |
| `TestPriceRounding` | 1 | round(price) / int(price) なし |
| `TestBalanceLocked` | 1 | reserved 解析 / locked=0.0 なし |

---

## 5. 変更ファイル

| ファイル | 変更概要 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | Fix 1,3,6,7,8,9: lockfile, SIGTERM, cleanup, preflight, loss_cap, dead code |
| `ztb/trading/live/exchanges/coincheck/adapter.py` | Fix 2,4,5: rate limit, round(), balance |
| `tests/unit/v460/test_regime_detector.py` | +12 テストケース |

---

## 6. 044# 指摘との対応表

| 044# 指摘# | 対応 | 備考 |
|------------|------|------|
| 1 (G1.1 乖離) | 未対応 | §9 P1 — side別 offset 後にGate整合 |
| 2 (データ汚染) | 未対応 | §9 P0-2 — clean/quarantine 分離は別タスク |
| 3 (FINAL 7日) | 未対応 | P1 — fill_quality.py 改修が必要 |
| 4 (AS_raw 分母) | 未対応 | P1 — coverage フィールド追加 |
| 5 (方策A衝突) | 検証済 | **方向性は正しい**。目標一本化は P1 |
| 6 (多重起動) | ✅ Fix 1 | lockfile + PID + stale回収 |
| 7 (same_side未使用) | ✅ Fix 9 | 削除して max_preflight_skip に置換 |
| 8 (無限スキップ) | ✅ Fix 7 | SAFE_STOP 実装 |
| 9 (time_filter) | 未対応 | §9 P1-4 — A/B 検証が必要 |
| 10 (stale全キャンセル) | 未対応 | P1 — run_id フィルタは Coincheck API 制約 |

### 追加修正 (044# 未指摘)

| ID | 対応 | 備考 |
|----|------|------|
| E-1 (rate limit) | ✅ Fix 2 | Coincheck IP ban 防止 |
| A-1 (SIGTERM) | ✅ Fix 3 | Windows graceful shutdown |
| E-3 (int→round) | ✅ Fix 4 | sell 側 PnL 改善に寄与 |
| E-4 (balance) | ✅ Fix 5 | pre-flight 精度向上 |
| A-4 (cleanup) | ✅ Fix 6 | 注文残存防止 |
| A-7 (loss_cap) | ✅ Fix 8 | 168h テスト対応 |

---

## 7. 残タスク

- [ ] 044# §9 P0-2: clean/quarantine 自動分離と再集計コマンド
- [ ] 044# §9 P0-3: side別 offset 最適化 (PnL 主指標)
- [ ] 044# §9 P0-4: time_filter ON/OFF 効果検証
- [ ] 044# §9 P0-5: soft/hard 二段 loss cap
- [ ] 044# §4 P1: Gate 基準と FINAL 条件の整合
- [ ] 044# §4 P1: AS coverage フィールド追加
- [ ] E-1 検証: 本番環境で rate limit 動作確認
- [ ] E-3 検証: round() による sell PnL 改善の統計検証
