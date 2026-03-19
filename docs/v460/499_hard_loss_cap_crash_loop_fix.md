# 499# hard_loss_cap crash loop 修正 — cumulative_pnl_jpy 当日 UTC スコープ化

## 発生事象

3/19 19:25〜20:31 (UTC) にかけて、`hard_loss_cap` による kill → watchdog 自動再起動 → 即座に再 kill の **crash loop** が発生。合計 4 回の停止後、watchdog の crash loop 検知 (max_restarts=3/60min) により自動再起動が抑止され、以降 5 分毎に `crash_loop` アラートが Discord に通知された。

### タイムライン

| UTC | イベント | run_id |
|-----|---------|--------|
| 19:24:55 | start | `1773948293_917d78fb` |
| 19:25:52 | **stop (hard_loss_cap)** — 1サイクルで即発動 | 同上 |
| 19:29:52 | start (watchdog auto-restart) | `1773948591_0c64f8fb` |
| 19:32:13 | **stop (hard_loss_cap)** | 同上 |
| 19:34:55 | start (watchdog auto-restart) | `1773948892_a13bbb23` |
| 19:35:21 | **stop (hard_loss_cap)** | 同上 |
| 19:39:46 | crash_loop 検知 | watchdog |
| 20:29:57 | start (手動再起動) | `1773952193_8b665134` |
| 20:31:23 | **stop (hard_loss_cap)** | 同上 |

## 根本原因

### cumulative_pnl_jpy の全期間合算問題

`resume_from_existing()` → `iter_fill_records_glob()` が **全日付 (2/13〜3/19 = 35日分)** のレコードを読み込み、`cumulative_pnl_jpy` を全レコードから再計算していた。

```
全期間 cumulative_pnl_jpy = -1449.3 JPY (35日分、4146 fills)
```

一方、`loss_cap_jpy` は `残高 × loss_cap_ratio (0.05)` で動的計算される。残高が約 25,000 JPY の場合:

```
loss_cap_jpy = 25,000 × 0.05 = 1,250 JPY
|-1449| > 1,250 → 即座に hard_loss_cap 発動
```

### death spiral の構造

```
起動 → resume → 全期間レコード読込 → cumPnL=-1449
  → loss_cap_jpy=1250 (動的計算)
  → |-1449| > 1250 → hard_loss_cap kill
  → watchdog 検知 → 自動再起動
  → 同じレコード再読込 → 同じ結果 → 即 kill
  → ∞ loop (watchdog max_restarts で最終的に停止)
```

### 見落とされていた設計欠陥

- `cumulative_pnl_jpy` は **推定 PnL** (`post_fill_30s_pnl` × price × qty) の全期間合算
- 推定 PnL は実際の約定損益とは異なるため、長期運用で必ず偏りが蓄積
- `_process_daily_reset` が `cumulative_pnl_jpy` をリセットしていなかった
- `daily_drawdown_guard` は bps ベースで日次リセットが機能していたが、JPY 絶対額ベースの `hard_loss_cap` にはリセット機構がなかった

## 修正内容

### A. resume 時の当日スコープ化 (`orchestrator_lifecycle.py`)

```python
# Before (全期間合算)
for r in clean_records:
    pnl_jpy = compute_record_pnl_jpy(r)
    if pnl_jpy is not None:
        st.cumulative_pnl_jpy += pnl_jpy

# After (当日UTC分のみ)
from ztb.data.raw_paths import utc_day_str_from_timestamp
_utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")
for r in clean_records:
    pnl_jpy = compute_record_pnl_jpy(r)
    if pnl_jpy is not None:
        _r_day = utc_day_str_from_timestamp(r.timestamp)
        if _r_day == _utc_today:
            st.cumulative_pnl_jpy += pnl_jpy
```

Codex 494# の `utc_day_str_from_timestamp` を活用。`btc_delta` / `adverse` は全期間維持（観測用、loss_cap 判定とは独立）。

### B. 日替わりリセット追加 (`orchestrator_pre_cycle.py`)

`_process_daily_reset` に `st: RunSessionState | None` パラメータを追加:

```python
# 499# fix: cumulative_pnl_jpy は当日スコープ → 日替わりでゼロリセット
if st is not None:
    st.cumulative_pnl_jpy = 0.0
if self._soft_loss_cap_triggered:
    self._soft_loss_cap_triggered = False
```

### C. 呼び出し側対応 (`fill_loop_orchestrator.py`)

```python
# Before
self._process_daily_reset()

# After
self._process_daily_reset(st)
```

## 設計意図

| 機構 | スコープ | 判定基準 | 役割 |
|------|---------|---------|------|
| `hard_loss_cap` | **当日 UTC** (499# 修正後) | JPY 絶対額 | 当日の異常損失に対する絶対額安全弁 |
| `soft_loss_cap` | 当日 UTC (連動リセット) | JPY 絶対額 | 当日のロット縮小プレ警告 |
| `daily_drawdown_guard` | 日次 (既存リセット済) | bps | 日次 PnL 管理・halt 制御 |
| `btc_delta` / `adverse` | 全期間 | 観測用 | loss_cap 判定とは独立 |

## テスト

[test_499_loss_cap_daily_scope.py](../../tests/unit/v460/test_499_loss_cap_daily_scope.py) — 7 テスト

- `TestResumeCumulativePnlDailyScope`: resume 時に前日以前のレコードが除外されること
- `TestDailyResetCumulativePnl`: 日替わりで cumPnL がゼロリセットされること、None st でもクラッシュしないこと
- `TestCrashLoopPrevention`: 35日分のレコードがあっても当日分で判定されること

## 日次内訳データ (参考)

```
fill_records_20260213 〜 20260319:
  Total filled: 4,146
  Total cumPnL (全期間): -1,449.3 JPY
  3/19 当日分: -267.5 JPY (89 fills)
  最大日次損失: -267.5 JPY (3/19)
  最大日次利益: +104.7 JPY (2/24)
```

## コミット

- `b21303140`: 499# fix: hard_loss_cap crash loop — cumulative_pnl_jpy を当日UTCスコープに修正
