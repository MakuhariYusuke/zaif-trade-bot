# 268# DD halt JST 日付リセット + 本番インシデント対応

| 項目 | 値 |
|---|---|
| Issue | 268# |
| 種別 | bugfix / incident |
| フェーズ | phg (横断品質改善) |
| Commit | `4b863d211` |
| テスト | 3688 passed, 32 skipped (+8 from 267#) |
| 発生日時 | 2026-03-03 10:51 JST ～ 2026-03-04 06:04 JST (約19h停止) |

---

## 背景 — 本番インシデント

2026-03-03 10:51 JST に DD hard halt がトリガーされ、bot が約 19 時間停止。
3 つの原因が収束して最大影響を発生させた。

## 根本原因分析

### 原因 1: UTC 日境界問題 (主因)

`DailyDrawdownGuard._utc_today()` が UTC 00:00 = **JST 09:00** で日付リセット。
JST 10:51 に halt → 次のリセットは翌 JST 09:00 → **最大約 22 時間** の halt。

```
JST 10:51  DD hard halt トリガー
JST 09:00  ← 次の日付リセット (UTC 00:00)
= 約 22h の halt ウィンドウ
```

### 原因 2: cooldown_release 未搭載版が稼働

本番は 246# の `cooldown_release` (2h 後の部分解除) 導入前のコードで稼働中。
cooldown_release があれば 2h 後に lot 30% で再開できたが、旧コードでは日替わりまで完全停止。

### 原因 3: 267# デプロイが UTC 21h に重複

267# commit 後のデプロイ作業が hard halt 継続中の時間帯に重なり、
実質的に halt 解除を待つだけの状態が続いた。

## 修正内容

### `_utc_today()` → `_today()` (設定可能 TZ)

```python
# Before (266#以前)
@staticmethod
def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

# After (268#)
def _today(self) -> str:
    return datetime.now(self._day_reset_tz).strftime("%Y%m%d")
```

### `day_reset_utc_offset_hours` パラメータ追加

| 層 | 変更 |
|---|---|
| `DailyDrawdownGuard.__init__` | `day_reset_utc_offset_hours: float = 0.0` 引数追加 |
| `DailyDrawdownGuard.__init__` | `self._day_reset_tz = timezone(timedelta(hours=...))` |
| `FillTestConfig` | `dd_day_reset_utc_offset_hours: float = 9.0` (デフォルト JST) |
| `run_fill_test.py` | config → `DailyDrawdownGuard` コンストラクタに渡す |
| YAML パーサー | `loss_control.daily_drawdown.day_reset_utc_offset_hours` サポート |

### 影響範囲

| 箇所 | 影響 |
|---|---|
| `maybe_reset_day()` | `_utc_today()` → `_today()` |
| `import_state()` | `_utc_today()` → `_today()` |
| docstring | UTC固定の記述を「設定可能なタイムゾーン」に更新 |

### 最大 halt 時間の短縮

| 条件 | Before (UTC) | After (JST) |
|---|---|---|
| JST 10:51 に halt | 翌 JST 09:00 = **22h** | 翌 JST 00:00 = **13h** |
| JST 23:59 に halt | 翌 JST 09:00 = **9h** | JST 00:00 = **1min** |
| 平均的な halt | ~14h | ~8h |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/daily_drawdown_guard.py` | `_utc_today`→`_today`, `_day_reset_tz` 追加 |
| `scripts/v460/lib/fill_config.py` | `dd_day_reset_utc_offset_hours` 追加 |
| `scripts/v460/run_fill_test.py` | config パラメータ渡し |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +8 テスト (TZ リセット検証) |
| `tests/unit/v460/test_215_dd_fix_alert_mode.py` | TZ パラメータ対応 |
| `CHANGELOG.md` | 268# セクション追加 |

## テスト追加 (+8)

TZ 日付リセットの境界条件テスト:
- JST 23:59 → 00:00 リセット検証
- UTC offset 0 / 9 での日付判定
- `import_state` の TZ 対応
- 既存テストの互換性維持

## 3/4 デプロイ後タイムライン

268# は 12:20 JST にデプロイ。以下のパターンが繰り返し発生:

| 時刻 (JST) | イベント |
|---|---|
| 06:04 | 267# で再起動 (sha=`227024d549aa`), state restored: pnl=-50bps, halted |
| 06:04 | cooldown_release: 69172s >= 7200s → 部分解除 |
| 06:10 | 2 回目の再起動 (同 sha) |
| 07:00 | sell per-side halt 継続、buy に切替 |
| 07:04 | **deadlock**: buy=JPY不足, sell=per-side halt → 223# 拒否 |
| 07:14 | Per-side halt released (15 cycles exhausted) |
| 07:16 | balance_forced → degraded liquidation sell 開始 |
| 07:36 | **PER-SIDE HALT**: sell -38.21bps → 再 halt |
| 08:00 | Per-side halt released (15 cycles) |
| **09:00** | **Day reset**: 20260303→20260304 (UTC=00:00, **旧コード最後**) |
| 09:49 | **PER-SIDE HALT**: sell -34.12bps |
| 10:09–11:15 | deadlock × 12 cycles (6min間隔) |
| 11:21 | Per-side halt released |
| 11:29 | **PER-SIDE HALT**: sell -32.15bps → 即再 halt |
| 11:56–12:14 | deadlock × 4 cycles |
| **12:20** | **268# デプロイ** (sha=`4b863d211de0`), state: pnl=-24.38bps |
| 12:21 | 2 回目起動 (同 sha), state restored |
| 12:20–13:21 | deadlock 継続 (sell per-side halt 未解除) |
| 13:27 | Per-side halt released |
| 13:45 | sell filled @ ¥10,666,367 (pnl=-5.46bps) |
| 13:47 | **PER-SIDE HALT**: sell -37.61bps → **4 回目の halt** |
| 14:00 | buy filled @ ¥10,690,939 (pnl=-1.62bps) ← 最後の fill |
| 14:05– | **deadlock 継続中**: buy=JPY¥1,028不足, sell=per-side halt |

## 発見された改善課題

### I1 🔴 CRITICAL — Balance-forced デッドロック

**症状**: buy=JPY 不足 + sell=per-side halt → 完全停止 (6min 間隔で空サイクル)

**コード**: `fill_loop_orchestrator.py` L1699:
```python
# 223# P0: balance_forced 後に per-side halt を再チェック
if self._daily_drawdown_guard.is_side_halted(next_side):
    logger.warning(
        "[223#] balance_forced → sell is per-side halted — "
        "refusing to bypass halt (safety > liveness)"
    )
```

**原因**: 223# は「safety > liveness」原則で per-side halt を尊重するが、
buy が JPY 不足で不可能な状態では sell が唯一の回復手段。
sell を拒否すると JPY を得る方法がなくなり **完全デッドロック**。

**提案 (269# 候補)**:
- balance_forced + per-side halt + 反対側不可 の 3 条件が揃った場合のみ
  「liquidation sell」として per-side halt をオーバーライド
- lot は recovery scale (0.35×) または min lot で制限
- 専用ログ/メトリクスで追跡

### I2 🟡 — Per-side halt → release → 即再 halt パターン

**症状**: halt 解除 → 1-2 fills → 累計 PnL が閾値以下 → 即再 halt (90min 無駄)

**タイムライン例**:
- 07:14 released → 07:36 re-halt (22min, 数 fill)
- 08:00 released → 09:49 re-halt (1h49min, day reset 込み)
- 11:21 released → 11:29 re-halt (**8min, 1 fill**) ← 最悪
- 13:27 released → 13:47 re-halt (20min, 2 fills)

**原因**: per-side halt は累計 PnL を追跡するが、release 時にリセットしない。
recovery 期間 (lot scale 0.35×) でも 1 回の負 PnL で -30bps 閾値を再突破。

**提案**: release 時に per-side PnL アキュムレータを部分リセット (例: 50%) するか、
release 後 N cycles は再 halt 猶予期間を設ける。

### I3 🟡 — デッドロック中のサイクルカウント

**症状**: balance_forced_halt_block の空サイクルが per-side halt の 15-cycle カウントに含まれる。
結果的に ~90min のタイマーとして機能するが、「市場状況を見て回復判断」ではなく
「固定時間待ち」になっている。

**提案**: 空サイクル (skip) を halt cycle カウントから除外するか、
デッドロック検出時は即座に limited release するか検討。

### I4 🟢 — JST リセット未検証

268# のデフォルト `day_reset_utc_offset_hours=9.0` は config に明示設定されておらず、
`FillTestConfig` のデフォルト値に依存。次回の JST 00:00 (= UTC 15:00) リセットで
正常動作を確認する必要がある。
