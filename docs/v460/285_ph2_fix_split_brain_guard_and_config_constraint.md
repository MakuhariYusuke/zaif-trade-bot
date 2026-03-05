# 285# 283#/284# P0 対応: Split-Brain 検知 + 設定相互制約

| 項目 | 値 |
|------|-----|
| ドキュメント番号 | 285 |
| 種別 | fix |
| フェーズ | ph2 G1.1-exec (fill 品質計測) |
| 前提 | 282# デッドロック修正, 283# Codex レビュー, 284# Gemini 3.1 Pro レビュー |
| 日付 | 2026-03-06 |

---

## 背景

283# (Codex) / 284# (Gemini 3.1 Pro) が 271#〜282# を横断レビューし、以下の P0 所見を提示:

1. **P0-1: Split-Brain (run_id 重複稼働)** — 2 つの run_id (`1772714694_d20307ad` / `1772714739_b393955d`) が 21:45〜22:37 JST に重複記録。多重起動の検出手段が JSONL レコードに不足
2. **P0-2: 設定相互制約不足** — `per_side_dd_halt_cycles=0` (永続封鎖) + `inventory_escape_enabled=False` の組合せが設定レベルで禁止されておらず、282# デッドロックの再発を許容

本ドキュメントはこれらの妥当性評価と対応実装を記録する。

---

## 妥当性評価

### P0-1: Split-Brain

| 観点 | 評価 |
|------|------|
| 283# の指摘 | **妥当** — FillRecord に pid 情報がなく、同一 JSONL に複数プロセスが書き込んだ場合の事後検出が不可能 |
| 既存ロック機構 | `lock_manager.py`: `O_CREAT\|O_EXCL` atomic + psutil PID/cmdline チェック + heartbeat staleness (300s) — 実行時の排他は一定レベルで担保済み |
| 284# の提案 (portalocker) | **将来課題**: 現行機構で実用上十分。OS レベルロック強化は R-xx として管理 |
| 対応方針 | FillRecord に `pid` フィールドを追加し、事後分析での多重起動検出を可能にする |

### P0-2: 設定相互制約

| 観点 | 評価 |
|------|------|
| 283# の指摘 | **妥当** — `halt_cycles=0` は永続封鎖を意味し、IE 無効なら脱出口がゼロ。282# デッドロックのパターンそのもの |
| 対応方針 | `__post_init__` バリデーションで当該組合せを `ValueError` で即時拒否 |

### 282# ドキュメント修正

| 観点 | 評価 |
|------|------|
| HIGH-2: 時刻境界 | **妥当** — 283# が示す実データでは per_side halt 開始は 13:17:02 (文書記載の 13:08 は不正確) |
| 用語 | **妥当** — "issue 番号" → "ドキュメント番号" (ユーザー指摘による正式呼称) |

---

## 実装

### 1. FillRecord `pid` フィールド追加 (P0-1)

**ファイル**: `ztb/metrics/fill_quality.py`

```python
# 285# 283# P0-1: Split-Brain 検知用
# 同一時刻帯に複数 run_id/pid が存在すれば多重起動を検出可能
pid: int | None = None  # os.getpid() at record creation
```

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py` — `_build_fill_record()`

```python
"pid": os.getpid(),  # 285# 283# P0-1
```

**ファイル**: `scripts/v460/lib/fill_record_helpers.py` — `_build_skip_record()`

```python
pid=os.getpid(),  # 285# 283# P0-1
```

### 2. 設定相互制約バリデーション (P0-2)

**ファイル**: `scripts/v460/lib/fill_config.py` — `FillTestConfig.__post_init__`

```python
# 285# 283# P0-2: per-side halt + IE 相互制約
if (self.per_side_dd_enabled
        and self.per_side_dd_halt_cycles == 0
        and not self.inventory_escape_enabled):
    raise ValueError(
        "per_side_dd_halt_cycles=0 (永続封鎖) と "
        "inventory_escape_enabled=False の組合せは禁止: "
        "脱出口がゼロとなり 282# デッドロック再発の危険"
    )
```

### 3. 282# ドキュメント修正

- 時刻境界: 13:08 → 13:17 (5 箇所)
- 用語: "関連 issue 番号の参照網" → "関連ドキュメント番号の参照網"

---

## テスト

**ファイル**: `tests/unit/v460/test_285_split_brain_guard.py` — 9 件

### TestConfigMutualConstraint (4 件)
| テスト | 検証内容 |
|--------|----------|
| `test_halt_cycles_zero_ie_disabled_raises` | halt=0 + IE無効 → ValueError |
| `test_halt_cycles_zero_ie_enabled_ok` | halt=0 + IE有効 → 許可 (IE が脱出口) |
| `test_halt_cycles_positive_ie_disabled_ok` | halt>0 + IE無効 → 許可 (自然解除) |
| `test_per_side_dd_disabled_no_constraint` | dd無効 → どの組合せでも OK |

### TestFillRecordPidField (5 件)
| テスト | 検証内容 |
|--------|----------|
| `test_pid_field_exists` | pid フィールド存在 + デフォルト None |
| `test_pid_field_set` | pid 明示設定 |
| `test_pid_in_to_dict` | to_dict() に pid 含有 |
| `test_pid_from_dict` | from_dict() で pid 復元 |
| `test_pid_none_from_old_record` | pid なし旧レコードの後方互換 |

**結果**: v460 ユニットテスト 3883 件全パス (285# 9 件含む)

---

## 283#/284# 残課題 (将来対応)

| # | 優先度 | 内容 | 状態 |
|---|--------|------|------|
| P0-3 | P0 | Dual-boot ロック統合テスト | 未着手 |
| P1-4 | P1 | buy_dynamic_kill inventory 連動緩和 | 未着手 |
| P1-5 | P1 | Forced buy 独立 KPI 追跡 | 未着手 |
| P1-6 | P1 | Buy-side AS 防御 (offset 連動, forced buy 遅延執行) | 未着手 |
| R-xx | P2 | OS レベルロック強化 (portalocker/msvcrt) | 未着手 (284# 提案) |

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `ztb/metrics/fill_quality.py` | FillRecord に `pid` フィールド追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | `_build_fill_record` に `pid: os.getpid()` |
| `scripts/v460/lib/fill_record_helpers.py` | `_build_skip_record` に `pid=os.getpid()` |
| `scripts/v460/lib/fill_config.py` | `__post_init__` に相互制約バリデーション |
| `docs/v460/282_ph2_fix_balance_forced_halt_deadlock.md` | 時刻境界 + 用語修正 |
| `tests/unit/v460/test_285_split_brain_guard.py` | テスト 9 件 |
