# 693# Codex 5タスク レビュー＆ staleness init fix

## 概要

- Codex commit `04390da32` で納品された 5 タスク（690#/688# 系）を全件レビュー
- Critical bug 1件を修正：`entry_gate_guard.py` の staleness 初期化バグ
- 追加 auto-refactor: `94ccbc0b1`, `25dcffe18`（fill_quality.py 分割）

## Codex タスク評価一覧

| # | タスク | 評価 | 主要課題 |
|---|--------|------|----------|
| 1 | Entry Gate Guard (691#) | C+ → 修正済 | `last_calibration_update_ts=0.0` → 起動直後 stale 判定 |
| 2 | Skip Audit (cancel_reason_taxonomy) | B | 動的 reason 検証ギャップ、コメント検出が弱い |
| 3 | Timeout Priority (regime×side) | A- | regime validation 欠如（軽微） |
| 4 | Offset Pipeline (stage disable) | B- | テスト数値未検証、パラメータ脆弱 |
| 5 | Analysis Protocol (688 CLI) | B+ | 型安全性（`.get()` パターン）、閾値ハードコード |

## 修正内容

### entry_gate_guard.py: staleness 初期化バグ修正

**問題**: `EntryGateGuardState()` のデフォルト `last_calibration_update_ts=0.0` が epoch 基準のため、起動直後の最初の EV≤0 判定で必ず stale 扱い → `auto_disable` が即発動し、entry_gate guard が無効化される。

**修正**:
```python
# Before
self._state = EntryGateGuardState()

# After
self._state = EntryGateGuardState(last_calibration_update_ts=time.time())
```

- 起動直後は grace period（`staleness_threshold_sec` 分）が有効
- `notify_calibration_update()` が呼ばれなくても、threshold 経過後に stale 判定される

### test_690_entry_gate_guard.py: テスト修正

`test_missing_calibration_update_is_treated_as_stale` を grace period 挙動に合わせて更新:
1. 起動直後 → stale ではない（suppress=False）
2. `last_calibration_update_ts` を 120秒戻す → stale 判定 → auto_disable

## テスト結果

- `test_690_entry_gate`: 7/7 pass
- 全体: 実行中（29%時点で failure なし）

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/entry_gate_guard.py` | staleness 初期値を `time.time()` に修正 |
| `tests/unit/v460/test_690_entry_gate_guard.py` | grace period テスト修正 |
