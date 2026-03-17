# 459# Hot-Reload 横展開 + 再起動耐性 + ranging_buy ソフト化

> **種別**: impl (実装)  
> **フェーズ**: ph2 (G1.1-exec)  
> **依存**: 458# macro sell protection, 195# velocity skip ソフト化, 179# _effective_sleep  
> **コミット**: `f34eb085c` → `f840d0e0a` → `b1f526531` → `a49406574`  
> **最終更新**: 2026-03-17

---

## 1. 背景

458# macro sell protection 実装後、以下の 3 つの運用課題が顕在化:

1. **ranging_buy_low_vol ハードスキップ** — low vol 環境での buy 全スキップが
   機会損失を拡大 (実測 low vol 群 pnl30 = +3.06bps で収益性あり)
2. **config hot-reload 未到達** — `maybe_reload()` が `_post_cycle_sleep()` のみで
   呼ばれ、連続ゲートブロック (7h+) 中に YAML 変更が反映されない
3. **SIGTERM 時の状態消失** — `_cleanup_sync` (atexit) が sync-only で
   `_finalize_run` (async) の最終状態保存ロジックに到達不能

---

## 2. 変更内容

### 2.1 ranging_buy_low_vol ソフト化 (`f34eb085c`)

**cycle_gate_aggregator.py** `_check_ranging_buy_low_vol`:

- `ranging_buy_low_vol_as_offset: true` 時は hard skip せず `GateCheckResult(blocked=False)` を返却
- maker_price の `low_vol_offset_boost` に offset 拡大を委譲 (195# パターン準拠)

**fill_config.py** 設定追加:

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `ranging_buy_low_vol_as_offset` | `bool` | `False` | ソフトモード切替 |

**config_hot_reload.py**: `_HOT_RELOADABLE_FIELDS` に追加 → ライブ切替対応

**データ根拠** (3/16 実績):

| 区分 | n | pnl30 | AS率 |
|---|---|---|---|
| low vol (ratio < 0.75) | 12 | +3.06 bps | 33% |
| high vol (ratio ≥ 0.75) | 71 | +0.49 bps | 28% |

### 2.2 Gate Block hot-reload 修正 (`f840d0e0a`)

**根本原因**: 連続ゲートブロック中は `_post_cycle_sleep()` に到達せず
`maybe_reload()` が呼ばれない設計欠陥。

**修正**: gate block ループ内の sleep 前に `self._config_reloader.maybe_reload(self)` を挿入。

### 2.3 _effective_sleep 横展開 (`b1f526531`)

179# `_effective_sleep()` に `maybe_reload()` を統合し、全 sleep パスで
config 変更を検出するよう横展開:

**fill_loop_orchestrator.py**:
```python
async def _effective_sleep(self, *, multiplier=1.0, max_override=0.0):
    # ... interval 計算 ...
    # 459# 横展開: 全 skip/halt/error パスが経由する sleep 前に reload 検出
    self._config_reloader.maybe_reload(self)
    await asyncio.sleep(_sleep)
```

**orchestrator_mid_cycle.py** — `narrow_spread_pause` も `_effective_sleep` 経由に統一:
```python
if gate_result.blocking_reason == "narrow_spread_pause":
    await self._effective_sleep(max_override=self.config.narrow_spread_pause_sec)
```

**orchestrator_balance.py** — `preflight_pause` に明示的 `maybe_reload` 挿入:
```python
self._config_reloader.maybe_reload(self)
await asyncio.sleep(pause_sec)
```

**カバーされる sleep パス** (9+):
gate_block / halt (drawdown) / toxicity_skip / one_sided_skip /
degraded_liquidation / exception / SkipGate threshold breach /
narrow_spread_pause / preflight_pause

### 2.4 再起動耐性: _cleanup_sync 最終状態保存 (`a49406574`)

**問題**: `_cleanup_sync` (atexit handler) は同期コンテキストだが、
状態保存が `_finalize_run` (async) 内のみで実行されるため、
SIGTERM / Ctrl+C 時に最終状態が保存されない。

**修正**:

**fill_loop_orchestrator.py**: `_session_state` 属性を追加し `run_continuous` 開始時に保持:
```python
self._session_state = st  # 459# cleanup_sync からの最終状態保存用
```

**orchestrator_lifecycle.py** `_cleanup_sync`:
```python
st = getattr(self, "_session_state", None)
if st is not None:
    self._state_persistence.save(self._build_state_snapshot(
        total_count=st.total_count,
        filled_count=st.filled_count,
        cumulative_pnl_jpy=st.cumulative_pnl_jpy,
    ))
    logger.info("[cleanup] Final state snapshot saved")
```

---

## 3. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/cycle_gate_aggregator.py` | ranging_buy_low_vol ソフト化 |
| `scripts/v460/lib/fill_config.py` | `ranging_buy_low_vol_as_offset` 追加 |
| `scripts/v460/lib/config_hot_reload.py` | hot-reloadable fields 追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | `_effective_sleep` に `maybe_reload` 統合, `_session_state` 保持 |
| `scripts/v460/lib/orchestrator_lifecycle.py` | `_cleanup_sync` に最終状態スナップショット保存 |
| `scripts/v460/lib/orchestrator_balance.py` | `preflight_pause` に `maybe_reload` 挿入 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | `narrow_spread_pause` を `_effective_sleep` 経由に統一 |
| `configs/v460/fill_test.yaml` | `ranging_buy_low_vol_as_offset: true`, `low_vol_offset_boost: 1.5` |

---

## 4. テスト結果

2182 passed, 125 skipped, 0 failed (全 4 コミット各段階で確認)
