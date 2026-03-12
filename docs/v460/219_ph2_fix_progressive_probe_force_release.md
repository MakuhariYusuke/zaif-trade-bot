# 219# Progressive Probe + Force Release: DynamicKill 回復の高速化

| key | value |
|---|---|
| 対象 | 218# probe の改良 — 回復速度の大幅向上 |
| commit | `e9a979dbe` |
| テスト | 2940→2949 passed / 0 failed |
| 新規テスト | 9件 (`test_219_progressive_probe.py`) |

---

## 問題点: 218# Probe の限界

218# probe は `max_stale=30` (60min) で1回だけ kill 解除 → rolling50 に1件追加 → 微動 → 再 kill → 次の probe まで 60min 待ち。回復に数時間かかる。

---

## 修正内容

### Fix 1: `max_stale_kill_cycles` 30→10

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| 変更 | デフォルト 30→10 (60min→20min で初回 probe) |

### Fix 2: Progressive probe interval — `_effective_probe_interval()`

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| ロジック | probe 間隔を半減: base 10 → 5 → 3 → 2 → 2 |
| 新フィールド | `_consecutive_probes: int`, `min_probe_interval: int = 2` |
| 効果 | 連続 probe が加速し、rolling window に高頻度でデータ注入 |

### Fix 3: Force release after N probes

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| ロジック | `_consecutive_probes >= max_force_release_probes` → kill 永久解除 |
| 新フィールド | `_force_released: bool`, `max_force_release_probes: int = 5` |
| リセット | `track()` で `_consecutive_probes=0`, `_force_released=False` |
| 効果 | 最悪でも 5 probes (≈40min) で完全復帰 |

---

## 本番確認

- 再起動後 20min で probe 発火 (14:54) ✅
- Buy fill @ 10,508,393 JPY, pnl=-3.03bps
- rolling50 mean: -1.283→-1.278bps (微改善)

---

## テスト (9件)

`tests/unit/v460/test_219_progressive_probe.py`:

- progressive interval halves
- effective interval calculation
- force release after N probes
- force release ends on track
- zero disables force release
- track resets consecutive probes
- export/import state
- BuyManager works
- default config values
