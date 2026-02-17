# 100# ph2: 追加構造問題（098/099 未カバー）

| key | value |
|---|---|
| 番号 | 100 |
| フェーズ | ph2 |
| 種別 | structural audit |
| 対象 | `scripts/v460/run_fill_test.py`, `ztb/metrics/fill_quality.py` |
| 作成日 | 2026-02-17 |
| 概要 | 098/099 で既出の 12 項目に加え、新たに **7 件** の構造問題を発見 |

---

## §1 E3 (120s) mid 計測タイミングの誤り 【重大度: HIGH】

**場所**: [run_fill_test.py L1691-L1708](scripts/v460/run_fill_test.py#L1691-L1708)

```python
# 047# E3: +30s (=60s) 計測
await asyncio.sleep(self.config.post_fill_wait_sec)  # +30s  ← L1691
mid_60s_after = await self._get_mid_price()           # ← 約定後60sのはず

# 047# E3: +60s (=120s) 計測
await asyncio.sleep(self.config.post_fill_wait_sec * 2)  # +60s  ← L1702
mid_120s_after = await self._get_mid_price()              # ← 約定後120sのはず
```

**問題**: `post_fill_wait_sec` のデフォルトは `30.0` 秒。実際のタイムライン:
- 0s: 約定
- 30s: `mid_30s_after` 取得 ✅
- +30s (= 60s): `mid_60s_after` 取得 ✅
- **+60s (= 120s)**: `mid_120s_after` 取得 ✅ ... **ただし early_exit が発動した場合は不正確**

early_exit 発動時（L1647: `early_exit_triggered = True, break`）は 30s を消化せずに break する。その後 `remaining` チェック（L1651-L1653）で残時間を sleep しないため、`mid_30s_after` の取得時刻が実際には 15s や 20s になる。E3 の `+30s` と `+60s` の sleep は `mid_30s_after` 取得後の相対時間なので、全体のオフセットがずれる。

**影響**: early_exit 発動したサイクルの `mid_60s_after`, `mid_120s_after` は実際のタイムスタンプと不一致。PnL 計測のラベルが信頼できない。

---

## §2 soft_loss_cap レジューム非復元 【重大度: HIGH】

**場所**: [run_fill_test.py L480](scripts/v460/run_fill_test.py#L480), [L1837-L1842](scripts/v460/run_fill_test.py#L1837-L1842)

```python
# __init__
self._soft_loss_cap_triggered: bool = False   # ← L480: 常に False 初期化

# run_continuous
cumulative_pnl_jpy = 0.0                      # ← L1837
for r in clean_records:                        # ← L1838: 既存レコードから累積PnL計算
    ...
    cumulative_pnl_jpy += (...)                # ← L1840
```

**問題**: レジューム時に `cumulative_pnl_jpy` は既存レコードから復元されるが、`_soft_loss_cap_triggered` は **常に `False`** に初期化される。

前回のランで soft cap が発動済み（ロット半減済み）でも、再起動後は:
1. `cumulative_pnl_jpy` が前回の損失を再現するため再び soft cap 判定に入る
2. `_soft_loss_cap_triggered = False` なので再度ロット半減が発火
3. **二重半減**: `0.001 → 0.0005`（最小ロット未満になる可能性）

ただし L2065 で `max(self.config.order_quantity, self._current_lot / 2)` のガードがあるため即座に壊れはしないが、前回既に半減済みのロットをさらに半減しようとする無駄な判定が走り、ログが誤解を招く。

---

## §3 balance_shrink → 動的ロット適応の競合 【重大度: MEDIUM】

**場所**: [run_fill_test.py L2021-L2027](scripts/v460/run_fill_test.py#L2021-L2027) vs [L960-L986](scripts/v460/run_fill_test.py#L960-L986)

```python
# balance_shrink 解除時
if self._balance_shrink_active:
    self._current_lot = self._pre_shrink_lot   # ← L2024: 原値に復元
    self._balance_shrink_active = False

# _check_balance_for_side: 残高不足時の自動縮小
self._current_lot = new_lot                    # ← L961: 残高に合わせて縮小
```

**問題**: `_check_balance_for_side()` での in-place ロット縮小（L961, L986）は `_pre_shrink_lot` を更新しない。

シナリオ:
1. `_current_lot = 0.005` (方策B 適応済み)
2. `_check_balance_for_side` が残高不足で `_current_lot = 0.003` に縮小
3. preflight 連続失敗 → `balance_shrink` 発動 → `_pre_shrink_lot = 0.005` (init時の値) を保存して `_current_lot = 0.0025` に半減
4. 成功 → `_current_lot = _pre_shrink_lot = 0.005` に復元

**L961 の縮小は _pre_shrink_lot を更新しないため**、balance_shrink 解除時に手順2のロット制約が無視され、再び残高不足 → preflight 失敗のループに入る。

---

## §4 loss_cap_jpy 更新と soft_cap_jpy 計算の不整合 【重大度: MEDIUM】

**場所**: [run_fill_test.py L2056-L2077](scripts/v460/run_fill_test.py#L2056-L2077), [L2384](scripts/v460/run_fill_test.py#L2384)

```python
# soft cap 計算 (サイクルごと)
soft_cap_jpy = (
    self.config.loss_cap_jpy                    # ← 動的更新される
    * self.config.soft_loss_cap_ratio           # 0.02
    / self.config.loss_cap_ratio                # 0.05
)

# 動的更新 (50サイクルごと)
self.config.loss_cap_jpy = new_cap              # ← L2384
```

**問題**: `_update_dynamic_loss_cap()` は `loss_cap_jpy` を残高変動に応じて変更するが、`soft_cap_jpy` は毎サイクル `loss_cap_jpy` から再計算される。

残高が増えると `loss_cap_jpy` が増加し、soft_cap も連動して **上昇** する。つまり:
- 損失蓄積中に残高が（外部入金等で）増えると、soft cap が「遠のく」
- 逆に残高減少（出金）で soft cap が突然近づき、想定外の早期発動

soft_cap は run 開始時の残高でスナップショットすべきか、明示的に独立管理すべき。`loss_cap_ratio` / `soft_loss_cap_ratio` の比率計算 (`0.02 / 0.05 = 0.4`) は `loss_cap_jpy` の 40% を soft として使う意図だが、動的更新の連動はドキュメントに明記されていない。

---

## §5 JSONL 書込の非アトミック性 【重大度: MEDIUM】

**場所**: [fill_quality.py L385-L387](ztb/metrics/fill_quality.py#L385-L387)

```python
with open(p, "a", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
```

**問題 5a — シグナル中断時の部分書込**:
`save_fill_records` はバッチ内の全レコードを1行ずつ append する。バッチの途中で SIGINT/SIGTERM が来た場合:
- atexit handler (`_cleanup_sync`) が同じバッチを `_emergency_dump` で再書込
- 結果: 一部レコードが重複（前半は通常保存、全体が emergency dump）
- `load_fill_records` は cycle_id 重複を検知しないため、集計時にダブルカウント

**問題 5b — lockfile による排他はプロセス間のみ**:
`_acquire_lock()` はファイルベースの排他ロックだが、`save_fill_records` の `open("a")` 自体には OS レベルのファイルロックがない。外部ツール（結果解析スクリプト等）が同時に read すること自体は安全だが、別プロセスが同じ JSONL を append した場合、行が interleave する可能性がある。

---

## §6 _check_balance_for_side の一方向ロット縮小 【重大度: MEDIUM】

**場所**: [run_fill_test.py L960-L986](scripts/v460/run_fill_test.py#L960-L986)

```python
# sell 側: BTC 残高不足 → ロット縮小
self._current_lot = new_lot    # L961

# buy 側: JPY 残高不足 → ロット縮小
self._current_lot = affordable_lot   # L986
```

**問題**: `_check_balance_for_side` は呼ばれるたびに `_current_lot` を **下方向にのみ** 修正する。残高が回復しても元のロットに戻す仕組みがない（balance_shrink 解除は別のコードパス）。

サイクルが進む中で:
1. BTC 売り残高不足 → `_current_lot = 0.002`
2. 次サイクルで buy（JPY 十分）→ `_current_lot = 0.002` のまま実行
3. buy 成功 → balance_shrink は preflight 3連続失敗でしか発動しないため、L2024 の復元パスに到達しない
4. 以降すべてのサイクルが縮小ロットで実行される

**方策B (`enable_dynamic_lot`)** が有効なら `_try_auto_lot_size()` が50サイクルごとにロットを再計算するが、方策B無効時は永続的にロットが縮み続ける。

---

## §7 early_exit_triggered 後の mid_30s_after タイミング不正 【重大度: LOW】

**場所**: [run_fill_test.py L1628-L1663](scripts/v460/run_fill_test.py#L1628-L1663)

```python
# early exit loop
for tick in range(ticks):       # ticks = 30/5 = 6
    await asyncio.sleep(monitor_sec)   # 5s
    ...
    if interim_pnl < -threshold:
        early_exit_triggered = True
        break                   # ← ここで break (例: tick=2, 15s 経過)

# remaining 計算
elapsed_monitor = (tick + 1) * monitor_sec   # = 15s
remaining = self.config.post_fill_wait_sec - elapsed_monitor  # = 15s
if remaining > 0 and not early_exit_triggered:
    await asyncio.sleep(remaining)   # ← early_exit 時はスキップ
# ↓
mid_30s_after = await self._get_mid_price()  # ← 実際は 15s 後に取得
```

**問題**: early_exit 発動時、残りの sleep がスキップされるため `mid_30s_after` は実際には 15s 後等の mid を取得している。
- `post_fill_30s_pnl` ラベルが「30s PnL」ではない実質データに汚染
- AS 判定 (`adverse_selected`) もこの mid を基準にしているため、判定精度に影響
- FillRecord には実測時刻のフィールドがなく、事後に判別不能

---

## まとめ

| # | 問題 | 重大度 | 修正コスト | 影響 |
|---|---|---|---|---|
| §1 | E3 計測 early_exit 連動ズレ | HIGH | 低 | PnL ラベル汚染 |
| §2 | soft_loss_cap レジューム非復元 | HIGH | 低 | 二重ロット半減 |
| §3 | balance_shrink vs 動的縮小の _pre_shrink_lot 不整合 | MED | 低 | 復元ロット過大 → preflight ループ |
| §4 | loss_cap_jpy 動的更新と soft_cap の連動 | MED | 中 | soft cap の意図しない変動 |
| §5 | JSONL 非アトミック書込 + 重複リスク | MED | 中 | レコード重複 → 集計誤差 |
| §6 | _check_balance_for_side の一方向ロット縮小 | MED | 低 | ロット永続縮小 |
| §7 | early_exit 後の mid_30s_after タイミング | LOW | 低 | AS 判定ノイズ |
