# 276# BlockingPolicy 抽出設計 — Blocking/Skip ロジック完全調査

## 1. run_continuous 全 Blocking/Skip 決定ポイントマップ

`run_continuous` (L1242–L2487) のメインループ内で `continue` に到達する全パスを列挙。

### BP-1: Daily Drawdown Halt (L1367)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1336–L1390 |
| **条件** | `self._daily_drawdown_guard.is_halted()` |
| **読み取り** | `_halt_start_cycle`, `_halt_iter_count`, `_cycle_count` |
| **書き込み** | `_halt_start_cycle`, `_halt_iter_count`, guard_fire (`dd_halt`), `_last_state_save_time` |
| **副作用** | skip record 生成 (N回毎), batch flush, state save (N回毎), MCB/SAD feed, lock heartbeat 更新, `_effective_sleep(multiplier=5.0)` |
| **sleep** | `_effective_sleep(multiplier=5.0)` — halt 中は 5x 間隔 |

### BP-2: Operator Halt (L1390)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1388–L1401 |
| **条件** | `load_alert_mode(self._results_dir).halt` |
| **読み取り** | `_results_dir` |
| **書き込み** | (なし、メトリクスのみ) |
| **副作用** | skip record (OPERATOR_HALT), batch flush, lock heartbeat, `_effective_sleep(multiplier=5.0)` |

### BP-3: MCB HALT (L1415)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1400–L1415 |
| **条件** | `_mcb.check(time.time()).level == MCBLevel.HALT` |
| **読み取り** | `_mcb`, `_maker_price.last_mid_price` |
| **書き込み** | MCB 内部 state (update), guard_fire (`mcb_halt`) |
| **副作用** | skip record (MCB_HALT), batch flush, lock heartbeat, `_effective_sleep(multiplier=5.0)` |

### BP-4: SAD FROZEN (L1444)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1425–L1444 |
| **条件** | `_sad.check(time.time()).level == SADLevel.FROZEN` |
| **読み取り** | `_sad`, `_maker_price.last_spread_raw` |
| **書き込み** | SAD 内部 state, guard_fire (`sad_frozen`) |
| **副作用** | skip record (SAD_FROZEN), batch flush, lock heartbeat, `_effective_sleep(multiplier=5.0)` |

### BP-5: MCB×SAD AND Escalation (L1471)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1459–L1471 |
| **条件** | `_mcb_warning and _sad_warning` (両方が同時に WARNING 以上) |
| **読み取り** | `_mcb_warning`, `_sad_warning` (ローカル変数) |
| **書き込み** | guard_fire (`mcb_sad_escalation`) |
| **副作用** | skip record (MCB_SAD_ESCALATION), batch flush, lock heartbeat, `_effective_sleep(multiplier=5.0)` |

### BP-6: Hard Skip UTC Hour (L1496)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1476–L1496 |
| **条件** | `utc_hour in self.config.hard_skip_utc_hours` |
| **読み取り** | `config.hard_skip_utc_hours`, `_in_hard_skip_hour` |
| **書き込み** | `_in_hard_skip_hour`, guard_fire (`hard_skip_utc`) |
| **副作用** | skip record (初回のみ), batch flush, lock heartbeat, `_effective_sleep()` |

### BP-7: Per-side DD Both Halt (L1558)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1541–L1558 |
| **条件** | `_daily_drawdown_guard.is_side_halted(next_side) AND is_side_halted(alt)` |
| **読み取り** | `_daily_drawdown_guard` (per-side halt state) |
| **書き込み** | guard_fire (`per_side_dd_both_halt`), `untick_side_halt()` |
| **副作用** | skip record (PER_SIDE_DD_HALT), batch flush, lock heartbeat, `_effective_sleep(multiplier=5.0)` |

### BP-8: Toxic Veto Both-Blocked (L1589)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1574–L1589 |
| **条件** | `next_side in _toxic_veto AND (alt in _toxic_veto OR alt is per-side halted)` |
| **読み取り** | `_toxic_veto`, `_daily_drawdown_guard` |
| **書き込み** | guard_fire (`toxic_veto_block`), `_tick_toxic_veto("both-blocked")` |
| **副作用** | skip record (TOXIC_FILL_SIDE_VETO), batch flush, `_effective_sleep()` |

### BP-9: Phantom Side Veto Both-Blocked (L1618)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1603–L1618 |
| **条件** | `_phantom_guard.is_side_vetoed(next_side) AND (_phantom_guard.is_side_vetoed(alt) OR per-side halted(alt))` |
| **読み取り** | `_phantom_guard`, `_daily_drawdown_guard` |
| **書き込み** | guard_fire (`phantom_veto_block`) |
| **副作用** | skip record (PHANTOM_SIDE_VETO), batch flush, `_effective_sleep()` |

### BP-10: Time Filter Both Sides (L1671)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1636–L1671 |
| **条件** | `_is_time_filtered(next_side) AND _is_time_filtered(alt_side)` |
| **読み取り** | `_time_filter.in_filter`, `_time_filter.last_heartbeat_time`, `config.heartbeat_interval_sec` |
| **書き込み** | `_time_filter.on_enter()`, `_time_filter.last_heartbeat_time`, guard_fire (`time_filter_both_sides`) |
| **副作用** | skip record (初回), heartbeat log (N秒毎), batch flush, lock heartbeat, `_effective_sleep()` |

### BP-11: Time Filter 086# Deadlock Wait (L1710)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1686–L1710 |
| **条件** | `side_filtered AND alt_side == _last_side AND consecutive_086_wait <= max_wait` |
| **読み取り** | `_time_filter.consecutive_086_wait`, `config.max_086_consecutive_wait`, `_last_side` |
| **書き込み** | `_time_filter.consecutive_086_wait`, `_time_filter.on_enter()` |
| **副作用** | skip record (TIME_FILTER_086_DEADLOCK), batch flush, `_effective_sleep()` |

### BP-12: Balance Forced + Per-side Halt Block (L1831)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1804–L1831 |
| **条件** | `balance_forced AND is_side_halted(next_side) AND NOT _inventory_escape` |
| **読み取り** | `_daily_drawdown_guard`, `_toxic_veto`, `_last_side` |
| **書き込み** | guard_fire (`balance_forced_halt_block`), `untick_side_halt()`, `_tick_toxic_veto("halt_block")`, `_last_side`, `_last_state_save_time` |
| **副作用** | skip record (PER_SIDE_DD_HALT), batch flush, `_maybe_skip_state_save()`, `_effective_sleep()` |

### BP-13: Preflight Balance Shrink (L1889 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1870–L1893 |
| **条件** | `両 side 残高不足 → preflight_skip_count >= balance_shrink_consecutive & lot > min_lot` |
| **読み取り** | `_preflight_skip_count`, `_current_lot`, `config.balance_shrink_consecutive` |
| **書き込み** | `_current_lot` (縮小), `_balance_checker.balance_shrink_active`, `_preflight_skip_count` (リセット) |
| **副作用** | skip record (PREFLIGHT_INSUFFICIENT), batch flush, `_effective_sleep()` |

### BP-14: Preflight Pause (L1921 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1900–L1921 |
| **条件** | `preflight_skip_count >= preflight_pause_threshold & preflight_pause_count < max_pauses` |
| **読み取り** | `_preflight_skip_count`, `_preflight_pause_count`, config thresholds |
| **書き込み** | `_preflight_pause_count++`, `_preflight_skip_count = 0` |
| **副作用** | skip record (PREFLIGHT_PAUSE), batch flush, `asyncio.sleep(pause_sec)` (**直接 sleep!**) |

### BP-15: Preflight SAFE_STOP (L1935 付近 — break)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1928–L1940 |
| **条件** | `_preflight_skip_count >= config.max_preflight_skip` |
| **読み取り** | `_preflight_skip_count`, `config.max_preflight_skip` |
| **書き込み** | `_kill_switch.kill("preflight_skip_exceeded")` |
| **副作用** | kill switch → ループ終了 (break), `_effective_sleep()` (直前の skip パス) |

### BP-16: One-sided Freeze Skip (L1966 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1951–L1973 |
| **条件** | `_one_sided_freeze_remaining > 0 AND (_frozen_side is None OR _frozen_side == next_side)` |
| **読み取り** | `_one_sided_freeze_remaining`, `_one_sided_frozen_side` |
| **書き込み** | `_one_sided_freeze_remaining--`, guard_fire (`one_sided_freeze_skip`), `_last_side` |
| **副作用** | skip record, batch flush, `_effective_sleep()` |

### BP-17: One-sided Cooldown Skip (L1993 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L1980–L2002 |
| **条件** | `_one_sided_cooldown_remaining > 0 AND (_frozen_side is None OR _frozen_side == next_side)` |
| **読み取り** | `_one_sided_cooldown_remaining`, `_one_sided_frozen_side` |
| **書き込み** | `_one_sided_cooldown_remaining--`, guard_fire (`one_sided_cooldown_skip`), `_last_side` |
| **副作用** | skip record, batch flush, `_effective_sleep()` |

### BP-18: Balance Forced Skip (L2064 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L2050–L2073 |
| **条件** | `_balance_forced AND config.skip_balance_forced AND NOT (original_also_insufficient OR deadlock_limit超過) AND NOT rescue_enabled` |
| **読み取り** | `_balance_forced_skip_count`, config deadlock params |
| **書き込み** | `_balance_forced_skip_count++`, `_last_side` |
| **副作用** | skip record (BALANCE_FORCED_SKIP), batch flush, `_effective_sleep()` |

### BP-19: CycleGateAggregator Blocked (L2213 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L2150–L2215 |
| **条件** | `_gate_result.blocked` (CycleGateAggregator.evaluate() の結果) |
| **読み取り** | `_gate_result`, `_trending_sell_skip_count`, `_consecutive_gate_blocks`, quiescence config |
| **書き込み** | `_trending_sell_skip_count` (条件付き), `_consecutive_gate_blocks++`, guard_fire, `_last_side`, `_last_state_save_time` |
| **副作用** | skip record, batch flush, `_maybe_skip_state_save()`, `narrow_spread_pause: asyncio.sleep(config.narrow_spread_pause_sec)` or `_effective_sleep()` |

### BP-20: Toxicity Participation Skip (L2248 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L2233–L2260 |
| **条件** | `_gate_result.participation_rate < 1.0 AND random.random() > participation_rate` |
| **読み取り** | `_gate_result.participation_rate` |
| **書き込み** | guard_fire (`toxicity_participation_skip`), `_last_side` |
| **副作用** | skip record, batch flush, `_effective_sleep()` |

### BP-21: Degraded Liquidation Duty Skip (L2282 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L2269–L2296 |
| **条件** | `_degraded_liquidation AND _duty > 1 AND (counter % duty) != 1` |
| **読み取り** | `_degraded_liquidation_duty_counter`, config.degraded_liquidation_duty_cycle |
| **書き込み** | `_degraded_liquidation_duty_counter++`, guard_fire (`degraded_liquidation_duty_skip`), `_last_side` |
| **副作用** | skip record, batch flush, `_effective_sleep()` |

### BP-22: Exception Continue (L2414 付近)
| 項目 | 詳細 |
|------|------|
| **行範囲** | L2399–L2420 |
| **条件** | `run_single_cycle` で `Exception` 発生 |
| **読み取り** | `_recovery_scale` (ローカル) |
| **書き込み** | `_last_side`, DD guard recovery counter 復元 |
| **副作用** | error log, `_balance_checker.restore_lot_after_dust_sweep()`, `_effective_sleep()` |

---

## 2. State 結合分析

### 高結合状態変数 (3+ 決定ポイントで共有)

| 状態変数 | 参照 BP | 種別 |
|----------|---------|------|
| `_daily_drawdown_guard` | BP-1, 7, 8, 9, 12 | R/W (halt/per-side) |
| `_toxic_veto` | BP-8, 12, サイクル末尾 | R/W |
| `_last_side` | BP-11, 12, 16, 17, 18, 19, 20, 21, 22 | W |
| `_guard_fire_counts` (via `_inc_guard_fire`) | **全 BP** | W |
| `st.batch` / `st.total_count` | **全 BP** | W |
| `_last_state_save_time` | BP-1, 12, 19 | R/W |
| `_consecutive_gate_blocks` | BP-19 (R/W) | R/W |
| `_trending_sell_skip_count` | BP-19 (conditional) | R/W |
| `_in_hard_skip_hour` | BP-6 | R/W |
| `_halt_start_cycle`, `_halt_iter_count` | BP-1 | R/W |
| `_one_sided_freeze_remaining` | BP-16 | R/W |
| `_one_sided_cooldown_remaining` | BP-17 | R/W |
| `_one_sided_frozen_side` | BP-16, 17 | R |
| `_balance_forced_skip_count` | BP-18, サイクル後 | R/W |
| `_preflight_skip_count` | BP-13, 14, 15 | R/W |
| `_phantom_guard` | BP-9, L1521–1533 | R |

### 結合クラスター

```
Cluster A: "System-Level Halt" (外部条件によるフルブロック)
  BP-1 (DD Halt), BP-2 (Operator), BP-3 (MCB), BP-4 (SAD), BP-5 (MCB×SAD)
  → 共通: multiplier=5.0, 独立評価, 相互依存なし

Cluster B: "Side Selection Blocking" (side 封鎖)
  BP-7 (Per-side DD both), BP-8 (Toxic veto both), BP-9 (Phantom veto both)
  → 共通: _daily_drawdown_guard, _toxic_veto, side 代替 → 両方封鎖で skip

Cluster C: "Time/Schedule Gate" 
  BP-6 (Hard skip UTC), BP-10 (Time filter both), BP-11 (086# deadlock)
  → 共通: UTC 時間帯ベース, _time_filter, _in_hard_skip_hour

Cluster D: "Balance/Preflight Gate"
  BP-12 (balance_forced + halt), BP-13 (balance shrink), BP-14 (preflight pause), BP-15 (SAFE_STOP), BP-18 (balance_forced skip)
  → 共通: _preflight_skip_count, _balance_forced_skip_count, _current_lot

Cluster E: "One-sided Escalation"
  BP-16 (freeze), BP-17 (cooldown)
  → 共通: _one_sided_freeze/cooldown_remaining, _one_sided_frozen_side

Cluster F: "Per-cycle Gate" (CycleGateAggregator 委譲済み)
  BP-19 (gate blocked), BP-20 (toxicity), BP-21 (degraded duty)
  → 共通: _gate_result, _cycle_gate, _consecutive_gate_blocks
```

---

## 3. 既存ヘルパーメソッド (既に抽出済みの責務)

| メソッド | 行 | 使用箇所 (BP) | 役割 |
|----------|-----|---------------|------|
| `_tick_toxic_veto(context)` | L415–L428 | BP-8, 12, サイクル末尾 | toxic veto カウンタ減算 + 期限切れ除去 |
| `_maybe_skip_state_save(st, context)` | L433–L448 | BP-12, 19 | skip パス中の定期 state 保存 |
| `_feed_mcb_sad()` | L453–L466 | BP-1 | halt 中の MCB/SAD price/spread 更新 |
| `_effective_sleep(multiplier, max_override)` | L573–L595 | ほぼ全 BP | regime 別 sleep + 乗数合成 |
| `_make_loop_skip_record(...)` | L597–L617 | ほぼ全 BP | skip record 生成 wrapper |
| `_opposite_side(side)` | L468–L470 | BP-7,8,9,10,11,18 等 | 反対 side 計算 |
| `_is_side_killed(side)` | L148–L172 | BP-19 (gate 入力) | dynamic kill 判定 |
| `_inc_guard_fire(name)` | L402–L406 | 全 BP | guard 発火カウンタ |

### 未抽出の共通パターン (Skip Side Effect Ceremony)

ほぼ全 BP に共通する「Skip Ceremony」が見られる:
```python
# パターン: Skip Ceremony (16/22 BP で出現)
st.batch.append(self._make_loop_skip_record(...))
st.total_count += 1
st.batch = self._batch_persistence.maybe_flush(st.batch, reason)
self._update_lock_heartbeat()  # 一部の BP のみ
await self._effective_sleep(multiplier=...)
continue
```

---

## 4. 提案: 抽出境界

### 4.1 BlockingPolicy クラス (推奨)

**責務**: 「このサイクルを実行すべきか」の判定と、skip 時の共通副作用を管理

```
BlockingPolicy
├── SystemHaltChecker  ← BP-1,2,3,4,5 (Cluster A)
│   - check() → BlockDecision(reason, sleep_mult)
│   - 状態: _halt_start_cycle, _halt_iter_count, _in_hard_skip_hour
│
├── SideBlockChecker   ← BP-7,8,9 (Cluster B)  
│   - check(side) → SideBlockResult(blocked, alt_side, reason)
│   - 状態: DD guard, _toxic_veto, _phantom_guard の参照のみ
│
├── TimeGateChecker    ← BP-6,10,11 (Cluster C)
│   - check(side) → TimeGateResult(blocked, alt_side)
│   - 状態: _time_filter, config.hard_skip_utc_hours
│
├── BalanceGateChecker ← BP-12,13,14,15,18 (Cluster D)
│   - check(side, balance_forced) → BalanceGateResult
│   - 状態: _preflight_skip_count, _balance_forced_skip_count
│
└── EscalationChecker  ← BP-16,17 (Cluster E)
    - check(side) → EscalationResult(blocked, remaining)
    - 状態: _one_sided_freeze/cooldown_remaining
```

BP-19,20,21 (Cluster F) は既に `CycleGateAggregator` に委譲済みのため抽出不要。

### 4.2 抽出の難所

1. **Skip Ceremony の統一**: 16/22 BP で同一パターン → `_execute_skip(st, reason, side, sleep_mult)` ヘルパーで一元化可
2. **BP-12 の複雑性**: balance_forced + per-side halt + inventory_escape の 3 方向分岐。単独で ~40 行。`BalanceGateChecker` 内部で inventory_escape 判定まで含めるか、別途 `InventoryEscapePolicy` にするかの判断が必要
3. **`_last_side` の書き込み**: 11/22 BP で `_last_side = next_side` を行う。これは skip 時の side 交互保証であり、BlockingPolicy 外で一元管理すべき (skip / execute 両方で更新するため)

### 4.3 クリーンな抽出ステップ案

```
Step 1: _execute_skip(st, reason, side, sleep_mult=1.0) ヘルパー導入
        — Skip Ceremony を 5 行 → 1 行に
Step 2: SystemHaltChecker 抽出 (BP-1,2,3,4,5)
        — 最も独立性が高く、他の BP との相互依存が少ない
Step 3: SideBlockChecker + TimeGateChecker 抽出 (BP-6~11)
        — side 代替ロジックの共通化
Step 4: EscalationChecker 抽出 (BP-16,17)
        — freeze/cooldown の対称性を活用
Step 5: BalanceGateChecker 抽出 (BP-12~15,18)
        — 最も複雑、最後に行う
```

---

## 5. 市場理論による実コードロジック改善機会

### 5.1 ハードコード `multiplier=5.0` (halt sleep) — L1385, L1399, L1413, L1443, L1470, L1556

```python
await self._effective_sleep(multiplier=5.0)  # 6 箇所で同一定数
```

**問題**: halt 中の sleep 倍率 `5.0` は根拠不明の magic number。

**理論的改善**:
- **Brunnermeier & Pedersen (2009) liquidity spiral**: halt 復帰時の市場状態は halt 期間中に変化しうる。5.0x 固定ではなく、MCB/SAD の最新シグナルから halt 解除までの推定時間を動的に計算し、exponential backoff with jitter を適用すべき。
- 具体案: `_HALT_SLEEP_MULT = 5.0` をクラス定数化 + config 外部化 (最低限)、将来的には MCB volatility σ に比例するスケーリング。

### 5.2 `_HALT_PERSIST_INTERVAL = 10` (L1350)

ループ内ローカル定数。halt 中の state save 間隔。

**改善**: クラス定数に昇格し、config.halt_persist_interval で外部化可能に。
理論的には halt 期間の予測分布 (Poisson 近似) に基づく適応的間隔が最適だが、複雑性に見合わないため定数外部化で十分。

### 5.3 `UNKNOWN_REGIME_MAX_CONSECUTIVE = 10` (cycle_gate_aggregator.py L139)

unknown regime で 10 回連続 block → 強制通過。

**問題**: `10` の根拠が不明。

**理論的改善**:
- Hamilton (1989) regime-switching モデルのフィルター収束速度に基づくべき。regime detector の buffer_size / stability_threshold から逆算して「N サイクルで regime が安定する期待値」を使うのが理論的に正しい。
- 具体的には `regime_detector.config.buffer_size / stability_window_multiplier` で算出可能。

### 5.4 `multiplier=3.0` for phantom detection (L1534)

```python
await self._effective_sleep(multiplier=3.0)
```

**問題**: phantom position 検出後の sleep 倍率 `3.0` の根拠が不明。

**理論的改善**: phantom 検出は「在庫推定エラー」を意味し、Ho & Stoll (1981) では在庫不確実性が高い場合にスプレッドを広げて対応する。sleep ではなく offset 拡大で対応する方が理論的に整合的。

### 5.5 `_STATE_SAVE_INTERVAL_SEC = 300.0` (L134)

5 分間隔の state 保存。

**改善**: config 外部化すべき (現在はクラス定数のみ)。サイクル間隔が 120s なら ~2.5 サイクル分。サイクル間隔が動的に変化する regime-aware 設計では、「N サイクルに 1 回」に変更した方が一貫性がある。

---

## 6. skip_gate_evaluator `except Exception` 分析

全 13 箇所の `except Exception` を評価:

| # | 行 | try の内容 | 狭い例外に変更可能か | 推奨 |
|---|-----|-----------|---------------------|------|
| 1 | L194 | `SkipGate.load()` + 初期化全体 | △ `ImportError | FileNotFoundError | pickle.UnpicklingError` | 部分的に可。pickle の壊れ方が多様なため `Exception` が妥当。ただし `KeyboardInterrupt`, `SystemExit` の巻き込みを防ぐため、catch 後に `if isinstance(e, (KeyboardInterrupt, SystemExit)): raise` を追加すべき |
| 2 | L464 | `warm_start_skip_gate_thresholds()` | ○ `FileNotFoundError | ValueError | KeyError` | warm start は optional。現在の `Exception` は妥当 (non-fatal) |
| 3 | L494 | `ScoreCalibrator.load()` | ○ `FileNotFoundError | pickle.UnpicklingError` | 狭められるが効果は限定的 |
| 4 | L527 | side model `_load_gate_from_path()` | △ | L194 と同様、pickle の多様な失敗パターンのため `Exception` が実用的 |
| 5 | L562 | alt model `_load_gate_from_path()` | △ | L527 と同上 |
| 6 | L625 | `alt_gate.evaluate(features)` | **○ narrowable** | ML 推論は `ValueError | TypeError | KeyError` に限定可能。`RuntimeError` (ONNX/torch) も追加。**改善推奨** |
| 7 | L834 | hot-reload: `_load_gate_from_path()` | △ | L194 同様 |
| 8 | L846 | `SkipGate` import (side hot-reload) | ○ `ImportError` に狭められる | **改善推奨**: `ImportError` のみが妥当 |
| 9 | L868 | side model first load via hot-reload | △ | L527 同様 |
| 10 | L886 | side model hot-reload | △ | L527 同様 |
| 11 | L1022 | `adapter.get_recent_trades()` | ○ | ネットワーク系: `(ConnectionError, TimeoutError, OSError, ValueError)` に狭められるが、各ブローカー SDK の例外クラスが多様。**ログが debug なので許容範囲** |
| 12 | L1050 | `adapter.get_orderbook()` + OB パース | △ | L1022 同様。SDK 依存 |
| 13 | L1254 | evaluate 本体 (全 ML パイプライン) | **要注意** | 全 ML パイプラインを包む最外層。`Exception` は妥当だが、`KeyboardInterrupt` を巻き込む可能性がある。`except Exception as e:` → `except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:` に狭めるか、`BaseException` 分離を入れることを推奨 |

### 優先改善対象

1. **L846** (`ImportError` に狭める) — 即座に可能、リスクなし
2. **L625** (`ValueError | TypeError | KeyError | RuntimeError` に狭める) — ML 推論の失敗パターンは限定的
3. **L1254** — `BaseException` 分離 (KB interrupt 防止) を追加

### 総合評価

大半の `except Exception` は「外部依存 (pickle, SDK, ネットワーク) の失敗パターンが多様」なため実用的に broad catch が妥当。ただし **L846 の `ImportError` 限定化** と **L1254 の `BaseException` 分離** は改善すべき。

---

## 付録: run_continuous Blocking Flow 全体図

```
while time < end_time:
  ├── Day Reset (reset counters, veto, kill mgrs)
  │
  ├── [BP-1] DD Halt?           → sleep 5x, continue
  ├── [halt 終了記録]
  ├── [BP-2] Operator Halt?      → sleep 5x, continue
  ├── [BP-3] MCB HALT?          → sleep 5x, continue
  ├──   MCB WARNING → offset/interval 乗算
  ├── [BP-4] SAD FROZEN?        → sleep 5x, continue
  ├──   SAD DRY/WIDE → offset/interval/lot 乗算
  ├── [BP-5] MCB×SAD Escalation? → sleep 5x, continue
  ├── [BP-6] Hard Skip UTC Hour? → sleep, continue
  ├── Phantom Guard reconcile
  ├── Side halt tick
  ├── Toxic veto init
  ├── next_side = _next_side()
  │
  ├── [side 代替チェック]
  │   ├── [BP-7] Per-side DD both halt? → sleep 5x, continue
  │   ├── [BP-8] Toxic veto both?       → sleep, continue
  │   ├── [BP-9] Phantom veto both?     → sleep, continue
  │   └── side 切替 (alt_side)
  │
  ├── [BP-10] Time filter both? → sleep, continue
  ├── [BP-11] 086# deadlock?    → sleep, continue
  ├── Time filter side switch
  │
  ├── Regime detector update
  ├── Preflight balance check
  │   ├── opposite 残高 OK → balance_forced
  │   │   ├── [BP-12] per-side halt block? → sleep, continue
  │   │   │   └── [269#] Inventory Escape → fallthrough
  │   │   └── balance_forced freq tracking
  │   ├── [BP-13] Balance shrink     → sleep, continue
  │   ├── [BP-14] Preflight pause    → sleep(pause_sec), continue
  │   ├── [BP-15] SAFE_STOP          → break
  │   └── both insufficient → sleep, continue
  │
  ├── [BP-16] One-sided freeze?  → sleep, continue
  ├── [BP-17] One-sided cooldown? → sleep, continue
  ├── [BP-18] Balance forced skip? → sleep, continue
  │
  ├── CycleGateAggregator.evaluate()
  │   ├── [BP-19] Gate blocked?   → sleep, continue
  │   ├── Pass → reset counters
  │   ├── [BP-20] Toxicity participation skip? → sleep, continue
  │   └── [BP-21] Degraded duty skip? → sleep, continue
  │
  ├── ═══ run_single_cycle() ═══
  │   └── [BP-22] Exception → sleep, continue
  │
  ├── post-cycle processing
  ├── progress log + adaptation
  └── cycle-end sleep (with veto tick, one-sided check)
```
