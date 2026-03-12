# 224# 後続作業: B1/B2 + 盲点修正

| 区分 | 概要 | ファイル | 重要度 |
|------|------|----------|--------|
| B1 | halt解除後ソフトリカバリ (lot半減) | daily_drawdown_guard / orchestrator / executor | MEDIUM |
| B2 | 日替わり × dynamic kill 矛盾検出+reset | orchestrator / sell_dynamic_kill | MEDIUM |
| B3 | kill mgr hot-reload 時の状態保存 | run_fill_test | CRITICAL |
| B4 | gate block guard_fire_counts 記録 | orchestrator | MEDIUM |
| B5 | soft_drawdown_interval_multiplier 永続化 | resilience / orchestrator | HIGH |
| B6 | is_kill_active() 副作用なし検査メソッド | sell_dynamic_kill | LOW |

---

## §1 B1: halt解除後ソフトリカバリ

### 問題
per-side halt が `tick_side_halt()` で解除された直後、即座にフルロットで取引再開される。
市場状況が halt トリガ時と変わっていない可能性があり、解除直後の大損リスクが高い。

### 実装

1. **`DailyDrawdownState`** に `side_recovery_remaining_buy/sell: int = 0` 追加
2. **`DailyDrawdownGuard.__init__`** に `per_side_recovery_cycles` (=5) と `per_side_recovery_lot_scale` (=0.5) パラメータ追加
3. **`tick_side_halt()`**: halt 解除時に `side_recovery_remaining = per_side_recovery_cycles` をセット
4. **`get_recovery_lot_scale(side)`**: リカバリ残 > 0 の場合 `per_side_recovery_lot_scale` を返しデクリメント、0 なら 1.0
5. **Orchestrator**: `run_single_cycle()` 直前に `get_recovery_lot_scale()` を呼び `_halt_recovery_lot_mult` にセット
6. **Executor**: `_alert_lot_mult` の後に `_halt_recovery_lot_mult` を適用 (floor = `config.order_quantity`)
7. **Config/YAML**: `per_side_recovery_cycles: 5`, `per_side_recovery_lot_scale: 0.5` 追加
8. **State persistence**: export/import/metrics に recovery フィールド追加

### 設計判断
- リカバリカウンタは「実際にそのサイドで取引実行されたサイクル」でのみデクリメント
  - gate block でスキップされた場合はデクリメントされない (意図通り)
- 日替わりリセットでリカバリ状態もクリア (fresh DailyDrawdownState)
- `_alert_lot_mult` × `_halt_recovery_lot_mult` は乗算的に適用 (二重に慎重)

---

## §2 B2: 日替わりリセット × dynamic kill 矛盾検出

### 問題
`maybe_reset_day()` は per-side halt/PnL を全クリアするが、`DynamicKillManager` の
rolling window (`_pnl_history`) は cross-day で残存。日替わり後に kill がアクティブなまま
halt だけ解除されると、「kill で取引ブロックされるが halt は解除済み」という矛盾が発生。

### 実装

1. **`DynamicKillManager.is_kill_active()`**: 副作用なしで kill 状態を検査する新メソッド
   - `check_kill()` は cooldown デクリメント等の副作用があるため dayReset では使用不可
   - `_cooldown > 0` または `rolling_mean < threshold` なら True
2. **Orchestrator**: `maybe_reset_day()` 直後に sell/buy kill mgr の `is_kill_active()` を呼出
   - active なら WARNING ログ + `_km.reset()` + `_inc_guard_fire("day_reset_kill_conflict")`

### 設計判断
- `is_kill_active()` は default threshold のみ使用 (regime 情報なし)
  - 日替わり境界では regime 文脈が不定のため default で十分
- kill が active なら全 reset — PnL 履歴も含めてクリア
  - 前日の PnL データで当日の取引を制限するのは不合理

---

## §3 B3: hot-reload kill mgr 状態保存 (CRITICAL修正)

### 問題
`_rebuild_sell_kill_mgr()` / `_rebuild_buy_kill_mgr()` は新規インスタンスを生成するため、
蓄積済みの `_pnl_history` (rolling window PnL) が**全消失**。
YAML で dynamic kill 閾値を微調整するだけで kill 判定に必要な直近 PnL 履歴がゼロクリアされ、
kill が一時的に完全無効化 → **安全弁消失 → 大損リスク**。

### 修正
`_rebuild_daily_drawdown_guard()` と同様のパターンで `export_state() → 再構築 → import_state()` を追加。

---

## §4 追加修正

### B4: gate block guard_fire_counts
- gate blocked 時に `_inc_guard_fire(f"gate_{blocking_reason}")` を記録
- `gate_buy_dynamic_kill`, `gate_sell_dynamic_kill`, `gate_unknown_regime_buy_skip` 等が累積

### B5: soft_drawdown_interval_multiplier 永続化
- `FillTestState` に `soft_drawdown_interval_multiplier: float = 1.0` フィールド追加
- `_build_state_snapshot()` でエクスポート、`resume_from_existing()` でインポート
- soft DD 発動中 (interval 3x) にクラッシュ → 再起動しても乗数が維持される

### B6: is_kill_active() メソッド
- `DynamicKillManager.is_kill_active() -> (bool, float|None, int)`
- `check_kill()` の副作用なし版 — cooldown/stale_counter を変更しない
- B2 の day-reset 検査で使用

---

## §5 セルフレビュー (224# 実装)

| ID | 項目 | 結果 |
|----|------|------|
| R1 | `get_recovery_lot_scale()` 1サイクル1回呼出し | ✅ try 直後、`run_single_cycle` 直前で1回のみ |
| R2 | B2 `is_kill_active()` 副作用なし | ✅ R2当初の`check_kill()`問題を修正 |
| R3 | Recovery カウンタとサイクルスキップの関係 | ✅ gate block/balance check でスキップされた場合はデクリメントされない |
| R4 | `_halt_recovery_lot_mult` 初期値 | ✅ クラスレベルで 1.0 宣言、毎サイクル `get_recovery_lot_scale()` で再設定 |
| R5 | Recovery × 日替わりクリア | ✅ fresh DailyDrawdownState で recovery=0 |
| R6 | export/import 整合性 | ✅ recovery フィールドを export_state/import_state に追加 |
| R7 | hot-reload kill mgr 状態保存 | ✅ export → rebuild → import で PnL 履歴維持 |

---

## §6 盲点分析 (224# 後)

### 継続監視 (対応不要・経過観察)
| ID | 概要 | 重要度 | 理由 |
|----|------|--------|------|
| C1 | lot 乗数チェーンの floor 動作 | LOW | 各段階で `max(order_quantity, ...)` が適用され実害なし |
| C2 | soft DD lot 半減 × regime mult 打ち消し | LOW | regime mult > 1.0 での半減打ち消しは限定的 |
| C3 | `_consecutive_unknown_blocks` stale 蓄積 | LOW | regime=unknown 以外の gate block 後もカウンタが残る。超過時は safe bypass |
| C4 | time_filter / preflight の fire count 未記録 | LOW | cancel_reason で記録済み、fire_counts は追加的 |
| C5 | MCB/SAD hot-reload 非対応 | LOW | 現状パラメータ変更頻度が低い |

### 今後の改善候補 (225# 以降)
| ID | 概要 | 重要度 |
|----|------|--------|
| F1 | cross-day warmup vs kill reset 矛盾 | MEDIUM |
| F2 | 通常サイクルパスの state save interval 短縮 | MEDIUM |
| F3 | phantom position 検出 (restart 後の residual order 検知) | HIGH |

---

## §7 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/daily_drawdown_guard.py` | B1: recovery state + get_recovery_lot_scale() |
| `scripts/v460/lib/fill_loop_orchestrator.py` | B1/B2: recovery 連携 + day reset kill 検出 + gate fire count + soft_dd_mult 永続化 |
| `scripts/v460/lib/fill_cycle_executor.py` | B1: _halt_recovery_lot_mult 適用 |
| `scripts/v460/lib/fill_config.py` | B1: per_side_dd_recovery_cycles/lot_scale 設定 |
| `scripts/v460/run_fill_test.py` | B1: guard 初期化 + B3: kill mgr rebuild 状態保存 |
| `configs/v460/fill_test.yaml` | B1: recovery 設定追加 |
| `scripts/v460/lib/resilience.py` | B5: soft_drawdown_interval_multiplier フィールド |
| `ztb/risk/sell_dynamic_kill.py` | B6: is_kill_active() メソッド |
| `tests/unit/v460/test_224_halt_recovery_and_kill_reset.py` | **新規**: テスト16件 |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | 既存: metrics keys 更新 |
| `docs/v460/224_ph2_halt_recovery_and_kill_reset.md` | **本ドキュメント** |

## §8 テスト結果

- 既存テスト: 2982 passed → 2982 passed (0 failure, 1 fix for metrics keys)
- 新規テスト: 16 件 (B1: 10, B2: 4, B1-executor: 2)
- **合計: 2998 passed, 0 failed**
