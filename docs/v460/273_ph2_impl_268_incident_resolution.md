# 273# 268# インシデント残課題の解決

> **フェーズ**: ph2 (G1.1-exec)  
> **種別**: impl (実装修正)  
> **日付**: 2026-03-04  
> **前提**: 268# インシデント分析 → 271# 部分対策 → 272# DRY リファクタ → **本 273# で残課題完了**

---

## 1. 対象課題の一覧

268# で特定された問題 6 件のうち、271# で I1/I2 を解決済み。273# では残る I3/I5/I6/Pattern B を解消する。

| ID | 重要度 | 概要 | 271# 状況 | 273# 対策 |
|---|---|---|---|---|
| **I1** 🔴 | Critical | Balance-forced deadlock | ✅ 解決済 (Inventory Escape) | — |
| **I2** 🟡 | High | Per-side halt → release → 即再halt | ✅ 解決済 (PnL reanchor) | — |
| **I3** 🟡 | High | 空サイクルが halt カウントに含まれる | ❌ 未対策 | **✅ `untick_side_halt()`** |
| **I4** 🟢 | Low | JST リセット未検証 | ✅ 検証済 (272#) | — |
| **I5** 🟡 | High | `sell_dynamic_kill` 過剰持続 (92min) | ❌ 未対策 | **✅ `max_kill_duration_sec`** |
| **I6** 🟡 | High | halt 解除後再参入遅延 (18min) | ❌ 未対策 | **✅ `halt_recovery_active` grace** |
| **Pattern B** | — | kill ↔ halt 相互ロック | ❌ I5 起因 | **✅ I5 で解消** |

---

## 2. I5: Kill 時間上限 (`max_kill_duration_sec`)

### 2.1 問題

268# で `sell_dynamic_kill` が 92 分間持続。原因:
- kill 解除条件 = rolling PnL mean が閾値を上回る
- rolling PnL 更新 = 新約定データが `track()` に渡される
- halt 中は約定できない → `track()` が呼ばれない → rolling PnL が改善しない → kill 解除されない

これにより **kill ↔ halt 相互ロック (Pattern B)** が発生。

### 2.2 解決策

`DynamicKillConfig` に `max_kill_duration_sec` フィールドを追加 (デフォルト `0.0` = 無効)。
kill 発動時にタイムスタンプを記録し、`check_kill()` の冒頭（probe/cooldown チェックの前）で時間経過を確認。
上限超過で自動解除。

**理論的根拠**: ローリングウィンドウの情報鮮度。長時間更新されないウィンドウは現在の市場状態を反映せず、
判断材料として陳腐化する。30 分を上限とすることで、情報の失効を制度化。

### 2.3 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `ztb/risk/sell_dynamic_kill.py` | `max_kill_duration_sec` config, `_kill_activated_at` slot, 時間チェック in `check_kill()`, reset/export/import 対応 |
| `scripts/v460/lib/fill_config.py` | `sell_dynamic_kill_max_duration_sec`, `buy_dynamic_kill_max_duration_sec` フィールド + YAML parsing |
| `scripts/v460/run_fill_test.py` | Constructor + hot-reload rebuild で `max_kill_duration_sec` を配線 |
| `configs/v460/fill_test.yaml` | `max_kill_duration_sec: 1800` (30 分) を sell/buy 両方に設定 |

### 2.4 タイムスタンプのライフサイクル

```
track() 呼出 → _kill_activated_at = None (新データでリセット)
check_kill() kill 発火 → _kill_activated_at = time.time() (初回のみ記録)
check_kill() 時間超過 → cooldown=0, _kill_activated_at=None, return killed=False
reset() → _kill_activated_at = None
export_state() → {"kill_activated_at": float|None}
import_state() → _kill_activated_at を復元
```

---

## 3. I3: 空サイクル halt カウント除外 (`untick_side_halt()`)

### 3.1 問題

`tick_side_halt()` はサイクル冒頭で無条件に `side_halt_remaining` をデクリメントする。
デッドロック中は全サイクルが `balance_forced_halt_block` または `per_side_dd_both_halt` の continue パスに入り、
実質的な取引試行がないまま halt カウンタが消費される。結果として halt が早期解除され、
十分な冷却期間が確保できない。

### 3.2 解決策

`DailyDrawdownGuard.untick_side_halt()` メソッドを追加。
デッドロックの continue パス直後に呼び出すことで、`tick_side_halt()` のデクリメントを補償する。

```python
def untick_side_halt(self) -> None:
    """273# I3: 空サイクル (halt + balance_forced 等) の halt カウント補償."""
    # sell
    if self._state.side_halted_sell and remaining < max:
        self._state.side_halt_remaining_sell += 1
    # buy (同様)
```

### 3.3 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/daily_drawdown_guard.py` | `untick_side_halt()` メソッド追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | `balance_forced_halt_block` / `per_side_dd_both_halt` パスで呼び出し |

### 3.4 安全性

- `side_halt_remaining` は `per_side_halt_cycles` を超えない (bounds check)
- halt されていない side は無操作
- 冪等性: tick + untick = 元の値

---

## 4. I6: Halt 解除後 Gate Grace Period (`halt_recovery_active`)

### 4.1 問題

halt 解除後、オーケストレータは通常のゲート評価を行う。
ソフトゲート (unknown_regime, ranging_buy_low_vol, trending_sell_skip, velocity_skip) が
halt 解除後のリカバリ期間中にブロックを続け、最初の約定まで 18 分の遅延を生じた。

### 4.2 解決策

`DailyDrawdownGuard.is_in_recovery(side)` メソッドで副作用なくリカバリ状態を照会。
リカバリ中は `CycleGateAggregator.evaluate()` に `halt_recovery_active=True` を渡し、
**ソフトゲート (Gate 1, 2, 3, 6, 7) をバイパス**。**ハードゲート (Gate 4, 5, 8, 9) は維持**。

### 4.3 ソフト vs ハード分類

| Gate | 名前 | 種別 | Recovery 中 |
|---|---|---|---|
| 1 | `unknown_regime_buy_skip` | Soft (政策) | **バイパス** |
| 2 | `ranging_buy_low_vol` | Soft (政策) | **バイパス** |
| 3 | `trending_sell_skip` | Soft (政策) | **バイパス** |
| 4 | `buy_dynamic_kill` | Hard (安全) | 維持 |
| 5 | `sell_dynamic_kill` | Hard (安全) | 維持 |
| 6 | `velocity_skip` | Soft (政策) | **バイパス** |
| 7 | `unknown_regime_sell_skip` | Soft (政策) | **バイパス** |
| 8 | `narrow_spread` | Hard (市場) | 維持 |
| 9 | `maker_price_precheck` | Hard (市場) | 維持 |

**理論的根拠**: halt 解除後の最優先はポジション再構築。
市場条件チェック (spread, price, kill) は安全に不可欠だが、
政策的スキップ (regime, velocity) は短期的に緩和しても安全性に影響しない。
リカバリ期間は既存の `per_side_recovery_cycles` で制限される。

### 4.4 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/daily_drawdown_guard.py` | `is_in_recovery(side)` メソッド追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | ゲート評価前に recovery 照会 + evaluate() に渡す |
| `scripts/v460/lib/cycle_gate_aggregator.py` | `halt_recovery_active` パラメータ追加, Gate 1/2/3/6/7 に grace 条件 |

---

## 5. Pattern B: Kill ↔ Halt 相互ロックの解消

268# で発見された Pattern B (sell_dynamic_kill ↔ per-side halt 相互ロック) は以下の組み合わせで解消:

1. **I5** `max_kill_duration_sec=1800`: kill が 30 分で自動解除 → ロックの一方を強制的に断つ
2. **I3** `untick_side_halt()`: デッドロック中の halt カウンタ浪費を防止 → halt の意味ある冷却を保証
3. **I6** `halt_recovery_active`: halt 解除後のソフトゲートバイパス → 再参入遅延を解消

---

## 6. テスト

### 6.1 新規テスト

`tests/unit/v460/test_273_kill_time_limit_halt_untick_recovery_grace.py` — **23 テスト**

| クラス | テスト数 | 対象 |
|---|---|---|
| `TestKillTimeLimit` | 7 | I5: 時間超過解除, 未達持続, disabled, track/reset/export/import |
| `TestUntickSideHalt` | 5 | I3: 補償, bounds, noop, buy 側, デッドロック保持 |
| `TestIsInRecovery` | 3 | I6: halt 前, 解除後, consume 後 |
| `TestHaltRecoveryGraceInGate` | 3 | I6: ソフト通常ブロック, recovery バイパス, ハード維持 |
| `TestPatternBMitigation` | 2 | Pattern B: kill 自動解除, halt 保持 |
| `TestConfigWiring` | 3 | YAML → Config → Manager の配線 |

### 6.2 回帰テスト

```
pytest tests/unit/v460/ -x -q
→ 3743 tests (3711 passed, 32 skipped, 0 failures)
```

---

## 7. 変更ファイル一覧

| ファイル | 行数変化 | 対象 Issue |
|---|---|---|
| `ztb/risk/sell_dynamic_kill.py` | +36 | I5 |
| `scripts/v460/lib/fill_config.py` | +8 | I5 |
| `scripts/v460/run_fill_test.py` | +4 | I5 |
| `configs/v460/fill_test.yaml` | +2 | I5 |
| `scripts/v460/lib/daily_drawdown_guard.py` | +37 | I3, I6 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +8 | I3, I6 |
| `scripts/v460/lib/cycle_gate_aggregator.py` | +6 | I6 |
| `tests/unit/v460/test_273_*.py` | +280 (新規) | 全 |

---

## 8. 残課題

268# の全 6 件 (I1–I6) + Pattern B が解決済み。次回以降の観察ポイント:

- **max_kill_duration_sec の最適値**: 1800 秒は理論値。実運用でのチューニング候補。
- **ソフトゲート grace のリカバリ期間**: `per_side_recovery_cycles` に依存。短すぎれば grace 効果が薄い。
- **untick の監視**: ログで untick 発生頻度を追跡し、デッドロック頻度の推移を確認。
