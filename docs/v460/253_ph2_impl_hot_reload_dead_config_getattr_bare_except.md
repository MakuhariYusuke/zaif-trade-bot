# 253# 実装: hot_reload 配線漏れ / dead config 削除 / getattr 排除 / bare except 改善

| 項目 | 内容 |
|------|------|
| 前提 | 252# self-review + codebase sweep (253# pre_impl) |
| テスト | 3526 件 pass (+19) |
| 対象 | 保守性向上・配線漏れ修正・型安全化・可観測性改善 |

## P1-1: `sell_asymmetric_high_vol_enabled` hot_reload + YAML 配線

252# で追加した `sell_asymmetric_high_vol_enabled` が `_HOT_RELOADABLE_FIELDS` 
に登録されていなかった (hot_reload 効かない)。

### 修正
- `config_hot_reload.py`: `_HOT_RELOADABLE_FIELDS` に追加
- `fill_test.yaml`: `loss_control` セクションに `sell_asymmetric_high_vol_enabled: false` 追加

### 理論根拠
Glosten-Milgrom モデルに基づく sell asymmetric 制御は、
市場レジーム変化に即応しなければ意味がない。
hot_reload 対象化により、プロセス再起動なしで有効/無効を切替可能に。

---

## P1-2: `balance_forced_apply_trending_offset` dead config 完全削除

234# で gate bypass 廃止時に dead config 化した `balance_forced_apply_trending_offset` を、
TODO(235#) の指示通り完全削除。

### 削除箇所 (4ファイル)
1. `fill_config.py`: フィールド定義削除
2. `fill_config.py`: YAML parse (`止血.get(...)`) 削除
3. `config_hot_reload.py`: `_HOT_RELOADABLE_FIELDS` から除去
4. `fill_test.yaml`: エントリ削除（コメント残存）

### テスト修正 (4ファイル)
- `test_196_velocity_proportional_trending_soft.py`: 参照削除
- `test_197_boost_optimization_gate_integration.py`: クラス `TestBalanceForcedTrendingOffset` 再構築
- `test_234_gate_bypass_removal.py`: `TestDeadConfigDeprecation` → 削除検証に変更
- `test_169_config_hot_reload.py`: soft-guard フィールド一覧から除去

---

## P1-3: `fill_cycle_executor.py` getattr 6件排除

### 背景
Mixin パターンで `self` の属性が orchestrator 側で定義されるため、
`getattr(self, attr, default)` で防御的にアクセスしていた。
しかしクラスレベルデフォルト宣言で型安全に直接参照可能。

### 対応
1. クラスレベルデフォルト追加:
   - `_alert_offset_mult: float = 1.0`
   - `_alert_lot_mult: float = 1.0`
   - `_halt_recovery_lot_mult: float = 1.0`
   - `_daily_drawdown_guard: DailyDrawdownGuard | None = None`
2. TYPE_CHECKING import: `DailyDrawdownGuard`
3. getattr 5箇所 → 直接参照に変更
4. `_postonly_crossing_streak` の getattr+加算 → 直接 `+= 1`
5. `macro_regime_conflict_action` の getattr → `self.config.` 直接参照

### 据え置き
- `getattr(order, "order_id", None)`: 外部 API レスポンスオブジェクトの検査で、
  型が不定のため getattr は正当。

---

## P1-4: `event_logger.py` TeeWriter bare except 改善

### 問題
`TeeWriter.write()` / `flush()` の `except Exception: pass` × 2 箇所。
例外が完全に消失し、デバッグ困難。

### 修正
`pass` → `logger.debug("TeeWriter.xxx failed for %s", type(w).__name__, exc_info=True)`

stderr ミラーリング用の低レイヤークラスのため、DEBUG レベルが適切
(WARNING 以上にすると stderr 自体のエラーループリスク)。

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `config_hot_reload.py` | +sell_asymmetric, -balance_forced |
| `fill_config.py` | フィールド/YAML parse 削除 |
| `fill_cycle_executor.py` | クラスレベルデフォルト追加, getattr×6排除 |
| `event_logger.py` | bare except→logger.debug |
| `fill_test.yaml` | +sell_asymmetric, -balance_forced |
| `test_253_*.py` | 新規 19 テスト |
| `test_196_*.py` | dead config 参照削除 |
| `test_197_*.py` | dead config テスト再構築 |
| `test_234_*.py` | 削除検証に変更 |
| `test_169_*.py` | soft-guard フィールド更新 |
