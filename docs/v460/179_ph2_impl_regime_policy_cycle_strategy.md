# 179# — RegimePolicyConfig + CycleStrategy + Chase 実装

> **分類**: ph2_impl  
> **前提**: 178# 177レビュー評価  
> **コミット**: 本作業

---

## §1 概要

178# で承認された Phase 1–3 実装:

| 施策 | 内容 | ファイル |
|------|------|----------|
| S1 | `_effective_sleep()` 一元化 | `fill_loop_orchestrator.py` |
| S2 | `RegimePolicyConfig` 分離 | `regime_policy.py` (新規) |
| S3 | `CycleStrategy` Protocol | `regime_policy.py` |
| C | Dynamic Cycle Interval | `DefaultCycleStrategy.effective_interval()` |
| D | Regime-linked Post-Fill Wait | `DefaultCycleStrategy.effective_post_fill_wait()` |
| Chase | Stale reprice aggressive 拡張 | `order_monitor.py` + `fill_cycle_executor.py` |

---

## §2 新ファイル: `regime_policy.py` (250 行以下)

### RegimePolicyConfig

```python
@dataclass
class RegimePolicyConfig:
    dynamic_cycle_enabled: bool = False
    cycle_intervals: dict[str, float]    # regime → interval (sec)
    dynamic_wait_enabled: bool = False
    post_fill_wait: dict[str, dict[str, float]]  # regime → {side → wait_sec}
    chase_enabled: bool = False
    chase_drift_bps: float = 3.0
    chase_max_reprice: int = 5
    chase_regimes: list[str]
    # 停止条件
    api_error_rate_threshold: float
    fill_rate_floor: float
    pnl_floor_bps: float
```

### CycleStrategy Protocol

```python
@runtime_checkable
class CycleStrategy(Protocol):
    def effective_interval(self, regime: str | None) -> float: ...
    def effective_post_fill_wait(self, side: str, regime: str | None) -> float: ...
    def is_chase_enabled(self, regime: str | None) -> bool: ...
    def chase_drift_bps(self) -> float: ...
    def chase_max_reprice(self) -> int: ...
```

### DefaultCycleStrategy

- `dynamic_cycle_enabled=False` → base 固定
- `dynamic_cycle_enabled=True` → regime lookup + fallback
- Fallback: 停止条件トリガーで一定時間 ranging モードに自動退避

---

## §3 S1: `_effective_sleep()` — 14 箇所 → 1 メソッド

**Before**: `await asyncio.sleep(self.config.cycle_interval_sec)` が 14 箇所に散在  
**After**: `await self._effective_sleep()` で CycleStrategy に委譲

```python
async def _effective_sleep(self, *, multiplier: float = 1.0) -> None:
    regime = self._current_regime_value()
    base = self._cycle_strategy.effective_interval(regime)
    await asyncio.sleep(base * multiplier)
```

- skip/halt/error continue: `await self._effective_sleep()`
- daily drawdown halt: `await self._effective_sleep(multiplier=5.0)`
- 正常サイクル完了: rapid_exit ロジック + strategy.effective_interval()

結果: orchestrator 1248 → 1162 行 (MAX 1200 以下に復帰)

---

## §4 C: Dynamic Cycle Interval

YAML:
```yaml
regime_policy:
  dynamic_cycle:
    enabled: true
    intervals:
      ranging: 120.0
      trending: 60.0
      trending_up: 60.0
      trending_down: 60.0
      high_vol: 120.0
```

効果: trending 時のサイクル頻度を 2× に上昇 → 価格追従性向上

---

## §5 D: Regime-linked Post-Fill Wait

YAML:
```yaml
regime_policy:
  dynamic_wait:
    enabled: true
    waits:
      ranging: {buy: 30, sell: 90}
      trending_up: {buy: 15, sell: 45}
      trending_down: {buy: 45, sell: 15}
```

- `PnlMeasurer.measure()` に `wait_sec_override` を追加
- `fill_cycle_executor._measure_post_fill_pnl()` から CycleStrategy 経由で注入
- 順方向側の wait を短縮 → 利確速度向上

---

## §6 Chase: Stale Reprice Aggressive 拡張

既存の stale order 検出 & cancel-replace を拡張:

- `chase_enabled=True` + trending regime → 低 drift 閾値 + 高 reprice 上限
- `OrderMonitor.monitor()` に `chase_drift_bps_override`, `chase_max_reprice_override` を追加
- `fill_cycle_executor._monitor_fill_polling()` が CycleStrategy から chase パラメータを取得

YAML:
```yaml
regime_policy:
  chase:
    enabled: true
    drift_bps: 3.0
    max_reprice: 5
    regimes: [trending_up, trending_down, trending]
```

---

## §7 Hot-Reload 対応

- `_HotReloadableRunner` Protocol に `_rebuild_cycle_strategy()` を追加
- `ConfigHotReloader._do_reload()` で YAML `regime_policy` セクション差分検知
- 変更検出 → `runner._rebuild_cycle_strategy()` で Strategy 再構築
- フォールバック: reload 失敗時は旧 Strategy を維持

---

## §8 安全設計: 停止条件

`RegimePolicyConfig` にフォールバック条件を内蔵:

| 条件 | 閾値 | 効果 |
|------|------|------|
| API エラー率 | > 3% (2h) | ranging フォールバック |
| fill_rate | < 35% (6h) | ranging フォールバック |
| avg pnl30 | < -0.8bps (6h) | ranging フォールバック |

`DefaultCycleStrategy.activate_fallback(duration_sec)` で時限フォールバック。

---

## §9 テスト

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestRegimePolicyConfig | 7 | YAML パース、デフォルト値 |
| TestCycleStrategyProtocol | 2 | Protocol 準拠 |
| TestEffectiveInterval | 7 | C: regime 別 interval |
| TestEffectivePostFillWait | 8 | D: regime×side wait |
| TestChase | 4 | Chase 有効/無効/パラメータ |
| TestFallback | 2 | Fallback 動作/期限切れ |
| TestOrderMonitorChaseIntegration | 1 | Chase パラメータ伝播 |
| TestPnlMeasurerOverride | 1 | wait_sec_override 受入 |
| TestEffectiveSleep | 2 | method 存在/async 確認 |
| TestHotReloadProtocol | 1 | Protocol 準拠 |
| TestRegimeSideMatrix | 30 | 全 regime×side 網羅 |
| **合計** | **65** | |

結果: 65/65 PASSED + 既存 99 テスト全 PASSED (regression-free)

---

## §10 変更ファイル一覧

| ファイル | 変更種別 | 行数変化 |
|----------|---------|---------|
| `lib/regime_policy.py` | 新規 | +245 |
| `lib/fill_loop_orchestrator.py` | 修正 | 1248 → 1162 (-86) |
| `lib/config_hot_reload.py` | 修正 | +15 |
| `lib/order_monitor.py` | 修正 | +6 |
| `lib/fill_cycle_executor.py` | 修正 | +15 |
| `lib/pnl_measurer.py` | 修正 | +10 |
| `run_fill_test.py` | 修正 | +20 |
| `test_179_*.py` | 新規テスト | +392 |
