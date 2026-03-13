# 181# C/D/Chase 有効化 + EV_weighted + Stop Condition Monitor

> **種別**: impl (ph2)  
> **日付**: 2026-02-28  
> **前提**: 179# RegimePolicyConfig/CycleStrategy 基盤, 180# from_yaml 堅牢化  
> **根拠**: 178# §1.3 EV_weighted 設計, 178# §2.1 C/D/Chase 有効化方針

---

## 1. 背景

179# で RegimePolicyConfig / DefaultCycleStrategy / Chase パラメータを実装したが、
全て `enabled: false` で本番に影響しない状態だった。

**問い: `enabled=false` で大値動きに対応できるのか?**

→ **否。** インフラだけでは何も変わらない。本チケットで有効化 + 安全弁を実装する。

## 2. 変更一覧

### 2.1 YAML: C/D/Chase 有効化 (`configs/v460/fill_test.yaml`)

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
  dynamic_wait:
    enabled: true
    waits:
      trending_up: { buy: 15.0, sell: 45.0 }
      trending_down: { buy: 45.0, sell: 15.0 }
      ranging: { buy: 30.0, sell: 90.0 }
      high_vol: { buy: 30.0, sell: 90.0 }
  chase:
    enabled: true
    drift_bps: 3.0
    max_reprice: 5
    regimes: [trending_up, trending_down]
  stop_conditions:
    api_error_rate_threshold: 0.03
    fill_rate_floor: 0.35
    pnl_floor_bps: -0.8
```

169# ConfigHotReloader により、次サイクルから自動反映される。

### 2.2 EV_weighted PnL (`fill_cycle_executor.py`)

```python
@staticmethod
def _compute_ev_weighted(pnl30, pnl120, *, w30=0.4, w120=0.6) -> float | None:
    if pnl30 is None: return None
    if pnl120 is None: return pnl30  # E3 サンプリング外
    return w30 * pnl30 + w120 * pnl120
```

- **178# §1.3 設計**: `0.4 * pnl30 + 0.6 * pnl120` で短期ノイズを抑制
- **E3 サンプリング不在時**: pnl120=None → pnl30 単独値にフォールバック
- FillRecord 構築時に自動計算、`ev_weighted_pnl` フィールドに格納

### 2.3 FillRecord 拡張 (`ztb/metrics/fill_quality.py`)

```python
ev_weighted_pnl: Optional[float] = None  # 0.4*pnl30 + 0.6*pnl120 (bps)
```

### 2.4 Stop Condition Monitor (`fill_loop_orchestrator.py`)

```python
def _check_regime_stop_conditions(self, filled_count, total_count) -> None:
```

- **実行タイミング**: 30 サイクルごと (~1h@120s, ~30min@60s)
- **fill_rate チェック**: `filled / total < fill_rate_floor (0.35)` → fallback 1h
- **avg_pnl30 チェック**: 直近 100 filled records の平均 < `pnl_floor_bps (-0.8)` → fallback 1h
- C/D/Chase 全 disabled 時はスキップ
- 最小サンプル 10 未満なら pnl チェック省略

## 3. CircuitBreaker 確認

158# で既に完全統合済:
- `ztb/utils/circuit_breaker.py` + `scripts/v460/lib/resilience.py`
- `fill_cycle_executor.py` L195, L494, L671 で呼び出し
- YAML `resilience:` セクションで設定済

→ **C (60s cycle) 有効化のブロッカーではない。**

## 4. テスト結果

| カテゴリ | テスト数 | 結果 |
|---|---|---|
| `_compute_ev_weighted` | 8 | ✅ ALL PASSED |
| `FillRecord.ev_weighted_pnl` | 4 | ✅ ALL PASSED |
| `_check_regime_stop_conditions` | 8 | ✅ ALL PASSED |
| **合計 (181#)** | **20** | **✅ 20/20 PASSED** |
| **回帰 (179# + 181#)** | **92** | **✅ 92/92 PASSED** |

## 5. ファイル行数

| ファイル | 変更前 | 変更後 | MAX |
|---|---|---|---|
| `fill_cycle_executor.py` | 675 | 700 | 700 |
| `fill_loop_orchestrator.py` | 1162 | 1198 | 1200 |
| `fill_quality.py` | 1245 | 1246 | — |
| `regime_policy.py` | 278 | 278 (変更なし) | — |

## 6. 残課題 (182# 以降)

| 優先度 | 課題 | 備考 |
|---|---|---|
| 🟡 | EV_weighted w30/w120 を YAML 外部化 | 現在は `_compute_ev_weighted()` のデフォルト引数 |
| 🟡 | Trend Mode 発動条件厳格化 | confidence + velocity + spread AND 条件 |
| 🟡 | 在庫偏り regime 別緩和 | `balance_forced_deadlock_limit` の regime 分岐 |
| 🟢 | 条件付き IOC (Phase 4) | Coincheck API IOC パラメータ要確認 |
| 🟢 | Mixin → 独立クラス化 | 長期改善、breaking change |
