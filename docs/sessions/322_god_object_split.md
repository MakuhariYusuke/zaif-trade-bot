# 322# God Object 分割 — maker_price.py Mixin 化

## 概要
`maker_price.py` MakerPriceCalculator (1,692 行, 上限 850 行の 199% 超過) を
3 つの Mixin モジュールに分割し、996 行 (41% 削減) に縮小。

## 変更概要

### 新規ファイル
| ファイル | 行数 | 責務 |
|---|---|---|
| `maker_regime_boost.py` | ~210 | Regime別 offset boost (7メソッド) |
| `maker_microstructure.py` | ~260 | σ推定, AS予約価格, Kyle λ, Amihud ILLIQ (6メソッド) |
| `maker_risk_guards.py` | ~250 | VG, Imbalance risk, Buy AS guard, Sell hour boost (4メソッド) |

### maker_price.py の変更
- `class MakerPriceCalculator(RiskGuardsMixin, MicrostructureMixin, RegimeBoostMixin):` — 3 Mixin 継承
- 17 メソッドを Mixin に移管、ローカル実装を削除
- 未使用 import (`datetime`, `timezone`, `FillTestRegime`) を除去
- **1,692 → 996 行 (696行削減、41%減)**

### テスト修正 (4ファイル)
| テスト | 修正内容 |
|---|---|
| `test_143_regime_utilization.py` | `getsource(MakerPriceCalculator)` → 個別メソッド |
| `test_157_regime_features.py` | `getsource` → `_resolve_trending_boost` |
| `test_fill_quality.py` | VG/trending source検査 → 個別メソッド |
| `test_306_proposals.py` | datetime mock パス → `maker_risk_guards` |
| `test_regime_detector.py` | FillTestRegime import 検査 → `maker_regime_boost` |

### Mixin アーキテクチャ
```
MakerPriceCalculator
  ├── RiskGuardsMixin        (VG, imbalance, buy_as_guard, sell_hour)
  ├── MicrostructureMixin    (σ, τ, AS shift, Kyle λ, Amihud ILLIQ)
  └── RegimeBoostMixin       (5 regime boost stages + dispatcher)
```

MRO (Method Resolution Order) により `hasattr()` と `inspect.getsource(method)` は保持。

## 再利用可能性分析
| Mixin | 再利用先候補 | 優先度 |
|---|---|---|
| RegimeBoostMixin | backtest regime 別 offset シミュレーション、regime 分類器統合 (4+ 実装並存) | P0 |
| MicrostructureMixin | σ 推定 (4 実装並存) の統合、Kyle λ/Amihud のオフライン分析 | P1 |
| RiskGuardsMixin | VPIN 計算の `ztb/features/microstructure.py` との共有、ToD 分析共通化 | P2 |

## テスト結果
- **4004 passed, 0 failed** (test_enricher_skip_gate.py, test_ml_pipeline.py は session037 WIP で除外)

## 残課題
- maker_price.py 996 行 (目標 850 まであと 146 行)
  - `compute()` 295 行、`_apply_spread_adaptive` 等は相互依存が強く分割コスト高
- fill_cycle_executor.py 1,502 行 — 次の分割対象候補
