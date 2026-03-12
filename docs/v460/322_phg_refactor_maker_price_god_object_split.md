# 322# God Object 分割 — maker_price.py Mixin 抽出

> **種別**: refactor  
> **起票**: 2026-03-07  
> **起源**: 321# §4 God Object 分割計画  
> **コミット**: `712b5b1ab`

---

## §1 概要

`maker_price.py` (1,692 行) は 321# §4.2 で特定された God Object。
3 つの Mixin モジュールに責務分離し、**1,692 → 996 行 (41% 削減)** を達成。

---

## §2 分割結果

| ファイル | 行数 | 責務 |
|---|---|---|
| `maker_price.py` | 1,692 → **996** (-41%) | 主ロジック: `compute()`, inventory, 各種 apply |
| `maker_regime_boost.py` **(NEW)** | **276** | regime 別 offset boost 7 メソッド |
| `maker_microstructure.py` **(NEW)** | **342** | σ推定, AS reservation, Kyle λ, Amihud ILLIQ |
| `maker_risk_guards.py` **(NEW)** | **280** | VG, imbalance リスク, buy AS guard, sell hour boost |

### 抽出メソッド

**RegimeBoostMixin** (maker_regime_boost.py):
- `_apply_regime_boosts` — regime 別 boost dispatcher
- `_regime_boost_trending_up` — 上昇トレンド boost
- `_regime_boost_trending_down` — 下降トレンド boost
- `_regime_boost_ranging` — レンジ regime 割引
- `_regime_boost_high_vol` — 高ボラ regime boost
- `_regime_boost_unknown` — unknown regime ハンドリング
- `_resolve_trending_boost` — trending boost 解決ヘルパー

**MicrostructureMixin** (maker_microstructure.py):
- `_estimate_sigma` — Parkinson σ 推定
- `_dynamic_tau` — GLFT τ 動的化
- `_apply_as_reservation_shift` — Avellaneda-Stoikov AS 価格シフト
- `_apply_kyle_lambda` — Kyle λ 情報非対称補正
- `_apply_amihud_illiq` — Amihud ILLIQ 非流動性補正
- `_get_depth` — OB depth 取得ヘルパー

**RiskGuardsMixin** (maker_risk_guards.py):
- `_apply_volatility_guard` — VG offset boost
- `_apply_imbalance_risk` — 在庫偏重リスク調整
- `_apply_buy_as_guard` — buy 側 AS 防御
- `_apply_sell_hour_boost` — 時間帯別 sell boost

### 継承チェーン

```
MakerPriceCalculator(RiskGuardsMixin, MicrostructureMixin, RegimeBoostMixin)
```

MRO により全メソッドがインスタンス上で解決。

---

## §3 テスト互換性対応

5 つのテストファイルが `inspect.getsource(MakerPriceCalculator)` でソースコードを検証していたため、
Mixin 分割後もテストが通るよう、対象メソッドの `inspect.getsource(method)` に変更。

| テストファイル | 変更内容 |
|---|---|
| `test_253` | `inspect.getsource` → メソッド単位検証 |
| `test_195` | 同上 |
| `test_239` | 同上 |
| `test_266` | 同上 |
| `test_196` | 同上 |

---

## §4 再利用可能性分析

322# 分割で抽出した 3 Mixin の、他モジュールでの再利用可能性を評価。

| 優先度 | 対象 | 内容 | 判定 |
|---|---|---|---|
| P0 | RegimeBoostMixin | regime 検出は `regime_detector.py` 固有ロジック。boost 計算は maker_price と密結合 | **再利用不可** — maker_price 専用 |
| P1 | MicrostructureMixin | σ推定/VPIN は他の 4 実装と入力形態が異なる (リアルタイム streaming vs batch) | **再利用困難** — 323# で精査済 |
| P2 | RiskGuardsMixin | price impact/spread proxy は backtest で使える可能性 | **将来検討** — backtest 移行時 |

---

## §5 テスト結果

- **4004 passed, 0 failed**
- 全 v460 テスト通過確認

---

## §6 変更ファイル一覧

| ファイル | 操作 |
|---|---|
| `scripts/v460/lib/maker_regime_boost.py` | NEW (276行) |
| `scripts/v460/lib/maker_microstructure.py` | NEW (342行) |
| `scripts/v460/lib/maker_risk_guards.py` | NEW (280行) |
| `scripts/v460/lib/maker_price.py` | MODIFIED (1692→996行) |
| `tests/unit/v460/test_253_*` | MODIFIED (inspect.getsource 修正) |
| `tests/unit/v460/test_195_*` | MODIFIED (同上) |
| `tests/unit/v460/test_239_*` | MODIFIED (同上) |
| `tests/unit/v460/test_266_*` | MODIFIED (同上) |
| `tests/unit/v460/test_196_*` | MODIFIED (同上) |

---

## §7 関連ドキュメント

| # | 関係 |
|---|---|
| 321 | §4 God Object 分割計画 (本実装の元) |
| 323 | fill_cycle_executor.py の分割 (本手法を踏襲) |
| 260 | compute() extract method — 先行する分割作業 |
