# 259# Codebase Sweep Report

## 概要

258# (AS Reservation Price / VPIN Continuous / RegimeDetectorLike Protocol) 実装後のコードベーススイープ。
残存する技術的負債、市場理論による補強余地、型安全向上箇所を洗い出す。

---

## P1: 今回実装対象

### MT-4: AS σ² 推定の改善 — RegimeDetector volatility_ratio 統合

**現状**: `_apply_as_reservation_shift()` (257#) の σ² 推定は Roll (1984) の
spread-based micro-volatility proxy のみ: `σ = spread / (2·mid)`

**問題**: bid-ask spread は流動性依存であり、実際のボラティリティと乖離する場合がある。
特に薄板時に spread が広がっても真のボラティリティは低い場合、AS shift が過大になる。

**改善**: `RegimeDetectorLike.last_volatility_ratio` は直近 window の
returns std / baseline std を返す。これを σ 推定に組み込み:

```
σ_enhanced = σ_roll × max(vol_ratio, σ_floor)
```

vol_ratio > 1.0 (高ボラ) → σ 増幅 → AS shift 増大 (正しくリスク回避)
vol_ratio < 1.0 (低ボラ) → σ 短縮 → AS shift 抑制 (不要な broad offset 防止)

**理論的根拠**: Avellaneda-Stoikov (2008) §3 で σ² は realized volatility を
想定。Roll (1984) proxy に realized vol ratio を乗ずることで、
microstructure noise と true volatility を分離する hybrid estimator になる。

**リスク**: 低 — `as_reservation_enabled=False` がデフォルトのため本番影響なし。
regime_detector が None の場合は既存の Roll proxy にフォールバック。

### F-3: adaptation_engine hasattr 残存 (2箇所)

**現状**: `adaptation_engine.py` L211, L379 で `hasattr(regime_detector, "current_regime")` を使用。
既に `RegimeDetectorLike` を import し型注釈も適用済みだが、hasattr が残存。

**修正**: Protocol 型なので `regime_detector is not None` チェックのみで十分。
`hasattr` 削除で 257# Protocol 化を完遂。

---

## P2: 次回以降 (優先度順)

### P2-1: OrderBookLevelLike Protocol (ob_utils / ob_recorder)

`ob_utils.py` L30,39 / `ob_recorder.py` L67,69 で `getattr(level, "price/quantity/size")` を
6箇所使用。`OrderBookLevelLike` Protocol を定義し isinstance check に置換。

| ファイル | 件数 | 影響 |
|---|---|---|
| ob_utils.py | 4 | extract_price/extract_size |
| ob_recorder.py | 2 | _extract_level 内 |

**→ 261# 実装済**: `@runtime_checkable OrderBookLevelLike` Protocol 定義。
ob_utils は直接属性アクセス、ob_recorder は isinstance + _to_finite_float 経由で型検証。

### P2-2: maker_price.py compute() 214行 → 150行以下

ファイル内 GOD OBJECT 警告 (L85-98) で `compute()` 150行上限を明記済みだが現在 214行。
loss_boost / FFD boost の 2ブロック (~50行) を `_apply_loss_boost()`, `_apply_ffd_boost()` に
抽出すれば 150行以下に収まる。パイプライン構造なので安全。

**→ 260# 実装済**: compute() 214→180行。`_apply_loss_boost()`, `_apply_ffd_boost()` 抽出。

### P2-3: _apply_regime_boosts() 153行 → 5分割

5つの独立 if ブロック (trending/high_vol/ranging/unknown/low_vol) を
`_regime_boost_trending()`, `_regime_boost_high_vol()` 等に分割。
各 ~30行、ロジック変更なし。

**→ 260# 実装済**: 5 sub-method + dispatcher パターン。各 ~30行。

### P2-4: skip_gate_evaluator getattr (4箇所)

| 行 | コード | 推奨 |
|---|---|---|
| L234,236 | `getattr(trade, key, default)` | TradeLike Protocol |
| L830,852 | `getattr(self, attr_path/attr_hash)` | dict 化 |
| L991,1015 | `getattr(adapter, "get_recent_trades/get_orderbook")` | GateAdapterProtocol |

**状態**: L234/236 は dict/object dual-format で構造的に必要。L830/852 はメタプログラミング。
L991/1015 は GateAdapterProtocol 定義で置換可能だが影響範囲大。保留。

### P2-5: `_last_ob_snapshot: object | None` → `OrderBookSnapshot | None`

maker_price.py L165。OB snapshot は `dict | SimpleNamespace` のいずれかが入る。
型を特定し Protocol or Union で明示化。

**→ 261# 実装済**: `OrderBookSnapshot` Protocol (bids/asks プロパティ) 定義。
`OrderbookProvider.get_orderbook()` 戻り値も `OrderBookSnapshot` に型安全化。

### P2-6: config_hot_reload.py L400 private attr getattr

`getattr(runner, "_fast_fill_defense", None)` — private attribute への getattr。
公開 property or Protocol method に置換。

**→ 261# 実装済**: `_HotReloadableRunner` Protocol で宣言済みのため直接参照に変更。

### P2-7: balance_checker.py adapter: object (3箇所)

L88, 109, 165 で `adapter: object`。BalanceAdapterProtocol の定義が必要。

**→ 261# 実装済**: `BalanceAdapterProtocol` + `_BalanceLike` Protocol 定義。
`adapter: object` → `adapter: BalanceAdapterProtocol` に置換。type: ignore 排除。

---

## P3: 市場理論 — 未実装の理論的補強

### P3-1: AS 最適スプレッド幅 (δ*)

$$\delta^* = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

現在は `spread * offset_ratio` のヒューリスティックだが、AS 理論は fill rate `k` を
含む最適スプレッド幅を導出可能。`k` の推定に fill_record の約定確率統計が必要。

### P3-2: Kelly Criterion lot sizing

$$f^* = \frac{p \cdot b - q}{b}$$

`_recent_records` deque で win_rate (p) / avg_win:loss (b) は計算可能だが、
short-horizon (< 100 fills) での統計的信頼性に課題。fractional Kelly (f*/2) で
保守的に運用する案。

### P3-3: Kyle's Lambda — 自己注文価格インパクト

$$\Delta P = \lambda \cdot Q$$

lot size に応じた価格インパクト推定。現在の offset は lot size に無関係だが、
理論的には大きい lot ほど自己約定で price impact を受ける。

### P3-4: Amihud 非流動性比率

$$ILLIQ = \frac{1}{N} \sum \frac{|R_i|}{V_i}$$

spread_adaptive の閾値動的化に使用可能。日次 ILLIQ が高い場合は
narrow_spread_bps 閾値を引き上げる。

---

## 統計サマリ

| カテゴリ | 件数 | 状態 |
|---|---|---|
| getattr 残存 | 35 | 15件 Protocol 化対象, 17件 構造上必要, 3件 dict 化推奨 |
| hasattr 残存 | 7 | 2件 P1 対象 (adaptation_engine), 3件 skip_gate, 2件 許容 |
| bare except | 0 | ✅ 完全排除済 |
| `: object` 型注釈 | 43 | ~20件 Protocol 化対象, ~23件 許容 |
| `Any` 型 | 0 | ✅ 完全排除済 |
| 150行超メソッド | 4 | compute() 214行, _apply_regime_boosts() 153行, evaluate() 346行, run_continuous() 1694行 |
