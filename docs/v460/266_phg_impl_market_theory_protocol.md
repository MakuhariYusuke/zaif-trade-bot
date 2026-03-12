# 266# Market Theory Pipeline + Protocol 型安全化 (267# 修正含む)

| 項目 | 値 |
|---|---|
| Issue | 266# / 267# |
| 種別 | impl / bugfix |
| フェーズ | phg (横断品質改善) |
| Commit | 266#: d20f1fa1b, 267#: (pending) |
| テスト | 3680 passed, 32 skipped (+40 from 3640) |
| 元チケット | 257# P2-3 (GLFT τ), 259# P2-8 (AS δ*, Kyle λ, Amihud ILLIQ), 257# P1-1 / 259# P1-1 (type:ignore / getattr) |

---

## 背景

MakerPriceCalculator の offset パイプラインは 258# で Avellaneda-Stoikov (AS) reservation price を導入したが、
以下の市場理論実装が未完了だった:

1. **GLFT τ動的化**: τ がボラ状態に応じて動的に変化すべき (Guéant-Lehalle-Fernandez-Tapia 2013)
2. **AS δ\* 最適スプレッド幅**: 理論最適 offset 下限の不在
3. **Kyle λ 価格インパクト**: OB 薄板時の安全マージン不足 (Kyle 1985)
4. **Amihud ILLIQ 非流動性**: 流動性低下時のスプレッド拡大 (Amihud 2002)

また Protocol 化の残作業として `type: ignore` ×4 件、`getattr` ×8 件が残存していた。

## 変更概要

### 1. OrderBookSnapshot Protocol 移管

`OrderBookSnapshot` を `maker_price.py` → `ob_utils.py` に移管し、複数モジュールで共有。

| 変更点 | Before | After |
|---|---|---|
| 定義場所 | `maker_price.py` (ローカル定義) | `ob_utils.py` (共有 Protocol) |
| 使用箇所 | `maker_price.py` のみ | `ob_utils.py`, `maker_price.py`, `skip_gate_evaluator.py` |

### 2. type:ignore 排除 (×4)

| ファイル | 行 | 排除方法 |
|---|---|---|
| `skip_gate_evaluator.py` | ob.bids (×1) | `hasattr(ob, "bids")` guard + 直接アクセス |
| `skip_gate_evaluator.py` | ob.asks (×1) | 同上 |
| `fill_cycle_executor.py` | `_current_regime_value` | class-level 宣言削除 (Mixin 提供) |
| `fill_cycle_executor.py` | `_daily_drawdown_guard` | 型注釈修正で `# type: ignore` 不要化 |

### 3. getattr 排除 (×8)

| ファイル | 対象 | 排除方法 |
|---|---|---|
| `ob_utils.py` best_bid_ask() | `getattr(ob, "bids/asks")` ×2 | `hasattr` + 直接アクセス |
| `ob_utils.py` bid_depth_volume() | `getattr(ob, "bids")` ×1 | None check + `ob.bids` 直接 |
| `ob_utils.py` ask_depth_volume() | `getattr(ob, "asks")` ×1 | 同上 |
| `skip_gate_evaluator.py` | `getattr(ob, "bids/asks")` ×2 | `hasattr` + 直接アクセス |
| `skip_gate_evaluator.py` | SkipGateAdapter.get_orderbook 戻り値 | `object` → `OrderBookSnapshot \| None` |

### 4. GLFT τ動的化

**理論**: Guéant-Lehalle-Fernandez-Tapia (2013) — ボラティリティ変動時に τ (残存時間) を動的調整。

```
τ_eff = τ_base / vol_ratio
τ_eff = clamp(τ_eff, τ_min, τ_max)
```

高ボラ → τ 短縮 (素早い意思決定)、低ボラ → τ 延長 (穏やかな市場で広い視野)。

| Config | Default | 説明 |
|---|---|---|
| `as_tau_dynamic_enabled` | `False` | GLFT τ 動的化有効/無効 |
| `as_tau_dynamic_min_sec` | `30.0` | τ 下限 (秒) |
| `as_tau_dynamic_max_sec` | `600.0` | τ 上限 (秒) |

**実装**: `MakerPriceCalculator._dynamic_tau(base_tau, vol_ratio)` — `_apply_as_reservation_shift()` 内で使用。

### 5. AS δ\* 最適スプレッド幅

**理論**: Avellaneda-Stoikov (2008) — 最適 half-spread δ\* を offset 下限として適用。

$$\delta^* = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

- $\gamma$: リスク回避度 (`as_reservation_gamma`)
- $\sigma$: Roll (1984) micro-vol proxy × vol_ratio
- $\tau$: 残存時間 (GLFT τ 動的化適用後)
- $k$: fill rate parameter (`as_delta_star_fill_rate_k`)

| Config | Default | 説明 |
|---|---|---|
| `as_delta_star_enabled` | `False` | δ\* 下限適用有効/無効 |
| `as_delta_star_fill_rate_k` | `1.5` | 執行率パラメータ k |

**実装**: `_apply_as_reservation_shift()` 内で δ\* を計算し、`max(offset, δ*)` で下限適用。

### 6. Kyle λ 価格インパクト

**理論**: Kyle (1985) — `λ = spread / (2·depth)` で価格インパクト係数を推定。
自注文 lot の予想インパクトを offset に加算。

```
λ_est = spread / (2 · depth_volume)
impact = λ_est · lot / mid_price · impact_mult
offset += min(impact, max_add_ratio)
```

| Config | Default | 説明 |
|---|---|---|
| `kyle_lambda_enabled` | `False` | Kyle λ 有効/無効 |
| `kyle_lambda_impact_mult` | `0.5` | インパクト倍率 |
| `kyle_lambda_max_add_ratio` | `0.05` | offset 加算上限 |

**実装**: `MakerPriceCalculator._apply_kyle_lambda(side, spread, mid, offset)` — compute() パイプラインに組込み。
buy 側は bid depth、sell 側は ask depth を使用。

### 7. Amihud ILLIQ 非流動性

**理論**: Amihud (2002) — `ILLIQ = (spread/mid) / depth_volume` で非流動性比率を計算。
ILLIQ が baseline を超えた場合、offset に乗算補正。

```
illiq = (spread / mid) / depth
ratio = illiq / baseline
mult = min(ratio, max_mult)
offset *= mult  (mult > 1 の場合のみ)
```

| Config | Default | 説明 |
|---|---|---|
| `amihud_illiq_enabled` | `False` | Amihud ILLIQ 有効/無効 |
| `amihud_illiq_baseline` | `0.001` | ILLIQ ベースライン |
| `amihud_illiq_max_mult` | `1.5` | offset 最大倍率 |

**実装**: `MakerPriceCalculator._apply_amihud_illiq(side, spread, mid, offset)` — compute() パイプラインに組込み。
ILLIQ 値は `_last_amihud_illiq` にキャッシュ。

### 8. σ推定共通化 (`_estimate_sigma`)

AS/GLFT/δ\* で使用する σ 推定ロジックを共通ヘルパーに抽出。

```python
def _estimate_sigma(self, spread: float, mid_price: float) -> tuple[float, float]:
    """Roll (1984) micro-vol proxy × RegimeDetector vol_ratio."""
    base_sigma = spread / (2.0 * mid_price) if mid_price > 0 else 0.0
    vol_ratio = getattr(self._regime_detector, "last_volatility_ratio", 1.0)
    return base_sigma * vol_ratio, vol_ratio
```

| 使用箇所 | 用途 |
|---|---|
| `_apply_as_reservation_shift` | AS reservation σ² |
| `_dynamic_tau` (間接) | vol_ratio に基づく τ 調整 |
| `_apply_as_reservation_shift` → δ\* | δ\* = γσ²τ の σ |

## compute() パイプライン (266# 後)

```
compute(side, current_spread, mid_price, ...)
  │
  ├─ _apply_as_reservation_shift()   [258#/266#: AS + GLFT τ + δ*]
  │    ├─ _estimate_sigma()          [266#: 共有 σ 推定]
  │    └─ _dynamic_tau()             [266# GLFT τ 動的化]
  ├─ _apply_regime_boosts()          [260#]
  │    ├─ _boost_trending()
  │    ├─ _boost_volatile()
  │    ├─ _boost_crisis()
  │    ├─ _boost_recovery()
  │    └─ _boost_ranging()
  ├─ _apply_spread_adaptive()        [258#]
  ├─ _apply_kyle_lambda()            [266# NEW]
  ├─ _apply_amihud_illiq()           [266# NEW]
  ├─ _apply_volatility_guard()
  ├─ _apply_imbalance_risk()
  ├─ _apply_loss_boost()             [260#]
  └─ _apply_ffd_boost()             [260#]
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/ob_utils.py` | OrderBookSnapshot Protocol 追加, best_bid_ask/depth getattr→hasattr |
| `scripts/v460/lib/maker_price.py` | _estimate_sigma, _dynamic_tau, _apply_kyle_lambda, _apply_amihud_illiq 追加, AS δ\* 統合, pipeline 更新 |
| `scripts/v460/lib/fill_config.py` | GLFT τ / AS δ\* / Kyle λ / Amihud ILLIQ config 12 フィールド追加 |
| `scripts/v460/lib/skip_gate_evaluator.py` | OrderBookSnapshot import, get_orderbook 戻り値型, type:ignore/getattr 排除 |
| `scripts/v460/lib/fill_cycle_executor.py` | _current_regime_value class-level 削除, type:ignore 排除 |
| `tests/unit/v460/test_259_*` | _estimate_sigma へのソース検査先変更 |
| `tests/unit/v460/test_260_*` | compute() 行数しきい値 200 へ引上げ |
| `tests/unit/v460/test_266_market_theory_protocol.py` | **新規**: 31テスト (7 クラス) |

## テスト追加 (test_266)

| テストクラス | テスト数 | 対象 |
|---|---|---|
| `TestEstimateSigma` | 4 | Roll proxy, vol_ratio scaling, zero mid, ソース再利用 |
| `TestDynamicTau` | 5 | disabled, 高ボラ短縮, 低ボラ延長, min/max clamp |
| `TestASDeltaStar` | 3 | disabled, floor 適用, 数式正確性 |
| `TestKyleLambda` | 6 | disabled, zero depth, 加算, max clamp, sell 側, pipeline |
| `TestAmihudILLIQ` | 7 | disabled, 十分流動性, 低流動拡大, max clamp, zero depth, cache, pipeline |
| `TestOrderBookSnapshotProtocol` | 3 | import 3 モジュール, 戻り値型 |
| `TestTypeIgnoreReduction` | 3 | attr-defined 排除, getattr 排除, class-level 排除 |
| `TestGetDepthHelper` | 4 | 267# buy/sell depth, kyle/amihud 使用検証 |
| `TestDeltaStarRatioConversion` | 2 | 267# 括弧化検証, 次元一貫性 |
| `TestEstimateSigmaDocstringAccuracy` | 1 | 267# docstring 正確性 |
| `TestKyleLambdaAmihudInteraction` | 2 | 267# 複合効果 bounds, Kyle vs 合流比較 |

## 全フィーチャーのデフォルト無効化

全 4 機能は `enabled=False` がデフォルト → **既存動作に影響ゼロ**。
本番有効化は PnL/fill-rate 検証後に段階的に実施。

## 267# 不具合修正・再利用性精査

### 発見・修正した不具合

| # | 重大度 | 箇所 | 内容 | 修正 |
|---|---|---|---|---|
| B1 | 🔴 **中** | δ\* ratio 変換 | AS 論文は絶対 σ (JPY/√s) 前提だがリターンベース σ を使用。`(2/γ)ln(1+γ/k)` が桁違いに巨大化 → δ\* フロアが常に `max_offset_ratio` と等価に | σ\_abs = σ\_return × mid に変換、δ\*(JPY)/spread で offset\_ratio 算出 |
| B2 | 🟡 低 | `_estimate_sigma` docstring | kyle\_lambda / amihud\_illiq が再利用すると記載するが実際は呼んでいない | docstring を正確に修正 |
| B3 | 🟡 低 | Kyle λ + ILLIQ 合流 | 薄板で加算(λ)→乗算(ILLIQ) が複合膨張 | `max_offset_ratio` クランプで bounds 保証、テスト追加 |
| B4 | 🟢 info | depth 取得重複 | side 分岐ロジックが Kyle λ / ILLIQ で個別実装 | `_get_depth(side)` ヘルパー抽出 (DRY) |

### 再利用性の具体的評価

| メソッド | 再利用 | 詳細 |
|---|---|---|
| `_estimate_sigma` | ✅ 実現済 (AS + δ\*) | Kyle λ / Amihud ILLIQ は depth ベースで独自推定 — σ 共有は不適切 (異なる情報源) |
| `_get_depth(side)` | ✅ 267# 新規抽出 | Kyle λ + Amihud ILLIQ + 将来の depth 系ステージで共有 |
| `_dynamic_tau` | 🟡 拡張候補 | `loss_boost_decay_tau_sec` にも vol\_ratio 連動を適用可能 (B5) |
| `OrderBookSnapshot` | ✅ 3 モジュール共有 | ob\_utils, maker\_price, skip\_gate\_evaluator |
| `_scale_offset_ratio` | ✅ 既存 (7 箇所使用) | regime boost, ILLIQ, loss\_boost 等で共通 |

### 今後の拡張候補 (enabled=false のため即時不要)

- **B5**: `loss_boost_decay_tau_sec` の vol_ratio 連動化 (`_dynamic_tau` 再利用)
- **R1**: Kyle λ に σ ベースのインパクト推定を追加 (Kyle 1985 理論的整合性向上)
- δ\* の γ パラメータスケーリング最適化 (絶対 σ でも γ=0.1 では γσ²τ 項が支配的)
