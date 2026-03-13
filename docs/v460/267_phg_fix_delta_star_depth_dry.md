# 267# δ* 次元修正 + _get_depth DRY + docstring 正確化

| 項目 | 値 |
|---|---|
| Issue | 267# |
| 種別 | bugfix |
| フェーズ | phg (横断品質改善) |
| Commit | `227024d54` |
| 親 Issue | 266# (Market Theory Pipeline) |
| テスト | 3680 passed, 32 skipped (+9 tests from 266#) |

---

## 背景

266# で実装した Market Theory Pipeline の品質精査中に 4 件の不具合・改善点を発見。
全機能が `enabled=false` のため本番影響はないが、有効化前に修正が必要。

## 修正一覧

| # | 重大度 | 箇所 | 内容 | 修正 |
|---|---|---|---|---|
| B1 | 🔴 中 | δ\* ratio 変換 | AS 論文は絶対 σ (JPY/√s) 前提だがリターンベース σ を使用 → `(2/γ)ln(1+γ/k)` が桁違いに巨大化し δ\* フロアが常に `max_offset_ratio` と等価に | σ\_abs = σ\_return × mid\_price に変換、δ\*(JPY)/spread で offset\_ratio 算出 |
| B2 | 🟡 低 | `_estimate_sigma` docstring | kyle\_lambda / amihud\_illiq が再利用すると記載するが実際は depth ベースで独自推定 | docstring を正確に修正 |
| B3 | 🟡 低 | Kyle λ + ILLIQ 合流 | 薄板で加算(λ)→乗算(ILLIQ) が複合膨張 | `max_offset_ratio` クランプで bounds 保証、テスト追加 |
| B4 | 🟢 info | depth 取得重複 | side 分岐ロジックが Kyle λ / ILLIQ で個別実装 | `_get_depth(side)` ヘルパー抽出 (DRY) |

## 修正詳細

### B1: δ* 次元修正 (σ_return → σ_abs)

**問題**: AS (Avellaneda-Stoikov 2008 §4) の δ\* 公式:

$$\delta^* = \gamma\sigma^2\tau + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$$

は σ が **絶対価格 (JPY/√s)** ベースを前提とする。
266# では `_estimate_sigma` のリターンベース σ (無次元, ≈ 0.00003) をそのまま使用していたため、
σ² が ≈ 10⁻⁹ と極小になり、第 1 項が無視される一方、第 2 項 `(2/γ)ln(1+γ/k)` が ≈ 3.0 となり
δ\* が常に `max_offset_ratio` にクランプされていた。

**修正**:
```python
# Before (266#)
delta_star = gamma * sigma_sq * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)
delta_star_ratio = delta_star / spread * mid_price  # 次元不整合

# After (267#)
sigma_abs = sigma * mid_price          # リターン → 絶対価格 (JPY)
sigma_abs_sq = sigma_abs * sigma_abs
delta_star_jpy = gamma * sigma_abs_sq * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)
delta_star_ratio = delta_star_jpy / spread  # δ*(JPY) → offset_ratio (無次元)
```

### B4: `_get_depth(side)` DRY 抽出

```python
def _get_depth(self, side: str) -> float:
    """267# DRY: side に応じた板 depth volume を返す."""
    return self._last_bid_depth if side == "buy" else self._last_ask_depth
```

Kyle λ と Amihud ILLIQ の両方で使用。将来の depth 系ステージでも共有可能。

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/maker_price.py` | `_get_depth` 抽出, δ\* σ\_abs 変換, docstring 修正 |
| `tests/unit/v460/test_266_market_theory_protocol.py` | +9 テスト (3 クラス) |
| `docs/v460/266_phg_impl_market_theory_protocol.md` | 267# addendum セクション追加 |

## テスト追加 (+9)

| テストクラス | テスト数 | 対象 |
|---|---|---|
| `TestGetDepthHelper` | 4 | buy/sell depth 取得, Kyle/ILLIQ 使用検証 |
| `TestDeltaStarRatioConversion` | 2 | 括弧化検証, 次元一貫性 |
| `TestEstimateSigmaDocstringAccuracy` | 1 | docstring 正確性 |
| `TestKyleLambdaAmihudInteraction` | 2 | 複合効果 bounds, Kyle vs 合流比較 |

## 再利用性評価

| メソッド | 再利用 | 詳細 |
|---|---|---|
| `_estimate_sigma` | ✅ 実現済 (AS + δ\*) | Kyle λ / Amihud ILLIQ は depth ベースで独自推定 — σ 共有は不適切 |
| `_get_depth(side)` | ✅ 267# 新規抽出 | Kyle λ + Amihud ILLIQ + 将来 depth 系で共有 |
| `_scale_offset_ratio` | ✅ 既存 (7 箇所) | regime boost, ILLIQ, loss\_boost 等で共通 |

## 今後の拡張候補

- **B5**: `loss_boost_decay_tau_sec` の vol\_ratio 連動化 (`_dynamic_tau` 再利用)
- **R1**: Kyle λ に σ ベースのインパクト推定を追加 (Kyle 1985 理論的整合性向上)
- δ\* の γ パラメータスケーリング最適化 (絶対 σ でも γ=0.1 では γσ²τ 項が支配的)
