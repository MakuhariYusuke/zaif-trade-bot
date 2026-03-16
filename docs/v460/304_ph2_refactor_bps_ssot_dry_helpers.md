# 304# BPS_FACTOR SSOT + DRY ヘルパー + マジックナンバー排除

> **文書番号**: 304#  
> **種別**: `refactor` (リファクタリング)  
> **作成日**: 2026-03-06  
> **コミット**: `407be0006` (refactor), `6abbddbdd` (hot_swap PID fix)  
> **テスト**: 3973 passed, 0 failed (57s)

---

## §1 概要

303# レビュー応答実装の直後に実施した品質リファクタリング。
コードベース全体の DRY 原則準拠度を向上させ、変更時の修正漏れリスクを排除した。

### 変更サマリ

| カテゴリ | 内容 | 対象ファイル数 |
|---|---|---|
| **SSOT 統一** | `BPS_FACTOR = 10_000` を 6 ファイルから 1 ファイルに集約 | 7 (新規1 + 修正6) |
| **DRY ヘルパー** | 重複コードブロックを static method に抽出 | 2 |
| **マジックナンバー排除** | 即値リテラルを名前付き定数に置換 | 1 |
| **バグ修正** | 重複 return 文の除去、None ガード最適化 | 2 |
| **ドキュメント強化** | docstring への理論参照・注意記載追加 | 1 |
| **運用スクリプト修正** | hot_swap_restart.ps1 PID 判定バグ修正 | 1 |

---

## §2 変更詳細

### 2.1 BPS_FACTOR SSOT (`constants.py` 新規作成)

**問題**: `_BPS_FACTOR = 10_000` が以下 6 ファイルに独立定義されていた:
- `adaptation_engine.py`
- `maker_price.py`
- `order_monitor.py`
- `velocity_math.py`
- `fill_record_helpers.py`
- `pnl_measurer.py`

**解決**: `scripts/v460/lib/constants.py` を新設し、`BPS_FACTOR: Final[int] = 10_000` を SSOT として定義。
全 6 ファイルが `from scripts.v460.lib.constants import BPS_FACTOR` に統一。

```python
# scripts/v460/lib/constants.py
from typing import Final
BPS_FACTOR: Final[int] = 10_000
```

**効果**: 将来の bps 基準変更時に 1 箇所の修正で済む。`Final` 型ヒントにより再代入を静的検出。

### 2.2 DRY ヘルパー: `_recalc_price_with_new_offset()`

**問題**: `fill_cycle_executor.py` 内の reprice / offset 変更後の価格再計算ロジックが 2 箇所に重複。
mid を逆推定 → 新 offset で再計算する同一パターン。

**解決**: `@staticmethod _recalc_price_with_new_offset(side, order_price, spread, old_ratio, new_ratio)` を抽出。

```python
@staticmethod
def _recalc_price_with_new_offset(
    side: str, order_price: float, spread_at_order: float | None,
    old_ratio: float, new_ratio: float,
) -> float:
    """mid 逆推定 → 新 offset ratio で price 再算出."""
    if spread_at_order is None or spread_at_order <= 0:
        return order_price
    if side == "buy":
        mid_est = order_price + spread_at_order * old_ratio / 2
        return round(mid_est - spread_at_order * new_ratio / 2)
    else:
        mid_est = order_price - spread_at_order * old_ratio / 2
        return round(mid_est + spread_at_order * new_ratio / 2)
```

### 2.3 DRY ヘルパー: `_side_pnl_bps()`

**問題**: `pnl_measurer.py` 内の buy/sell PnL bps 計算が 4 箇所に重複。
buy は mid 上昇が利益、sell は mid 下落が利益、という同一パターン。

**解決**: `@staticmethod _side_pnl_bps(side, mid_at_fill, mid_after)` を抽出し、4 call sites を統一。

```python
@staticmethod
def _side_pnl_bps(side: str, mid_at_fill: float, mid_after: float) -> float:
    """side 別 PnL bps 計算 (buy: mid上昇が利益, sell: mid下落が利益)."""
    if side == "buy":
        return (mid_after - mid_at_fill) / mid_at_fill * _BPS_FACTOR
    return (mid_at_fill - mid_after) / mid_at_fill * _BPS_FACTOR
```

### 2.4 マジックナンバー排除 (`daily_drawdown_guard.py`)

| 即値 | 名前付き定数 | 用途 |
|---|---|---|
| `0.01` | `_WARMUP_REPAIR_EPS` | warmup 期間の PnL 修復 ε (浮動小数点比較閾値) |
| `1.0` | `_WARMUP_REPAIR_MIN_PNL` | warmup 期間の最低 PnL (これ以上でないと修復不要) |

### 2.5 バグ修正

#### 2.5.1 `maker_price.py`: 重複 return 文

```python
# Before (バグ)
    return effective_offset_ratio
    return effective_offset_ratio  # 到達不能

# After
    return effective_offset_ratio
```

#### 2.5.2 `fill_cycle_executor.py`: `_consecutive_no_feasible` None ガード

**問題**: `_consecutive_no_feasible` が `None` の可能性があり、4 箇所で `if self._consecutive_no_feasible is not None` ガードが必要だった。

**解決**: 初期値を `None` → `{}` (空辞書) に変更。ガード不要に。

#### 2.5.3 `fill_cycle_executor.py`: cancel_reason リテラル → enum

`cancel_reason = "unknown"` → `cancel_reason = CR.UNKNOWN` に変更。
`cancel_reasons.py` モジュールの enum 定数を使用し、typo リスクを排除。

### 2.6 ドキュメント強化 (`ab_judgment.py`)

- `_norm_cdf()`: Abramowitz & Stegun (1965) Handbook of Mathematical Functions, Eq. 7.1.26 への理論参照を追加
- `counterfactual_pnl30_bps`: 本値は約定時点 mid 基準であり、30s 後の市場変動のみを反映する旨の staleness note を追加

### 2.7 hot_swap_restart.ps1 PID 判定修正

**問題**: `Start-Process` が返す PID はランチャープロセス。実際の Python プロセスは子プロセスとして起動されるため、PID が不一致で正常起動をエラーと誤判定していた。

**解決**: ロックファイルに記録された PID の alive チェックに変更。親 PID が死んでいても子プロセス (= bot 本体) が生存していれば正常と判定。

---

## §3 理論的背景

### BPS_FACTOR の意味

basis point (bp) = 0.01% = 1/10,000。HFT / マーケットメイキングにおいて、
取引コスト・スプレッド・PnL を bps 単位で表現するのは業界標準
(Aldridge 2013, "High-Frequency Trading: A Practical Guide to Algorithmic Strategies")。

`BPS_FACTOR = 10_000` は `pnl_ratio * 10_000` で bps 換算を行う定数であり、
6 ファイルに渡る SSOT 化は数値一貫性の担保に直結する。

### mid 逆推定の妥当性

`_recalc_price_with_new_offset()` は、Avellaneda & Stoikov (2008) の
reservation price モデルにおける `δ±` (bid/ask offset) の動的調整を
mid 逆推定経由で実現している。

```
order_price = mid ∓ spread × offset_ratio / 2
∴ mid = order_price ± spread × offset_ratio / 2
new_price = mid ∓ spread × new_ratio / 2
```

spread が不明または 0 以下の場合はフォールバックとして元価格を返す安全設計。

---

## §4 テスト結果

| スイート | 結果 |
|---|---|
| 全 v460 テスト | 3973 passed, 0 failed |
| 実行時間 | 57.08s |
| 回帰テスト | 既存テストに変更なし、全パス |

---

## §5 関連文書

| 文書 | 関係 |
|---|---|
| [303#](303_ph2_resp_301_302_review_response.md) | 直前の実装 (本リファクタの対象) |
| [301#](301_ph2_rev_292_300_multifaceted_review.md) | Codex レビュー (F6 offset stage 記録は次回) |
| [302#](302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md) | Gemini 3.1 Pro レビュー |
| [300#](300_ph2_rev_ab_test_deep_analysis.md) | A/B テスト深層分析 |

---

## §6 残タスク (303# からの引き継ぎ)

| # | タスク | 優先度 | 状態 |
|---|---|---|---|
| F3 | hot-reload E2E テスト追加 | P2 | 保留 |
| F4 | forced buy α/repair 分離評価 | P2 | Dashboard 拡張で対応可 |
| F5 | BH/bootstrap 統計強化 | P2 | サンプル蓄積後 |
| F6 | offset stage-by-stage 記録 | P1 | 次回実装 |
| 盲点2 | Taker 執行化 (Toxic Forced Repair) | P1 | Phase A 凍結解除後 |
