# 239# Feasible Quote Proactive Calculation: InfeasibleQuoteError + 制約前方移動

## 概要

232# §1.5 [P1] feasible set collapse 対応。
制約チェックの実行順序を最適化し、専用例外 `InfeasibleQuoteError` を導入。
構造的に不可能なサイクルの **早期離脱** と **型安全な分類** を実現。

## 背景・問題

### 旧構造の問題

1. **sell_max_spread_jpy チェックが offset 計算後**: inventory skewing, sell offset floor, regime boosts 等の重い計算を行った後に spread 超過を検出し ValueError。構造的に不可能な sell サイクルで無駄な計算が走っていた。

2. **文字列パースによるエラー分類**: executor が `str(e).lower()` で `"sell_guard"` や `"spread too narrow"` を string match — 脆弱でリファクタリング耐性が低い。

3. **fallback price 処理の重複**: `InfeasibleQuoteError` 用と generic Exception 用で ~20 行の同一コードが重複。

### 市場理論的根拠

Avellaneda-Stoikov (2008): maker の quote は best bid/ask 内で feasible set を形成。
`min_spread_jpy` (下界) と `sell_max_spread_jpy` (上界) の 2 制約が feasible set の境界条件。

- 両制約とも spread のみに依存（offset 不要）
- offset 計算前にチェック可能 → 早期離脱で O(1) 判定
- 構造的に空集合の場合: `min_spread > sell_max_spread` → どの offset でも不可能

## 実装内容

### A. InfeasibleQuoteError (新規)

`scripts/v460/lib/maker_price.py`

```python
class InfeasibleQuoteError(ValueError):
    __slots__ = ("reason",)
    def __init__(self, reason: str, msg: str) -> None: ...
```

| reason | 意味 |
|--------|------|
| `"spread_too_narrow"` | spread < min_spread_jpy |
| `"sell_guard_reject"` | sell 時 spread > sell_max_spread_jpy |

- `ValueError` サブクラス → 既存 `except ValueError`/`except Exception` との後方互換を維持
- `__slots__` でメモリ効率確保

### B. compute() 制約順序変更

変更前:
```
1. OB fetch
2. min_spread check → ValueError
3. offset 決定 (base, inv_skew, sell_floor)
4. sell_max_spread check → ValueError  ← ここ
5. regime/spread_adaptive/VG/imbalance/loss_boost/FFD
6. finalize
```

変更後:
```
1. OB fetch
2. min_spread check → InfeasibleQuoteError  ← 前方
3. sell_max_spread check → InfeasibleQuoteError  ← 前方移動
4. offset 決定 (base, inv_skew, sell_floor)
5. regime/spread_adaptive/VG/imbalance/loss_boost/FFD
6. finalize
```

**効果**: spread > sell_max_spread の場合、inventory skewing 〜 finalize の全計算をスキップ。

### C. fill_cycle_executor.py

- `InfeasibleQuoteError` import 追加
- `except InfeasibleQuoteError as e:` を `except Exception as e:` の前に追加
  - `e.reason` で直接分類 — 文字列パース不要
  - no_feasible_quote 連続カウンタ処理をここに集約
- generic `except Exception` から `sell_guard`/`spread_too_narrow` の string match を削除

### D. _make_price_error_skip() ヘルパー (新規)

fallback price 取得 + skip record 生成の共通処理を抽出:
- `InfeasibleQuoteError` catch と generic Exception catch の両方から呼び出し
- ~20 行のコード重複を排除
- run_single_cycle の行数制限 (710 行) 内を維持

### E. テスト

22 新テスト (6 クラス):

| クラス | テスト数 | 内容 |
|--------|----------|------|
| TestInfeasibleQuoteError | 6 | ValueError 互換, reason 属性, slots |
| TestSellMaxSpreadEarlyBailout | 4 | sell/buy 分岐, spread_too_narrow, 無制限 |
| TestSellGuardSingleLocation | 2 | raise 1 箇所のみ, 旧 ValueError 不在 |
| TestExecutorInfeasibleCatch | 4 | except 存在, string match 除去, import |
| TestMakePriceErrorSkipHelper | 3 | メソッド存在, シグネチャ, 委譲確認 |
| TestFeasibleQuoteTheory | 3 | 制約非交差, feasible window, ソース順序 |

## テスト結果

- 変更前: 3264 tests passed
- 変更後: **3286 tests passed** (+22 新規)
