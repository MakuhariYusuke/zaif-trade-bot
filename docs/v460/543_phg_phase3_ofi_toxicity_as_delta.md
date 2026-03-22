# 543# Phase 3 実装: OFI-Lite / Toxicity Budget 独立化 / A-S δ* 計測

> 540# §6 Phase 3 アクションプラン実装  
> 日付: 2026-03-23

---

## §1 概要

540# §6 で定義された Phase 3「Prediction Hub への移行（中期）」の 3 項目を実装。
いずれも **計測・分離フェーズ** であり、pipeline への影響は offset_stages dict への記録のみ。

| # | 施策 | 種別 | 状態 |
|---|------|------|------|
| 3-1 | OFI-Lite: cycle 間 depth delta | Code | ✅ |
| 3-2 | Toxicity Budget 独立化 | Refactor | ✅ |
| 3-3 | A-S reference spread (δ*) 計測 | Code | ✅ |

---

## §2 Phase 3-1: OFI-Lite (Cont-Kukanov-Stoikov 2014)

### 設計思想

Order Flow Imbalance (OFI) を cycle 間の OB snapshot 差分から近似する。
**新規 API 呼び出し不要** — 既存の `calculate_imbalance()` が取得する OB snapshot の前回値を保存して差分を計算。

### 数式

$$\text{OFI} = \frac{\Delta V_{\text{bid}} - \Delta V_{\text{ask}}}{|\Delta V_{\text{bid}}| + |\Delta V_{\text{ask}}|}$$

- $\Delta V_{\text{bid}} = \sum_{i=1}^{5} V_{\text{bid},i}^{(t)} - V_{\text{bid},i}^{(t-1)}$
- $\Delta V_{\text{ask}} = \sum_{i=1}^{5} V_{\text{ask},i}^{(t)} - V_{\text{ask},i}^{(t-1)}$
- 結果: $[-1, +1]$。+1 = 強い買い圧力, -1 = 強い売り圧力

### 実装

| ファイル | 変更内容 |
|----------|----------|
| `ztb/trading/pricing/ofi_lite.py` | **新規**: 純粋関数 `compute_ofi_lite()` |
| `scripts/v460/lib/maker_price.py` | `_prev_ob_snapshot` / `_last_ofi_lite` スロット追加。`calculate_imbalance()` で前回 snapshot 保存 + OFI 計算。`offset_stages["ofi_lite"]` に記録 |

### 利用可能データ

- `last_ofi_lite` プロパティで最新値を取得可能
- `offset_stages["ofi_lite"]` でテレメトリに記録（0 以外の場合のみ）

---

## §3 Phase 3-2: Toxicity Budget 独立化

### 設計思想

`sell_dynamic_kill.py` の `assess_toxicity()` メソッド内にあった 4 段階判定ロジック (Glosten-Milgrom 1985) を、
kill メカニズムから独立した純粋関数として抽出。
将来の kill 撤去時にも Toxicity graduated response を保存できるようにする。

### 4 段階応答

| Level | Score 範囲 | Offset Mult | Participation |
|-------|-----------|-------------|---------------|
| GREEN | `< warn_level` | 1.0× | 100% |
| YELLOW | `[warn, caution)` | 1.0× → 2.0× (線形補間) | 100% |
| ORANGE | `[caution, 1.0)` | 2.0× → 3.0× + 参加率低下 | 100% → 33% |
| KILL | `≥ 1.0` | 3.0× | 0% |

### 実装

| ファイル | 変更内容 |
|----------|----------|
| `ztb/risk/toxicity_budget.py` | **新規**: 純粋関数 `assess_toxicity_score()` |
| `ztb/risk/sell_dynamic_kill.py` | `assess_toxicity()` を純粋関数に委譲（遅延 import で循環回避） |

### 循環 import 回避

```
toxicity_budget.py → imports ToxicityAssessment, ToxicityLevel from sell_dynamic_kill.py
sell_dynamic_kill.py → lazy import assess_toxicity_score (メソッド内import)
```

---

## §4 Phase 3-3: A-S Reference Spread (δ*) 計測

### 設計思想

Avellaneda-Stoikov (2008) の最適スプレッド下限 δ* を計算済みだが、
値が debug ログにしか出力されず観測不可だった。
offset_stages dict に記録することで、live テレメトリから A-S 理論値と実測値の乖離を観測可能にする。

### 数式

$$\delta^* = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$

- $\gamma$: リスク回避度
- $\sigma$: 価格ボラティリティ
- $\tau$: 残存時間
- $\kappa$: 流動性パラメータ

### 実装

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/maker_microstructure.py` | `_last_as_delta_star_ratio` 型スタブ + `_apply_as_reservation_shift()` 内でキャッシュ |
| `scripts/v460/lib/maker_price.py` | `_last_as_delta_star_ratio` スロット + `offset_stages["as_delta_star"]` 記録 |

---

## §5 テスト結果

- 既存テスト 3806 passed, 9 skipped (pre-existing failures: test_143, test_260, test_336 を除外)
- 新規モジュール import 検証: OK
- 循環 import: 遅延 import で解消済み

---

## §6 次ステップ

- Phase 4-1: OFI-Lite を signal として pipeline に接続（spread_adapt の入力として）
- Phase 4-2: A-S δ* と実測 offset の乖離度を定量評価
- Phase 4-3: Toxicity Budget を sell_dynamic_kill から完全分離（kill binary 撤去の前段階）
