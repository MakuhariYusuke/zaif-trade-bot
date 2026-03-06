# 297# F-4: ノンパラメトリック検定 + 多重比較補正の ab_judgment 統合

> **文書番号**: 297#
> **種別**: `impl` (実装)
> **作成日**: 2026-03-06
> **前提**: [297_ph2_rpt_f4_g2_pre_analysis.md](297_ph2_rpt_f4_g2_pre_analysis.md) §1.3 推奨方針

---

## §1 概要

gate_c3_comparison.py (v459) から Mann-Whitney U / Cliff's Delta / Holm-Bonferroni の
3関数を ab_judgment.py に cherry-pick し、`evaluate_ab_variant()` の統計検定パイプラインを強化。

**目的**: パラメトリック (Welch's t) + ノンパラメトリック (Mann-Whitney U) の二重検定で
A/B判定の頑健性を向上。Holm-Bonferroni で多重検定の偽陽性を制御。

## §2 変更ファイル

### scripts/v460/lib/ab_judgment.py

| 追加関数 | 行数 | 説明 |
|---------|------|------|
| `_norm_cdf(z)` | 14L | 標準正規分布 CDF (A&S 7.1.26 erfc 近似) |
| `_mann_whitney_u(x, y)` | 22L | Mann-Whitney U 検定 (O(n×m) 全ペア + 正規近似) |
| `_cliffs_delta(x, y)` | 24L | Cliff's Delta 効果量 + 4段階解釈 |
| `_holm_bonferroni(p_values, alpha)` | 17L | Holm-Bonferroni 多重比較補正 |

#### ABJudgmentResult 新フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `mann_whitney_p_value` | `float \| None` | Mann-Whitney U 検定 p値 |
| `cliffs_delta_value` | `float \| None` | Cliff's Delta 効果量 |
| `cliffs_delta_interpretation` | `str` | negligible / small / medium / large |
| `holm_significant` | `list[bool] \| None` | [Welch t, Mann-Whitney] Holm補正後の有意性 |

#### evaluate_ab_variant() 統合

既存の Welch's t + Cohen's d に加え、`len(v_pnl) >= 10 and len(c_pnl) >= 10` の条件下で
Mann-Whitney U + Cliff's Delta を追加実行。両 p値を Holm-Bonferroni で補正。

#### summary() 更新

```
[stat] Welch t: p=0.0234 (Holm ✓), Cohen's d=0.512
[stat] Mann-Whitney: p=0.0312 (Holm ✓), Cliff's δ=0.387 (medium)
```

### tests/unit/v460/test_160_ab_judgment.py

新規テストクラス (19 テスト):

| クラス | テスト数 | 検証内容 |
|--------|---------|---------|
| `TestNormCdf` | 2 | z=0→0.5, z=±1.96→0.975/0.025, extreme values |
| `TestMannWhitneyU` | 4 | 同一分布, 明確な差, 空入力, U統計量 |
| `TestCliffsDelta` | 4 | 大効果, 微小効果, 空入力, 対称性 |
| `TestHolmBonferroni` | 6 | 全有意, 全非有意, 部分有意, step-down 棄却, 空, 単一 |
| `TestF4Integration` | 3 | フィールド格納, サンプル不足時None, summary表示 |

## §3 gate_c3 元実装からの改善点

1. **`_norm_cdf` バグ修正**: 元実装は erfc→CDF 変換が不正確 (`exp(-z²/2)` を直接使用)。
   A&S 7.1.26 に忠実な `erfc(|z|/√2) → CDF` 変換に修正。
2. **numpy 依存排除**: `math.exp`, `math.sqrt` を使用し numpy 不要化 (`_norm_cdf`, `_holm_bonferroni`)。
3. **型安全**: `List[float]` → `np.ndarray` (既存コードと整合), 戻り値タプルに解釈文字列追加。

## §4 G-2 調査結果 (実装不要)

G-2 の CircuitBreaker / DrawdownController fill_loop 統合は **既に完了済み** であることが判明:

| 項目 | 統合場所 | 実装 |
|------|---------|------|
| CircuitBreaker (API障害遮断) | `fill_cycle_executor.py` L735-746, L1245, L1427 | `_circuit_breaker` (ztb CircuitBreaker) |
| DrawdownController (日次PnL制限) | `fill_loop_orchestrator.py` 18箇所 | `_daily_drawdown_guard` (DailyDrawdownGuard) |

事前分析 (rpt) の「❌ 未統合」は誤り。ztb/risk/drawdown_controller.py は SAC 訓練専用で、
fill_loop には独自の `DailyDrawdownGuard` (`scripts/v460/lib/daily_drawdown_guard.py`) が存在。

## §5 テスト結果

- **既存テスト**: 68 passed (回帰なし)
- **新規テスト**: 19 passed
- **合計**: 87 passed
- **全体 v460**: 3952 passed, 32 skipped
