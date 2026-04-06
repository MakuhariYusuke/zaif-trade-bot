# Codex Prompt CX1: skip_gate score 品質分析・二峰性検証

## 背景
skip_gate の score average が +0.33→-0.31 に崩壊した（705# 分析）。
706# は「selector として機能していない」と結論。707# は触れていない。
708# の検証で、score=0.74 の高スコア fill もあり「二峰性」が疑われる。

## タスク

### Phase 1: score 分布の定量分析

1. `results/v460/fill_test/fill_records_20260401.jsonl` から `20260406.jsonl` の全レコードから
   `skip_gate_score`, `skip_gate_reason`, `skip_gate_bypassed`, `skip_gate_forced_pass`,
   `skip_gate_skipped`, `side`, `regime`, `post_fill_30s_pnl`, `date_str` or `timestamp` を抽出

2. Pre (Apr 1-3) と Post (Apr 4-6) に分割

3. 各期間の score 分布:
   - ヒストグラム (20 bins, -2 to +2)
   - mean, std, skew, kurtosis
   - 二峰性検定 (Hartigan's dip test or visual bimodality coefficient)

4. score vs PnL 相関:
   - Pearson, Spearman
   - score quintile 別 PnL 平均

5. bypass/forced_pass の内訳:
   - bypass 時の score vs normal 時の score
   - forced_pass が発動する条件の特定

### Phase 2: 原因の分離

6. model drift vs feature shift:
   - skip_gate の入力特徴量（OBI, spread, regime, etc）の分布を pre/post で比較
   - 特徴量シフトで score 変化が説明可能か（permutation importance 的アプローチ）

7. skip_gate model の architecture/path を特定:
   - `configs/v460/fill_test.yaml` の `skip_gate.model_path` を確認
   - model が最後に retrain された日付を確認
   - 学習データの vintage vs 現在の市場との乖離

### Phase 3: 改善案

8. `bypass_mode=false` の影響シミュレーション:
   - Post fill records で score < threshold (0.1) の fill を除外した場合の counterfactual PnL

9. threshold 再調整の提案:
   - 現在 threshold=0.1 で概ね全通過
   - optimal threshold を grid search (0.0, 0.1, 0.2, ..., 0.8) で counterfactual 推定

## 成果物
- `analysis_results/708_skip_gate_quality.json` に結果を出力
- 改善案を `docs/v460/708_skip_gate_recommendations.md` にまとめ
- 必要なテストがあれば `tests/unit/v460/` に追加

## 制約
- `git commit --no-verify` を使用
- テストは `python -m pytest tests/unit/v460/ -x --tb=short` で確認
- YAML/コード変更は行わない（分析のみ）
