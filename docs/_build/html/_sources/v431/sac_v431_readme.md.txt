# SAC v431: Advanced Learning Framework

## Overview

SAC v431は、unified_analyzeを軸とした統合型高度学習フレームワークです。v430の報酬関数設計ミスを修正し、v428の値張り付き防止メカニズムと高度な学習手法を統合しています。

## Key Improvements (v430 → v431)

### 1. 報酬関数再設計
- **sell_penalty** → **sell_bonus**: 負のペナルティから正のボーナスへ
- **hold_penalty** → **hold_bonus**: HOLD行動の適切な報酬化
- **buy_bonus** 削除: 不要な非対称性を排除

### 2. v428技術統合
- **対称アクション閾値**: ±0.3333で値の張り付きを防止
- **アンサンブルシステム**: 5つの専門化モデル統合
- **ベイズ最適化**: 高度なハイパーパラメータチューニング

### 3. 高度学習手法
- **カリキュラム学習**: 4段階の段階的学習
- **マルチステージ学習**: 探索→活用→微調整の3段階
- **アンサンブル学習**: 多様な市場条件対応

### 4. Unified Analysis統合
- **自動レポート生成**: トレーニング・バックテスト・リスク分析
- **ベンチマーク比較**: 過去モデルとの性能比較
- **包括的評価**: 単一コマンドでの完全分析

## Configuration Structure

```json
{
  "version": "v431",
  "reward_function": {
    "sell_bonus": 0.352,      // 正のボーナス（v430のsell_penaltyを反転）
    "hold_bonus": 0.005,      // 正のボーナス（v430のhold_penaltyを反転）
    "trading_bonus": 0.004    // 取引ボーナス
  },
  "action_conversion": {
    "symmetric_thresholds": true,  // v428の値張り付き防止
    "action_threshold": 0.3333
  },
  "advanced_learning": {
    "curriculum_learning": { /* 4段階カリキュラム */ },
    "multi_stage_training": { /* 3段階マルチステージ */ },
    "ensemble_training": { /* 5モデルアンサンブル */ }
  },
  "unified_analysis_integration": {
    "automated_reporting": true,
    "benchmarking": true
  }
}
```

## Usage

### Training

```bash
# カリキュラム学習
python ztb/training/scripts/train_sac_v431_advanced.py --mode curriculum

# マルチステージ学習
python ztb/training/scripts/train_sac_v431_advanced.py --mode multi_stage

# アンサンブル学習
python ztb/training/scripts/train_sac_v431_advanced.py --mode ensemble

# 標準学習
python ztb/training/scripts/train_sac_v431_advanced.py --mode standard
```

### Analysis

```bash
# 包括的分析
python ztb/analysis/v431/sac_v431_comprehensive_analysis.py \
  --backtest_results results/sac_v431_backtest_results.json \
  --compare_models sac_v431_standard sac_v430_standard \
  --risk_assessment results/sac_v431_backtest_results.json \
  --ensemble_models models/sac_v431_ensemble_*.zip
```

### Unified Analysis Integration

```bash
# バックテスト分析
python ztb/analysis/unified_analyze.py comparative analyze_backtest \
  --results results/sac_v431_backtest_results.json \
  --output reports/sac_v431_backtest_analysis.md

# バージョン比較
python ztb/analysis/unified_analyze.py comparative versions \
  --versions v431 v430 v428 \
  --output reports/sac_v431_version_comparison.md
```

## Expected Improvements

### Performance Targets
- **取引数**: v430の0件 → 1,500+件 (実験結果に基づく)
- **勝率**: 安定した50%以上の実現
- **リスク管理**: 最大ドローダウン-60%以内に抑制
- **適応性**: 多様な市場条件での安定した性能

### Technical Benefits
- **学習効率**: カリキュラム学習による段階的スキル習得
- **堅牢性**: アンサンブルによるリスク分散
- **保守性**: unified_analyze統合による統一的評価
- **拡張性**: モジュール化された高度学習手法

## Development Roadmap

### Phase 1: Core Implementation ✅
- [x] 報酬関数再設計
- [x] v428技術統合
- [x] 高度学習手法実装
- [x] Unified Analysis統合

### Phase 2: Validation (Next)
- [ ] トレーニング実行テスト
- [ ] バックテスト検証
- [ ] パフォーマンス比較
- [ ] リスク評価

### Phase 3: Optimization (Future)
- [ ] ハイパーパラメータ再調整
- [ ] アンサンブル重み最適化
- [ ] リアルタイム適応機能
- [ ] 拡張分析機能

## Files Structure

```
configs/v431/
├── sac_v431_advanced.json          # メイン設定ファイル

ztb/training/scripts/
├── train_sac_v431_advanced.py      # 高度トレーニングスクリプト

ztb/analysis/v431/
├── sac_v431_comprehensive_analysis.py  # 包括的分析スクリプト

models/
├── sac_v431_*.zip                  # トレーニング済みモデル

results/
├── sac_v431_*.json                 # バックテスト結果

reports/
├── sac_v431_*.md                   # 分析レポート
```

## Dependencies

- Stable Baselines3
- PyTorch
- NumPy, Pandas
- unified_analyze framework
- ensemble_system components

## Notes

- v431はv430の報酬関数設計ミスを根本的に修正
- v428の成熟した技術を最大限活用
- unified_analyzeを中心とした統合アプローチ
- 段階的検証による安定した開発プロセス
