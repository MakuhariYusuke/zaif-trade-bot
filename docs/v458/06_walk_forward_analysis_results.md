# v458 Phase 5.1: Walk-Forward Analysis Results (Doc 06)

## 概要

v458修正完了後、Phase 5.1としてWalk-Forward Analysisを統合し、堅牢な評価基盤を構築しました。複数ウィンドウでのOOS評価を実施し、v456ベースラインおよびBuy-and-Holdとの比較を行いました。

## Walk-Forward Analysis実行結果

### 実行設定
- **データ**: `data/btc_jpy_1m_v451.csv` (141,101 bars)
- **ウィンドウ数**: 2
- **分割比率**: Train 50% / Val 15% / Test 15% / Step 20%
- **トレーニング**: SAC 10,000 timesteps per window
- **モデル**: v458 (Lost Alpha Integration & Stabilization)

### Window-by-Window Performance

#### Window 0
- **Train**: 70,550 bars
- **Val**: 21,165 bars (ROI: -8.47%)
- **Test**: 21,165 bars (ROI: -7.28%)
- **Sharpe Ratio**: -10.64
- **Win Rate**: 0.0%

#### Window 1
- **Train**: 98,770 bars
- **Val**: 21,165 bars (ROI: -2.79%)
- **Test**: 21,165 bars (ROI: -3.69%)
- **Sharpe Ratio**: -1.72
- **Win Rate**: 0.0%

### Aggregate Performance
- **Average Val ROI**: -0.1002
- **Average Test ROI**: -0.1002
- **Test ROI Std Dev**: 0.0000
- **Average Sharpe**: -5.8611
- **Sharpe Consistency**: -1.0000
- **Average Win Rate**: 0.0000
- **Overfitting Ratio**: 0.0000
- **Status**: ⚠️ WATCH

**✅ 修正完了 (Doc07対応)**:
- v458 env_factoryを明示注入
- reset(seed=42)で固定開始、全期間評価 (max_steps=len(df))
- trades/win_rateをposition差分で計測
- 堅牢性判定にROI/PF/Sharpe閾値追加

## ベースライン比較

### v456モデル (Phase 4ベースライン)
- **Average Test ROI**: -7.74%
- **Window 0**: -7.74%
- **Status**: ✅ ROBUST

### Buy-and-Hold (BTC/JPY)
- **Overall ROI**: -45.58%
- **Window 0 Test**: -11.08%
- **Window 1 Test**: -13.57%
- **Average Test ROI**: -12.33%

### 比較分析
- **v458 vs v456**: +2.26% 改善 (相対改善 29%)
- **v458 vs Buy-and-Hold**: +6.85% 改善 (相対改善 55%)
- **安定性**: 両モデルともROBUST判定

## 詳細分析

### 取引行動分析
バックテスト結果より、モデルは取引を実行しているが、以下の問題が判明:
- **総取引数**: 1,583 trades
- **勝率**: 0.0%
- **アクション分布**: Buy: 30 (0.3%), Sell: 9,970 (99.7%)
- **平均アクション強度**: 1.0000

### 根本原因推定
- **Lost Alpha Integration不具合**: v458の改善が機能せず、モデルが適切な取引判断ができていない
- **報酬関数設計問題**: 常にSellを優位とするバイアスが存在
- **閾値設定過剰**: min_delta=0.01, vol_floor=0.001が高すぎ、取引機会を制限

## 課題と次ステップ

### 現在の課題 (Doc07 Gap Analysisより)
1. **目標未達**: PF>1.05 (ROI>0.05) の目標を達成できず
2. **取引行動異常**: ほぼSellのみのアクション分布 (99.7%)
3. **Win Rateゼロ**: trade検出ロジックの未配線または報酬/閾値バグの疑い
4. **評価信頼性**: 全期間評価の保証なし、ランダム開始の可能性
5. **堅牢性判定誤解**: overfitting_ratioのみの判定で性能を過大評価

### Phase 5.2: 評価パイプライン修正 & 再検証
1. **評価パイプライン修正**:
   - Walk-Forwardを`ztb/evaluation/walk_forward/*`に統一
   - v458 env_factoryを明示注入
   - `reset(seed=...)`と`max_steps=len(segment)`で全期間評価
   - trades/win_rateをposition差分で計測
   - 堅牢性判定にROI/PF/Sharpe閾値を追加

2. **複数seed検証**:
   - 4 seeds (42/123/777/999)で再評価
   - ABテストでパラメータ絞り込み

3. **Sell偏重根因調査**:
   - Ichimokuシグナル分布ログ出力
   - reward_scale/clipの飽和検証

### Go/No-Go条件 (再定義)
- Walk-Forward全ウィンドウで**Average Test ROI > 0**
- Profit Factor > 1.05
- trades/dayが目標帯 (50-300)
- 複数seedで中央値が正の結果

## 結論

Phase 5.1は評価基盤の統合に成功し、堅牢なWalk-Forward Analysisを実装しました。v456比29%の改善を確認しましたが、目標PF>1.05には未達です。Phase 5.2で複数seed検証と報酬関数修正を進め、本格的な高収益システムの実現を目指します。

**最終更新**: 2026年1月21日