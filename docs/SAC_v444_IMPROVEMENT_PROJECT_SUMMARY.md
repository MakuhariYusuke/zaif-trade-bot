# SAC v444 アクションバイアス改善プロジェクト - 全体概要

**作成日**: 2025-11-05
**対象版**: SAC v444.3, v444.4, v444.5
**優先度**: 🔴 高

---

## 📌 プロジェクト概要

SAC v444モデルで確認された重大なアクションバイアス（SELL偏重）を系統的に改善するプロジェクト。

**現状の問題**:
- Mean Reward: -9845.19（エクストリーム負）
- BUY Action: 18.00% （期待値: 30-40%）
- SELL Action: 66.85% （期待値: 30-40%）
- 根本原因: balance_penalty_scale = 1000.0 が過度に大きい

---

## 🎯 改善目標

| 項目 | 現状 | 目標 | 優先度 |
|------|------|------|--------|
| Mean Reward | -9845 | > -5000 | 🔴 P1 |
| BUY Ratio | 18% | 30-40% | 🔴 P1 |
| SELL Ratio | 66.85% | 30-40% | 🔴 P1 |
| BUY/SELL差 | 48.85% | < 10% | 🔴 P1 |
| HOLD Ratio | 15.15% | 20-30% | 🟡 P2 |
| Continuous Mean | -0.4968 | ±0.1 | 🟡 P2 |

---

## 📦 納品物一覧

### 1️⃣ 設定ファイル（3個）

#### Phase 1: Minimal Penalty
**ファイル**: `config/sac_v444_3_balanced_penalty_scale_200.json`
- balance_penalty: **200.0** (75% 削減)
- buy_action_bonus: 10.0
- sell_action_bonus: 5.0
- hold_action_bonus: 2.0

#### Phase 2: Moderate Penalty
**ファイル**: `config/sac_v444_4_balanced_penalty_scale_300.json`
- balance_penalty: **300.0** (70% 削減)
- buy_action_bonus: 15.0
- sell_action_bonus: 10.0
- hold_action_bonus: 3.0

#### Phase 3: Higher Penalty
**ファイル**: `config/sac_v444_5_balanced_penalty_scale_500.json`
- balance_penalty: **500.0** (50% 削減)
- buy_action_bonus: 20.0
- sell_action_bonus: 15.0
- hold_action_bonus: 5.0

---

### 2️⃣ Trainingスクリプト（2個）

#### 単一設定用
**ファイル**: `quick_train_v444_configurable.py`

```bash
# 基本的な使用方法
python quick_train_v444_configurable.py \
  --config config/sac_v444_3_balanced_penalty_scale_200.json \
  --verbose

# 出力: モデル保存 + トレーニングログ
```

**特徴**:
- JSON設定ベースの柔軟な実行
- 詳細なロギング
- エラーハンドリング

#### 複数設定用
**ファイル**: `quick_train_v444_multi_config.py`

```bash
# すべての設定でテスト
python quick_train_v444_multi_config.py --compare

# 特定の設定のみテスト
python quick_train_v444_multi_config.py --config scale_200
```

**特徴**:
- 自動比較レポート生成
- 複数設定の並列テスト対応
- 結果の可視化

---

### 3️⃣ 分析ツール（1個）

**ファイル**: `analysis/parameter_tuning_analysis.py`

```bash
python analysis/parameter_tuning_analysis.py
```

**出力**:
1. パラメータ効果分析レポート
   - `analysis/parameter_tuning_recommendations_YYYYMMDD_HHMMSS.txt`

2. 比較可視化
   - `analysis/parameter_tuning_analysis_YYYYMMDD_HHMMSS.png`
   - グラフ内容:
     - Penalty Scale 比較
     - Action Bonuses 比較
     - 期待報酬範囲
     - ターゲット行動比率

---

### 4️⃣ ドキュメント（3個）

#### A. 詳細デバッグガイド
**ファイル**: `docs/SAC_v444_DEBUG_GUIDE.md`

内容:
- 問題分析（根本原因の特定）
- 改善戦略（段階的アプローチ）
- テスト実行ガイド
- トラブルシューティング

#### B. 実装ガイド
**ファイル**: `docs/IMPLEMENTATION_GUIDE.md`

内容:
- 準備完了事項の確認
- ステップバイステップ実行手順
- チェックリスト
- 結果記録テンプレート

#### C. 本ドキュメント
**ファイル**: `SAC_v444_IMPROVEMENT_PROJECT_SUMMARY.md`

---

## 🔧 技術的背景

### 根本原因分析

```
Balance Penalty の計算:
penalty = balance_penalty_scale * |buy_ratio - sell_ratio|
         = 1000.0 * |0.18 - 0.6685|
         = 1000.0 * 0.4885
         = 488.5

毎ステップ 488.5 の負のペナルティが 2000 ステップ適用:
総ペナルティ ≈ 977,000 損失

結果: Mean Reward -9845 (ほぼ完全に penalty で構成)
```

### 改善メカニズム

1. **Balance Penalty 削減**
   - scale_200: -97.7/step (79.9% 削減)
   - scale_300: -146.6/step (70% 削減)
   - scale_500: -244.3/step (50% 削減)

2. **Action Bonuses 増強**
   - 多様なアクション選択を動機付け
   - SELL bias に対抗

3. **期待される効果**
   - Reward: -5000～-500 （現在から大幅改善）
   - Action Distribution: より バランス化
   - Trading Performance: 安定化

---

## 📈 期待される進行状況

### Week 1: テスト実行
```
Day 1-2: scale_200 テスト
  ↓ 
  結果分析 → Mean Reward -5000～-2000？
  
Day 3-4: scale_300 テスト
  ↓
  比較分析 → scale_200 より改善？
  
Day 5-6: scale_500 テスト
  ↓
  全体比較 → 最適設定選択？
```

### Week 2: 詳細検証
```
Day 1-2: 選択設定で長時間training（10000+ steps）
Day 3-4: Backtest で trading performance 検証
Day 5-6: Fine-tuning & Production readiness
```

---

## 🚀 実行手順（クイックスタート）

### Step 1: 準備確認
```bash
# 設定ファイルが存在するか確認
ls -la config/sac_v444_*.json | grep "scale_"

# 出力:
# sac_v444_3_balanced_penalty_scale_200.json
# sac_v444_4_balanced_penalty_scale_300.json
# sac_v444_5_balanced_penalty_scale_500.json
```

### Step 2: Phase 1 テスト実行
```bash
python quick_train_v444_configurable.py \
  --config config/sac_v444_3_balanced_penalty_scale_200.json \
  --verbose
```

**実行時間**: 10-30分

**期待される出力**:
```
================================================================================
🚀 Starting training: SAC v444.3: Balance Penalty Optimization Phase 1...
================================================================================
[Training progress logs...]
Model saved to: models/sac_v444_3_final_model_scale_200
✅ Training completed!
```

### Step 3: 結果分析
```bash
# パラメータ分析
python analysis/parameter_tuning_analysis.py

# 出力:
# - analysis/parameter_tuning_recommendations_*.txt
# - analysis/parameter_tuning_analysis_*.png
```

### Step 4: 複数設定の比較（オプション）
```bash
# すべての設定をテスト
python quick_train_v444_multi_config.py --compare

# 出力:
# - results/training_comparison_report_*.txt
```

---

## 📊 成功指標

### Success Criteria (優先度順)

✅ **Priority 1: Mean Reward Improvement**
- 現在: -9845
- 目標: > -5000
- チェック方法: training ログから確認

✅ **Priority 2: BUY/SELL Balance**
- 現在: BUY 18%, SELL 66.85% (差: 48.85%)
- 目標: BUY 30-40%, SELL 30-40% (差: < 10%)
- チェック方法: action distribution 統計

✅ **Priority 3: HOLD Action Increase**
- 現在: 15.15%
- 目標: 20-30%
- チェック方法: action ratio

✅ **Priority 4: Training Stability**
- 目標: Loss がスパイクしない
- チェック方法: tensorboard ログ確認

---

## 🔍 監視メトリクス

### Training中に確認

```
[Step 1000] 
  • Mean Reward: -8500 (改善中?)
  • BUY Actions: 25% (増加中?)
  • SELL Actions: 60% (減少中?)

[Step 2000]
  • Mean Reward: -6000 (更に改善?)
  • BUY Actions: 35% (理想値接近?)
  • SELL Actions: 40% (バランス化?)

[Step 3000]
  • Mean Reward: -4000 (ターゲット達成?)
  • BUY Actions: 38% (成功!)
  • SELL Actions: 35% (成功!)
```

---

## ⚠️ リスク & 対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 改善がない | 🔴 高 | config 正しく読み込まれているか確認 |
| Training 不安定 | 🔴 高 | batch_size, learning_rate を削減 |
| 過度なペナルティ | 🟡 中 | bonus を増加、penalty を削減 |
| メモリ不足 | 🟡 中 | buffer_size を削減 |

---

## 💡 ベストプラクティス

1. **段階的テスト**
   - 1つのセットで十分なテストを実施
   - 性急に結論を出さない

2. **結果記録**
   - 各テストの結果を詳細に記録
   - パラメータとの関連性を分析

3. **バックアップ**
   - 重要なモデルを保存
   - 結果を version 管理

4. **比較分析**
   - 3つのセット全て実行してから最適化を決定
   - 単一の指標ではなく複合的に判断

---

## 📞 サポート

### 問題が発生した場合

1. **ログを確認**
   ```bash
   tail -f logs/quick_train_v444_configurable.log
   ```

2. **設定を検証**
   ```bash
   python -c "import json; print(json.dumps(json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json')), indent=2))"
   ```

3. **ドキュメント参照**
   - `docs/SAC_v444_DEBUG_GUIDE.md` (詳細)
   - `docs/IMPLEMENTATION_GUIDE.md` (実装)

---

## 📝 変更ログ

### v1.0 (2025-11-05)
- ✅ 3つの設定ファイル作成（scale 200, 300, 500）
- ✅ Configurable training スクリプト作成
- ✅ パラメータ分析ツール作成
- ✅ 詳細ドキュメント作成

---

## 🎓 参考資料

- **SAC アルゴリズム**: Soft Actor-Critic (entropy regularization ベース)
- **Balance Penalty**: Action diversity を強制するペナルティ
- **Action Bonuses**: 特定アクションの選択を動機付け

---

## 🏁 結論

このプロジェクトで作成された3つの設定ファイルとツールにより、**段階的かつ系統的に**アクションバイアスを改善できます。

**推奨実行順序**:
1. `quick_train_v444_configurable.py` で scale_200 テスト
2. 結果確認後、scale_300 テスト
3. 全体比較で最適設定を決定
4. 選択設定で詳細検証

**期待される成果**:
- Mean Reward: 50-80% 改善
- Action Distribution: バランス化
- Trading Performance: 向上

---

**最終更新**: 2025-11-05
**ステータス**: 🟢 Ready for Implementation
