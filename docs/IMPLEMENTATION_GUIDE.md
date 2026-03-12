# SAC v444 アクションバイアス改善 - 実装ガイド

## 📋 準備完了事項

### ✅ 作成された設定ファイル

1. **config/sac_v444_3_balanced_penalty_scale_200.json**
   - balance_penalty: 200.0 (元: 1000.0)
   - buy_action_bonus: 10.0
   - sell_action_bonus: 5.0
   - hold_action_bonus: 2.0
   - total_timesteps: 3000

2. **config/sac_v444_4_balanced_penalty_scale_300.json**
   - balance_penalty: 300.0
   - buy_action_bonus: 15.0
   - sell_action_bonus: 10.0
   - hold_action_bonus: 3.0
   - total_timesteps: 3000

3. **config/sac_v444_5_balanced_penalty_scale_500.json**
   - balance_penalty: 500.0
   - buy_action_bonus: 20.0
   - sell_action_bonus: 15.0
   - hold_action_bonus: 5.0
   - total_timesteps: 3000

### ✅ 作成されたツール

1. **quick_train_v444_configurable.py**
   - 設定ファイルベースのtraining実行
   - ロギングと結果保存
   - エラーハンドリング

2. **quick_train_v444_multi_config.py**
   - 複数設定の自動テスト
   - 比較レポート生成

3. **analysis/parameter_tuning_analysis.py**
   - パラメータ効果の分析
   - 視覚化による比較
   - 推奨事項の提示

### ✅ ドキュメント

1. **docs/SAC_v444_DEBUG_GUIDE.md**
   - 詳細なデバッグガイド
   - テスト実行ガイド
   - トラブルシューティング

---

## 🚀 実装ステップ

### Phase 1: 最初のテスト（scale_200）

#### Step 1.1: 設定確認
```bash
python -c "import json; c=json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json')); print('Balance Penalty:', c['environment']['behavior_optimization']['balance_penalty'])"
```
期待される出力: `Balance Penalty: 200.0`

#### Step 1.2: Training実行
```bash
python quick_train_v444_configurable.py --config config/sac_v444_3_balanced_penalty_scale_200.json --verbose
```

**実行時間**: 約 10-30分（データサイズとマシン性能による）

**確認項目**:
- [ ] Training が正常に開始する
- [ ] Reward が表示される
- [ ] Model が保存される

#### Step 1.3: 結果分析
```bash
python -c "
import json
# Check if model was saved
from pathlib import Path
model_path = Path('models/sac_v444_3_final_model_scale_200')
print('✅ Model exists' if model_path.exists() else '❌ Model not found')
"
```

---

### Phase 2: 第2のテスト（scale_300）

#### Step 2.1: Training実行
```bash
python quick_train_v444_configurable.py --config config/sac_v444_4_balanced_penalty_scale_300.json --verbose
```

#### Step 2.2: 結果比較
テスト後に比較分析を実行:
```bash
python analysis/parameter_tuning_analysis.py
```

---

### Phase 3: 第3のテスト（scale_500）

#### Step 3.1: Training実行
```bash
python quick_train_v444_configurable.py --config config/sac_v444_5_balanced_penalty_scale_500.json --verbose
```

---

### Phase 4: 全体比較と最適化

#### Step 4.1: 複数設定の自動テスト
```bash
python quick_train_v444_multi_config.py --compare
```

このコマンドは:
1. すべての設定でtraining実行
2. 結果の比較レポート生成
3. 推奨設定を提示

#### Step 4.2: 結果の確認

結果は以下の場所に保存されます:
- `results/training_comparison_report_YYYYMMDD_HHMMSS.txt`
- `analysis/parameter_tuning_recommendations_YYYYMMDD_HHMMSS.txt`
- `analysis/parameter_tuning_analysis_YYYYMMDD_HHMMSS.png`

---

## 📊 期待される結果

### Mean Reward の改善

| Config | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| scale_200 | -9845 | -5000～-2000 | 50-80% ✅ |
| scale_300 | -9845 | -4000～-1500 | 60-85% ✅✅ |
| scale_500 | -9845 | -3000～-500 | 70-95% ✅✅✅ |

### Action Distribution の改善

| Action | Current | Target (scale_200+) |
|--------|---------|-------------------|
| BUY | 18.00% | 30-40% |
| SELL | 66.85% | 30-40% |
| HOLD | 15.15% | 20-30% |

### Continuous Action Distribution

| Metric | Current | Target |
|--------|---------|--------|
| Mean | -0.4968 | -0.1～0.1 |
| Skewness | 0.9268 (SELL bias) | < 0.3 (balanced) |
| Extreme Negative | 54.90% | < 30% |

---

## 🔍 監視すべきメトリクス

各training中に以下を監視してください:

### ログから確認する項目
1. **Reward推移**
   ```
   Step 1000: Mean Reward = -9000 (OK)
   Step 2000: Mean Reward = -7000 (Improving!)
   Step 3000: Mean Reward = -5000 (Target!)
   ```

2. **Loss値**
   ```
   Actor Loss が大きくスパイクしていないか
   Critic Loss が安定しているか
   ```

3. **Action Distribution**
   ```
   BUY Actions が増えているか
   SELL Actions が減っているか
   ```

### テンソルボードから確認

```bash
tensorboard --logdir=tensorboard/
```

確認する項目:
- `rollout/ep_rew_mean`: エピソード報酬
- `train/actor_loss`: アクタロス
- `train/critic_loss`: クリティックロス

---

## ✅ チェックリスト

### 実装前
- [ ] すべての設定ファイルが作成されているか確認
- [ ] quick_train_v444_configurable.py が実行可能か確認
- [ ] ディレクトリ構造が正しいか確認

### Phase 1 (scale_200) 実行後
- [ ] Training が正常に完了したか
- [ ] Model ファイルが保存されたか
- [ ] Reward が改善されたか確認

### Phase 2 (scale_300) 実行後
- [ ] Phase 1 との比較で、さらに改善しているか
- [ ] Action Distribution が期待値に近いか

### Phase 3 (scale_500) 実行後
- [ ] 3 つすべてを比較分析
- [ ] 最適な設定を選択

### 最終確認
- [ ] 最適設定で Backtest 実行
- [ ] 結果をドキュメント化

---

## 🐛 トラブルシューティング

### 問題: "Config file not found"
```bash
# 設定ファイルの存在確認
ls -la config/sac_v444_3_balanced_penalty_scale_200.json
```

### 問題: Import エラー
```bash
# 必要なパッケージをインストール
pip install stable-baselines3 pandas numpy
```

### 問題: CUDA メモリ不足
```bash
# CPUで実行
export CUDA_VISIBLE_DEVICES=""
python quick_train_v444_configurable.py --config config/sac_v444_3_balanced_penalty_scale_200.json
```

### 問題: Training が遅い
```bash
# バッチサイズを削減
# config内の batch_size を 256 → 128 に変更
```

---

## 📝 結果記録テンプレート

各テスト後に以下をドキュメント化してください:

```
## Test: scale_200

**実行日時**: 2025-11-05 15:30

**設定パラメータ**:
- balance_penalty: 200.0
- buy_action_bonus: 10.0
- sell_action_bonus: 5.0
- hold_action_bonus: 2.0
- total_timesteps: 3000

**結果**:
- Mean Reward: [結果値]
- BUY Ratio: [結果値]
- SELL Ratio: [結果値]
- HOLD Ratio: [結果値]

**分析**:
[観察事項]

**次のステップ**:
[推奨アクション]
```

---

## 🎯 次のステップ

### 短期（今日～明日）
1. [ ] scale_200 でテスト実行
2. [ ] scale_300 でテスト実行
3. [ ] 結果比較分析

### 中期（今週）
1. [ ] scale_500 でテスト実行
2. [ ] 最適設定を選択
3. [ ] Backtest 実行

### 長期（来週）
1. [ ] 選択設定で長時間training（10000+ steps）
2. [ ] Fine-tuning
3. [ ] Production deployment

---

## 📚 関連ファイル

**設定ファイル**:
- `config/sac_v444_3_balanced_penalty_scale_200.json`
- `config/sac_v444_4_balanced_penalty_scale_300.json`
- `config/sac_v444_5_balanced_penalty_scale_500.json`

**Trainingスクリプト**:
- `quick_train_v444_configurable.py`
- `quick_train_v444_multi_config.py`

**分析ツール**:
- `analysis/parameter_tuning_analysis.py`

**ドキュメント**:
- `docs/SAC_v444_DEBUG_GUIDE.md` (詳細版)

---

**最後に**: このプロセスは段階的です。急がず、各フェーズの結果を詳しく確認してから次に進んでください。データに基づいた判断を常に心がけてください。

成功を祈ります！ 🚀
