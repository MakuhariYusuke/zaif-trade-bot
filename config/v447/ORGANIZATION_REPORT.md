# V447設定ファイル整理完了レポート

## 📋 実施内容

### 1. 全設定ファイルの検証
- **対象**: config/v447/ 内の全JSONファイル (10件)
- **結果**: 全てのファイルが正常に検証完了 ✅
- **修正**: `sac_v447_1m_multiframe_balance_aggressive.json` のカンマ欠落を修正

### 2. 標準設定ファイルの決定
**推奨ベースライン**: `config/v447/sac_v447_1m_multiframe_config.json`

**選定理由**:
- ファイル名が最もシンプルで識別しやすい
- 既存README.mdで標準として文書化済み
- バランスの取れたパラメータ設定
  - LR: 0.0001 (標準)
  - ent_coef: 0.01 (標準)
  - balance_penalty: 5.0 (保守的)
  - entropy_regularization: 0.02

### 3. ドキュメント整備
以下のドキュメントを作成・更新:

1. **`config/v447/README.md`** (完全リライト)
   - 全設定ファイルの一覧表
   - 使用例とCLIオーバーライド説明
   - reward_components永続化の説明

2. **`config/v447/USAGE_GUIDE.md`** (新規作成)
   - 設定ファイル選択ガイド
   - ABテスト推奨アプローチ
   - 次のステップ

3. **`tools/analyze_v447_configs.py`** (新規作成)
   - JSON妥当性検証
   - パラメータ比較表示
   - 推奨設定の提示

### 4. 検証ツール作成
- `tests/validate_v447_base.py`: ベース設定の動作確認
- `tools/analyze_v447_configs.py`: 全設定ファイルの分析

## 📊 設定ファイル一覧

| No | ファイル名 | LR | EntCoef | BalPen | EntReg | 用途 |
|----|-----------|-----|---------|--------|--------|------|
| 1 | **sac_v447_1m_multiframe_config.json** | 0.0001 | 0.01 | 5.0 | 0.02 | **標準ベースライン** |
| 2 | sac_v447_1m_multiframe_balance_adjusted.json | 0.0001 | 0.01 | 1.5 | 0.03 | 中程度ペナルティ |
| 3 | sac_v447_1m_multiframe_balance_aggressive.json | 0.0001 | 0.02 | 2.5 | 0.04 | 強めペナルティ |
| 4 | sac_v447_1m_multiframe_balance_shaping.json | 0.0001 | 0.01 | 1.0 | 0.02 | balance_shaping重視 |
| 5 | sac_v447_1m_multiframe_small_balance_penalty.json | 0.0001 | 0.01 | 0.2 | 0.03 | 弱いペナルティ |
| 6 | sac_v447_1m_multiframe_entropy_lr_lower.json | 0.00005 | 0.02 | 1.5 | 0.05 | 低LR+高エントロピー |
| 7 | sac_v447_1m_multiframe_entropy_lr_lower_skew_small.json | 0.00005 | 0.02 | 1.5 | 0.05 | +skewness調整 |
| 8 | sac_v447_1m_multiframe_combined_entropy_lr_balance.json | 0.00005 | 0.02 | 0.2 | 0.05 | 低LR+弱ペナルティ |
| 9 | sac_v447_1m_multiframe_skewness_penalty.json | 0.00005 | 0.02 | 0.2 | 0.05 | skewness重視 |
| 10 | sac_v447_1m_multiframe_skip_feature_filter.json | 0.0001 | 0.01 | 1.5 | 0.03 | 特徴量フィルタスキップ |

## 🚀 推奨使用方法

### ABテスト + reward_components分析
```bash
python tools/ab_param_search.py \
  --template config/v447/sac_v447_1m_multiframe_config.json \
  --grid config/ab/ab_grid_fine_tuning.json \
  --timesteps 5000 \
  --seeds 3 \
  --fast-mode
```

### 単独実行 (CLIオーバーライド活用)
```bash
python ztb/training/unified_trainer/main.py \
  --config config/v447/sac_v447_1m_multiframe_config.json \
  --timesteps 10000 \
  --seed 42 \
  --log-level INFO
```

## 🔑 重要ポイント

### CLI引数の優先度
unified_trainerは**CLI引数を常に優先**します:
- `--timesteps` → 設定ファイルの`total_timesteps`を上書き
- `--seed` → ランダムシードを固定
- `--log-level` → ログレベルを変更
- `--fast-mode` → feature_set='minimal'に変更

### reward_components永続化
全ての設定ファイルで以下が自動記録されます:
- balance_penalty
- skew_penalty
- balance_shaping
- entropy_shaping
- action_bonus
- final_reward

→ `reports/training_report_*.json` の`reward_components`セクションに保存

## 📝 次のアクション

1. **ベースライン実行**: 標準設定で動作確認
   ```bash
   python ztb/training/unified_trainer/main.py \
     --config config/v447/sac_v447_1m_multiframe_config.json \
     --timesteps 3000
   ```

2. **reward_components確認**: 永続化が正常動作していることを確認
   ```bash
   # 最新レポートのreward_components表示
   python -c "import json; from pathlib import Path; report = max(Path('reports').glob('training_report_*.json'), key=lambda p: p.stat().st_mtime); data = json.load(open(report)); print('reward_components' in data)"
   ```

3. **ABグリッド探索**: 体系的なパラメータ探索
   ```bash
   python tools/run_ab_searches.py \
     --template config/v447/sac_v447_1m_multiframe_config.json \
     --timesteps 5000 \
     --seeds 3 \
     --grids config/ab/ab_grid_fine_tuning.json \
     --fast-mode
   ```

## ✅ 検証済み

- [x] 全JSONファイルの構文検証
- [x] ベース設定ファイルの読み込み確認
- [x] パラメータ比較表の作成
- [x] ドキュメント整備
- [x] 検証ツールの作成

---

**整理完了日**: 2025-11-20  
**対象バージョン**: v447.1  
**設定ファイル総数**: 10件 (全て正常)
