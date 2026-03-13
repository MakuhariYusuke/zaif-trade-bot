# SAC v447 Configuration Files

このディレクトリには、SAC v447実験用の設定ファイルが含まれています。

## 🎯 推奨ベースライン

**`sac_v447_1m_multiframe_config.json`** をABテスト・パラメータサーチのベーステンプレートとして使用してください。

### 特徴
- 1分足データ + マルチタイムフレーム対応
- reward_components永続化対応
- CLI引数で柔軟にオーバーライド可能

## 📊 設定ファイル一覧

### ベースライン
| ファイル名 | LR | EntCoef | BalPen | EntReg | 用途 |
|-----------|-----|---------|--------|--------|------|
| **sac_v447_1m_multiframe_config.json** | 0.0001 | 0.01 | 5.0 | 0.02 | 標準ベースライン |

### バリエーション(balance_penalty調整)
| ファイル名 | LR | EntCoef | BalPen | EntReg | 特徴 |
|-----------|-----|---------|--------|--------|------|
| sac_v447_1m_multiframe_balance_adjusted.json | 0.0001 | 0.01 | 1.5 | 0.03 | 中程度のペナルティ |
| sac_v447_1m_multiframe_balance_aggressive.json | 0.0001 | 0.02 | 2.5 | 0.04 | 強めのペナルティ |
| sac_v447_1m_multiframe_balance_shaping.json | 0.0001 | 0.01 | 1.0 | 0.02 | balance_shaping重視 |
| sac_v447_1m_multiframe_small_balance_penalty.json | 0.0001 | 0.01 | 0.2 | 0.03 | 弱いペナルティ |

### バリエーション(学習率・エントロピー調整)
| ファイル名 | LR | EntCoef | BalPen | EntReg | 特徴 |
|-----------|-----|---------|--------|--------|------|
| sac_v447_1m_multiframe_entropy_lr_lower.json | 0.00005 | 0.02 | 1.5 | 0.05 | 低LR+高エントロピー |
| sac_v447_1m_multiframe_entropy_lr_lower_skew_small.json | 0.00005 | 0.02 | 1.5 | 0.05 | +skewness調整 |
| sac_v447_1m_multiframe_combined_entropy_lr_balance.json | 0.00005 | 0.02 | 0.2 | 0.05 | 低LR+弱ペナルティ |

### バリエーション(その他)
| ファイル名 | LR | EntCoef | BalPen | EntReg | 特徴 |
|-----------|-----|---------|--------|--------|------|
| sac_v447_1m_multiframe_skewness_penalty.json | 0.00005 | 0.02 | 0.2 | 0.05 | skewness重視 |
| sac_v447_1m_multiframe_skip_feature_filter.json | 0.0001 | 0.01 | 1.5 | 0.03 | 特徴量フィルタスキップ |

## 🚀 使用例

### Option A: ベースライン + CLIオーバーライド (推奨)
ABテストでパラメータを動的に変更:

```bash
python tools/ab_param_search.py \
  --template config/v447/sac_v447_1m_multiframe_config.json \
  --grid config/ab/ab_grid_fine_tuning.json \
  --timesteps 5000 \
  --seeds 3 \
  --fast-mode
```

### Option B: 特定バリアント直接比較
既存の設定ファイルを直接比較:

```bash
python tools/ab_test_runner.py \
  --configs config/v447/sac_v447_1m_multiframe_entropy_lr_lower.json \
            config/v447/sac_v447_1m_multiframe_balance_shaping.json \
  --seeds 3 \
  --jobs 1
```

### Option C: unified_trainerで単独実行
```bash
python ztb/training/unified_trainer/main.py \
  --config config/v447/sac_v447_1m_multiframe_config.json \
  --timesteps 10000 \
  --log-level INFO
```

## 📝 CLIオーバーライド対応パラメータ

unified_trainerはCLI引数を優先します:

| 引数 | 説明 | 例 |
|-----|------|-----|
| `--timesteps` / `-s` | 訓練ステップ数 | `-s 5000` |
| `--log-level` / `-l` | ログレベル | `-l WARNING` |
| `--seed` | 乱数シード | `--seed 42` |
| `--fast-mode` | 高速モード (feature_set=minimal) | `--fast-mode` |

## 🔬 reward_components永続化

全ての設定ファイルは以下のreward_componentsを自動的に記録します:

- `balance_penalty`: アクション分布の偏りペナルティ
- `skew_penalty`: 偏度(skewness)ペナルティ
- `balance_shaping`: バランス誘導報酬
- `entropy_shaping`: エントロピー正則化報酬
- `action_bonus`: アクション固有ボーナス
- `final_reward`: 最終報酬値

これらは`reports/training_report_*.json`の`reward_components`セクションに保存されます。

## 🛠️ 検証ツール

設定ファイルの妥当性検証:

```bash
python tools/analyze_v447_configs.py
```

## 📚 旧README内容

元のusage exampleは以下を参照:
```bash
python ztb\training\unified_trainer\main.py \
  --config config\v447\sac_v447_1m_multiframe_config.json \
  -s 2000 \
  -l WARNING
```

分析ツール:
```bash
# Action distribution分析
python tools\analysis\action_distribution_window.py --reports reports --start 1000 --end 2000

# タイムフレーム変換
python tools\data\convert_timeframe.py --input data\btc_jpy_1m_dataset.csv --targets 5m 15m 1h --outdir data

# データ検証
python tools\data\validate_dataset.py --path data\btc_jpy_1m_dataset.csv --resample-to 5m 15m
```

---

**最終更新**: 2025-11-20  
**バージョン**: v447.1
