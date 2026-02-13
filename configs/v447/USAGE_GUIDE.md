# V447設定ファイル使用ガイド

## 検証結果サマリー

全10ファイルがJSON構文的に正常です✅

## 推奨標準設定

### 🎯 ベーステンプレート
**`config/v447/sac_v447_1m_multiframe_config.json`**

**推奨理由:**
1. ファイル名が最もシンプルで識別しやすい
2. balance_penalty=5.0 で比較的保守的
3. 標準的なLR(0.0001)とent_coef(0.01)
4. README.mdで既に標準として文書化済み

## ABテスト用推奨アプローチ

### ✅ 推奨: ベースライン + AB Grid
```bash
python tools/ab_param_search.py \
  --template config/v447/sac_v447_1m_multiframe_config.json \
  --grid config/ab/ab_grid_fine_tuning.json \
  --timesteps 5000 \
  --seeds 3 \
  --fast-mode
```

**メリット:**
- 1つのベースラインから系統的に探索
- reward_components永続化で詳細分析可能
- パラメータ組合せを柔軟に制御

### 代替: 特定バリアント直接比較
既に有望な設定が分かっている場合:

```bash
python tools/ab_test_runner.py \
  --configs config/v447/sac_v447_1m_multiframe_entropy_lr_lower.json \
            config/v447/sac_v447_1m_multiframe_balance_shaping.json \
  --seeds 3 \
  --jobs 1
```

## 設定ファイル選択ガイド

### 学習安定性を重視
→ `sac_v447_1m_multiframe_entropy_lr_lower.json`
- LR=0.00005 (低め)
- ent_coef=0.02 (高め)
- entropy_regularization=0.05

### バランス探索を重視  
→ `sac_v447_1m_multiframe_balance_shaping.json`
- balance_penalty=1.0 (中程度)
- balance_shaping重視設計

### 弱ペナルティで自由度高く
→ `sac_v447_1m_multiframe_small_balance_penalty.json`
- balance_penalty=0.2 (弱い)
- エージェントの自由度を最大化

### 標準的な設定で開始
→ `sac_v447_1m_multiframe_config.json` (推奨)
- balance_penalty=5.0 (保守的)
- 標準LR・エントロピー

## CLI引数オーバーライド

unified_trainerはCLI引数を**常に優先**します:

| 引数 | 設定ファイルのキー | 優先度 |
|-----|------------------|--------|
| `--timesteps` | `training.total_timesteps` | CLI > config |
| `--seed` | (なし) | CLI > random |
| `--log-level` | (なし) | CLI > WARNING |
| `--fast-mode` | `training.features.*` | CLI > config |

**実例:**
```bash
# 設定ファイル: total_timesteps=2000
# CLI: --timesteps 10000
# → 実際に実行されるのは 10000 steps
```

## 次のステップ

1. **ベースライン実行**
   ```bash
   python ztb/training/unified_trainer/main.py \
     --config config/v447/sac_v447_1m_multiframe_config.json \
     --timesteps 5000
   ```

2. **reward_components確認**
   ```bash
   python -c "import json; print(json.load(open('reports/training_report_*.json'))['reward_components'])"
   ```

3. **ABグリッド探索**
   ```bash
   python tools/run_ab_searches.py \
     --template config/v447/sac_v447_1m_multiframe_config.json \
     --timesteps 5000 \
     --seeds 3 \
     --grids config/ab/ab_grid_fine_tuning.json \
     --fast-mode
   ```

4. **結果分析**
   ```bash
   python tools/analyze_v447_configs.py
   ```

---

**作成日**: 2025-11-20  
**目的**: v447設定ファイル群の整理と使用方針の明確化
