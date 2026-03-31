# 676# SAC Sidecar P0 修正 — confidence 復活 + deploy gate 品質強化

## 概要
675# で診断した「SAC sidecar 実質機能停止」の P0 修正を実施。binding constraint が dead_zone ではなく confidence であることを深掘りで確認し、4つの config 変更と 1つのコード変更を適用。

## 675# 深掘り: 問題の正確な因果関係

```
SAC training:
  OOS gross_roi ≈ 0.031% (mark-to-market do-nothing ROI)
  env_metrics.total_trades = 0 (SAC は HOLD に収束)
  ↓
confidence 計算:
  confidence = (roi - gate) / (confidence_roi_full - gate)
            = (0.00031 - 0) / (0.005 - 0)
            = 0.063
  ↓
sidecar offset:
  magnitude = max_boost(0.20) × shaped(0.64) × confidence(0.063)
           ≈ 0.007 bps = 0.74 JPY < tick size
  → 実質ゼロ
```

**binding constraint は confidence, not dead_zone。**
bias=0.628 は dead_zone=0.10 を余裕で通過するが、confidence=0.063 により最終 offset が tick size 未満に潰れる。

## 変更内容

### Config 変更

#### 1. `configs/v460/fill_test.yaml`
| パラメータ | 変更前 | 変更後 | 根拠 |
|-----------|--------|--------|------|
| `sidecar.dead_zone` | 0.10 | 0.05 | 弱 signal も活用 (binding constraint ではないが余裕度を確保) |

#### 2. `configs/v460/experiments/g2_sac_train.yaml`
| パラメータ | 変更前 | 変更後 | 根拠 |
|-----------|--------|--------|------|
| `incremental_timesteps` | 15000 | 25000 | 15K/8064 rows = 1.86 epoch → 25K = 3.1 epoch。収束改善 |
| `min_trade_count` | 3 | 50 | OOS 2016 rows で 50 trades = 2.5% frequency が意味のある最低ライン |
| `confidence_roi_full` | 0.005 | 0.002 | 現 ROI≈0.031% で confidence 0.063→0.157。tick size 超えの最低ライン |
| `min_profit_factor` | (新設) | 0.8 | PF<0.8 は明らかに学習失敗 (PF≥1.0 で収支均衡) |

### コード変更

#### `scripts/v460/ml/sac_retrain_scheduler.py`
- `SACRetrainConfig` に `min_profit_factor: float = 0.0` 追加
- `from_yaml_dict()` で `min_profit_factor` のパース追加
- `__post_init__()` で値域バリデーション追加
- deploy gate に profit_factor チェック追加 (gross_roi → trade_count → **profit_factor** → deploy の順)
- 600# conditional neutral fallback パターンを踏襲

## 期待効果

### confidence の変化

| 指標 | 変更前 | 変更後 (config のみ) | 変更後 (SAC 改善後) |
|------|--------|---------------------|---------------------|
| confidence_roi_full | 0.005 | 0.002 | 0.002 |
| OOS ROI (現状) | 0.031% | 0.031% | ≥0.05% (目標) |
| confidence | 0.063 | 0.157 | ≥0.25 |
| sidecar offset | 0.007 bps | 0.018 bps | ≥0.03 bps |
| 実額 (BTC@10.6M) | 0.74 JPY | 1.9 JPY | ≥3.2 JPY |

Config のみで ~2.5x 改善だが、依然として微弱。SAC の学習品質改善 (timesteps 増加 + PF gate による feedback) との合わせ技で有意な水準を目指す。

### deploy gate の品質改善

| Gate 条件 | 変更前 | 変更後 | 効果 |
|-----------|--------|--------|------|
| gross_roi > 0 | ✅ (do-nothing で通過) | ✅ (変更なし) | - |
| trade_count > 3 | ✅ (1413 > 3) | trade_count > 50 | do-nothing モデルが reject される可能性 |
| profit_factor > 0.8 | (なし) | ✅ (新設) | 損失優位モデルの deploy 防止 |

**注意**: trade_count=0 (env_metrics.total_trades=0) のモデルでも、evaluate_model_oos() の `trades_count` は position_manager 経由の正当な値。eval_result["trade_count"] が 0 < 50 で reject される可能性が高い。SAC が HOLD に収束している限り deploy されなくなる = フェイルセーフ。

## retrain_scheduler 再起動要否

- `fill_test.yaml` の dead_zone: **hot-reload 対応** (次サイクルで反映)
- `g2_sac_train.yaml` の変更: **retrain_scheduler 要再起動** (config は起動時に読み込み)

## テスト結果
- 全テスト: 2225 passed, 126 skipped, 0 failed
- SAC retrain scheduler テスト: 52 passed
- Sidecar 関連テスト: 144 passed, 13 skipped

## 次のステップ
- retrain_scheduler を再起動して新 config 適用
- 次回 retrain で min_trade_count=50 / min_profit_factor=0.8 の gate 効果を観察
- deploy 失敗が続く場合は P1 施策 (simple_reward, γ=0.95, gradient_steps=2) へ移行
