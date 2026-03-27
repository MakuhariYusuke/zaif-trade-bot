# 645# P0: 退化 sell モデル無効化 + degenerate 検出ガード

## 背景

ボット残高が 2mBTC 以下に低下。14日間で -529 bps (1,751 fills)。
根本原因の調査を実施。

## 発見: 退化 (degenerate) sell モデル

### 症状
- `skip_gate_score` = **-0.3016** の fill が **193件** (全 sell の 24%)
- 全193件が **完全同一のスコア** `-0.3015559311057293`
- `threshold_used == score` → adaptive threshold がこの値に収束 → **判別能力ゼロ**
- regime 分布: 173件 ranging, 16件 trending_up, 4件 trending_down

### インパクト
| セグメント | 件数 | 平均PnL | 勝率 | 損失寄与 |
|-----------|------|---------|------|---------|
| 退化 sell | 193 | **-0.685** | 45.6% | **-132 bps (全体の24.9%)** |
| 正常 sell | 620 | -0.269 | 49.8% | -166 bps |
| 全体 | 1,752 | -0.304 | 49.2% | -532 bps |

### 原因
- sell モデル `skip_gate_lgbm_pnl120_sell.pkl` は **2026/2/24** 作成 (1ヶ月以上古い)
- 学習データ **n=229** に対し **300本のツリー** → 重度の過学習
- 入力空間の大部分で同一リーフに落ち、定数値を出力
- adaptive threshold がその定数値に収束し、全件パスを許可

### 対比: unified モデル
| 指標 | sell model | unified model |
|------|-----------|---------------|
| 更新日 | 2/24 | **3/27** |
| 平均PnL | -0.367 | **-0.001** |
| 勝率 | 47.8% | **57.6%** |
| Score-PnL相関 | -0.004 | **+0.102** |

## 修正内容

### P0: YAML — sell side model 無効化
- `model_path_sell: null` (旧: `models/v460/skip_gate_lgbm_pnl120_sell.pkl`)
- `model_path_sell_short: null` (旧: `models/v460/skip_gate_lgbm_pnl30_sell.pkl`)
- → unified model (3/27更新, 正の相関) にフォールバック

### P1: degenerate model 検出ガード
- `_check_model_degeneracy()` を `SkipGateModelLoaderMixin` に追加
- 12パターンの合成入力で予測を実行
- 40% 超の予測が同一値 → **退化と判定、ロード拒否**
- 初回ロード・hot-reload 双方に適用
- Pipeline なし (テスト/モック) はスキップ

### 影響ファイル
- `configs/v460/fill_test.yaml` — model_path_sell / model_path_sell_short を null 化
- `scripts/v460/lib/skip_gate_model_loader.py` — degeneracy check 追加
- `tests/unit/v460/test_645_degeneracy_check.py` — 5テスト新規
- `tests/unit/v460/test_336_yaml_code_drift_prevention.py` — allowlist 更新

## テスト結果
- v460 unit: **4145 passed**, 5 skipped, 0 failed
- 645# degeneracy: **5/5 passed**
