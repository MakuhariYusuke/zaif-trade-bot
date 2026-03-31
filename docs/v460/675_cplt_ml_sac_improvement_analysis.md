prompts\codex_task_ppo_foundation.md
docs\v460\675_cplt_ml_sac_improvement_analysis.md
さて、テストについては引続きお願いしたいところですが、並行して別タスクをお願いしたいです。具体的にはPPOの復活支援ということでおねがいします。又適宜PPO関連でも重複解消出来るかどうかも確認するのと、その他デバッグもお願いします。ドキュメント発行時番号重複に注意して下さい。
# 675# ML/SAC 総合分析 — 現状の問題と改善ロードマップ

## 概要
674# 自己レビューで判明した「feature MI ≤ 3%」「parameter golf」問題を受け、ML/SAC レイヤー全体の構造的問題を分析。SAC sidecar が実質機能停止していることを確認し、改善ロードマップと v461 世代 PPO 統合の設計を策定。

## 診断結果

### SAC Sidecar の実質機能停止

retrain history（直近5回、2026-03-30〜31）:

| timestamp | gross_roi | trade_count | env_metrics.total_trades | env_metrics.net_pnl |
|-----------|-----------|-------------|--------------------------|---------------------|
| 03-30 20:15 | 0.0226% | 1439 | 0 | 0.0 |
| 03-30 21:41 | 0.0132% | 1511 | 0 | 0.0 |
| 03-31 02:08 | 0.0157% | 1507 | 0 | 0.0 |
| 03-31 07:09 | 0.0405% | 1361 | 0 | 0.0 |
| 03-31 15:01 | 0.0313% | 1413 | 0 | 0.0 |

**全 retrain で `total_trades: 0`, `net_pnl: 0.0`, `final_balance == initial_balance`**

- `trade_count` は OOS 期間の行数ベースの算出値で、環境内の実際の取引ではない
- SAC の continuous action が `continuous_to_discrete_threshold: 0.10` を超えない（HOLD 収束）
- `min_trade_count: 3` の deploy gate を通過するが、実質学習は空

ROI が微正なのは mark-to-market 方式で position=0 のまま価格ドリフトを受けている「do nothing ROI」。

### Deploy Gate の構造的欠陥

```
deploy条件: gross_roi > 0.0 AND trade_count > 3
 ↓
trade_count = OOS期間の行数 (≈2016) → 常に > 3
gross_roi = mark-to-market drift → trending期は常に > 0
 ↓
結果: 取引ゼロのモデルが必ず deploy される
```

### sidecar signal の実効性

直近 deploy されたモデル出力:
- `directional_bias: 0.628` (有意)
- `confidence: 0.063` (低い)
- fill_test での適用: `sidecar_offset_bps: -0.004bps` (無視できる微小値)

#### 676# 深掘り: binding constraint は dead_zone ではなく confidence

初期分析では「dead_zone=0.10 で offset が潰される」と推定したが、深掘りで訂正:

- `abs(bias)=0.628 > dead_zone=0.10` → dead_zone は通過する
- 実際の binding constraint は **confidence=0.063**
- `magnitude = max_boost(0.20) × shaped(normalized(0.628)) × confidence(0.063) ≈ 0.007 bps`
- BTC@10.6M JPY で 0.007bps = **0.74 JPY** — tick size 未満で実効ゼロ

confidence = (OOS_ROI - 0) / (confidence_roi_full - 0) = 0.031% / 0.5% = 0.063

**つまり SAC の OOS ROI が低すぎて confidence が潰れ、sidecar 全体が常時無効化。**

### 報酬関数の複雑性

| ファイル | 行数 | 問題 |
|---------|------|------|
| `reward_calculator.py` | 1,919行 | 12段 curriculum + shaping + signal + asymmetric scaling |
| `heavy_env/core.py` | 1,656行 | 環境本体も肥大 |

`use_simple_reward: False`（デフォルト）→ 12段カリキュラムが動作するが、SAC の credit assignment を破壊。v459 で `use_simple_reward: True` (PnL直結 + clip[-1,1]) が最良結果を出した。

### 特徴量の信号弱度

17 特徴量（FeatureRegistry）:
1. price_velocity, 2. micro_trend, 3. price_acceleration
4. volume_surge, 5. momentum_divergence, 6. tick_volume_ratio
7. order_flow_imbalance, 8. micro_volatility, 9. spread_pressure
10. momentum_burst, 11. liquidity_surge, 12. realized_volatility
13. parkinson_sigma, 14. vpin_proxy, 15. kyle_lambda_proxy
16. amihud_illiq, 17. ema_velocity_bps

observation_shape = [20] (17 特徴量 + 3 env_internal = position, portfolio_value_normalized, ...)

672# 分析: 全特徴量の mutual information ≤ 3% (PnL予測にほぼ無関係)。すべて OHLCV proxy であり、real 板データ（depth, trade flow）を含まない。

## 現行 SAC 訓練パラメータ

```yaml
# g2_sac_train.yaml (sac_retrain セクション)
sac_retrain:
  rolling_window_days: 7           # 直近7日の OHLCV
  incremental_timesteps: 15000     # warm-start 15K steps
  retrain_interval_sec: 7200       # 2h毎
  min_trade_count: 3               # Deploy Gate (構造的欠陥)

# SAC hyperparameters
sac_hyperparameters:
  gamma: 0.80                     # 短期割引（HOLD偏重を悪化）
  ent_coef: "auto"                # entropy auto-tuning
  learning_rate: 3.0e-4
  batch_size: 256
  buffer_size: 100000
  learning_starts: 1000
  gradient_steps: 1

# fill_test.yaml sidecar
sidecar:
  enabled: true
  max_boost_bps: 0.20
  dead_zone: 0.10                  # confidence < 0.10 で offset=0
  shaping: quadratic               # 弱signal抑制
```

## 改善提案

### P0: SAC 実効性の回復 (即時適用可能)

| # | 施策 | 根拠 | 懸念 |
|---|---|---|---|
| **P0-1** | `use_simple_reward: true` 追加 | 12段 curriculum は dead code 化。simple reward (PnL直結 + clip) が v459 最良結果 | curriculum 関連テストの影響確認要 |
| **P0-2** | sidecar `dead_zone: 0.10 → 0.05` | 現 confidence ~0.06 で dead_zone 0.10 だと offset ≈ 0 | offset noise 増加の可能性 |
| **P0-3** | `incremental_timesteps: 15K → 25K` | 15K / 8064 train rows = 1.86 epoch — 収束不足 | 計算時間 +70% (~15分 → ~25分) |
| **P0-4** | `min_trade_count: 3 → 50` (OOS gate) | trade_count の算出方法が行数ベースで gate が無意味化 | 過度に厳格化すると deploy ゼロ |
| **P0-5** | deploy gate に `env_metrics.total_trades > 0` 条件追加 | 取引ゼロモデルの deploy を防止 | 全モデルが reject される場合は threshold 調整要 |

### P1: 報酬構造の根本改善 (中期)

| # | 施策 | 根拠 |
|---|---|---|
| **P1-1** | `trade_pnl_mode` を "realized" に変更 | mark-to-market は position=0 でも drift で正の reward → HOLD 誘因 |
| **P1-2** | `γ: 0.80 → 0.95` | 短期割引 + 微小 step PnL → discount 過多で HOLD が最適解化 |
| **P1-3** | `gradient_steps: 1 → 2` | v459 E設定で gradient_steps=2 が最良。sample efficiency 向上 |
| **P1-4** | `hold_penalty_multiplier: 0.0 → 0.001` | 軽微な HOLD ペナルティで取引頻度を誘導 |

### P2: v461 世代 — マルチアルゴリズム基盤

| # | 施策 | 概要 |
|---|---|---|
| **P2-1** | **PPO sidecar の並行訓練** | SAC = offset continuous, PPO = side selection discrete |
| **P2-2** | **Ensemble gating** | SAC / PPO / SkipGate の出力を weighted merge |
| **P2-3** | **Real feature integration** | Coincheck websocket → real-time depth/trade → feature upgrade |

## PPO 統合の具体設計 (P2-1)

### 既存 PPO コードベース

PPO 関連コードは広範に存在する（120+ ファイル）。主要なもの:

**コア実装:**
- `ztb/training/algorithms/ppo/ppo_algorithm.py` — AlgorithmFactory 統合
- `ztb/training/unified_trainer/algorithms/ppo_trainer.py` — UnifiedTrainer PPO モジュール
- `ztb/training/core/ppo_trainer.py` — コア PPO trainer
- `ztb/training/custom_ppo.py` — カスタム PPO 実装
- `sb3_contrib/__init__.py` — MaskablePPO ラッパー

**テスト:**
- `tests/training/test_ppo_trainer.py`
- `tests/unit/algorithms/test_ppo_algorithm.py`
- `tests/unit/training/test_ppo_trainer.py`
- `tests/integration/test_custom_ppo_integration.py`

**設定:**
- `ztb/training/config/ppo_config.py` — PPO hyperparameter defaults
- `ztb/training/constants.py` — PPO 関連定数

**過去バージョンの実験:**
- `experiments/train_sac_v443_2_*.py` — v443 で PPO/SAC 比較実験
- `scripts/training/train_v445_*.py` — v445 で direct training
- `ztb/training/archive/ppo_trainer_old.py` — 旧 PPO trainer

**インフラ:**
- `ztb/utils/training_utils.py` — `load_model()` が PPO/MaskablePPO を自動検出
- `ztb/trading/environment/heavy_env/core.py` — 離散/連続両方の action space 切替
- `ztb/trading/environment/utils/config.py` — `use_continuous_actions: false` で離散3行動

### 提案アーキテクチャ

```
┌──────────────────────────────────────────────────┐
│                v461 Dual-Sidecar Architecture     │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────┐    ┌──────────────┐             │
│  │  SAC Sidecar │    │  PPO Sidecar │             │
│  │  (continuous)│    │  (discrete)  │             │
│  │  offset bias │    │  side signal │             │
│  └──────┬──────┘    └──────┬───────┘             │
│         │                   │                     │
│         ▼                   ▼                     │
│  ┌──────────────────────────────────┐            │
│  │      Ensemble Gating Layer       │            │
│  │  confidence-weighted merge       │            │
│  │  SAC offset × PPO side_prob      │            │
│  └──────────────┬───────────────────┘            │
│                 │                                 │
│                 ▼                                 │
│  ┌──────────────────────────────────┐            │
│  │    fill_test Execution Pipeline   │            │
│  │    cycle_gate_aggregator.py       │            │
│  └──────────────────────────────────┘            │
└──────────────────────────────────────────────────┘
```

**PPO sidecar の役割**: side selection の改善
- 現在 `side_selector.py` は rule-based (scoring function)
- PPO が 3-class 離散行動 (buy/sell/skip) を出力
- confidence threshold 付きで rule-based 結果をオーバーライド

**PPO が SAC より有効な根拠 (この問題設定):**
1. **離散行動**: BUY/SELL/SKIP は本質的に離散判断。SAC の連続出力 → threshold 離散化で情報損失
2. **On-policy**: 非定常市場では off-policy replay buffer が stale data を含む
3. **安定性**: SAC の entropy auto-tuning が HOLD 収束するのに対し、PPO の clipped objective はより安定した探索を維持

**リスク:**
- PPO は sample efficiency が低い → 50K steps では不足 (200K+ 必要)
- Dual-sidecar は複雑性増加、計算コスト 2x

### 実装ステップ (Codex タスクとして分離)

1. 既存 PPO コード (120+ files) の棚卸し・整理
2. PPO テストの整備（現状テストの動作確認と修正）
3. `ppo_sidecar_train.yaml` 設計 (discrete 3-action, realized PnL reward)
4. `ppo_retrain_scheduler.py` — SAC scheduler と同構造で作成
5. `ppo_sidecar_signal.json` — 出力形式設計
6. `cycle_gate_aggregator.py` への PPO signal reader 追加
7. A/B テスト基盤

## 推奨アクション順序

1. **P0-1〜P0-5**: SAC 実効性回復 (deploy gate 修正 + simple reward 有効化)
2. **P1-1〜P1-4**: 報酬構造の簡素化 (simple reward 確認後)
3. **P2-1**: PPO sidecar prototype (v461 として Codex にタスク委託)

## 関連ドキュメント
- 672# 深層分析 (MI ≤ 3%, α 分布)
- 674# 自己レビュー (pre-669 theory validation)
- 670# 振り返り starting point
