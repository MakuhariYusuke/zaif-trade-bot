# 356a# ph3 計画: SAC 訓練 — vXXX 資産活用による加速戦略

> **目的**: ph3 G2-train Gate 突破に向けた SAC 4-seed 訓練の包括的実行計画  
> **前提**: 355# B1-B5 ブロッカー + 既存 vXXX 資産の棚卸し結果 (356b#)  
> **作成日**: 2026-03-09

---

## §1 エグゼクティブサマリ

ph3 の大義は **SAC 強化学習エージェントによる方向性 Alpha の獲得** である。  
ph2 で蓄積した maker 執行品質 (fill test) と組み合わせることで、  
「いつ・どちら側に注文を出すか」を学習済みエージェントが決定する体制へ移行する。

**G2-train Gate 条件** (gate_thresholds.yaml):

| チェック | 条件 | 閾値 |
|---|---|---|
| E1: positive_seed_ratio | gross ROI > 0 の seed ≥ 75% | 0.75 |
| E2: ic_seed_std | IC の seed 間 σ ≤ 0.03 | 0.03 |
| E3: convergence | 30K 以降 ROI 変動 ≤ 5% | 5.0% |
| E4: worst_seed_roi | worst seed ROI > −2% | −0.02 |

> 測定方法: **4 seed × 50K steps** → 結果集約 → G2 judgment

### 結論: 5 ブロッカー + 2 設計判断 = 推定 **4-7 日**

---

## §2 vXXX 資産棚卸し

### §2.1 SAC 訓練コード — 3 重実装の整理状態

| # | ファイル | LOC | 用途 | ph3 での利用 |
|---|---|---|---|---|
| **A** | `ztb/training/sac_trainer.py` | ~300 | スタンドアロン。`BaseTrainer` + `RegimeAdaptiveTrainerMixin` → `UnifiedTrainer` 委譲 | ❌ 不使用 |
| **B** | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | ~1900 | 巨大クラス。checkpoint/callback/regime adaptation 完備 | ⚠️ 設計判断次第 |
| **C** | `scripts/v460/lib/tasks/sac_train.py` | ~370 | v460 task。**SB3 `SAC` 直接使用**。Protocol ベースで軽量 | ✅ メインパス |

**063# で 246 行削除した整理は既に完了**。しかし A/B/C の 3 系統並存は **B5 ブロッカー** として残存。

**判断**: C (task_sac_train.py) を ph3 メインパスとし、B の checkpoint/callback 機能を必要に応じて部分取込する。  
理由: C は v460 experiment framework (run_experiment.py) と直結しており、G2 gate への結果伝搬パスが最短。

### §2.2 既存モデル (v459 era)

```
models/
├── e0_sac_scalping_basic_btcjpy_*.zip (3)    # 基本スキャルピング
├── e1_sac_scalping_enhanced_*.zip (3)         # 強化版
├── e2a_sac_scalping_regime_adaptive_*.zip (5) # レジーム適応
├── e2b_sac_scalping_regime_trend_*.zip (4)    # トレンド特化
└── sac_model.zip (1)                           # 汎用
```

- 全 16 ファイル × ~3.1 MB (SB3 format)
- **v459 の observation space** とは次元が異なるため **直接 transfer learning は不可**
- ただし **ハイパーパラメータ知見** は活用可能 (§4.2 参照)

### §2.3 ハイパーパラメータ知見 (v459 → v460)

| パラメータ | v459 値 | v460 base.yaml | SB3 デフォルト | 採用判断 |
|---|---|---|---|---|
| `gamma` | **0.80** | 0.99 | 0.99 | **0.80 採用** — 短期スキャルピング指向 (v451 由来の実績値) |
| `buffer_size` | **100,000** | — | 1,000,000 | **100,000 採用** — 50K steps では巨大バッファ不要 |
| `learning_starts` | **100** | — | 1,000 | **100 採用** — 早期学習開始で 50K ステップを有効活用 |
| `batch_size` | **256** | 256 | 256 | 256 維持 |
| `learning_rate` | **3e-4** | 0.0003 | 3e-4 | 3e-4 維持 |
| `tau` | **0.005** | 0.005 | 0.005 | 0.005 維持 |
| `ent_coef` | **"auto"** | — | "auto" | "auto" 維持 |
| `total_timesteps` | 10,000 (phase0) | **50,000** | — | **50,000 採用** (G2 gate 基準) |

### §2.4 特徴量体系 — 2 系統の断絶

#### 系統 A: v460 Microstructure Features (build_features.py)

```python
V460_FEATURES = [
    "bid_ask_spread",       # 板スプレッド
    "depth_imbalance",      # 板深度非対称
    "trade_flow_imbalance", # 約定フロー非対称
    "vwap_deviation",       # VWAP 乖離
    "trade_intensity",      # 約定密度
    "order_flow_toxicity",  # 注文フロー毒性
    "price_impact",         # 価格インパクト
    "micro_return_vol",     # ミクロリターン変動
    "bid_depth_slope",      # Bid 深度勾配
    "ask_depth_slope",      # Ask 深度勾配
]
```

- **10 特徴量**: OB (板情報) + Trades (約定履歴) ベース
- ph1 G1-info で統計的有意性検証済み
- **問題**: HeavyTradingEnv は OHLCV DataFrame を入力とする → **OB/Trades 系特徴量は直接利用不可**

#### 系統 B: FeatureRegistry (ztb/features/)

- **~191 特徴量**: `@register("name")` デコレータで登録
- すべて **OHLCV ベース** の技術指標 (price_velocity, micro_trend, volume_surge, etc.)
- HeavyTradingEnv は FeatureRegistry を **自動発見・適用** する
- v459 モデルはこの系統で訓練されている

**結論**: ph3 では **FeatureRegistry 系統を主力とし、v460 microstructure features は ph4 以降の拡張で統合**する。

### §2.5 HeavyTradingEnv — 特徴量取込メカニズム

```
EnvironmentConfig.feature_names が設定されている場合:
  → その特徴量だけを使用 (明示的モード)
  → DataFrame にないカラムがあれば ValueError

feature_names が None の場合:
  → DataFrame の全カラムから自動発見
  → exclude_by_default (ts, timestamp, exchange, pair, episode_id) を除外
  → FeatureSetConfig / curated_features_list / MTF / 相関削減を適用
```

**意味**: `feature_names` を明示的に設定すれば、任意のサブセットで訓練可能。  
これが B3 修正の核心。

### §2.6 Gate インフラ

| コンポーネント | 実装状態 | 場所 |
|---|---|---|
| G2 judgment 関数 | ✅ 完全実装 | `run_gate_check.py` L248-320 |
| G2 thresholds | ✅ 設定済み | `gate_thresholds.yaml` L55-80 |
| run_experiment.py G2 dispatch | ❌ **未実装** | `_evaluate_gate()` — G1 のみ、G2 は `"PENDING"` |
| g2_sac_train.yaml | ❌ **不在** | configs/v460/experiments/ |
| Multi-seed orchestrator | ❌ **不在** | 4 seed の連続実行 + 結果集約ロジック |

### §2.7 過去ドキュメントの知見活用

| # | 文書 | 活用ポイント |
|---|---|---|
| 015# | SAC 実装調査 & オンライン学習設計 | SAC 3 系統の初期マッピング、オンライン学習の設計方針 |
| 021# | コード重複 & リファクタリング分析 | SAC 重複箇所の特定 → 063# で解消済み |
| 063# | SAC 重複実装の整理 | 246 行削除。A/B/C 3 系統は残存するが主要重複は解消 |
| 111# | v456–v459 レガシー資産・教訓 | gamma=0.80 の根拠、FeatureRegistry の歴史 |
| 034# | エージェント行動空間・執行パラメータ制御の歴史的分析 | action_space_type="1d_position" の選定理由 |

---

## §3 ブロッカー解消計画

### B1: `g2_sac_train.yaml` 不在 (High / 0.5 日)

**課題**: run_experiment.py は `configs/v460/experiments/g2_sac_train.yaml` を参照するが、ファイルが存在しない。

**解決策**: G1 実験 YAML の構造を踏襲して作成。

```yaml
# configs/v460/experiments/g2_sac_train.yaml
_base: configs/v460/base.yaml
_gate: G2-train
_task: sac_train

data:
  ohlcv_path: data/v460_ohlcv_1m.parquet    # v460 OHLCV 1分足
  # v460_features_path は ph4 以降

features:
  selected:
    # FeatureRegistry 系統から SAC に適した特徴量を選定
    # (§4.1 で詳細設計)
    - price_velocity
    - micro_trend
    - price_acceleration
    - volume_surge
    - momentum_divergence
    - tick_volume_ratio
    - order_flow_imbalance
    - micro_volatility
    - spread_pressure
    - momentum_burst
    - liquidity_surge
    - realized_volatility

environment:
  initial_balance: 10000000.0      # 1000万円 (v459 踏襲)
  transaction_cost: 0.001          # 0.1% (Zaif Maker)
  max_position_size: 0.01          # 0.01 BTC
  action_space_type: "1d_position" # Continuous [-1, +1] (SAC 用)
  use_continuous_actions: true
  reward_settings:
    trend_guidance_enabled: false
    cooldown_steps: 5
    edge_threshold: 0.001
    hold_penalty: 0.0
    asymmetric_loss_weight: 1.0
  execution_model:
    costs:
      fee_rate: 0.001
      slippage_rate: 0.0005
      slippage_model: "fixed"
    risk:
      max_position: 0.01
      max_daily_loss_pct: 0.03
      consecutive_loss_limit: 5

sac_hyperparameters:
  gamma: 0.80                      # v459 短期指向 (v451 由来)
  ent_coef: "auto"
  learning_rate: 3.0e-4
  batch_size: 256
  buffer_size: 100000              # v459 踏襲
  learning_starts: 100             # v459 踏襲
  tau: 0.005
  train_freq: 1
  gradient_steps: 1

training:
  total_timesteps: 50000           # G2 gate 基準
  checkpoint_interval: 5000        # 10 チェックポイント
  eval_episodes: 10

seeds: [42, 123, 456, 789]        # G2 gate: 4-seed
```

**作業内容**:
1. 上記 YAML を `configs/v460/experiments/g2_sac_train.yaml` に作成
2. G1 YAML との構造整合性テスト

---

### B2: 特徴量次元体系の断絶 (High / 1-2 日)

**課題**: v460 microstructure 10 特徴量と FeatureRegistry ~191 特徴量の間にアダプターがない。

**決定**: ph3 では **FeatureRegistry 特徴量のみ使用**。理由:

1. HeavyTradingEnv が FeatureRegistry と直結 (自動発見メカニズム)
2. v459 モデルの訓練実績がある
3. v460 microstructure features は OB/Trades 入力が必要で、OHLCV ベースの HeavyTradingEnv とは incompatible
4. microstructure features の統合は ph4 の streaming_pipeline 拡張で対応

**特徴量選定方針**:

```
Phase 3 Feature Set (12-15 features, 全て FeatureRegistry 系):
├── 価格系 (4):  price_velocity, micro_trend, price_acceleration, momentum_burst
├── 出来高系 (3): volume_surge, tick_volume_ratio, liquidity_surge
├── 構造系 (3):   order_flow_imbalance, spread_pressure, micro_volatility
└── 統合系 (2-3): momentum_divergence, realized_volatility, [mean_reversion_signal]
```

**作業内容**:
1. FeatureRegistry から SAC 学習に適した特徴量サブセットを選定
2. `g2_sac_train.yaml` の `features.selected` に記載
3. 訓練データ (OHLCV parquet) にこれら特徴量が含まれることを確認

---

### B3: feature_columns の env 未注入 (Medium / 0.5 日)

**課題**: `task_sac_train.py` L178-215 で `feature_columns` を構築するが `EnvironmentConfig.feature_names` に設定していない。

**影響**: config で指定した特徴量が**無視**され、DataFrame の全カラムが自動発見される → observation space 次元が不定。

**修正案** (task_sac_train.py `_create_training_env`):

```python
# 修正前:
env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()

# 修正後:
env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()
if feature_columns:
    env_config.feature_names = feature_columns  # ★ 明示的注入
```

**作業内容**:
1. `_create_training_env()` 修正 — `feature_columns` → `env_config.feature_names`
2. `env_info["feature_columns_used"]` に実際に使用された特徴量を記録
3. 単体テスト追加: 指定した特徴量のみが observation space に反映されることを確認

---

### B4: Multi-seed ラッパー不在 (Medium / 1-2 日)

**課題**: run_experiment.py は `--seed` で**単一 seed のみ実行**。G2 gate は 4 seed の結果集約を要求する。

**G2 judgment が期待する入力フォーマット**:

```json
{
  "seed_results": [
    {"seed": 42,  "gross_roi": 0.05, "ic_mean": 0.03},
    {"seed": 123, "gross_roi": 0.03, "ic_mean": 0.02},
    {"seed": 456, "gross_roi": 0.04, "ic_mean": 0.025},
    {"seed": 789, "gross_roi": -0.01, "ic_mean": 0.01}
  ],
  "convergence": {
    "roi_variance_pct_after_30k": 3.5
  }
}
```

**task_sac_train.py の現在の出力フォーマット**:

```json
{
  "algorithm": "SAC",
  "seed": 42,
  "total_timesteps": 50000,
  "training_time_sec": 120.5,
  "model_path": "models/sac_v460_s42.zip",
  "env_info": {"obs_dim": 12, "action_dim": 1},
  "checkpoint_metrics": [...],
  "eval_metrics": {"gross_roi": ..., "ic_mean": ...}
}
```

**解決策**: 2 層アーキテクチャ

```
[run_multi_seed.py / run_experiment.py --multi-seed]
  ├── seed=42:  run_experiment.py → task_sac_train → result_42.json
  ├── seed=123: run_experiment.py → task_sac_train → result_123.json
  ├── seed=456: run_experiment.py → task_sac_train → result_456.json
  └── seed=789: run_experiment.py → task_sac_train → result_789.json
  └── _aggregate_seed_results() → g2_aggregated.json
      └── run_g2_judgment(g2_aggregated.json) → PASS/FAIL
```

**設計選択肢**:

| 案 | 概要 | Pro | Con |
|---|---|---|---|
| **A: run_experiment.py 拡張** | `seeds` リストを config から読み、for-loop で実行 | 既存フレームワーク活用 | run_experiment.py の肥大化 |
| **B: 外部ラッパースクリプト** | `run_g2_multi_seed.py` が run_experiment.py を 4 回呼出し | separation of concerns | 新規ファイル追加 |
| **C: task 内 multi-seed** | `task_sac_train` が 4 seed を内部ループ | 最小変更 | task の責務超過 |

**推奨: 案 A** — run_experiment.py に `_run_multi_seed()` を追加。理由:
- config の `seeds` フィールドは base.yaml に既に定義済み
- gate 評価パスも同一ファイル内で完結
- 新規ファイル追加を最小限に抑える (copilot-instructions 準拠)

**作業内容**:
1. run_experiment.py に `_run_multi_seed(config, seeds)` 関数追加
2. `task_sac_train` の出力に `gross_roi` / `ic_mean` フィールドを保証
3. `_aggregate_seed_results()` で G2 judgment フォーマットに変換
4. convergence 計算: checkpoint_metrics から 30K 以降の ROI 分散を算出
5. `_evaluate_gate()` に G2 ディスパッチ追加
6. 単体テスト: 4 seed 結果の集約・判定テスト

---

### B5: SACTrainer 3 重実装 (Low / 設計判断)

**課題**: A (ztb/training/sac_trainer.py), B (unified_trainer/algorithms/sac_trainer.py), C (tasks/sac_train.py) の 3 系統。

**判断**: ph3 では **C を使用**。A/B は deprecated 扱いとし、ph3 完了後に整理。

**根拠**:
- C は v460 experiment framework と直結
- C は SB3 SAC を直接使用しており、デバッグ・理解が容易
- B の 1900 行は過剰 (checkpoint, regime adaptation, callback が task_sac_train + run_experiment.py に分散すべき機能)
- A は B への薄いラッパーで独立した価値なし

**将来方針** (ph4+):
- B の有用機能 (checkpoint manager, structured logger) を独立モジュールとして抽出
- A/B のクラス本体は `archived/` に移動
- C をベースに refactored SACTrainer を構築

---

## §4 実装ロードマップ

### §4.1 フェーズ分割

```
Phase 3-A: 基盤整備 (2-3 日)
├── [P3A-1] g2_sac_train.yaml 作成                         ← B1
├── [P3A-2] task_sac_train.py feature_columns 修正         ← B3
├── [P3A-3] 訓練データ準備 (OHLCV parquet + 特徴量確認)    ← B2
└── [P3A-4] 単体テスト追加 (env feature + sac train)

Phase 3-B: Multi-seed & Gate 接続 (1-2 日)
├── [P3B-1] run_experiment.py multi-seed 実行               ← B4
├── [P3B-2] _aggregate_seed_results() 実装
├── [P3B-3] _evaluate_gate() G2 ディスパッチ追加
└── [P3B-4] convergence 計算 (checkpoint 30K 以降 ROI 分散)

Phase 3-C: 訓練実行 & Gate 判定 (1-2 日)
├── [P3C-1] 4-seed × 50K steps 訓練実行
├── [P3C-2] G2 gate 判定
├── [P3C-3] 結果分析・ドキュメント化
└── [P3C-4] FAIL 時の改善サイクル計画
```

### §4.2 ハイパーパラメータ戦略

**初回訓練 (baseline)**:
- v459 の実績値をベースに v460 向け調整
- gamma=0.80, buffer=100K, lr=3e-4 (v459 踏襲)
- 50K steps (G2 gate 基準)

**FAIL 時の改善方向**:

| E1 (positive_seed) FAIL | E2 (IC std) FAIL | E3 (convergence) FAIL | E4 (worst_seed) FAIL |
|---|---|---|---|
| 特徴量追加/変更 | lr 調整、seed 間ハイパラ固定 | learning_starts 増加、buffer 拡大 | gamma 上げ (保守化) |
| reward shaping | ent_coef 固定 (auto→0.1等) | gradient_steps 増加 | transaction_cost 精査 |
| データ拡張 | batch_size 増加 | early stopping 導入 | reward clipping |

### §4.3 特徴量候補の詳細根拠

| 特徴量 | FeatureRegistry 名 | 採用理由 |
|---|---|---|
| price_velocity | `price_velocity` | 瞬時価格変化率。トレンド検知の最基本指標 |
| micro_trend | `micro_trend` | 短期トレンド方向。SAC の方向性判断に直結 |
| price_acceleration | `price_acceleration` | トレンド加速/減速。エントリー/エグジットタイミング |
| volume_surge | `volume_surge` | Z-score ベース出来高急増。流動性イベント検知 |
| momentum_divergence | `momentum_divergence` | Fast/Slow 乖離。逆張りシグナル |
| tick_volume_ratio | `tick_volume_ratio` | 直近出来高比率。短期流動性変化 |
| order_flow_imbalance | `order_flow_imbalance` | 疑似注文フロー非対称。OHLCV からの推定板状況 |
| micro_volatility | `micro_volatility` | 短期ボラティリティ。リスク調整の入力 |
| spread_pressure | `spread_pressure` | スプレッド圧力。maker 収益性の proxy |
| momentum_burst | `momentum_burst` | 価格×出来高バースト。強いエントリーシグナル |
| liquidity_surge | `liquidity_surge` | 流動性急増。約定しやすさの変化検知 |
| realized_volatility | `realized_volatility` | 実現ボラティリティ。ポジションサイジング入力 |

---

## §5 run_experiment.py G2 対応設計

### §5.1 現在のフロー (G1 のみ)

```python
# run_experiment.py (現状)
def run(config_path, seed_override=None):
    cfg = load_config(config_path)
    task_fn = TASK_DISPATCH[cfg["_task"]]  # "sac_train" → task_sac_train
    results = task_fn(cfg)                  # 単一 seed 実行
    gate_result = _evaluate_gate(gate, results, cfg)  # G1 のみ
    # → gate_result は G2 の場合 "PENDING"
```

### §5.2 改修後のフロー (G2 対応)

```python
# run_experiment.py (改修案)
def run(config_path, seed_override=None):
    cfg = load_config(config_path)
    gate = cfg.get("_gate", "")
    
    if "G2" in gate:
        # Multi-seed 実行
        seeds = cfg.get("seeds", [42])
        results = _run_multi_seed(cfg, seeds)   # ★ 新規
    else:
        task_fn = TASK_DISPATCH[cfg["_task"]]
        results = task_fn(cfg)
    
    gate_result = _evaluate_gate(gate, results, cfg)
    # ...

def _run_multi_seed(cfg: dict, seeds: list[int]) -> dict:
    """4-seed 訓練実行 + 結果集約."""
    seed_results = []
    all_checkpoints = []
    
    for seed in seeds:
        seed_cfg = {**cfg, "training": {**cfg.get("training", {}), "seed": seed}}
        task_fn = TASK_DISPATCH[cfg["_task"]]
        result = task_fn(seed_cfg)
        
        seed_results.append({
            "seed": seed,
            "gross_roi": result["eval_metrics"]["gross_roi"],
            "ic_mean": result["eval_metrics"].get("ic_mean", 0.0),
        })
        all_checkpoints.append(result.get("checkpoint_metrics", []))
    
    # Convergence 計算: 30K 以降の ROI 分散
    convergence = _compute_convergence(all_checkpoints, window_start=30000)
    
    return {
        "seed_results": seed_results,
        "convergence": convergence,
        "raw_results": {s: r for s, r in zip(seeds, seed_results)},
    }

def _compute_convergence(all_checkpoints, window_start=30000):
    """30K step 以降の ROI 変動を算出."""
    roi_values = []
    for cp_list in all_checkpoints:
        for cp in cp_list:
            if cp.get("timestep", 0) >= window_start:
                roi_values.append(cp.get("roi", 0.0))
    
    if len(roi_values) < 2:
        return {"roi_variance_pct_after_30k": 0.0}
    
    roi_var_pct = (max(roi_values) - min(roi_values)) * 100
    return {"roi_variance_pct_after_30k": roi_var_pct}
```

### §5.3 `_evaluate_gate()` G2 ディスパッチ追加

```python
def _evaluate_gate(gate: str, results: dict, cfg: dict) -> str:
    if "G1" in gate:
        # ... 既存の G1 処理
    elif "G2" in gate:
        from scripts.v460.run_gate_check import run_g2_judgment
        # results は既に seed_results + convergence フォーマット
        judgment = run_g2_judgment_from_dict(results)  # ★ dict 版の judgment
        results["g2_judgment_cache"] = judgment
        return "PASS" if judgment["gate_result"] == "PASS" else "FAIL"
    return "PENDING"
```

---

## §6 task_sac_train.py 修正計画

### §6.1 B3 修正: feature_columns → env_config.feature_names

```diff
 def _create_training_env(df, cfg):
     env_cfg = section(cfg, "environment")
     feature_cfg = section(cfg, "features")
     selected_raw = feature_cfg.get("selected", [])
     feature_columns = [str(col) for col in selected_raw] if isinstance(selected_raw, list) else []
 
     env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()
+    if feature_columns:
+        env_config.feature_names = feature_columns
+        env_config.use_continuous_actions = True  # SAC 用
 
     env = HeavyTradingEnv(df=df, config=env_config)
```

### §6.2 出力スキーマ拡張

task_sac_train の `eval_metrics` に `gross_roi` と `ic_mean` を確実に含める:

```python
# _evaluate_trained_model 内
eval_metrics = {
    "gross_roi": total_pnl / initial_balance,  # ★ 必須
    "ic_mean": information_coefficient,          # ★ 必須
    "sharpe_ratio": sharpe,
    "max_drawdown": max_dd,
    "win_rate": wins / total_trades,
    "total_trades": total_trades,
}
```

### §6.3 Convergence メトリクス出力

checkpoint_metrics に ROI を含める:

```python
# _train_with_checkpoints 内
checkpoint_metrics.append({
    "timestep": current_step,
    "roi": episode_roi,           # ★ convergence 計算用
    "reward_mean": mean_reward,
    "episode_length": ep_len,
})
```

---

## §7 リスク評価

### §7.1 技術リスク

| リスク | 深刻度 | 影響 | 緩和策 |
|---|---|---|---|
| OHLCV データ品質不足 | Medium | 特徴量算出不可 | data/ 内の既存 parquet を事前検証 |
| FeatureRegistry 特徴量の env 互換性 | Medium | observation space 不整合 | B3 修正 + 単体テスト |
| 50K steps での学習不足 | Low | G2 E1/E4 FAIL | v459 では 10K で phase0 通過実績あり |
| checkpoint_metrics 形式不統合 | Medium | convergence 計算不可 | §6.3 の出力拡張 |

### §7.2 スケジュールリスク

| リスク | 影響 | 緩和策 |
|---|---|---|
| fill test 中に ph3 作業が bot に影響 | 低 — 訓練コードは bot と独立 | 別プロセスで実行 |
| G2 FAIL → 改善サイクル | +2-5 日 | ハイパラ / 特徴量 / データの 3 軸で iterative 改善 |
| FeatureRegistry 特徴量の選定ミス | G2 FAIL | SHAP 分析 (164# 手法) で事後検証 + 差替え |

### §7.3 設計リスク

| リスク | 影響 | 対策 |
|---|---|---|
| v459 gamma=0.80 が v460 データに不適 | E4 FAIL | 0.90, 0.95 のグリッドサーチを FAIL 時に実施 |
| FeatureRegistry 特徴量のみでは alpha 不足 | 全 E FAIL | ph4 microstructure 統合を前倒し |
| multi-seed の実行時間過大 | スケジュール遅延 | 4 seed × 50K = 20 万 steps — v459 では ~10 分/seed |

---

## §8 依存関係と前提条件

### §8.1 ph2 fill test との関係

- ph3 訓練コードは **fill test と完全独立** (別プロセス)
- fill test 中の bot は `git_sha=819ec73b2081` で固定
- ph3 の code changes は bot 再起動まで影響しない

### §8.2 データ要件

| データ | 必要性 | 現状 |
|---|---|---|
| OHLCV 1m parquet | 必須 | `data/` 配下に v459/v460 データが存在する前提 |
| FeatureRegistry 特徴量 | 必須 — env が自動計算 | DataFrame に OHLCV カラムがあれば FeatureRegistry が算出 |
| OB/Trades データ | 不要 (ph3 scope 外) | ph4 で streaming_pipeline 経由で統合 |

### §8.3 テスト要件

- 全 v460 テスト (4194 件) が引き続き PASS すること
- 新規テスト: feature injection + multi-seed + G2 judgment

---

## §9 成功基準

| 基準 | 定義 |
|---|---|
| **ph3 完了** | G2-train Gate PASS (E1-E4 全合格) |
| **中間マイルストーン 1** | g2_sac_train.yaml + B3 修正完了、単一 seed で訓練実行可能 |
| **中間マイルストーン 2** | 4-seed 訓練完了、結果 JSON 集約完了 |
| **後続判断** | G2 PASS → ph4 (オンライン学習設計) へ移行 |
| **G2 FAIL 時** | §4.2 の改善方向に従い最大 3 イテレーション |

---

## §10 クリティカルパスと工数見積もり

```
Day 1:  [P3A-1] g2_sac_train.yaml 作成
        [P3A-2] task_sac_train.py B3 修正
        [P3A-3] 訓練データ確認 + 特徴量選定 finalize

Day 2:  [P3A-4] 単体テスト追加
        [P3B-1] run_experiment.py multi-seed
        [P3B-2] _aggregate_seed_results()

Day 3:  [P3B-3] _evaluate_gate() G2 dispatch
        [P3B-4] convergence 計算
        [P3B-?] 統合テスト

Day 4:  [P3C-1] 4-seed × 50K 訓練実行 (~40-60 分)

Day 5:  [P3C-2] G2 gate 判定
        [P3C-3] 結果分析・ドキュメント化
        [P3C-4] (FAIL 時) 改善計画策定

───────────────────────────────────────
Best case:  4 日 (G2 PASS)
Expected:   5-6 日 (1 回の改善サイクル)
Worst case: 7-10 日 (3 回の改善サイクル)
```

---

## §11 自己批判的レビュー

### §11.1 この計画の弱点

1. **FeatureRegistry 特徴量のみで alpha 取れるか不明** — v460 の本来の強みは microstructure features (OB/Trades)。OHLCV ベースの技術指標だけでは v459 と大差ない可能性がある。
   - **対策**: 初回 G2 FAIL 時に microstructure features 統合の ph4 前倒しを検討

2. **gamma=0.80 の根拠が v451** — 3 メジャーバージョン前の知見。v460 の取引環境 (Coincheck BTC/JPY) に適合しない可能性。
   - **対策**: baseline 実行後、gamma=[0.80, 0.90, 0.95, 0.99] の 4 値グリッドサーチ

3. **IC (Information Coefficient) の算出方法が未定義** — G2 の E2 check で IC を要求しているが、task_sac_train.py の eval_metrics に IC 算出ロジックがない。
   - **対策**: checkpoint_metrics の予測/実績相関として Spearman ρ を算出する実装が必要

4. **convergence の定義が曖昧** — "30K 以降の ROI 変動 ≤ 5%" の「変動」は max-min なのか標準偏差なのか。
   - **対策**: run_gate_check.py L298-301 参照 — `roi_variance_pct_after_30k` は config 値をそのまま使用。定義を YAML に明記する。

### §11.2 見落としている可能性

- HeavyTradingEnv の OnlineScaler (v455) が FeatureRegistry 特徴量に干渉する可能性
- action_space_type = "1d_position" と continuous_actions = True の二重指定による衝突
- SB3 の SAC は normalize_observations パラメータがある — env の OnlineScaler との重複

---

## §12 355# ブロッカーとの対応表

| 355# B# | 本ドキュメントでの対応 | 節 |
|---|---|---|
| B1: g2_sac_train.yaml 不在 | YAML テンプレート作成計画 | §3 B1, §4.1 P3A-1 |
| B2: 特徴量次元体系の断絶 | FeatureRegistry 単独使用の決定 + 選定根拠 | §2.4, §3 B2, §4.3 |
| B3: feature_columns の env 未注入 | 具体的 diff 付き修正計画 | §3 B3, §6.1 |
| B4: multi-seed ラッパー不在 | run_experiment.py 拡張設計 | §3 B4, §5 |
| B5: SACTrainer 3 重実装 | C をメインパス、A/B は deprecated 判断 | §2.1, §3 B5 |

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-09 | 1.0 | 初版 (vXXX 資産棚卸し + 5 ブロッカー解消計画 + 実装ロードマップ) |
| 2026-03-09 | 1.1 | 枝番付与 (356a/356b), B1 g2_sac_train.yaml 作成, B3 feature_columns 修正, B4 G2 dispatch + multi-seed 実装 |
