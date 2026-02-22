# 015 — ph3 計画: SAC 実装調査 & オンライン学習設計

> **目的**: 本文書は v460 ph3 (G2-train) に向け、現コードベースの SAC 関連実装を棚卸しし、  
> ライブ取引中のオンライン学習アーキテクチャを設計するためのものである。  
> 別の AI コーディングエージェントによるレビューを前提に記述している。

---

## 目次

1. [エグゼクティブサマリー](#1-エグゼクティブサマリー)
2. [SAC 実装ランドスケープ](#2-sac-実装ランドスケープ)
3. [訓練 ↔ 推論ギャップ分析](#3-訓練--推論ギャップ分析)
4. [v459 の教訓と根本原因](#4-v459-の教訓と根本原因)
5. [オンライン学習インフラの現状](#5-オンライン学習インフラの現状)
6. [v460 ph3 に向けた SAC 設計提案](#6-v460-ph3-に向けた-sac-設計提案)
7. [ライブ取引中のオンライン学習設計](#7-ライブ取引中のオンライン学習設計)
8. [未解決課題 & レビュー依頼事項](#8-未解決課題--レビュー依頼事項)

---

## 1. エグゼクティブサマリー

### 現状の結論

| 項目 | 状態 |
|------|------|
| SAC トレーナー実装数 | **6 系統** （統合が必要） |
| v459 SAC バックテスト結果 | **SAC ≈ Random** （Mann-Whitney p=0.64） |
| 訓練-推論次元不一致 | **致命的**: 88+ 次元で訓練 → 5 次元で推論 |
| オンライン学習エンジン | コード存在するが **MarketDataStream が DUMMY** |
| v460 maker 0% 環境での見通し | **手数料障壁撤廃**により根本的に異なるゲーム |

### 最重要リスク

1. **特徴量次元ミスマッチ** — 訓練と推論で使う特徴量が根本的に異なる
2. **6 系統の SAC 重複** — どれが正式なのか不明確で保守不能
3. **MarketRegimeDetector / MarketDataStream が DUMMY** — オンライン学習の中核が未実装
4. **SB3 `model.train()` の誤用** — AdaptiveSACCore がバッチを組み立てても SB3 に注入していない

---

## 2. SAC 実装ランドスケープ

### 2.1 全 6 系統の比較

| # | ファイル | 行数 | 役割 | 依存 | v460 推奨 |
|---|---------|------|------|------|-----------|
| 1 | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | 1,990 | 最新・最完全の SB3 ベーストレーナー | HeavyTradingEnv, SB3-SAC, CheckpointManager | **◎ 主テーナー** |
| 2 | `ztb/training/trainers/sac_trainer.py` | 735 | 旧トレーナー（EnsembleMixin 付き） | SB3-SAC, SACMetricsCallback | △ 廃止候補 |
| 3 | `ztb/training/sac_trainer.py` | 400 | ファサード → UnifiedTrainer へ委譲 | UnifiedTrainer | △ 薄いラッパー |
| 4 | `ztb/training/sac.py` | 490 | SACSuite CLI ツール | SACTrainer, argparse | ○ CLI は維持 |
| 5 | `ztb/training/adaptive_sac_core.py` | 763 | V433 カスタム PyTorch 適応型 SAC | AdaptiveSACPolicy, MarketRegimeDetector(DUMMY) | ✗ DUMMY 多すぎ |
| 6 | `ztb/training/v435/train_sac_v435.py` | 14 | 最小スタブ | — | ✗ 削除 |

**補助モジュール**:
- `ztb/training/sac_v430_training_optimizations.py` (422L): `GradientAccumulator`, `DynamicLRScheduler` ユーティリティ
- `ztb/training/online_learning_engine.py` (808L): オンライン学習エンジン（V433、後述）

### 2.2 統一 SACTrainer (#1) の詳細

```
SACTrainer(BaseAlgorithmTrainer)
├── SB3 SAC("MlpPolicy")
├── HeavyTradingEnv + Monitor + DummyVecEnv + VecNormalize
├── CheckpointManager (lz4, async, 1000-step interval)
├── TrainingStateManager (resume with replay_buffer)
├── ConfigurationManager (validation)
├── Market Regime Adaptation (MarketRegimeClassifier, optional)
├── Feature Set Resolution (v451, v454, default, etc.)
├── Reward Params Verification (Expected vs Actual logging)
└── Cost Breakdown Analysis (Gross/Fees/Slippage/Net)
```

**ハイパーパラメータ（デフォルト定数）**:
- `learning_rate`: 3e-4
- `buffer_size`: 1,000,000
- `batch_size`: 256
- `tau`: 0.005
- `gamma`: 0.99
- `ent_coef`: "auto"（自動エントロピー調整）
- `learning_starts`: 定数から（デフォルト値は constants.py で管理）

### 2.3 AdaptiveSACCore (#5) の詳細

カスタム PyTorch ネットワーク `AdaptiveSACPolicy(nn.Module)` を持つ:

```
AdaptiveSACPolicy
├── regime_encoder: Linear(4→32→16)  # volatility, trend, volume, confidence
├── policy_net: [obs_dim+16 → 256 → 256 → action_dim]  (LayerNorm 付き)
├── value_net: [obs_dim+16 → 256 → 256 → 1]
├── q1_net / q2_net: Twin Q-network
└── forward(obs, regime_features) → (policy, value, q1, q2)
```

**致命的問題**:
- `MarketRegimeDetector` は常に `"neutral"` を返す DUMMY
- `_get_current_market_data()` は常に `None` を返す
- `_perform_online_update()` でバッチを numpy に変換するが**使わない**（変数に代入せず捨てている）
- `self.sac_model.train(batch_size=len(batch))` は SB3 の `model.train()` を呼ぶが、これは **SB3 のリプレイバッファから** サンプリングするため、外部バッチは無視される

```python
# adaptive_sac_core.py L600-615 — _perform_online_update() のバグ
batch = [self.online_buffer[i] for i in batch_indices]
np.array([exp[0] for exp in batch])  # ← 結果を捨てている！
np.array([exp[1] for exp in batch])  # ← 結果を捨てている！
# ...
self.sac_model.train(batch_size=len(batch))  # ← SB3 内部バッファから学習、外部データ無視
```

---

## 3. 訓練 ↔ 推論ギャップ分析

### 3.1 次元ミスマッチ

```
【訓練時】                              【推論時】
HeavyTradingEnv / FastIntradayEnvV456    action_prediction.py
   88+ 次元の観測空間                      features[:5] のみ使用
   ├── [0:30]  Base OHLCV                  └── 先頭 5要素だけ取り出し
   ├── [30:57] MTF (5m/15m/1h)                 または 5未満なら zero-pad
   ├── [57:63] Cyclical time
   ├── [63:69] Global (spread, returns)
   ├── [69:82] Regime 13D
   └── [82:88] Account metrics
```

**結果**: モデルが 88 次元の入力空間で学習した重みは、推論時に 5 次元しか与えられないため **完全に無意味**。

### 3.2 行動空間の離散化

```python
# action_prediction.py — SAC continuous → discrete 変換
threshold = 0.1
if action_val > threshold:   → BUY (1)
elif action_val < -threshold: → SELL (2)
else:                        → HOLD (0)
```

- SAC が連続値で学習した微妙なポジションサイジングが、粗い 3 値に離散化される
- `FastIntradayEnvV456` は `Box([-1,0], [1,1])` の 2D 行動空間（target_position + ttl_fraction）を使うが、推論側は 1D しか見ていない

### 3.3 モデルレジストリの不在

- モデルファイルパスの命名規則でアルゴリズムを推定（`if "sac" in path_lower`）
- 観測空間次元・行動空間次元・特徴量セット等のメタデータが保存されない
- 訓練と推論で同じ前処理パイプラインを共有する仕組みがない

---

## 4. v459 の教訓と根本原因

### 4.1 バックテスト結果（Phase 4.5 最終判定: NO-GO）

| 戦略 | Net ROI | Gross PnL | Fees | Trades |
|------|---------|-----------|------|--------|
| Random | -15.02% | ≈ 0 | 15,023 | 950 |
| BuyAndHold | -0.29% | 0 | 20 | 1 |
| Momentum_RSI | -5.59% | +7,051 | 12,455 | 729 |
| **SAC (P1-1)** | **-14.99%** | **+1,171** | **16,161** | **1,024** |

- **Mann-Whitney U 検定**: p=0.6422（SAC vs Random は統計的に非有意）
- **手数料率**: Gross PnL +1,171 に対し Fees 16,161 → **手数料が利益の 1,380%**
- **BUY:SELL 比**: 512:512 → **方向性エッジ未学習**

### 4.2 根本原因の分解

```
原因 1: 手数料構造  ←←← v460 で解決（maker 0%）
  taker 0.1% × 高頻度取引 = 利益の完全吸収

原因 2: 特徴量の予測力不足
  RSI系 1分足のみ → マイクロストラクチャ情報なし ←← v460 で 10 特徴追加

原因 3: 訓練-推論ギャップ
  88次元→5次元の次元ミスマッチ ←←← **未解決** ★★★

原因 4: 行動空間設計の粗さ
  連続値 → 3値離散化で情報損失 ←←← 検討必要
```

### 4.3 v460 で解消される要因

- **手数料 0%** (maker-only): コスト障壁が完全に消える
- **マイクロストラクチャ 10 特徴**: bid_ask_spread, depth_imbalance, trade_flow_imbalance, vwap_deviation, trade_intensity, order_flow_toxicity, price_impact, micro_return_vol, bid_depth_slope, ask_depth_slope
- **ph1 (G1-info) で事前検証**: XGBoost Walk-Forward で特徴量の予測力を RL 以前に確認

### 4.4 v460 でも残る課題

- **訓練-推論ギャップ** → ph3 設計で解消必須
- **モデルレジストリ** → メタデータ管理の設計必要
- **オンライン学習ストリーム** → リアルタイムデータ連携が未実装

---

## 5. オンライン学習インフラの現状

### 5.1 アーキテクチャ概要（V433 設計）

```
OnlineLearningEngine (808L)
├── Thread 1: _learning_loop()
│    └── _perform_learning_update()
│         └── AdaptiveSACCore.online_learn() × mini_batch
├── Thread 2: _monitoring_loop()
│    ├── _check_performance()
│    └── _check_resources() (psutil)
├── Thread 3: _backup_loop()
│    └── _perform_backup() (torch.save)
├── Async: start_online_learning()
│    └── _data_processing_loop()
│         ├── MarketDataStream.get_latest_data() ← DUMMY (None)
│         ├── _preprocess_market_data()
│         ├── _extract_features() (SMA, RSI, MACD, Volume)
│         ├── _generate_experiences_from_data()
│         └── ConceptDriftDetector.check_drift()
└── Components:
     ├── OnlineExperienceBuffer (deque, max 50K, prioritized sampling)
     ├── ConceptDriftDetector (reward window前半 vs 後半比較)
     └── ExperienceTuple (obs, action, reward, next_obs, done)
```

### 5.2 DUMMY / 未実装コンポーネント

| コンポーネント | 状態 | 影響 |
|-------------|------|------|
| `MarketDataStream` | **DUMMY** — 常に `None` を返す | データ取得ループが空回り |
| `MarketRegimeDetector` | **DUMMY** — 常に `"neutral"` | レジーム適応が機能しない |
| `_get_current_market_data()` | **DUMMY** — 常に `None` | 市場データ取得不能 |
| `_perform_online_update()` の SB3 統合 | **バグ** — 外部バッチ無視 | オンライン学習が実質的に無効 |

### 5.3 UnifiedTrainer との統合

`trainer.py` L1913-2130 で V433 コンポーネントを初期化:

```python
# trainer.py L1970-1980
self.online_learning_config = OnlineLearningConfig(...)
self.online_learning_engine = OnlineLearningEngine(
    self.online_learning_config, self.adaptive_sac_core
)

# trainer.py L2122-2130 — 非同期スレッドで起動
def run_online_learning():
    loop.run_until_complete(online_engine.start_online_learning())
online_thread = threading.Thread(target=run_online_learning, daemon=True)
online_thread.start()
```

観測空間/行動空間の次元は config から取得（デフォルト: obs_dim=10, action_dim=3）。

### 5.4 OnlineLearningEngine の特徴量パイプライン

```python
# online_learning_engine.py — 独自の簡易特徴量（メインパイプラインと非共有）
features["sma_5"]  = data["close"].rolling(5).mean()
features["sma_20"] = data["close"].rolling(20).mean()
features["rsi"]    = 100 - (100 / (1 + gain/loss))
features["macd"]   = ema_12 - ema_26
features["macd_signal"] = features["macd"].ewm(span=9).mean()
features["volume_sma_5"] = data["volume"].rolling(5).mean()
features["volume_ratio"]  = data["volume"] / features["volume_sma_5"]
```

**問題**: メインの `ztb/features/` パイプライン（microstructure 含む 88+ 次元）と**完全に別物**。
環境で学習した知識がオンライン学習ループで再利用不能。

---

## 6. v460 ph3 に向けた SAC 設計提案

### 6.1 推奨: 統一 SACTrainer + FastIntradayEnvV456

**根拠**:
- 統一 SACTrainer (#1) は最も完成度が高い（checkpoint, resume, regime, cost analysis）
- `FastIntradayEnvV456` は固定 88 次元で再現性が高い
- v460 の microstructure 特徴量は `[0:30]` 基本特徴に追加可能（パイプライン拡張）

```
推奨アーキテクチャ:

FeaturePipeline (ztb/features/)
  └── microstructure.py (10 features)
  └── base features (OHLCV, etc.)
  └── MTF, cyclical, regime, account
       │
       ▼
FastIntradayEnvV456 (固定 N 次元)
       │
       ▼
SACTrainer (unified_trainer/algorithms/sac_trainer.py)
  └── SB3 SAC("MlpPolicy")
       │
       ▼
ModelRegistry (新規: メタデータ付きモデル保存)
  └── obs_dim, action_dim, feature_set, scaler_params, version
       │
       ▼
ActionPrediction (live_trader/)
  └── ModelRegistry.load() → 正しい次元で推論
```

### 6.2 廃止対象

| ファイル | アクション |
|---------|-----------|
| `ztb/training/trainers/sac_trainer.py` (#2) | 非推奨マーク → 次バージョンで削除 |
| `ztb/training/sac_trainer.py` (#3) | ファサード維持 or 非推奨 |
| `ztb/training/adaptive_sac_core.py` (#5) | DUMMY 多数、設計を #1 に吸収して廃止 |
| `ztb/training/v435/train_sac_v435.py` (#6) | 即削除 |
| `ztb/training/online_learning_engine.py` | 再設計（§7 参照） |

### 6.3 訓練-推論ギャップの解消

#### 方式 A: ModelRegistry による次元統一（推奨）

```python
@dataclass
class ModelMetadata:
    algorithm: str          # "sac"
    version: str            # "v460"
    observation_dim: int    # 88 or 98 (microstructure追加後)
    action_dim: int         # 2 (target_position, ttl_fraction)
    action_space_type: str  # "continuous"
    feature_set: str        # "v460_microstructure"
    feature_names: List[str]
    scaler_type: str        # "GroupedFeatureScaler"
    scaler_params_path: str # scaler のパス
    training_env: str       # "FastIntradayEnvV456"
    created_at: str
    training_config: Dict[str, Any]
```

モデル保存時に `metadata.json` を同梱し、推論時に自動ロード:

```python
# action_prediction.py — 改修案
class ActionPrediction:
    def __init__(self, live_trader):
        self.metadata = ModelMetadata.load(model_path / "metadata.json")
        self.scaler = load_scaler(self.metadata.scaler_params_path)

    def predict_action(self, raw_features: Dict[str, float]) -> int:
        # メタデータに基づいて正しい次元の特徴量ベクトルを構築
        obs = self._build_observation(raw_features, self.metadata.feature_names)
        obs_scaled = self.scaler.transform(obs)
        action, _ = self.model.predict(obs_scaled.reshape(1, -1))
        return self._interpret_action(action, self.metadata)
```

#### 方式 B: 環境ラッパーによる推論環境再現

```python
class InferenceEnvWrapper:
    """訓練環境と同一の観測空間を推論時にも再現"""
    def __init__(self, model_path: Path):
        self.env = FastIntradayEnvV456.from_metadata(model_path / "env_config.json")
        self.feature_pipeline = FeaturePipeline.from_metadata(model_path / "features.json")

    def prepare_observation(self, market_data: Dict) -> np.ndarray:
        features = self.feature_pipeline.transform(market_data)
        return self.env.normalize_observation(features)
```

**推奨**: 方式 A。軽量で、環境全体を再生成する方式 B よりも保守が容易。

### 6.4 行動空間の設計選択

| 選択肢 | 説明 | 利点 | 欠点 |
|--------|------|------|------|
| **3 値離散 (BUY/HOLD/SELL)** | 現行の閾値判定 | シンプル | ポジションサイズ不可 |
| **2D 連続 (position, ttl)** | FastIntradayEnvV456 形式 | 細粒度 | 探索空間が大きい |
| **1D 連続 (target_position)** | -1〜+1 のポジション比率 | 中間的 | TTL 管理が別途必要 |

**ph3 では 1D 連続を推奨**: maker-only で TTL は maker 側ロジックで管理するため、RL が決めるべきは「どの程度のポジションを取るか」のみ。

---

## 7. ライブ取引中のオンライン学習設計

### 7.1 設計方針

```
【原則】
1. ライブ取引は "推論モード" が最優先 — 学習は非同期バックグラウンド
2. 学習の暴走が取引に影響してはならない — モデル更新はアトミック
3. 概念ドリフト検知は必要だが、反応は慎重に — 急激な適応は逆効果
4. 本番モデルの更新は "影モデル" による検証後のみ
```

### 7.2 アーキテクチャ

```
                    ┌─────────────────────┐
  Market Data ────▶│  LiveTrader (推論)   │──▶ Orders
  (WebSocket)      │  ├── model_prod      │
                   │  ├── ActionPrediction │
                   │  └── ExperienceLogger │──┐
                   └─────────────────────┘   │
                                              │ experiences (obs, action, reward, done)
                                              ▼
                   ┌─────────────────────────────────────┐
                   │  OnlineLearningEngine (バックグラウンド)  │
                   │  ├── ExperienceBuffer (Ring, 100K)   │
                   │  ├── model_shadow (影モデル)          │
                   │  ├── ConceptDriftDetector             │
                   │  ├── ShadowModelEvaluator             │
                   │  └── ModelSwapGate                    │
                   └─────────────────────────────────────┘
                              │
                    影モデルが prod より                 
                    統計的に優位なら                    
                              ▼
                   ┌─────────────────────┐
                   │  Model Swap (atomic) │
                   │  prod ← shadow       │
                   └─────────────────────┘
```

### 7.3 コンポーネント設計

#### 7.3.1 ExperienceLogger

```python
@dataclass
class LiveExperience:
    timestamp: datetime
    observation: np.ndarray      # 完全な N 次元観測（ModelMetadata に基づく）
    action: np.ndarray           # モデル出力（連続値）
    reward: float                # 実現 PnL（遅延評価）
    next_observation: np.ndarray
    done: bool
    market_regime: str           # 推定レジーム
    execution_info: Dict         # fill_rate, latency, slippage
```

**報酬の遅延評価**: maker 注文は即座に約定しないため、報酬計算にはタイムラグが生じる。

```
注文発行 ──── 約定待機 ──── 約定確認 ──── PnL 確定 ──── 報酬計算
  t=0          t=1~60s       t=?           t=?+1         t=?+1
                                     └── ここでやっと reward が確定
```

**設計**: Experience は `reward=None` で一旦バッファに投入し、約定確認後に遅延更新する。

#### 7.3.2 ConceptDriftDetector（改修案）

現行の実装（報酬ウィンドウの前半と後半を比較）は単純すぎる。

```python
class ImprovedConceptDriftDetector:
    """改修版: 複数指標による概念ドリフト検知"""

    def __init__(self, config):
        self.window_size = config.drift_window  # 500
        self.significance_level = 0.05

    def check_drift(self, metric_history: List[float]) -> DriftResult:
        # 1. Page-Hinkley Test — 平均のシフトを逐次検知
        ph_result = self._page_hinkley_test(metric_history)

        # 2. Kolmogorov-Smirnov Test — 分布の変化を検知
        ks_result = self._ks_test(
            metric_history[:len(metric_history)//2],
            metric_history[len(metric_history)//2:]
        )

        # 3. Feature distribution shift — 入力特徴量の分布変化
        feature_drift = self._check_feature_drift()

        return DriftResult(
            is_drifting=ph_result.detected or ks_result.detected,
            severity=max(ph_result.severity, ks_result.severity),
            drift_type="reward" if ph_result.detected else "feature" if feature_drift else "none"
        )
```

#### 7.3.3 ShadowModelEvaluator

```python
class ShadowModelEvaluator:
    """影モデルの評価 — prod との比較"""

    def __init__(self, min_eval_episodes: int = 50):
        self.min_eval_episodes = min_eval_episodes
        self.prod_results = deque(maxlen=200)
        self.shadow_results = deque(maxlen=200)

    def should_swap(self) -> Tuple[bool, Dict]:
        if len(self.shadow_results) < self.min_eval_episodes:
            return False, {"reason": "insufficient_data"}

        # Welch's t-test for unequal variance
        t_stat, p_value = stats.ttest_ind(
            list(self.shadow_results), list(self.prod_results),
            equal_var=False
        )

        # 影モデルが統計的に優位 (片側検定) かつ実質的に改善
        improvement = np.mean(list(self.shadow_results)) - np.mean(list(self.prod_results))

        swap = (
            p_value < 0.05 and          # 統計的有意
            t_stat > 0 and              # 影 > prod
            improvement > 0.001 and     # 実質的改善 (0.1%)
            len(self.shadow_results) >= self.min_eval_episodes
        )

        return swap, {
            "t_stat": t_stat,
            "p_value": p_value,
            "improvement": improvement,
            "prod_mean": np.mean(list(self.prod_results)),
            "shadow_mean": np.mean(list(self.shadow_results)),
        }
```

#### 7.3.4 ModelSwapGate

```python
class ModelSwapGate:
    """アトミックなモデル交換"""

    def __init__(self, max_swaps_per_day: int = 3):
        self.max_swaps_per_day = max_swaps_per_day
        self.swap_count_today = 0

    def execute_swap(self, live_trader, shadow_model_path: Path) -> bool:
        if self.swap_count_today >= self.max_swaps_per_day:
            return False  # 1日のスワップ回数制限

        # 1. 全ポジションをフラットに
        live_trader.close_all_positions()

        # 2. モデルのアトミックスワップ
        with live_trader.model_lock:
            live_trader.model = load_model(shadow_model_path)
            live_trader.metadata = ModelMetadata.load(shadow_model_path / "metadata.json")

        # 3. スワップ記録
        self.swap_count_today += 1
        return True
```

### 7.4 学習スケジュール

```
├── 毎秒: ExperienceLogger が市場データ + 行動 + 報酬を記録
├── 毎 100 experiences: 影モデルに対してミニバッチ更新（10 gradient steps）
├── 毎 30 分: ConceptDriftDetector チェック
├── 毎 1 時間: ShadowModelEvaluator で prod vs shadow 比較
├── swap 条件成立時: ModelSwapGate で安全にスワップ
├── 毎日 00:00 UTC: swap_count リセット、日次レポート生成
└── 毎週: 完全再訓練（全蓄積データで shadow を初期化）
```

### 7.5 SB3 との統合方法

SB3 の SAC は内部にリプレイバッファを持つ。オンライン学習のアプローチ:

#### 方式 1: SB3 リプレイバッファへの直接注入（推奨）

```python
# SB3 のリプレイバッファに外部経験を注入
from stable_baselines3.common.buffers import ReplayBuffer

def inject_experience(model: SAC, exp: LiveExperience):
    """SB3 リプレイバッファに直接経験を追加"""
    model.replay_buffer.add(
        obs=exp.observation,
        next_obs=exp.next_observation,
        action=exp.action,
        reward=np.array([exp.reward]),
        done=np.array([exp.done]),
        infos=[{}],
    )

# バッファに十分溜まったら学習
if model.replay_buffer.size() >= model.learning_starts:
    model.train(gradient_steps=10, batch_size=256)
```

**利点**: SB3 の学習ループ（critic/actor loss, entropy, target update）がそのまま使える
**注意**: `model.train()` は内部的に `self.replay_buffer.sample()` するため、注入した経験がサンプリングされるには十分なバッファサイズが必要

#### 方式 2: 完全カスタム学習ループ

SB3 を推論専用にし、学習は PyTorch で直接行う。
AdaptiveSACCore の設計思想に近いが、DUMMY を排除した形で再実装が必要。

**推奨**: 方式 1。SB3 の成熟した実装を活かしつつ、最小限の拡張で実現可能。

### 7.6 リスク管理

| リスク | 対策 |
|--------|------|
| 学習の暴走（gradient explosion） | gradient clipping, lr のウォームアップ |
| オーバーフィッティング（直近データに過適応） | Experience Replay のバッファサイズを大きく（100K+） |
| 概念ドリフトの誤検知 | 複数検定の組合せ + 最低サンプル数制約 |
| モデルスワップ時の損失 | ポジションクローズ → スワップ → 緩やかな再開 |
| メモリリーク | ExperienceBuffer のリングバッファ化 + psutil 監視 |
| 学習スレッドのクラッシュ | デーモンスレッド + watchdog + 自動再起動 |

---

## 8. 未解決課題 & レビュー依頼事項

### 8.1 レビュアーへの質問

1. **SB3 リプレイバッファ注入方式 vs カスタム PyTorch 学習ループ** — どちらが v460 の maker-only 戦略に適切か？SB3 の SAC 実装は十分に柔軟か？

2. **影モデルの評価期間** — 50 エピソード（≈ 数時間相当）は十分か？統計的検出力を確保しつつ適応速度も維持するバランスは？

3. **報酬の遅延評価** — maker 注文の約定遅延がある環境で、TD 学習の bootstrap 推定はどの程度バイアスされるか？n-step returns の n はどう設定すべきか？

4. **6 系統の SAC 統合方針** — #1 に集約して他を deprecate する方針で問題ないか？#5 (AdaptiveSACCore) のレジーム適応設計で #1 に取り込むべき要素はあるか？

5. **行動空間** — maker-only 環境で 1D 連続（target_position のみ）は適切か？あるいは離散 3 値で十分か？ポジションサイジングの価値をどう評価すべきか？

6. **ConceptDriftDetector の検定方式** — Page-Hinkley + KS で十分か？暗号資産のボラティリティ特性を考慮した非パラメトリック手法の提案は？

7. **メモリ/計算資源** — VPS (2-4 vCPU, 4-8GB RAM) でオンライン学習スレッドを常駐させることの実現可能性は？SB3 リプレイバッファ 100K の推定メモリ使用量は？

### 8.2 ph3 開始前に完了すべき前提

| # | 項目 | 依存 |
|---|------|------|
| 1 | ph2 G1.1-exec Gate 通過 | fill_rate ≥ 70% |
| 2 | ModelRegistry 実装 | §6.3 方式 A |
| 3 | ActionPrediction の次元ミスマッチ修正 | ModelRegistry 依存 |
| 4 | SAC トレーナー統合（#2, #5, #6 の deprecation） | — |
| 5 | FastIntradayEnvV456 に microstructure 特徴量を統合 | ztb/features/microstructure.py |
| 6 | ExperienceLogger の設計・実装 | 報酬遅延評価含む |

### 8.3 ph3 スコープの明確化

ph3 (G2-train) の **Go 条件** (000# §2.5):

| 条件 | 閾値 |
|------|------|
| gross > 0 の seed 比率 | ≥ 3/4 (75%) |
| IC の seed 間標準偏差 | ≤ 0.03 |
| 学習曲線の収束 | 30K 以降で ROI 変動 ≤ 5% |
| worst-seed 下限 | ROI > −2% |

**スコープ外** (ph3 では扱わない):
- ライブ取引でのオンライン学習実行（ph5 の範囲）
- 取引所間の切替ロジック（ph4 以降）
- 複数通貨ペア対応

---

> **文書管理**  
> - 作成日: 2025-01-28  
> - フェーズ: ph3 計画  
> - 前提文書: 000# (プロジェクト提案), 100# (v459 Phase4.5 完了報告)  
> - 次ステップ: レビュー結果を受けて `016_ph3_resp_015.md` を作成
