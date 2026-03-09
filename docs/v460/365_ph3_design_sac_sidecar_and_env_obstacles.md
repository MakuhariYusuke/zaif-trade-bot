# 365# ph3 設計: SAC Sidecar アーキテクチャ + HeavyTradingEnv 阻害要因分析

| 項目 | 値 |
|---|---|
| 文書番号 | 365# |
| フェーズ | ph3 (G2-train) |
| 前提 | 356#, 358#, 361#, 362#, 364# B4 |
| 作成 | Copilot 038 |
| ステータス | **ACTIVE** |

---

## §1 エグゼクティブサマリ

本文書は 364# B4/P1-3「ph3 sidecar 設計文書」を充足する。

### 3つの柱

1. **SAC Sidecar Architecture**: SAC を直接トレーダーではなく「方向性 Regime Prior」として ph2 に注入する設計 (361#/362# 合意)
2. **HeavyTradingEnv 阻害要因**: 市場理論ベースの 15 機構が SAC 学習を歪める分析と対処法
3. **Warm-Start Incremental Training**: チェックポイント + replay buffer 永続化による蓄積型学習の設計

### 結論

| 判断 | 決定 |
|---|---|
| SAC の役割 | Directional Bias [-1, +1] を出力する **Sidecar** (直接執行ではない) |
| env 市場理論機構 | g2_sac_train.yaml で**全て無効** (§3 で確認済)。追加のガード不要 |
| 定期再学習 | retrain_scheduler は SkipGate 専用。SAC 用は **新規作成** (§5) |
| Replay buffer | SB3 標準の `save/load_replay_buffer()` を活用。**未実装→P1** |
| 特徴量ギャップ | Sidecar であれば OHLCV で**構造的に許容可能** (§2.3) |

---

## §2 SAC Sidecar Architecture

### §2.1 361#/362# の合意要旨

> **「SAC に板 (Quote) を出させるな。方針 (Regime) を出させよ。」** — 362#

2つの独立したレビュー (361#: Copilot, 362#: Gemini) が同一結論に到達:

| 項目 | 直接執行モデル (却下) | Sidecar モデル (合意) |
|---|---|---|
| SAC 出力 | BUY/SELL/HOLD の注文指示 | Directional Bias [-1, +1] |
| 出力頻度 | 毎ステップ (1分) | 数分に 1 回のシグナル更新 |
| ph2 との関係 | ph2 を置き換える | ph2 に**注入**して非対称化 |
| 必要な精度 | 高 (直接 PnL に直結) | 中 (方向性の先読みで十分) |
| 特徴量要件 | microstructure 必須 | OHLCV ベースで合理的 |

### §2.2 Sidecar データフロー

```
┌──────────────────────────────────────────────────────────┐
│ SAC Sidecar (数分毎に推論)                                │
│                                                          │
│  [OHLCV 1m 特徴量 12次元]                                │
│       ↓                                                  │
│  SAC Actor Network                                       │
│       ↓                                                  │
│  directional_bias: float  [-1.0 .. +1.0]                │
│       ↓                                                  │
│  ┌─────────────────────────────────┐                     │
│  │ bias > +0.3  → BUY bias        │                     │
│  │ bias < -0.3  → SELL bias       │                     │
│  │ else         → NEUTRAL         │                     │
│  └─────────────────────────────────┘                     │
└──────────────┬───────────────────────────────────────────┘
               │ sidecar_signal.json (atomic write)
               ▼
┌──────────────────────────────────────────────────────────┐
│ ph2 fill_test (毎サイクル読込)                            │
│                                                          │
│  cycle_gate_aggregator.py                                │
│       ↓                                                  │
│  sidecar_bias = read_sidecar_signal()                    │
│       ↓                                                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ BUY bias:  buy_offset += sidecar_boost              │ │
│  │            sell_offset -= sidecar_boost              │ │
│  │ SELL bias: sell_offset += sidecar_boost              │ │
│  │            buy_offset -= sidecar_boost               │ │
│  │ NEUTRAL:   no change                                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  → Buy/Sell の指値攻撃性を非対称に調整                    │
│  → Asymmetric Maker として機能                            │
└──────────────────────────────────────────────────────────┘
```

### §2.3 特徴量ギャップの構造的許容性

| 観点 | 直接執行の場合 | Sidecar の場合 |
|---|---|---|
| **必要な予測対象** | 「この注文が fill されるか」 | 「今後数分で価格はどちらに動くか」 |
| **必要な時間軸** | ミリ秒〜秒 (microstructure) | 分〜数分 (medium-term trend) |
| **OHLCV 情報量** | 不十分 (板・約定情報がない) | **十分** (方向性予測には OHLCV で合理的) |
| **環境と現実の乖離** | 致命的 (fill probability が未知) | 許容可能 (価格方向性は共通) |

**結論**: Sidecar アーキテクチャでは OHLCV ベースの FeatureRegistry 12 特徴量で**構造的に問題ない**。362# が指摘した次元ギャップは「SAC の責務を方向性予測に限定する」ことで解消される。

### §2.4 注入ポイント設計

fill_test パイプライン上の SAC 注入ポイント候補:

| # | 注入ポイント | メリット | デメリット |
|---|---|---|---|
| **A** | `cycle_gate_aggregator.py` の offset 計算 | 既存の offset 機構に乗る。影響範囲が明確 | offset 以外の調整ができない |
| **B** | `fill_loop_orchestrator.py` の cycle 判定前 | cycle skip にも影響可能 | 影響範囲が広すぎる |
| **C** | `fill_config.py` の動的パラメータ | config ベースで clean | hot-reload 頻度が高すぎる |

**推奨: A** — offset 計算への注入が最小侵襲。`sidecar_boost` の値域を ±0.1 bps 程度に制限すれば、暴走リスクも低い。

---

## §3 HeavyTradingEnv 市場理論機構 — SAC 学習阻害分析

### §3.1 機構一覧と干渉リスク

HeavyTradingEnv には 15 の市場理論ベース機構が組み込まれている。SAC 学習への干渉を深刻度順に整理する。

| # | 機構 | ファイル | 深刻度 | g2_sac_train.yaml で無効? |
|---|---|---|---|---|
| 1 | **HybridConfig entry/exit override** | core.py L980-1120 | ★★★★★ | ✅ 無効 (`hybrid_config` 未設定) |
| 2 | **CurriculumManager + 報酬ステージ遷移** | balance_curriculum.py | ★★★★★ | ✅ 無効 (`curriculum_stage` 未設定 → `"simple"` fallback) |
| 3 | **ThresholdManager 動的閾値** | threshold_manager.py | ★★★★☆ | ⚠️ 部分有効 (後述) |
| 4 | **BehavioralPenaltyCalculator** | behavioral_penalty_calculator.py | ★★★★☆ | ✅ 無効 (reward_settings 未設定) |
| 5 | **SignalIntegrator 動的重み** | signal_integrator.py | ★★★★☆ | ✅ 無効 (`signal_guidance_enabled=False`) |
| 6 | **ActionValidator マスク** | action_validator.py | ★★★☆☆ | ⚠️ 部分有効 (資金チェックは常時作動) |
| 7 | **DynamicRewardShaper** | dynamic_reward_shaper.py | ★★★☆☆ | ✅ 無効 (reward_settings 未設定) |
| 8 | **PositionManager フィルタ** | position_manager.py | ★★★☆☆ | ⚠️ 部分有効 (後述) |
| 9 | **連続→離散変換 (根本構造)** | action_executor.py | ★★☆☆☆ | **常時作動** (設計上不可避) |
| 10 | **TP/SL 強制清算** | core.py L1140-1220 | ★★☆☆☆ | ⚠️ 設定依存 |
| 11 | **Drawdown ペナルティ** | core.py L1410-1425 | ★★☆☆☆ | **常時作動** (threshold=20%) |
| 12 | **Bankruptcy ペナルティ** | core.py L1395-1405 | ★★☆☆☆ | **常時作動** (PV<2000 で done) |
| 13 | **RegimeClassifier 報酬乗数** | core.py L1300-1370 | ★★☆☆☆ | ✅ 無効 (`advanced_market_regime` 未設定) |
| 14 | **AsymmetricRewardScaler** | asymmetric_reward_scaler.py | ★☆☆☆☆ | ✅ 中立 (デフォルト乗数=1.0) |
| 15 | **DomainRandomizer** | core.py L475-490 | ★☆☆☆☆ | ✅ 無効 (`domain_randomization` 未設定) |

### §3.2 現状の安全性評価

**g2_sac_train.yaml は市場理論機構を一切有効化していない。**

以下の YAML キーが**存在しない**ことで、大半のリスクは回避済み:

```yaml
# g2_sac_train.yaml に存在しないキー (= 全てデフォルトで無効):
curriculum_learning: ...        # → CurriculumManager 無効
curriculum_stage: ...           # → "simple" fallback
hybrid_config: ...              # → HybridConfig entry/exit override 無効
domain_randomization: ...       # → DomainRandomizer 無効
advanced_market_regime: ...     # → RegimeClassifier 無効
signal_guidance_enabled: ...    # → SignalIntegrator 無効
adaptive_threshold_mode: ...    # → Z-Score 動的閾値 無効
enable_forced_diversity: ...    # → Forced Diversity 無効
```

### §3.3 残存リスク — 常時作動する機構

以下の 5 機構は YAML 設定に関わらず**常時作動**し、SAC 学習に影響する:

#### (A) 連続→離散変換 (ActionExecutor)

| 項目 | 詳細 |
|---|---|
| **問題** | SAC の連続出力 `[-1, +1]` が `±0.3333` 閾値で BUY/HOLD/SELL に硬直分割 |
| **影響** | 閾値境界での報酬勾配不連続、Action Collapse リスク |
| **対処法** | **現時点では許容**。Sidecar モデルでは SAC 出力をそのまま `directional_bias` として使い、env 内の離散化は訓練時のみの問題。SB3 SAC は `ent_coef="auto"` で exploration を維持するため、Action Collapse は自動温度調整で緩和される |
| **将来対応** | 連続行動空間をそのまま使う env (LiteTradingEnv) への移行を ph4 で検討 |

#### (B) ActionValidator 資金チェック

| 項目 | 詳細 |
|---|---|
| **問題** | portfolio_value が不足すると BUY/SELL がマスクされ HOLD に強制変換。SAC は**マスクを直接 observe しない** |
| **影響** | Ghost action 問題 — SACが BUY を選んでも HOLD が実行され、HOLD の報酬を BUY の Q 値更新に使う |
| **対処法** | `info["action_masks"]` を observation に追加するか、`max_position_size=0.01 BTC` (≈15万円) に対して `initial_portfolio_value=1000万円` であり、資金不足はほぼ発生しない。**実質的リスクは低い** |
| **将来対応** | MaskablePPO / Action Masking 対応 SAC への移行 (SB3-Contrib) |

#### (C) Drawdown ペナルティ

| 項目 | 詳細 |
|---|---|
| **問題** | drawdown > 20% で持続的ペナルティ。50K steps ではここに到達する可能性あり |
| **影響** | 大きな損失後にペナルティ累積で Q 値推定が歪む |
| **対処法** | **許容** — リスク管理として合理的。50K steps + initial_PV=1000万 + max_pos=0.01BTC のスケールでは drawdown 20% 到達は稀。到達した場合は学習の早期段階で方策が悪い証拠 |

#### (D) Bankruptcy ペナルティ

| 項目 | 詳細 |
|---|---|
| **問題** | PV < 2000 で done=True + 巨大負ペナルティ。Q 値推定に外れ値 |
| **影響** | episode 終了と同時なので 1 回限り。replay buffer 内の sample としては稀 |
| **対処法** | **許容** — 発生確率が極めて低い (1000万→2000 は 99.98% 損失)。SAC の gradient clipping で Q 値への影響は限定的 |

#### (E) ThresholdManager 基本閾値

| 項目 | 詳細 |
|---|---|
| **問題** | `adaptive_threshold_mode=False` でも基本閾値 (±0.3333) は常時適用 |
| **影響** | (A) と同根。連続→離散の固定閾値 |
| **対処法** | (A) と同じ。**現時点では許容** |

### §3.4 要対応事項の整理

| 優先度 | 対処 | 工数 | 効果 |
|---|---|---|---|
| **P0** | なし (g2_sac_train.yaml で主要リスクは既に回避済み) | — | — |
| **P1** | `g2_sac_train.yaml` に**明示的無効化コメント**を追加 (意図の文書化) | 0.5h | 将来の混乱防止 |
| **P2** | `action_masks` を observation に含める env 拡張 | 2-4h | Ghost action 解消 |
| **P3** | LiteTradingEnv (連続行動空間直接使用) の設計 | 1-2d | 根本解決 |

### §3.5 将来の env 拡張時の注意事項

ph4 以降で HeavyTradingEnv にカリキュラム学習や HybridConfig を**有効化して SAC を訓練する場合**、以下の問題が顕在化する:

| 機構 | 顕在化する問題 | 対処法 |
|---|---|---|
| CurriculumManager | 報酬関数の非定常性。replay buffer 内 sample の報酬が不一致 | **replay buffer flush** — ステージ遷移時にバッファをクリア |
| HybridConfig ZScore entry | SAC の行動選択を完全上書き → credit assignment 崩壊 | **ZScore/Pullback を OFF**。entry は SAC に委ねる |
| HybridConfig TP/SL exit | 全 exit を HOLD に差し替え → SAC が exit を学習できない | exit override を OFF、TP/SL は報酬シェーピングで誘導 |
| BehavioralPenalty | PnL 無関係なバランスペナルティが Q 値を歪める | ペナルティ強度を 0.01 以下に低減 or 完全無効化 |
| SignalIntegrator | グランビル/ダウ理論がReward Hacking の温床 | weight を 0.0 に設定し SAC を pure PnL で訓練 |
| DynamicRewardShaper | レジーム依存の報酬スケール変動 | レジーム情報を observation に含め、スケーリングは無効化 |
| Emergency intervention (-500) | Q 値推定を壊滅的に歪める外れ値 | **絶対に有効化しない** |

---

## §4 Warm-Start Incremental Training 設計

### §4.1 現状の問題

| 項目 | 現状 | 問題 |
|---|---|---|
| モデル保存 | `model.save(path)` — 重みのみ | replay buffer 破棄 |
| モデルロード | `SAC.load(path)` — 重みのみ | 過去の学習経験が消失 |
| 訓練方式 | 毎回 0 からフルトレーニング | catastrophic forgetting リスク |
| Replay buffer | SB3 標準で `save/load_replay_buffer()` 利用可能 | **sac_algorithm.py / sac_train.py で未使用** |

### §4.2 Warm-Start フロー

```
[初回訓練]
  model = SAC(env)
  model.learn(50_000 steps)
  model.save("sac_v460.zip")
  model.save_replay_buffer("sac_v460_buffer.pkl")
                    ↓
[2回目以降: Warm-Start]
  model = SAC.load("sac_v460.zip", env=new_env)
  model.load_replay_buffer("sac_v460_buffer.pkl")
  model.learn(10_000~15_000 additional steps)     ← 追加学習のみ
  ↓ OOS validation gate
  ├── PASS → model.save() + save_replay_buffer() → デプロイ
  └── FAIL → ロールバック (前回モデルを維持)
```

### §4.3 SB3 の replay buffer 機能

`.venv` にインストール済みの SB3 は以下をサポート:

```python
# OffPolicyAlgorithm (SAC の親クラス) が提供
model.save_replay_buffer("buffer.pkl")      # pickle 形式で保存
model.load_replay_buffer("buffer.pkl")      # 復元 + truncate_last_traj

# CheckpointCallback も対応
CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    save_replay_buffer=True,     # ★ buffer も同時保存
    save_vecnormalize=True,
)
```

### §4.4 必要な実装変更

#### (1) sac_algorithm.py — replay buffer ラッパー追加

```python
# ztb/training/algorithms/sac/sac_algorithm.py に追加

def save_replay_buffer(self, path: str | Path) -> None:
    """Replay buffer を pickle 形式で保存."""
    if self.model is None:
        raise RuntimeError("Model not initialized")
    self.model.save_replay_buffer(str(path))

def load_replay_buffer(self, path: str | Path) -> None:
    """保存済み replay buffer を復元."""
    if self.model is None:
        raise RuntimeError("Model not initialized")
    self.model.load_replay_buffer(str(path))
```

#### (2) sac_train.py — warm-start パス追加

```python
# scripts/v460/lib/tasks/sac_train.py の _train_with_checkpoints() 修正

def _train_with_checkpoints(model, env, cfg):
    # Warm-start: 既存モデル + buffer があればロード
    pretrained_path = cfg.get("training", {}).get("pretrained_model_path")
    buffer_path = cfg.get("training", {}).get("pretrained_buffer_path")

    if pretrained_path and Path(pretrained_path).exists():
        model = SAC.load(pretrained_path, env=env)
        if buffer_path and Path(buffer_path).exists():
            model.load_replay_buffer(buffer_path)

    # 訓練実行 (warm-start 時は追加ステップのみ)
    ...

    # 保存時に buffer も同時保存
    model.save(str(model_path))
    model.save_replay_buffer(str(model_path.with_suffix(".buffer.pkl")))
```

#### (3) g2_sac_train.yaml — warm-start 設定追加

```yaml
training:
  total_timesteps: 50000
  checkpoint_interval: 5000
  val_ratio: 0.2
  # Warm-start (2回目以降に有効化)
  # pretrained_model_path: "models/v460/sac_v460_seed42.zip"
  # pretrained_buffer_path: "models/v460/sac_v460_seed42.buffer.pkl"
  # incremental_timesteps: 15000  # warm-start 時の追加ステップ数
```

### §4.5 Replay Buffer サイズ管理

| パラメータ | 値 | 根拠 |
|---|---|---|
| buffer_size | 100,000 | g2_sac_train.yaml 設定値 |
| 1 sample のメモリ | ~500 bytes (obs=12dim + action=1dim + reward + next_obs + done) | 推定値 |
| 満杯時のディスク容量 | ~50 MB (pickle) | 100K × 500B |
| warm-start 後の混合比 | 旧データ 100K + 新データ 10-15K → 自動上書き (FIFO) | SB3 ReplayBuffer の標準動作 |

buffer_size=100K に対して total_timesteps=50K なので、初回訓練では buffer は半分のみ使用。warm-start で 10-15K 追加すると旧データの**一部が新データで置換**される (FIFO)。これは catastrophic forgetting 緩和に有利。

---

## §5 SAC Retrain Scheduler 設計

### §5.1 既存インフラとの関係

| 項目 | retrain_scheduler.py (SkipGate) | sac_retrain_scheduler.py (新規) |
|---|---|---|
| 対象モデル | LightGBM (SkipGate) | SB3 SAC |
| 再訓練データ | fill_records_*.jsonl | OHLCV 1m parquet (rolling window) |
| 再訓練頻度 | ~1 時間 (固定 interval + trigger) | 1-4 時間 (新規データ蓄積量ベース) |
| Gate 検証 | Walk-Forward OOS eval | val_env OOS eval (E1-E4 サブセット) |
| デプロイ | atomic model swap + hot-reload | atomic model swap + signal file 更新 |
| 起動方式 | fill_test_cli.py から子プロセス | 独立プロセス or fill_test_cli.py 拡張 |

### §5.2 スケジューラのメインループ

```python
# scripts/v460/ml/sac_retrain_scheduler.py (概念設計)

def run_sac_retrain():
    """SAC sidecar モデルの定期再訓練ループ."""

    cfg = load_config("configs/v460/experiments/g2_sac_train.yaml")
    model_path = Path("models/v460/sac_sidecar.zip")
    buffer_path = Path("models/v460/sac_sidecar.buffer.pkl")
    signal_path = Path("cache/sidecar_signal.json")

    while not shutdown_event.is_set():
        # 1. トリガー判定: 新規 OHLCV データの蓄積量チェック
        if not should_retrain(cfg):
            shutdown_event.wait(timeout=300)  # 5分ごとにチェック
            continue

        # 2. データ準備: rolling window (直近 7 日分)
        df = load_rolling_ohlcv(days=7)

        # 3. Warm-start 訓練
        env = create_training_env(df, cfg)
        if model_path.exists():
            model = SAC.load(str(model_path), env=env)
            if buffer_path.exists():
                model.load_replay_buffer(str(buffer_path))
            incremental_steps = cfg.get("training", {}).get(
                "incremental_timesteps", 15000
            )
        else:
            model = create_sac_model(env, cfg)
            incremental_steps = cfg["training"]["total_timesteps"]

        model.learn(total_timesteps=incremental_steps)

        # 4. OOS 検証
        val_env = create_val_env(df, cfg)
        metrics = evaluate_model(model, val_env)

        if metrics["gross_roi"] > 0:  # 最低条件: 正の ROI
            # 5. Atomic デプロイ
            tmp_path = model_path.with_suffix(".tmp.zip")
            model.save(str(tmp_path))
            model.save_replay_buffer(str(buffer_path))
            tmp_path.rename(model_path)  # atomic rename

            # 6. Sidecar signal 更新
            write_sidecar_signal(signal_path, model, env)
        else:
            log.warning("OOS validation failed, keeping previous model")

        # 7. 次回まで待機
        shutdown_event.wait(timeout=effective_interval)
```

### §5.3 Sidecar Signal ファイルフォーマット

```json
{
  "timestamp": "2026-03-10T12:00:00+09:00",
  "model_version": "sac_sidecar_v460_20260310_1200",
  "directional_bias": 0.42,
  "confidence": 0.78,
  "regime_hint": "trending_up",
  "features_snapshot": {
    "price_velocity": 0.0023,
    "micro_trend": 0.15,
    "momentum_burst": 0.08
  },
  "training_metrics": {
    "gross_roi": 0.032,
    "total_timesteps": 65000,
    "buffer_utilization": 0.65
  }
}
```

fill_test 側は `sidecar_signal.json` を atomic read し、`directional_bias` のみを使用。他のフィールドは診断用。

---

## §6 retrain_scheduler (SkipGate) との区別

### §6.1 「1時間毎に再学習」の正体

現在「1時間毎に再学習」で動いているのは **retrain_scheduler.py = SkipGate (LightGBM)** の定期再学習である。SAC の定期再学習は**未実装**。

| 項目 | SkipGate retrain (既存) | SAC retrain (未実装→本文書で設計) |
|---|---|---|
| ファイル | `scripts/v460/ml/retrain_scheduler.py` (2131行) | `scripts/v460/ml/sac_retrain_scheduler.py` (新規) |
| 呼出元 | `fill_test_cli.py` L293 子プロセス | 独立プロセス (Phase 1) |
| 入力データ | `fill_records_*.jsonl` | OHLCV 1m parquet |
| モデル形式 | LightGBM pickle | SB3 SAC zip + replay buffer pkl |
| hot-reload | SkipGateEvaluator が mtime 監視で auto-load | sidecar_signal.json の atomic read |
| 計算量 | 軽い (~10秒) | 重い (~5-10分 for 15K steps) |

### §6.2 並行運用の設計

```
fill_test_cli.py 起動
├── fill_test 本体 (PID xxx)
├── retrain_scheduler.py (SkipGate, 1時間毎, PID yyy)
└── sac_retrain_scheduler.py (SAC sidecar, 2-4時間毎, PID zzz)  ← 新規
```

- SkipGate と SAC は**完全独立**のプロセスとして並行動作
- CPU 競合を避けるため、SAC retrain は SkipGate retrain と**時間をずらす** (offset 30分等)
- SAC retrain は GPU が利用可能なら GPU を使用 (MLP policy なので CPU でも十分高速)

---

## §7 実装優先順位

| 優先度 | タスク | 工数 | 依存 | 効果 |
|---|---|---|---|---|
| **P1** | replay buffer 永続化 (sac_algorithm.py + sac_train.py) | 2-4h | なし | ✅ **完了** |
| **P2** | g2_sac_train.yaml に明示的無効化コメント追加 | 0.5h | なし | ✅ **完了** |
| **P3** | SAC 出力を directional_bias として定義 (interface 設計) | 2-4h | なし | ✅ **完了** — `sidecar_types.py` |
| **P4** | sidecar_signal.json writer/reader 実装 | 4-8h | P3 | ✅ **完了** — `sidecar_signal_io.py` |
| **P5** | cycle_gate_aggregator.py への sidecar 注入 | 4-8h | P4 | ✅ **完了** — `_apply_sidecar_offset()` |
| **P6** | sac_retrain_scheduler.py 新規作成 | 1-2d | P1, P3 | ⏳ 未着手 |
| **P7** | action_masks の observation 埋め込み | 2-4h | なし | ⏳ 未着手 (低優先) |
| **P8** | LiteTradingEnv (連続行動空間) 設計 | 1-2d | なし | ⏳ 未着手 (ph4) |

### クリティカルパス

```
P1 → P6 (warm-start → retrain scheduler)
P3 → P4 → P5 (interface → signal → fill_test 統合)
```

P1 と P3 は独立なので並行着手可能。

---

## §8 356# との関係

本文書は 356# の**補完文書**として位置付ける。

| 356# の範囲 | 本文書の追加 |
|---|---|
| G2-train Gate 突破の実行計画 | SAC の長期運用設計 (warm-start + retrain) |
| B1-B5 ブロッカー解消 | env 阻害要因の体系的分析 |
| FeatureRegistry 特徴量選定 | OHLCV 限定の構造的正当化 |
| multi-seed 訓練設計 | sidecar 統合設計 |
| — | HeavyTradingEnv 15 機構の SAC 干渉分析 |
| — | replay buffer 永続化設計 |
| — | sac_retrain_scheduler 設計 |

356# は「G2 Gate 突破」、365# は「G2 通過後の運用基盤」をそれぞれカバーする。

---

## §9 リスク評価

### §9.1 Sidecar アプローチのリスク

| リスク | 深刻度 | 緩和策 |
|---|---|---|
| SAC の方向性予測が無価値 | High | sidecar_boost を ±0.1 bps に制限。NEUTRAL 時は影響ゼロ |
| 予測が逆方向 → fill_test の悪化 | High | bias の絶対値が閾値以下なら NEUTRAL 判定。confidence-weighted で影響制限 |
| sidecar_signal.json の読み書き競合 | Low | atomic write (tmp → rename) で排他制御 |
| SAC retrain の CPU 占有 | Medium | nice/priority 設定 + SkipGate retrain との時間オフセット |

### §9.2 replay buffer 永続化のリスク

| リスク | 深刻度 | 緩和策 |
|---|---|---|
| buffer pickle のファイルサイズ (最大50MB) | Low | ディスク容量は十分。定期的な old buffer の削除 |
| buffer 内の古い sample が学習を歪める | Medium | FIFO で自動上書き。buffer_size=100K で直近約 70 時間分 |
| buffer の obs 次元が変わった場合のロード失敗 | High | 特徴量変更時は buffer を破棄し cold-start |

### §9.3 残存する根本的課題

| 課題 | 本文書での評価 | 対応フェーズ |
|---|---|---|
| OHLCV のみで方向性 Alpha が取れるか | 「Sidecar では構造的に許容」だが未検証 | ph3 G2 gate で検証 |
| gamma=0.80 の妥当性 | v451 由来、v460 未検証 | G2 FAIL 時に 0.90/0.95 でグリッドサーチ |
| env の報酬関数が PnL 以外の成分を含む | "simple" ステージでも基本的 step_pnl + trade_pnl は有効 | §3.5 の注意事項を遵守 |

---

## §10 自己批判的レビュー

### §10.1 本設計の弱点

1. **Sidecar の boost 値 (±0.1 bps) が小さすぎて効果不明** — fill_test の offset は現在 ±数 bps のオーダー。0.1 bps の上乗せが統計的に有意な差を生むか疑問。
   - **対策**: 初期の boost 値はバックテストで調整。効果が不明なら 0.3-0.5 bps まで段階的に引き上げ。

2. **sac_retrain_scheduler の rolling window 7 日は根拠薄弱** — BTC/JPY の regime 遷移期間に対して 7 日は短すぎる可能性。逆に長すぎて古い regime のデータが混在する可能性もある。
   - **対策**: rolling window を config 化し、7/14/30 日でオフライン比較実験。

3. **replay buffer の FIFO 上書きで catastrophic forgetting が完全解消されるわけではない** — 全く新しい市場環境が来た場合、古い環境の知識は失われる。
   - **対策**: 定期的な full retrain (月 1 回等) + warm-start の併用。reservoir sampling も検討。

4. **ActionValidator の Ghost action 問題を「許容」としたが、本当に発生しないか未検証** — `max_position_size=0.01 BTC` と `initial_PV=1000万` で資金不足がほぼ起きないという推定は、連続損失シナリオで崩れる可能性。
   - **対策**: 訓練ログで action_masks の reject 率をモニタリングし、5% 超なら P7 (action_masks observation 埋め込み) を前倒し。

5. **SkipGate retrain と SAC retrain の CPU 競合** — 両プロセスが同時に走ると fill_test 本体のレイテンシに影響する可能性。
   - **対策**: mutex or time-offset で排他制御。Process affinity で CPU core を分離。

### §10.2 西洋的思考への固執リスク

- Sidecar パターンは Kubernetes/microservices 文脈の概念。金融 HFT では**co-located strategy** の方が適切な場合もある。
- 「分離して注入」が常に最適とは限らない。SAC を fill_test の内部ループに直接組み込む方が低レイテンシになる可能性。ただし、現時点では SAC の信頼度が低いため分離が妥当。

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-10 | 1.0 | 初版 (Sidecar 設計 + env 阻害分析 + warm-start + retrain scheduler) |
| 2026-03-15 | 1.1 | P1/P2 完了マーク, P3-P5 実装完了 (sidecar_types + signal_io + gate injection) |
