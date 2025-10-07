# 補足機能実装ガイド - Gradient Probe Guard & Enhanced Ensemble

**実装日**: 2025年10月7日  
**機能**: grad_probesゼロ張り付き検出 + アンサンブル強化

---

## 📋 目次

1. [概要](#概要)
2. [Gradient Probe Guard](#gradient-probe-guard)
3. [Enhanced Ensemble Aggregator](#enhanced-ensemble-aggregator)
4. [設定方法](#設定方法)
5. [使用例](#使用例)
6. [トラブルシューティング](#トラブルシューティング)

---

## 概要

本ドキュメントは、2つの重要な補足機能の実装ガイドです:

1. **Gradient Probe Guard**: grad_probesのゼロ張り付き（特にSELL）を検出し、自動停止&診断データアーカイブ
2. **Enhanced Ensemble Aggregator**: confidence-weighted voting、失格モデル自動除外

---

## Gradient Probe Guard

### 機能説明

Gradient Probe Guardは、学習中にgrad_probes（特にSELL行動の勾配）がゼロに張り付く問題を検出し、自動的に学習を停止します。停止時には以下のデータを保存します:

- **Replay Buffer**: リプレイバッファの状態
- **Model State**: モデルの重みとパラメータ
- **Diagnostics**: 勾配履歴、統計情報
- **Manifest**: 停止理由、設定、メトリクス
- **TensorBoard Events**: TensorBoardログ

### アーキテクチャ

```
CompositeTrainingCallback
  └─ GradProbeGuard
       ├─ _on_step() ← 毎ステップ勾配チェック
       ├─ _extract_grad_stats() ← 勾配抽出
       ├─ _check_zero_gradients() ← ゼロ検出
       └─ _handle_zero_gradient_halt() ← 停止&アーカイブ
```

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `zero_threshold` | `1e-8` | ゼロと見なす閾値 |
| `consecutive_zeros` | `5` | 連続ゼロ回数（これを超えると停止） |
| `check_interval` | `1000` | チェック間隔（ステップ数） |
| `monitor_actions` | `["SELL", "BUY", "HOLD"]` | 監視する行動 |
| `critical_actions` | `["SELL"]` | 停止トリガーとなる行動 |
| `save_replay_buffer` | `true` | リプレイバッファを保存するか |
| `save_model_state` | `true` | モデルを保存するか |
| `save_diagnostics` | `true` | 診断データを保存するか |
| `archive_dir` | `"grad_probe_archives"` | アーカイブ保存先 |

### 検出ロジック

```python
# 疑似コード
for each step:
    if step % check_interval == 0:
        grad_stats = extract_gradient_norms()
        
        for action in critical_actions:
            if grad_norm[action] < zero_threshold:
                consecutive_zeros[action] += 1
            else:
                consecutive_zeros[action] = 0
            
            if consecutive_zeros[action] >= threshold:
                # 停止トリガー
                halt_training()
                save_archive()
                return False
```

### アーカイブ構造

停止時に以下の構造でアーカイブが作成されます:

```
grad_probe_archives/
└── grad_zero_session_20251007_120000/
    ├── manifest.json                    # 停止理由、設定、メトリクス
    ├── model/
    │   └── model.zip                    # モデルの状態
    ├── replay_buffer/
    │   └── replay_buffer.pkl            # リプレイバッファ
    ├── diagnostics/
    │   ├── gradient_history.json        # 勾配履歴（最大1000ステップ）
    │   └── final_stats.json             # 最終統計
    └── tensorboard_events/
        └── events.out.tfevents.*        # TensorBoardログ
```

### manifest.json の例

```json
{
  "session_id": "ensemble_B_1M",
  "timestamp": "20251007_120000",
  "halt_reason": "SELL action gradient stuck at zero for 5 checks (step 150000)",
  "halt_step": 150000,
  "config": {
    "zero_threshold": 1e-8,
    "consecutive_zeros": 5,
    "check_interval": 1000,
    "monitor_actions": ["SELL", "BUY", "HOLD"],
    "critical_actions": ["SELL"]
  },
  "final_stats": {
    "step": 150000,
    "action_grads": {
      "SELL": 0.0,
      "BUY": 0.0012,
      "HOLD": 0.0008
    },
    "grad_norms": {
      "SELL": 0.0,
      "BUY": 0.0012,
      "HOLD": 0.0008
    },
    "is_zero": {
      "SELL": true,
      "BUY": false,
      "HOLD": false
    },
    "consecutive_zero_count": {
      "SELL": 5,
      "BUY": 0,
      "HOLD": 0
    }
  },
  "history_length": 150
}
```

---

## Enhanced Ensemble Aggregator

### 機能説明

Enhanced Ensemble Aggregatorは、複数モデルの予測を高度に集計します。主な機能:

1. **Confidence-Weighted Voting**: Sharpe ratio × 信頼度で重み付け
2. **失格モデル自動検出**: all-masked多発やSharpe低下を自動検出し、weight=0化
3. **重みキャリブレーション**: 評価データで各モデルの性能を測定し、最適な重みを算出

### アーキテクチャ

```
EnsembleAggregator
  ├─ calibrate_weights() ← 重みキャリブレーション
  │    ├─ 各モデルのSharpe ratio計算
  │    ├─ 各モデルの信頼度計算
  │    ├─ all-masked率計算
  │    ├─ 失格判定
  │    └─ weight計算（Sharpe × confidence）
  └─ predict() ← アンサンブル予測
       └─ confidence_weighted ← 重み付き投票
```

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `method` | `"confidence_weighted"` | 集計方法 |
| `disqualification_threshold` | `0.5` | all-masked率の閾値（これ以上で失格） |
| `min_sharpe_threshold` | `-2.0` | 最低Sharpe ratio（これ以下で失格） |
| `use_confidence_scaling` | `true` | 信頼度スケーリングを使用するか |

### 失格条件

モデルは以下の条件で失格（weight=0）となります:

1. **Sharpe ratio が閾値以下**: `sharpe < min_sharpe_threshold` (デフォルト: -2.0)
2. **all-masked率が閾値以上**: `masked_rate >= disqualification_threshold` (デフォルト: 50%)

### all-masked検出ロジック

```python
# 各ステップで行動確率の標準偏差をチェック
action_probs = model.predict_proba(obs)
if np.std(action_probs) < 1e-6:
    # 全行動が同じ確率 = マスクされている
    masked_step_count += 1

# エピソード終了時
if masked_step_count / total_steps > 0.5:
    # エピソードの半分以上がmasked
    masked_episode_count += 1

# 全エピソード終了後
masked_rate = masked_episode_count / n_episodes
if masked_rate >= disqualification_threshold:
    # 失格
    model_weight = 0.0
```

### 重み計算ロジック

```python
# 1. 各モデルの性能評価
for model in models:
    sharpe = evaluate_sharpe(model, eval_env)
    confidence = evaluate_confidence(model, eval_env)
    masked_rate = evaluate_masked_rate(model, eval_env)
    
    # 2. 失格判定
    if sharpe < min_sharpe_threshold or masked_rate >= disqualification_threshold:
        weight = 0.0
    else:
        # 3. 重み計算
        if use_confidence_scaling:
            weight = max(sharpe, 0.0) * confidence
        else:
            weight = max(sharpe, 0.0)

# 4. 正規化
total = sum(weights)
if total > 0:
    normalized_weights = [w / total for w in weights]
else:
    # 全失格の場合は均等重み（警告表示）
    normalized_weights = [1.0 / n_models] * n_models
```

### キャリブレーション出力例

```
🔧 Calibrating model weights with 50 episodes...
   - Confidence scaling: True
   - Disqualification threshold (all-masked): 50.0%
   - Min Sharpe threshold: -2.0

  Model 1 (ensemble_A_1M):
    Sharpe: 4.5000
    Confidence: 0.8500
    All-masked rate: 10.00%

  Model 2 (ensemble_B_1M):
    Sharpe: 5.2000
    Confidence: 0.9000
    All-masked rate: 15.00%

  Model 3 (ensemble_C_1M):
    Sharpe: -2.5000
    Confidence: 0.7000
    All-masked rate: 5.00%

  ⚠️  Model 3 (ensemble_C_1M) DISQUALIFIED: Sharpe -2.5000 < -2.0

🚫 1 model(s) disqualified

📊 Calibrated weights:
  ✅ Model 1 (ensemble_A_1M): 0.4500
  ✅ Model 2 (ensemble_B_1M): 0.5500
  ❌ DISQUALIFIED Model 3 (ensemble_C_1M): 0.0000
```

---

## 設定方法

### 設定ファイル（template.json / template_2M.json）

#### Gradient Probe Guard設定

```json
{
  "enable_grad_probe_guard": false,
  "grad_probe_config": {
    "zero_threshold": 1e-8,
    "consecutive_zeros": 5,
    "check_interval": 1000,
    "monitor_actions": ["SELL", "BUY", "HOLD"],
    "critical_actions": ["SELL"],
    "save_replay_buffer": true,
    "save_model_state": true,
    "save_diagnostics": true,
    "archive_dir": "grad_probe_archives",
    "save_tensorboard_events": true,
    "save_environment_state": true
  }
}
```

#### 推奨設定

- **100kテスト**: `enable_grad_probe_guard: false` (短時間なので不要)
- **1M学習**: `enable_grad_probe_guard: false` (オプション)
- **2M学習**: `enable_grad_probe_guard: true` (推奨、`check_interval: 5000`)

### アンサンブル設定

アンサンブルツールの引数で設定:

```bash
python scripts/ensemble_aggregator.py \
    --model-dirs checkpoints/ensemble_*/checkpoint_1000000 \
    --method confidence_weighted \
    --eval-data ml-dataset-enhanced.csv \
    --calibrate \
    --n-eval 100
```

---

## 使用例

### 例1: 2M学習でGrad Probe Guardを有効化

```bash
# 1. 設定ファイル作成
cp configs/train/template_2M.json configs/train/my_model_2M.json

# 2. enable_grad_probe_guard を true に設定
# （template_2M.jsonはデフォルトでtrue）

# 3. 学習実行
python -m ztb.training.unified_trainer \
    --config configs/train/my_model_2M.json \
    --log-level INFO
```

停止時の出力例:

```
ZERO GRADIENT DETECTED: SELL action has zero gradients for 5 consecutive checks (threshold: 5)
🛑 TRAINING HALTED: SELL action gradient stuck at zero for 5 checks (step 150000)
📦 Saving diagnostics to: grad_probe_archives/grad_zero_my_model_2M_20251007_120000
✅ Manifest saved: grad_probe_archives/grad_zero_my_model_2M_20251007_120000/manifest.json
✅ Gradient history saved: grad_probe_archives/grad_zero_my_model_2M_20251007_120000/diagnostics/gradient_history.json
✅ Final stats saved: grad_probe_archives/grad_zero_my_model_2M_20251007_120000/diagnostics/final_stats.json
✅ Model saved: grad_probe_archives/grad_zero_my_model_2M_20251007_120000/model/model.zip
✅ TensorBoard events saved: grad_probe_archives/grad_zero_my_model_2M_20251007_120000/tensorboard_events
📦 Archive complete: grad_probe_archives/grad_zero_my_model_2M_20251007_120000
🛑 Training halted due to: SELL action gradient stuck at zero for 5 checks (step 150000)
```

### 例2: アンサンブル集計（信頼度スケーリング + 失格検出）

```bash
# 重みキャリブレーション + 評価
python scripts/ensemble_aggregator.py \
    --model-dirs \
        checkpoints/ensemble_A_1M/checkpoint_1000000 \
        checkpoints/ensemble_B_1M/checkpoint_1000000 \
        checkpoints/ensemble_C_1M/checkpoint_1000000 \
    --method confidence_weighted \
    --eval-data ml-dataset-enhanced.csv \
    --calibrate \
    --n-eval 100 \
    --output ensemble_1M_results.json
```

出力例:

```
📦 Loading 3 models...
  ✅ Loaded: ensemble_A_1M
  ✅ Loaded: ensemble_B_1M
  ✅ Loaded: ensemble_C_1M
✅ Loaded 3/3 models

🔧 Calibrating model weights with 50 episodes...
   - Confidence scaling: True
   - Disqualification threshold (all-masked): 50.0%
   - Min Sharpe threshold: -2.0

  Model 1 (ensemble_A_1M):
    Sharpe: 4.5000
    Confidence: 0.8500
    All-masked rate: 10.00%

  Model 2 (ensemble_B_1M):
    Sharpe: 5.2000
    Confidence: 0.9000
    All-masked rate: 55.00%

  ⚠️  Model 2 (ensemble_B_1M) DISQUALIFIED: All-masked 55.0% >= 50.0%

  Model 3 (ensemble_C_1M):
    Sharpe: 3.8000
    Confidence: 0.7500
    All-masked rate: 20.00%

🚫 1 model(s) disqualified

📊 Calibrated weights:
  ✅ Model 1 (ensemble_A_1M): 0.5700
  ❌ DISQUALIFIED Model 2 (ensemble_B_1M): 0.0000
  ✅ Model 3 (ensemble_C_1M): 0.4300

📊 Evaluating ensemble with 100 episodes...
  Episode 10/100 completed
  ...
  Episode 100/100 completed

✅ Evaluation complete:
   Mean reward: 285.50 ± 42.30
   Sharpe ratio: 6.7500
   Action distribution: BUY=32.00%, HOLD=48.00%, SELL=20.00%

✅ Results saved to ensemble_1M_results.json
```

### 例3: アーカイブの分析

```bash
# 1. アーカイブディレクトリを確認
ls grad_probe_archives/

# 2. manifestを確認
cat grad_probe_archives/grad_zero_my_model_2M_20251007_120000/manifest.json | jq .

# 3. 勾配履歴を可視化（Pythonスクリプト）
python scripts/analyze_gradient_archive.py \
    --archive-dir grad_probe_archives/grad_zero_my_model_2M_20251007_120000
```

---

## トラブルシューティング

### Grad Probe Guard関連

#### 問題1: 勾配がゼロにならないのに停止する

**原因**: `zero_threshold`が高すぎる

**解決策**:
```json
{
  "grad_probe_config": {
    "zero_threshold": 1e-10  // より厳しい閾値
  }
}
```

#### 問題2: すぐに停止してしまう

**原因**: `consecutive_zeros`が少なすぎる

**解決策**:
```json
{
  "grad_probe_config": {
    "consecutive_zeros": 10  // より多くの連続ゼロを許容
  }
}
```

#### 問題3: アーカイブが大きすぎる

**原因**: TensorBoardイベントが大量

**解決策**:
```json
{
  "grad_probe_config": {
    "save_tensorboard_events": false  // TensorBoardイベント保存を無効化
  }
}
```

### アンサンブル関連

#### 問題4: 全モデルが失格になる

**原因**: 閾値が厳しすぎる

**解決策**:
```python
aggregator = EnsembleAggregator(
    model_paths=model_paths,
    disqualification_threshold=0.7,  # 70%に緩和
    min_sharpe_threshold=-3.0,  // -3.0に緩和
)
```

#### 問題5: all-masked検出が敏感すぎる

**原因**: 標準偏差の閾値が高すぎる

**解決策**: `ensemble_aggregator.py`の`calibrate_weights`内で調整:
```python
# 行110付近
if np.std(action_probs) < 1e-4:  # 1e-6 → 1e-4に緩和
    masked_step_count += 1
```

#### 問題6: 信頼度スケーリングで重みが偏る

**原因**: 信頼度の影響が大きすぎる

**解決策**:
```bash
# 信頼度スケーリングを無効化
python scripts/ensemble_aggregator.py \
    --model-dirs checkpoints/ensemble_*/checkpoint_1000000 \
    --method confidence_weighted \
    --eval-data ml-dataset-enhanced.csv \
    --calibrate \
    --n-eval 100
```

内部的に`use_confidence_scaling=False`を設定する場合は、`calibrate_weights`の引数を変更:
```python
aggregator.calibrate_weights(eval_env, n_episodes=50, use_confidence_scaling=False)
```

---

## ベストプラクティス

### Grad Probe Guard

1. **100kテスト**: 無効化（短時間なので不要）
2. **1M学習**: オプション（問題が出たら有効化）
3. **2M学習**: 有効化推奨（`check_interval: 5000`）
4. **並列実行時**: 各セッションで個別にアーカイブ作成

### アンサンブル

1. **モデル数**: 3-5モデルが最適（2モデルは不安定、6+は冗長）
2. **キャリブレーション**: 必ず実施（`--calibrate`）
3. **評価エピソード数**: 100エピソード推奨（50は最小、200は過剰）
4. **信頼度スケーリング**: デフォルトで有効、問題があれば無効化
5. **失格閾値**: デフォルト（masked=50%, Sharpe=-2.0）から開始し、必要に応じて調整

---

## まとめ

本実装により、以下が可能になりました:

✅ **Gradient Probe Guard**:
- SELL勾配のゼロ張り付きを自動検出
- 学習停止 + 完全な診断データアーカイブ
- 問題の早期発見とデバッグ時間短縮

✅ **Enhanced Ensemble Aggregator**:
- Sharpe × 信頼度による最適重み付け
- 失格モデル（all-masked、低Sharpe）の自動除外
- より堅牢なアンサンブル予測

これらの機能により、長時間学習の安定性とアンサンブルモデルの品質が大幅に向上します。

---

**次のステップ**: COMPREHENSIVE_TRAINING_GUIDE.mdに統合し、100k→1M→2M学習ワークフローを実行してください。
