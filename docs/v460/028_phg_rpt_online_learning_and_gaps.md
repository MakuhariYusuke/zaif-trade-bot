# 028# Phase G — 取引中学習の方策検討 + コードベースギャップ分析

| key | value |
|-----|-------|
| type | rpt (調査レポート) |
| scope | cross-gate (ph2→ph5) |
| status | done |

---

## §1 背景

fill test (ph2) は新版コードで継続中。本レポートではまだ着手されていない「取引中の学習」について、既存実装の棚卸し・方策の整理・000# への位置づけ提案を行う。

併せてコードベース全体を走査し、v460 の Gate 進行に影響するギャップを網羅報告する。

---

## §2 既存「学習・適応」モジュールの棚卸し

### §2.1 モジュール一覧と稼働状況

| モジュール | パス | 状態 | v460 参照 |
|-----------|------|------|----------|
| OnlineLearningPipeline | `ztb/adaptation/online_learning/` | **DEAD** — circular import で disabled | なし |
| RetrainingTrigger | `ztb/adaptation/retraining/` | **DEAD** — 型定義のみ | なし |
| ConceptDriftDetector | `ztb/adaptation/concept_drift/` | **DEAD** — 設定のみ | なし |
| AB Testing | `ztb/adaptation/ab_testing/` | **DEAD** — 参照ゼロ | なし |
| Safety | `ztb/adaptation/safety/` | **DEAD** — 参照ゼロ | なし |
| Operations | `ztb/adaptation/operations/` | **DEAD** — 参照ゼロ | なし |
| Monitoring | `ztb/adaptation/monitoring/` | ALIVE (types のみ) | comprehensive_backtest |
| Explainability | `ztb/adaptation/explainability/` | ALIVE | sac_algorithm.py |
| ContinualLearning | `ztb/adaptation/continual_learning.py` | ALIVE | unified_trainer |
| MetaLearning | `ztb/adaptation/meta_learning.py` | ALIVE | unified_trainer |
| OnlineLearningEngine | `ztb/training/online_learning_engine.py` (808L) | **DEAD** — `enable_v433_adaptive=False`(デフォルト) | なし (V433 遺産) |

**結論**: 6 モジュール + V433 OnlineLearningEngine が完全に dead code。v460 scripts / live_trader からの参照はゼロ。

### §2.2 OnlineLearningPipeline の問題

1. **circular import で事実上 disabled**
   - `create_algorithm_trainer()` → `OnlineLearningSACTrainer` → `SACAlgorithmTrainer` → `EnsembleMixin` → `unified_trainer/__init__` → `trainer.py` → `create_algorithm_trainer()` → **循環**
   - `raise NotImplementedError("Online learning training temporarily disabled due to circular import issues")`

2. **SB3 SAC と完全に非互換**
   - `OnlineLearningPipeline` は raw `torch.nn.Module` 前提（`pipeline.py` L44）
   - 初期化時に `torch.nn.Linear(10, 1)` ダミーモデルを使用（`trainer.py` L43）
   - SB3 SAC の `ReplayBuffer` / `policy.actor` / `policy.critic` への接続なし
   - `load_online_model()` の `load_state_dict` 行がコメントアウト + `# TODO: algorithm統合`

3. **v459 時代の設計遺産** — v460 で使うには根本からの再設計が必要

### §2.3 Fine-tuning の既存実装

| 資産 | パス | 流用可能性 |
|------|------|-----------|
| `_set_fine_tune_learning_rate()` | `ztb/training/algorithms/sac/sac_algorithm.py` | **HIGH** — LR 変更ロジック |
| `run_fine_tuning.py` | `scripts/v452/` | **MEDIUM** — SAC.load → 低 LR 追加学習パターン |
| `checkpoint_manager.py` | `ztb/training/checkpoint/` | **HIGH** — replay buffer save/load 実装あり |
| `grad_probe_guard.py` | `ztb/training/utils/` | **LOW** — 緊急バッファ保存のみ |

---

## §3 取引中学習の方策

### §3.1 方策 A: パラメータ適応（最軽量）

**概要**: SAC モデル自体は変えず、fill_test 実測データからスプレッドオフセット・発注タイミングを自動調整。

| 項目 | 内容 |
|------|------|
| 入力 | fill_records (JSONL), FillMetrics |
| 出力 | FillTestConfig パラメータ更新 |
| 実装量 | ~100 行 |
| リスク | 極小 — モデル非接触 |
| Phase | **ph2–ph5** で即時適用可 |

**例**: fill_rate < 80% → `spread_offset_ratio` を 0.2 → 0.3 に増加（板の内側へ寄せて約定率向上）。AS_ratio > 15% → `spread_offset_ratio` を縮小（板の外側に退避して逆選択を回避）。

### §3.2 方策 B: 定期バッチ再訓練（中量）

**概要**: 日次/週次で直近データを用いて SAC を fine-tune (warm start)。

| 項目 | 内容 |
|------|------|
| 入力 | 直近 N 日分の OHLCV + マイクロストラクチャ特徴量 |
| 出力 | 新モデル (.zip) |
| 手順 | (1) SAC.load → (2) 低 LR で追加学習 → (3) G3 指標検証 → (4) model hot-swap |
| 流用 | `_set_fine_tune_learning_rate()`, `checkpoint_manager`, `run_fine_tuning.py` パターン |
| 実装量 | ~300 行 + model hot-swap 機構 |
| リスク | 中 — catastrophic forgetting 防止策が必要 (frozen layers, EWC) |
| Phase | **ph5 (G4-live)** |

### §3.3 方策 C: オンライン学習（重量, v461+）

**概要**: 取引ループ内でリアルタイムにモデルを更新。

| 項目 | 内容 |
|------|------|
| 前提 | SB3 SAC ⇔ OnlineLearningPipeline ブリッジ、circular import 解消 |
| 実装量 | ~1000 行 + 根本リファクタリング |
| リスク | 高 — 不安定性, catastrophic forgetting, 未検証コード多数 |
| Phase | **v461 以降** — v460 scope 外 |

### §3.4 推奨

**v460 では方策 A + B を採用。方策 C は v461 課題。**

---

## §4 000# への位置づけ提案

### §4.1 §3.6 G4-live への追記案

現在の §3.6 に以下を追加:

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| パラメータ適応 | fill_rate/AS_ratio の移動平均を監視し offset 自動調整が稼働中 | FillMetrics 日次集計 → ConfigUpdate ログ (JSONL) |
| 定期再訓練 | 日次 fine-tune が G3 指標を維持 | 新旧モデルの G3 比較 |
| Model hot-swap | 新モデルへの無停止切替が動作確認済 | 手動テスト |

### §4.2 §6 リスクへの追記案

| 重要度 | リスク | 緩和策 |
|--------|--------|--------|
| ⭐⭐ | 市場レジーム変化によるモデル陳腐化 | 定期再訓練 (方策 B) + G3 指標モニタリング。陳腐化検知は fill_rate/AS_ratio の 7 日移動平均で監視 |

---

## §5 コードベースギャップ一覧

### F1: [CRITICAL] live_trader にマイクロストラクチャ特徴量パスなし

- **箇所**: `ztb/trading/live_trader/feature_computation.py`
- **問題**: ライブ推論時の特徴量計算が OHLCV のみ (open=high=low=close のドージ + volume=1000 のモック値)。v460 必須のマイクロストラクチャ カラム (`best_bid`, `best_ask`, `spread`, `depth_imbalance`, `trade_flow_imbalance` 等) が完全欠損
- **影響**: **学習時と推論時で完全に異なる特徴量を使う**致命的乖離。v460 core proposition の否定
- **ブロッカー**: ph5 (G4-live)
- **対策**: `FeatureComputation` にリアルタイム板情報取得ロジック追加 + `add_microstructure_features()` のライブ呼び出しパス実装

### F2: [CRITICAL] SB3 SAC ⇔ OnlineLearning ブリッジ断絶

- **箇所**: `ztb/adaptation/online_learning/pipeline.py`, `trainer.py`
- **問題**: §2.2 に記載。raw torch.nn 前提、ダミーモデル、ReplayBuffer 非統合
- **ブロッカー**: **No (v460)** — 方策 C (オンライン学習) は v461 scope 外。方策 B (fine-tune) では SB3 標準パスを使うため本モジュール不要
- **対策**: v460 では方策 B を採用し、本モジュールには手を入れない

### F3: [HIGH] Model hot-reload 機構なし

- **箇所**: `ztb/trading/live_trader/model_manager.py`
- **問題**: `load_model()` は起動時 1 回のみ。ファイルウォッチャー、定期チェック、hot-swap 一切なし
- **ブロッカー**: ph5 (G4-live) — 7 日連続稼働中にモデル更新不可
- **対策**: `ModelManager.check_model_update(interval_sec=300)` 追加。ファイル mtime/SHA256 比較 → 不一致なら reload

### F4: [HIGH] 潜在的循環依存 (online_learning / unified_trainer)

- **箇所**: `ztb/training/unified_trainer/algorithms/__init__.py` L70-95
- **構造**: `algorithms` → `OnlineLearningSACTrainer` → `SACAlgorithmTrainer` → `EnsembleMixin` → `unified_trainer/__init__` → `trainer.py` → `algorithms` → **潜在的循環**
- **現状**: import文がコメントアウトされ `raise NotImplementedError(...)` で即座にガード。Python の import システムは発動しないため、現時点ではアクティブな循環ではない
- **ブロッカー**: ph3+ (SAC + online_learning 統合を解禁する場合に顕在化)
- **対策**: `sac_trainer.py` の `EnsembleMixin` import を遅延 import に変更

### F5: [MEDIUM] Dead code 6 モジュール

- **箇所**: `ztb/adaptation/` 配下の online_learning, retraining, concept_drift, ab_testing, safety, operations
- **問題**: v460 からの参照ゼロ。保守コスト・混乱リスク
- **対策**: ph3 以降で復活判断。不要なら `archived/` へ

### F6: [MEDIUM] Replay buffer が v460 スクリプト未統合

- **箇所**: `scripts/v460/` 全般
- **問題**: `checkpoint_manager.py` に save/load 実装はあるが、v460 トレーニングスクリプトからの参照なし
- **ブロッカー**: ph3 (学習再開時にバッファ消失リスク)

### F7: [LOW] Gate G2/G3/G4 未実装

- **箇所**: `scripts/v460/run_gate_check.py`
- **問題**: G0/G1/G1.1 の 3/6 のみ実装。G2/G3/G4 のスタブなし
- **対策**: 各 Phase 開始前にスタブ + 閾値定義を追加

---

## §6 優先順位マトリクス

| 優先度 | 項目 | 対象 Phase | 実装時期 |
|--------|------|-----------|---------|
| **P0** | 方策 A: パラメータ適応 | ph2 (現在) | **即時** |
| **P1** | F1: live_trader 特徴量パス | ph5 | ph3 設計時に着手 |
| **P1** | F3: Model hot-reload | ph5 | ph5 開始前 |
| **P2** | 方策 B: 定期再訓練 | ph5 | ph4 完了後 |
| **P2** | F4: Circular import 解消 | ph3 | ph3 開始時 |
| **P3** | F7: Gate G2/G3/G4 スタブ | ph3/4/5 | 各 Phase 開始前 |
| **P4** | F5: Dead code 整理 | — | 余裕時 |
| **Scope外** | 方策 C: オンライン学習 | v461 | — |

---

## §7 000# 改訂事項 (提案)

| §番号 | 改訂内容 |
|-------|---------|
| §3.6 G4-live | パラメータ適応・定期再訓練・model hot-swap の条件を追加 |
| §6 リスク | 「市場レジーム変化によるモデル陳腐化」リスクを追加 |
| §2 Phase 定義 | ph5 の「成果物」に「パラメータ適応ログ, 再訓練パイプライン」を追加 |

**改訂は外部レビュー後に実施する。**
