# v456 Implementation Checklist: 実装チェックリスト

> **Version**: v456.1 (Revised after External Review)  
> **Date**: 2026-01-13  
> **Status**: Draft (Revised)

---

## ⚠️ 外部レビュー追加チェック項目 (CRITICAL)

### 0.1. データ整合性チェック（最優先）
- [ ] **MTFリサンプリング リーク検出テスト**
  - [ ] `tests/unit/features/test_mtf_no_future_leak.py` 作成
  - [ ] 10:07:00時点で10:05バー使用を検証
  - [ ] 10:00:00時点（境界）でのテスト
  - [ ] バックテストとライブの挙動一致確認

- [ ] **正規化パイプライン分離確認**
  - [ ] OnlineScaler対象グループ: `base_features`, `global_continuous` のみ
  - [ ] 除外グループ確認: `cyclical_time`, `regime_onehot`, `mtf_categorical`
  - [ ] テスト: One-Hot特徴量がスケーリングされていないこと

- [ ] **タイムゾーン一貫性テスト**
  - [ ] Naive timestamp入力時のエラー/警告確認
  - [ ] バックテストデータのtz-aware化確認
  - [ ] ライブデータのtz処理確認

### 0.2. Train-Live Parity チェック
- [ ] **Soft Filter環境内統合確認**
  - [ ] `FastIntradayEnvV456.step()` 内でフィルタ適用
  - [ ] 学習時と推論時で同一ロジック使用
  - [ ] `info["gated_action"]` のログ出力確認

- [ ] **Calibration Gate環境内統合確認**
  - [ ] EV判定が`step()`内で実行される
  - [ ] 外部でのフィルタリングがないこと

### 0.3. 報酬シェーピング キャリブレーション
- [ ] **シェーピング比率検証**
  - [ ] サンプル100エピソードで検証
  - [ ] `avg_shaping / avg_pnl < 0.5` を確認
  - [ ] 違反時: 係数自動調整または手動修正

### 0.4. バックテスト分割・データリーク対策
> **⚠️ 第2次レビュー追加**: 時系列データでのTrain/Validation分割でリーク防止が不十分だと
> 本番でパフォーマンス大幅劣化のリスクあり

- [ ] **Embargo期間の設定**
  - [ ] Train/Validation境界に embargo 期間を設定（推奨: 24h～48h）
  - [ ] 境界前後のデータが相互に影響しないことを確認
  ```python
  BACKTEST_SPLIT_CONFIG = {
      "train_end": "2025-10-31T23:59:59Z",
      "embargo_days": 2,  # 2日間のデータを除外
      "validation_start": "2025-11-03T00:00:00Z",
  }
  ```

- [ ] **Purged K-Fold CV（推奨）**
  - [ ] 時系列データ用のPurged CV実装
  - [ ] 各Foldで前後のembargo期間を除外
  - [ ] テスト: 同一データが複数Foldに出現しないこと
  ```python
  def purged_kfold_split(df, n_splits=5, embargo_hours=48):
      """
      Purged K-Fold for time-series data
      
      各fold境界の前後embargo_hours分のデータを除外し、
      時間的リークを防止
      """
      ...
  ```

- [ ] **Forward Chaining Validation（代替案）**
  - [ ] 時系列順でのTrain拡張型CV
  - [ ] 未来データへのリーク完全防止
  
- [ ] **検証テスト**
  - [ ] Train期間のデータがValidation期間に影響しないことを確認
  - [ ] Validation/Test分割も同様にembargo設定

---

## 概要

このドキュメントは、v456の実装における全タスクとチェックポイントを管理します。
各フェーズの完了条件と検証項目を明確化し、品質を担保します。

---

## Phase 1: 基盤構築 (Week 1-2)

### 1.1. Cyclical Time Features
- [ ] **実装**
  - [ ] `ztb/features/time/cyclical.py` 作成
  - [ ] `calc_cyclical_time_features()` 関数実装
  - [ ] 6特徴量（hour_sin/cos, minute_sin/cos, dow_sin/cos）
  - [ ] ⚠️ `validate_and_convert_timestamp()` でTZ検証
  
- [ ] **テスト**
  - [ ] 単体テスト `tests/unit/features/test_cyclical_time.py`
  - [ ] 値域チェック: 全て[-1, +1]の範囲内
  - [ ] 周期性チェック: 23:59と00:00が近い値
  - [ ] ⚠️ Naive timestamp拒否テスト
  
- [ ] **統合**
  - [ ] `FastIntradayEnv`への組み込み
  - [ ] Observation Space次元の更新
  - [ ] ⚠️ OnlineScaler対象外に設定

### 1.2. MTF Features Pipeline
- [ ] **実装**
  - [ ] `ztb/features/generators/multi_timeframe/v456_engine.py` 作成
  - [ ] 5m/15m/1h データリサンプリング
  - [ ] 各タイムフレームで9特徴量 × 3 = 27特徴量
  - [ ] ⚠️ `get_mtf_closed_bar()` でクローズドバーのみ使用
  
- [ ] **テスト**
  - [ ] リサンプリング精度テスト
  - [ ] タイムスタンプアライメントテスト
  - [ ] NaN/Inf検出テスト
  - [ ] ⚠️ **未来データリーク検出テスト** ← 最重要
  
- [ ] **検証**
  - [ ] サンプルデータでの特徴量可視化
  - [ ] EMA方向の正確性確認

### 1.3. Global Market Features
- [ ] **実装**
  - [ ] `ztb/features/global_market.py` 拡張
  - [ ] `GlobalMarketFeatureEngineerV456` クラス
  - [ ] Lead-Lag特徴量6種
  - [ ] ⚠️ FX調整スプレッド (`global_fx_adjusted_spread`)
  - [ ] ⚠️ データ鮮度フラグ (`global_data_stale_flag`)
  
- [ ] **テスト**
  - [ ] データマージテスト（タイムスタンプずれ対応）
  - [ ] Forward-fill正常動作テスト
  - [ ] 相関計算精度テスト
  - [ ] ⚠️ FX未取得時のフォールバック動作テスト
  
- [ ] **検証**
  - [ ] Binance/Zaifデータでのバックテスト
  - [ ] Lead-Lag効果の統計的検証

---

## Phase 2: フィルタリング強化 (Week 3-4)

### 2.1. Soft Filter Implementation
- [ ] **実装**
  - [ ] `ztb/trading/filters/soft_filter.py` 作成
  - [ ] `SoftFilter` クラス
  - [ ] Time-based restriction
  - [ ] Regime-based restriction
  - [ ] ⚠️ **環境step()内部への統合** ← Train-Live Parity
  
- [ ] **設定**
  - [ ] `REGIME_CONSTRAINTS` 辞書定義
  - [ ] `TIME_RESTRICTION_CONFIG` 辞書定義
  
- [ ] **テスト**
  - [ ] 14:00/17:00/01:00でのポジション乗数確認
  - [ ] `high_volatility_ranging`での制限確認
  - [ ] 閾値調整の動作確認
  - [ ] ⚠️ 学習時・推論時の同一動作確認

### 2.2. Calibration Gate Integration
- [ ] **確認**
  - [ ] 既存`CalibrationGate`の動作確認
  - [ ] EV計算式の検証
  - [ ] コストモデルパラメータの確認
  
- [ ] **拡張**
  - [ ] MTF情報を考慮したEV計算
  - [ ] レジーム別の統計分離
  - [ ] ⚠️ **環境step()内部への統合**
  
- [ ] **テスト**
  - [ ] EV > 0 判定の正確性
  - [ ] Edge caseのハンドリング
  - [ ] ⚠️ ゲーティング前後のアクション差分ログ

### 2.3. MTF Trend Filter
- [ ] **実装**
  - [ ] 1h足トレンド逆行時のBUY/SELL禁止ロジック
  - [ ] 設定可能なフィルター強度
  
- [ ] **テスト**
  - [ ] フィルター適用時のアクション変化確認
  - [ ] 誤フィルタリング率の計測

---

## Phase 3: 統合シグナルシステム (Week 4-5)

### 3.1. Signal Fusion Engine
- [ ] **実装**
  - [ ] `ztb/trading/signal/fusion_engine.py` 作成
  - [ ] RL Action + Technical Signal融合
  - [ ] Filter Mode / Boost Mode切り替え
  
- [ ] **テスト**
  - [ ] 融合ロジックの単体テスト
  - [ ] モード切り替えテスト

### 3.2. Dynamic TP/SL
- [ ] **実装**
  - [ ] レジーム連動TP/SL設定
  - [ ] Trailing Stop対応
  
- [ ] **設定**
  ```python
  TP_SL_CONFIG = {
      "strong_trend": {"tp": None, "trailing_pct": 0.02},
      "high_volatility_ranging": {"tp": 0.013, "sl": 0.008},
      "default": {"tp": 0.01, "sl": 0.01},
  }
  ```

### 3.3. Entry System Integration
- [ ] **実装**
  - [ ] `IntegratedEntrySystem` クラス完成
  - [ ] 全フィルターの統合
  
- [ ] **テスト**
  - [ ] End-to-endエントリー判定テスト
  - [ ] パフォーマンステスト（レイテンシ計測）

---

## Phase 4: 報酬関数拡張 (Week 5)

### 4.1. MTF Alignment Bonus
- [ ] **実装**
  - [ ] `calc_mtf_alignment_bonus()` 関数
  - [ ] `compute_hft_reward()`への統合
  
- [ ] **テスト**
  - [ ] ボーナス計算の正確性
  - [ ] 値域[-0.2, +0.15]の確認

### 4.2. Balance Enforcement
- [ ] **実装**
  - [ ] `calc_balance_enforcement_penalty()` 関数
  - [ ] BUY/SELL比率トラッキング
  
- [ ] **テスト**
  - [ ] 偏り検出の正確性
  - [ ] ペナルティスケールの適切性

### 4.3. Reward Function Testing
- [ ] **テスト**
  - [ ] 単体テスト: 各項の計算
  - [ ] 統合テスト: 総報酬の挙動
  - [ ] 回帰テスト: v455との比較

---

## Phase 5: モデル構造進化 (Week 6)

### 5.1. GRU Policy Network
- [ ] **実装**
  - [ ] `ztb/models/gru_policy.py` 作成
  - [ ] `GRUPolicyNetwork` クラス
  - [ ] Hidden state管理
  
- [ ] **テスト**
  - [ ] Forward passの形状確認
  - [ ] Gradient flowの確認
  
- [ ] **統合**
  - [ ] SAC Trainerとの統合
  - [ ] Sequence batch処理

### 5.2. Curriculum Learning
- [ ] **実装**
  - [ ] `ztb/training/curriculum_v456.py` 作成
  - [ ] 4ステージ定義
  - [ ] 進行判定ロジック
  
- [ ] **テスト**
  - [ ] ステージ進行テスト
  - [ ] データフィルタリングテスト

---

## Phase 6: バックテスト検証 (Week 7)

### 6.1. バックテスト実行
- [ ] **準備**
  - [ ] テストデータの分割確認
  - [ ] 評価指標の実装確認
  
- [ ] **実行**
  - [ ] Training Setでの学習
  - [ ] Validation Setでのハイパラ調整
  - [ ] Test Setでの最終評価
  
- [ ] **結果記録**
  - [ ] 評価指標の記録
  - [ ] ベースライン（v455）との比較

### 6.2. Walk-Forward Analysis
- [ ] **実行**
  - [ ] 90日Train / 30日Test ウィンドウ
  - [ ] 複数期間での検証
  
- [ ] **分析**
  - [ ] 期間ごとのパフォーマンス変動
  - [ ] Overfitting検出

---

## Phase 7: 最終調整 (Week 8)

### 7.1. ハイパーパラメータ最終調整
- [ ] **Optuna実行**
  - [ ] 探索空間の定義
  - [ ] 100トライアル以上の探索
  
- [ ] **最適値の記録**
  - [ ] 報酬関数パラメータ
  - [ ] モデルパラメータ
  - [ ] フィルターパラメータ

### 7.2. ドキュメント整備
- [ ] **更新**
  - [ ] `00_improvement_proposal.md` 最終版
  - [ ] `01_technical_specification.md` 最終版
  - [ ] `02_feature_engineering_spec.md` 最終版
  - [ ] `04_final_results.md` 作成

### 7.3. コードレビュー
- [ ] **セルフレビュー**
  - [ ] コードスタイル確認
  - [ ] 型ヒント確認
  - [ ] docstring確認
  
- [ ] **ピアレビュー**
  - [ ] 外部レビュー依頼
  - [ ] フィードバック反映

---

## 品質ゲート

### Gate 1: Phase 2完了時
| 項目 | 基準 | 結果 |
|-----|-----|-----|
| 特徴量NaN率 | < 0.1% | [ ] |
| 特徴量Inf率 | 0% | [ ] |
| 単体テスト通過率 | 100% | [ ] |

### Gate 2: Phase 4完了時
| 項目 | 基準 | 結果 |
|-----|-----|-----|
| Validation Return | > -5% | [ ] |
| Training安定性 | 崩壊なし | [ ] |
| 統合テスト通過率 | 100% | [ ] |

### Gate 3: Phase 6完了時
| 項目 | 基準 | 結果 |
|-----|-----|-----|
| Test Return | > 0% | [ ] |
| Sharpe Ratio | > 0.5 | [ ] |
| Max Drawdown | < 15% | [ ] |

### Gate 4: 最終リリース前
| 項目 | 基準 | 結果 |
|-----|-----|-----|
| Test Return | > +5% | [ ] |
| Profit Factor | > 1.3 | [ ] |
| Walk-Forward一貫性 | 全期間プラス | [ ] |
| ペーパートレード | 24h安定稼働 | [ ] |

---

## リスク管理チェック

### 実装時のリスク
- [ ] メモリリークチェック（長時間実行）
- [ ] CPU/GPU使用率モニタリング
- [ ] 例外ハンドリングの網羅性

### 運用時のリスク
- [ ] サーキットブレーカー動作確認
- [ ] 日次損失リミット動作確認
- [ ] アラート通知設定

---

## 成果物一覧

### コード成果物
| ファイル | 説明 | ステータス |
|---------|------|----------|
| `ztb/features/time/cyclical.py` | 時刻特徴量 | [ ] |
| `ztb/features/generators/multi_timeframe/v456_engine.py` | MTF特徴量 | [ ] |
| `ztb/trading/filters/soft_filter.py` | Soft Filter | [ ] |
| `ztb/trading/signal/fusion_engine.py` | シグナル融合 | [ ] |
| `ztb/models/gru_policy.py` | GRU Policy | [ ] |
| `ztb/training/curriculum_v456.py` | Curriculum | [ ] |
| `scripts/v456/train.py` | 学習スクリプト | [ ] |
| `scripts/v456/backtest.py` | バックテスト | [ ] |

### ドキュメント成果物
| ファイル | 説明 | ステータス |
|---------|------|----------|
| `docs/v456/00_improvement_proposal.md` | 改善提案書 | [x] |
| `docs/v456/01_technical_specification.md` | 技術仕様書 | [x] |
| `docs/v456/02_feature_engineering_spec.md` | 特徴量設計書 | [x] |
| `docs/v456/03_implementation_checklist.md` | 本ファイル | [x] |
| `docs/v456/04_final_results.md` | 最終結果 | [ ] |

### 設定ファイル成果物
| ファイル | 説明 | ステータス |
|---------|------|----------|
| `config/v456/env_config.json` | 環境設定 | [ ] |
| `config/v456/reward_config.json` | 報酬設定 | [ ] |
| `config/v456/model_config.json` | モデル設定 | [ ] |
| `config/v456/training_config.json` | 学習設定 | [ ] |

---

## 署名欄

| 役割 | 名前 | 日付 | 署名 |
|-----|-----|-----|-----|
| 設計者 | - | - | - |
| 実装者 | - | - | - |
| レビュアー | - | - | - |
| 承認者 | - | - | - |
