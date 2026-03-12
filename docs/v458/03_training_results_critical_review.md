# 03. v458 Training Results: Critical Review & Improvement Plan

対象: `docs/v458/02_training_validation_results.md`

## 0. 総評（率直な評価）
現状の結果は「**良い兆候はあるが、結論は早い**」という段階です。  
学習ステップが 1,000 と短く、**OOS検証が未実施**で、**取引回数が極端に少ない**ため、  
「Lost Alpha の回復」を断定するには根拠が不足しています。

## 1. 重大な懸念点（優先度順）

### Critical
1) **学習ステップ不足 + ガイダンス未減衰**
   - `guidance_decay_steps=20000` に対して学習は 1,000 steps。  
     ほぼ全期間が「教師強制」状態で、**“自立した学習成果”を評価できていない**。
2) **検証データが訓練と同一**
   - `config/v458/base/config.yaml` の `training_data` と `validation_data` が同一。  
     **OOS評価が無く、過学習・偶然勝ちの排除ができない**。
3) **取引回数が 5 回のみ**
   - 141k stepsで 5トレードは**HFT/短期戦略の目的に合致しない**。  
     buy-and-holdに近く、**市場トレンドに“乗っただけ”**の可能性が高い。
4) **行動空間が 1d_position（TTL無効）**
   - `FastIntradayEnvV456` では 1d_position だと TTL が無効化。  
     **時間制約による頻度調整が働かず、極端な低頻度化を助長**。

### High
5) **「Global features」を実際は使っていない可能性**
   - `fast_intraday_env_v456.py` では global 特徴量が 0 埋め。  
     結果ドキュメントの「Global features活用」と **実装が不整合**。
6) **`execution_model` / `dynamic_threshold` が FastEnv で未配線**
   - `config/v458/base/config.yaml` に記載はあるが、  
     `FastIntradayEnvV456` では未活用の可能性が高い。  
     **「効いている前提」で評価してしまうリスク**。
7) **MTF overflow warning**
   - `np.float32` キャストの overflow は「非クリティカル」ではなく、  
     **NaN混入→学習崩壊の前兆**になりうる。

### Medium
8) **reward_clip=1.0 の影響が不明**
   - `reward_scale=100000` と `reward_clip=1.0` は、  
     **報酬の飽和を招いて学習を鈍化**させる可能性がある。  
     raw reward 分布の確認が必要。
9) **Seed安定性の検証不足**
   - v457での「bimodal不安定」は seed 依存が本質だった。  
     単一seedの勝ちは再現性が担保されない。

## 2. 既存資産を活かした改善方針（再利用優先）

### A. 評価の信頼性を上げる（即対応）
- **OOS分割を明示**  
  - `config/v458/base/config.yaml` の `training_data` / `validation_data` を分離。  
  - 時系列 split（例: 70/15/15）で walk-forward を再現。
- **ベースライン比較を必須化**  
  - `scripts/v457/backtest_v457.py` の `--baseline long/short/flat` を利用し、  
    **buy-and-holdと差分で性能を判断**する。  
- **評価指標の拡充**  
  - `profit_factor`, `expectancy`, `avg_win/avg_loss`, `max_dd`, `trades/day` を追加。  
  - v456 KPI基準（`docs/v456/00_improvement_proposal.md`）を採用。

### B. 頻度制御とコスト構造の再確認（v457知見の活用）
- v457グリッドサーチ結果（`docs/v457/19_v458_grid_search_results.md`）では  
  **頻度増加 = 手数料死**が再現。  
- そのため **cooldown強化 + edge判定**を優先（`docs/v457/20_v458_grid_search_review.md`）。  
  - `cooldown_steps: 30〜60`
  - `min_edge_mult: 1.5` 以上
  - `vol_floor` の引き上げ

### C. ガイダンスは「段階制御」へ
- v457の seed instability 解決案（`docs/v457/32_seed_stability_test.md`）を再利用。  
  - `BalanceCurriculumManager` + `SignalRewardIntegrator` を使い  
    **段階的にIchimoku重みを落とす**方式に戻す。  
  - `ztb/trading/environment/components/reward/balance_curriculum.py`  
    `ztb/trading/strategies/signal_reward_integrator.py`

### D. Dynamic Threshold の既存実装を接続
- `ThresholdManager` を FastEnv 側に接続し、  
  `dynamic_threshold_mode` を **実際に動作させる**。  
  - `ztb/trading/environment/components/threshold_manager.py`  
  - `docs/v450/01_dynamic_thresholding.md` の設計を参照

### E. コスト/実行モデルの活用（v455資産）
- `docs/v455/00_high_frequency_trading_proposal.md` にある  
  **Calibration Gate / Integrated Signal System** を再利用し、  
  **期待値がプラスのトレードだけ許可**する。
- 既存 `ActionSignalGuide` と統合し、  
  RL単独でのエントリーを減らす。

## 3. 具体的な改善アクション（短期→中期）

### Short-term (今週)
1) OOSデータ分割 + baseline比較を追加  
2) 2M steps × seeds [42,123,777,999] 実施  
3) `reward_clip` を外した比較 run を実施  
4) MTF overflow の原因調査（float64化 or clip）

### Mid-term (次フェーズ)
1) TTLを復活（2d action）し、頻度と保有時間を制御  
2) cooldown + edge判定で「回転数を下げて勝つ」構造を構築  
3) Dynamic Threshold を FastEnv へ接続し、v450知見を再活用

### Long-term
1) Calibration Gate + Integrated Signal System を実装  
2) Regime別のパラメータ最適化（v456案）  
3) Paper trading で OOS+実執行リスクを検証

## 4. 判定基準（Go / No-Go）
- **Seeds 3本以上で Net PnL > 0 が再現**  
- **Profit Factor > 1.05**  
- **Trades/day が KPI帯に入る**（v456基準: 50〜300）  
- **OOSで buy-and-hold を上回る**

---
参考:  
- `docs/v457/20_v458_grid_search_review.md`  
- `docs/v457/32_seed_stability_test.md`  
- `docs/v456/00_improvement_proposal.md`  
- `docs/v455/00_high_frequency_trading_proposal.md`  
- `scripts/v457/backtest_v457.py`
