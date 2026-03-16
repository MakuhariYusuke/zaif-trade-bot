# v458 Refactor & Reuse Strategy (Doc 05)

対象: `docs/v458/04_challenges_and_next_steps.md`

## 1. 現状の評価（批判的に見るべき点）
- 成果は出ているが、**評価基盤はまだ弱い**。Walk-Forward未統合、指標の母数が小さい可能性が残る。
- Profit Factor などの改善値は大きいが、**trade数と期間のバランスが不明**で、過大評価の恐れがある。
- 既存の強力な資産があるにも関わらず、**統合が部分的**。再実装が増えると品質低下リスクが上がる。

結論として、今必要なのは「新機能追加」ではなく、**既存資産の統合と評価の信頼性確保**。

## 2. 既存資産の再利用マップ（優先度順）

### A. Walk-Forward評価（最優先）
- 既存実装:
  - `scripts/v456/phase4/run_walk_forward_analysis.py`
  - `scripts/v456/phase4/PHASE4_GUIDE.md`
  - `ztb/evaluation/walk_forward/*`
  - `ztb/optimization/parallel/window_evaluator.py`（並列化）
- 方針:
  - v458の環境/設定を流し込み、**同一インターフェースでWalk-Forwardを回す**。
  - 既存の結果集計（Reporter）を使い、評価指標を統一する。

### B. ハイパーパラメータ最適化（既存ツール活用）
- 既存実装:
  - `tools/ab_test_runner.py`
  - `tools/ab_param_search.py`
  - `ztb/training/hyperparameter_optimizer.py`
  - `ztb/training/unified_optimizer.py`
  - `config/ab/*`
- 方針:
  - v458のconfigを母体に**少数パラメータのABテスト**を先に回す。
  - 大規模探索は後回しにし、seed安定性とOOS優先。

### C. Live/Paper Trading（実運用評価）
- 既存実装:
  - `ztb/trading/live/simulation/paper_trader.py`
  - `ztb/trading/live_trader/*`
  - `ztb/trading/production/paper_trading_manager.py`
  - `ztb/trading/environment/bridge.py`
  - `docs/implementation/PHASE4_LIVE_PAPER_TRADE_INTEGRATION.md`
- 方針:
  - v458は**paper_trader系の統合済み実装を優先**。個別スクリプトを増やさない。

### D. リスク管理・安全停止
- 既存実装:
  - `ztb/trading/production/circuit_breaker.py`
  - `ztb/trading/production/virtual_portfolio_manager.py`
  - `ztb/trading/production/risk_based_allocator.py`
  - `emergency_stop.py`, `recovery_system.py`, `rollback_manager.py`
- 方針:
  - 評価と同時に「リスク判定の閾値」を明文化し、paper trading で検証。

### E. エントリーゲート（v455資産）
- 既存実装:
  - `ztb/trading/signal/entry_system.py`
  - `ztb/trading/signal/calibration_map.py`
  - `docs/v455/00_high_frequency_trading_proposal.md`
- 方針:
  - **IntegratedEntrySystem + CalibrationGate をフィルタとして復活**。
  - RLの無駄打ちを削減し、「期待値の低い取引」を防止。

### F. 動的閾値・頻度制御（v450/v457資産）
- 既存実装:
  - `ztb/trading/environment/components/threshold_manager.py`
  - `docs/v450/01_dynamic_thresholding.md`
  - `docs/v457/20_v458_grid_search_review.md`
- 方針:
  - cooldown + edge判定を先に固定化し、動的閾値は**統合後に再評価**。

## 3. 重点リファクタリング案（重複削減）

### 3.1 Training/Backtest統一
- v458は v457の標準パイプラインを使い、**独自スクリプトを最小化**する。
- `scripts/v457/train.py` と `scripts/v457/backtest_v457.py` の構造を踏襲し、  
  v458専用の差分だけを config で管理する。

### 3.2 評価指標の標準化
- `scripts/v457/backtest_v457.py` の統計項目を**基準メトリクスとして固定**。
- `profit_factor`, `expectancy`, `max_drawdown`, `trades/day` を必須化。

### 3.3 リスク管理の一元化
- backtest, paper trading, live の**リスク判定ロジックを共通化**。
- VirtualPortfolioManager と RiskBasedAllocator を中心に組み立てる。

## 4. 実装ロードマップ（Phase 5案）

### Phase 5.1: 評価基盤の統合 ✅完了
1) Walk-Forward評価を v458 に接続 ✅
2) baseline比較 (buy/hold, flat, short) を標準化 ✅
3) 3-4 seed + OOS固定評価をセット化 ✅

**完了成果**:
- Walk-Forward Analysis: 2ウィンドウ評価完了
- ベースライン比較: v456, Buy-and-Hold比較実施
- 複数seed検証: seed=123,42,randomで安定性確認
- 評価指標統一: Profit Factor, Expectancy, Sharpe標準化

### Phase 5.2: 取引頻度制御 + ゲート導入
1) cooldown + edge判定の安定化  
2) IntegratedEntrySystemでフィルタリング  
3) Paper tradingで trade/day と PF の安定性を検証

### Phase 5.3: 本番想定の検証
1) PaperTraderで実運用負荷を測る  
2) RiskBasedAllocatorで段階的配分テスト  
3) CircuitBreaker/RecoverySystemを実データで評価

## 5. 進捗判定の基準（Go/No-Go）
- Walk-Forwardで**全ウィンドウがPF>1.05**  
- OOSで buy-and-hold を上回る  
- Trades/day が目標帯に収まる  
- Seedごとの差分が許容範囲内（中央値評価）

## 6. 次に出すべき成果物（Doc 05の役割）
このドキュメントは「**再実装を止め、既存資産を統合する設計図**」として位置付ける。  
次の成果物は以下を満たす必要がある:

1) v458評価基盤がWalk-Forwardに統合されていること  
2) ABテスト/ハイパラ探索が既存ツールで回せること  
3) Live/Paper trading用の統合ルートが一本化されていること  
4) リスク管理が backtest/paper/live で同一基準になっていること

---
参考:
- `scripts/v456/phase4/run_walk_forward_analysis.py`
- `ztb/evaluation/walk_forward/*`
- `tools/ab_test_runner.py`
- `ztb/trading/signal/entry_system.py`
- `ztb/trading/production/risk_based_allocator.py`
