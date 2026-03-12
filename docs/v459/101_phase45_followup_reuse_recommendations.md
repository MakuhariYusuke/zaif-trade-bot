# 101# 100#フォローアップレビュー: 過去vXXX資産の再利用提案とPhase C実装優先順位

**Date**: 2026-02-08  
**対象**: `100_phase45_completion_report.md` + 過去vXXXドキュメント + 既存実装棚卸し  
**目的**: 高収益化に向けて、再利用価値の高い資産を厳選し、Phase Cの実行順を明確化する

---

## 0. 結論（先に要点）

1. `100`のNO-GO判定は妥当。**SACがRandom同等**の段階でPhase 5本格移行は早い。  
2. ただし、既存コード資産は十分ある。新規開発よりも**評価基盤の再利用と接続修正**が先。  
3. 再利用は「そのまま使える資産」と「修正前提資産」を分けるべき。特に以下は優先度が高い。  
   - `ztb/metrics/metrics.py`（PF/Sharpe/MaxDD/WinRate）
   - `ztb/evaluation/walk_forward/*`（OOS評価）
   - `ztb/trading/environment/components/threshold_manager.py`（取引頻度制御）
   - `ztb/trading/execution/*` + `ztb/trading/cost/venue_transaction_cost_manager.py`（約定・コスト現実化）
4. 逆に、以下は現状のまま再利用しない方がよい。  
   - `scripts/v458/run_walk_forward_v458.py`（script依存が強く脆い）
   - `ztb/analysis/baseline_comparison.py`（`BaselineComparisonEngine`重複定義）
   - `scripts/v459/check_data_leakage.py`（簡易/プレースホルダ検査が多い）
5. `100`の11章は**方向性としては概ね正しい**が、実装前提にズレがある。  
   - `min_holding_period` の現行値認識
   - `dynamic_threshold_mode` のHeavyEnv側配線
   - `stage1_*` YAML（complex path）と`PnL-only`方針の混線
   - `compute_hft_reward` のHeavyEnv未接続

---

## 1. 100#に対する追加の批判的観点

## 1.1 統計の独立性

`BuyAndHold` と `Momentum_RSI` は実装上ほぼ決定的で、seedを変えても同値になりやすい。  
この場合、`n=4`を独立サンプルとして扱うと過大評価になる。

- 対象: `scripts/v459/run_baselines.py`
- 含意: deterministic baselineは「seed複製」ではなく、**window分割（時期差）で標本を増やす**方が正しい。

## 1.2 小標本での近似検定

`gate_c3_comparison.py` のMann-Whitneyは近似実装。`n=4`規模では近似誤差の影響が出やすい。

- 対象: `scripts/v459/gate_c3_comparison.py`
- 含意: Phase C最終判定は、可能ならexact検定系に寄せるか、少なくともwindowを増やして近似依存を減らす。

## 1.3 指標定義の曖昧さ（Gross/Net）

`BuyAndHold`で`Gross PnL=0`は、実現損益中心の定義に依存している可能性が高い。  
保有中の含み損益を明示しないと戦略比較で誤解が起きる。

- 関連: `ztb/trading/environment/components/position_manager.py`（`realized_pnl`, `unrealized_pnl`, `gross_pnl`）
- 含意: 「実現ベース」と「時価評価ベース」をKPIとして分離し、同じ表に混在させない。

## 1.4 OOS未確定

`P1`は`walk_forward`無効で、訓練・評価分離が弱い。

- 対象: `scripts/v459/run_phase45_p1.py`
- 含意: Phase Cで改善が見えた時点で、**即OOS（4split）へ移行**しないと誤学習を見抜けない。

---

## 2. 再利用資産の優先度マップ

## A. 即採用（そのまま使える）

| 資産 | パス | 使い方 | 期待効果 |
|---|---|---|---|
| Metrics計算基盤 | `ztb/metrics/metrics.py` | PF/Sharpe/MaxDD/WinRateを統一算出 | `100`で未計測のKPIを即補完 |
| Walk-Forward分割 | `ztb/evaluation/walk_forward/splitter.py` | embargo付き時系列分割を標準化 | OOS判定の信頼性向上 |
| Walk-Forward評価器 | `ztb/evaluation/walk_forward/evaluator.py` | window単位で評価を統一 | 評価ロジックの再利用 |
| 回帰テスト群 | `tests/unit/evaluation/test_walk_forward_*` | 改修時の回帰確認 | 既存不具合の再発防止 |
| 取引頻度制御 | `ztb/trading/environment/components/threshold_manager.py` | threshold/adaptive制御で過剰売買を抑制 | 手数料負けの縮小 |
| コールバック | `ztb/training/callbacks/advanced_callbacks.py` | EarlyStopping/BestModel保存 | 長時間学習の無駄削減 |
| ベースライン実験 | `scripts/v459/run_baselines.py` | 同条件比較の土台 | Gate C3再実施を効率化 |

## B. 条件付き採用（修正・前提確認後に使う）

| 資産 | パス | 採用条件 | 背景教訓 |
|---|---|---|---|
| v451設定群（γ=0.80, loss重み） | `config/v451/sac_v451_optimized.json` | 現環境でA/B再検証 | v451成功の再現性確認 |
| V457RewardCalculator | `ztb/trading/environment/components/calculators/v457_reward_calculator.py` | simple pathとの同条件比較 | v456の過剰シェーピング反省 |
| 1D action単純化思想 | `docs/v457/27_v457_3_analysis.md` | TTL/行動空間の寄与を分離検証 | churn抑制の有効性 |
| 特徴量相関削減 | `ztb/features/core/registry.py` + `ztb/preprocessing/feature_correlation_filter.py` | 分析JSONとfeature登録を明示 | 「8特徴固定」の妥当性再点検 |
| 特徴量アブレーション | `ztb/benchmarks/ablate_features.py` | seed/window付きで実行 | どの特徴が効くか可視化 |

## C. 非推奨（現状のまま再利用しない）

| 資産 | パス | 理由 |
|---|---|---|
| v458実行スクリプト本体 | `scripts/v458/run_walk_forward_v458.py` | script import依存が強く、再利用時に壊れやすい |
| Baseline比較エンジン | `ztb/analysis/baseline_comparison.py` | 同名クラス重複で挙動が読みづらい |
| leakage checker（簡易版） | `scripts/v459/check_data_leakage.py` | プレースホルダ検査が多く、保証水準が不足 |

---

## 3. 過去vXXX教訓の「再利用の仕方」

## 3.1 v451（成功設定）は「盲信」ではなく再現実験で使う

- 参照: `docs/v457/01_legacy_asset_analysis.md`, `config/v451/sac_v451_optimized.json`
- 適用: `gamma=0.80`, `loss_multiplier=1.2`をPhase Cで**独立要因として検証**。
- 注意: v451の市場局面と現在局面は異なる。単発再現で本採用しない。

## 3.2 v457.2（Tiny Edge）を現在結果と接続する

- 参照: `docs/v457/23_v457_2_strategy_plan.md`
- 適用: Gross/Net差分を「1取引あたり」で常時監視する。
- 具体KPI: `avg_gross_pnl_per_trade`, `avg_fee_per_trade`, `edge_to_cost_ratio`。

## 3.3 v457.3（行動空間単純化）を再評価する

- 参照: `docs/v457/27_v457_3_analysis.md`
- 適用: 行動空間が過剰なら、1D position中心に戻して取引回数とPF変化を確認。

## 3.4 v456（複雑報酬の失敗）をガードレールにする

- 参照: `docs/v456/59_V456_FINAL_RETROSPECTIVE.md`
- 適用: penalty項を増やす前に、PnL-onlyで改善余地を確認する順序を厳守。

## 3.5 v454（逆説的確信）は診断器として使う

- 参照: `docs/v454/01_roadmap_signal_quality.md`
- 適用: action強度bin別の勝率/損益を出し、強シグナルが本当に優位か監視する。

---

## 4. Phase C実行順（最小改修版）

## C0: 計測統一（最優先）

1. KPI算出を`ztb/metrics/metrics.py`に寄せる。  
2. `realized`/`unrealized`/`mark-to-market`を明示的に分離。  
3. deterministic baselineはseed複製ではなくwindow複製で比較。

## C1: コスト圧縮（手数料負け対策）

1. `ThresholdManager`で`continuous_threshold`を段階引上げ。  
2. `min_holding_period`と`allow_reverse/enforce_reverse_cooldown`を同時最適化。  
3. 評価はROI単体でなく、`gross_per_trade`と`fee_per_trade`を必須化。

## C2: 約定モデル・コストモデルの現実化

1. `ztb/trading/execution/realistic.py`でスリッページ/遅延感度を見る。  
2. `ztb/trading/execution/pseudo_hft.py`で高頻度寄りの頑健性を見る。  
3. `ztb/trading/cost/venue_transaction_cost_manager.py`でvenue別感度を算出。

## C3: 報酬・ハイパラ再現実験

1. baseline: PnL-only（現行）  
2. variant-1: v451系（γ=0.80, loss_weight=1.2）  
3. variant-2: V457RewardCalculator  
4. ここで初めて200Kへ拡張（50Kで優位が出ないものは延長しない）。

## C4: OOS最終判定

1. `ztb/evaluation/walk_forward/splitter.py`で4split。  
2. `4 seeds × 4 splits`で比較。  
3. Random/BuyHold/Momentum超過を判定してPhase 5可否を確定。

---

## 5. 直近の実行提案（高収益化に向けた現実的な順序）

1. **48時間以内**: C0 + C1（計測統一と過剰売買抑制）  
2. **次の48時間**: C2（約定/コスト感度） + C3の軽量版（50K）  
3. **勝ち筋が出た条件のみ**: C4（4split OOS）へ進む

---

## 6. 最終コメント

`100`で得られた「SAC≒Random」という事実は重いが、打ち手は残っている。  
鍵は「新機能追加」ではなく、**既存資産を正しい順序で接続して誤判定を減らすこと**。  
Phase Cでは、まず計器を整え、次にコストを削り、最後にOOSで勝ち筋を検証する。

---

## 7. 100# 9章ソースパス検証結果（コード照合）

| 9章記載 | 実在 | 照合結果 |
|---|---|---|
| `ztb/trading/environment/components/position_manager.py` | ✅ | `buy_count/sell_count` 属性追加、約定時インクリメント、`reset()`でゼロクリアを確認 |
| `ztb/trading/environment/components/calculators/reward_calculator.py` | ✅ | `position_change_penalty/threshold` の設定値化を確認（ハードコード固定ではない） |
| `scripts/v459/run_phase45_p1.py` | ✅ | `hold_penalty_multiplier=1.0` を確認 |
| `scripts/v459/run_phase45_p1_subprocess.py` | ✅ | seed別 `stdout/stderr` ログ保存を確認 |
| `scripts/v459/run_baselines.py` | ✅ | Random / BuyAndHold / Momentum_RSI 実装を確認 |
| `scripts/v459/gate_c3_comparison.py` | ✅ | Mann-Whitney / Cliff's delta / Holm補正ロジックを確認 |
| `tests/unit/trading/components/test_gate05_reward_purity.py` | ✅ | テストメソッド16件を確認（Gate 0.5 + C0の検証項目あり） |

補足:
- 依存不足（`pydantic`）のため、この環境ではテスト実行までの再確認は未実施。  
- ただし、9章に記載されたファイルパスと変更意図は、静的コード照合では整合している。

---

## 8. 100# 11章方向性の妥当性判定

| 施策 | 判定 | コメント |
|---|---|---|
| 11.1 A-1 `continuous_threshold` 引き上げ | ✅ 妥当 | 過剰売買抑制の主軸として有効。先行実施すべき |
| 11.1 A-2 `min_holding_period` 導入 | ⚠️ 要修正 | 現行値は0ではなく、`EnvironmentConfig`既定は3。基準値認識を修正すべき |
| 11.1 動的閾値（z-score/adaptive）の即利用 | ⚠️ 条件付き | `ThresholdManager`機能はあるが、HeavyEnv設定経路で`dynamic_threshold_mode`等が十分露出していない |
| 11.2 B-1 `stage1_trade_reduced.yaml`活用 | ⚠️ 条件付き | 当該YAMLは`use_simple_reward: false`。PnL-only方針とは別トラックとして扱うべき |
| 11.2 B-2 `V457RewardCalculator`活用 | ✅ 条件付き妥当 | `custom_reward_params.type=\"pnl_centered\"` 指定時のみ有効。明示配線が前提 |
| 11.2 B-3 `compute_hft_reward`統合 | ⚠️ 要実装 | 関数は存在するがHeavyTradingEnvのreward経路へ直接は接続されていない |
| 11.3 C-0 相関フィルタ不発疑惑 | ✅ 妥当 | `analysis_file`不在時は`cls.list()`にフォールバックする実装で、懸念は合理的 |
| 11.3 C-4 特徴量拡張再試験 | ✅ 妥当 | RSI偏重是正として有効。まず小規模seedで選別が現実的 |
| 11.4 D ステップ拡大後回し | ✅ 妥当 | 50Kで優位が出る条件に絞ってから200Kへ延長すべき |

---

## 9. 101追記修正（11章を実行可能にするための具体化）

1. `min_holding_period` の基準値を **0→3（既定値）** に修正。  
2. HeavyEnvで動的閾値を実験対象にするなら、`EnvironmentConfig` と `from_dict()` に以下を明示配線する。  
   - `dynamic_threshold_mode`
   - `z_score_window`
   - `z_score_threshold`
   - `z_score_method`
3. 報酬実験を2系統に分離する。  
   - 系統A: `use_simple_reward=True` のPnL-only微調整（因果分離優先）  
   - 系統B: `stage1_trade_reduced.yaml` 等のcomplex reward比較（別評価）
4. `V457RewardCalculator`は設定スイッチ明示で実験する。  
   - `reward_settings.custom_reward_params.type = "pnl_centered"`
5. `compute_hft_reward`は「即統合」ではなく、HeavyEnv用アダプタ実装後に導入する。  
6. deterministic baseline（BuyHold/Momentum）はseedではなくwindowで標本を増やす。  

上記6点を反映すれば、11章の方向性は「提案」から「実行可能な計画」に引き上げられる。
