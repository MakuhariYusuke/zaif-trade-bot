# v456 Code Review Response (Implementation Stuck Analysis)

> **Version**: v456.1  
> **Date**: 2026-01-14  
> **Reviewer**: Codex  
> **Scope**: docs/v456 + `25_ai_code_review_prompt.md`

---

## 1. 最重要診断（結論）

最も可能性の高い失敗要因は以下の複合です。

1) **特徴量の大半がランダムノイズ**で、学習信号が崩壊している。  
2) **reward と balance の結合**が強く、初期数ステップで drawdown に到達して即終了する設計になっている。  
3) **環境設定・評価条件がドキュメント間で不一致**で、改善効果の比較が不能になっている。  
4) **同一データでの評価**により、評価結果が意味を持たない。

---

## 2. Critical Findings

### 2.1 ランダム特徴量の投入
- **症状**: MTF/Regime など未取得特徴量を `np.random.randn()` で埋めている。  
- **影響**: 学習はランダムノイズに適応し、実データに一般化しない。  
- **参照**: `docs/v456/25_ai_code_review_prompt.md`

### 2.2 reward が balance を直接変動させる設計
- **症状**: 初回アクションで reward が大きく負となり、balance が即時減少 → drawdown 条件で終了。  
- **影響**: エピソード長 1〜2 ステップ化、学習自体が成立しない。  
- **参照**: `docs/v456/20_stage1_implementation_findings.md` `docs/v456/17_root_cause_analysis.md`

### 2.3 環境設定のドリフト
- **症状**: initial_balance、max_position、drawdown_limit、max_steps が文書間で不一致。  
  - 例: 124.01 JPY / max_position=None と 100,000 JPY / 0.01 BTC の混在。  
- **影響**: 実験結果の比較が無意味化し、改善の再現性が失われる。  
- **参照**: `docs/v456/16_week4_real_account_training.md` `docs/v456/25_ai_code_review_prompt.md` `docs/v456/22_fix_implementation_success.md`

### 2.4 Train/Eval データリーク
- **症状**: トレーニングと同一データで評価している。  
- **影響**: 指標が無効。過学習・過小評価どちらも判別不能。  
- **参照**: `docs/v456/25_ai_code_review_prompt.md`

---

## 3. Major Findings

### 3.1 観測空間・特徴量定義の不一致
88D/85D/91Dや、Global/Accountの次元数が複数パターンで記載されている。  
**環境のshape不整合やログ解釈ミスの原因**になる。  
- **参照**: `docs/v456/02_feature_engineering_spec.md` `docs/v456/14_week3_kickoff.md` `docs/v456/07_revised_action_plan.md`

### 3.2 SafeIntradayEnvWrapperによるtrain-live差異
Wrapperで drawdown/reward を変更しているが、ライブ適用が明記されていない。  
**train/live パリティが崩れる可能性**。  
- **参照**: `docs/v456/22_fix_implementation_success.md` `docs/v456/24_final_execution_report.md`

### 3.3 Actionスケールの異常疑い
`pos=-100.000` 等の記録があり、行動→ポジション変換の単位ズレが疑われる。  
**過大手数料や意図しない売買**につながる。  
- **参照**: `docs/v456/20_stage1_implementation_findings.md`

---

## 4. Moderate Findings

### 4.1 訓練ステップ不足
30K timesteps は 88D 特徴量 + SAC には不足の可能性が高い。  
**まず安定動作→長期訓練**が必要。  
- **参照**: `docs/v456/22_fix_implementation_success.md`

### 4.2 1分足はHFTではない
実質は「短期イントラデイ」であり、HFT向け設計期待と乖離。  
**目的と粒度の再定義**が必要。  
- **参照**: `docs/v456/25_ai_code_review_prompt.md`

---

## 5. 質問への回答

### Q1: 40の合成特徴量をどう置換すべきか
- **必須**: ランダム特徴量は即時廃止。  
- **推奨**: 1m OHLCV から算出可能な指標（RSI/MACD/ATR/BB/ADX）を **MTFで再計算**。  
- **代替**: 未取得特徴量は **削除 or 0固定 + mask** にしてモデルが欠損を学習できるようにする。

### Q2: 報酬関数のスケーリング
- reward は **PnL基準**で設計し、**balance更新はPnL基準で別計算**に分離。  
- Trade-based PnL + step-level cost の併用が安定。  
- Sharpe/Sortinoは「学習報酬」ではなく**評価指標**として使用。

### Q3: データリーク対策
- 時系列 split (train/val/test) + embargo gap を導入。  
- Walk-forward: 90日 train / 30日 test の rolling を推奨。  
- **評価は必ずOOS**で実施。

### Q4: BUY偏重の理由
- 行動→ポジション変換の**スケール/符号**確認が最優先。  
- reward がSELLよりBUYに有利になっていないか検証。  
- 直すなら **離散行動 (BUY/SELL/HOLD)** の暫定導入が有効。

### Q5: Hyperparameter
- `initial_balance` と `max_position` は **データ価格・手数料に対し現実的な損益スケール**に一致させる。  
- LRスケジュールは有効だが、**まず環境とデータの修正が優先**。

### Q6: モデル構造
- 現段階は **MLPで十分**。  
- GRU/LSTMはデータ品質が確定してから導入する。

### Q7: 診断アプローチ
- 1ステップごとに **rewardとPnLの差分**、fee/slippage、balance更新をログ化。  
- action分布・ポジション量の分布、done理由の割合を記録。  
- **乱数特徴量の割合**を毎回チェック。

### Q8: ベースライン比較
- RSI/MACD等の簡易ルール戦略を必ず実装し、RLの最低ラインを設定。  
- BTC/JPYの1分足では **手数料込みでプラス**が出ることが最低基準。

---

## 6. 最優先アクションプラン（Top 3）

1. **ランダム特徴量の撤廃 + 特徴量定義の統一**  
   - 影響: ★★★★★ / 工数: 中  
   - 実施内容: 欠損列は即エラー or 0+mask、MTF/RegimeをOHLCから再計算。

2. **報酬と資金更新の分離**  
   - 影響: ★★★★★ / 工数: 中〜高  
   - 実施内容: balanceはPnLで更新し、rewardは学習用にスケール調整。

3. **評価パイプラインの再構築（OOS）**  
   - 影響: ★★★★☆ / 工数: 中  
   - 実施内容: 時系列split + walk-forward、記録の再現性確保。

---

## 7. 代替アプローチ

- **ルールベース + 監督学習**で方向予測し、RLはサイズ調整に限定。  
- **PPO/DQN** への切替は二次的。まずはデータ品質と環境設計を安定化。  
- **TCN/Transformer**は効果検証後に検討。

---

## 8. テスト戦略

- **Unit**: feature生成・MTFクローズバー・スケーリング範囲。  
- **Integration**: env.step()のbalance更新/termination条件。  
- **Backtest**: OOS期間のみで統計的指標を計測。  
- **Regression**: 重要メトリクス（episode length/hold率/fee支払額）を固定ベースラインで検証。

---

## 9. ドキュメントギャップ

- 設定値の**単一ソース**がない。  
- **観測空間の定義**が複数バージョンで混在。  
- **評価結果**が予測値と実測値で混在。  
  - 予測と実測を明示的に分離すべき。

---

## 10. 追加の気づき

- 実残高 (124 JPY) での学習は **feeインパクトが過大**で学習が崩壊しやすい。  
  → 学習は **スケールした仮想資金**、本番は別評価で分離すべき。  
  参照: `docs/v456/16_week4_real_account_training.md`

---

## 総合評価

**実装着手可否**: 条件付き可  
**条件**:  
1) ランダム特徴量の撤廃  
2) reward/balance分離  
3) 評価のOOS化
