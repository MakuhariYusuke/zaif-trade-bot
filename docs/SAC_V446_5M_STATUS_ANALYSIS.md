## SAC v446 5m Training Status Analysis — 2025-11-15

### 1. 目的
現在までの `sac_v446_5m_100k_config` トレーニング結果をレビューし、現行5分足パイプラインの課題を抽出し、分析と対策方向性を文章化する。

### 2. 現状確認
- **データ & 環境**: `data/btc_jpy_5m_dataset.csv` を用いたSAC学習（シャープ比最適化報酬、ケリー式ポジションサイズ、SL/TP、0.001の取引コスト）。
- **学習条件**: バッファ50k（レポート内 `sac.buffer_size` ）、学習率0.0003、バッチ256、総ステップ数10万。検証/ロギング（1000ステップごとの保存・評価）が有効。CUDA最適化は有効だが並列化は使用せず。
- **トレーニング結果**（`training_stats`）: 10万ステップで約4,219秒（約70分）。1秒あたり23.7ステップ、training_efficiency 0.0237。
- **パフォーマンス**: 最終報酬 `-80.0`、主導的アクションはBUY 52.8%。HOLD 17.8%、SELL 29.4%。アクション多様度0.805。

### 3. 抽出された課題
1. **収益性不足** — 最終 reward が `-80.0` で明確な損失。損益だけでなくリスクリターン（シャープ比等）が未記載だが、モデルは利益を出せていない。
2. **偏った行動傾向** — BUY比率が52.8%で、売買両方に対する自信が偏っており、SELL/HOLD比率の低さが損失につながっている可能性。リスク管理（ケリー/ドローダウン）が期待通り機能していない兆候。
3. **低トレーニング効率** — 単純な1 GPU/1 CPU実行でステップレート23.7/s。データ最適化は有効だが、並列学習やキャッシュを活用しておらず、同じリソースでの改善余地あり。
4. **モニタリング不足** — training_stats に各エポックの reward や loss、validation スコアが記録されておらず、学習曲線が描けないため学習停止条件（Early stopping）も機能せず。

### 4. 深堀り分析
- **損失の原点**: BUY偏重はゼロサム環境で反対注文不足となりがち。報酬関数が買い方向で高いリターンを期待しすぎており、SELL/HOLD シグナルのスケーリングが下がっている可能性。`reward_scaling` 6.0 と高めで、買いシグナルが過学習気味。
- **リスク管理との乖離**: ケリー式ポジションが設定されているが、実際の action_distribution ではサイドライン（HOLD）を含めて 30% 程度しかない。drawdown limit を 0.1 に設定しているが、トレーニング結果では負債が蓄積しており、スケーリング/フィルターの再検討が必要。
- **効率面**: training_efficiency 0.0237 というのは「ステップ／秒」が少なく、バッチ内でのトレーニングが毎ステップで再帰している。`train_freq`:1・`gradient_steps`:1 でステップごとに更新しているが、これを `gradient_steps`>1 にすると GPU がより活用できる見込み。
- **メトリクス不足**: 提供されている評価`metrics`（total_return, sharpe_ratio など）と実際のログのリンクがない。これらを `training_stats` に組み込むことで、負の報酬を示すタイムラインを確認する必要がある。

### 5. 推奨アクション
1. **報酬/アクション再スケーリング**: BUY/Sell/HOLD の reward shaping を再調整し、SELL/ HOLD にも十分なフィードバックを注入。シャープ比最適化に集中しすぎないこと。
2. **Validation metrics のロギング**: Eval callback で `mean_reward`/`std_reward` をロギングし、`training_stats` に追加する。early_stopping を有効化する準備として、progressive reward に基づく判断を記録。
3. **学習負荷改善**: `gradient_steps` を 4〜8 に増やし、GPU の利用率を引き上げる。並列環境（VecEnv で n_envs>1 など）で mini-batch の多様性を保ちながら step rate を維持。
4. **リスク制御の検証**: `risk_management` の設定（drawdown limit, stop_loss/take_profit）をバックテストし、特にケリーが過剰にポジションを拡大させていないか検証。
5. **報告更新**: 毎回の training_report に validation の時間系列を含め、CHANGELOG や docs に最新課題を追加し自動化。

### 6. 追跡事項
- モデルの reward の分布（ポジティブ vs ネガティブ）、validation ステップでの mean_reward、gradient_norm。これらを追跡するためのダッシュボード/スプレッドシートを併用。
- 5分足検証用スクリプトでは、このモデルと`data/btc_jpy_real_dataset.csv` の backtest を差分比較し、BUY偏重が現実世界でも同様か確認。

---
文責: GitHub Copilot (Drafted 2025-11-15)