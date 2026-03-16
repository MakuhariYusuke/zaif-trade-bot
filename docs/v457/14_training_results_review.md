# 14. v457 Training Results Review (補足指摘と参照先)

対象: `docs/v457/13_training_results_success.md`

## A. 見落とし・未記載で再現性に影響する点

1) **データ条件の明示不足**  
   - 学習/評価に使ったデータの**期間・ソース・分足粒度・欠損補正**が未記載。  
   - 同じ「20k steps」でも相場環境で結果が変わるため、**開始/終了日時**と**データファイル名**は記録必須。

2) **学習・評価の分離が不明**  
   - Train/Testが同一データの可能性がある。  
   - Over-trading改善の検証は**未見データでの再現性**が重要。

3) **SAC設定の再現性不足**  
   - `gamma / learning_rate / batch_size / seed` など未記載。  
   - 20k stepsは**シード差の影響が大きい**ため、最低3seed評価が必要。

4) **報酬スケールの記録欠如**  
   - `reward_scale / reward_clip` と `hold_ramp` の**単位感**が記載されていない。  
   - 例: `hold_ramp=200` でも `reward_scale=100000` なら学習影響は小さく見える等。

5) **取引コストの内訳が不明**  
   - 「手数料+スリッページ負け」とあるが、**費用の割合**が未記載。  
   - 例: 1トレード平均コスト / 1トレード平均利益 を併記すると原因が明確になる。

6) **取引頻度の評価が粗い**  
   - 1228 trades / 10k steps だけでは判断が難しい。  
   - **平均保有時間、平均デルタ量、往復回数、Trade Size分布**が不足。

7) **min_delta調整案の適用経路が未確認**  
   - `FastIntradayEnvV456` は `min_delta` を内部パラメータで持つが、  
     **現状のfactoryではconfigから渡っていない**ため調整が無効化される可能性がある。  
   - 変更するなら**注入経路の確認**が必要。

8) **丸め誤差の確認**  
   - 行動分布の合計が 100.1% (38.5 + 51.9 + 9.7) になっている。  
   - 小さいが、集計処理の端数ルールは統一した方が良い。

## B. 追加で押さえるべき分析・ログ

- **期待値分解**  
  - WinRateだけでなく、**avg win / avg loss / profit factor / expectancy**を記録。  
  - 高勝率でも損失が大きいと赤字になるので原因が特定しやすい。

- **edge_shortfall / trade_cost / vol_ratio のログ**  
  - v455で推奨されていた指標。  
  - Over-tradingが「コスト無視」か「低ボラ滞在」かを分離できる。

- **seed別の分散**  
  - 20k stepsは分散が大きいので、**中央値**で比較が安全。

## C. 参考になりそうな vXXX（該当ファイル）

1) **v437: 取引頻度制御の実装知見**  
   - 取引間クールダウン/最大取引数/取引ペナルティが整理されている。  
   - `docs/README_v437.md`

2) **v455: min_edge_mult / vol_floor の感度分析とログ設計**  
   - 3x3グリッドや `edge_shortfall` 可視化の設計が明確。  
   - `docs/v455/11_sensitivity_and_training_plan.md`  
   - `docs/v455/12_sensitivity_results_and_training_conclusion.md`

3) **v452: ThresholdManagerの安全化と相対閾値**  
   - 閾値の**固定値依存を避ける**仕組みとバリデーション改善。  
   - `docs/v452/changes_v452.md`

4) **v420 Hold Relaxed: HOLD対策の副作用と頻度偏り**  
   - HOLD封じで高頻度化する副作用が詳細に書かれている。  
   - `docs/bug_fixes/SAC_V420_HOLD_RELAXED_LEARNING_PROCESS.md`

## D. 軽微だが有用な指摘

- `hold_grace` を固定にするより、**相場ボラに応じて伸縮**させると「早逃げ強制」の副作用が減る。  
- `min_delta` を上げる場合は **`max_position_size` による実効値**を必ず併記。  
- 手数料負けが主因なら、**「エントリー時ペナルティ」よりもクールダウン**の方が副作用が少ない場合がある。  

## E. 短期の追加タスク（最小）

1) 評価ログに `avg win / avg loss / profit factor / trade_cost` を追加  
2) seedを3つに増やし中央値比較  
3) `min_delta` / `cooldown_steps` を動かせるように注入経路を整理
