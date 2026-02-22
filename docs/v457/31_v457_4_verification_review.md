# 31. v457.4 Verification Review & System Alignment Proposals

対象: `docs/v457/30_v457_4_verification.md`

## 1. 総評
1D Action のネイティブ実装で「Sell Only」へ崩れるのは、  
**Action 空間の問題というより“探索不足 + TTL 実装の影響”が濃い**。  
計画を前進させるには、**TTL を完全に無効化する設計**と  
**評価の安定化（seed/区間/基準）**が同時に必要。

## 2. 使える手（即効性のある提案）
1) **1D では TTL を完全に無効化**  
   - `FastIntradayEnvV456` は 1D でも `position_ttl` を減衰・強制クローズする。  
   - 「Hold しているのに周期的に強制フラット→再入場」が起きうる。  
   - **1D モード時は TTL チェック/減衰をスキップ**するのが本命。

2) **seed 固定 + 複数 seed で統計化**  
   - v457.3/4 の差は “初期探索の運” で説明可能。  
   - **最低 3 seed / 2 区間で再現性を確認**する。

3) **ベースライン比較を固定化**  
   - “Always Long / Always Short / Always Flat” の  
     3 つを同じ区間で比較するだけで、  
     **Buy&Hold 偶然勝利かどうかが即判別**できる。

4) **action 分布のログを必須化**  
   - `action[0]` 分布と、（2D の場合）`action[1]` 分布を保存。  
   - 早期に “張り付き崩壊” を検知できる。

5) **`min_edge_mult` と `edge_penalty_rate` を活用**  
   - `fast_intraday.py` の既存機能で  
     “値幅が足りないトレード”を抑制できる。  
   - いきなり fee 重視より **min_edge 強化の方が安定**。

## 3. v457 全体の整合性を高める視点
1) **Action mode の一貫性**  
   - config / training / backtest で `action_space_type` を統一。  
   - `create_fast_intraday_env_v456` が `action_space_type` を渡せるか再確認。

2) **PnL の定義を統一**  
   - `reward` は Net PnL だが、`total_pnl` は取引時のみ更新。  
   - **報酬・評価・ログの PnL 定義がズレている可能性**が高い。  
   - v457 全体で “評価に使う PnL 指標” を固定すべき。

3) **データ区間の固定化**  
   - 学習/検証区間が “同一データだけど開始位置がランダム” だと  
     結果が不安定になる。  
   - **区間と seed をログ保存**して再現性を担保する。

4) **v457 系ドキュメントの参照表を作る**  
   - v457.1/2/3/4 の横串を通す “Index” を用意し、  
     用語・指標・設定ファイル名の齟齬を減らす。

## 4. リファクタリング提案
1) **Action 解析ロジックの分離**  
   - `FastIntradayEnvV456.step()` 内の  
     1D/2D 分岐と TTL 処理を `ActionProcessor` に切り出し、  
     仕様変更を一箇所に集中させる。

2) **TTL の責務整理**  
   - 1D モードは “TTL 不使用” と明示し、  
     `position_ttl` の更新・減衰を無効化する設計にする。

3) **Accounting の統一**  
   - `balance` / `total_pnl` / `reward` の計算を  
     `TradeAccounting` のようなコンポーネントへ整理し、  
     **評価指標のブレを抑える**。

4) **設定読み込みの統一**  
   - `ztb/training/utils/v457_config_utils.py` を中心に  
     config の読み込みを一本化する。

5) **ログ出力の標準化**  
   - trades.json / action 分布 / TTL 強制クローズ回数を  
     “v457 共通ログ”として固定フォーマット化する。  

## 5. 妥当性検討（チェックリスト）
1) **学習/検証区間と seed の固定**  
   - `reset()` のランダム開始位置は seed に依存する。  
   - **区間ID + seed をログに残す**ことで v457.3/4 の差を切り分ける。

2) **Action/TTL の挙動確認**  
   - 1D でも TTL 減衰が走るため、  
     **強制クローズ回数と cooldown 回数**を記録する。  
   - 1D なのに “TTL=原因の売買” が残っていないかを検証。

3) **reward_scale / reward_clip の同一性**  
   - v457.3 と v457.4 の比較は **報酬スケールが揃っている前提**が必要。  
   - クリップが違うと “Sell Only / Buy Only” に偏る要因になる。

4) **PnL 定義の整合**  
   - `reward` と `total_pnl` の計算がずれるため、  
     **評価指標は Net PnL 基準で統一**する必要がある。

5) **ベースライン比較**  
   - Always Long/Short/Flat を同一区間で回し、  
     **“Buy & Hold 偶然勝利”の可能性**を排除する。

## 6. 追加の気づき（既存実装の流用候補）
- **TTL/1D の比較基準**: `ztb/trading/environment/wrappers/fixed_ttl_wrapper.py`  
  - 1D ネイティブ実装の回帰テストに使える。
- **Action 解析/変換**: `ztb/trading/environment/components/action_executor.py`  
  - ActionProcessor の雛形として転用できる。
- **頻度制御ロジック**: `ztb/trading/environment/components/position_manager.py`  
  - `min_holding_period` / `enforce_reverse_cooldown` で TTL の代替にできる。
- **再現性**: `ztb/utils/seed_manager.py`, `ztb/training/scripts/run_seal.py`  
  - seed と環境スナップショットの記録に直結する。
- **統計検証**: `scripts/analysis/statistical_sampling_framework.py`  
  - 複数 seed の安定性検証を自動化できる。
- **行動診断**: `ztb/analysis/action_confidence_diagnostics.py`  
  - “Sell Only / Buy Only” が行動強度の偏りかどうか判別できる。
- **trades.json 連携**: `backtest_v456.py` / `scripts/v457/backtest_v457.py`  
  - 実測の fee/slippage を含む履歴保存が可能。
- **評価指標の統一**: `scripts/analysis/analyze_backtest_v456.py`, `utils/results_utils.py`  
  - PF / Expectancy を共通フォーマットで算出できる。

## 7. 結論
v457.4 の “Sell Only” は 1D 実装バグというより、  
**TTL の残存 + 探索不安定性 + 評価の再現性不足**が主因。  
まずは **TTL 無効化 + seed/区間固定 + baseline 比較**を実施し、  
その上で reward / frequency 制御を強化すると、  
v457 全体の整合性と再現性が大きく上がる。
