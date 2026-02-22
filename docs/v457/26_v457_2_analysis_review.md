# 26. v457.2 Analysis Review & System-Level Advice

対象: `docs/v457/25_v457_2_analysis.md`  
目的: v457.2 の失速原因を俯瞰し、システム全体の改善方針を整理する

## 1. 即時の診断（v457.2結果から見えること）
1) **「Buy Only」よりも「TTL=短期 + 再入場ループ」の疑いが濃い**  
   - `FastIntradayEnvV456` は行動の第2要素が TTL で、`ttl_fraction` が 0 に寄ると TTL=1 になる。  
   - TTL 失効 → cooldown → 再エントリーが繰り返され、**約 6 ステップに1回の売買**が起きる。  
   - 10,000 steps で 1,583 trades は **TTL=1 + cooldown=5** に近い数値。  
   - まず **action[1]（TTL）の分布ログ**が必須。

2) **報酬は既に Net PnL だが、スケール/クリップが学習を壊している可能性**  
   - `compute_hft_reward()` は `pnl - fee - slippage` を返す。  
   - `reward_scale` と `reward_clip` が不適切だと、**極端な負の報酬が常時クリップされ学習が崩壊**する。  
   - v457.2 の `reward_scale=10.0` は “効きすぎ” の可能性が高い。

3) **`fee_penalty_weight` の導入が正しく反映されていない可能性**  
   - `reward_settings` は `compute_hft_reward()` の引数に直接渡る。  
   - 未定義キー（例: `fee_penalty_weight`）は **例外 or 無視**のどちらかになり得る。  
   - 学習が動いているなら、**実際には penalty が適用されていない疑い**もある。

## 2. システム全体で詰まりやすいポイント
### A. 行動設計と環境挙動
- 2D action（target_position + TTL）の場合、**TTL側の崩壊が売買頻度を支配**する。  
- `min_delta` / `cooldown_steps` / `max_delta_per_step` は環境側の“頻度制御”だが、  
  **TTLが短いと意味が薄れる**。

### B. 報酬定義とスケーリング
- Net PnL そのものは既に使っているため、**効き目はスケールとクリップ設計次第**。  
- 既存の `alpha/beta/gamma` は実装上 “無効” なので、設定しても反映されない。  
- “費用重視”をやるなら、**`fee_paid` を直接倍率で重くするか、`min_edge_mult` を上げる方が現実的**。

### C. 評価方法の歪み
- **学習と同じデータで backtest**しているため、汎化を測れていない。  
- `total_pnl` は “取引ステップのみ更新”なので、**報酬と評価で PnL定義がズレる**。  
- **trades.json が無く、fee/slippage 実測ログが残らない**。

### D. データ品質とレジーム設計
- “低ボラ期間の重み付け”が曖昧。  
  ATR/price を閾値化して **低ボラ期間の定義を固定**した方が実装しやすい。  
- データ源（Yahoo/bitFlyer/coincheck）の混在で、  
  **欠損・重複・時刻ズレの検証コストが増える**。

## 3. 既存資産の再活用ポイント
- `ztb/trading/rewards/fast_intraday.py`  
  - `min_edge_mult` / `edge_penalty_rate` / `hold_grace` / `hold_ramp` を使って  
    「十分な値幅が見込める時だけ取引する」設計が可能。
- `ztb/trading/environment/fast_intraday_env_v456.py`  
  - `min_delta` / `cooldown_steps` / `max_ttl_steps` の組み合わせで  
    **TTL依存の暴走を止められる**。
- `ztb/trading/environment/utils/fast_intraday_env_v456_utils.py`  
  - `env_config` 注入が可能なので **v457.2 config の統一管理に向く**。

## 4. 推奨アクション（優先順）
1) **TTL行動の分布を可視化**  
   - `action[1]` の平均/分布を記録して “TTL=0付近” が起きていないか検証。
2) **TTLを一旦固定化して 1D action に縮退**  
   - まず `target_position` のみで学習し、頻度制御は環境側で行う。  
3) **reward_scale / reward_clip を再設計**  
   - クリップを外すか、負の報酬が潰れない範囲に設定。  
4) **`fee_penalty_weight` を確実に反映**  
   - `compute_hft_reward()` に明示実装するか、既存の `min_edge_mult` を強化。  
5) **評価ログの強化**  
   - trades.json に fee/slippage 実測を保存し、  
     Net PF / Expectancy / Avg Win-Loss を固定出力化。

## 5. 結論
v457.2 の崩壊は「重い手数料」そのものより、  
**Action設計（TTL）と報酬スケールが学習を壊している可能性が高い**。  
最初に TTL を固定化し、スケール/クリップを正常化した上で再評価すべき。  
その上で “費用上回る値幅のみ取引” の設計（min_edge 強化）を行うと、  
Profit-first の理念が初めて学習に反映される。
