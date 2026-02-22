# 24. v457.2 Strategy Plan Review (Profit-First Retraining)

対象: `docs/v457/23_v457_2_strategy_plan.md`  
参照: `docs/v457/17_v457_enhancement_roadmap.md`, `docs/v457/22_v457_1_phase2_frequency_control_review.md`

## 1. 総評
Profit-first の方向性は妥当で、現状の「Tiny Edge + Fatal Cost」を正面から扱う意図は正しい。  
ただし、**既存実装が既に Net PnL を報酬にしている点と、報酬スケール/設定キーの不整合**が見落とされている。  
ここを整えないと「新しい設定を入れても実質効果が出ない」可能性が高い。

## 2. 重要な修正点・注意点
1) **Reward は既に Net PnL ベース**  
   - `FastIntradayEnvV456` は `compute_hft_reward()` を使い、`pnl - fee - slippage` が基本。  
   - 問題は「報酬の定義」ではなく、**報酬スケールとクリップがコストを埋没させている可能性**。  
   - `reward_scale` / `reward_clip` の再調整が必須。

2) **reward_settings のキー不整合**  
   - `reward_settings` は `compute_hft_reward()` の引数として渡る。  
   - そのため `fee_penalty_weight` / `holding_penalty` は **未対応キーで例外の原因**になる。  
   - 既存キーに寄せるか、`compute_hft_reward()` の拡張が必要。

3) **JSON コメントは実ファイルでは無効**  
   - `config/v457_2/train_config.json` で `//` コメントを使うと読み込み失敗する。  
   - `.jsonc` にするか、コメントを削除した正規 JSON にする。

4) **ent_coef だけでは張り付きが止まらない可能性**  
   - `ent_coef` は探索を促すが、コスト重視と同時に入れると学習が不安定になる。  
   - **行動抑制は環境側（min_delta / cooldown / max_delta）と合わせて設計**した方が安定。

5) **Hold 強化は “凍結復活” のリスク**  
   - `holding_penalty = 0` は HOLD の張り付き再発リスク。  
   - `hold_grace` + 低い `hold_ramp` で「長期保有は弱く罰する」構成の方が安全。

## 3. 既存実装の再利用候補（活用優先）
- `ztb/trading/rewards/fast_intraday.py`  
  - 既に Net PnL を採用。  
  - `min_edge_mult`, `edge_penalty_rate`, `vol_floor`, `vol_floor_penalty`, `hold_grace`, `hold_ramp` が使える。  
  - **Profit-first の「費用上回る値幅がない限り入らない」思想に直結**。

- `ztb/trading/environment/fast_intraday_env_v456.py`  
  - `min_delta`, `max_delta_per_step`, `cooldown_steps`, `max_ttl_steps` で取引頻度制御が可能。  
  - `reward_scale` / `reward_clip` を調整してコスト罰を強調できる。

- `ztb/trading/environment/utils/fast_intraday_env_v456_utils.py`  
  - env_config から `reward_settings` と env_kwargs を注入可能。  
  - v457.2 の `config/v457_2` と相性が良い。

- `ztb/training/utils/v457_config_utils.py`  
  - 設定読み込みの既存ユーティリティ。  
  - `train_v456_simple.py` に流用すれば v457.2 の外部 JSON を自然に取り込める。

- `ztb/trading/environment/factory_v456.py`  
  - Regime 特徴量生成が既存で入っている。  
  - 低ボラ期間の重点学習やサンプリング設計に活用可能。

## 4. 追加提案（計画を強化する観点）
- **報酬コンポーネントの可視化**  
  - `reward_info`（pnl, fee, slippage, penalty）をログ化し、  
    「コストがどれだけ学習に効いたか」を定量で確認する。

- **低ボラ期間の定義を明文化**  
  - ATR/price の分位点で「低ボラ期間」を定義し、  
    学習データの重みづけやサブセット学習に反映する。

- **評価指標の固定化**  
  - Net PF / Expectancy / Avg Win-Loss を固定出力し、  
    “Profit-first が本当に効いたか” を毎回比較できるようにする。

## 5. 推奨アクション（優先順）
1) `reward_settings` キーを **`compute_hft_reward()` 互換に修正**  
2) `reward_scale` / `reward_clip` を **コストが埋もれない値に調整**  
3) `train_v456_simple.py` に **v457.2 config 読み込みを組み込む**  
4) HOLD 張り付き防止のため **`hold_grace` + `hold_ramp` を最小限導入**
