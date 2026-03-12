# v458 追加レビューと提案 (01)

## 1. 重要な欠落・不整合（要修正）
- `scripts/v458/train_v458_main.py` が実行不能。`create_fast_intraday_env_v456` の呼び出し引数が不一致、`SimpleTrainingCallback` が `self.model.episode_rewards` を参照（SB3に存在しない）、末尾に `</content>` 等の混入があり SyntaxError になる。
- `tests/verification/test_v458_features.py` も末尾タグ混入で SyntaxError。加えて、MTF因果性の検証が弱く、仕様担保になっていない。
- 仕様書の観測次元内訳が実装とズレている。実装は `30(base)+27(mtf)+6(cyclical)+6(global)+13(regime)+6(account)=88` だが、`00_project_proposal_v458.md` では account が未記載で `Regime/Global=25` と記載。
- `reward_scale`/`reward_clip` 前提が不一致。仕様は `reward_scale=1.0` だが、実装デフォルトは `reward_scale=100000` + `reward_clip=1.0`。`reward_scale=1.0` だとほぼ全報酬がクリップされる可能性が高い。
- `guidance_decay_steps` が config から渡せる導線が弱い。`create_fast_intraday_env_v456` の許可キーに未登録で、設定変更が反映されない。

## 2. 短期修正タスク（MVP整備）
- v458専用 config を新規作成（例: `config/v458/base/config.yaml`）。
  - `training.environment.action_space_type = "1d_position"`
  - `guidance_decay_steps` を明示
  - `reward_scale` と `reward_clip` を整合（1.0にするなら clip を外す／拡大）
  - `sac_hyperparameters` と seed を明示
- `scripts/v458/train_v458_main.py` を v457 方式へ寄せる。
  - `create_fast_intraday_env_v456(df=df, env_config=env_config)` で呼ぶ
  - Callback は v457 の `SimpleTrainingCallback` を流用
  - 末尾タグ混入の除去、`env is None` のガード追加
- `tests/verification/test_v458_features.py` を修正。
  - 末尾タグ混入の除去
  - MTF因果性は「全データで算出した特徴」と「t時点までのデータで再算出した特徴」の一致チェックで検証
  - Guidance decay は early/late で penalty 影響が単調減少することを統計的に検証
  - `action_space_type` は training と一致させる
- `create_fast_intraday_env_v456` の `known_utils_keys` に `guidance_decay_steps` を追加し、config反映を保証。
- `docs/v458/00_project_proposal_v458.md` に観測次元内訳・reward scale/clip の前提を追記。

## 3. 実験・検証の提案
- Seed 3本（42/123/777）に加えてもう1本を追加し、崩壊判定基準を明文化（PnL>0, DD<xx%, トレード数>yy）。
- Ablation: `guidance_decay_steps`（0/50k/100k）と Guidance無効の比較。
- 漏洩版（v457.4）との A/B 比較で「性能低下の妥当性」を確認。
- 時系列分割（train/val/test）を明示し、`scripts/v457/backtest.py` を流用した OOS 評価を標準化。
- 主要メトリクスを追加出力（reward components、penalty内訳、action分布、TTL発動回数）。

## 4. データ/特徴量の整合
- `timestamp` のタイムゾーン統一（JST/UTC）。Cyclical features の「危険時間帯」分析と一致させる。
- Global features を使う場合は `GlobalMarketFeatureEngineerV456` 連携を実装。使わないなら観測から外すかスケーラ対象から外す。
- MTF の `ffill` に上限（欠損ギャップ制限）を入れ、長い欠損区間の特徴量伝播を抑制。

## 5. 既存知見の再利用候補（中期）
- v455: Calibration Gate / Realistic Execution Model の再接続（live/backtest の乖離抑制）。
- v456: Soft Filter / regime-aware sizing / Cyclical Time Features の再評価。
- v457: Anti-freeze reward のパラメータと config schema の踏襲。
