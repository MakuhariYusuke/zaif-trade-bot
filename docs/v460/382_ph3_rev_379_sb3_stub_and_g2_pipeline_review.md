# 382# ph3 レビュー: SB3 スタブ修正と G2 SAC パイプライン再点検

**Date**: 2026-03-12  
**Scope**: `prompts/382_codex_sb3_stub_review.md`, `docs/v460/000`, `356`, `358`, `379`, 現行ワーキングツリー実装  
**Conclusion**: SB3 シャドウイングの根本修正自体は妥当。ただし、現行の G2/OOS 評価経路には gate 判定を歪める blocking issue が残っており、提示された post-fix 数値はまだ gate-grade の根拠として扱わない方がよい。

---

## 1. 主要所見

### CRITICAL-1: checkpoint 評価が training env を直接進めており、SB3 の内部 rollout 状態と衝突しうる

**根拠**:
- `scripts/v460/lib/tasks/sac_train.py:307-315` で `model.learn(..., reset_num_timesteps=False)` の直後に同じ `env` を `_checkpoint_eval_roi()` へ渡している
- `scripts/v460/lib/tasks/sac_train.py:349-358` で `_checkpoint_eval_roi()` がその `env` を `reset()` / `step()` している

**問題**:
- SB3 の off-policy 学習は `reset_num_timesteps=False` 時に env を連続使用する前提で内部状態を持つ
- その途中で外側から同じ env を進めると、モデル側が保持する rollout 状態と env 実状態がずれる
- これは checkpoint ROI の信頼性だけでなく、その後の学習自体も汚染する

**推奨**:
- checkpoint 評価は training env と完全分離した専用 eval env で行う
- あるいは checkpoint callback + cloned env に寄せ、学習ループ外から training env を触らない

### CRITICAL-2: OOS 評価が「holdout 全体」ではなく、同じ先頭 10K step を 3 回繰り返している

**根拠**:
- `configs/v460/experiments/g2_sac_train.yaml:95,103` で `val_ratio: 0.2`, `n_episodes: 3`
- `scripts/v460/lib/sac_common.py:116-139` で各 episode ごとに `env.reset()` 後、`max_steps_per_episode=10_000` で打ち切っている
- `ztb/trading/environment/heavy_env/core.py:667-699` で `random_start=False` のとき `current_step=0` に戻る

**問題**:
- 各 episode は毎回 validation slice の先頭から始まるため、実質的に同じ 10K step を 3 回再生している
- `trade_count` は 3 回分加算される一方、`gross_roi` は同一区間の平均、`gross_pnl` は最終 episode だけ (`scripts/v460/lib/sac_common.py:133-139`)
- したがって prompt 記載の `OOS gross_roi` / `trade_count` は holdout 20% 全体の評価値ではなく、しかも内部整合も取れていない

**補足**:
- `evaluate_model_oos()` の多エピソード挙動はローカルで再現確認済み。`trade_count` は総和、`gross_pnl` は最後の episode の値になる

**推奨**:
- G2 用 OOS は原則 1 pass で holdout 全区間を走査する
- 速度都合で上限を残すなら、episode ごとに非重複 window を明示的に切る
- `gross_pnl` も `gross_roi` と同じ集約規約に統一する

### HIGH-1: 1 seed がクラッシュしても G2 が PASS しうる

**根拠**:
- `scripts/v460/run_experiment.py:245-253` で seed 失敗時に `gross_roi=0.0` を入れて継続
- `scripts/v460/run_experiment.py:327-366` の G2 判定は `error` を見ず、ROI のみで判定

**影響**:
- 3 seed が正、1 seed が例外でも `positive_seed_ratio=0.75` を満たしうる
- `worst_seed_roi` も `0.0 > -0.02` で通る
- 実際に `_evaluate_g2_from_results()` をローカル実行すると、このケースは `PASS` になった

**推奨**:
- 1 seed でも例外が出たら `gate_result=FAIL` または `ERROR`
- 少なくとも `n_successful_seeds == len(seeds)` を必須条件に加える

### HIGH-2: validation env が OOS データ自身で scaler を再学習しており、評価条件が training 時と不一致

**根拠**:
- `scripts/v460/lib/tasks/sac_train.py:120,156,224-235` は train 用 / val 用で別々に `_create_training_env()` を呼ぶ
- `ztb/trading/environment/heavy_env/core.py:546-555` は `config.train_end_index` が無ければ env 内で scaler を計算
- `ztb/trading/environment/heavy_env/mixins/initialization.py:556-566` はその場合「entire dataset を使うので leakage の恐れ」と明記
- `scripts/v460/lib/config_loader.py:150-173` では `data.train_end_index` を必須扱いにしているが、`task_sac_train()` 側では env へ注入していない

**問題**:
- train env は train slice 統計、val env は val slice 統計で正規化される
- これは「train 時と異なる feature scale で推論している」状態で、OOS の純度も再現性も落ちる

**推奨**:
- train env で計算した `scaler_mean/std` を val env へ引き渡す
- もしくは `train_end_index` 契約を `EnvironmentConfig` に正しく配線する

### HIGH-3: `_sb3_test_stub/` を残すかどうか以前に、テスト基盤全体が SB3 を広域にモックしている

**根拠**:
- `ztb/support/sb3_compat.py:14-45` は import 失敗時に `stable_baselines3` をその場で生成する
- `tests/conftest.py:46-52` で `ensure_sb3_compat()` を常時実行
- `tests/conftest.py:308-388`, `587-623`, `1114-1129` などで SB3 の stub / repair を多重に注入
- 一方で本番経路の `scripts/v460/lib/tasks/sac_train.py:77-83,131-146,266-283` と `scripts/v460/diagnose_sac_actions.py:15-18,45-47` は `import_real_sb3()` を使わず、通常 import に戻っている

**問題**:
- `_sb3_test_stub/` を削除しても、CI/pytest 側のマスキングは残る
- 「SB3 が壊れていてもテストだけ通る」再発経路がまだ存在する

**推奨**:
- `_sb3_test_stub/` は本番コードから完全分離し、必要ならテスト専用 fixture へ移す
- `tests/conftest.py` の global stub 注入を縮小し、必要テストだけ局所 mock に寄せる
- SAC 実行系は direct import をやめて import helper を一元化する

### MEDIUM-1: 仕様と実験前提のドリフトが残っている

**根拠**:
- `docs/v460/000_ph0_plan_project_proposal.md:30-32` は「maker-only / 手数料 0%」を前提にしている
- しかし `configs/v460/experiments/g2_sac_train.yaml:65` は `transaction_cost: 0.001`
- `docs/v460/000_ph0_plan_project_proposal.md:189-190` の E2 は IC seed std、現実装 `configs/v460/gate_thresholds.yaml:61-66` は ROI seed std

**問題**:
- 今回の「ROI が負なのは 0.1% コストのため」という解釈は、少なくとも 000 提案書とは前提が一致していない
- G2 仕様の更新も proposal に完全反映されていないため、文書横断で判定軸が揺れている

**推奨**:
- `000` を SSOT として更新するか、逆に「現行運用仕様は 000 から変更済み」と明示する
- 0% maker 前提で評価したいのか、Coincheck 0.1% を現実前提にするのかを gate 定義から固定する

### LOW-1: `sitecustomize.py` と関連文書に後片付け不足がある

**根拠**:
- `sitecustomize.py:22-67` は SB3 関連ロジックが実質 dead code
- `docs/v460/379_pre366_sac_integration.md:5` は `2026-03-25` と `2026-03-11` が併記され、日付整合が取れていない

**所見**:
- 直接の機能バグではないが、再発防止と調査容易性の観点では削る方がよい

---

## 2. 依頼事項への直接回答

### 6.1 `sitecustomize.py` のデッドコード

- SB3 固有の `_prefer_local_package()` と `_replace_stub_with_filebacked()` は完全削除でよい
- `sitecustomize.py` 自体は、今の内容だけなら不要
- ただし `CHANGELOG.md:2585` にある「Windows torch bootstrap」のために残したいなら、その責務だけを持つ最小実装へ縮退すべき

### 6.2 `_sb3_test_stub/` の存続判断

- 本番コード配下に置く必要は薄い
- ただし削除前に `tests/conftest.py` と `ztb/support/sb3_compat.py` の global stub 注入を先に整理しないと、問題の本体は残る
- 結論としては「`_sb3_test_stub/` 削除は賛成。ただし順番はテスト基盤の局所 mock 化が先」

### 6.3 `import_real_sb3()` の信頼性

- 一時的な quarantine としては許容
- ただし end-state には向かない。`sys.modules` purge / `sys.path` 手術は副作用が強い
- `__version__` チェック単独は弱いので、残すなら `sb3.__file__` が project root 配下でないことまで確認した方がよい
- さらに重要なのは、helper を作ったのに `task_sac_train.py` が使っていない点

### 6.4 パフォーマンス修正の妥当性

- `inspect.signature()` キャッシュ自体は妥当。`__init__` 初期化に寄せると型と可読性が少し良くなる程度で、優先度は低い
- `_CHECKPOINT_EVAL_MAX_STEPS=5_000` と `max_steps_per_episode=10_000` は「粗い診断」なら可だが、「gate 判定」には短すぎる
- 特に現状は短いだけでなく、同じ prefix を再利用しているため評価バイアスが大きい

### 6.5 閾値 `0.10` の妥当性

- 暫定 stopgap としては理解できる
- ただし post-fix の実 SB3 行動分布に基づく再較正がまだ不足している
- `adaptive_threshold_mode` や学習化は、まず評価経路を正してからでよい。今は閾値改善より測定系修正が先

### 6.6 訓練パラメータの改善方向

- 17 feature / 1.2M 行規模の SAC として 50K steps は短い
- ただし今は学習長を増やす前に、checkpoint/OOS 評価の妥当性を直す方が先
- コスト 0% 学習は「方向性 alpha を見る」目的なら一案だが、仕様書との整合を先に決めるべき
- curriculum も同様で、計測系が安定してから導入判断すべき

---

## 3. 優先アクション

1. checkpoint 評価を training env から切り離す
2. OOS 評価を「holdout 全区間 1 pass」へ変更し、`gross_roi/trade_count/gross_pnl` の集約規約を整理する
3. seed 例外時は G2 を即 FAIL/ERROR にする
4. train scaler を val env に再利用する
5. SB3 stub をテスト専用 fixture に閉じ込め、`sitecustomize.py` / `_sb3_test_stub/` / `tests/conftest.py` を整理する

---

## 4. 検証メモ

- 対象 docs: `000`, `356`, `358`, `379`, `index.md`, prompt `382`
- 対象 code: `sitecustomize.py`, `ztb/support/sb3_compat.py`, `scripts/v460/lib/sac_common.py`, `scripts/v460/lib/tasks/sac_train.py`, `scripts/v460/run_experiment.py`, `ztb/trading/environment/...`
- 実行確認:
  - `pytest tests/unit/v460/test_356_g2_sac_blockers.py -q` は **45 tests passed**
  - ただしコマンド全体は repository-wide coverage threshold 未達で exit 1
  - 現行テストは上記 critical/high issue を捕捉していない
