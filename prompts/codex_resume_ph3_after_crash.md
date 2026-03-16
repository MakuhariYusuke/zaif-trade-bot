# Codex 再開プロンプト: ph3 Sidecar 作業の続き

あなたは `zaif-trade-bot` リポジトリで、IDE クラッシュにより中断した ph3 Sidecar 系作業の続きを担当する Codex です。

## 最重要前提

- **profit-first** を守ること
- **別のCodexが `prompts/codex_test_cleanup_and_perf.md` に基づいてテスト整理を進めている**ため、原則として `tests/`, `pytest.ini`, その prompt には触れないこと
- まず `git status --short` を確認し、**自分が行う変更と無関係な差分は触らないこと**
- 既存の未コミット作業を巻き戻さないこと

## まず読むべき文書

優先順:

1. `docs/v460/377_ph3_unified_direction_course_correction.md`
2. `docs/v460/375_ph3_rev_369_374_profit_first_and_sac_scope_review.md`
3. `docs/v460/376_gemini_ph3_comprehensive_review_and_v456_revival.md`
4. 必要に応じて `docs/v460/374_ph3_design_sac_continuous_value_and_market_theory_integration.md`
5. `docs/v460/index.md`

## ここまでで確認できている状況

- `375#` では 374# を profit-first で是正済み
  - `3.1 = 縮小版でのみ GO`
  - `3.2 = HOLD`
  - `3.3 / 3.4 = NO-GO`
- `376#` では追加で以下が指摘されている
  - `build_features.py` に M2-M5 事前計算が無い
  - `FastIntradayEnvV456` が埋もれている
- `377#` では **「Phase 3.1 コード実装完了、SAC live 起動待ち」** まで進んでいる
  - 文書上は `374# impl 82675725d` と記載あり
  - ただし retained artifacts 上は **SAC live presence 0/4** のまま

## 377# から推定される主残作業

### P0: Phase 3.1 が本当に live で動くかの確認

以下を確認し、未達なら必要最小限の修正を行うこと。

1. `scripts/v460/lib/sidecar_types.py`
   - `compute_sidecar_offset_bps_v2()`
   - `_shaping_fn()`
   - default が `max_boost_bps=0.15` 系に収束しているか

2. `scripts/v460/lib/fill_config.py`
3. `scripts/v460/lib/fill_config_parser.py`
4. `scripts/v460/lib/config_hot_reload.py`
5. `scripts/v460/lib/cycle_gate_aggregator.py`
6. `scripts/v460/lib/fill_config_validation.py`
7. `configs/v460/fill_test.yaml`

上記が 377# 記載どおり実装済みかを確認する。

### P1: SAC scheduler を実際に 1 回起動し、artifact を生成させる

想定コマンド:

```bash
./.venv/Scripts/python.exe scripts/v460/ml/sac_retrain_scheduler.py --config configs/v460/experiments/g2_sac_train.yaml --once
```

確認対象:

- `cache/sidecar_signal.json` が生成されるか
- `logs/sac_retrain_history.jsonl` が生成されるか
- 実行失敗なら、原因が
  - config 読込
  - parquet 読込
  - env 作成
  - model load/save
  - signal write
  のどこかを特定すること

### P2: Phase 3.1 live presence の 4 項目確認

377# §7.1 の 4 項目を現実に埋めること。

1. `cache/sidecar_signal.json` 更新
2. `logs/sac_retrain_history.jsonl` 履歴
3. `fill_records` に `sidecar_offset_bps` / `sidecar_bias` non-null が出るか
4. `fill_test.log` に sidecar 関連ログが出るか

必要なら以下も確認:

- `results/v460/fill_test/logs/fill_test.log`
- `results/v460/fill_test/fill_records_*.jsonl`

### P3: P0-P2 が通ったら、最小限の文書更新

- 原則は `docs/v460/377_ph3_unified_direction_course_correction.md` の末尾追記でよい
- もし更新量が大きいなら **次番ドキュメント** を新規作成し、`docs/v460/index.md` を更新する
- 文書には以下を必ず書くこと
  - 何を確認したか
  - どこで失敗したか / 成功したか
  - live presence が何項目埋まったか
  - 次にやるべきこと

## 今はやらないこと

以下は 375# / 377# の判断により **まだ着手しない**。

- 3.2 の M2-M5 parquet 統合の本格実装
- 3.3 multi-dimensional action
- 3.4 closed-loop reward
- `FastIntradayEnvV456` の本格復活
- 大規模 trainer 再統合
- テスト整理タスクへの介入

## ただし例外

もし P1 の scheduler 起動が、明らかに以下の軽微問題で止まる場合は修正してよい。

- import path のズレ
- config key 名のズレ
- signal/history path 作成漏れ
- 軽微な logging / metrics key ミス

ただし、**大工事になりそうなら止めて、現状と blocker を文書化すること**。

## レビュー観点

- `375#` の profit-first を破らないか
- Sidecar の数値感が過大になっていないか
- 実装済みと文書記載が食い違っていないか
- retained artifact / log / fill_records で裏取りできているか

## 成果物

最低限ほしい成果物:

1. 実コード確認結果
2. scheduler 実行結果
3. sidecar live presence 4項目の達成/未達
4. 文書更新 (`377` 追記 or 新規 doc)
5. `index.md` 更新（新規 doc の場合のみ）

## 補足

- `377#` には「build_features.py M2-M5 proxy 追加着手」とあるが、**今はそこを本作業の主対象にしないこと**
- 今回の本丸は **Phase 3.1 が本当に live に存在する状態まで持っていくこと** である
- まずは「動いている証跡」を作ること。利益検証はその次
