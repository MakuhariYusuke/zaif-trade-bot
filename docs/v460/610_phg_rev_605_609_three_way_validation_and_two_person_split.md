# 610# 605#-609# 三者主張の検証と二名へのタスク再配分

- **日付**: 2026-03-25
- **目的**: 605#-609# の三者の主張をコード・設定・テストに照らして検証し、実作業を二人へ衝突なく再配分する
- **入力**: `605#`, `606#`, `607#`, `608#`, `609#`, `configs/v460/fill_test.yaml`, `scripts/v460/lib/*`, `scripts/v460/ml/sac_retrain_scheduler.py`, 関連ユニットテスト

---

## §0 総括

三者の中で最も堅いのは `607#`、最も発想価値が高いのは `608#`、最もバランス良く補正しているのは `609#` である。

ただし、現時点でそのまま実行計画に採れるのは一部だけで、次のように整理するのが安全である。

1. **605#-607# 著者**: 実装・運用感覚は強いが、`605#` の運用前提には古い値が混ざっていた。`606#` はその是正として有益だが、repo 現在値と食い違う箇所もある。`607#` は最も実証的で、今回の裏取りでも強い。
2. **608# 著者**: 問題設定の方向は良い。特に Attribution と Sidecar 再実効化の発想は価値が高い。一方、TTL・ceiling・Stage Max Mult など前提値の誤認があり、そのまま実装に降ろすと再び「ハルシネーション連鎖」になる。
3. **609# 著者**: 608# の補正役としてかなり優秀。ただし `in-process 推論は read_sidecar_signal を差し替えるだけ` という含みは楽観的で、live 側の feature contract 問題をやや過小評価している。

従って、作業は**二人**に対して以下のように切るのが良い。

- **担当A**: 実装・運用・runtime 検証担当（605#-607# 系）
- **担当B**: 分析・数理・設計仕様担当（608# 系）

`609#` は第三者レビューとして残し、**受け入れ基準の番人**に回すのが最も機能する。

---

## §1 三者の主張に対する判定

### 1.1 605#-607# 著者への判定

#### 支持できる点

- `607#` の hot-reload 再構築は実装裏付けがある
- `607#` の「fill_test は同期ループ」という構造理解は概ね妥当
- `606#` の「605# の stale assumption を訂正する必要がある」という問題意識は正しい
- `605#` の回顧文書としての価値は高い。何が解け、何がまだ残るかの棚卸しとしては有用

#### 補正が必要な点

**(A) 605# の Tier 0/Tier 1 は、そのまま実行計画にしてはいけない**

`605#` は `offset_ceiling_ratio_*`, `composite_risk_enabled`, `TTL`, `sell_dynamic_kill duration`, `Stage Max Mult` などで古い前提を含んでいた。これは `606#` が正しく訂正している。

**(B) 606# は「正誤表」としては有益だが、実装完了報告としては強すぎる**

今回の repo 確認では:

- `configs/v460/fill_test.yaml:375` `spread_anomaly_detector.enabled=false`
- `configs/v460/fill_test.yaml:386` `micro_circuit_breaker.enabled=false`
- `configs/v460/fill_test.yaml:1186` `entry_gate_enabled=false`

であり、`606#` が書く「After: enabled=true」「flat field → nested block 化」は**現在の repo 状態そのものではない**。一方で parser 側は nested `entry_gate:` を読める実装を既に持つ（`scripts/v460/lib/fill_config_parser.py:1162`）。

つまり `606#` は、

- **設計/変更意図としては良い**
- しかし **現 working tree の config 実態とはまだ一致していない**

という位置づけで読むべきである。

**(C) 607# は三者の中で最も堅い**

以下はコード実在を確認した。

- `scripts/v460/lib/config_hot_reload.py:630`
- `scripts/v460/lib/config_hot_reload.py:631`
- `scripts/v460/run_fill_test.py:664`
- `scripts/v460/run_fill_test.py:682`
- `scripts/v460/lib/spread_anomaly_detector.py:179`

加えて関連テストも通った。

- `tests/unit/v460/test_169_config_hot_reload.py`
- `tests/unit/v460/test_211_spread_anomaly_detector.py`
- `tests/unit/v460/test_211_micro_circuit_breaker.py`
- `tests/unit/v460/test_211_mcb_sad_escalation.py`
  - **80 passed**

したがって `607#` は「実装修正済み + テスト裏付けあり」の文書として扱ってよい。

### 1.2 608# 著者への判定

#### 支持できる点

- Attribution を最優先に置くのは正しい
- `581#` の additive pipeline を「理論」ではなく「実効」で評価しようとする姿勢は正しい
- Sidecar を retrain 成功時ファイル更新だけに依存させる構造が弱い、という認識は正しい
- Alpha 層を「生かす」ために実効反映率を見る、という視点は 0# の方針にも合う

#### 反証・補正

**(A) TTL 600s 前提は誤り**

- `scripts/v460/lib/sidecar_types.py:45` で `DEFAULT_SIGNAL_TTL_SEC = 7800.0`

この点は `609#` の指摘どおり。ただし 608# が狙った「有効率が低い理由の再定義が必要」という問題意識自体は残る。

**(B) Stage Max Mult 不在を前提にした議論は誤り**

- `606#` 正誤表の通り、2.0 cap は既に実装済み

したがって 608# は「いま multiplicative chain が uncontrolled に膨張している」というより、**cap 付きの multiplicative / additive 併存期にある**と読み替える必要がある。

**(C) `±1.0bps` への Action Range 拡大は早い**

608# の方向性は理解できるが、現 config は `configs/v460/fill_test.yaml:632` で `sidecar.max_boost_bps: 0.20`。この段階で 1.0bps へ飛ぶのは、既存ガードとの相互作用評価を飛ばしすぎる。

**(D) Nyquist-Shannon の持ち込みは不要**

これは 609# が正しく補正している。ここは signal processing の権威づけより、

- 観測更新頻度
- 推論コスト
- live feature availability

の3点で実務的に語る方が強い。

### 1.3 609# 著者への判定

#### 支持できる点

- TTL 問題の真因を「TTL 値そのもの」ではなく「retrain 成功依存」に置き直した点
- 608# の価値を消さず、誤った前提だけを潰している点
- Attribution を Phase 1 / Phase 2 に分けた点
- `608# 著者` と `605#-607# 著者` の性質に応じてタスク分離しようとした点

#### 補正が必要な点

**(A) in-process inference は「差し替えだけ」で済む話ではない**

`609#` は方向としてはかなり正しいが、難所を一つ軽く見ている。

- `scripts/v460/lib/orchestrator_mid_cycle.py:149` は現状 `read_sidecar_signal_with_status()` で JSON を読むだけ
- 一方、`scripts/v460/ml/sac_retrain_scheduler.py:1005` の signal 更新は training env の最新 observation を使う
- `scripts/v460/ml/sac_retrain_scheduler.py:87` は `btc_jpy_1m_full_registry_features.parquet` 前提
- `scripts/v460/ml/sac_retrain_scheduler.py:844` では `feature_columns` を env へ与えている

つまり、**live cycle 側には training env と同じ feature builder がまだ無い**。従って in-process inference は有望だが、「ファイル I/O を関数呼び出しに変えるだけ」の工数ではない。まず feature contract を切る必要がある。

**(B) 追加提案の中には今すぐやらない方が良いものもある**

- `Multi-Horizon SAC`: 早すぎる
- `Adversarial Scenario Testing`: 価値はあるが今の主要ボトルネックではない
- `eDRC 再活性化`: additive A/B が先

#### 609# に対する総評

`609#` は三者の中で**最も実務に近いレビュー文書**である。ただし「今すぐ実装できること」と「設計として正しいが前提整備が要ること」の線引きを、さらに明確にした方が良い。

---

## §2 追加で拾うべき盲点

### 2.1 config/document parity の崩れが続いている

今回の範囲だけでも、

- `605#` が stale config を前提にした
- `606#` がそれを正した
- しかし repo の current YAML は `606#` 文面とも一致していない
- `608#` は `606#` の訂正を十分に反映せず再び計画を立てた

という流れになっている。

これは実装ミスというより、**「effective config を機械的に可視化する仕組み」が不足している**ことが根本原因である。

### 2.2 Attribution は新スクリプト乱立より既存分析器拡張でやるべき

608# / 609# は Attribution Analyzer を別立てで考えているが、保守性の観点では

- `scripts/v460/analysis/analyze_fill_logs.py`

に `offset_stages` / `executor_offset_stages` セクションを追加する方がよい。既に同スクリプトは additive 判定や executor stage JSON を読む基盤を持っている。

### 2.3 inventory-aware ceiling は有望だが、既存 inventory 制御と二重化しやすい

609# の追加提案 `Risk-Budget-Aware Ceiling` 自体は面白いが、現行には既に `inventory_skewing` 系の思想がある。ここにさらに ceiling 側の inventory 連動を足すと、

- side selection
- offset adjustment
- ceiling override

の三重在庫制御になり、保守性が落ちる。

### 2.4 120s cycle を Binance HFT と比較しすぎない方がよい

609# の `Decision Latency Budget` は、**測ること自体**は支持する。ただし現システムは Coincheck / maker / sync-loop 前提であり、<100ms HFT を直接ベンチマークにすると議論が滑る。まずは

- ファイル I/O
- 推論
- 価格計算
- REST API

の内訳をログ化して、現実的な改善余地を測るのが先である。

---

## §3 二人へのタスク再配分

### 3.1 担当A: 実装・runtime・検証担当（605#-607# 系）

担当A は「今 repo にあるものを正しく動かし、誤差を消す」仕事に集中するのがよい。

| # | タスク | 優先度 | 根拠 |
|---|---|---|---|
| A1 | `606#` と current YAML の差分是正。少なくとも `spread_anomaly_detector`, `micro_circuit_breaker`, `entry_gate` の文書と実 config を一致させる | P0 | 現状 repo と文書がズレている |
| A2 | `607#` hot-reload 修正の live 検証。`true -> false -> true` の切替で MCB/SAD state 継承と効力変化を記録 | P0 | 実装済み資産の本番検証 |
| A3 | `experimental_additive_pipeline.enabled` の A/B を実施し、既存 `analyze_fill_logs.py` に clamp_rate / info_loss を追加 | P1 | 608#/609# の中心仮説を最短距離で検証 |
| A4 | 起動時と hot-reload 時に critical config snapshot を INFO ログへ出す | P1 | 605#→608# のハルシネーション連鎖を断つ |
| A5 | `entry_gate` は nested block を使うなら YAML 移行まで完了、やらないなら flat flag 維持と明記 | P1 | 606# の文書先行状態を解消 |

### 3.2 担当B: 分析・数理・仕様担当（608# 系）

担当B は「理論を現在値に合わせて正しく定義し直す」役割に集中するのがよい。

| # | タスク | 優先度 | 根拠 |
|---|---|---|---|
| B1 | `608#` を current repo 前提で改稿。TTL 600s, ceiling 0.15-0.35, Stage Max Mult 未実装などの誤前提を除去 | P0 | 設計文書の土台修正 |
| B2 | Attribution 指標定義を既存ログ前提に落とす。`offset_stages` と `executor_offset_stages` から何を取るかを仕様化 | P1 | 新規スクリプト乱立を防ぎつつ実装に渡せる |
| B3 | in-process inference の前に、live 側で供給可能な feature と `sac_retrain_scheduler` の `feature_columns` の差分表を作る | P1 | 609# の楽観を埋める最重要前提 |
| B4 | Sidecar action range は `0.20 -> 0.30 -> 0.50bps` の ladder と abort 条件まで定義し、`1.0bps` は保留にする | P1 | 現段階では段階投入が妥当 |
| B5 | realized spread persistence / toxicity EWMA を既存 Toxicity Budget とどう接続するかを整理 | P2 | 609# 追加提案の中で比較的 ROI が高い |

### 3.3 第三者レビュー役（609# 系）

ユーザー要望どおり実作業は二人に割るべきなので、`609#` 著者には**実装 ownership を持たせない**方がよい。役割は以下で十分である。

- A/B 実験の acceptance criteria 定義
- `608#` 改稿版の factual review
- additive A/B と sidecar feature contract のレビューゲート

つまり、**実装者ではなく検収者**に置くのが最も噛み合う。

---

## §4 その他の提案

### 4.1 提案P1: runtime config 自己申告

毎起動 / hot-reload ごとに以下を1行で出す。

- `git_sha`
- `config_hash`
- `offset_ceiling_ratio_buy/sell`
- `experimental_additive_pipeline.enabled`
- `spread_anomaly_detector.enabled`
- `micro_circuit_breaker.enabled`
- `entry_gate_enabled`
- `sidecar TTL / max_boost_bps`

これだけで 605#→606#→608# のような認識齟齬の多くが止まる。

### 4.2 提案P2: Attribution は `analyze_fill_logs.py` 内へ統合

新しい `attribution_analyzer.py` を増やすより、既存 `analyze_fill_logs.py` に以下の節を足す方が保守しやすい。

- `pre_clamp vs post_clamp`
- `executor_offset_stages` の stage 使用率
- `tox_buffer / liq_buffer` 分布
- sidecar boost の clamp 到達率

### 4.3 提案P3: in-process inference の前に feature contract テストを作る

Sidecar を live cycle に入れるなら、先に

- required feature list
- live で供給可能な feature list
- 欠損時 fallback

をテストで固定した方がよい。ここを飛ばして推論を組み込むと、再び silent neutral 化が起きる。

---

## §5 結論

- **605#**: retrospective として有用だが、実行計画は stale 前提を含む
- **606#**: 方向は良いが、repo 現在値と文書がまだズレている
- **607#**: 今回範囲で最も堅い。実装とテストが伴っている
- **608#**: 発想価値は高いが、現状のまま実装計画にすると危険
- **609#**: 補正として優秀。ただし in-process inference の実装難度をやや軽く見ている

次に進むなら順序はこれがよい。

1. **担当A** が config/document parity と additive A/B の計測土台を固める
2. **担当B** が 608# を current repo 前提で改稿し、live feature contract を定義する
3. **609# 系** が acceptance reviewer として実験条件と完了基準を固定する

この順なら、二人の作業がぶつからず、三者の強みだけを残せる。
