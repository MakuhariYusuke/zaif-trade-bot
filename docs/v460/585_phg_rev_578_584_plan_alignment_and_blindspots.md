# 585# [review] 578#-584# 計画整合レビューと盲点補完

> **Date**: 2026-03-24  
> **Scope**: 578#-584#, 000#, 関連実装, 既存分析ログ  
> **Conclusion**: 578#-583# は方向性としては概ね妥当。ただし「実装済み」と「live で実証済み」が混在している。584# は 000# との位相合わせと優先順位の補正が必要。

---

## 1. エグゼクティブサマリ

今回の 578#-584# で見えている流れは大筋では正しい。
特に 578# の監査観点、579# の「乗算鎖のブラックボックス化」批判、583# の責務分割は支持できる。

一方で、次の 4 点はそのまま進めると詰まりやすい。

1. **000# とのフェーズ整合が崩れている**。584# の `ph6` は 000# 上の公式フェーズではない。
2. **実装完了と運用実証を混同している**。580#-582# はコード上の修正と live 観測の成立がまだ一致していない。
3. **A/B 判定の可観測性が未確立**。584# の additive 検証基準は、現状の telemetry だけでは信用し切れない。
4. **優先順位が少し逆転している**。000# は SAC を Driver ではなく Sidecar と定義しており、Execution 層の是正より先に P6 をクリティカルパスへ置く論拠は弱い。

---

## 2. 強く補正したい点

### 2.1 584# は 000# を改訂しない限り「ph6」ではなく ph5.x の改善計画として扱う方が安全

000# では正式フェーズが `ph0`-`ph5` と `ph3.1` までしか定義されていない。`ph6` を先に立てると、Gate の因果追跡と index 管理がずれやすい。

- 000# §2 では `ph5 = Paper trading 運用検証`
- 584# の内容は実質的には **paper trading 中に見つかった Execution 不全の是正** であり、性質としては `ph5 remediation` か `ph5.5` に近い

**判定**:
- 584# の問題意識自体は妥当
- ただし位相は `ph6` より **`ph5.x 再整備`** の方が 000# と整合する

**推奨**:
- 584# をそのまま使うなら、先に 000# Appendix A へ「ph5.5 / ph6 追加」の改訂を入れる
- 000# を触らないなら、584# は `ph5 remediation` として再ラベルした方が安全

### 2.2 579#-582# の「修正済み」はコード上ではかなり進んでいるが、live 実証済みとは言い切れない

578# が指摘した 2 本柱、すなわち

- `spread_capture_bps` / `adverse_selection_cost_bps` の永続化漏れ
- `edrc_hard_cap` の適用順序

について、**repo 上のコード修正自体は確認できた**。

- `ztb/metrics/fill_quality.py` に `spread_capture_bps` / `adverse_selection_cost_bps` が存在
- `scripts/v460/lib/fill_config.py` では `hour_ceiling_mult` の後に `edrc_hard_cap` を適用

しかし、既存分析スクリプトで 2026-03-23 のデータを確認すると、依然として次の状態だった。

- `analyze_fill_logs --date-from 2026-03-23 --date-to 2026-03-23`
  - `spread_capture_bps 未記録 (0/90 fills)`
  - `Buffer Decomposition: (no additive pipeline data — tox_buffer not found in stages)`

つまり現状は、

- **コード修正は前進**している
- だが **live JSONL で期待どおりに観測できたことまでは未証明**

という整理が正しい。

**判定**:
- 578# の監査論点は支持
- 579#-582# の「完了」表現は少し強すぎる

**推奨**:
- 580#-582# は「実装完了」ではなく **「コード反映済み・live 可観測性は要再確認」** と書く方が安全
- Phase 6 相当へ進む前に、same-run / same-SHA で `spread_capture_bps` と `tox_buffer` / `liq_buffer` の出現確認を先に置く

### 2.3 584# P1 の A/B 検証は、現状の telemetry 前提のままだと判定を誤る可能性がある

584# は additive pipeline の A/B を既存の `analyze_fill_logs.py` で比較する方針だが、ここはまだ前提が甘い。

2026-03-23 の fill records を見ると、`execution_additive_enabled=true` のレコードは存在する一方、`executor_offset_stages` 側は依然として multiplicative 風の JSON で、`tox_buffer` / `liq_buffer` が入っていないものが残っている。

さらに repo 上の現行コードを読むと、`scripts/v460/lib/fill_cycle_executor.py` の `_build_fill_record(...)` 呼び出しでは `execution_sigma` / `execution_adverse_ofi` / `execution_additive_enabled` を明示的に渡していない。

この 2 点を合わせると、少なくとも今は

- **実行系 runtime**
- **リポジトリ上の現行コード**
- **分析スクリプトの分類ロジック**

の 3 者が完全に揃っているとは言い切れない。

**判定**:
- 584# の P1 自体は最優先候補でよい
- ただし **今の telemetry のままでは additive vs multiplicative の attribution を誤る恐れがある**

**推奨**:
- P1 着手前に「判定の母集団」を固定する
  - same-SHA
  - same-run_id
  - process restart 済み
- `Execution Quality Comparison` の結果を意思決定に使う前に、少なくとも 1 run 分は `tox_buffer` / `liq_buffer` が JSONL に出ていることを確認する

### 2.4 584# の「YAML を切り替えるだけで A/B」が成り立つとは限らない

584# は additive A/B を比較的軽く始められる前提で書かれているが、repo 上では

- `experimental_additive_pipeline` / `edrc_*` は `fill_config_parser.py` で読み込まれる
- しかし `config_hot_reload.py` の hot-reload 対象には入っていない

そのため、**実行中プロセスに対する YAML ホットスワップだけでは additive/eDRC 切替が反映されない可能性が高い**。

同様に `entry_gate_*` も hot-reload 対象には見当たらない。

**判定**:
- 584# の「コード変更不要」は概ね正しい
- ただし **restart / run_id 分離 / 反映確認** は必要

**推奨**:
- 584# P1/P4/P5 には「プロセス再起動 or 新 run_id での実施」を明記する
- 000# の運用規約どおり、変更前後データは混在させず分離する

### 2.5 P6 retrain_scheduler は「未起動」より「fresh/stale/error の安定化」の問題として再定義した方がよい

584# は `cache/sidecar_signal.json` が neutral のままという前提で P6 を置いているが、2026-03-24 時点の `cache/sidecar_signal.json` は既に neutral ではない。

また 2026-03-23 の fill records では `sidecar_signal_status` が混在していた。

- `fresh = 25`
- `stale = 57`
- `error = 19`

したがって今の問題は

- signal が存在しない
- neutral しか出ていない

ではなく、むしろ

- **fresh で安定供給されない**
- stale / error がまだ多い

という可用性問題に近い。

しかも 000# §0.1 は SAC を **Driver ではなく Sidecar** と定義している。よって Execution 層の構造欠陥が残る段階で P6 を最上位クリティカルパスに置くのは、000# の設計思想とも少しズレる。

**判定**:
- P6 自体は必要
- ただし役割は「本番 Execution 改善の主役」ではなく **alpha 補助系の安定化**

**推奨**:
- 584# P6 は「neutral fallback 解消」から「fresh/stale/error 改善」へ書き換える
- 優先順位は P1 の下、または P1 と独立並行に置くのが自然

### 2.6 584# P2 Smart Preflight は考え方は良いが、設計スケッチがまだ薄い

P2 は `preflight_insufficient` を減らしたいという着眼点自体は非常に良い。
ただし提案されている pseudo-code は、そのままでは入らない。

現状確認できた事実:

- `scripts/v460/lib/orchestrator_balance.py` に `_resolve_balance_and_preflight(...)` は存在
- `scripts/v460/lib/maker_price.py` には inventory skew の tanh 平滑化が存在
- しかし `get_inventory_skew_score()` は存在しない
- `smart_preflight_enabled` / `preflight_skip_inv_threshold` も未定義

つまり P2 は「小変更」ではなく、**新しい判定面を追加する設計タスク** である。

加えて、ここは市場理論だけでなく Gate 理論の注意も必要。
preflight を早めに skip すると API コール削減には効くが、`G1.2-full F1/F1b` を改善するとは限らない。下手をすると単に attempted 母数を減らし、見かけ上の rate だけ動かす危険がある。

**推奨**:
- P2 をやるなら、まず観測を足す
  - `preflight_candidate_but_skipped`
  - `skip 後に opposite side で fill した率`
  - `preflight skip による lost opportunity`
- 成功基準は `preflight_insufficient 率` だけでなく、`F1/F1b`, `sum_pnl`, `attempted 母数` をセットで見る

---

## 3. 中重要度の補強・反論

### 3.1 580# / 581# の「True Additive」は現状では Execution 後段に限定して書く方が正確

582# の additive 実装は repo 上で確認でき、RMS 結合のテストも通っている。
しかし、`maker_price.py` 側では依然として

- `inv_skew`
- `as_shift`
- `regime`
- `spread_adapt`
- `kyle`
- `amihud`

といった前段ステージが逐次適用されている。

したがって、現状は **end-to-end の真の加法化** ではなく、より正確には

> 「Execution offset pipeline の additive 化」

である。

**支持**:
- 乗算爆発を後段で抑える方向性は妥当

**反証・補足**:
- 文書表現として「システム全体の true additive」と読むと強すぎる

### 3.2 582# の Toxicity / Liquidity 分離は設計方向として正しいが、成功基準 `相関 < 0.5` は弱い

相関が低いことは「別のものを測っている」ヒントにはなるが、収益改善の証明にはならない。

profit-first で見るなら、本当に欲しいのは次の比較。

1. `tox_buffer` 上位デシルで AS 率が上がるか
2. `liq_buffer` 上位デシルで spread_capture が増えるか
3. `tox_buffer` が高いときに quote を引いた方が純利益が改善するか

**推奨**:
- V4 は `相関 < 0.5` を補助指標へ降格
- 主判定は `AS率`, `spread_capture`, `post_fill_30s_pnl`, `clamp率` の条件付き比較へ置き換える

### 3.3 579# の refactoring 提案は良いが、責務の置き場所は既存分割に合わせた方がよい

579# は `ztb/trading/inventory_manager.py` のような統合ヘルパーを提案しているが、現行 v460 の責務分割は

- 在庫連動 offset: `scripts/v460/lib/maker_price.py`
- 残高 / preflight: `scripts/v460/lib/orchestrator_balance.py`

に寄っている。

この状態で新しい `inventory_manager` を中央集権的に足すと、逆に god object を再生成しやすい。

**推奨**:
- 新クラスを 1 個増やすより
  - `maker_price.py` 側に inventory signal API を追加
  - `orchestrator_balance.py` 側に preflight policy を切り出す
- という 2 点分割の方が SRP を守りやすい

### 3.4 584# の基準線数値は、必ずフィルタ条件を明記した方がよい

既存分析スクリプトで `2026-03-12` から `2026-03-22` を再集計すると、次の値になった。

- `Total=6034`
- `Filled=1602`
- `Fill rate=26.5%`
- `buy avg_pnl30=-0.44bps`
- `sell avg_pnl30=-0.18bps`
- `git_sha_unique=60`

一方、584# の冒頭数値は

- Fill Rate 25.2%
- Buy -0.28bps
- Sell +0.21bps
- preflight_insufficient 34.7%

となっており、少なくとも分母かフィルタ条件が異なる。

**判定**:
- 584# の危機感は妥当
- ただし baseline 数字は **再現条件を明記しないと意思決定に使いづらい**

**推奨**:
- `date_from/date_to`, `run_id`, `git_sha`, `filled only / total`, `attempted only` を併記する
- 以後の比較は same-run / same-SHA を原則にする

### 3.5 `experimental_additive_pipeline` と `execution_additive_enabled` の二枚看板は整理した方がよい

現行コード上、実際の分岐は `experimental_additive_pipeline` が持っている。
一方 `execution_additive_enabled` は config/hot-reload/telemetry 側に残っているが、責務が見えにくい。

この状態だと、584# のような計画書で

- どのフラグが実際のロジック分岐か
- どのフラグが観測上のラベルか

が混線しやすい。

**推奨**:
- 実行切替フラグを `experimental_additive_pipeline` に一本化
- `execution_additive_enabled` は telemetry 用 legacy 名なら、その旨を明記するか廃止候補に回す

### 3.6 `additive_base_bps` は現時点では未使用ノブに見える

`additive_base_bps` は YAML と parser と dataclass には存在するが、実処理での使用箇所は確認できなかった。

**推奨**:
- 使わないなら削る
- 将来用なら、584# の時点では「未使用」と明記する

---

## 4. 支持できる点

以下は今回の流れの中で、かなり支持しやすい。

1. **578# の監査スタンス**
   - 実装ミスと構造問題を分けて見ているのは良い
2. **579# の inventory skew tanh 平滑化評価**
   - `maker_price.py` に実装も確認でき、方向として自然
3. **583# の責務分割**
   - `offset_pipeline.py` / `multiplicative_pipeline.py` 分割
   - `fill_cycle_executor.py` の phase helper 化
   は保守性改善として素直に良い
4. **584# P1 を上位に置いたこと自体**
   - additive / clamp / telemetry の整理を先にやるのは合理的
5. **584# P5 の Entry Gate 再評価**
   - 555# 系資産を捨てずに活かす発想は 000# の「既存成果最大活用」と整合する

---

## 5. 今やる順番

現時点で一番詰まりにくい順番は次の通り。

1. **584# を ph5.x remediation として位置付け直す**  
   000# を直すか、584# の名前を直すかのどちらかを先に決める。

2. **P1 の前に telemetry parity を確立する**  
   same-SHA / same-run で
   - `spread_capture_bps`
   - `adverse_selection_cost_bps`
   - `tox_buffer`
   - `liq_buffer`
   が本当に JSONL に出ることを確認する。

3. **P1 Additive A/B を restart 前提で実施する**  
   hot-reload 前提ではなく、反映確認済み run として比較する。

4. **P2 Smart Preflight は観測追加を先に入れる**  
   いきなり skip せず、まず「skip したら何を失うか」を見える化する。

5. **P6 は fresh/stale/error 安定化として並行処理する**  
   execution 是正の主軸には置かない。

6. **P3 / P4 / P5 は P1 の観測成立後に再優先順位付けする**  
   現時点では additive 実効性が未確定なので、buy/sell 調整や eDRC 有効化を先に重ねると attribution がさらに濁る。

---

## 6. 著者別に返したい修正ポイント

### 578# / 579# 著者向け
- 「コード修正済み」と「live で確認済み」を分けて書く
- `inventory_manager` 新設より、既存の `maker_price.py` / `orchestrator_balance.py` の責務を尊重した分割案へ寄せる

### 580# / 581# / 582# / 583# 著者向け
- `True Additive` はまず `Execution-stage additive` と書く
- `additive_base_bps` や `execution_additive_enabled` の責務を整理する
- live JSONL での観測成立を確認できるまで「完了」は少し抑える

### 584# 著者向け
- ph6 の位相を 000# と揃える
- P6 を「neutral fallback 解消」から「fresh/stale/error 改善」へ再定義する
- P1/P4/P5 は hot-reload ではなく restart / run_id 分離前提に修正する
- V4 の成功基準を profit-first 指標に差し替える
- P2 pseudo-code は存在しない API 依存を消して、観測追加込みで設計し直す

---

## 7. 最終判定

- **578#**: 支持。監査観点は妥当。
- **579#**: おおむね支持。ただし refactor 先の責務配置は補正が必要。
- **580# / 581# / 582#**: 方向性は支持。ただし「完了」より「コード反映済み・live要確認」が正確。
- **583#**: 支持。保守性改善として価値が高い。
- **584#**: 本文の問題意識は良い。ただし **位相・優先順位・可観測性前提** を直してから着手した方が成功率が高い。

総じて、今回の流れは迷走というより、**実装速度が可観測性と運用規律を追い越した状態** と見るのが正確である。
したがって次の一手は新機構の追加ではなく、**000# と runtime と分析基盤の三者を再整列すること** である。
