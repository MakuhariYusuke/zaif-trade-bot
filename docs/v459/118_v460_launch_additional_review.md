# 118# v460始動 追加レビュー（設計・命名以外の観点）

| 項目 | 内容 |
|------|------|
| 対象 | `117_v460_doc00_design_and_naming_reform.md` |
| 前提 | v459最終結論は `116#`（Phase E No-Go, v460移行推奨） |
| 目的 | v460開始時に、命名/文書設計以外で見落としやすいリスクを先に潰す |

---

## §0 結論（先に要点）

1. `117#` の命名改革と 00# 軽量化方針は妥当で、**そのまま採用可能**。  
2. ただし v460 の成否は命名より、**実験統治・執行実現性・データ準備**で決まる。  
3. 追加で必須なのは次の3点:  
   - Gate 1 を「特徴量情報量」だけでなく「執行可能性（maker fill）」まで拡張  
   - K2 の結論を過信せず、ラベル設計（horizon/目的関数）を再定義  
   - run manifest を実装レベルで強制し、再現性を文書ではなく成果物で担保  

---

## §1 117# の良い点（維持推奨）

1. `NNN_` 3桁化と `ph/type` 明示は、v459で発生した索引崩壊をほぼ解消できる。  
2. `rev/resp` の規格化はレビュー連鎖の追跡性を大幅に改善する。  
3. 00# を「判定基準原典」に寄せる方針は正しい。  
4. AIプロンプトと日次ログを `docs/` から分離する判断は妥当。  

---

## §2 追加で補うべき盲点（重要度順）

| Severity | 観点 | 盲点 | リスク | 補正案 |
|---|---|---|---|---|
| **CRITICAL** | 執行実現性 | `maker-only` 前提はあるが、fill rate/queue/latency の Gate が未定義 | 机上では勝てるが実運用で約定せず崩壊 | Gate 1.5 を追加し、`fill_rate_90p`, `cancel_ratio`, `queue_wait` を必須指標化 |
| **CRITICAL** | 情報量判定 | K2は「次足符号×8特徴」での判定。これのみで「特徴量に情報なし」と断定すると早計 | v460で有効な horizon/task を捨てる可能性 | Gate 1 を multi-target 化（h1/h5/h15, direction + magnitude + volatility） |
| **HIGH** | コスト判定 | v459で `eval_cost` 計測不整合が一部存在 | コスト関連の結論の信頼区間が広い | v460 Day1で fee model integration test を最優先実装 |
| **HIGH** | 実験統治 | 命名規則はあるが run manifest のスキーマが未定義 | 「文書上の再現」しか残らず再検証不能 | `run_id/config_hash/data_hash/artifacts/status` をJSONLで強制 |
| **HIGH** | Gate互換性 | 0#/v459 と v460 の Gate 番号の意味が変わる可能性 | 比較不能、意思決定の混乱 | `G1-info`, `G2-train`, `G3-pnl`, `G4-live` のように名前付きで固定 |
| **MEDIUM** | データ供給 | マイクロストラクチャ特徴量導入方針はあるが取得・保存仕様が未定義 | 実装が始まらず設計だけ進む | 取得対象（板/約定）・粒度・欠損処理・保持期間を 001 で先に固定 |
| **MEDIUM** | 言語方針 | `英語のみ` を厳格化すると議論速度が落ちる恐れ | 文書運用が形骸化 | 「ファイル名は英語、本文は日本語可」に緩和 |
| **LOW** | 規則一貫性 | `PhX` と `ph0` の表記が117内で混在 | 小さな運用摩擦 | すべて小文字 `ph0/ph1/phg` に統一 |

---

## §3 v460開始時の推奨 Gate 再設計

### §3.1 Gate構造（実運用寄り）

| Gate | 目的 | 失敗時の扱い |
|------|------|-------------|
| `G0-data` | データ品質と再現性基盤（hash/manifest） | 即停止、学習禁止 |
| `G1-info` | 特徴量情報量（非RL上限, multi-target） | FAILなら特徴量再設計へ戻る |
| `G1.5-exec` | maker執行可能性（fill/latency） | FAILなら戦略クラス変更 |
| `G2-train` | 学習安定性（seed分散、再現） | FAILなら学習器/報酬見直し |
| `G3-pnl` | コスト込み収益性（PF/ROI/Sharpe） | FAILなら設計見直し |
| `G4-live` | Paper trading運用検証 | FAILなら本番禁止 |

### §3.2 最低通過条件（初期案）

1. `G1-info`: OOSで `IC > 0.02` を少なくとも1 horizonで再現。  
2. `G1.5-exec`: maker想定で `fill_rate_90p >= 90%`。  
3. `G2-train`: 4 seed中3以上で `gross > 0`。  
4. `G3-pnl`: `avg_gross_per_trade > avg_fee_per_trade` かつ `PF > 1.05`。  

---

## §4 命名規則への補強提案

### §4.1 推奨ファイル名

`NNN_phX_type_subject[_refNNN].md`

- `NNN`: 3桁連番  
- `phX`: `ph0/ph1/.../phg`（小文字固定）  
- `type`: `plan/rev/resp/rpt/fix/ext/meta`  
- `refNNN`: レビュー対象番号（必要時のみ）

### §4.2 補助規則

1. 同名禁止（subject は一意）。  
2. `rev` は必ず `refNNN` を持つ。  
3. 章追加で方針が変わったら、文書末尾に `Change Log` を追記。  

---

## §5 v460最初の7文書（実行順）

1. `000_ph0_plan_project_proposal.md`  
2. `001_ph0_plan_data_contract.md`  
3. `002_ph0_plan_gate_spec.md`  
4. `003_ph0_plan_experiment_manifest.md`  
5. `004_ph0_rpt_g0_data_validation.md`  
6. `005_ph0_rpt_g1_feature_info_test.md`  
7. `006_ph0_rev_000_ref000.md`  

注: 117# の `001_ph0_plan_architecture.md` は残しつつ、`data_contract` と `manifest` を先に確定させると実装停滞を防げる。

---

## §6 48時間の現実的アクション

1. `docs/v460/` と `experiments/v460/` を作成し、命名ルールを README 化。  
2. `G0-data` 用に `manifest.jsonl` スキーマを固定。  
3. K2再試験（multi-target版）を 1 本だけ実施。  
4. maker執行の簡易シミュレーション指標（fill_rate）を `G1.5` として定義。  

---

## §7 最終コメント

v460の失敗パターンは「設計は綺麗だが、実験統治と執行現実で崩れる」形。  
`117#` を基礎に、`118#` の補正（特に `G1.5-exec` と manifest 強制）を入れれば、v459で起きたループ再発をかなり防げる。  
