# 515# PHG: remaining deferred actions screening plan

## 目的

514# で deferred docs 全体の棚卸しを行った。
本書はその次段として、

- まだ docs 上で future 扱いだが、今やれそうなもの
- 今は触らないほうが安全なもの

を切り分け、以後の実装/文書更新の順序を固定するための計画書である。

## 対象文書

- `121_ph2_plan_model_replacement.md`
- `158_phg_rpt_backlog_audit_and_phase_d_priorities.md`
- `index.md`

## スクリーニング結果

### A. 今やれるもの

#### 1. `121# D1 lib -> ztb`

判定: docs 更新は今やる価値が高い

理由:

- 主要 canonical 化は既に session037 で前倒し済み
- ただし residual な `maker_price` / `skip_gate_evaluator` / `order_monitor` / `ab_judgment`
  は残るため、done 扱いではなく「残る本命」表現が必要

対応方針:

- stale な `v461` 固定表現を修正
- 残課題は限定的に明記

#### 2. `121# D9 VG イベント JSONL`

判定: docs 更新は今やる価値が高い

理由:

- 372# で既に構造化ログ化済み
- deferred のまま残すと誤読を招く

対応方針:

- done として追記
- ただし downstream analysis の継続余地は別物として扱う

#### 3. `158# P2-5 skip_gate.py モジュール配置`

判定: docs 更新は今やる価値が高い

理由:

- canonical import 収束はかなり進んでいる
- ただし façade/shim を残す意味はまだある

対応方針:

- 「未着手」から「大幅前進済み」へ修正
- 将来の完全収束とは分けて記述

#### 4. `158# P3-1 SkipGate 単体テスト拡充`

判定: docs 更新は今やる価値が高い

理由:

- session037 で runtime/result/fill-record 境界までかなり補強済み
- もはや future 大項目ではなく継続保守レベル

対応方針:

- future 一覧の表現を弱める
- 「残りは追加補完レベル」と書く

### B. 今は future 維持でよいもの

#### 1. `121# D2 utils 70+`
#### 2. `121# D5 UnifiedTrainer`
#### 3. `121# D12 WebSocket API 活用`
#### 4. `121# D13 event-driven cycle`
#### 5. `158# P3-2 utils 70+`
#### 6. `158# P3-4 UnifiedTrainer`
#### 7. `158# P3-6 asyncio.to_thread`

理由:

- いずれも session037 の前倒し範囲を越える
- ここを中途半端に触ると docs より code backlog のほうが先に壊れる

## 更新ルール

1. 「done」と「かなり前進」を混同しない
2. shim / façade が残るものは、未完と明示する
3. docs の目的が historical record の場合、当時の判断は削除しない
4. `index.md` は要約だけに留め、詳細は個別文書へ寄せる

## 実行順

1. `121#` 更新
2. `158#` 更新
3. `index.md` low priority リスト追随
4. その後に second wave として
   - `RewardCalculator` 分割
   - `UnifiedTrainer`
   - `WebSocket / event-driven`
   の docs 整合を再確認

## 判定

「今やれる deferred 項目の docs 更新」は今回の範囲でかなり消化できる。

一方で、`UnifiedTrainer` / `utils` / `event-driven` は、
現時点では無理に「前倒し済み」と見せない方が安全である。
