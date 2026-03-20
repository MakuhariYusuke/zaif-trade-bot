# 514# PHG: deferred docs refresh / carry-forward audit plan

## 目的

`docs/v460` には、当時は妥当だったが現在は実装進捗とズレた
`v461` 以降送り / 後日 / 棚上げの記述が残っている。

本書は、それらを

1. 今すぐ更新すべきもの
2. 一部だけ更新すべきもの
3. まだ将来課題として維持すべきもの

に分け、以後の docs 保守を迷いなく進めるための監査・計画書である。

## 背景

session037 で次がかなり前倒しされた。

- `lib -> ztb` の主要移行
- `skip_gate` の canonical import 収束
- `maker_price` / `skip_gate_evaluator` の split-first
- observability / diagnostics の拡充

この結果、古い文書の一部は「当時の判断」と「現在の状態」が混ざり始めている。

## 監査結果

### A. すぐ更新すべき文書

#### 1. `113_ph2_impl_resilience_r1_split.md`

理由:

- `R3 SkipGate テスト不足`
- `R5 lib -> ztb 移動`

が現在はそのままではない。

方針:

- 当時の表は残す
- 2026-03-21 補遺で「その後の前倒し」を追記

#### 2. `118_phg_rpt_backlog_deep_analysis.md`

理由:

- `R5 / E3 lib -> ztb`
- `G1-4 / E11 skip_gate.py`

が未着手前提のまま残っている。

方針:

- 「全部古い」とは扱わない
- `lib -> ztb` / `skip_gate` は更新
- `UnifiedTrainer` / event-driven cycle は将来維持

#### 3. `168_phg_rpt_comprehensive_improvement_hodl_vs_trading.md`

理由:

- `P2-5 skip_gate.py モジュール配置` が `v461` 扱いのまま

方針:

- 収益直結ではない点は維持
- ただし「未着手の v461 課題」ではなく「実装前進済み」に改める

#### 4. `420_ph2_impl_observability_deferred_items.md`

理由:

- 420# 以後に observability がかなり増えている
- 先送り事項の中身が実質 2 件へ絞れている

方針:

- 先送り節そのものは残す
- 補遺で「その後どこまで観測基盤が前進したか」を追記

### B. 次点で更新候補

#### 1. `121_ph2_plan_model_replacement.md`

理由:

- `D2 utils 70+`
- `D5 UnifiedTrainer`
- `D13 event-driven cycle`

などは依然として future 寄りだが、
`lib -> ztb` や `skip_gate` に絡む記述は一部 stale の可能性がある。

#### 2. `158_phg_rpt_backlog_audit_and_phase_d_priorities.md`

理由:

- P3/v461+ の棚卸し母表として使われやすい
- ただし一括更新より、今回更新した 113/118/168 の反映を見た後で
  読み直すほうが安全

### C. まだ将来課題として維持すべき文書・項目

#### 維持でよいもの

- `utils/` 70+ ファイル分割
- `UnifiedTrainer` God Object
- event-driven cycle
- online learning
- `asyncio.to_thread` 残件

理由:

- いずれも session037 の前倒し範囲を超える
- 今は「未着手」より「未着手のままで妥当」

## 今回の実更新

今回実際に更新した対象:

- `106#`
- `108#`
- `113#`
- `118#`
- `168#`
- `420#`
- `502#`
- `505#`

## 更新ルール

今後、deferred docs を直すときは次のルールで揃える。

1. 当時の判断は消さない
2. 現状との差分は `補遺` か `現時点の要約` で追記する
3. `未着手` と `将来維持` を混同しない
4. `lib -> ztb` / `skip_gate` / observability は session037 進捗へ追随させる
5. `UnifiedTrainer` / event-driven / online learning は軽々しく done 扱いしない

## 次の更新優先順位

1. `121_ph2_plan_model_replacement.md`
2. `158_phg_rpt_backlog_audit_and_phase_d_priorities.md`
3. `index.md` の low priority / v461+ リストと、今回更新済み docs の整合確認

## 判定

deferred docs のうち、今すぐ触る価値が高かったものは今回ほぼ着手できた。

残る課題は、

- stale になった future 表現の二次棚卸し
- 真に将来課題のものを誤って「消化済み」に見せないこと

の 2 点である。
