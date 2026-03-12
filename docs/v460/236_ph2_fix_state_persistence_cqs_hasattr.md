# 236# State Persistence / CQS 分離 / hasattr 排除 / per-side 化

## 概要

232# (Codex) + 233# (Gemini) 外部レビューで指摘された残課題と、
235# セルフレビューで洗い出した C-7 / C-9 を合わせて修正。
**コード品質・堅牢性** に焦点を当てた改善バッチ。

## 修正内容

### C-7 [HIGH]: 234# エスカレーションカウンタの状態永続化漏れ

**問題**: 234# で追加した 3 カウンタ (`_degraded_liquidation_duty_counter`,  
`_one_sided_cooldown_remaining`, `_one_sided_freeze_remaining`) が  
`FillTestState` に含まれず、クラッシュ再起動時にエスカレーション状態が消失。
例: freeze 期間中にプロセス再起動 → freeze 解除＝安全機構の空振り。

**修正**:
- `FillTestState` に 4 フィールド追加 (3 カウンタ + `consecutive_no_feasible`)
- `_build_state_snapshot()` で保存
- `_restore_common_state()` で復元 (> 0 の場合のみ、ログ出力付き)

### C-9 [HIGH]: `consecutive_no_feasible` per-side 化

**問題**: グローバル `int` カウンタだったため、buy 側の narrow 連続と sell 側の
guard_reject が合算され、片側だけでは閾値未達なのに `NO_FEASIBLE_QUOTE` 誤発動。

**修正**:
- `_consecutive_no_feasible: int → dict[str, int] | None` に変更
- 全使用箇所で `side` をキーとして get/set
- リセットも per-side

### 232# §1.8 [HIGH]: FFD getter 副作用の CQS 分離

**問題**: `get_boost_multiplier()` が TTL decay (状態変更) を内包 — getter が
副作用を持つ CQS 違反。呼び出し順序に依存するバグの温床。

**修正**:
- `maybe_expire_boost(side)`: TTL decay ロジック (副作用あり)
- `get_boost_multiplier(side)`: 純粋 getter (状態読み取りのみ)
- `maker_price.py` で `maybe_expire_boost()` → `get_boost_multiplier()` の順で呼び出し

### C-4 [MEDIUM]: hasattr パターンの排除 (4 箇所)

`hasattr` はアトリビュートの存在を動的チェックする anti-pattern。
クラスレベルデフォルト値で属性の存在を保証し、静的解析・型安全性を向上。

| 場所 | 旧 | 修正 |
|------|-----|------|
| `fill_cycle_executor.py` (2箇所) | `hasattr(self, "_current_regime_value")` | 直接呼び出し (Mixin で定義保証) |
| `fill_record_helpers.py` (2箇所) | `hasattr(self, "_trending_sell_skip_count")` 他 | orchestrator にクラスレベルデフォルト宣言 |
| `skip_gate_evaluator.py` | `hasattr(self, "_last_reload_check")` | クラスレベル `_last_reload_check: float | None = None` + None ガード |

### [LOW]: Dead import 排除

- `config_hot_reload.py`: 未使用 `import sys` 削除
- `fill_loop_orchestrator.py`: 未使用 `Optional` import 削除

## 変更ファイル

| ファイル | 変更内容 |
|---------|----------|
| `resilience.py` | `FillTestState` に 4 フィールド追加 |
| `fill_loop_orchestrator.py` | snapshot 保存・復元追加、クラスレベルデフォルト、Optional 削除 |
| `fill_cycle_executor.py` | `_consecutive_no_feasible` per-side 化、hasattr 排除 |
| `fast_fill_defense.py` | CQS 分離: `get_boost_multiplier` / `maybe_expire_boost` |
| `maker_price.py` | `maybe_expire_boost()` 呼び出し追加 |
| `fill_record_helpers.py` | hasattr 排除 |
| `skip_gate_evaluator.py` | `_last_reload_check` クラスレベルデフォルト |
| `config_hot_reload.py` | dead import 削除 |
| `test_229_cleanup_counter_rename.py` | CQS 対応更新 |
| `test_230_ffd_deadzone_streak_guards.py` | CQS 対応更新 + hasattr テスト反転 |
| `test_236_state_persistence_cqs.py` | **新規** 28 テスト |

## テスト結果

```
3223 passed, 0 failed
```

## 残課題

| 優先度 | 課題 | 備考 |
|--------|------|------|
| P0 | `status_unknown` → phantom position quarantine | order_monitor: 不明注文の隔離・残高照合 |
| P1 | participation budget (binary→continuous) | Kill Gate を毒性バジェットに設計変更 |
| P1 | feasible quote proactive calculation | maker_price 前の制約交点事前計算 |
| P1 | Liveness constraint relaxation | 長時間無取引を正常状態として許容 |
| P2 | Guard reason classification | market vs system の分類リファクタ |
| P2 | Macro/mid horizon separation | 2 階層アーキテクチャ設計 |
