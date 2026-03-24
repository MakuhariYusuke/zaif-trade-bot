# 597# 死コード注釈: 190# A/B + one_sided_balance

## 背景

596# で `skip_gate_primary_max_consecutive_skip` を evaluator-level の安全弁として導入し、
190# A/B の「旧 ev_weighted ハードゲート側の安全弁」に依存しなくても連続 skip を止められる状態になった。

また、`one_sided_balance` は 522# の `balance_forced` 廃止で trigger 自体が消えており、
実運用では `True` にならない死コード状態に入っている。

このため、いま残っている該当コードは挙動変更ではなく、
**死コードとして明示して保守判断を誤らないようにする** ことが主目的になる。

## 今回の整理

### 1. 190# A/B の旧ハードゲート分岐に dead-code 注釈を追加

- 対象: `scripts/v460/lib/skip_gate_ev_weighted.py`
- `skip_gate_ev_as_offset_enabled=True` の本番設定では当該分岐に到達しないことを明記
- コメント:
  - `DEAD CODE (596#): offset mode では到達不可、primary safety valve で代替`

### 2. `one_sided_balance` 系の死コード注釈

- 対象:
  - `scripts/v460/lib/orchestrator_pre_cycle.py`
  - `scripts/v460/lib/fill_config.py`
- コメント:
  - `DEAD CODE (596#): 522# で trigger 消失、常時 False`

### 3. config コメントの現状追随

- `skip_gate_ev_max_consecutive_skip`
- `skip_gate_ev_one_sided_threshold_shift`

に対して、
`596# primary safety valve で代替済み`
を追記し、旧安全弁の立ち位置を現状に合わせた。

## 判断

- 今回は削除ではなく注釈に留めた
- 理由:
  1. 190# / 193# / 596# の系譜を読める状態で残した方がレビューしやすい
  2. 既存テストが旧フィールド存在自体を見ているため、まずは意味付けの更新が安全
  3. 本当の削除は、YAML / hot-reload / source-contract の再棚卸しを伴う別タスクが妥当

## 確認

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_336_yaml_code_drift_prevention.py \
  tests/unit/v460/test_190_ev_weighted_safety.py \
  tests/unit/v460/test_596_primary_consecutive_skip_safety.py \
  tests/unit/v460/test_593_ev_toxic_skip_and_cap_hit_veto.py \
  tests/unit/v460/test_193_ev_offset.py \
  -x --tb=short --no-cov
```

## 次の自然な一手

1. 旧 190# A/B フィールドを「存在は維持するが dead-code」扱いで関連 docs にも揃える
2. `one_sided_balance` の本削除可否を、522# 系の履歴と hot-reload 影響込みで判断する
