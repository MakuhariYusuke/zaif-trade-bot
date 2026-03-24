# 598# 次のデススパイラル候補 調査メモ

## 位置付け

この系統の cleanup / 調査は、以後この `598#` に集約して更新する。

- 597# は 190# A/B + `one_sided_balance` の個別 cleanup 記録
- 598# は、その次に再燃しうる候補の調査・判断・追記の正本

## 背景

596# で primary safety valve を導入し、597# で 190# A/B と `one_sided_balance`
に dead-code 注釈を付けた。

ここで次に危ないのは、

- runtime では実質使われない
- しかし config / YAML / parser / テストだけ残る

という「保守判断を誤らせる死コード予備軍」である。

## 調査結果

### A. `inventory_escape_*` / `recovery_skew_*`

対象:

- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fill_config_parser.py`
- `configs/v460/fill_test.yaml`
- `tests/unit/v460/test_285_split_brain_guard.py`
- `tests/unit/v460/test_346_fill_config_validation.py`

観察:

1. runtime 側での参照は実質消えている
2. parser / YAML / validation test は後方互換のため残っている
3. `fill_config.py` 自体にも `522# 撤廃` と明記されている

判断:

- **dead-code 予備軍ではなく、ほぼ dead-code 本体**
- ただし parser と validation test が existence / backward-compat を見ているため、
  いきなり削除するより
  **注釈で「読み取り専用の残置」であることを固定する** 方が安全

今回の対応:

- `fill_config.py` に `DEAD CODE (598#)` 注釈を追加
- `fill_config_parser.py` に後方互換読み取り専用の注釈を追加
- `test_285_split_brain_guard.py` / `test_346_fill_config_validation.py` の説明も
  「runtime で効く前提」から「legacy read-only field の存在確認」へ補正

### B. `skip_gate_ev_*` 旧 hard-gate 系

対象:

- `skip_gate_ev_max_consecutive_skip`
- `skip_gate_ev_one_sided_threshold_shift`
- `skip_gate_ev_weighted.py` の旧 hard-gate 分岐

観察:

1. 本番 YAML では `ev_as_offset_enabled: true`
2. 旧 hard-gate 分岐は本番主経路では通らない
3. ただし hot-reload / YAML drift / legacy existence test が残る

判断:

- runtime 的には dead-code に近い
- ただし **互換性レイヤとしての存在はまだ意図的**
- 597# の dead-code 注釈で、現時点では十分

### C. `execution_additive_enabled` と `experimental_additive_pipeline`

対象:

- `fill_config.py`
- `fill_config_parser.py`
- `config_hot_reload.py`
- `fill_record_builder.py`
- `analyze_fill_logs.py`

観察:

1. logic 分岐は `experimental_additive_pipeline`
2. outward telemetry は `execution_additive_enabled`
3. parser 側に mismatch warning がある
4. hot-reload には両方が残っている

判断:

- これは dead-code ではなく **二重意味の温床**
- ただし現状は
  - ロジック
  - telemetry
  の役割分離が入っているので、直ちに危険ではない
- 次に危険化するなら、docs/test がこの役割分離を忘れた時

## 型安全の確認

次の targeted mypy を実行し、今回の調査対象の config 線に新しい型崩れがないことを確認した。

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/lib/fill_config.py \
  scripts/v460/lib/fill_config_parser.py \
  scripts/v460/lib/config_hot_reload.py
```

結果:

```text
Success: no issues found in 3 source files
```

## まとめ

優先度順では次の通り。

1. `inventory_escape_*` / `recovery_skew_*`
   - dead-code 注釈を付けた上で、将来削除候補として整理
2. `skip_gate_ev_*`
   - 597# の注釈で現状十分、次は docs/test の意味付け整理
3. additive dual-line
   - dead-code ではなく contract drift 防止が主眼

## 追加確認

### additive dual-line の守り

次の 2 点をテストで固定した。

1. `execution_additive_enabled` は telemetry 契約として hot-reload 対象に残す
2. `inventory_escape_*` / `recovery_skew_*` は legacy read-only 扱いのため hot-reload 対象にしない

また、parser は

- `execution_additive_enabled`
- `experimental_additive_pipeline`

が不一致のとき warning を出すことで、
「logic は additive、telemetry は multiplicative」などの読み違いを検知できる。

### analysis / observability 側の low-risk 型 sweep

`hindsight_filter.py` は TypedDict の整理範囲が大きく、別タスク向きだった。
一方で次の 2 本は low-risk に targeted mypy を進められると判断した。

- `compare_regime_ab.py`
  - detector subclass の内部属性型を宣言
- `side_regime_dashboard.py`
  - YAML judgment mapping と per-regime criteria の cast を明示

## 次の自然な一手

1. `inventory_escape_*` / `recovery_skew_*` を「読み取り専用 legacy field」として docs 側にも明記
2. `skip_gate_ev_*` の existence/hot-reload test を、legacy compatibility 契約として整理
3. additive dual-line については telemetry と logic の責務分離を保つ focused test を追加
