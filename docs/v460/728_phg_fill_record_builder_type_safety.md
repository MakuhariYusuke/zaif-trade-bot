# 728# FillRecord builder type-safety cleanup

## 目的

fill関連分割の継続として、`FillRecord` 再構築経路の重複と局所 `type: ignore` を減らす。
挙動は変えず、既存の payload sanitization と typed builder へ寄せる。

## 実装

- `ztb.metrics.fill_record_builders.build_typed_fill_record()` で constructor を typed callable として扱い、`type: ignore[call-arg]` を削除。
- `FillRecord.from_dict()` を `build_typed_fill_record()` 経由に変更し、sanitize + constructor 呼び出しの重複を排除。
- `fill_quality.py` から不要になった `fill_record_payloads` import を削除。

## 横展開メモ

- 今回は `FillRecord` の dataclass constructor 型と動的 payload filtering の境界だけを整理した。
- 次の候補は、テスト側に散在する `FillRecord(...)` の大型fixtureを `build_fill_record()` / shared factory に寄せること。
- `Callable[..., FillRecordT]` cast は dataclass constructor の動的 keyword payload を型検査に伝えるための境界で、`Any` と inline ignore は増やしていない。

## 検証結果

- `python3 -m py_compile ztb/metrics/fill_record_builders.py ztb/metrics/fill_quality.py` PASS
- `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_fill_quality.py::TestFillRecord tests/unit/v460/test_695_fill_record_enrichment.py tests/unit/v460/test_687_state_separation.py -x --tb=short --no-cov` PASS: `30 passed in 2.03s`
- `rg "type: ignore\\[(arg-type|call-arg)\\]" ztb/metrics/fill_quality.py ztb/metrics/fill_record_builders.py` found no remaining target ignores.
