# Codex Prompt: Protocol 688 NFQ フィルタ修正 (700# Task 1)

## Goal
`protocol_688.py` の NFQ セクションが全キャンセルを返すセマンティックバグを修正し、NFQ (no_feasible_quote) キャンセルのみを正確にフィルタする。

## Context
- **697# 指摘**: `_cancel_payload(records)` は `not filled` で全キャンセルを返すが、NFQ セクションでは `cancel_reason == "no_feasible_quote"` のみが対象であるべき。
- **影響**: 過去の protocol 688 ベースの NFQ 分析がすべて汚染されている。
- **場所**: `scripts/v460/analysis/protocols/protocol_688.py`, L349:
  ```python
  "nfq": _cancel_payload(records),
  ```

## 現行コード

### `_cancel_payload` (L152-159)
```python
def _cancel_payload(records: list[Record]) -> dict[str, object]:
    cancels = [record for record in records if not _record_bool(record, "filled")]
    counts = collections.Counter(_record_str(record, "cancel_reason", "unknown") for record in cancels)
    return {
        "total": len(cancels),
        "reasons": dict(counts),
    }
```
→ **問題**: 全キャンセルを返す。NFQ 固有ではない。

## Implementation

### 1. 新関数 `_nfq_payload` を追加

```python
def _nfq_payload(records: list[Record]) -> dict[str, object]:
    """NFQ (no_feasible_quote) キャンセルのみを抽出して集計する."""
    nfq_records = [
        record for record in records
        if not _record_bool(record, "filled")
        and _record_str(record, "cancel_reason", "") == "no_feasible_quote"
    ]
    # NFQ 発生の regime 分布
    regime_counts = collections.Counter(
        _record_str(record, "regime", "unknown") for record in nfq_records
    )
    # NFQ 発生の side 分布
    side_counts = collections.Counter(
        _record_str(record, "side", "unknown") for record in nfq_records
    )
    total_cancels = sum(1 for r in records if not _record_bool(r, "filled"))
    return {
        "nfq_total": len(nfq_records),
        "cancel_total": total_cancels,
        "nfq_ratio": len(nfq_records) / total_cancels if total_cancels > 0 else 0.0,
        "regime": dict(regime_counts),
        "side": dict(side_counts),
    }
```

### 2. L349 の呼び出しを差し替え

```python
# Before:
"nfq": _cancel_payload(records),

# After:
"nfq": _nfq_payload(records),
"cancels": _cancel_payload(records),  # 全キャンセルは別キーで保持
```

**注意**: `_cancel_payload` は他セクション等で汎用キャンセル集計に使われる可能性があるため削除しない。

### Test file: `tests/unit/v460/test_700_protocol_688_nfq_fix.py`

1. `test_nfq_payload_filters_only_nfq` — NFQ + 他キャンセルを混在させ、NFQ のみ抽出を確認
2. `test_nfq_payload_empty_records` — 空リストで {"nfq_total": 0, "cancel_total": 0, ...} を確認
3. `test_nfq_payload_no_nfq_cancels` — キャンセルはあるが NFQ はゼロのケース
4. `test_nfq_payload_regime_distribution` — regime 分布が正しく集計されるか
5. `test_cancel_payload_unchanged` — 既存 `_cancel_payload` の動作が変わらないことを確認
6. `test_protocol_output_has_both_keys` — 出力に "nfq" と "cancels" 両方存在

## Constraints
- `_cancel_payload` の関数シグネチャ・戻り値は変更しない
- `_nfq_payload` の戻り値は JSON serializable
- `_record_str` / `_record_bool` ヘルパーを使用（新規ヘルパー不要）
- Run: `python -m pytest tests/unit/v460/test_700_protocol_688_nfq_fix.py -x --tb=short -q`
