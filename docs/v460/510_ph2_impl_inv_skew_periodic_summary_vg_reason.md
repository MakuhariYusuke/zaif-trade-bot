# 510# inv_skew_factor fill_record 追加 / 周期的サマリ統計 / VG boost 理由粒度向上

## 概要

fill_record と周期ログの可観測性を3方面から強化。
在庫偏重状態、VG 発動理由、regime 別 skip rate を定量追跡可能にする。

---

## 変更内容

### 1. inv_skew_factor の fill_record 追加

**問題:** `_last_inv_skew_factor` は `maker_price.py` 内部変数のみで、
fill_record にも外部プロパティにも公開されていなかった。
在庫偏重による offset 調整がヒンドサイト分析で不可視。

**修正:**

| 対象 | 変更 |
|------|------|
| `maker_price.py` | `last_inv_skew_factor` プロパティ追加 |
| `fill_quality.py` | `inv_skew_factor: float | None` フィールド追加 |
| `fill_record_builder.py` | `inv_skew_factor` フィールド populate |

### 2. VG boost 理由の粒度向上 (velocity vs VPIN 分離)

**問題:** `vg_reason` は `maker_risk_guards.py` 内のローカル変数で、
ログ文字列としてのみ使用。fill_record では `vg_triggered=True` と
`vg_boost_factor` しか記録されず、velocity/VPIN どちらが主因か判別不能。

**修正:**

| 対象 | 変更 |
|------|------|
| `maker_price.py` | `_last_vg_reason` スロット + `last_vg_reason` プロパティ追加 |
| `maker_risk_guards.py` | `_last_vg_reason` に `"velocity"` / `"vpin"` / `"velocity+vpin"` を構造化保存 |
| `fill_quality.py` | `vg_reason: str | None` フィールド追加 |
| `fill_record_builder.py` | `vg_reason` フィールド populate |

**分類ロジック:**
```python
if velocity_boost > 1.0 and vpin_boost > 1.0:
    reason = "velocity+vpin"
elif velocity_boost > 1.0:
    reason = "velocity"
else:
    reason = "vpin"
```

### 3. 周期的サマリ統計

**問題:** progress log にはサイクル数・fill rate・PnL のみ。
regime 別の skip rate、VG 発動率、inv_skew 稼働率が不可視。

**修正:**

`RunSessionState` に追加カウンタ:
```
regime_cycle_counts: dict[str, int]   # regime→cycle数
skip_by_regime: dict[str, int]        # regime→skip数
vg_fire_count: int                    # VG発動回数
vg_reason_counts: dict[str, int]      # reason→count
inv_skew_active_count: int            # inv_skew≠0のサイクル数
```

`orchestrator_post_cycle.py` に追加ログ:
```
[510# regime] trending_up=120(skip 8%), ranging=80(skip 25%), unknown=10(skip 40%)
[510# VG] fires=45 (18.8%), reasons: vpin=30, velocity=10, velocity+vpin=5
[510# inv_skew] active=90/200 (45.0%)
```

---

## テスト

`tests/unit/v460/test_506_sell_improvements.py` に追加:

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestInvSkewFactorFillRecord` | 3 | FillRecord フィールド存在、to_dict、MakerPrice プロパティ |
| `TestVGReasonGranularity` | 4 | FillRecord フィールド、有効値、MakerPrice プロパティ、分類ロジック |
| `TestPeriodicSummaryCounters` | 2 | RunSessionState フィールド存在、カウンタ increment |

合計: 34 tests pass（既存 25 + 新規 9）

---

## 影響範囲

- `scripts/v460/lib/maker_price.py`: __slots__ + init + property 追加 (既存コードパス不変)
- `scripts/v460/lib/maker_risk_guards.py`: vg_reason 永続化 (既存ロジック不変)
- `scripts/v460/lib/fill_record_builder.py`: 2 フィールド追加
- `scripts/v460/lib/fill_loop_orchestrator.py`: RunSessionState に 5 フィールド追加
- `scripts/v460/lib/orchestrator_post_cycle.py`: counter increment + summary log 追加
- `ztb/metrics/fill_quality.py`: FillRecord に 2 フィールド追加
