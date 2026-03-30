# 658# セルフレビュー: 657# ログ・可読性・observability改善

SHA: `38519db72`

## 概要

657# (B-3 / A-4 / A-5) 実装直後のセルフレビュー。
ログノイズ削減、dead code 除去、pipeline observability 補完、可読性向上の4点を修正。

## 変更内容

### 1. inv_skew ログ最適化 (maker_price.py)

| 項目 | Before | After |
|------|--------|-------|
| ログレベル | 毎サイクル `logger.info` | 毎サイクル `logger.debug` + 60秒毎 `logger.info` |
| フィールド | `imbalance=` `raw=` `smo=` `mult=` | `imb=` `max_f=` `raw=` `smo=` `mult=` |

- **`max_f=` フィールド追加**: regime別 max_factor (ranging=0.40 / trending=0.15) が毎ログで確認可能
- **60秒 time throttle**: `_inv_skew_last_info_time` (monotonic) で INFO を間引き、ログノイズを大幅削減
- `__slots__` に `_inv_skew_last_info_time` を追加

### 2. dead code 削除 (skip_gate_evaluator.py)

```python
# 削除
_conditions_met = sum([_cond_spread, _cond_obi, _cond_vpin, _cond_vel])
```

657# の A-4 実装で段階制（条件充足数によるスコアリング）から全条件充足の二値判定に変更された際に、
`_conditions_met` が未使用になっていた。

### 3. toxic_sell_veto 可読性向上 (skip_gate_evaluator.py)

分岐ロジックを `_soft_mode` 変数に抽出:

```python
# Before: インライン条件式
if (
    self._config.toxic_sell_veto_as_offset_enabled
    or _decay < 0.5
):

# After: 意図が明確な変数
_soft_mode = (
    self._config.toxic_sell_veto_as_offset_enabled
    or _decay < 0.5
)
if _soft_mode:
```

コメントも補強: `分岐: as_offset=true→常にソフト / false→decayが50%未満でソフト化へフォールバック / それ以外→hard skip`

### 4. fill_config.py コメント修正

```python
# Before (誤): 段階制: 条件充足数 ≤ soft_max_conditions → offset boost, 全条件充足 → hard skip
# After (正): 全条件充足時でも hard skip せず offset boost で保守的に発注
```

657# A-4 で段階制から全条件ソフト化に変更された設計を正しく反映。

### 5. multiplicative_pipeline observability (multiplicative_pipeline.py)

`_exec_stages` JSON に `toxic_veto` ステージを追加:

```python
_exec_stages: dict[str, float | None] = {
    "ev": ...,
    "velocity": ...,
    "toxic_veto": sg_toxic_veto_offset_mult if _toxic_veto_offset_applied else None,  # 追加
    "trending": ...,
    "toxicity": ...,
    "vg_supp": ...,
    "alert": ...,
}
```

fill_record の `executor_offset_stages_json` に記録され、A-4 の offset boost が実際に適用されたかを事後分析可能に。

## 変更ファイル

| ファイル | 変更種別 |
|----------|----------|
| `scripts/v460/lib/maker_price.py` | log level + max_f field + throttle |
| `scripts/v460/lib/skip_gate_evaluator.py` | dead code削除 + _soft_mode 抽出 |
| `scripts/v460/lib/fill_config.py` | コメント修正 |
| `scripts/v460/lib/multiplicative_pipeline.py` | toxic_veto stage 追加 |

## 影響範囲

- **機能変更なし** — ログ・コメント・observability のみ
- 既存の動作・閾値・判定ロジックは一切変更していない
