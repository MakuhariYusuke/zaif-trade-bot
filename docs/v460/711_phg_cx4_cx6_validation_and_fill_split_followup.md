# 711# CX4-CX6 検証と fill/PPO/SAC follow-up

## 概要

`CX4/CX5/CX6` の prompt を current runtime と照合し、妥当な部分は活かしつつ、危ない前提は補正して実装した。
同時に `fill_quality` の長大化をもう一段分割し、heavy test の固定費も削減した。

---

## prompt 検証結果

### CX4 skip_gate bypass gradual

- 方向性は妥当
- ただし global `bypass_mode` を即置換すると後方互換を壊すため、
  `bypass_mode_buy` / `bypass_mode_sell` を追加し、global 設定を fallback とした
- dryrun 分析は `skip_gate_as_prob` を優先し、未記録環境では `skip_gate_score` に fallback する方が安全

### CX5 calibration retrain

- 方向性は妥当
- hidden task は `calibration_batch.py` の date range 未対応だった
- そのため CLI は `--date-from/--date-to` を追加してから分析導線を整えた
- entry-gate EV の比較は、現行 runtime が使っている fallback calibration path に合わせる必要があり、
  prompt 例のまま独自 cost 項を足す実装は採用しなかった

### CX6 OBI U-shape redesign

- 問題設定は正しい
- ただし `linear` を壊す改修は危険なので、opt-in の mode 拡張として入れた
- `linear` は完全後方互換、`absolute/quadratic/excess` を比較可能にした
- runtime と analysis の数式 drift を避けるため、shared helper 化を優先した

---

## 今回の実装判断

1. `fill_quality` は小さく切り続ける
   - 今回は record payload shaping を `ztb/metrics/fill_record_payloads.py` へ分離
   - `FillRecord` 本体は `fill_quality.py` に残し、builder は薄い wrapper にした
2. heavy test は smoke と trainable を混ぜない
   - `test_enricher_skip_gate.py` の real-data smoke は最小 sample だけを使う
   - copy は `deep=False` を使い、不要なメモリ複製を減らした
3. PPO/SAC は runtime より test fixed-cost を優先
   - 現時点の支配点は `enricher` 実データ setup であり、scheduler 側は secondary

---

## 検証結果

- `py_compile`: pass
- focused CX4-CX6/config subset:
  - `136 passed in 3.78s`
- heavy subset:
  - `tests/unit/v460/test_fill_quality.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
  - `359 passed, 1 skipped, 5 warnings in 7.00s`

slowest:

1. `test_enrichment_with_real_data` setup `0.17s`
2. `test_oos_failed_keeps_fresh_signal` call `0.13s`
3. `test_raw_load_cache_is_bounded_and_clearable` call `0.09s`

---

## 残課題

1. `test_enricher_skip_gate.py` の real-data setup をさらに削る
2. `fill_quality.py` の残る report shaping を切る
3. PPO/SAC scheduler の exception-path fixed-cost をもう一段落とす

## 結論

提案していた方向性は概ね問題なかった。
ただし、prompt 文をそのまま当てるのではなく、

- backward compatibility
- runtime 既存契約
- analysis/runtime drift 防止

の 3 点で補正するのが正解だった。
