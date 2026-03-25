# 622# SAD/MCB 有効化 + skip_gate 診断ログ改善

- **日付**: 2026-03-25
- **著者**: Copilot
- **コミット**: `f0af0c80d` (SAD/MCB), `47fc069ea` (診断ログ)
- **種別**: config / impl
- **目的**: 605#-608# 監査で検出された SAD/MCB 未有効化を解消し、620# バグ調査で判明した診断ログの盲点を改善

---

## §1 背景

### SAD/MCB 未有効化

606# で SAD (Spread Anomaly Detector) / MCB (Micro Circuit Breaker) の有効化が決定されたが、YAML の `enabled: false` のまま残存していた。605#-608# 監査で検出。

607# で hot-reload 対応済み（`config_hot_reload.py` に prefix `"sad_"` / `"mcb_"` 登録、`_rebuild_sad` / `_rebuild_mcb` コールバックで状態継承つきコンポーネント再構築）のため、fill test 再起動不要で反映可能。

### 診断ログの盲点

620# バグ調査時に以下の盲点が判明:

1. **skip_gate `evaluate()` のログが皆無**: 判定結果・使用閾値・regime を知る手段がなく、regime_thresholds bypass バグの発見が遅れた
2. **分析スクリプトが `skip_gate_threshold_used` を未使用**: fill_record に記録済みの閾値が分析に活用されておらず、adaptive threshold の異常動作を可視化できなかった

---

## §2 SAD/MCB 有効化

### 変更内容 (`configs/v460/fill_test.yaml`)

```yaml
# Before
spread_anomaly_detector:
  enabled: false  # 未インスタンス化

micro_circuit_breaker:
  enabled: false  # 未インスタンス化

# After
spread_anomaly_detector:
  enabled: true   # 607# hot-reload 対応済み

micro_circuit_breaker:
  enabled: true   # 607# hot-reload 対応済み
```

### hot-reload 動作

| 項目 | 内容 |
|------|------|
| チェック間隔 | 120 秒 |
| prefix 登録 | `"sad_"` → `_rebuild_sad`, `"mcb_"` → `_rebuild_mcb` |
| 再構築 | `_HOT_RELOADABLE_FIELDS` に `mcb_enabled` / `sad_enabled` 含む |
| 状態継承 | 再構築時に既存インスタンスの内部状態を引き継ぎ |

fill test 再起動なしで次回 hot-reload チェック時に反映。

---

## §3 skip_gate 診断ログ

### evaluate() INFO 診断ログ

`ztb/ml/skip_gate.py` の `evaluate()` 末尾に以下のログを追加:

```
[skip_gate] buy/ranging: pnl=+0.152 th=0.080 regime_floor=0.050 -> pass
[skip_gate] sell/trending: pnl=-0.031 th=0.120 -> skip
```

記録項目:
- `side` / `regime`: 判定対象
- `pred_pnl`: LGBM 予測 PnL (bps)
- `threshold_used`: 実効閾値（adaptive + regime_floor enforcement 後）
- `regime_floor`: regime_thresholds 設定値（該当時のみ）
- `reason`: 最終判定 (pass / skip / force_pass)

### 効果

- 620# 類似バグの即座発見が可能に（regime_floor が閾値に反映されていない場合、`th < regime_floor` となりログで検出可能）
- per-fill の判定根拠がログで完全にトレース可能

---

## §4 分析スクリプト改善

### threshold_used 分布分析

`scripts/v460/analysis/analyze_fill_logs.py` の `section_skip_gate()` に以下を追加:

1. **Threshold Used 分布** (pass fills): min / p50 / max
2. **Margin 分布** (score - threshold): 閾値ギリギリの pass が多いか可視化
3. **Regime 別 Threshold 分布**: regime ごとの閾値差を確認（bypass 検出）

出力例:
```
  --- Threshold Used (passed fills) ---
    min=0.050  p50=0.080  max=0.120  n=2341
    margin(score-th): min=0.001  p50=0.045  max=0.312
  --- Threshold by Regime ---
    ranging: min=0.050 p50=0.080 max=0.080 n=1205
    trending: min=0.100 p50=0.120 max=0.120 n=1136
```

620# 調査であれば、regime 間で閾値に差がないことが一目で判明し bypass バグを即検出できた。

---

## §5 テスト

全 2237 テストが pass（skip_gate ユニットテスト 206 件含む）。

---

## §6 まとめ

| 項目 | 内容 |
|------|------|
| SAD/MCB | `enabled: true` に変更、hot-reload で反映 |
| skip_gate 診断 | evaluate() に INFO ログ追加（regime + 閾値 + 判定） |
| 分析改善 | threshold_used 分布 + regime 別分析を追加 |
| 再起動 | 不要（hot-reload） |
