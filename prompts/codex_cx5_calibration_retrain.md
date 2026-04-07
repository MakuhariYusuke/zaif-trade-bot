# CX5: CalibrationMap 再学習 + entry_gate 復活準備

## 背景

entry_gate は CalibrationMap ベースの EV 判定を行うが、現在 100% suppressed (dead layer)。
原因: CalibrationMap の EV が P75=-1.914 に集中しており、どの threshold でもほぼ全件が negative EV → auto_disable。

708# 検証結果:
```
P5=-1.914  P25=-1.914  med=-1.914  P75=-1.914  P95=-1.337
EV >= -0.5:  0.0%
EV >= -1.0:  0.9%
EV >= -1.5: 19.1%
EV >= -2.0: 100.0%
```

## 現在の構成

```yaml
entry_gate:
  enabled: false                  # observe mode
  max_consecutive_blocks: 5
  buy_suppress_ev_threshold: -0.5  # 704# buy 側の軽度負EVは suppress
```

- CalibrationMap: `models/v460/entry_gate_calibration.json`
- Batch trainer: `scripts/v460/ml/calibration_batch.py` (554#)
- Online update: `scripts/v460/lib/orchestrator_post_cycle.py` L112-140

## 要件

### 1. CalibrationMap 再学習

`scripts/v460/ml/calibration_batch.py` を直近データ（2026-04-01～04-06）で実行し、新しい calibration state を生成。

```bash
.venv/Scripts/python.exe -m scripts.v460.ml.calibration_batch \
  --results-dir results/v460/fill_test \
  --date-from 2026-03-01 --date-to 2026-04-06 \
  --output models/v460/entry_gate_calibration_retrained.json
```

### 2. 再学習後の EV 分布分析 CLI

`scripts/v460/analysis/entry_gate_ev_distribution.py` を作成:

```bash
.venv/Scripts/python.exe -m scripts.v460.analysis.entry_gate_ev_distribution \
  --calibration-path models/v460/entry_gate_calibration_retrained.json \
  --results-dir results/v460/fill_test \
  --date-from 2026-04-04 --date-to 2026-04-06 \
  --json
```

出力:
- EV 分布統計 (P5, P25, med, P75, P95)
- threshold 別 pass/block 率: -0.5, -1.0, -1.5, -2.0
- side 別 EV 分布
- regime 別 EV 分布
- 旧 CalibrationMap との比較テーブル

### 3. buy_suppress_ev_threshold 最適化

threshold 候補 [-0.5, -1.0, -1.5, -1.8, -2.0] の counterfactual:
- 各 threshold で block される fill の avg PnL (post_fill_30s_pnl)
- 通過する fill の avg PnL
- net impact 推定

### 4. テスト

- `tests/unit/v460/test_710_entry_gate_ev_distribution.py`:
  - CLI 引数パース
  - EV 分布計算ロジック
  - threshold 比較ロジック

### 5. 安全制約

- 再学習結果は `models/v460/entry_gate_calibration_retrained.json` に出力（既存を上書きしない）
- 有効化は手動 YAML 変更 (`enabled: true` + calibration_map_path 変更) のみ
- 再学習後も EV 分布が集中している場合は WARNING 出力

## 影響範囲

| ファイル | 変更 |
|---------|------|
| `scripts/v460/analysis/entry_gate_ev_distribution.py` | 新規 |
| `scripts/v460/ml/calibration_batch.py` | 既存（確認のみ、変更最小限） |
| `tests/unit/v460/test_710_entry_gate_ev_distribution.py` | 新規テスト |

## 成功基準

1. 再学習後の EV 分布が `-1.914` 集中から改善
2. threshold 別 counterfactual で optimal threshold を特定
3. 既存テストの regression なし
