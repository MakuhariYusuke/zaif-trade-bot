# 555# CalibrationMap ランタイム統合

## 概要

546# §B で推奨された CalibrationMap を fill_test ランタイムに統合する。
554# で構築済みの offline batch (`models/v460/entry_gate_calibration.json`) を
fill_test 起動時にロードし、CycleGateAggregator に CalibrationGate
(EV ベースエントリー判定) を組み込む。
約定後には CalibrationMap の online 更新を行い、実運用中の性能統計を追跡する。

## 前提: CalibrationMap offline batch 状況 (554#)

- **データソース**: `fill_records/` JSONL (15,531 records, 38日分)
- **有効約定数**: 4,718 filled records (filled=True & pnl ≠ None)
- **出力先**: `models/v460/entry_gate_calibration.json`
- **CalibrationMap パラメータ**: EWMA tau=100, n_min=30, Beta prior (α=2, β=2)
- **Global 統計**: n_eff ≈ 200, p_win_lcb ≈ 0.38
- **Regime 別統計** (554# batch 出力):
  - `trending`: n_eff ≈ 73, p_win_lcb ≈ 0.33
  - `ranging`: n_eff ≈ 96, p_win_lcb ≈ 0.39
  - `high_vol`: n_eff ≈ 21 (n_min 未到達 → fallback 対象)
  - `unknown`: n_eff ≈ 10 (n_min 未到達 → fallback 対象)
- **Action Bin 分布**: Buy/Sell が主体 (Strong_Buy/Sell は稀, Neutral は fill されにくい)
- **構築ツール**: `scripts/v460/ml/calibration_batch.py` (`build_calibration_map()`, `load_calibration_state()`)

## 統合設計

### 1. YAML 設定 (`configs/v460/fill_test.yaml`)

```yaml
entry_gate:
  enabled: false                    # 初期は観測のみ (log_only モード)
  calibration_map_path: "models/v460/entry_gate_calibration.json"
  probability_mode: "lcb"           # lcb / mean / ucb
  ewma_tau: 100.0
  n_min: 30.0
  fee_rate: 0.0                     # maker fee (Coincheck maker=0)
  c_spread: 0.3                     # spread cost weight
  c_vol: 0.2                        # volatility cost weight
  c_imp: 0.5                        # market impact weight
  online_update: true               # 約定後に CalibrationMap を更新
```

### 2. FillTestConfig 拡張

`fill_config.py` に `entry_gate_*` フィールドを追加。
`fill_config_parser.py` に `_parse_entry_gate_section()` を追加。

### 3. FillTestRunner 起動時ロード

`run_fill_test.py` の `__init__` で:
1. `load_calibration_state()` で CalibrationMap インスタンスを生成
2. CalibrationGate を初期化
3. CycleGateAggregator に注入

### 4. CycleGate 統合 (Gate 10: calibration_gate)

- Soft Gate として Gate 9 の後に評価
- `CalibrationGate.evaluate()` を呼び出し、EV ≤ 0 なら blocked
- `enabled=false` の場合は log_only (blocking せず EV を記録)
- `GateCheckResult` として audit trail に記録

### 5. Post-cycle online 更新

`orchestrator_post_cycle.py` の `_process_post_cycle()` で:
- `record.filled == True` かつ `pnl_jpy != None` の場合
- `CalibrationMap.update(regime, action, gross_pnl, step)` を呼び出し

### 6. 状態永続化

- `FillTestState` に `calibration_map_state: dict | None` フィールド追加
- `_build_state_snapshot()` で `CalibrationMap.get_state()` をエクスポート
- `_restore_common_state()` で `CalibrationMap.load_state()` をインポート

## ztb 既存実装の活用

| コンポーネント | パス | 活用内容 |
|---|---|---|
| CalibrationMap | `ztb/trading/signal/calibration_map.py` | EWMA 統計 + 階層的 fallback |
| CalibrationGate | 同上 | EV ベースエントリー判定 + コストモデル |
| calibration_batch | `scripts/v460/ml/calibration_batch.py` | `load_calibration_state()` 関数 |

新規実装は最小限 (config 拡張 + 統合グルーコード) に抑え、
既存の CalibrationMap/CalibrationGate をそのまま利用する。

## リスクと緩和策

1. **Cold start**: offline batch で初期状態を提供 → n_eff ≈ 200 で起動
2. **過剰 blocking**: `enabled: false` で観測のみから開始 → EV 分布を確認後に有効化
3. **状態消失**: FillTestState 永続化で再起動時に CalibrationMap 復元
4. **コストモデル精度**: Coincheck maker fee = 0 で fee_rate=0.0 設定 → spread/vol/impact のみ
