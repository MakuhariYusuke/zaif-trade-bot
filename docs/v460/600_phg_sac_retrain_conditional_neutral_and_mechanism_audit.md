# 600# SAC retrain conditional neutral fallback + unused mechanism audit

## 概要
SAC retrain scheduler の2つの設計バグを修正し、過去ドキュメントで有用と判断されながら未活用のメカニズムを監査した。

## 発見した問題

### Bug 1: OOS 失敗時の無条件 neutral fallback (CRITICAL)
- **症状**: 3/23 08:23 に成功 deploy (ROI=+0.000534) → 21:09 に retrain 失敗 → neutral fallback push → **成功 deploy 済みの信号が消滅**
- **根本原因**: `retrain_once()` の OOS 失敗パスが常に `_push_neutral_fallback()` を呼び出し、既存の active signal を上書き
- **影響**: 成功モデルの directional bias が失われ、sidecar が neutral に固定化

### Bug 2: 同一データウィンドウでの warm-start 過学習
- **症状**: 3/23 deploy 成功後、同じ rolling window で 3 回連続 warm-start → OOS 性能劣化
- **根本原因**: `retrain_once()` がデータウィンドウの変化を検出せず、同一データで繰り返し warm-start
- **影響**: CPU リソースの無駄遣い + 過学習によるモデル性能悪化

## 修正内容

### Fix 1: Conditional neutral fallback (600#)
- `_is_signal_fresh_and_active()` ヘルパー追加
  - 既存 sidecar signal が non-neutral かつ `max_signal_staleness_hours` (default: 24h) 以内なら True
  - True の場合は neutral push をスキップし、既存信号を保持
- OOS failure / trade_count gate failure の両方に適用
- error / import failure 時は従来通り neutral (安全側に倒す)

### Fix 2: Data window staleness guard (600#)
- `_last_deployed_val_ts_max` をモジュール変数で追跡
- deploy 成功時に val_df の最終 timestamp を記録
- 次回 retrain 時、val_timestamp_max が前回 deploy と同一なら即座に skip
- 新しいデータが到着するまで無駄な warm-start を防止

### Config 追加
- `SACRetrainConfig.max_signal_staleness_hours: float = 24.0`
- YAML: `sac_retrain: max_signal_staleness_hours: 24.0`

## 変更ファイル
- `scripts/v460/ml/sac_retrain_scheduler.py` — 両修正の実装
- `tests/unit/v460/test_sac_retrain_scheduler.py` — テスト更新 + 新規テスト追加
- `configs/v460/fill_test.yaml` — buy model unified fallback (前セッション)

## テスト結果
- 全 45 テスト PASSED (8.02s)
- 新規テスト: `test_oos_failed_keeps_fresh_signal`, `test_from_yaml_dict_600_max_signal_staleness(_default)`

---

## 未活用メカニズム監査結果

### 537# 前提の修正
- ✅ `composite_risk_enabled` — 既に `true` (threshold=1.0, 540# 調整済み)
- ✅ `micro_timeout` — 既に `true` (Step 3 完了, 496#)

### 有効化推奨 (P0-P1: YAML 1 行変更で即座に可能)

| 優先度 | 機能 | 現状 | 有効化努力 | 期待効果 |
|--------|------|------|-----------|----------|
| P0 | **entry_gate** (555#/589#) | `false` | Trivial | toxic low-EV エントリー排除 (+0.5-1.5 bps) |
| P1 | **spread_anomaly_detector** (513#) | `false` | Trivial | DRY market 防御 (offset拡大+interval延長+lot縮小) |
| P1 | **micro_circuit_breaker** (513#) | `false` | Trivial | 高ボラ防御 (-30 to -50 bps リスク軽減) |

### 慎重な検討が必要 (P2-P3)

| 優先度 | 機能 | 現状 | 有効化努力 | 備考 |
|--------|------|------|-----------|------|
| P2 | **additive_pipeline** (572#) | `false` | Moderate | 乗算→加算のパラダイムシフト、A/B テスト必須 |
| P3 | **eDRC** (575#) | α=β=0 | Significant | 576# インシデント後無効化、パラメータ再推定必要 |

### 無効のまま据置き (上記以外)
- adaptation (122# 因果分離問題), lot_sizing/confidence_lot/kelly (no-op)
- imbalance/smart_side (板情報不要), early_exit (137% 損失生成)
- microprice_side (AS seeker 問題), narrow_spread_pause (AB テスト未実施)
