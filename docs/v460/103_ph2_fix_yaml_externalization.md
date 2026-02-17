# 103# YAML 設定外部化 — マジックナンバー一掃

## 概要

100#/102# の実装で追加されたハードコード値および、既存の散在マジックナンバーを
`FillTestConfig` + `configs/v460/fill_test.yaml` に外部化。
Config drift（デフォルト値と YAML 値の乖離）も同時に修正。

## A. Config Drift 修正 (デフォルト値 → YAML と一致)

| フィールド | 旧 default | 新 default | YAML 値 |
|---|---|---|---|
| `order_timeout_sec` | 300.0 | **90.0** | 90.0 |
| `as_deadzone_bps` | 0.5 | **2.5** | 2.5 |
| `regime_min_confidence` | 0.4 | **0.3** | 0.3 |
| `skip_gate_adaptive_step` | 0.02 | **0.05** | 0.05 |

## B. YAML 未記載の既存フィールド追加

| フィールド | 値 | 追加先 |
|---|---|---|
| `batch_flush_interval_sec` | 600.0 | `tuning` セクション |
| `heartbeat_interval_sec` | 900.0 | `tuning` セクション |

## C. 新規 YAML 化パラメータ (tuning セクション)

| フィールド | 値 | 用途 | 旧ハードコード箇所 |
|---|---|---|---|
| `max_offset_ratio` | 0.30 | offset 比率の上限キャップ | 5箇所 (spread_adaptive, imbalance, adapt) |
| `min_offset_ratio` | 0.01 | offset 比率の下限 | wide spread floor |
| `loss_cap_update_interval` | 50 | loss_cap 残高更新サイクル | `__init__` |
| `min_loss_cap_jpy` | 50.0 | 動的 loss_cap 最低保証 (JPY) | `_update_dynamic_loss_cap` |
| `mid_trend_validity_sec` | 300.0 | mid trend 有効判定秒数 | `_compute_maker_price` |
| `balance_margin_ratio` | 1.01 | buy 残高チェックのマージン | `_check_balance_for_side` (4箇所) |
| `balance_shrink_consecutive` | 3 | shrink 発動の連続失敗回数 | `run_continuous` |
| `balance_shrink_divisor` | 2 | ロット分割係数 | `run_continuous` |
| `skip_gate_recent_trades_limit` | 50 | SkipGate 約定取得件数 | `run_single_cycle` |
| `status_unknown_retry_delays` | None (=[2,3,5]) | ステータス不明リトライ遅延 | ポーリングループ |
| `rate_limit_min_backoff_sec` | 5.0 | rate-limit 最低バックオフ | 注文リトライ |
| `save_retry_backoff_sec` | 0.5 | 保存リトライの初期バックオフ | `_try_save_batch` |
| `regime_warmup_multiplier` | 3 | regime warm-up ウィンドウ倍率 | `run_continuous` |
| `e3_60s_multiplier` | 2.0 | E3 60s 計測倍率 | `run_single_cycle` |
| `e3_120s_multiplier` | 4.0 | E3 120s 計測倍率 | `run_single_cycle` |
| `adapt_min_side_samples` | 20 | side 別適応の最小サンプル下限 | `_try_auto_adapt` |

## D. FastFillDefense 更新

- `FastFillDefenseConfig` に `max_offset_ratio` / `min_offset_ratio` を追加
- boost cap 計算が `0.30 / max(base, 0.01)` → `config.max_offset_ratio / max(base, config.min_offset_ratio)` に

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_fill_test.py` | FillTestConfig drift修正 + 18フィールド追加 + from_yaml mapping + 16箇所のハードコード置換 |
| `scripts/v460/lib/fast_fill_defense.py` | FastFillDefenseConfig に 2 フィールド追加 + cap計算の外部化 |
| `configs/v460/fill_test.yaml` | `tuning` セクション新設 (18 パラメータ) |
| `tests/unit/v460/test_094_stale_order.py` | drift修正に伴うアサーション更新 |
| `tests/unit/v460/test_fill_quality.py` | balance_shrink テストを config ベースに |
| `tests/unit/v460/test_regime_detector.py` | deadzone default テスト更新 |

## テスト結果

- 806 passed, 0 failed

## パフォーマンス影響

- なし (config 参照は O(1) 属性アクセス)
