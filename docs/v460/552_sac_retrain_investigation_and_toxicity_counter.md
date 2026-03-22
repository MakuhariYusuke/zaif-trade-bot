# 552# SAC Retrain Scheduler 調査レポート — OOS Gate 持続失敗の根本原因

## 概要

SAC sidecar が 2026-03-19T16:34 以降、neutral fallback (bias=0.0, confidence=0.0) を継続出力。
fill_test は stale signal (TTL=7800s) として無視し、事実上 sidecar が 100% dead の状態。

## 1. 根本原因: OHLCV データファイルの更新停止

| 項目 | 値 |
|------|-----|
| データファイル | `data/btc_jpy_1m_full_registry_features.parquet` |
| 最終更新日時 | 2026-03-11 12:13 |
| ファイルサイズ | 171,674,434 bytes (~164MB) |
| 現在日時 | 2026-03-23 |
| **データ鮮度** | **12日間更新なし** |

`rolling_window_days=7` の設定により、retrain scheduler は毎回同一のデータスライスを使用:

- train: timestamp 1770009060 → 1770568020 (8,064 rows)
- val: timestamp 1770568140 → 1770708300 (2,016 rows)

## 2. Retrain 履歴

| 日時 (UTC) | status | gross_roi | trade_count | 備考 |
|------------|--------|-----------|-------------|------|
| 03-11 05:30 | error | - | - | DLL 初期化失敗 (torch c10.dll) |
| 03-11 05:31 | error | - | - | 同上 |
| 03-19 10:14 | error | - | - | Timestamp 型変換エラー |
| 03-19 10:15 | **deployed** | +6.8e-05 | 1,363 | **唯一の成功** |
| 03-19 16:34 | oos_failed | -5.5e-05 | 1,269 | ← ここから neutral fallback |
| 03-19 17:13 | oos_failed | -5.5e-05 | 1,269 | 同一データ・同一結果 |
| 03-19 20:51 | oos_failed | -5.5e-05 | 1,269 | |
| 03-20 15:51 | oos_failed | -5.5e-05 | 1,269 | debug_details 付き |
| 03-21 08:55 | oos_failed | -5.5e-05 | 1,269 | |
| 03-21 12:16 | oos_failed | -5.5e-05 | 1,269 | |
| 03-22 10:22 | oos_failed | -5.5e-05 | 1,269 | 最新 (24h前) |

### 重要な観察

1. **OOS 環境で取引ゼロ**: `env_metrics.total_trades=0, net_pnl=0.0`
   - `trade_count=1269` は `evaluate_model_oos` の集約値
   - SAC モデルがバリデーション環境で一貫して行動不活性
2. **gross_roi が全て -5.5e-05**: warm-start が同一データ上で同一重みを微調整 → 収束先が同一
3. **train/val の行数が不変**: データ更新がないため rolling window が毎回同じスライス

## 3. OOS Gate の仕組み

```
sac_retrain_scheduler.py → retrain_once()
  ├─ データロード: ohlcv_path から rolling_window_days 分を切り出し
  ├─ train_val_split: val_ratio=0.2 (最後の20%)
  ├─ warm-start: incremental_timesteps=15,000 追加学習
  ├─ OOS 評価:
  │   ├─ gross_roi > min_gross_roi (0.0) → ✅ PASS
  │   └─ trade_count >= min_trade_count (3) → ✅ PASS (通常)
  └─ 失敗時: _push_neutral_fallback() → cache/sidecar_signal.json に neutral 書込
```

OOS gate 自体は正常動作。**データが古いことが問題**。

## 4. 影響分析

- sidecar offset boost = 0 bps (全サイクル)
- sidecar dead → fill_test は 540# 以前の offset 戦略のみで動作
- 546# A (max_boost_bps 調整) の効果がゼロ
- 547# HIGH-2 (same-SHA validation) において sidecar 効果の測定不能

## 5. 修正方針

### 即座に対応可能 (P0)

1. **OHLCV データの更新パイプライン確立**
   - `scripts/v460/add_market_theory_features.py` は既存データに特徴量を追加するスクリプト
   - 新規1分足データの取得と parquet 更新を自動化する cron/scheduler が必要
   - Coincheck API から OHLCV を定期取得 → parquet に append → 特徴量再計算

2. **stale data guard 導入** (retrain scheduler)
   - parquet の最終タイムスタンプが N 日以上古い場合、retrain をスキップしログに警告
   - 同一データで繰り返し retrain する無駄を防止

### 中期 (P1)

3. **replay buffer のリセット検討**
   - 現在の buffer (`sac_sidecar.buffer.pkl`) に stale data が蓄積
   - データ更新後に cold-start (50K steps) で再訓練する選択肢

4. **min_gross_roi 閾値の見直し**
   - 現在 0.0 (> 0 で通過)
   - マーケットメイカー文脈では micro-positive ROI でも有用
   - しかし現在の -5.5e-05 はゲートが正しく機能している証拠

## 6. 関連ログ・ファイルの所在

| ファイル / ディレクトリ | 内容 |
|------------------------|------|
| `logs/sac_retrain_history.jsonl` | SAC retrain 全履歴 (status, gross_roi, trade_count, debug_details)。§2 の表はこのファイルから抽出 |
| `logs/g2_sac_train_run.log` | G2 SAC 初回訓練 stdout (2026-03-12) |
| `logs/g2_sac_train_stderr.log` | G2 SAC 初回訓練 stderr (37.6KB, DLL エラー等を含む) |
| `logs/g2_sac_train_stderr_run1.log` / `_run2.log` | 追加ラン stderr |
| `logs/g2_sac_train_background.log` / `_err.log` | バックグラウンド起動ログ |
| `models/v460/sac_sidecar.zip` | デプロイ中の SAC sidecar モデル |
| `models/v460/sac_sidecar.buffer.pkl` | SAC replay buffer (stale data 蓄積の可能性) |
| `cache/sidecar_signal.json` | SAC inference → fill_test の信号ファイル (neutral fallback 含む) |
| `configs/v460/experiments/g2_sac_train.yaml` | SAC 訓練・retrain のマスター設定 |
| `scripts/v460/ml/sac_retrain_scheduler.py` | retrain 実行スクリプト (`retrain_once()`) |
| `logs/tensorboard/SAC_*` | SB3 TensorBoard ログ |

---

## 7. 546# D 実装 (本コミットで完了)

本調査と並行して、546# D の Toxicity Distribution Counter を実装:

- `RunSessionState.toxicity_level_counts`: side×level 別カウンタ
- `RunSessionState.sidecar_nonzero_count`: sidecar bias≠0 カウンタ
- `orchestrator_mid_cycle.py`: `_evaluate_and_handle_cycle_gate` 内で toxicity level / sidecar nonzero を追跡
- `orchestrator_post_cycle.py`: 進捗ログに ORANGE+KILL 率 / sidecar nonzero 率を出力
- `test_551_toxicity_distribution_counter.py`: 20 テスト全 PASS
