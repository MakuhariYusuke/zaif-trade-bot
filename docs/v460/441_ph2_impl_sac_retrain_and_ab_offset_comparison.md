# 441# SAC Retrain 有効化 + 440# A/B Offset 比較ツール

| 項目 | 内容 |
|---|---|
| 番号 | 441# |
| 分類 | ph2_impl (Phase2 Implementation) |
| 対象 | 437# §7 Phase 2 SAC Retrain + 440# A/B 検証 |
| 前提 | 440# regime-side offset committed (307a3e78e) |
| 目的 | SAC 定期再訓練の設定完成 + offset 変更の Before/After 比較基盤構築 |

---

## §1 SAC Retrain Phase 2: 設定完成

### §1.1 実装済み資産

| コンポーネント | ファイル | 状態 |
|---|---|---|
| SAC Sidecar Retrain Scheduler | `scripts/v460/ml/sac_retrain_scheduler.py` (365行) | ✅ 完全実装 |
| SkipGate Retrain Scheduler | `scripts/v460/ml/retrain_scheduler.py` (880行) | ✅ 完全実装 |
| Ops スクリプト (start/stop/status) | `ops/windows/retrain_scheduler.ps1` | ✅ 完全実装 |
| Hot-Swap Restart | `ops/windows/hot_swap_restart.ps1` | ✅ 完全実装 |
| fill_test.yaml retrain セクション | `configs/v460/fill_test.yaml` L837-870 | ✅ 設定済み |

### §1.2 本セッションで追加

**`configs/v460/experiments/g2_sac_train.yaml`** に `sac_retrain:` セクションを追加:

```yaml
sac_retrain:
  rolling_window_days: 7
  model_name: "sac_sidecar.zip"
  buffer_name: "sac_sidecar.buffer.pkl"
  signal_path: "cache/sidecar_signal.json"
  incremental_timesteps: 15000
  check_interval_sec: 300        # 5分間隔でポーリング
  retrain_interval_sec: 7200     # 最短2時間間隔
  retrain_interval_max_sec: 14400
  min_new_rows: 120
  min_gross_roi: 0.0
  n_eval_episodes: 1
  min_trade_count: 3
  confidence_roi_full: 0.005
```

### §1.3 起動手順

```powershell
# Step 1: ワンショットテスト
.venv\Scripts\python.exe scripts\v460\ml\sac_retrain_scheduler.py `
  --config configs/v460/experiments/g2_sac_train.yaml --once

# Step 2: バックグラウンド起動
.\ops\windows\retrain_scheduler.ps1 -Action start `
  -Config configs/v460/experiments/g2_sac_train.yaml

# Step 3: 状態確認
.\ops\windows\retrain_scheduler.ps1 -Action status
```

### §1.4 未実装・次期課題

| 課題 | 優先度 | 備考 |
|---|---|---|
| 自動監視・アラート | P2 | retrain 失敗時の Slack/email 通知 |
| ロールバック手順 | P2 | モデル劣化時の自動切り戻し |
| Adaptive interval | P3 | レジーム別の retrain 頻度調整 |

---

## §2 440# A/B Offset 比較ツール

### §2.1 設計

`scripts/v460/analysis/ab_offset_comparison.py` — fill_records を Before/After で分割し、regime×side 別に PnL/fill_rate/AS率を比較。Welch t検定で統計的有意性を評価。

### §2.2 ベースライン (Before: 31日分, 2/13-3/15)

| Regime | Side | n | fill_rate | pnl30 (bps) | AS率 | 440# 変更 |
|---|---|---|---|---|---|---|
| **ranging** | **buy** | **1321** | **32.0%** | **-0.407** | **23.5%** | discount 0.90→**1.15** |
| ranging | sell | 1335 | 36.2% | -0.135 | 29.1% | discount 0.90→**0.85** |
| unknown | buy | 47 | 81.0% | -1.384 | 38.3% | boost 2.0 (既存) |
| unknown | sell | 46 | 79.3% | -0.388 | 26.1% | boost **1.3** (新規) |

### §2.3 使い方

```bash
# ベースライン保存
python scripts/v460/analysis/ab_offset_comparison.py --save-baseline

# デプロイ後の比較 (split-date でデプロイ日を指定)
python scripts/v460/analysis/ab_offset_comparison.py --compare --split-date 2026-03-16

# 現在のベースライン表示
python scripts/v460/analysis/ab_offset_comparison.py --show-baseline
```

### §2.4 受入基準

| 指標 | 基準 | 根拠 |
|---|---|---|
| buy+ranging pnl30 改善 | Δ ≥ +0.2 bps | -0.407→-0.2 以上で効果あり |
| fill_rate 低下 | ≤ 10% 相対 | offset 拡大で fill_rate 低下は想定内 |
| 全体 pnl30 | 悪化なし (Δ ≥ 0) | target 以外への副作用なし |
| p-value | < 0.05 | 統計的有意 |

---

## §3 テスト

| テストファイル | テスト数 | 状態 |
|---|---|---|
| `test_441_ab_offset_comparison.py` | 11 | ✅ 全パス |
| `test_sac_retrain_scheduler.py` | 31 | ✅ 全パス (YAML 変更影響なし) |
| `test_440_regime_side_offset.py` | 19 | ✅ 全パス |

---

## §4 結論

1. **SAC Retrain**: YAML 設定完成、36テスト全パス、起動準備完了
2. **A/B 比較ツール**: ベースライン取得済み、デプロイ後即座に比較可能
3. **次のステップ**: 440# offset 変更を production にデプロイ → 数日後に `--compare` で効果検証
