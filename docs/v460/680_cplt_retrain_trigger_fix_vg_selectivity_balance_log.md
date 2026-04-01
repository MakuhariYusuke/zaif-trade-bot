# 680# retrain trigger 修正 + VG 選択性回復 + balance ログ詳細化

## 概要
674# ログ分析（7時間稼働、120 records）から発見された3つの問題を修正。
SAC retrain の 2回目以降不発バグ（構造的デッドロック）、VG の非選択的発火（94%）、
balance insufficient の診断不足を解消。

## ログ分析結果 (674# SHA `000427953`, PID=36764)

### 定量サマリ

| 指標 | 値 | 672# 比較 | 評価 |
|------|-----|-----------|------|
| Total records | 120 | — | — |
| Fill rate | 29.2% (35/120) | ~20% | ✅ 改善 |
| avg_pnl30 | +0.14 bps | -0.23 bps | ✅ 方向改善（CI含0） |
| avg_pnl120 | +4.10 bps | — | ✅ 明確に正 |
| AS率 | 25.7% (9/35) | 23.7% | ≈同水準 |
| NFQ率 | 24.2% (29/120) | 13.6% | cap_bps 2.0 で増加（想定内） |
| MCB halt | 10/120 | — | restart直後σスパイク含む |
| Insufficient balance | 25.8% (31/120) | — | 🔴 新規発見 |

### Regime × Side パフォーマンス

| Regime | Side | n | avg_pnl30 | AS% | 評価 |
|--------|------|---|-----------|-----|------|
| trending_up | sell | 9 | +1.78 bps | 11% | ✅ 順張り好調 |
| ranging | mixed | 11 | +1.09 bps | — | ✅ 最良レジーム |
| trending_up | buy | 9 | -1.30 bps | 33% | ⚠️ 逆張り被弾 |
| trending_down | sell | 4 | -2.67 bps | 50% | 🔴 最悪コンボ |

## 発見された問題と修正

### 1. SAC retrain 2回目以降不発バグ（構造的デッドロック） 🔴

**症状**: 起動時1回だけ retrain 実行（00:01-00:12）、以降7時間 retrain なし。

**根本原因**:
```
retrain trigger: mtime(OHLCV) 変化で発火
  ↓
ensure_data_fresh(): max_data_stale_hours=48h → OHLCV は 15h old = "fresh"
  ↓
OHLCV ファイル更新されない → mtime 不変
  ↓
trigger の data_unchanged で永久ブロック
```

649# の fix は「データ鮮度チェックを retrain trigger から分離」したが、
更新 *頻度* の問題は未解決だった（stale=48h ≫ retrain_interval=2h）。

**修正 (2段階)**:
1. `max_data_stale_hours: 48 → 1.5`: retrain_interval(2h) 内に OHLCV 更新を保証
2. `DataFileRetrainTrigger.MAX_STALENESS_MULT = 3.0`: mtime 不変でも
   3×interval (6h) 経過で time_forced フォールバック発火（watchdog timer パターン）

### 2. VG (Volatility Guard) 非選択的発火 🟡

**症状**: fill 35件中 VG 発火 33件 (94%)。VG 発火群 pnl30=-0.08bps vs 非発火 +1.46bps。

**原因**: VPIN avg=0.68 に対して `vpin_continuous_min=0.40` → ほぼ全 cycle で発火。
情報理論的にはフィルタのエントロピー H(filter) ≈ 0 で、フィルタなしと等価。

**修正**: `vpin_continuous_min: 0.40 → 0.50`
- 新 norm = (0.68-0.50) / (0.80-0.50) = 0.60 → 二次曲線 boost = 1 + (1.5-1)×0.36 = 1.18
- 旧 norm = (0.68-0.40) / (0.80-0.40) = 0.70 → boost = 1 + (1.5-1)×0.49 = 1.245
- VPIN < 0.50 のサイクルでは boost=1.0 (中立) → 選択性回復

注意: n=3 (非VG) はサンプル不足。統計的結論ではなく情報理論的な選択性改善が主目的。

### 3. Balance insufficient ログ詳細化 🟡

**症状**: 31/120 (25.8%) が insufficient。buy 17 / sell 16 でほぼ均等。
BTC@10.6M × lot=0.001 ≈ 10,600 JPY に対し残高 ~21,700 JPY → 1 fill で片側枯渇。

**修正**: `orchestrator_balance.py` の insufficient ログに `jpy=, btc=, lot=` を追加。
根本原因は lot=0.001 vs 残高の構造的問題であり、lot 変更はリスクバランスに影響するため
ログ可視化のみ実施。

### 4. retrain trigger ログ INFO 昇格

`data_unchanged` 理由の trigger skip が DEBUG レベル → 7時間の沈黙に気付けなかった。
INFO に昇格し `since_last_retrain` 秒数を表示。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/experiments/g2_sac_train.yaml` | `max_data_stale_hours: 48→1.5` |
| `configs/v460/fill_test.yaml` | `vpin_continuous_min: 0.40→0.50` |
| `scripts/v460/ml/sidecar_scheduler_common.py` | `MAX_STALENESS_MULT=3.0` time_forced fallback |
| `scripts/v460/lib/orchestrator_balance.py` | insufficient ログに JPY/BTC/lot 詳細追加 |
| `scripts/v460/ml/sac_retrain_scheduler.py` | trigger skip ログ INFO 昇格 + elapsed 表示 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | `vg_vpin_continuous_min` allowlist 追加 |
| `tests/unit/v460/test_sac_retrain_scheduler.py` | time_forced fallback テスト追加 |

## Sell offset clamp 分析（修正不要の結論）

Sell 18件中 16件 (89%) が pre_clamp offset > ceiling=0.40 で clamp。
しかし **clamp 群 pnl30=+0.87bps vs unclamped -2.21bps** → ceiling は保護的に機能中。
672# の「offset pipeline は fill 制御に限定的に有効」と整合。操作不要。

## 関連ドキュメント
- 675# ML/SAC 総合分析 (問題の初期特定)
- 676# SAC sidecar P0 修正 (confidence/deploy gate)
- 679# SAC 報酬・γ 根本修正 (use_simple_reward, γ=0.95)
- 649# データ鮮度チェック分離
