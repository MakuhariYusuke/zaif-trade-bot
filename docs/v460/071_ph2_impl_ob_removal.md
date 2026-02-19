# 071# ph2 実装: 板情報 (OB) 除去 — 価格ベース回帰

| key | value |
|---|---|
| 番号 | 071 |
| フェーズ | ph2 |
| 種別 | impl |
| 参照 | `scripts/v460/ml/skip_gate.py`, `scripts/v460/run_fill_test.py`, `configs/v460/fill_test.yaml` |
| 作成日 | 2026-02-16 |
| テスト | 647 passed |
| 目的 | SkipGate から板情報 (OB depth/volume) 特徴量を除去し、v459 方式の価格ベース判断に回帰 |

---

## §0 エグゼクティブサマリ

**070# の結論 (全 ML モデル ROC-AUC ≤ 0.54) を受け、OB depth/volume 特徴量をゼロシグナルとして廃止。**

- `GATE_FEATURE_COLS`: 19 → 16 (OB 3 特徴量除去)
- S1 imbalance フィルター: **無効化** (enabled: false)
- S2 smart_side: **無効化** (enabled: false, imbalance 依存のため)
- OB quality check / fallback model アーキテクチャ: **完全除去**
- OB は引き続き **価格発見** (`_compute_maker_price`, `_get_mid_price`) にのみ使用

### 削除した特徴量

| 特徴量 | 説明 | 削除理由 |
|--------|------|----------|
| `spread_bps_ob` | best_bid/ask からのスプレッド (bps) | IC ≈ 0, 070# で無信号確認 |
| `depth_imbalance_ob` | bid/ask volume 比率 | IC ≈ 0, 070# で無信号確認 |
| `side_aligned_imbalance` | side × depth_imbalance 交互作用 | 上記 2 つの派生、同様に無信号 |

---

## §1 変更詳細

### 1.1 `scripts/v460/ml/skip_gate.py`

| 変更 | 内容 |
|------|------|
| `GATE_FEATURE_COLS` | `spread_bps_ob`, `depth_imbalance_ob`, `side_aligned_imbalance` 除去 (19→16) |
| `SkipGateConfig` | `ob_freshness_sec` フィールド除去 |
| `OB_CRITICAL_FEATURES` | クラス変数ごと除去 |
| `_check_ob_quality()` | メソッド除去 (~30行) |
| `_fallback` / `set_fallback()` | fallback model アーキテクチャ除去 |
| `evaluate()` | `ob_age_sec` パラメータ除去、OB quality → fallback 委譲ロジック除去 |
| `build_features_from_market_state()` | `best_bid`, `best_ask`, `bid_vol_5`, `ask_vol_5` パラメータ除去、OB 特徴量計算コード除去 |

### 1.2 `scripts/v460/run_fill_test.py`

| 変更 | 内容 |
|------|------|
| `FillTestConfig` | `skip_gate_fallback_path`, `skip_gate_ob_freshness_sec`, `ob_fail_max_consecutive`, `ob_fail_offset_boost` 除去 |
| YAML パース | `fallback_path`, `ob_freshness_sec`, `ob_max_consecutive_fail`, `ob_fail_offset_boost` 除去 |
| Runner init | fallback model ロード、`_consecutive_ob_failures` カウンター除去 |
| `run_single_cycle()` | SkipGate 用 OB fetch 除去、OB quality check 除去、連続失敗 safety stop 除去 |
| `FillRecord` | `ob_quality_ok`, `ob_age_ms` フィールド設定除去 |

### 1.3 `configs/v460/fill_test.yaml`

| 変更 | 内容 |
|------|------|
| `imbalance.enabled` | `true` → `false` |
| `smart_side.enabled` | `true` → `false` |
| `skip_gate` セクション | `fallback_path`, `ob_freshness_sec`, `ob_max_consecutive_fail`, `ob_fail_offset_boost` 除去 |

---

## §2 残存 OB 使用 (継続)

以下は **価格発見** のための OB 使用であり、ML 特徴量ではないため継続:

| 箇所 | 用途 |
|------|------|
| `_compute_maker_price()` | best_bid/ask から指値価格を算出 |
| `_get_mid_price()` | best_bid/ask の中値で PnL 測定 |
| `feature_enricher.py` | AS 教師データ作成時の OB エンリッチメント (学習用、推論パス外) |

---

## §3 テスト影響

| テストファイル | 変更 |
|--------------|------|
| `test_enricher_skip_gate.py` | `Test065SkipGateTwoTier` → `Test065SkipGateNoOB` (OB なし evaluate テスト) |
| 同上 | `Test068OBQualityCheck` → `Test071OBRemoved` (OB メソッド不在確認) |
| 同上 | `Test058MarketStateFeatures` — 全 `build_features_from_market_state` 呼出から OB パラメータ除去 |
| `test_fill_test_config.py` | S1/S2 YAML テストを `enabled: false` に更新 |
| 同上 | `fallback_path` 3 テスト → `test_071_no_fallback_path_in_config` (廃止確認) |

**結果: 647 passed, 0 failed**

---

## §4 残存 `GATE_FEATURE_COLS` (16)

```python
GATE_FEATURE_COLS = [
    "side_buy",           # buy=1, sell=0
    "spread_jpy",         # 指値スプレッド (JPY)
    "spread_bps",         # spread / mid_price × 10000
    "offset_ratio",       # offset / mid_price
    "regime_trending",    # レジーム one-hot
    "regime_volatile",
    "regime_ranging",
    "hour_sin",           # 時刻 (sin/cos)
    "hour_cos",
    "trade_count_60s",    # 直近約定データ
    "trade_volume_60s",
    "buy_ratio_60s",
    "vpin_60s",
    "trade_intensity",    # count × volume
    "buy_pressure",       # buy_ratio × volume
    "spread_regime_interaction",  # spread × regime
]
```

---

## §5 次ステップ

| 優先度 | アクション | 根拠 |
|--------|-----------|------|
| **P0** | fill test 再投入 (dry-run) | OB 除去後の動作確認・データ蓄積 |
| P1 | データ蓄積→800+ サンプル | 070# の結論: ML 有効化の最低条件 |
| P2 | trade-only 特徴量の有効性評価 | 16 特徴量での ROC-AUC 再測定 |
