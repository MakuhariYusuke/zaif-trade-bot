# 072# ph2 準備: OB 特徴量トグル実装

| key | value |
|---|---|
| 番号 | 072 |
| フェーズ | ph2 |
| 種別 | impl |
| 参照 | `scripts/v460/ml/skip_gate.py`, `scripts/v460/run_fill_test.py`, `configs/v460/fill_test.yaml` |
| 作成日 | 2026-02-16 |
| テスト | 658 passed |
| 目的 | ph2 通過後に速やかに OB 特徴量を復元できるよう、config 駆動のトグルを実装 |

---

## §0 エグゼクティブサマリ

**071# で除去した OB 特徴量を、YAML フラグ 1 行の変更で即座に復元できる仕組みを実装。**

現在: `use_ob_features: false` (071# 状態維持)
復元: `use_ob_features: true` に変更するだけで OB 特徴量が復活

### 復元時の変更箇所 (ph2 通過後)

```yaml
# configs/v460/fill_test.yaml
skip_gate:
  use_ob_features: true   # ← false → true に変更するだけ
```

追加で S1/S2 も復元する場合:
```yaml
imbalance:
  enabled: true
smart_side:
  enabled: true
```

---

## §1 実装詳細

### 1.1 `skip_gate.py`

| 変更 | 内容 |
|------|------|
| `_BASE_FEATURE_COLS` | 16 cols: trade/price 系 (常時使用) |
| `_OB_FEATURE_COLS` | 3 cols: `spread_bps_ob`, `depth_imbalance_ob`, `side_aligned_imbalance` |
| `get_gate_feature_cols(use_ob)` | **新規関数**: `use_ob=False` → 16 cols, `True` → 19 cols |
| `GATE_FEATURE_COLS` | 後方互換: `get_gate_feature_cols(False)` で初期化 |
| `SkipGateConfig.use_ob_features` | **新規フィールド**: `bool = False` |
| `build_features_from_market_state` | OB パラメータ (`best_bid/ask`, `bid/ask_vol_5`) を Optional で再追加、`use_ob_features` フラグで条件生成 |

### 1.2 `run_fill_test.py`

| 変更 | 内容 |
|------|------|
| `FillTestConfig.skip_gate_use_ob_features` | **新規**: `bool = False` |
| YAML パース | `skip_gate.use_ob_features` → `skip_gate_use_ob_features` マッピング追加 |
| Runner init | `use_ob_features` を SkipGateConfig に注入、ログ出力に追加 |
| `run_single_cycle()` | `use_ob_features=True` 時のみ `get_orderbook(depth=5)` を SkipGate 用に取得し、`build_features_from_market_state` に渡す |

### 1.3 `fill_test.yaml`

```yaml
skip_gate:
  use_ob_features: false   # 072# OB トグル (ph2 通過後に true へ)
```

---

## §2 テスト追加

### `test_enricher_skip_gate.py` — `Test072OBToggle` (9 テスト)

| テスト | 検証内容 |
|--------|----------|
| `test_get_gate_feature_cols_no_ob` | `use_ob=False` → 16 cols, OB なし |
| `test_get_gate_feature_cols_with_ob` | `use_ob=True` → 19 cols, OB あり |
| `test_build_features_without_ob` | 特徴量辞書に OB キーなし |
| `test_build_features_with_ob` | OB 特徴量の値妥当性 (spread_bps, depth_imbalance, alignment) |
| `test_build_features_with_ob_sell_side` | sell 側の符号反転 |
| `test_build_features_with_ob_missing_data` | OB データなし → NaN フォールバック |
| `test_skip_gate_config_use_ob_default` | デフォルト `False` |
| `test_feature_count_consistency` | `get_gate_feature_cols` と `build_features` の出力キー一致 |

### `test_fill_test_config.py` (3 テスト)

| テスト | 検証内容 |
|--------|----------|
| `test_072_use_ob_features_default_false` | デフォルト `False` |
| `test_072_use_ob_features_from_yaml` | YAML `true` → config `True` |
| `test_072_yaml_roundtrip_use_ob_features` | YAML ↔ config roundtrip |

**結果: 658 passed, 0 failed**

---

## §3 アーキテクチャ

```
fill_test.yaml
  skip_gate.use_ob_features: false/true
       │
       ▼
FillTestConfig.skip_gate_use_ob_features
       │
       ▼
SkipGateConfig.use_ob_features
       │
       ├─ false → build_features(use_ob=False) → 16 features
       │           OB fetch スキップ
       │
       └─ true  → get_orderbook(depth=5) ──┐
                   build_features(           │
                     best_bid/ask,           │
                     bid/ask_vol_5,          │
                     use_ob=True             │
                   ) → 19 features ◄────────┘
```

---

## §4 次ステップ

| 条件 | アクション |
|------|-----------|
| ph2 通過 (800+ サンプル、ROC-AUC > 0.60) | `use_ob_features: true` に切替 |
| S1/S2 復元検討 | `imbalance.enabled: true`, `smart_side.enabled: true` |
| モデル再学習 | OB 特徴量込みの 19-feature モデルを `train_and_save_as_skip_gate` で学習 |
