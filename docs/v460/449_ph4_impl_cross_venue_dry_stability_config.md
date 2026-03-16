# 449# Cross-Venue DRY + 安定性 + Config 拡張

**種別**: impl (改善)  
**前提**: 448# (`9f83850d9`) — F3/F2 修正完了後の継続改善  
**コミット**: (本文末尾に記載)

---

## 背景

446#/447# レビューの P0 修正 (F2/F3) は 448# で完了。
本 449# は以下 4 視点からの横断改善:

| 視点 | 対象 | 根拠 |
|---|---|---|
| **重複排除 (DRY)** | spread_bps 3 重計算 | 同一式が 3 箇所に散在 |
| **計算量削減** | hint=None ログの冗長再計算 | 既計算値の再利用 |
| **市場理論補強** | confidence_floor ハードコード | Kyle (1985) λ の information floor に相当 |
| **動作安定性** | depth_imbalance_threshold ハードコード / getattr パターン | 設定可能化 + 安定パターン統一 |

---

## 変更一覧

### 1. spread_bps 重複計算の排除 (DRY)

**問題**: `(ref_mid - loc_mid) / loc_mid × 10,000` が 3 箇所で計算されていた:
- `fill_cycle_executor.py` L200: EMA 更新用
- `cross_venue_lead_lag.py` L230: hint 計算内部
- `fill_cycle_executor.py` L261: hint=None ログ

**修正**: `compute_cross_venue_lead_lag_hint()` に `precomputed_point_spread_bps` パラメータを追加。
caller が既に計算済みの値を渡せば内部で再計算をスキップ。
hint=None ログも既存変数 `point_spread_bps` を再利用。

### 2. confidence_floor の config 化 + 理論的根拠付与

**問題**: `base_conf = max(0.33, ...)` の `0.33` が理論的根拠なくハードコードされていた。

**修正**:
- `compute_cross_venue_lead_lag_hint()` に `confidence_floor` パラメータを追加 (デフォルト `0.33`)
- `FillTestConfig.cross_venue_confidence_floor` を新設
- YAML `confidence_floor` キーでオーバーライド可能

**理論根拠**: Kyle (1985) の informed trader モデルにおける λ (market impact coefficient) は、
小さな価格乖離にも情報価値が存在することを示す。confidence floor はこの
"information minimum" に対応し、spread が参照値より小さくても一定の信号強度を保持する。

### 3. depth_imbalance_threshold の config 化

**問題**: `maker_risk_guards.py` で `0.1` がハードコードされていた。

**修正**:
- `FillTestConfig.cross_venue_depth_imbalance_threshold: float = 0.1` を新設
- `maker_risk_guards.py` で `cfg.cross_venue_depth_imbalance_threshold` を参照
- YAML `depth_imbalance_threshold` キーで調整可能

### 4. getattr 統一 (安定性)

**問題**: `fill_cycle_executor.py` で `_run_id` / `_git_sha` を `try/except AttributeError` で取得していた。

**修正**: `getattr(self, "_run_id", "")` / `getattr(self, "_git_sha", "")` に統一。
6 行 → 2 行に削減、意図が明確。

---

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/cross_venue_lead_lag.py` | `precomputed_point_spread_bps` + `confidence_floor` パラメータ追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | 計算済み値の渡し + hint=None ログの DRY + getattr 統一 |
| `scripts/v460/lib/fill_config.py` | `cross_venue_depth_imbalance_threshold`, `cross_venue_confidence_floor` 追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML マッピング 2 件追加 |
| `scripts/v460/lib/maker_risk_guards.py` | ハードコード `0.1` → config 参照 |
| `tests/unit/v460/test_439_cross_venue_lead_lag.py` | `TestPrecomputedSpreadBps` (2 tests) + `TestConfidenceFloorParam` (2 tests) 追加 |
| `tests/unit/v460/test_336_fill_config_parser.py` | `test_cross_venue_449_config_fields` 追加 |

## テスト結果

```
68 passed in 1.35s
  - test_439_cross_venue_lead_lag: 35 passed (29 + 6 new)
  - test_336_fill_config_parser: 31 passed (30 + 1 new)
  - test_336_yaml_code_drift_prevention: 4 passed (既存: 新 config はデフォルト一致のため drift なし)
```

## 後方互換性

- `precomputed_point_spread_bps=None` (デフォルト) → 従来通り内部計算
- `confidence_floor=0.33` (デフォルト) → 既存挙動と完全一致
- `depth_imbalance_threshold=0.1` (デフォルト) → 既存挙動と完全一致

---

## コミット

`449# DRY+安定性: spread_bps 3重計算排除 + confidence_floor/depth_imb_threshold config化 + getattr統一`
