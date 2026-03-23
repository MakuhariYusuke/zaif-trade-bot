# 569# 計測基盤の補完とパラメータ即時修正 (P1-P3)

**タイプ**: impl  
**日付**: 2026-03-23  
**前提**: 565# §3.2「計測基盤修正後に実施すべきもの」および §3.1 の残件  

---

## 概要

567# で追加した `analyze_fill_logs.py` の Execution Quality セクションを機能させるための根本改修 (P1相当) と、565# で指摘されたパラメータ系の即時修正（P1〜P3）を実施。

## 修正内容

### 1. `spread_capture_bps` 等の記録処理漏れ修正 (I2有効化)
- **ファイル**: `scripts/v460/lib/fill_record_builder.py`
- **問題**: `pnl_measurer.py` で計算した `spread_capture_bps` および `adverse_selection_cost_bps` が、`FillRecord` 生成時に辞書へマッピングされておらず、JSONLに出力されていなかった。
- **対応**: `_build_fill_measurement_fields` 内で明示的に値を記録するように修正。これにより、次回稼働以降から Execution Quality 分析が機能する。

### 2. P1: offset ceiling の引上げ
- **ファイル**: `configs/v460/fill_test.yaml`
- **内容**: 567# のI3で判明した「buyのpre_clamp中央値が0.30を超えている」事実に基づき、天井を緩和。
  - `offset_ceiling_ratio_buy`: 0.30 → 0.35
  - `offset_ceiling_ratio_sell`: 0.30 → 0.40

### 3. P2: CV favorable_tighten の Sell 側無効化
- **ファイル**: `scripts/v460/lib/maker_risk_guards.py`
- **内容**: 562# / 564# で指摘された「sell側はASリスクが高いため、ポジティブシグナルによるスプレッド縮小（タイト化）を行うべきではない」という提案を実装。`side == "sell"` の場合は `favorable_tighten` をスキップするように修正。

### 4. P3: Stage max_mult の導入 (乗数チェーン爆発防止)
- **ファイル**: `scripts/v460/lib/pre_order_adjustments.py`
- **内容**: `_apply_offset_multiplier` において、1段ごとの multiplier の上限を `2.0` に制限。複数の強気シグナルが重なった際の無用なスプレッド拡大を防ぐ (562# P-C 提案)。

### 5. 盲点6: Kelly と lot_sizing の矛盾解消
- **ファイル**: `configs/v460/fill_test.yaml`
- **内容**: `lot_sizing: enabled: false` にもかかわらず `kelly: enabled: true` となっており、計算だけが走る無意味な設定になっていたため、一時的に `kelly: enabled: false` に修正。

### 6. テスト保守: YAML ドリフト判定の修正
- **ファイル**: `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
- **内容**: 以前の 555# 変更によりコードデフォルトと一致するようになった `entry_gate_calibration_map_path` を `KNOWN_YAML_OVERRIDES` から削除 (pytest 落下解消)。

## 次のステップ
- **M1-M3**: `analyze_fill_logs.py` および `analyze_regime_moves.py` に対する更なる分析ロジック（体制遷移時のAS、自己相関、曜日効果）の拡充。
- **Task A (Gemini)**: eDRC の数理仕様に基づいた、乗法チェーンから加法スコアパイプラインへの全面改修の策定待ち。
