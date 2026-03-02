# 227# Ranging×OBI方向非対称 + Velocity EMAフィルタ + import最適化 + getattr排除 + Config検証

> **種別**: fix  
> **フェーズ**: ph2 (G1.1-exec)  
> **前提**: 226# (`43b09080e`)  
> **日付**: 2026-03-04

---

## 概要

深層コードベース分析に基づく 6 カテゴリの改善を実施。市場理論 (AS, OBI) による信号精度向上、ホットパス性能改善、コード品質強化を同時に達成。

## 変更一覧

### C1: Ranging × OBI 方向非対称 (CRITICAL — 収益直結)

**課題**: Ranging (mean-reversion) 市場で offset discount が方向に無関心。OBI が明確な方向性を示していても buy/sell 対称に discount を適用していた。

**理論根拠**: Avellaneda-Stoikov 2008 — 情報非対称性リスクは OBI で推定可能。bid-heavy (imbalance > 0) は上昇圧力を示唆し、mean-reversion 環境では buy が有利 (反転で利食い可能)。

**実装**:
- `maker_price.py` の ranging offset discount ブロックに OBI 方向性ロジック追加
- `imbalance > threshold` の場合:
  - bid-heavy → buy discount 1.0 (中立、非対称で有利に) / sell discount 強化
  - ask-heavy → sell discount 1.0 (中立) / buy discount 強化  
- `ranging_obi_asymmetry_factor` (default=0.0: 無効) で強度制御
- `ranging_obi_threshold` (default=0.1) で適用閾値制御
- clamp: `[min_ratio, 1.0]` で安全な範囲に制限

**ファイル**: `scripts/v460/lib/maker_price.py` (~L521-555)

### C3: Velocity EMA ノイズフィルタ (CRITICAL — 信号品質)

**課題**: `compute_instant_velocity_bps()` の即時速度が Coincheck の薄板環境で bid-ask bounce ノイズに影響されやすい。

**理論根拠**: 低流動性市場では単一サンプルの価格変化がノイズ支配的。EMA による平滑化で真のトレンド速度を抽出。

**実装**:
- `_smoothed_velocity_bps` スロットを `MakerPriceCalculator` に追加
- `compute()` 内で raw velocity に EMA 適用: `smoothed = α * raw + (1-α) * prev`
- `velocity_ema_alpha` (default=1.0: 無効) で制御。α=1.0 で raw パススルー (後方互換)
- 初回サンプルはそのまま通過

**ファイル**: `scripts/v460/lib/maker_price.py` (~L825-839)

### H1+H5: Lazy Import 排除 (HIGH — ホットパス性能)

**課題**: `fill_loop_orchestrator.py` の hot loop 内で 4 つの lazy import、`maker_price.py` の `compute()` / `set_loss_boost()` 内で `import math` / `import time`。

**実装**:
- orchestrator: `load_alert_mode`, `MCBLevel`, `SADLevel`, `datetime/timezone` をファイル先頭に移動
- orchestrator: 3 つの追加 `from datetime import datetime, timezone` lazy import も削除
- maker_price: `import math` をファイル先頭に移動、`compute()` 内の `import math as _math` 削除
- maker_price: `set_loss_boost()` 内の `import time as _time` 削除 (既存 top-level `time` を使用)

**推定効果**: ~5μs/cycle 削減 + コード品質 smell 排除

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`, `scripts/v460/lib/maker_price.py`

### H2: getattr → 直接アクセス (HIGH — 可読性・性能)

**課題**: `fill_loop_orchestrator.py` で ~14 箇所の `getattr(self, "_attr", default)` が使用されていたが、全対象属性にクラスレベルデフォルト宣言あり。

**実装**:
- `getattr(self, "_soft_drawdown_interval_multiplier", 1.0)` → `self._soft_drawdown_interval_multiplier` 等
- `getattr(self._maker_price, "property", None)` → `self._maker_price.property` (4 箇所)
- 全対象属性のクラスレベルデフォルト存在をテストで検証

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`

### M1: Config バリデーション強化 (MEDIUM — 安全性)

**`__post_init__` に 4 ルール追加**:
- `loss_boost_decay_tau_sec > 0` (τ=0 で ZeroDivisionError 防止)
- `ranging_obi_asymmetry_factor ∈ [0, 1]`
- `ranging_obi_threshold >= 0`
- `velocity_ema_alpha ∈ (0, 1]`

**ファイル**: `scripts/v460/lib/fill_config.py`

## テスト

`tests/unit/v460/test_227_ranging_obi_velocity_ema_import_fix.py` — **21 テスト**:

| クラス | テスト数 | 内容 |
|---|---|---|
| TestRangingObiAsymmetry | 4 | bid-heavy buy/sell, below threshold, factor=0 |
| TestVelocityEma | 3 | EMA smoothing, alpha=1 passthrough, first sample |
| TestConfigValidation | 8 | tau, obi_factor, obi_threshold, ema_alpha 各境界 |
| TestImportOptimization | 5 | math, datetime, MCBLevel, SADLevel, load_alert_mode |
| TestOrchestratorClassLevelAttrs | 1 | class-level defaults 存在確認 |

**結果**: 3067 passed, 0 failed, 19 warnings

## Config 変更

| パラメータ | デフォルト | 説明 | YAML |
|---|---|---|---|
| `ranging_obi_asymmetry_factor` | 0.0 | OBI 方向非対称強度 [0,1] | ✅ |
| `ranging_obi_threshold` | 0.1 | OBI 非対称適用閾値 | ✅ |
| `velocity_ema_alpha` | 1.0 | velocity EMA α (0,1] | ✅ |

**全パラメータ default=無効** — 後方互換性完全保持。

## 未実装 (次回以降)

- **C2**: inventory skew time-weighted decay — inv_fill_history にタイムスタンプ追加、指数重み付け (τ=30min)
- **H3**: `hasattr(self, "_mcb"/"_sad")` → 直接アクセス (state_snapshot)
- **H4**: `is_buy_killed`/`is_sell_killed` の遅延評価 (side 依存)
- **M2**: `get_recovery_lot_scale()` → `consume_recovery_cycle()` リネーム
- **M5**: AS optimal spread κ/γ 較正
