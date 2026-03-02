# 228# Inventory Time-Decay + hasattr排除

> **種別**: fix  
> **フェーズ**: ph2 (G1.1-exec)  
> **前提**: 227# (`7d93ad859`)  
> **日付**: 2026-03-04

---

## 概要

C2: 在庫偏重 (inv_skew) に時間減衰を導入し、古い fill 履歴の影響を自然に減衰。H3: `hasattr()` を完全排除しクラスレベル宣言に統一。

## 変更一覧

### C2: Inventory Skew Time-Decay (CRITICAL — 収益直結)

**課題**: 現行の inv_skew は直近 N fill を等重みで扱う。30 分前の fill と 30 秒前の fill が同じ重みでは、市場環境の変化に追随できない。

**理論根拠**: Guéant-Lehalle-Fernandez-Tapia (2013) — 在庫リスクは最終約定からの経過時間とともに情報価値が減衰する。MM のポジションリスク管理はリアルタイム性が重要。

**実装**:
- `_inv_last_update_time` スロットを `MakerPriceCalculator` に追加
- `update_inventory()` で `time.time()` を記録
- `_decayed_imbalance(now)` メソッド: `raw * exp(-elapsed/τ)` を返す
- `inv_net_imbalance` property が time-decay 適用後の値を返す
- `compute()` 内の inv_skew ブロックで `_decayed_imbalance(now)` を使用
- `_effective_sell_offset_floor()` の inv bypass 判定でも適用
- `inv_decay_tau_sec` (default=0.0: 無効) で制御
- 226# P5 の O(1) 計算量を完全保持

**ファイル**: `scripts/v460/lib/maker_price.py`

### H3: hasattr 完全排除 (HIGH — コード品質)

**課題**: `fill_loop_orchestrator.py` で `hasattr(self, "_mcb")` 等の動的チェックが 7 箇所。Mixin パターンでの防御だが、クラスレベルデフォルトで同等の安全性を確保可能。

**実装**:
- `_mcb`, `_sad`, `_cycle_strategy` にクラスレベル `None` デフォルトを追加
- `hasattr(self, "_mcb") and self._mcb is not None` → `self._mcb is not None` (4 箇所)
- `hasattr(self, "_cycle_strategy")` → `self._cycle_strategy is not None` (3 箇所)
- `hasattr(self._regime_detector, "current_regime")` → 削除 (冗長)
- 結果: `fill_loop_orchestrator.py` から `hasattr` が **0** に

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`

## テスト

`tests/unit/v460/test_228_inv_decay_hasattr_removal.py` — **17 テスト**:

| クラス | テスト数 | 内容 |
|---|---|---|
| TestInvDecayTimeDomain | 8 | τ=0 無効, 時間減衰, 5τ 近似ゼロ, fill リセット, 負値, 符号保持, 未 fill, property |
| TestInvDecayInCompute | 1 | compute() 内で decayed factor が時間経過で縮小 |
| TestInvDecayConfigValidation | 3 | 負値 ValueError, ゼロ有効, 正値有効 |
| TestInvDecayYaml | 1 | YAML parser |
| TestHasattrRemoval | 4 | _mcb/_sad/_cycle_strategy class-level, hasattr=0 検証 |

**結果**: 3084 passed, 0 failed, 19 warnings

## Config 変更

| パラメータ | デフォルト | 説明 | YAML |
|---|---|---|---|
| `inv_decay_tau_sec` | 0.0 | 在庫偏重時間減衰 τ (秒, 0=無効) | ✅ |

**推奨起動値**: `decay_tau_sec: 1800` (30 分)
