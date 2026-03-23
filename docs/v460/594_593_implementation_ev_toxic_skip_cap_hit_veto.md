# 594# 593# 採用項目 A/B 実装

## 概要
593# 検証で採用が確認された改善項目 A (ev_toxic_skip) と B (CV cap_hit sell veto) を実装。

## 変更内容

### A: ev_toxic_skip — 中間帯スキップ (593# 優先度 A)
- **背景**: EV score が emergency(-8.0) と warning(-4.0) の間にある toxic flow は、offset 修飾 (widen) では防げないが emergency ほど極端ではない
- **EV 分布根拠**: filled p10=-2.09, skipped mean=-2.06, skipped p10=-5.02 → -5.0 を閾値に設定
- **実装**: `_ev_weighted_as_offset()` に emergency skip の直後、warning zone の前に toxic skip チェックを追加
- **閾値**: `skip_gate_ev_toxic_skip_threshold: -5.0` (デフォルト)
- **動作**: ev_score < -5.0 → `should_skip=True, reason="ev_toxic_skip"` でスキップ

### B: CV cap_hit sell veto 昇格 (593# 優先度 B)
- **背景**: sell × CV applied n=3, PnL=-6.67, 全件 cap_hit=true → widen が上限に張り付いて防御不能
- **実装**: `_apply_cross_venue_lead_lag_guard()` で cap_hit 検出後、sell 側かつ `cap_hit_sell_veto_enabled=true` の場合に veto へ昇格
- **設定**: `cross_venue_cap_hit_sell_veto_enabled: true` (YAML で有効、デフォルト False)
- **動作**: cap_hit + sell → `_cross_venue_lead_lag_vetoed = True` → InfeasibleQuoteError

## 変更ファイル

| ファイル | 変更内容 |
|----------|---------|
| `scripts/v460/lib/fill_config.py` | `skip_gate_ev_toxic_skip_threshold`, `cross_venue_cap_hit_sell_veto_enabled` フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML キーマッピング追加 (cv, skip_gate 両セクション) |
| `scripts/v460/lib/skip_gate_ev_weighted.py` | `_ev_weighted_as_offset()` に toxic skip ロジック追加 |
| `scripts/v460/lib/maker_risk_guards.py` | `_apply_cross_venue_lead_lag_guard()` に cap_hit sell veto ロジック追加 |
| `configs/v460/fill_test.yaml` | `ev_toxic_skip_threshold: -5.0`, `cap_hit_sell_veto_enabled: true` |
| `tests/unit/v460/test_593_ev_toxic_skip_and_cap_hit_veto.py` | 12 テスト (A: 6, B: 3, Config: 3) |

## テスト結果
- 593# 新規テスト: 12/12 passed
- 既存テスト回帰 (193#, 439#, 190#): 101/101 passed

## EV スキップ階層 (実装後)
```
ev_score < -8.0  → emergency_skip (ハードスキップ, 193#)
ev_score < -5.0  → ev_toxic_skip  (中間帯スキップ, 593# A) ← NEW
ev_score < -4.0  → warning zone   (offset × 0.7 保守化, 200#)
ev_score >= -4.0 → 通常 offset 修飾
```
