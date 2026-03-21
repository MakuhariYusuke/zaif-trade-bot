# 522# Balance-Forcing 完全撤廃 + 切り捨て/丸め無効化パラメータ監査

## 概要

348# で `balance_forced` を廃止したが、実質的な後継機能として
`balance_switch`・`recovery_skew`・`inventory_escape` が残存していた。
これらは「上がり続けるのに売らせる」「下がって買わせる」問題の原因であり、
522# で全て完全撤廃する。

また、切り捨て・値丸め・ceiling clamp により**実質無意味**になっている
コンフィグパラメータを水平監査した結果を記録する。

## §1 Balance-Forcing 完全撤廃

### §1.1 撤廃対象

| 機能 | ファイル | 行 | 説明 | 影響データ |
|------|----------|-----|------|-----------|
| `balance_switch` | orchestrator_balance.py L62-121 | side 切替 | 残高不足の requested side → opposite に即座切替 | 163/332 cycles (49%) |
| `recovery_skew` | orchestrator_balance.py L80-87 | kill bypass | opposite が kill-gated → kill を bypass して wide offset で通す | 25/332 cycles (7.5%) |
| `inventory_escape` | orchestrator_balance.py L152-198 | halt bypass | per-side halt を duty cycle で貫通 | N/A |

### §1.2 新アーキテクチャ

```
旧: requested side 残高不足 → opposite に切替して取引実行 (forced)
新: requested side 残高不足 → side を freeze → サイクル skip
    → 次サイクルで side_selector が自然に opposite を選択
```

**原則**: No Trade = Normal (250#)。残高不足時に forced な取引を行わない。

### §1.3 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `orchestrator_balance.py` | `_resolve_balance_and_preflight()` 簡素化、`_handle_inventory_escape_or_halt()` 削除 |
| `orchestrator_pre_cycle.py` | CycleContext から `inventory_escape`・`recovery_skew` フィールド削除 |
| `orchestrator_mid_cycle.py` | gate evaluator への `recovery_skew` パラメータ削除、`balance_forced_switch` 記録削除、inventory_escape 関連ロジック削除 |
| `cycle_gate_aggregator.py` | `evaluate()` から `recovery_skew` パラメータ・ロジック全削除 |
| `fill_config.py` | `inventory_escape_enabled`/`recovery_skew_enabled` デフォルト False |
| `fill_config_validation.py` | IE 必須バリデーション削除 (per_side_dd_halt_cycles=0 制約) |
| `config_hot_reload.py` | IE/RS の hot-reload エントリ削除 |
| `fill_test.yaml` | `inventory_escape_enabled: false`、`recovery_skew_enabled: false` |

### §1.4 テスト更新

| テストファイル | 変更 |
|----------------|------|
| `test_091_fixes.py` | balance_switch→freeze+skip パターンに更新 |
| `test_285_split_brain_guard.py` | IE 制約バリデーションの期待値を「エラーなし」に更新 |
| `test_292_observability.py` | hot-reload フィールドから IE/RS 除外 |
| `test_346_fill_config_validation.py` | IE 制約テストの期待値を「エラーなし」に更新 |

## §2 切り捨て・丸め無効化パラメータ監査

### §2.1 P0: Double-Ceiling Clamp (要対応)

| 項目 | 詳細 |
|------|------|
| 場所 | `maker_price.py:_apply_final_offset_ceiling()` + `offset_pipeline.py:execution_final_clamp` |
| 問題 | maker_price で ceiling clamp → offset_pipeline で multiplier 適用 → 再度 ceiling clamp。中間 clamp により multiplier が実値でなく clamp 値に作用 |
| 影響 | multiplier < 1.0 のケースで情報損失。例: base=0.28, mult=0.8 → 0.224 のはずが、pre-clamp で 0.25 → mult=0.8 → 0.20 |
| 対策案 | maker_price の中間 ceiling を削除し、offset_pipeline の execution_final_clamp のみで制御。**次回 fill test で検証** |

### §2.2 P1: dd_soft_lot_scale vs min_lot Floor

| 項目 | 詳細 |
|------|------|
| 場所 | lot scaling ロジック |
| 問題 | `order_quantity: 0.001` × `dd_soft_lot_scale: 0.5` = 0.0005 BTC < `min_lot: 0.001` → `max(min_lot, 0.0005)` = 0.001。scaling 無効 |
| 影響 | DD soft lot scaling が事実上のno-op |
| 対策案 | min_lot > base_lot × scale の場合は warning ログ追加 |

### §2.3 P2: sell_guard.offset_floor Dead Code

| 項目 | 詳細 |
|------|------|
| 場所 | `fill_test.yaml` sell_guard.offset_floor: 0.05 |
| 問題 | Pipeline 出力は常に ceiling 未満 (0.05–0.25)。`max(ceiling_result, floor)` で floor=0.05 は常に不要 |
| 影響 | パラメータが存在するが効果なし |
| 対策案 | 文書化のみ (floor は安全弁として残す) |

### §2.4 P3: Disabled Features with Full Config

以下の disabled feature は完全な設定ツリーを保持しているが、全てno-op:

| Feature | Enabled Flag | Dead Config Lines |
|---------|-------------|-------------------|
| `spread_adaptive` | `false` (L317) | ~10 行 |
| `early_exit` | `false` (L371) | ~3 行 |
| `imbalance` | `false` (L291) | ~5 行 |
| `smart_side` | `false` | ~3 行 |
| `microprice_side` | `false` | ~2 行 |
| `enable_dynamic_lot` | `false` | ~5 行 |
| `enable_auto_adapt` | `false` | ~7 行 |

対策: 将来の有効化に備えて残置。config audit としての記録のみ。

### §2.5 P4: Timeout Shadowing

| 項目 | 詳細 |
|------|------|
| 場所 | `order_timeout_sec: 90.0` vs `order_timeout_sec_sell: 75.0` vs `macro_sell_timeout_*` |
| 問題 | macro timeout (6-12s) が active な場合は sell timeout (75s) が shadow される |
| 影響 | 意図的な設計だが、優先順位がドキュメント化されていない |

## §3 次回 Fill Test 検証項目

1. balance-forcing 撤廃による balance_switch=0 cycles の確認
2. sell 側のスキップ増加とPnL改善の検証
3. P0 double-ceiling 修正の実装 (maker_price ceiling 削除)
