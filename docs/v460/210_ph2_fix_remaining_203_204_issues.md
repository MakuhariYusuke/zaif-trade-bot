# 210# 203#/204# 残課題解消 — FFD hot-reload同期, velocity配線, one-sided永続化, spread staleness, DRY snapshot

> **日付**: 2026-03-02  
> **前提**: 209# (セルフレビュー + 監査) 完了後、205# で整理された 203#/204# 残課題を解消  
> **コミット**: `391f3421c`

---

## 1. 背景

205# (200–204 レビュー) で §4–§8 に残課題として整理された項目、および
209# §4 で「監査で検出した未修正残課題」として記録された H2/H3/M5/L-2 の 4 件を
本チケットで解消する。加えて、FillTestState 構築の DRY 違反を共通化した。

---

## 2. 修正内容 (5件 + セルフレビュー 4件)

### H2: FFD hot-reload stale reference (HIGH)

| 項目 | 内容 |
|---|---|
| 問題 | `config_hot_reload.py` の `_rebuild_fast_fill_defense()` が新 FFD インスタンスを `runner._fast_fill_defense` に設定するが、`MakerPriceCalculator._fast_fill_defense` は旧参照を保持し続ける。hot-reload 後に FFD パラメータ変更が反映されない |
| 影響 | hot-reload でしか発生しないが、YAML 変更が maker 価格計算に反映されないサイレント障害 |
| 修正 | (1) `MakerPriceCalculator.update_fast_fill_defense(ffd)` public setter 追加、(2) `_HotReloadableRunner` Protocol に `_fast_fill_defense: object` 型宣言追加、(3) rebuild 後に `runner._maker_price.update_fast_fill_defense(_ffd)` で同期 |
| ファイル | `config_hot_reload.py`, `maker_price.py` |

### H3: Velocity dead code wiring (HIGH)

| 項目 | 内容 |
|---|---|
| 問題 | `fill_loop_orchestrator.py` で `_cycle_gate.evaluate()` 呼び出し時に `price_velocity_60s` 引数を渡しておらず、`CycleGateAggregator._check_velocity_skip()` が常に Gate 5 を通過させていた |
| 影響 | `velocity_skip_as_offset_enabled=True` (ソフトモード、デフォルト) では実害なし。hard mode 時は velocity skip が完全に無効化 |
| 修正 | (1) `price_velocity_60s=self._maker_price.last_mid_trend_bps` を evaluate 呼び出しに追加、(2) `last_mid_trend_bps` property を MakerPriceCalculator に追加、(3) `_check_velocity_skip` docstring にシグナル元差異を明記 |
| 備考 | `last_mid_trend_bps` は OB mid 瞬時速度であり、本来設計意図の `trade_vel_60s` (約定ベース 60 秒速度) とは異なる。符号規約は同一であり、閾値比較の意味論は保たれる。将来的に trade_vel_60s が実装された際は差し替え推奨 |
| ファイル | `fill_loop_orchestrator.py`, `maker_price.py`, `cycle_gate_aggregator.py` |

### L-2: one_sided_consecutive_count persistence (MEDIUM)

| 項目 | 内容 |
|---|---|
| 問題 | 207# で追加した `_one_sided_consecutive_count` が `FillTestState` dataclass に含まれておらず、再起動時にカウンタがリセットされる。片側連続実行制限が再起動後に無効化 |
| 修正 | (1) `FillTestState` に `one_sided_consecutive_count: int = 0` フィールド追加、(2) 状態復元処理 (2ブランチ: with/without regime_detector) で `_one_sided_consecutive_count` を復元、(3) 保存ヘルパーに含める |
| ファイル | `resilience.py`, `fill_loop_orchestrator.py` |

### M5: Spread staleness guard (MEDIUM)

| 項目 | 内容 |
|---|---|
| 問題 | `MakerPriceCalculator.last_spread` は `compute()` 実行時に更新されるキャッシュ値だが、Gate 8 (spread barrier) がこの値を参照して skip 判定を行うと、Gate 8 のブロックにより `compute()` が呼ばれず、stale な spread が永続するフィードバックループが発生し得る |
| 修正 | `_last_spread_time: float | None` を追加し、`last_spread` property で 60 秒超なら `None` を返却。stale spread によるブロックを防止 |
| ファイル | `maker_price.py` |

### DRY: FillTestState snapshot 共通化

| 項目 | 内容 |
|---|---|
| 問題 | `fill_loop_orchestrator.py` 内で `FillTestState(...)` を 3 箇所 (halt保存・progress保存・final保存) で同一 13+ フィールドを個別構築しており、フィールド追加時の変更漏れリスクが高い |
| 修正 | `_build_state_snapshot(total_count, filled_count, cumulative_pnl_jpy)` ヘルパーメソッドを追加し、3 箇所を統一呼び出しに置換。戻り値は `object` 型 (循環 import 回避) |
| ファイル | `fill_loop_orchestrator.py` |

---

## 3. セルフレビュー修正 (4件)

| ID | 重要度 | 内容 | 対応 |
|---|---|---|---|
| D | MEDIUM | `_HotReloadableRunner` Protocol に `_fast_fill_defense` 型宣言欠落 | `_fast_fill_defense: object` 追加 |
| E | MEDIUM | `_check_velocity_skip` で受け取る値が trade_vel_60s ではなく mid_trend_bps であることが未文書化 | docstring + インライン NOTE 追加 |
| F | LOW | config_hot_reload が `_fast_fill_defense` private slot に直接書込 | `update_fast_fill_defense()` setter 経由に変更 |
| H | LOW | コメント「4箇所」とあるが実際は「3箇所」 | 数値修正 |

---

## 4. 変更ファイル一覧

| ファイル | 変更量 | 変更内容 |
|---|---|---|
| `scripts/v460/lib/config_hot_reload.py` | +12 | FFD 同期 + Protocol 型宣言 |
| `scripts/v460/lib/cycle_gate_aggregator.py` | +3 | docstring 更新 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +56/−34 | `_build_state_snapshot` + velocity配線 + one_sided復元 |
| `scripts/v460/lib/maker_price.py` | +23/−1 | spread staleness + mid_trend_bps + FFD setter |
| `scripts/v460/lib/resilience.py` | +2 | one_sided_consecutive_count フィールド |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +235 | 11 新規テスト |

合計: **+331/−34** (6 files)

---

## 5. テスト

### 新規テスト (11件)

| クラス | テスト数 | 内容 |
|---|---|---|
| `TestFillTestStateOneSidedPersistence210` | 2 | フィールドデフォルト値、asdict→filter round-trip |
| `TestSpreadStaleness210` | 3 | fresh spread 返却、stale (>60s) None 返却、未設定 None 返却 |
| `TestMidTrendBpsProperty210` | 2 | 初期値 None、set & get |
| `TestFFDHotReloadSync210` | 1 | rebuild→setter→参照一致検証 |
| `TestVelocityGateWiring210` | 3 | hard block、None通過、soft mode通過 |

### 結果

- **210# テスト**: 11 passed
- **v460 全体**: 1172 passed, 5 failed (全て既存の無関係な失敗)

---

## 6. 残課題 (本チケット外)

| ID | 重要度 | 内容 | 理由 |
|---|---|---|---|
| H4 | HIGH | SellDynamicKillManager rolling PnL window 非永続化 | fill_records warmup 同様の仕組みが必要 (要設計) |
| M1 | MEDIUM | warmup `_warmup_daily_drawdown_from_records` の 2 回走査 | 数千件で顕在化、実運用では軽微 |
| M7 | MEDIUM | Partial fill ハンドリング不在 | Coincheck 0.001 BTC では稀 |
| 204# I | MEDIUM | Per-fill loss cap: offset boost 未実装 (interval延長+side封鎖のみ) | 効果検証後に追加予定 |
| C (SR) | LOW | spread staleness 60s がハードコード | Config 外部化は優先度低 |
| I (SR) | LOW | `_build_state_snapshot` の統合テスト不在 | 個別フィールドテストでカバー |
| P2 | LOW | σ-linked offset, OFI/PIN toxic flow detection | 205# §7 長期施策 |
