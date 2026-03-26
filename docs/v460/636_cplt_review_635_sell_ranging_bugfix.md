# 636# Review: 635# sell/ranging 3層防御のバグ修正とテスト整備

## 概要

635# で実装した sell/ranging 3 層防御（ranging buy priority / sell_ranging_offset / no_feasible_quote freeze）のセルフレビュー。5 件のバグを発見・修正し、11 件の新規テストを追加。全 259 テスト通過。

**コミット**: `52ba93adc` (実装+修正), `600e1b2e0` (ドキュメント+残課題)

## 1. 発見・修正したバグ

### Bug 1: getattr 違反 (test_255 関連)

| 項目 | 詳細 |
|------|------|
| 場所 | `skip_gate_evaluator.py`, `orchestrator_post_cycle.py`, `side_selector.py` |
| 問題 | 635# で追加したコードが config 属性に `getattr()` でアクセスしていた |
| 影響 | test_255 (getattr 禁止テスト) が失敗する |
| 修正 | 直接属性アクセス (`self._config.xxx`) に統一 |

### Bug 2: offset_ceil 0.5 → 0.8（sell_ranging_offset 無効化）

| 項目 | 詳細 |
|------|------|
| 場所 | `configs/v460/fill_test.yaml` |
| 問題 | `sell_ranging_offset: 0.5` を追加したが、`offset_ceil: 0.5` のまま → 他の offset と合算するとすぐに ceiling 到達 → ペナルティが事実上 clamp されて無効 |
| 影響 | 635# の第 2 層防御（skip_gate penalty）が**完全に機能していなかった** |
| 修正 | `offset_ceil: 0.5` → `0.8` に引き上げ。sell_ranging_offset=0.5 が有効に機能する余地を確保 |

### Bug 3: ranging_buy_priority_max_consecutive 未定義

| 項目 | 詳細 |
|------|------|
| 場所 | `fill_config.py`, `fill_config_parser.py`, `fill_test.yaml` |
| 問題 | `SideSelector` が `self._config.ranging_buy_priority_max_consecutive` を参照するが、`FillTestConfig` にフィールドが存在せず、YAML パーサーにもマッピングがない |
| 影響 | config に `ranging_buy_priority_max_consecutive` がないのでデフォルト値 `3` が使われるが、YAML からの変更不能 |
| 修正 | FillTestConfig にフィールド追加 (default=3)、parser に smart_side セクションからのマッピング追加、YAML に値追加 |

### Bug 4: sell_ranging_offset YAML 未定義 + dead code

| 項目 | 詳細 |
|------|------|
| 場所 | `fill_test.yaml`, `fill_config_parser.py` |
| 問題 | `skip_gate_sell_ranging_offset` が FillTestConfig にはあるが YAML に記載なし → デフォルト値のみ |
| 影響 | hot-reload で sell_ranging_offset を調整不可 |
| 修正 | YAML に `sell_ranging_offset: 0.5` 追加、parser に sg_map マッピング追加 |

### Bug 5: VG velocity_threshold_bps テストドリフト

| 項目 | 詳細 |
|------|------|
| 場所 | `test_fill_quality.py` |
| 問題 | 630# で `sell_velocity_skip_threshold_bps` を `12.0` → `6.0` に変更したが、テスト側が `12.0` のまま |
| 影響 | テストが実態と乖離（ただし通過はしていた） |
| 修正 | テストの期待値を `6.0` に修正 |

## 2. 新規テスト: test_634_sell_ranging_suppression.py

11 テストを追加（全 3 層 + config 統合をカバー）:

### Layer 1: Ranging Buy Priority (4 tests)
| テスト | 検証内容 |
|--------|----------|
| `test_ranging_buy_priority_basic` | ranging 時に sell → buy に切り替わること |
| `test_ranging_buy_priority_max_consecutive` | 連続上限 (3) を超えたら sell を許可 |
| `test_non_ranging_no_priority` | trending 等では優先が発動しないこと |
| `test_frozen_side_overrides_priority` | freeze_side が ranging priority より優先 |

### Layer 2: Skip Gate Penalty (3 tests)
| テスト | 検証内容 |
|--------|----------|
| `test_sell_ranging_offset_config` | config フィールドが正しく設定されること |
| `test_sell_ranging_offset_in_source` | skip_gate_evaluator に penalty コードが存在 |
| `test_offset_ceil_accommodates_penalty` | offset_ceil >= sell_ranging_offset (clamp 回避) |

### Layer 3: No Feasible Freeze (2 tests)
| テスト | 検証内容 |
|--------|----------|
| `test_no_feasible_freeze_in_source` | orchestrator_post_cycle に freeze コードが存在 |
| `test_no_feasible_freeze_calls_freeze` | no_feasible_quote 時に freeze_side(2) が呼ばれること |

### Config Integration (2 tests)
| テスト | 検証内容 |
|--------|----------|
| `test_config_defaults` | FillTestConfig のデフォルト値が正しいこと |
| `test_yaml_round_trip` | YAML → config → 値の round-trip が正常 |

## 3. 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | offset_ceil 0.5→0.8, sell_ranging_offset 追加, ranging_buy_priority_max_consecutive 追加 |
| `scripts/v460/lib/fill_config.py` | `ranging_buy_priority_max_consecutive`, `skip_gate_sell_ranging_offset` フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | smart_side / skip_gate セクションのパーサーマッピング追加 |
| `scripts/v460/lib/skip_gate_evaluator.py` | sell/ranging penalty offset の適用コード (getattr→直接アクセス) |
| `scripts/v460/lib/side_selector.py` | ranging buy priority (getattr→直接アクセス) |
| `scripts/v460/lib/orchestrator_post_cycle.py` | no_feasible_quote → freeze_side(2) (getattr→直接アクセス) |
| `scripts/v460/analysis/analyze_fill_logs.py` | cancel reason breakdown by side/regime 追加 |
| `tests/unit/v460/test_634_sell_ranging_suppression.py` | 新規 11 テスト |
| `tests/unit/v460/test_fill_quality.py` | VG velocity 期待値 12→6 修正 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | 新規 config フィールド追加 |

## 4. offset_ceil 引き上げの副作用評価

`offset_ceil: 0.5 → 0.8` は skip_gate の全オフセット上限を引き上げるため、以下に影響:

| offset 種別 | 値 | ceil=0.5 時 | ceil=0.8 時 |
|-------------|-----|------------|------------|
| sell_ranging_offset | +0.5 | 他 offset と合算で即 clamp | 余裕あり |
| hour_offset (UTC14) | +0.5 | 他との合算で clamp | 余裕あり |
| hour_offset (UTC16) | +0.5 | 同上 | 余裕あり |
| narrow_spread_offset | +0.2 | 問題なし | 問題なし |

**リスク**: hour_offset の大きい時間帯（UTC14/16）で sell_ranging_offset と合算すると最大 1.0 → ceil=0.8 でも clamp される可能性。ただし、**clamp は安全方向**（過剰厳格化の抑制）なので実害は限定的。

## 5. 残課題

635# ドキュメントの末尾に追記済み（`600e1b2e0`）:

| 優先度 | 課題 | 状態 |
|--------|------|------|
| P1 | Hour skip UTC 6 時の過度な厳格化 | 未対応（要データ確認） |
| P2 | 630# 選別的 rollback 検討 | 保留（1-2 週間の実データ後） |
| P3 | Cross-venue sell 側チューニング | 据え置き |
| — | Sidecar status=error 100% | 据え置き（615# 級改修必要） |

## 6. 運用監視チェックリスト

- [ ] 1-2 日後: buy/sell 比率の改善確認（ranging 相場）
- [ ] 1 週後: no_feasible_quote 連続発生数の減少確認
- [ ] 1-2 週後: PnL 底上げ（sell/ranging 赤字縮小）、offset_ceil=0.8 副作用監視
- [ ] 2 週後: 630# パラメータの選別的見直し可否判断

---
*テスト結果: 259 passed (pytest tests/ -x --tb=short)*
*コミット: `52ba93adc` (impl+fix), `600e1b2e0` (docs)*
