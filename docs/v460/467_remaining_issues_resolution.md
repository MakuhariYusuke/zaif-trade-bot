# 467# 残課題解消: config_hash, deep-night ceiling, status_unknown_fast検知, hour-matched比較

**日付**: 2026-03-25
**ステータス**: 完了
**前提**: 461#-463# 残課題、465# model degeneration fix、466# memory leak fix

## 概要

461#-463# のレビューで発見された残課題を体系的に解消した。
464# の長期アーキテクチャ提案は別途として、運用上クリティカルな以下4項目を実装。

## 変更内容

### 1. FillRecord config_hash 追加 (462# 残課題)

**問題**: FillRecord にコード SHA (`git_sha`) はあるが設定識別子がなく、
同一コードでも YAML 設定変更時の効果分離が不可能だった。

**解決**:
- `FillRecord.config_hash: str | None` フィールド追加
- `manifest.compute_config_hash()` (SHA-256[:16]) で YAML 設定のハッシュ計算
- `FillRecordBuilderMixin._build_fill_record()` で payload に含める
- `run_fill_test.py __init__` で初期計算、`config_hot_reload.py` でリロード時再計算

**変更ファイル**:
| ファイル | 変更 |
|----------|------|
| `ztb/metrics/fill_quality.py` | `config_hash` フィールド追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | `_config_hash` クラス属性追加 |
| `scripts/v460/lib/fill_record_builder.py` | payload に `config_hash` 追加 |
| `scripts/v460/run_fill_test.py` | `__init__` で config_hash 計算 |
| `scripts/v460/lib/config_hot_reload.py` | YAML reload 時に config_hash 再計算 |

### 2. Deep-Night Ceiling 緩和機構 (461# P0)

**問題**: 461# データで JST 22-03h (UTC 13-18h) の AS 率が 57-100% に達するが、
offset ceiling が固定値のため防御的 offset 拡大が上限に制約される。

**解決**:
- `FillTestConfig.hour_ceiling_mult: dict[int, float]` フィールド追加
- `resolve_offset_ceiling(side, *, utc_hour=None)` に時間帯乗数を統合
- 後方互換: `utc_hour` 未指定時は既存動作と同一
- 3 箇所の呼び出し元 (`maker_price.py`, `offset_pipeline.py`, `orchestrator_post_cycle.py`) を更新
- `fill_config_parser.py` で YAML パース対応
- `config_hot_reload.py` で hot-reload 対応

**設定例** (YAML):
```yaml
hour_ceiling_mult:
  13: 2.0   # JST 22h — ceiling 2倍緩和
  14: 2.0   # JST 23h
  15: 2.0   # JST 00h
  17: 1.5   # JST 02h
  18: 1.5   # JST 03h
```

### 3. status_unknown_fast 連続検知 (461# P0)

**問題**: `status_unknown_fast` (注文ステータス API が None を返し、
リトライ全失敗 + elapsed < poll_interval×3) が連続発生しても検知・警告なし。
API 劣化や接続障害のサインを見逃す。

**解決**:
- `FillCycleExecutorMixin._consecutive_status_unknown_fast: int` カウンタ追加
- `_maybe_register_phantom()` 内でカウンタ更新:
  - `status_unknown_fast` → インクリメント
  - その他 (filled, timeout, cancel, status_unknown) → リセット
  - 3 連続以上で WARNING ログ出力

### 4. Hour-Matched Comparison ユーティリティ (462# 残課題)

**問題**: SHA/config 比較で時間帯の AS 率変動が交絡因子となり、
純粋なコード・設定効果の分離が困難。

**解決**:
- `scripts/v460/analysis/hour_matched_comparison.py` 新規作成
- 同一 UTC hour 内で A/B variant を比較 → 時間帯交絡を排除
- SHA 比較 (`--sha`) と config_hash 比較 (`--config-hash`) に対応
- Welch の t 検定で統計的有意性評価
- JSON 出力オプション (`--json`)

## テスト

`tests/unit/v460/test_467_remaining_issues.py` — 23 テスト:
- `TestFillRecordConfigHash` (4): from_dict 復元、デフォルト None
- `TestHourCeilingMult` (8): 乗数適用、サイド別、無効時、後方互換、hot-reload
- `TestConsecutiveStatusUnknownFast` (5): カウンタ増減、リセット、WARNING 閾値
- `TestHourMatchedComparison` (3): バケット計算、AS 率、unfilled
- `TestComputeConfigHash` (3): 決定性、キー順序独立、差分

## 残課題

| # | 優先度 | 内容 | 状態 |
|---|--------|------|------|
| 461# | P1 | ranging_low_vol_skip: buy-only soft mode の閾値最適化 | ✅ 468# 解消 |
| 461# | P1 | sell AS defense: 時間帯別 offset boost の実績評価 | ✅ 468# 解消 |
| 461# | P2 | deep-night hard_skip_utc_hours 拡張 (現在 [16,21] のみ) | ✅ 468# ceiling_mult で解消 |
| 464# | - | Adapter Strategy, ConfigMap 2.0, AB Test Framework | 長期提案・別途 |
