# 046# — 残タスク完了・ログ分析・再起動準備

**Session**: 046#  
**Parent**: 045# (d542cd8b1)  
**Date**: 2026-02-14  

---

## §1 ログ分析結果

`results/v460/fill_test/logs/fill_test.log` (1.69 MB) を解析。

### 定量サマリ
| 指標 | 値 |
|---|---|
| 総サイクル数 | 257 |
| 約定サイクル | 186 (fill rate ≈ 72.4%) |
| 残高不足エラー | 74 (28.8%) |
| ├ BTC 不足 (sell 側) | 42 |
| └ JPY 不足 (buy 側) | 32 |
| time_filter スキップ | 多数 (JST 17-04 UTC 8-19) |
| stale order キャンセル | 1 (起動時) |

### 発見事項
1. **Bug10 (NEW/HIGH)**: `insufficient_funds` エラー時にリトライが無駄に実行されていた。残高不足は 2 秒待っても回復しない。→ 即時 break で修正
2. **ゾンビプロセス由来ログ混在**: time_filter の sleep が 2 分間隔で連続出力 — 別プロセスの干渉。Bug7 (045#) のロックで解消済み
3. **`_update_dynamic_loss_cap` dead code**: `JPY_RESERVED`/`BTC_RESERVED` チェックが 045# E-4 修正後は不要に。修正
4. **balance API のゴミ通貨**: `_lending`, `_lend_in_use`, `_lent`, `_debt`, `_tsumitate` が不要な Balance オブジェクトとして生成されていた。除外フィルタ追加

---

## §2 実装内容

### Fix 1: Bug10 — insufficient_funds リトライスキップ (HIGH)
- **ファイル**: `scripts/v460/run_fill_test.py`
- **内容**: `run_single_cycle` の order retry ループで `cancel_reason == "insufficient_funds"` 時に即 `break`
- **効果**: API コール 50%削減 (1+1回 → 1 回)、rate limit 圧迫軽減
- **044# マッピング**: 新規発見 (ログ解析)

### Fix 2: soft/hard 二段 loss_cap (HIGH, 044# §9 P0-5)
- **ファイル**: `scripts/v460/run_fill_test.py`, `configs/v460/fill_test.yaml`
- **内容**:
  - `soft_loss_cap_ratio: float = 0.02` (残高の 2%) → ロット半減
  - `loss_cap_ratio: float = 0.05` (残高の 5%) → SAFE_STOP (既存)
  - `_soft_loss_cap_triggered` フラグで重複半減を防止
- **YAML**: `safety.soft_loss_cap_ratio: 0.02` 追加
- **効果**: 損失拡大を段階的に抑制。半減後も取引継続可能

### Fix 3: clean/quarantine データ分離 (HIGH, 044# §9 P0-2)
- **ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/run_fill_test.py`
- **内容**:
  - `filter_clean_records()` 関数追加: `git_sha` の有無でレコードを `(clean, quarantine)` に分割
  - レジューム時の累積 PnL 計算をクリーンレコードのみで実施
  - 149 件の汚染レコード (git_sha=blank) を quarantine 扱い
- **効果**: ゾンビプロセス由来データによるメトリクス汚染を防止

### Fix 4: balance 解析のゴミ通貨除外 (LOW)
- **ファイル**: `ztb/trading/live/exchanges/coincheck/adapter.py`
- **内容**: `get_balance` で `_lending`, `_lend_in_use`, `_lent`, `_debt`, `_tsumitate` サフィックスを除外
- **効果**: 不要な Balance オブジェクト生成を抑制、メモリ効率改善

### Fix 5: `_update_dynamic_loss_cap` dead code 除去 (LOW)
- **ファイル**: `scripts/v460/run_fill_test.py`
- **内容**: 045# E-4 で `reserved` が `locked` に統合されたため、`JPY_RESERVED`/`BTC_RESERVED` の個別チェックを除去
- **効果**: コード明確化

### Fix 6: YAML dead config 修正 (LOW)
- **ファイル**: `configs/v460/fill_test.yaml`
- **内容**: `max_consecutive_same_side: 2` (045# F7 で削除済み) → `max_preflight_skip: 10` に置換
- **効果**: YAML と code の整合性

---

## §3 テスト

### 新規テスト (11 件)
| クラス | テスト名 | 対象 |
|---|---|---|
| `TestBug10InsufficientFundsNoRetry` | `test_source_has_insufficient_funds_break` | Bug10 ソース検証 |
| | `test_cancel_reason_classification` | エラー分類ロジック |
| `TestSoftHardLossCap` | `test_config_has_soft_loss_cap_ratio` | 設定フィールド |
| | `test_soft_loss_cap_flag_initialized` | フラグ初期化 |
| | `test_soft_cap_ratio_less_than_hard` | 比率制約 |
| | `test_yaml_parser_handles_soft_loss_cap` | YAML 解析 |
| `TestCleanQuarantineFilter` | `test_filter_separates_by_git_sha` | 分離ロジック |
| | `test_filter_disabled` | 無効化 |
| | `test_all_clean` | 全クリーンケース |
| `TestBalanceCurrencyFilter` | `test_ignore_suffixes_in_source` | 除外サフィックス |
| | `test_loss_cap_no_dead_reserved_check` | dead code 除去確認 |

### テスト結果
- **451 passed** (440 既存 + 11 新規)
- 0 failed
- 6 warnings (scipy precision, pytest mark — 既知)

---

## §4 変更ファイル一覧

| ファイル | 変更行 |
|---|---|
| `scripts/v460/run_fill_test.py` | +42/-8 |
| `ztb/metrics/fill_quality.py` | +38/-0 |
| `ztb/trading/live/exchanges/coincheck/adapter.py` | +8/-3 |
| `configs/v460/fill_test.yaml` | +3/-2 |
| `tests/unit/v460/test_regime_detector.py` | +155/-0 |
| `docs/v460/046_ph2_remaining_fixes_and_log_analysis.md` | 本ドキュメント |

---

## §5 045# からの累積修正状況

| 044# ID | 重要度 | 状態 | 対応セッション |
|---|---|---|---|
| Bug7 | CRITICAL | ✅ 完了 | 045# |
| E-1 | CRITICAL | ✅ 完了 | 045# |
| A-1 | HIGH | ✅ 完了 | 045# |
| E-3 | HIGH | ✅ 完了 | 045# |
| E-4 | HIGH | ✅ 完了 | 045# |
| A-4 | HIGH | ✅ 完了 | 045# |
| F8 | MEDIUM | ✅ 完了 | 045# |
| A-7 | MEDIUM | ✅ 完了 | 045# |
| F7 | LOW | ✅ 完了 | 045# |
| P0-2 (clean/quarantine) | HIGH | ✅ 完了 | **046#** |
| P0-5 (soft/hard loss_cap) | HIGH | ✅ 完了 | **046#** |
| Bug10 (新規) | HIGH | ✅ 完了 | **046#** |
| balance ゴミ除外 | LOW | ✅ 完了 | **046#** |
| dynamic loss_cap dead code | LOW | ✅ 完了 | **046#** |
| YAML dead config | LOW | ✅ 完了 | **046#** |
