# 102# Structural Fixes Implementation

## 概要
100# で対応漏れとなった §1〜§7 + P1-5 の構造的課題を修正。  
パフォーマンス (計算コスト) を意識し、O(1) の set lookup や snapshot 方式を採用。

## 変更一覧

### §1 + §7: E3 計測タイミング修正 (HIGH → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | early_exit 発動時に 30s 計測が短縮されると、E3 (60s/120s) も同量前倒しされPnLラベルが汚染 |
| 修正 | E3 計測を「fill 後の絶対時刻基準」に変更。`t_post_fill_start` からの差分で残り sleep を動的計算 |
| ファイル | `scripts/v460/run_fill_test.py` |
| パフォーマンス | `time.time()` 1回 + `max(0, ...)` 1回の追加のみ、sleep 総時間は変化なし |

### §2: `_soft_loss_cap_triggered` レジューム復元 (HIGH → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | 再起動時に `False` 初期化 → 累積PnLが既にsoft cap超過でも二重ロット半減が発動 |
| 修正 | `resume_from_existing()` で復元した `cumulative_pnl_jpy` から論理的に判定して復元 |
| ファイル | `scripts/v460/run_fill_test.py` |
| パフォーマンス | 既存の cumPnL 計算に比較1回追加のみ (checkpoint I/O 不要) |

### §3: `_pre_shrink_lot` 整合性 (MED → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | `_check_balance_for_side()` がロット縮小時に `_pre_shrink_lot` を更新しない → 復元時に過大ロット |
| 修正 | sell/buy 両方の縮小パスで `_balance_shrink_active` でない場合に `_pre_shrink_lot = old_lot` を同期 |
| ファイル | `scripts/v460/run_fill_test.py` |

### §4: `soft_cap_jpy` 独立管理 (MED → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | 動的 `loss_cap_jpy` 更新に `soft_cap_jpy` が連動し、残高変動で cap が意図しないタイミングで変動 |
| 修正 | `_soft_cap_jpy_snapshot` を `run_continuous` 起動時にスナップショットし、以降はこれを参照 |
| ファイル | `scripts/v460/run_fill_test.py` |
| パフォーマンス | スナップショット保持で毎サイクルの除算計算を省略 |

### §5: JSONL 重複レコード防止 (MED → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | SIGINT 中断時に通常保存 + emergency dump で同一レコードが重複 → 集計誤差 |
| 修正 | `load_fill_records` に `cycle_id` ベースの dedup を追加 (set lookup O(1))。`load_fill_records_glob` でも cross-file dedup + emergency ディレクトリ自動統合 |
| ファイル | `ztb/metrics/fill_quality.py` |
| パフォーマンス | set(cycle_id) の O(1) lookup。書込側は変更なし (append 性能維持) |

### §6: `_check_balance_for_side` ロット復元 (MED → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | 残高回復後もロットが永続的に縮小したまま (`enable_dynamic_lot` 無効時) |
| 修正 | sell/buy 両方の balance check で、残高が `_pre_shrink_lot` 以上に回復した場合にロットを復元 |
| ファイル | `scripts/v460/run_fill_test.py` |
| パフォーマンス | 残高足りている場合の比較1回追加のみ |

### P1-5: regime detector warm-up (MED → 修正済み)

| 項目 | 内容 |
|------|------|
| 問題 | レジューム時に window=20 のバッファが空 → 20サイクル (約40分) regime 判定が unknown |
| 修正 | `resume_from_existing()` の既存レコード (filled + mid_at_fill) を直近 window*3 件まで事前投入 |
| ファイル | `scripts/v460/run_fill_test.py` |
| パフォーマンス | 最大 60 件の update() 呼出 (起動時1回のみ、ランタイム影響なし) |

## テスト

- **既存**: 794 passed (2テスト修正: cycle_id dedup 対応)
- **新規**: 12 passed (`test_101_structural_fixes.py`)
  - JSONL dedup (single/cross-file/emergency/no-dup), soft cap snapshot, pre_shrink_lot, lot restoration, regime warm-up, actual_measurement_sec

## パフォーマンス配慮

| 修正 | ランタイムコスト |
|------|----------------|
| E3 タイミング修正 | `time.time()` × 2 + `max()` × 2 (ナノ秒オーダー) |
| soft_cap resume | 比較1回 (起動時のみ) |
| JSONL dedup | set lookup O(1) per record (メモリ: cycle_id 文字列のみ) |
| regime warm-up | 最大60回 update() (起動時のみ、numpy 演算含み < 1ms) |
| soft_cap snapshot | float 1個の保持 (毎サイクルの除算省略) |
| lot restoration | balance check 内の比較1回 (既存 API 呼出に付随) |
