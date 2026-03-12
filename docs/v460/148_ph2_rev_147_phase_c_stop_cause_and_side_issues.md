# 148# 147補足レビュー: Phase C 停止原因再点検と別問題

**日時**: 2026-02-23  
**種別**: rev (レビュー)  
**Phase**: ph2  
**対象**: 147# `147_ph2_rpt_phase_c_24h_run_start.md`

---

## §1 結論 (先出し)

- 147# の主張「`2026-02-23 02:36:18` の停止は異常終了である」は妥当。
- ただし「外部要因が主因」の確度は **高→中高** に下げるべき。理由は、`run_fill_test.py` 側に **未捕捉例外で落ちる経路** が残っており、現状ログ設計だと stop reason が消えるため。
- 別問題として、**lock heartbeat 設計の閾値不整合** があり、将来的に「生存中プロセスを stale 判定して二重起動」し得る。収益性・安全性の両面で優先修正対象。

---

## §2 事実確認 (ログ/コード突合)

| 観点 | 確認結果 | 根拠 |
|---|---|---|
| 前回 run の最終ログ | `2026-02-23 02:36:18` で停止 | `results/v460/fill_test/logs/fill_test.log` |
| 停止種別ログ | `Fill test completed` / `Kill switch` / `LOSS CAP REACHED` なし | 同上 |
| 次回 run 開始 | `2026-02-23 03:37:52` で再開 | 同上 |
| ロック状態 | `03:37:49` に stale lock reclaim 実行 | 同上 (`[lock] Stale lockfile detected`) |
| 例外記録 | 末尾付近に `Traceback` なし | 同上 |
| main 例外処理 | `asyncio.run(...)` を `try/finally` で囲むのみ (except なし) → **148# P0 で top-level except 追加済** | `scripts/v460/run_fill_test.py:2159-2182` |
| kill 経路の記録 | kill時は `Kill switch ... activated` が出る設計 | `ztb/risk/circuit_breakers.py:40-51` |

---

## §3 147# ロジックの妥当性と不足

### 3.1 妥当

- 147# の「明示停止ログなし → 正常終了ではない」は正しい。
- `hours` 超過、`hard_loss_cap`、`KillSwitch` を否定するロジックも妥当。

### 3.2 不足 (見落とし)

1. **未捕捉例外停止の考慮不足** → **✅ 解決済** (148# P0: top-level except)  
   `run_continuous()` の while 全体は包括 try/except で囲まれておらず、`run_single_cycle` 外での例外はプロセス終了に直結し得た。`main()` に top-level `except Exception` を追加 (`run_fill_test.py:2159-2182`)。

2. **stderr 側でのみ残るクラッシュ証跡の考慮不足** → **✅ 解決済** (148# P1: stderr mirror)  
   `fill_test.log` に stop reason がなくても、親プロセス stderr に traceback が出た場合、`_TeeWriter` で `fill_test_stderr.log` にミラーリング (`run_fill_test.py:139-170`)。

3. **運用監視の最小要件不足** → **✅ 解決済** (148# P0: events.jsonl + P2-A: watchdog)  
   147# の P2-A/B/C は方向性正しいが、「停止理由の永続化」が未定義だった。`fill_test_events.jsonl` で start/stop/crash/signal を記録、`fill_test_watchdog.ps1` で死活監視。

---

## §4 別問題レビュー (重大度付き)

| # | 重大度 | 問題 | 影響 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | lock heartbeat 閾値不整合。`stale=1800s` に対して heartbeat 更新が実運用で疎になる | 生存プロセスを stale と誤判定し二重起動リスク。発注競合・評価汚染 | heartbeat を独立タスクで 60s 更新。`lock_stale_heartbeat_sec` は `>= 3 * heartbeat_period` |
| 2 | HIGH | 親プロセスの異常終了理由が必ず残らない | 停止原因特定不可、再発防止が遅れる | `main()` に top-level `except Exception` を追加し `traceback` をイベントログへ必ず保存 → **✅ `run_fill_test.py:2159-2182`** |
| 3 | MEDIUM | 起動ログ (`phase_c_start.log`) が運用解析しづらい | 監視自動化が困難、誤検知増加 | UTF-8 で stdout/stderr を分離保存 → **✅ `_TeeWriter` + `_setup_stderr_mirror()` at `run_fill_test.py:139-170`** |
| 4 | MEDIUM | retrain 設定の target/path ミスマッチ警告 (`target=pnl30` vs `...pnl120.pkl`) | モデル管理上の誤運用リスク、検証解釈ぶれ | → **✅ `_validate_side_target_path_mismatch()` at `retrain_scheduler.py:285-305`** |
| 5 | MEDIUM | trades health が `stale=83.1h` のまま | retrain の実効性低下、SkipGate品質低下 | → **✅ `trades_health_alert` event at `run_fill_test.py:1219-1231`** |

---

## §5 直近アクション提案 (Phase C を止めない前提)

| 優先 | アクション | 目的 | 工数目安 | ステータス |
|---|---|---|---|---|
| P0 | 停止理由イベントを `fill_test_events.jsonl` に必ず出力 (start/stop/crash/signal) | 止まった理由を推定でなく事実にする | 0.3日 | ✅ 実装済 `e21363a36` |
| P0 | lock heartbeat 更新を周期化 (60s) | stale 誤判定防止 | 0.2日 | ✅ 実装済 `e21363a36` |
| P1 | 起動方法を標準化し stderr を恒久保存 | 例外証跡の欠落防止 | 0.2日 | ✅ 実装済 |
| P1 | target/path ミスマッチの運用ガード追加 | モデル誤適用防止 | 0.2日 | ✅ 実装済 |
| P1 | trades stale 検知時の自動通知 | 学習品質の劣化早期検出 | 0.2日 | ✅ 実装済 |
| P2 | プロセス死活監視スクリプト | 停止検出・アラート | 0.2日 | ✅ 実装済 `fill_test_watchdog.ps1` |

---

## §6 147# への追記推奨ポイント

147# に追記するなら以下 3 点を追加すると論理が締まる:

1. 「外部要因」確度を **暫定** と明記し、未捕捉例外停止を同列候補に置く。 → **✅ 147# §3.3 に追記済**
2. 完了条件 (§5) に「`fill_test_events.jsonl` で stop reason が記録されること」を追加。 → **✅ 147# §5 に追記済**
3. 本番前必須対策 (§3.5) に「lock heartbeat 周期更新」を追加。 → **✅ 147# §3.5 に追記済**

---

## §7 実装トレース

§5 全アクションの実装箇所と検証結果:

| アクション | 実装箇所 | コミット | テスト |
|-----------|---------|---------|-------|
| P0: events.jsonl | `run_fill_test.py:95-132` (`_log_event()`) | `e21363a36` | 1440 passed |
| P0: heartbeat 60s | `fill_config.py:lock_heartbeat_period_sec=60.0` + `run_fill_test.py:1337-1343` (`_heartbeat_loop`) | `e21363a36` | 1440 passed |
| P1: stderr mirror | `run_fill_test.py:139-170` (`_TeeWriter`, `_setup_stderr_mirror`) | `a02dd9337` | 140 passed |
| P1: target/path guard | `retrain_scheduler.py:280-305` (`_validate_side_target_path_mismatch`) | `a02dd9337` | 140 passed |
| P1: trades stale | `run_fill_test.py:1219-1231` (`trades_health_alert` event) | `a02dd9337` | 140 passed |
| P2: watchdog | `ops/windows/fill_test_watchdog.ps1` | `4b91e2f7f` | 手動確認済 |

**次ステップ**: P2-B 自動再起動は [150#](150_ph2_plan_fill_test_auto_restart.md) で設計完了。実装は Phase C 完了後。

---

## §8 Codex レビュー依頼事項

本ドキュメントは 148# レビュー指摘への対応報告を兼ねる。以下の観点でのレビューを依頼:

### 8.1 実装の網羅性

- §4 で指摘した 5 問題すべてに対策が実施されたか
- 各実装の設計品質 (特に `_TeeWriter` の edge case、`_validate_side_target_path_mismatch` の検出精度)

### 8.2 残リスク評価

- P2-B 自動再起動未実装の影響度
- heartbeat が 60s 周期で更新されるが、現行 Phase C run は旧コードのため恩恵なし。次回 run 前に再起動必要か

### 8.3 コードベース確認依頼

```
scripts/v460/run_fill_test.py:
  L95-132:   _log_event() — イベント記録ロジック
  L139-170:  _TeeWriter / _setup_stderr_mirror — stderr ミラーリング
  L1219-1231: trades_health_alert イベント発行
  L1337-1343: _heartbeat_loop — 60s 周期 heartbeat
  L2159-2195: top-level except — crash 捕捉

scripts/v460/ml/retrain_scheduler.py:
  L280-305: _validate_side_target_path_mismatch — side 別ガード

scripts/v460/lib/fill_config.py:
  lock_heartbeat_period_sec, lock_stale_heartbeat_sec — heartbeat 設定値

ops/windows/fill_test_watchdog.ps1:
  全体: プロセス検出 (WMI)、lock parse、アラート判定
```

---

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `docs/v460/148_ph2_rev_147_phase_c_stop_cause_and_side_issues.md` | NEW: 147# 補足レビュー |
| `scripts/v460/run_fill_test.py` | P0: events.jsonl, heartbeat 60s / P1: stderr mirror, trades stale alert |
| `scripts/v460/lib/fill_config.py` | P0: heartbeat 設定値 |
| `scripts/v460/ml/retrain_scheduler.py` | P1: target/path mismatch guard |
| `ops/windows/fill_test_watchdog.ps1` | P2-A: 死活監視スクリプト |

---

## §9 Codex 深掘りレビュー追記 (2026-02-23)

### 9.1 指摘事項 (重大度順)

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | **CRITICAL** | `scripts/v460/run_fill_test.py:2165` | `runner._kill_switch.reason` を参照しているが、`KillSwitch` に `reason` プロパティは存在しない (`get_reason()` のみ)。kill 経路で `AttributeError` → `crash` 扱いになる。 | `runner._kill_switch.get_reason()` に修正し、`signal/hard_loss_cap/preflight_stop` の回帰テスト追加。 |
| 2 | **HIGH** | `scripts/v460/lib/fill_config.py:647-675` | `lock_heartbeat_period_sec` / `lock_stale_heartbeat_sec` が YAML マッピング対象外。`configs/v460/fill_test.yaml` から調整できず、運用で閾値変更不能。 | `tuning_map` に両キーを追加し、`fill_test.yaml` に明示値を追加。 |
| 3 | **HIGH** | `scripts/v460/run_fill_test.py:161-170` | 追記文書で「stdout/stderr 分離保存」と読めるが、実装は stderr のみ。stdout は起動方法依存。 | 仕様文言を「stderr ミラー」に修正、または stdout も同等実装。 |
| 4 | **MEDIUM** | `scripts/v460/run_fill_test.py:95-132`, `ops/windows/fill_test_watchdog.ps1:135-150` | `fill_test_events.jsonl` に複数プロセスが単純 append。競合時の行破損・順序逆転を排除していない。 | append 前の短時間 lock（`msvcrt`/lockfile）か、writer を 1 系統に統一。 |
| 5 | **MEDIUM** | `tests/unit/v460` 全体 | 148# 追加機能（event logger / stderr mirror / kill reason / watchdog 連携）の専用テストが見当たらない。 | `test_run_fill_test_events.py` を新設し、start/stop/crash/signal 各イベントを検証。 |

### 9.2 総評

- 148# の方向性（停止原因の可観測化）は妥当だが、**#1（kill reason 参照バグ）で可観測化自体が壊れるリスク**があるため最優先修正が必要。  
- 149/150 へ進む前に、#1 と #2 を先に塞ぐ方がデバッグ効率・運用品質ともに高い。

### 9.3 対応結果

| # | 対応 |
|---|------|
| 1 | **修正済**: `runner._kill_switch.reason` → `runner._kill_switch.get_reason()` (run_fill_test.py:2165) |
| 2 | **修正済**: `tuning_map` に `lock_heartbeat_period_sec`/`lock_stale_heartbeat_sec` 追加 + `fill_test.yaml` に明示値追加 |
| 3 | **修正済**: `_TeeWriter` docstring を「stderr ミラー専用」に修正 |
| 4 | **修正済**: `_log_event()` に `msvcrt.locking` によるファイルロックを追加 |
| 5 | **修正済**: `tests/unit/v460/test_148_fill_test_events.py` 新設 (16 テスト — event logger, TeeWriter, stderr mirror, KillSwitch reason 回帰) |
