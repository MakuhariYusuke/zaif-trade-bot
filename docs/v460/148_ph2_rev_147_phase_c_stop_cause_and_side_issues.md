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
| main 例外処理 | `asyncio.run(...)` を `try/finally` で囲むのみ (except なし) | `scripts/v460/run_fill_test.py:2027-2039` |
| kill 経路の記録 | kill時は `Kill switch ... activated` が出る設計 | `ztb/risk/circuit_breakers.py:40-51` |

---

## §3 147# ロジックの妥当性と不足

### 3.1 妥当

- 147# の「明示停止ログなし → 正常終了ではない」は正しい。
- `hours` 超過、`hard_loss_cap`、`KillSwitch` を否定するロジックも妥当。

### 3.2 不足 (見落とし)

1. **未捕捉例外停止の考慮不足**  
   `run_continuous()` の while 全体は包括 try/except で囲まれておらず、`run_single_cycle` 外での例外はプロセス終了に直結し得る。`scripts/v460/run_fill_test.py:1242-1719`

2. **stderr 側でのみ残るクラッシュ証跡の考慮不足**  
   `fill_test.log` に stop reason がなくても、親プロセス stderr に traceback が出ている可能性がある。`phase_c_start.log` が UTF-16 かつ `NativeCommandError` 形式で混在し、追跡性が低い。

3. **運用監視の最小要件不足**  
   147# の P2-A/B/C は方向性正しいが、「停止理由の永続化」が未定義。これがないと再発時に同じ推定を繰り返す。

---

## §4 別問題レビュー (重大度付き)

| # | 重大度 | 問題 | 影響 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | lock heartbeat 閾値不整合。`stale=1800s` に対して heartbeat 更新が実運用で疎になる | 生存プロセスを stale と誤判定し二重起動リスク。発注競合・評価汚染 | heartbeat を独立タスクで 60s 更新。`lock_stale_heartbeat_sec` は `>= 3 * heartbeat_period` |
| 2 | HIGH | 親プロセスの異常終了理由が必ず残らない | 停止原因特定不可、再発防止が遅れる | `main()` に top-level `except Exception` を追加し `traceback` をイベントログへ必ず保存 |
| 3 | MEDIUM | 起動ログ (`phase_c_start.log`) が運用解析しづらい | 監視自動化が困難、誤検知増加 | UTF-8 で stdout/stderr を分離保存 (`fill_test_stdout.log`, `fill_test_stderr.log`) |
| 4 | MEDIUM | retrain 設定の target/path ミスマッチ警告 (`target=pnl30` vs `...pnl120.pkl`) | モデル管理上の誤運用リスク、検証解釈ぶれ | model artifact naming/metadata を一致させる。運用判定時は metadata.target を必須チェック |
| 5 | MEDIUM | trades health が `stale=83.1h` のまま | retrain の実効性低下、SkipGate品質低下 | TradesRecorder の収集成否を日次監視し、stale 時は retrain を明示停止してアラート |

---

## §5 直近アクション提案 (Phase C を止めない前提)

| 優先 | アクション | 目的 | 工数目安 |
|---|---|---|---|
| P0 | 停止理由イベントを `fill_test_events.jsonl` に必ず出力 (start/stop/crash/signal) | 止まった理由を推定でなく事実にする | 0.3日 |
| P0 | lock heartbeat 更新を周期化 (60s) | stale 誤判定防止 | 0.2日 |
| P1 | 起動方法を標準化し stderr を恒久保存 | 例外証跡の欠落防止 | 0.2日 |
| P1 | target/path ミスマッチの運用ガード追加 | モデル誤適用防止 | 0.2日 |
| P1 | trades stale 検知時の自動通知 | 学習品質の劣化早期検出 | 0.2日 |

---

## §6 147# への追記推奨ポイント

147# に追記するなら以下 3 点を追加すると論理が締まる:

1. 「外部要因」確度を **暫定** と明記し、未捕捉例外停止を同列候補に置く。  
2. 完了条件 (§5) に「`fill_test_events.jsonl` で stop reason が記録されること」を追加。  
3. 本番前必須対策 (§3.5) に「lock heartbeat 周期更新」を追加。

---

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `docs/v460/148_ph2_rev_147_phase_c_stop_cause_and_side_issues.md` | NEW: 147# 補足レビュー |
