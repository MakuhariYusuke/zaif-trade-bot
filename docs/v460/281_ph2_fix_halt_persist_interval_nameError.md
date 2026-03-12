# 281# fix: NameError `_HALT_PERSIST_INTERVAL` — 278# config化の参照漏れ修正

| 項目 | 内容 |
|------|------|
| 日付 | 2026-03-05 |
| 起票 | Copilot |
| 前提 | v460 fill_test on Coincheck, HEAD=d8aebac68 (278#) |
| コミット | `91f050a76` |
| 重大度 | **CRITICAL** (プロセス即死) |

---

## 1. インシデント概要

278# のマジックナンバー根拠化で `_HALT_PERSIST_INTERVAL = 10` を `FillTestConfig.halt_persist_interval` に config 化した際、orchestrator 内の参照箇所 2 箇所のうち **1 箇所** で旧シンボル `_HALT_PERSIST_INTERVAL` への参照が残留。

fill_test 起動後、daily_drawdown halt に入った瞬間に `NameError: name '_HALT_PERSIST_INTERVAL' is not defined` でプロセスが即死する。

---

## 2. 根本原因

278# で実施した config 化の変更:

```python
# BEFORE (278# 以前):
_HALT_PERSIST_INTERVAL = 10  # モジュールレベル定数

# 参照箇所 (2箇所):
# 1) 条件判定: self._halt_iter_count % _HALT_PERSIST_INTERVAL == 0
# 2) ログメッセージ: f"(next log @+{_HALT_PERSIST_INTERVAL} iters)"

# AFTER (278#):
# モジュールレベル定数を削除 → self.config.halt_persist_interval に移行
# 参照箇所 1 は移行完了 → self.config.halt_persist_interval
# 参照箇所 2 は移行漏れ → _HALT_PERSIST_INTERVAL のまま (NameError)
```

**原因分析**: 278# では orchestrator 内の 73 行を変更 (6 箇所のマジックナンバー → config/導出)。大量変更の中でログメッセージ内の参照を見落とした。条件判定側 (L1410) は正しく `self.config.halt_persist_interval` に移行されていたが、同じブロック内のログメッセージ (L1434) が漏れた。

---

## 3. 修正内容

| 箇所 | 変更前 | 変更後 |
|------|--------|--------|
| L1431 (コメント) | `_HALT_PERSIST_INTERVAL 毎` | `halt_persist_interval 毎` |
| L1434 (f-string) | `{_HALT_PERSIST_INTERVAL}` | `{self.config.halt_persist_interval}` |

差分 (2 insertions, 2 deletions):

```diff
-                # 211#: halt サイクル可視化ログ (entering + _HALT_PERSIST_INTERVAL 毎)
+                # 211#: halt サイクル可視化ログ (entering + halt_persist_interval 毎)
                 if _should_record_halt:
                     logger.info(
                         f"[daily_drawdown] Halt cycle #{self._halt_iter_count}"
-                        f" (next log @+{_HALT_PERSIST_INTERVAL} iters)"
+                        f" (next log @+{self.config.halt_persist_interval} iters)"
                     )
```

---

## 4. 影響範囲

- **影響**: daily_drawdown halt に入るとプロセスが即死 → halt 中のログ出力・state 保存・MCB/SAD フィード全て停止
- **発生条件**: `daily_pnl_bps` が `hard_limit_bps` を超過した場合 (通常運用で発生する正常なフロー)
- **発見契機**: 280# コミット後の fill_test 起動テスト

---

## 5. 教訓

### 278# マジックナンバー大量一括置換の反省

1. **同一ブロック内の複数参照**: 条件式とログメッセージという 2 つの異なる文脈で同じシンボルが使われている場合、条件式のみ置換してログメッセージを見落とすパターン
2. **grep 漏れ**: `_HALT_PERSIST_INTERVAL` で grep すれば検出できたが、置換完了の確認が不十分だった
3. **テスト網の穴**: 278# のテスト 34 件は config フィールド・バリデーション・導出ロジックを検証したが、halt ログパスの実行フローは未カバー

### 対策

- config 化の際は `grep -r OLD_SYMBOL` で **全参照の消滅を確認** してからコミット
- ログメッセージ内の定数参照は見落としやすい — f-string 内のシンボルも grep 対象に含める

---

## 6. テスト

本修正はログメッセージの参照修正のみであり、新規テストの追加はなし。278# の既存テスト (34 件) で config フィールドの存在は検証済み。

halt パスの統合テストは 282# で 15 件追加され、halt カウントダウン・IE 動作を包括的にカバー。

---

## 7. 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | L1431 コメント + L1434 f-string の参照修正 (2行) |
