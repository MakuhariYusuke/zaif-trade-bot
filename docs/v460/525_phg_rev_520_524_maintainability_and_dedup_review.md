# 525# 520#-524# レビュー — 重複排除・保守性・実装境界の点検

> 更新: 2026-03-21 16:45 JST
> 対象: 520#-524#, 関連実装, 短時間ログ確認
> 注: 今回は fill test ログの追加収益論点を無理に広げず、**DRY / 保守性 / 撤去済み概念の残存**を中心に見た。

## 1. Findings

| # | 重要度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | MEDIUM | `scripts/v460/lib/fill_config.py:647`, `scripts/v460/lib/fill_config_parser.py:629`, `scripts/v460/lib/guard_reason_classifier.py:78` | 522# は「balance-forcing 完全撤廃」と書いているが、`inventory_escape` / `recovery_skew` の概念が config / parser / taxonomy に分散残置している | 「runtime path は撤去、legacy config は後方互換で残置」という表現に文書を補正し、実装側は `legacy_recovery_compat` 的な一箇所へ隔離する |
| 2 | MEDIUM | `scripts/v460/lib/orchestrator_guards.py:255`, `scripts/v460/lib/orchestrator_balance.py:107` | 524# の preflight open-order cleanup 提案は妥当だが、そのまま `_handle_preflight_failure()` に書くと startup stale cleanup と二重実装になる | `_cancel_stale_orders()` を文脈付き共有 helper に昇格し、startup / preflight recovery で同じ経路を使う |
| 3 | LOW | `scripts/v460/lib/maker_price.py:785` | 523# で二重 ceiling を撤去した後も、private な `_apply_final_offset_ceiling()` が未使用のまま残っている | 本当に互換面が不要なら削除。残すなら tombstone comment を明記し、「再利用禁止」を添える |
| 4 | LOW | `scripts/v460/lib/orchestrator_balance.py:61`, `scripts/v460/lib/balance_checker.py:115` | `_check_balance_for_side()` / `BalanceChecker.check()` は「不足時に True」を返すため、呼び出し側が二重否定になり読みづらい | `_is_balance_insufficient_for_side()` などへ改名し、真偽の意味を名前に寄せる |
| 5 | LOW | `docs/v460/520_phg_plan_remaining_deferred_actions_screening.md`, `docs/v460/521_phg_master_deferred_and_architecture_carryforward.md` | 520# と 521# の役割分離は良いが、module inventory が prose として重複し始めている | 521# に module status の単一表を置き、520# は「履歴 + 521# へのリンク」に寄せる |

---

## 2. 総評

520#-524# の流れは全体として悪くない。特に次の方向は支持できる。

- 520#/521# で deferred docs と carry-forward の **更新先を一本化**したこと
- 522# で「残高不足時に無理に opposite side を打たない」という **No Trade = Normal** を徹底したこと
- 523# で ceiling を `offset_pipeline` 側へ **単一化**したこと
- 524# で `preflight_skip_exceeded` を「単なる pause 調整」ではなく、**open order 残存と回復フローの欠落**として見たこと

ただし、レビュー者としては次を補正したい。

- 522# の「完全撤廃」は runtime path についてはほぼ正しいが、**実装面では legacy surface がまだ広い**
- 523# の「dead code cleanup」は方向として正しいが、**private dead method と legacy config はなお残る**
- 524# の提案は筋が良いが、**今ある stale-order cleanup を再利用しないと同種ロジックが二重化する**

---

## 3. 520#/521# の評価

### 3.1 良い点

520# が「履歴」、521# が「living document」という切り分けは合理的である。これは複数エージェント運用で強い。

また、521# の

- `scripts/v460` は orchestrator / run context / compatibility shim
- `ztb` は reusable domain logic

という整理も妥当で、今後の `lib -> ztb` 議論の土台として機能する。

### 3.2 改善したい点

ただし 520#/521# で module 名や carry-forward 状態が prose として重複し始めている。今はまだ読めるが、継続更新するとズレやすい。

保守性の観点では、521# に

- module
- current owner (`scripts` / `ztb` / shim)
- status (`done` / `converging` / `future`)
- next action

の表を置き、520# は「なぜその表に至ったか」の履歴だけに寄せた方が DRY である。

---

## 4. 522# のレビュー

### 4.1 方向性は支持

`balance_switch` / `recovery_skew` / `inventory_escape` を runtime path から外し、

- 残高不足なら skip
- side freeze により次サイクルで自然に opposite を選ばせる

へ寄せたのはよい。市場理論的にも、forced side switching は「損失確定方向へ participation させる」癖を持ちやすく、止める価値が高い。

### 4.2 ただし「完全撤廃」の書き方はやや強い

実装では次が残っている。

- `scripts/v460/lib/fill_config.py:647-652`
  - `inventory_escape_enabled`
  - `inventory_escape_duty_cycle`
  - `recovery_skew_enabled`
  - `recovery_skew_offset_mult`
- `scripts/v460/lib/fill_config_parser.py:629-633`
  - YAML から `inventory_escape_*` をまだ読む
- `scripts/v460/lib/guard_reason_classifier.py:78-79`
  - `inventory_escape_*` reason を taxonomy に保持

これは runtime path の未撤去というより、**後方互換のための概念残置**である。したがって 522# の表現は

- 「実行経路からは撤去」
- 「互換 surface は残置」

に補正した方が、読者にとって正確である。

### 4.3 保守性提案

今のままだと legacy 残置が

- config dataclass
- parser
- hot reload
- guard taxonomy
- tests

に分散している。これは次に誰かが掃除するときに負担が重い。

おすすめは二択である。

1. **短期**: `LEGACY_RECOVERY_COMPAT_FIELDS` のような一箇所へ明示的にまとめる
2. **中期**: 一回の breaking cleanup で完全削除する

中途半端に散らして残すのが一番つらい。

---

## 5. 523# のレビュー

### 5.1 二重 ceiling 撤廃は支持

`maker_price.py` 中間 clamp を外し、`offset_pipeline` の `execution_final_clamp` に寄せた判断は妥当である。SSOT 化としても筋が通っている。

### 5.2 ただし private dead method が残っている

`scripts/v460/lib/maker_price.py:785` の `_apply_final_offset_ceiling()` は、523# 後の `compute()` から呼ばれていない。private method なので、ここを「互換性のために残す」というのはやや弱い。

この状態の問題は、将来誰かが

- `execution_final_clamp` 側を直した
- しかし `_apply_final_offset_ceiling()` 側は放置した

となったとき、概念上の二重実装が再発しやすい点である。

推奨は以下。

- 本当に不要なら削除
- すぐ消せないなら docstring 冒頭に「unused legacy helper / do not reuse」と書く

### 5.3 もう一歩の整理案

523# は `dead code cleanup` を掲げているので、次にやるなら「削除」と「compat 残置」をレベル分けして書くと良い。

- Level A: runtime から参照が完全に消えた private helper → 削除候補
- Level B: 旧 YAML / 旧 test 互換のために残す public field → compat 管理へ隔離

この分類があると、cleanup の議論がぶれにくい。

---

## 6. 524# のレビュー

### 6.1 問題意識は妥当

`preflight_skip_exceeded` を

- 残高不足
- pause 回数
- safe stop

の数値問題だけでなく、**未解放 open order による両側膠着**として見たのは良い。ここは実装改善の価値が高い。

### 6.2 ただし実装は「足す」のではなく「共有化する」べき

既に `scripts/v460/lib/orchestrator_guards.py:255-286` に startup stale cleanup がある。524# の提案をそのまま `_handle_preflight_failure()` に書くと、

- open order 取得
- order cancel
- ログ整形
- 例外処理

が二重化する。

したがって実装するなら、次の形がよい。

- `_cancel_stale_orders()` を `startup` 専用 helper のまま使い回さない
- `cancel_open_orders_for_recovery(reason: str, source: str)` のような共有 helper に抽出する
- startup / preflight / 未来の deadlock recovery が同じ helper を使う

これなら DRY を守れるし、ログ prefix も `[startup]` 固定から脱却できる。

### 6.3 `orchestrator_balance.py` を再肥大化させない

`_handle_preflight_failure()` は既に

- skip record
- lot shrink
- pause
- safe stop

を抱えている。ここに open-order recovery を直書きすると、また monolith に戻る。

なので 524# を入れる場合は、

- `PreflightRecoveryPolicy`
- あるいは `_try_preflight_open_order_recovery()`

のような抽出メソッド化が望ましい。

### 6.4 短時間ログ所見

今回の短時間確認では、524# 以上に強い収益改善論点までは断定しない。

ただし、**open order 残存を見にいく recovery path 自体は保守性と停止回避の両面で入れる価値がある**。ここは profit-first とも矛盾しない。

---

## 7. 小さいが効く改善点

### 7.1 真偽値の名前を寄せる

`BalanceChecker.check()` は `scripts/v460/lib/balance_checker.py:125` のとおり「不足時に True」を返す。これ自体は正しいが、`scripts/v460/lib/orchestrator_balance.py` 側で

- `if not await self._check_balance_for_side(...)`

となるため、読み手に優しくない。

ここは

- `_is_balance_insufficient_for_side()`
- `_should_skip_for_balance()`

のように名前を寄せた方がよい。バグ修正ではなくても、境界修正が続く場所では効く。

### 7.2 legacy テストの隔離

`inventory_escape` 系は tests にも散って残っている。後方互換で残すなら、通常機能テストの中に薄く散らすより、`legacy_compat` 系に寄せた方が概念負債を閉じ込めやすい。

---

## 8. 結論

520#-524# は、全体としては良い整流である。特に

- deferred docs の SSOT 化
- balance-forcing からの撤退
- double ceiling の一本化
- preflight 停止を open-order 回復の問題として捉え直した点

は支持できる。

そのうえで、今回のレビューで一番言いたいのは次の 2 点である。

1. **撤去済み概念の legacy surface がまだ散っている**
2. **524# を実装するなら、既存 stale-order cleanup を共有化して入れるべき**

つまり、次にやるべきことは「大きな新機構」ではなく、

- compat 残置の隔離
- cleanup helper の共通化
- naming / dead private method の整理

である。ここを片付けると、以後の fill test 修正がかなり読みやすくなる。
