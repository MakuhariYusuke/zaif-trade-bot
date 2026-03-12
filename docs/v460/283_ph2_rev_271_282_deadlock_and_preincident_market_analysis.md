# 283# 271〜282レビュー: デッドロック再検証 + 発生前ログの市場理論分析

| 項目 | 内容 |
|---|---|
| 日付 | 2026-03-05 |
| 対象 | 271#〜282# ドキュメント / 実装 / 実ログ |
| 主眼 | 1) デッドロック事象の見落とし確認 2) デッドロック前ログの市場理論的再評価 3) 追加改善提案 |

---

## 1. 結論（先に要点）

- **282# の主修正（`untick_side_halt` 呼出し除去 + IE双方向化）は妥当**。実際に 2026-03-05 21:45 JST 以降で約定再開を確認。
- ただし、**「止まらなくなった」だけで「儲かる」には未到達**。post-fix 2 run 合算で `fills=14`, `p30 mean=-1.721bps`, `winrate=35.71%`。
- デッドロック前（同日 09:00〜13:17）の本質は、**buy 側の情報優位欠如 + kill主導の片側化 + 在庫枯渇からの forced buy 劣化**。
- 追加で重要な見落としとして、**run_id 混在（`1772714694_d20307ad` と `1772714739_b393955d` の時系列重複）**が確認され、単一実行保証/観測の信頼性にリスクあり。

---

## 2. 271〜282の妥当性レビュー

### 2.1 方向性評価

| 区間 | 評価 | 根拠 |
|---|---|---|
| 271#〜277# | 概ね妥当 | 269/270の論点を実装へ落とし込めている（IE, reanchor, state save, DRY） |
| 278#〜281# | 品質リスク露呈 | config化で NameError (`_HALT_PERSIST_INTERVAL`) を混入し、halt経路で即死 |
| 282# | 妥当（P0修正として有効） | deadlock 脱出に直結する2点を修正し、実約定再開を確認 |

### 2.2 282# 事実整合（ログ突合）

- deadlock run: `run_id=1772636854_01304838`, `git_sha=91f050a76aa8`
- 同 run 集計（`fill_records_20260305.jsonl`）:
  - `records=353`, `fills=25`
  - `cancel_reason=per_side_dd_halt` は **251件**
  - per_side_dd_halt は **連続251件（run内 103件目〜353件目）**
  - 時刻: `2026-03-05 13:17:02 JST` 〜 `2026-03-05 21:41:29 JST`
- 実装確認:
  - `scripts/v460/lib/fill_loop_orchestrator.py:1601` 付近（both_halt）で `untick_side_halt()` 呼出し除去
  - `scripts/v460/lib/fill_loop_orchestrator.py:1859` 付近（balance_forced_halt_block）で同除去
  - `scripts/v460/lib/fill_loop_orchestrator.py:1828` 付近で IE 条件を side 非依存化

---

## 3. デッドロック前ログの市場理論分析（2026-03-05 09:00〜13:17 JST）

### 3.1 実測サマリ

- 対象 run: `1772636854_01304838`
- 09:00〜13:17（per_side_halt開始直前）
  - `records=102`, `fills=25`
  - `p30 mean=-0.933bps`, `winrate=36.0%`
  - buy: `12 fills`, `mean=-1.001bps`, `winrate=25.0%`
  - sell: `13 fills`, `mean=-0.870bps`, `winrate=46.15%`
- cancel 主因（同区間）
  - `buy_dynamic_kill=52`（最大 23 連続、09:06:46〜09:47:01 JST）
  - `skip_gate=8`
  - `sell_guard_reject=4`

### 3.2 市場理論的解釈

1. **レンジ相場での maker 逆選択負け**
- 全 fill が `regime=ranging` なのに期待値が負。
- これは「spread 捕捉 < adverse selection + タイミング劣化」を示唆。

2. **buy 側の情報劣位が顕著**
- pre-deadlock AS 比率: 全体 `32.0%`、buy は `41.67%`、sell は `23.08%`。
- buy が短期ドリフトに対し不利な位置で約定している。

3. **kill 主導の片側化 → 在庫枯渇 → forced buy 品質劣化**
- `buy_dynamic_kill` 長時間作動で sell 側偏重。
- BTC 在庫を削り、`balance_forced` buy が増える。
- buy fill のうち `balance_forced=true` は `7/12`（58.33%）。
- `balance_forced` buy 平均 `-1.345bps`、通常 buy 平均 `-0.519bps`。

要するに、デッドロックは「事故」ではなく、
**収益性が弱い状態遷移（kill→片側化→在庫歪み→forced劣化）の終端で顕在化**している。

---

## 4. 見落とし・論理補強ポイント

### 4.1 CRITICAL

1. **run_id 重複稼働疑い（単一実行保証の毀損）**
- `1772714694_d20307ad` の記録期間: 21:45:12〜22:37:09 JST
- `1772714739_b393955d` の記録期間: 21:45:55〜22:42:38 JST
- 時間帯が重複しており、`fill_records_20260305.jsonl` に両 run が混在。
- `fill_test_events.jsonl` 末尾では `b393` の start はあるが stop が見えず、lock_conflict観測が不完全。

**影響**: deadlock脱出評価や post-fix PnL 解釈の信頼性を落とす。

### 4.2 HIGH

2. **282# 文書内の時刻・件数境界にズレ**
- 実データでは per_side_halt 連続区間は 13:17:02 開始。
- 13:08:53 の sell fill（-7.47bps）が存在し、「13:08まで25 fills」は境界次第で 24/25 が変わる。

3. **再発条件の設定ガードが弱い**
- `per_side_dd_halt_cycles=0` は「日替わりまで封鎖」挙動を許す設計。
- `inventory_escape_enabled=false` と組み合わさると長時間停止リスクが再燃しうる。
- `scripts/v460/lib/fill_config.py` に相互制約チェックが不足。

### 4.3 MEDIUM

4. **post-fix でも system guard 優位（自家中毒）**
- `fill_test_state.json` で `guard_category_totals`: `market=209`, `system=766`, `recovery=44`。
- 依然として市場要因より内部制御のブロックが支配。

5. **status_unknown_fast の残存**
- post-fix run で `status_unknown_fast=1` を確認。
- 単発だが、約定/取消の整合不確実性は deadlock 以前に PnL 評価を歪める。

---

## 5. 改善提案（優先度順）

### P0（即時）

1. **単一実行保証の監査を最優先**
- `fill_records` へ `pid` を記録。
- 「同一時刻帯に複数 run_id が出現したら CRITICAL alert」追加。
- `fill_test_events` に start/stop 対を強制（異常終了時も finally で stop 書き込み）。

2. **設定相互制約を追加**
- `per_side_dd_halt_cycles==0` かつ `inventory_escape_enabled==false` を禁止。
- `per_side_dd_halt_cycles` の最小値を運用既定で `>=1` に固定。

3. **deadlock再現テストを run_id/lock 観点まで拡張**
- 282# のロジックテストに加え、「二重起動時に2本目が絶対に記録を書かない」統合テストを追加。

### P1（収益改善の本線）

4. **buy_dynamic_kill の在庫連動緩和**
- 在庫が閾値以下なら buy kill 閾値を段階緩和（完全封鎖ではなく lot/offset 制御へ）。
- 「kill で守る」から「サイズを落として在庫維持」に転換。

5. **forced buy を独立評価して品質管理**
- 通常 buy と `balance_forced` buy の KPI を分離。
- `forced_buy_p30_mean` が悪化時は、sell 側の放出を先に抑制（在庫下限ベース）。

6. **buy 側 AS 対策の明示導入**
- buy 時のみ microprice/短期flow に応じて offset を追加上積み。
- `AS_ratio_buy` を閉ループで監視し、しきい値超過時に自動で quote を受け身化。

### P2（安定運用）

7. **評価窓を二層化**
- 現行30秒指標に加え、5分窓の side別期待値を導入。
- kill/halt 判定を「瞬間防御」と「在庫維持」を分離して最適化。

---

## 6. まとめ

- 271〜282の流れは、**deadlock解除の技術的方向性としては正しい**。
- ただし、現段階は「停止回避の確立」であり、**高収益化フェーズには未到達**。
- 次の勝ち筋は、
  - **(A) 実行基盤の信頼性（単一実行保証）を先に固める**
  - **(B) buy側逆選択と forced buy 劣化を直接叩く**
  の2本立て。

この2点を先に片付けない限り、以降のパラメータ調整は再び「事後的最適化」に戻る可能性が高い。
