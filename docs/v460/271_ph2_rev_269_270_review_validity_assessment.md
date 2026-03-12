# 269# / 270# 外部レビュー妥当性評価と対応状況

## 概要

269# (Codex 内部診断) と 270# (Gemini 3.1 Pro 外部レビュー) の 2 つのレビューが
v460 BTC/JPY maker bot の blocking architecture について独立に分析を行った。
本文書は両レビューの主張の妥当性、相互の一致/相違点、そして実装対応状況をまとめる。

---

## §1. 両者の一致する核心的診断（全て妥当）

### 1.1 Per-side Halt が主デッドロック要因

| 項目 | 内容 |
|------|------|
| 269# 根拠 | `fill_test_state.json`: `halted=false`, `side_halted_sell=true`。fill_records 28件中20件が `per_side_dd_halt` |
| 270# 根拠 | 269# の診断を追認。独自検証なし |
| **評価** | **正確**。aggregate halt は発動していない。per-side halt が balance_forced と組み合わさってデッドロックを形成 |

### 1.2 234# Degraded Liquidation が到達不能

| 項目 | 内容 |
|------|------|
| 269# 根拠 | `fill_loop_orchestrator.py:1699` の `is_side_halted` チェックが L2166 の degraded liquidation より前に `continue` |
| 270# 根拠 | 「非常口への廊下を封鎖する警備員」メタファー |
| **評価** | **正確**。223# の安全修正が 234# の縮退清算経路を意図せず遮断。実装上の見落とし |

### 1.3 Per-side Halt の Debt Trap（解除後即再 halt）

| 項目 | 内容 |
|------|------|
| 269# 根拠 | ログ: 13:27 release → 13:45 fill (`pnl=-5.46bps`) → 13:47 再halt（累積 `-37.61bps ≤ -30.0bps`） |
| 270# 根拠 | 「非定常確率過程において再参入を過去の累積経路で評価し続けてはならない」 |
| **評価** | **正確**。release 時に PnL アンカーがリセットされないため、初回負 fill で即再 halt が不可避 |

### 1.4 258#/264#/266# 市場理論が Live で Dormant

| 項目 | 内容 |
|------|------|
| 269# 根拠 | `fill_config.py` のデフォルト全 `False`、YAML 配線なし、fill_records で `macro_trend=null` |
| 270# 根拠 | 「フェラーリのエンジンを開かないトランクに放置」 |
| **評価** | **正確**。AS reservation, GLFT τ, δ*, Kyle λ, Amihud ILLIQ, VPIN continuous — 全 6 機能が無効だった |

---

## §2. 269# のみが指摘した問題（270# が見落とし）

| # | 指摘 | 妥当性 | 重要度 |
|---|------|--------|--------|
| 1 | `balance_forced_halt_recheck` 経路で state save が stale | **正確**。`saved_at=14:02` だがログは `14:59` まで進行。他の skip パス (gate_block) には 223# で追加済みだがこのパスに漏れ | P0 |
| 2 | `sell_dynamic_kill` の probe/force-release が quiescence 思想と矛盾 | **正確**。`max_stale_kill_cycles=10`, `max_force_release_probes=5` がハードコードで YAML 未露出。「No Trade = 正常」(242#) と矛盾 | P1 |
| 3 | guard_fire_counts の system 支配 (market=39 vs system=113) | **正確**。詰まりの主因がシステム側制御であることの実証 | 参考 |
| 4 | Liveness Budget 導入提案 | 設計として妥当だが、現時点では過剰な複雑化リスク。Inventory Escape + reanchor で十分か要検証 | P2 |

**270# はこれらを全てカバーしていない**。セカンドオピニオンとしての独立検証が不足している。

---

## §3. 270# のみの追加主張

| # | 主張 | 妥当性 |
|---|------|--------|
| 1 | Kelly Criterion が balance_forced の「究極の数理的回答」 | **理論的に正しいが時期尚早**。Kelly fraction ≤ 0 で自律 halt は理想的だが、現在の liveness 問題を先に解決しないと Kelly 自体が評価不能。269# の「liveness 修正後に P2 で導入」の方が実践的 |
| 2 | PnL 再アンカーは P0 に格上げすべき | **妥当**。Inventory Escape だけでは再 halt ループを断ち切れない。escape 実行 → 縮退 fill → 負 PnL → 即再 halt の連鎖が起きうる |

---

## §4. 優先度の不一致と判断

| 施策 | 269# | 270# | **採用判断** |
|------|------|------|-------------|
| Inventory Escape Mode | P0 | P0 | **P0** ✅ 実装完了 |
| Per-side PnL reanchor | P1 | P0 | **P0** (270# 側を採用: escape 後の即再 halt 防止が必須) ✅ 実装完了 |
| State save 追加 | P0 | 言及なし | **P0** (269# 側を採用) ✅ 実装完了 |
| Probe/force-release YAML 露出 | P1 | 言及なし | **P1** ✅ 実装完了 |
| 市場理論 YAML 配線 | P2 | P1 | **P0** (根本修正: そもそも有効化されていなかった) ✅ 実装完了 |
| Kelly Criterion live 有効化 | P2 | P1 | **P2** (269# 側: liveness 問題解決後) |
| Liveness Budget | P1 | 言及なし | **P2** (現修正群で効果検証後に再評価) |

---

## §5. 両レビューの品質評価

### 269# (Codex 内部診断)

| 観点 | 評価 |
|------|------|
| 実証性 | ◎ コード行番号・ログタイムスタンプ・state JSON の実測値に基づく一次情報 |
| 網羅性 | ◎ 7 Finding + 6 施策。probe/state save/guard_fire_counts など 270# が見落とした問題をカバー |
| 行動指針 | ○ 優先順位は保守的だが論理的 |
| 弱点 | △ fill_records 件数の不整合 (28 vs 65 — レビュー中のデータ増加と推定されるが断り書きなし) |

### 270# (Gemini 3.1 Pro 外部レビュー)

| 観点 | 評価 |
|------|------|
| 実証性 | △ 269# の診断を「100%正確」と全面追認するのみ。独自のコード・ログ精査なし |
| 網羅性 | × probe/state save/Liveness Budget を見落とし。核心 3 点に集中しすぎ |
| 行動指針 | ○ PnL 再アンカーの P0 格上げは的確な判断 |
| 弱点 | △ LaTeX 記法の崩壊 (`$p$` → `$`)、独立検証の欠如、「100%正確」は過大評価 |

### 総合判定

**269# が一次レビューとして質・量ともに上**。270# は 269# のメタ解釈として付加価値を持つが、
セカンドオピニオンとしての独立性は不十分。ただし **PnL 再アンカーの P0 格上げ判断は
270# の最大の貢献** であり、この点は採用した。

---

## §6. 実装対応状況（本 Issue で完了）

### ✅ 完了

| # | 施策 | 詳細 |
|---|------|------|
| 1 | **市場理論 YAML 配線** | AS reservation (γ/τ/τ_dynamic/δ*), Kyle λ, Amihud ILLIQ, VPIN continuous — 全 6 機能を YAML + parser + `enabled=true` で有効化 |
| 2 | **Inventory Escape Mode** | `fill_loop_orchestrator.py` L1701 の per-side halt チェック内に escape 分岐を新設。balance_forced + sell halt → duty cycle (1/5) で halt 貫通、degraded liquidation パラメータで縮退売却。config: `inventory_escape_enabled`, `inventory_escape_duty_cycle` |
| 3 | **State save 追加** | `balance_forced_halt_recheck` パスに `_STATE_SAVE_INTERVAL_SEC` ベースの state save を追加 |
| 4 | **Per-side PnL reanchor** | `DailyDrawdownState` に `side_reanchor_pnl_buy/sell` フィールド追加。`tick_side_halt()` で release 時に現 PnL を基準点として記録。`update_pnl()` で再 halt 判定を `(current - reanchor)` vs `reanchor_budget_bps` で行う。export/import で永続化 |
| 5 | **Probe/force-release YAML 露出** | `max_stale_kill_cycles`, `max_force_release_probes` を FillTestConfig + YAML に追加。live は `0` (無効) に設定 |
| 6 | **Inventory Escape カウンタ永続化** | `FillTestState` に `inventory_escape_duty_counter` フィールド追加。状態保存・復元に対応 |

### ❌ 未着手（P2 / 今後の検討事項）

| # | 施策 | 理由 |
|---|------|------|
| 1 | Kelly Criterion live 有効化 | Liveness 修正の効果検証後に評価。現時点では市場理論の有効化で十分 |
| 2 | Liveness Budget (段階的緩和順序) | 現修正群 (escape + reanchor) で deadlock 解消が見込まれるため、効果検証後に再評価 |
| 3 | InventoryEscapePolicy / KillGateRescue への責務分離 | 現実装は既存の degraded liquidation パラメータを流用。将来的には独立ポリシーに分離が望ましい |

---

## §7. 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | 市場理論セクション追加、inventory_escape 設定、probe disable、reanchor budget |
| `scripts/v460/lib/fill_config.py` | 市場理論 YAML parser、inventory_escape/probe/reanchor config fields |
| `scripts/v460/lib/fill_loop_orchestrator.py` | Inventory Escape Mode 分岐、state save、escape counter 永続化 |
| `scripts/v460/lib/daily_drawdown_guard.py` | Reanchor PnL fields、tick_side_halt reanchor、update_pnl reanchor 判定、export/import |
| `scripts/v460/run_fill_test.py` | DailyDrawdownGuard + DynamicKillConfig への新パラメータ配線 |
| `scripts/v460/lib/resilience.py` | `inventory_escape_duty_counter` field |
| `tests/unit/v460/test_226_*.py` | Source offset tolerance 拡大 (inventory escape 追加分) |
