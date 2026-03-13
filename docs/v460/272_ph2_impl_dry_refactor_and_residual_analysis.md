# 272# DRY リファクタ + 269# 残指摘の掘り下げ検証

> **日付**: 2026-03-04  
> **前提**: 269# (Codex), 270# (Gemini), 271# (妥当性評価 + 実装), 272# (本文書)  
> **対象コミット**: `2726a8952` (refactor: 272# DRY リファクタ)

---

## §1. DRY リファクタ — 抽出したヘルパーメソッド

269#/270# の実装 (271#) で `fill_loop_orchestrator.py` に追加したコードが
既存ロジックとの重複を生んでいた。272# はこれを解消する。

### 1.1 抽出した 4 メソッド

| メソッド | 型 | 重複箇所 | 削減効果 |
|----------|-----|---------|----------|
| `_tick_toxic_veto(context: str)` | instance | L1518 (both-blocked), L1741 (inventory_escape), L1763 (halt_block) → 3→1 | 各 5-7 行 × 3 → 呼出 1 行 × 3 |
| `_maybe_skip_state_save(st, context: str)` | instance | L1782 (halt_block save), L2156 (gate_block save) → 2→1 | 各 7 行 × 2 → 呼出 1 行 × 2 |
| `_feed_mcb_sad()` | instance | halt loop 内の MCB/SAD フィード (8 行) → 1→1 | 8 行 → 1 行 |
| `_opposite_side(side: str) -> str` | staticmethod | 5 箇所のインライン三項演算子 → 5→1 | 可読性向上 |

### 1.2 検討したが見送ったパターン

| パターン | 出現数 | 見送り理由 |
|---------|--------|-----------|
| Skip-Record-Flush-Continue | 18 箇所 | 各サイトの surrounding logic が異なる (`continue` はメソッド経由では制御不能)。パラメータが 8+ となり抽象化コスト > 利得 |
| `_is_sell_killed` / `_is_buy_killed` 対称ペア | 2 ペア (4 メソッド) | side パラメータ化は可能だが、全テストへの波及が大きい。P2 候補 |
| MCB check+update / SAD check+update | 各 1 箇所 | halt loop の feed-only と通常の check+update は分岐ロジックが異なる。feed のみ抽出済み |

### 1.3 テスト影響

- `test_226_loss_boost_decay_inv_skew_state.py`:
  - `test_orchestrator_source_has_halt_mcb_update`: `self._mcb.update` → `self._feed_mcb_sad()` のソース検査に更新
  - `test_orchestrator_source_has_halt_sad_update`: `_feed_mcb_sad` ヘルパー内に `self._sad.update` が含まれることを検証に変更
- **全 3688 テスト PASS** (exit code 1 は coverage threshold のみ)

---

## §2. 269# 指摘の検証状況 — 全項目トレーサビリティ

### 2.1 確認項目 (§1: 確認できた改善)

| 269# §項番 | 指摘内容 | 対応状況 |
|------------|---------|---------|
| §1.1 | 268# JST 日次リセットが実ログで確認 | ✅ 確認済み (271# §1) |
| §1.2 | guard_fire_counts / guard_category_totals の観測改善 | ✅ 確認済み (271# §1) |
| §1.3 | 249# dual_kill_quiescence が一部機能 | ✅ 確認済み (271# §1) |

### 2.2 主要 Findings (§2)

| 269# §項番 | 指摘内容 | 対応 | 検証状況 |
|------------|---------|------|---------|
| §2.1 [CRITICAL] | per-side halt + balance_forced = 主デッドロック | ✅ **Inventory Escape Mode 実装** (271#) | duty cycle 1/5 で halt 貫通。degraded liquidation パラメータで縮退売却 |
| §2.2 [CRITICAL] | 234# degraded_liquidation が到達不能 | ✅ **修正済み** (271#) | Inventory Escape が L1762 で halt を貫通し、degraded_liquidation フラグで `run_single_cycle` へ到達 |
| §2.3 [HIGH] | balance_forced_halt_recheck で state save stale | ✅ **修正済み** (271#) | `_maybe_skip_state_save` ヘルパー経由で時間ベース保存 |
| §2.4 [HIGH] | per-side halt の Debt Trap (即再 halt) | ✅ **修正済み** (271#) | PnL reanchor 方式: release 時に基準点記録、再 halt 判定は `(current - anchor)` vs `reanchor_budget_bps` |
| §2.5 [HIGH] | probe / force-release が quiescence と矛盾 | ✅ **修正済み** (271#) | YAML 露出 + live `max_stale_kill_cycles=0`, `max_force_release_probes=0` |
| §2.6 [MEDIUM] | 258#/264#/266# 市場理論が dormant | ✅ **修正済み** (271#) | AS reservation, Kyle λ, Amihud ILLIQ, VPIN continuous — 全 6 機能を YAML 有効化 |
| §2.7 [MEDIUM] | blocking ロジックの巨大横断責務 | 🔄 **一部改善** (272#) | ヘルパー抽出で重複削減。完全な Policy 分離は P2 |

### 2.3 根本原因 (§3)

| 269# §項番 | 指摘内容 | 対応状況 | 残課題 |
|------------|---------|---------|--------|
| §3.1 | alpha 取りと inventory 解消の経路混在 | ✅ **Inventory Escape Mode で分離** | 現状 sell のみ。buy 側ミラーは §3 で議論 |
| §3.2 | safety が各層で局所最適 → 全体 No Trade | 🔄 **部分対応** | Inventory Escape が上位で介入するが、Liveness Budget は未実装 |
| §3.3 | 評価単位が too short (30s/60s/120s 中心) | ❌ **未着手** | 5分/15分ホライゾンの上位意思決定昇格は P2 |

### 2.4 推奨施策 (§4)

| 269# §項番 | 施策 | 実装状況 | 備考 |
|------------|------|---------|------|
| §4.1 P0 | Inventory Escape Mode | ✅ **完了** | balance_forced + per-side halt + sell → duty 1/5 halt 貫通 |
| §4.2 P0 | balance_forced_halt_recheck に state save | ✅ **完了** | `_maybe_skip_state_save` で一元化 |
| §4.3 P1 | per-side halt → release 後アンカー方式 | ✅ **完了** (P0 格上げ) | `side_reanchor_pnl_*` + `per_side_reanchor_budget_bps=-15.0` |
| §4.4 P1 | probe/force-release YAML 殺し | ✅ **完了** | live: `0`, 検証時は明示 opt-in |
| §4.5 P1 | Liveness Budget 導入 | ❌ **未実装** (P2 据え置き) | inventory escape duty cycle が部分的に同等機能を提供 |
| §4.6 P2 | 市場理論 → inventory risk 用途に転用 | 🔄 **有効化のみ** | 現状は alpha パスの offset/lot 修飾。inventory escape 専用パラメータへの転用は将来 |

### 2.5 追加で見落としやすい点 (§5)

| 269# §項番 | 指摘内容 | 対応状況 | 分析 |
|------------|---------|---------|------|
| §5.1 | degraded_liquidation の責務曖昧 | ❌ **未対応** | Inventory Escape と KillGateRescue の責務分離は未実施。現状は `_inventory_escape` フラグで分離を示唆するのみ |
| §5.2 | macro_trend / macro_aligned が全件 null | ⚠️ **設計通りだが要判断** | `enable_macro_regime: false` が YAML で意図的。269# 指摘は「未統合のサイン」だが、Phase E 相当の判断待ち (下記 §3.4 で詳述) |
| §5.3 | policy 抽出不足 (BlockingPolicy等) | ❌ **未対応** | P2 候補。orchestrator 2603 行の構造的問題は残存 |

### 2.6 潜在的 Deadlock (§6)

| 269# §項番 | パターン | 検証状況 |
|------------|---------|---------|
| §6.1 | buy 不足 + sell per-side halt | ✅ **解決** (Inventory Escape で sell 側 halt 貫通) |
| §6.2.1 | sell 不足 + buy per-side halt (ミラー) | ⚠️ **未対応** (下記 §3.1 で分析) |
| §6.2.2 | aggregate halt + dual kill quiescence + balance_forced | 🔄 **246# cooldown release で部分対応**。完全検証は困難 |
| §6.2.3 | state stale + watchdog 判定 | ✅ **state save 追加で軽減** |

---

## §3. 掘り下げ検証 — 残存リスクと未対応事項

### 3.1 Inventory Escape の sell-only 問題

**現状**: `_ie_enabled and next_side == "sell"` のみ Escape が発動。

**ミラーケース**: BTC 残高十分だが JPY 不足 → sell は可能、buy 不可 → balance_forced で buy に切替 → buy per-side halt → **デッドロック**

**分析**:
- 現実的にこのパターンは発生しにくい。JPY 不足 (= BTC を大量保有) の状態で sell 不可になるのは、sell 側の per-side halt が先行している場合のみ
- 実運用では「JPY 不足 → sell に balance_forced」が圧倒的多数 (BTC を売って JPY 回収)
- buy 側の inventory escape は「JPY 在庫を解消するために BTC を買う」意味になり、これは在庫解消ではなくポジション構築

**判定**: 構造的非対称性は意図的。**buy 側 escape は現時点では不要**。ただし将来的に JPY→BTC のポジション逆転が起きた場合は要対応。

### 3.2 Liveness Budget の代替としての Duty Cycle 評価

269# §4.5 の Liveness Budget (段階的緩和順序) は未実装だが、以下が代替として機能:

1. **Inventory Escape Duty Cycle** (1/5): halt 貫通頻度の制限
2. **Degraded Liquidation Duty Cycle**: kill gate rescue の頻度制限
3. **Recovery Lot Scale** (224#): halt 解除後の慎重な再参入

ただし、269# が指摘する「`minutes_since_last_fill` / `inventory_excess` / `holding_risk_score` による統一的な緩和順序」は持っていない。

**判定**: 現修正群で deadlock 解消効果を 1 週間程度検証し、liveness が改善しなければ Liveness Budget を P1 に格上げ。

### 3.3 Kelly Criterion の接続状況

**コード配線は完了している**:
- `fill_config.py`: `enable_kelly_sizing: bool = False`
- `adaptation_engine.py`: `yaml_cfg.get("kelly", {})` で YAML → Kelly fraction 計算
- `lot_sizer.py`: `KellyEstimate`, `kelly_fraction`, `kelly_recommended_lot` 完備

**YAML 未設定**: `configs/v460/fill_test.yaml` に `kelly:` セクションがない。

**判定**: liveness 修正の効果検証後に YAML に `kelly:` セクションを追加して有効化。269# §4.6 のとおり、Kelly ≤ 0 で自律 halt は理想的だが、現時点では per-side halt + reanchor が同等の安全弁として機能。

### 3.4 MacroRegimeDetector (macro_trend / macro_aligned)

**状況**: `configs/v460/fill_test.yaml` の `regime.macro.enabled: false` で意図的に無効化。

269# §5.2 の指摘:
> `fill_records の macro_trend = null` は「未使用フィールド」ではなく、未統合のサイン。大きな相場感を使うための導線が live で閉じている。

**分析**:
- MacroRegimeDetector のコードは完成済み (`scripts/v460/lib/macro_regime.py`)
- `compose_regimes()` で micro/macro の合成も実装済み
- `conflict_action=log` で観測専用で起動可能
- 有効化すれば `fill_records` の `macro_trend` / `macro_aligned` が populate される

**判定**: `regime.macro.enabled: true` + `conflict_action: log` で観測モード有効化を推奨。意思決定への適用 (Phase E) は 1 週間の観測データ蓄積後。

### 3.5 Policy 分離の優先度

269# §5.3 の指摘する 3 つの Policy:

| Policy | 概要 | 現時点の実装場所 | 分離の緊急度 |
|--------|------|----------------|-------------|
| `BlockingPolicy` | 複数 guard の blocking 判定統合 | `fill_loop_orchestrator.py` inline (18 箇所の skip-record パターン) | 中 (可読性問題) |
| `InventoryEscapePolicy` | 在庫解消専用モードのパラメータ/判定 | `fill_loop_orchestrator.py` L1766-1796 | 低 (現状 30 行で自己完結) |
| `LivenessSupervisor` | liveness 統括 (緩和順序管理) | 未実装 | 低 (duty cycle で代替中) |

**判定**: 272# の DRY 抽出で重複は解消済み。Policy 分離は orchestrator が 3000 行を超える前の予防措置として P2 で計画。

---

## §4. 推奨アクション (残課題)

### 即時 (P1)

| # | 施策 | 詳細 | 効果 |
|---|------|------|------|
| 1 | MacroRegime 観測モード有効化 | `regime.macro.enabled: true`, `conflict_action: log` | macro_trend / macro_aligned データ蓄積。Phase E 判断材料 |
| 2 | Liveness Budget 効果検証 | Inventory Escape + reanchor の効果を 1 週間モニタリング | P1→P2 or 実装判断 |

### 中期 (P2)

| # | 施策 | 詳細 |
|---|------|------|
| 3 | Kelly Criterion YAML 有効化 | `kelly:` セクション追加、kelly_enabled=true |
| 4 | _is_sell_killed / _is_buy_killed のパラメータ化 | side 引数で統合、テスト更新 |
| 5 | BlockingPolicy / InventoryEscapePolicy 分離 | orchestrator 肥大化防止 |
| 6 | 評価ホライゾン拡大 (5min/15min) | §3.3 への対応。上位意思決定層の新設 |

### 不要 (対応済み or 設計通り)

| # | 項目 | 理由 |
|---|------|------|
| 7 | buy 側 inventory escape | 構造的非対称性は意図的 (§3.1) |
| 8 | day_reset と reanchor の相互作用 | `DailyDrawdownState()` 新規生成で自動リセット (検証済み) |
| 9 | inventory_escape の get_metrics 露出 | `guard_fire_counts` + `state save` で間接的に観測可能 |

---

## §5. 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | `_tick_toxic_veto`, `_maybe_skip_state_save`, `_feed_mcb_sad`, `_opposite_side` ヘルパー抽出 + 10 箇所の呼出置換 |
| `tests/unit/v460/test_226_loss_boost_decay_inv_skew_state.py` | MCB/SAD ソース検査を `_feed_mcb_sad` 対応に更新 |
| `docs/v460/index.md` | 272# エントリ追加 |
| `docs/v460/271_ph2_rev_269_270_review_validity_assessment.md` | リネーム (`269_270_` → `271_`) |
