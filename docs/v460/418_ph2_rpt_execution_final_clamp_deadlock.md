# 418# Execution Final Clamp + Route-to-Kill Deadlock 修正

**日付**: 2026-03-15  
**前提**: 416# (Codex Review), 417# (Gemini Second Opinion)  
**ステータス**: 実装完了・テスト済み

---

## 概要

416# / 417# のレビューで発見された **2つの構造欠陥** を修正。

### 問題 1: Post-Ceiling Offset Leak (P0 CRITICAL)

**根本原因**: `maker_price.py` の offset ceiling (L1012-1027) は正しく機能しているが、
その後の `fill_cycle_executor.py` 内 **6つの executor 側 multiplier** が ceiling を迂回し、
`effective_offset_ratio` が際限なく拡大していた。

| # | Multiplier | Ceiling 再適用? |
|---|-----------|:-:|
| 1 | EV offset (193#) | ❌ |
| 2 | Velocity offset (195#) | ❌ |
| 3 | Trending sell offset (196#) | ❌ |
| 4 | Toxicity offset (240#) | ❌ |
| 5 | VG sell supplement (202#) | ❌ |
| 6 | Alert mode offset (215#) | ❌ |

**実データ証拠** (416# §1.1):
- 3/11: `offset_stages.final = 0.300` → `effective_offset_used = 1.305` (×4.35倍)
- 3/14: `offset_stages.final = 0.498` → `effective_offset_used = 0.905` (×1.82倍)

**影響**: 405# が offset ceiling を 0.30→0.50 に引き上げたことで、post-ceiling multiplier と
組み合わさり offset が 1.0 超に膨張。sell AS rate 倍増 (22%→37%)、PnL 逆転 (+1.17→-1.34 bps)。

### 問題 2: Route-to-Kill Deadlock (P0 HIGH)

**根本原因**: buy 残高不足 → sell に切替 → sell が `sell_dynamic_kill` で gate-blocked →
cycle skip → ループ再開 → 再び buy 残高不足 → sell 切替 → gate-blocked → ...
の高速デッドスピラル。

**実データ証拠**: 3/12: `sell_dynamic_kill` = 266回 (3/11: 7回から38倍に急増)

---

## 修正内容

### Fix 1: Execution Final Clamp

**場所**: `fill_cycle_executor.py` L738付近 (全 multiplier 適用後、`t_submit` 直前)

**ロジック**:
1. サイド別 ceiling を解決 (`offset_ceiling_ratio_buy` / `_sell` / 共通)
2. `effective_offset_ratio > ceiling` の場合:
   a. **Hard skip** (optional): `effective_offset_ratio > ceiling × hard_skip_mult` なら
      cycle skip (CR.FINAL_CLAMP_HARD_SKIP) — 市場が極端すぎる
   b. **Normal clamp**: price を再計算し、offset を ceiling に切り詰め
3. 発火時は `execution_pre_clamp_offset` (pre-clamp 値) を FillRecord に記録

**設定フィールド** (`fill_config.py`):
```yaml
execution_final_clamp_enabled: true      # デフォルト有効 (安全策)
execution_final_clamp_hard_skip_mult: 0.0  # >0 で hard skip 有効 (例: 2.0)
```

### Fix 2: Route-to-Kill Deadlock 防止

**場所**: `orchestrator_balance.py` `_resolve_balance_and_preflight()` L73付近

**ロジック**: 残高不足で反対サイドに切替する前に `_is_side_killed()` を事前チェック。
kill-gated なら切替せず、`CR.ROUTE_TO_KILL_DEADLOCK` としてスキップ。
これにより gate 段階での高速ループが発生しない。

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_cycle_executor.py` | Final Clamp ロジック挿入 (L738), `execution_pre_clamp_offset` 引き回し |
| `scripts/v460/lib/fill_config.py` | `execution_final_clamp_enabled`, `execution_final_clamp_hard_skip_mult` 追加 |
| `scripts/v460/lib/cancel_reasons.py` | `FINAL_CLAMP_HARD_SKIP`, `ROUTE_TO_KILL_DEADLOCK` 追加 |
| `scripts/v460/lib/orchestrator_balance.py` | Route-to-Kill 事前チェック追加 |
| `scripts/v460/lib/fill_record_builder.py` | `execution_pre_clamp_offset` パラメータ追加 |
| `ztb/metrics/fill_quality.py` | `FillRecord.execution_pre_clamp_offset` フィールド追加 |
| `tests/unit/v460/test_418_final_clamp_deadlock.py` | 25テスト (全パス) |

---

## 416# / 417# レビュー採択・非採択マトリクス

| # | 提案 | 優先度 | 判定 | 備考 |
|---|------|--------|------|------|
| 416# §1.1 | post-ceiling multiplier leak | CRITICAL | ✅ 採択 | Final Clamp で修正 |
| 416# §2.1 | `execution_final_offset` 記録 | HIGH | ✅ 採択 | `execution_pre_clamp_offset` として実装 |
| 416# §3 | sell_dynamic_kill は balance noise でない | HIGH | ✅ 採択 | Route-to-Kill Deadlock 修正 |
| 416# §4.1 | SkipGate 側閾値は既に存在 | HIGH | ✅ 確認済み | 追加改修不要 |
| 416# §4.2 | gate_path / resolved_side_reason 記録 | MEDIUM | 🔄 後続 | 419# 以降で検討 |
| 416# §5 | git_sha 混在は hot reload 起因 | MEDIUM | ✅ 確認済み | SHA 別再集計は後続 |
| 416# §6 | 3/12 TypeError 既修正 | MEDIUM | ✅ 確認済み | 対処不要 |
| 416# §7 | buy も赤字 | MEDIUM | ✅ 確認済み | Final Clamp で buy ceiling (0.20) も適用 |
| 416# §8 | VG は continuous risk premium | MEDIUM | ✅ 確認済み | `vg_boost_factor` 分布の分析は後続 |
| 417# Action 1 | "The Final Clamp" | P0 | ✅ 採択 | 本実装 |
| 417# Action 2 | Orchestrator deadlock escape | P1 | ✅ 採択 | Route-to-Kill として実装 |
| 417# Action 3 | SkipGate pnl threshold 再調査 | P1 | 🔄 後続 | SHA 別再集計後に判断 |
| 417# Action 4 | SHA-based 再評価 | P2 | 🔄 後続 | 分析スクリプト必要 |
| 417# Self-review | Hard skip combo | P0 | ✅ 採択 | `execution_final_clamp_hard_skip_mult` で実装 |

---

## 追加発見した盲点・改善候補

### 盲点 1: `_apply_offset_multiplier` に ceiling awareness がない
`pre_order_adjustments.py` の `_apply_offset_multiplier()` は純粋な乗算のみで
ceiling/floor チェックを一切持たない。これは設計として正しい（single responsibility）が、
呼び出し側で ceiling を必ず再適用する責務が fill_cycle_executor に集約される。
→ Final Clamp で正しくカバー済み。

### 盲点 2: degraded_liquidation だけが `max_offset_ratio` clamp を持つ
6 multiplier の中で唯一 degraded_liquidation (234#) だけが `min(... max_offset_ratio)`
でクランプしている。しかしこの clamp は `max_offset_ratio = 0.30` であり、
`offset_ceiling_ratio_sell = 0.50` よりも厳しい制約。
つまり degraded_liquidation 時は正しく動作するが、他の multiplier は無防備だった。
→ Final Clamp が全 multiplier をカバーするため解消。

### 盲点 3: sidecar_offset_bps は ratio ではなく price 直接操作
sidecar (372#) は `order_price` を直接変動させるため `effective_offset_ratio` を
更新しない。Final Clamp は ratio ベースなので sidecar の影響は別軸。
→ sidecar による price 変動は ceiling の趣旨（offset ratio 制限）とは独立。
   ただし sidecar が大きすぎる場合の別途ガードは 419# 以降で検討。

### 改善候補 1: YAML config に `execution_final_clamp_hard_skip_mult` 追加
現在デフォルト 0.0 (hard skip 無効)。実運用での推奨値決定後に有効化する。
推奨候補: `2.5` (ceiling の 2.5 倍超で hard skip)。

---

## テスト結果

```
tests/unit/v460/test_418_final_clamp_deadlock.py: 25 passed
regression (test_405, test_346, test_306): 105 passed
```
