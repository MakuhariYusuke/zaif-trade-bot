# 223# 222# レビュー対応 — CRITICAL バグ修正 + メトリクス強化 + state save 改善

> **日付**: 2026-03-03
> **対象**: 222# (Codex + Gemini レビュー) の CRITICAL / HIGH 指摘への実装対応
> **変更ファイル**:
>   - `scripts/v460/lib/fill_loop_orchestrator.py` (+62 行)
>   - `scripts/v460/lib/cycle_gate_aggregator.py` (+2 行)
>   - `ztb/risk/sell_dynamic_kill.py` (+10 行)
> **テスト**: 2971 passed / 0 failed

---

## 0. 総括

222# で指摘された 5 件の findings に対し、以下の優先度で修正を実施した。

| 優先度 | 222# 指摘 | 対応 | 状態 |
|--------|-----------|------|------|
| **P0** | 1.1 CRITICAL: per-side halt が balance_forced で破られる | **修正済** — balance_forced 後に halt 再チェック | ✅ |
| **P0** | Gemini §6.5-2: balance_forced + side halt 順序修復 | 同上 (1.1 と同じ) | ✅ |
| **P1** | 1.5 MEDIUM: guard_fire_counts にデッドロック関連イベント未記録 | **修正済** — 7 種類の新メトリクス追加 | ✅ |
| **P1** | 1.4 MEDIUM: skip 連続時に state save が stale | **修正済** — 時間ベース state save 追加 | ✅ |
| **P1** | Gemini §6.5-1: DUAL KILL bypass 即時廃止 | **評価完了** — 段階的廃止を推奨 (§4 参照) | 📋 |

---

## 1. [P0] balance_forced + per-side halt 再チェック (1.1 CRITICAL 修正)

### 問題

`per-side halt` で sell が封鎖 → buy に切り替え → buy 残高不足 → `balance_forced` で sell に戻される → **halt 中の sell がそのまま実行される**

**実ログ証拠** (2026-03-02):
```
16:40:56  sell PnL -30.40bps <= -30.0bps — sell 封鎖 (cycles=15)
16:44:31  Per-side halt: sell blocked, switching to buy
16:44:32  buy insufficient, switching to sell immediately
16:44:33  balance_forced but one_sided_balance — proceeding with sell
16:44:34  Placed sell limit (HALT BREACHED)
16:46:19  Cycle 5542 result: filled=True, pnl=-1.15bps
```

### 修正

`fill_loop_orchestrator.py` の `balance_forced` で `next_side` が反転した直後に、`is_side_halted(next_side)` を再チェック。True ならサイクルを skip する。

```python
# 223# P0: balance_forced 後に per-side halt を再チェック
if self._daily_drawdown_guard.is_side_halted(next_side):
    logger.warning(
        f"[223#] balance_forced → {next_side} is per-side halted — "
        f"refusing to bypass halt (safety > liveness)"
    )
    self._inc_guard_fire("balance_forced_halt_block")
    batch.append(self._make_loop_skip_record(...))
    continue
```

**設計原則**: Safety > Liveness — halt は最優先のリスク制御であり、balance convenience のために破ってはならない。

---

## 2. [P1] guard_fire_counts メトリクス強化 (1.5 修正)

### 問題

222# §1.5: デッドロック関連の guard 発火 (`dual_kill_bypass`, `per_side_halt_switch`, `dynamic_kill_probe` 等) が `guard_fire_counts` に記録されていなかった。

### 追加したメトリクス (7 種)

| キー | 発火箇所 | 意味 |
|------|----------|------|
| `per_side_halt_switch` | orchestrator L909 | 片側 halt で反対に切り替え |
| `per_side_dd_both_halt` | orchestrator L897 | 両側 halt で全停止 |
| `balance_forced_halt_block` | orchestrator L1093 | balance_forced が halt で阻止された (223# 新設) |
| `dual_kill_bypass` | orchestrator L1405 | DUAL KILL bypass 発動 |
| `dynamic_kill_probe_sell` | orchestrator L77 | sell kill probe 発動 |
| `dynamic_kill_probe_buy` | orchestrator L108 | buy kill probe 発動 |
| `dynamic_kill_force_release_sell` | orchestrator L79 | sell force release 発動 |
| `dynamic_kill_force_release_buy` | orchestrator L110 | buy force release 発動 |

### 実装

- `CycleGateResult` に `dual_kill_bypassed: bool` フィールド追加
- `DynamicKillTelemetry` に `probe_fired: bool`, `force_release_fired: bool` フィールド追加
- orchestrator の各判定箇所でフラグを読み取り `_inc_guard_fire()` 呼び出し

---

## 3. [P1] skip-time lightweight state save (1.4 修正)

### 問題

222# §1.4: `_state_persistence.save()` の呼び出しが 3 箇所のみ:
1. DD halt 中 (10 iter 毎)
2. `progress_log_interval` サイクル毎 (通常 50 サイクル)
3. 最終保存

gate_block の `continue` パス（skip 連続）では save が発生しない → 5 時間以上 stale になるケースが発生。

### 修正

時間ベースの state save チェックを gate_block continue パスに追加。

```python
# 223# skip-time lightweight state save
_now_mono = time.monotonic()
if _now_mono - self._last_state_save_time >= self._STATE_SAVE_INTERVAL_SEC:
    self._state_persistence.save(self._build_state_snapshot(...))
    self._last_state_save_time = _now_mono
```

- **デフォルト間隔**: 300 秒 (5 分)
- **クラス変数** `_STATE_SAVE_INTERVAL_SEC` で設定可能
- ループ開始時に `_last_state_save_time = time.monotonic()` で初期化
- 既存の 3 save 箇所にもタイムスタンプ更新を追加

---

## 4. DUAL KILL bypass ポリシー評価

### Codex の立場 (keep with metrics)

- 1.2 では「1.1 を修正すれば bypass の直接的な危険は軽減される」と評価
- §1.5 ではメトリクス追加を推奨 → 観測可能化が先決

### Gemini の立場 (即時廃止)

- §6.2: Kelly 基準から EV 負 × 不確実性最大時のベットサイズ = 0
- 「両側 kill = レジーム崩壊」であり、bypass は統計的自殺
- §6.5-1: P0 として即時削除要請

### 評価

**Gemini の理論的主張は正しい**。両側 kill は市場の前提崩壊シグナルであり、強制通過は Kelly 基準に反する。

しかし **実装上の即時削除は不適切** と判断する。理由:

1. **219# probe/force_release が既に data-driven な解除機構を提供している**
   - probe: stale data 蓄積時に 1 取引許可して PnL を観測
   - force_release: 5 回連続 probe で改善なしなら kill 強制解除
   - DUAL KILL bypass はこれらと機能的に重複する「粗い解除」

2. **223# で balance_forced + halt 再チェックが実装された**
   - DUAL KILL bypass が gate を通過しても、per-side halt で最終的にブロックされる
   - 最も危険な経路 (halt 破り) は既に封鎖済み

3. **メトリクスが未収集のため、削除判断のエビデンスがない**
   - 本 223# で `dual_kill_bypass` カウンタを新設
   - 数日分のデータ収集後に、bypass 発動 + 後続 PnL の相関を分析すべき

### 結論: 段階的廃止ロードマップ

| フェーズ | 内容 | 時期 |
|----------|------|------|
| 223# (本対応) | メトリクス収集開始 + halt 再チェックで致命経路封鎖 | 即時 |
| 224# (次回) | 1 週間分のメトリクス分析 → bypass 発動時の PnL 相関評価 | +1 週間 |
| 225# (予定) | エビデンスに基づき bypass 廃止 or probe への統合判断 | +2 週間 |

**probe/force_release 機構が DUAL KILL bypass の上位互換であることを確認次第、bypass を廃止する。**

---

## 5. 222# 各指摘への個別回答

### 1.1 CRITICAL: per-side halt bypass → **§1 で修正済**
### 1.2 HIGH: DUAL KILL bypass の安全性 → **§4 で評価、段階的廃止ロードマップ策定**
### 1.3 HIGH: SHA 混在評価
- 運用上の課題であり、コード変更ではなく分析基盤の改善が必要
- `run_id × git_sha` ベースのマイクロエポック分析は今後のレポーティング改善で対応
### 1.4 MEDIUM: state stale → **§3 で修正済**
### 1.5 MEDIUM: metrics gap → **§2 で修正済**
### Gemini §6.5-1: DUAL KILL 即時廃止 → **§4 で段階的廃止を推奨**
### Gemini §6.5-2: balance_forced + halt 順序修復 → **§1 で修正済**
### Gemini §6.5-3: skip-time state save → **§3 で修正済**

---

## 6. 追加改善点 (222# 範囲外)

### 6.1 per-side halt 復帰後の "冷却期間" 不在

per-side halt は `halt_cycles=15` で解除されるが、解除直後に即座に元の side で取引再開される。解除直後は市場条件が改善したかの検証がない。

**提案**: halt 解除後の最初の 3 サイクルは lot を半減する「ソフトリカバリ」を検討。

### 6.2 日替わりリセットと per-side halt の相互作用

`_daily_drawdown_guard.maybe_reset_day()` で日替わりリセットされると、per-side halt もクリアされるが、underlying の PnL データ (dynamic kill の rolling window) は日をまたいで持ち越される。日替わり直後は kill と halt の状態に矛盾が生じうる。

**提案**: 日替わりリセット時に dynamic kill の rolling window も truncate するか、少なくとも warning を出す。

---

## 7. セルフレビュー (223# 実装の検証)

### 7.1 確認項目と結果

| # | 検証項目 | 結果 | 評価 |
|---|----------|------|------|
| R1 | `_is_sell_killed()` / `_is_buy_killed()` の呼び出し回数 — 副作用 (`_stale_counter++`) が 1 回/サイクルか | gate evaluate の引数として 1 回ずつ呼ばれる。OK | ✅ |
| R2 | `_make_loop_skip_record` の `balance_forced_switch` パラメータ存在確認 | L337-368 に定義あり。OK | ✅ |
| R3 | balance_forced halt block 後の `_last_side` 設定 — freeze + halt で空回りリスク | `freeze_remaining` がデクリメントされ数サイクルで解消。実害は軽微 | ⚠ LOW |
| R4 | skip-time state save が gate_block path のみの問題 — 他 continue パスも stale? | DD halt は独自 save あり。他はいずれも短 sleep + 次ループで normal path 到達。gate_block のみで十分 | ✅ |
| R5 | `_cycle_count` は `run_single_cycle` でのみ++ — skip 中は `progress_log_interval` 到達不能 | 223# skip-time save が正しく必要な理由。OK | ✅ |
| R6 | `dual_kill_bypassed` / `probe_fired` / `force_release_fired` のテストカバレッジ | テスト 11 件追加 (`test_223_review_response.py`) | ✅ |

### 7.2 盲点 — 今後の改善候補

| # | 盲点 | 重要度 | 対応方針 |
|---|------|--------|----------|
| B1 | per-side halt 解除直後のソフトリカバリ不在 | MEDIUM | §6.1 参照。halt_cycles 解除後 3 サイクル lot 半減の検討 |
| B2 | 日替わりリセット × dynamic kill rolling window の矛盾 | MEDIUM | §6.2 参照。日替わり時の warning 追加を検討 |
| B3 | balance_forced_halt_block で `_balance_forced_skip_count` 未インクリメント | OK | continue で抜けるため下流に影響なし |
| B4 | gate_block 以外の continue パスでの state stale | LOW | 短 sleep → 次ループで save 到達。gate_block のみ長時間固着リスクあり |

---

## テスト結果

```
2982 passed, 20 warnings in 167.46s
```

既存テスト 2971 件 + 新規 11 件 (`test_223_review_response.py`) 全件通過。
新規フィールド (`dual_kill_bypassed`, `probe_fired`, `force_release_fired`) はデフォルト値 `False` のため後方互換。
