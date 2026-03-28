# 648# σ Stale Feedback Loop 修正

## 概要

03/28 fill_test ログ分析により、**σ が 0.000775 で 22 時間以上固定**され、
ATR floor が常に市場スプレッドを上回り、結果として
**注文が完全に停止**するバグを発見・修正。

## 問題の経緯

### タイムライン (03/27-28 JST)

| 時刻 | 事象 |
|------|------|
| 03/27 17:01 | fill_test 再起動 (SHA f7faac4f12) |
| 03/27 17:04 | 647# コミット後の再起動 (SHA da8b07b1cd) |
| 03/27 19:00 | **σ が 0.000417 → 0.000775 にジャンプ** |
| 03/28 00:01-06:43 | buy/sell 交互、10 fills 成功 |
| 03/28 06:43 | **最後の buy fill** — btc≈0.002, jpy≈230 |
| 03/28 07:57-12:04 | sell timeout ×6（~1/時）、他は全て cancel |
| 03/28 12:04 | **最後の Coincheck 注文** |
| 03/28 12:04-17:04 | **完全停止** — 5 時間ゼロ注文 |
| 03/28 17:04 | 再起動で σ=0 にリセット → 復旧 |

### 根本原因: σ stale feedback loop

`compute()` 内のパイプライン実行順序に起因：

```
旧実装:
  1. _enforce_spread_guards()  ← stale _last_sigma を使用
  2. pipeline stages...
     └─ _apply_as_reservation_shift()
         └─ _estimate_sigma()   ← ここで _last_sigma を更新
```

**フィードバックループ：**
1. σ = 0.000775 にジャンプ（正当な市場ボラティリティ）
2. 次サイクル: `_enforce_spread_guards` が stale σ で ATR floor 計算
   - ATR = min(0.000775 × 10.6M × 1.2, 10.6M × 3.0/10000) = min(9,858, 3,180) = **3,180 JPY**
3. 市場スプレッド (~1,500-2,500) < 3,180 → **InfeasibleQuoteError**
4. `_estimate_sigma()` に到達不可 → σ = 0.000775 のまま永久固定
5. ステップ 2 に戻る（永久ループ）

### 複合原因: 在庫デッドロック

06:43 の buy fill で jpy ≈ 230（最低注文額 ~21,500 が必要）。
buy は `preflight_insufficient` で永久ブロック。
sell は σ stale → ATR floor 過大でブロック。
**両方向とも注文不可 → 完全停止。**

## 修正内容

### コード変更

**`scripts/v460/lib/maker_price.py`** — `compute()` メソッド:

```python
# 修正後: _estimate_sigma を spread guard の前に移動
mid_trend_bps = self._refresh_market_state(...)
self._estimate_sigma(spread, mid_price)  # 648# σ refresh (NEW)
self._enforce_spread_guards(...)
```

- `_estimate_sigma(spread, mid_price)` を `_enforce_spread_guards` の**前**に呼出し
- これにより spread guard は常に **fresh σ** でATR floor を計算
- `_apply_as_reservation_shift` 内での2回目の呼出しは冪等（Parkinson H/L tracking は monotonic）
- AS reservation の early return 条件（neutral inventory, disabled 等）に依存しない σ 更新を保証

### テスト変更

**`tests/unit/v460/test_239_feasible_quote.py`**:
- `TestATRFloorCap.test_atr_no_cap_blocks`: mult=2.0→3.0 に変更
  （fresh σ では ATR = spread × mult/2 → mult>2 で block。旧テストは stale σ 前提）
- `TestSigmaStaleRefresh` クラス追加（3テスト）:
  - `test_stale_sigma_does_not_block_fresh_spread`: stale σ があっても fresh σ で通過
  - `test_sigma_refreshed_before_guard`: σ が市場整合的な値に更新されること
  - `test_fresh_sigma_still_blocks_when_spread_narrow`: 正当なブロックは維持

**`tests/unit/v460/test_143_regime_utilization.py`**:
- mock regime_detector に `last_volatility_ratio = 1.0` 追加（σ refresh 対応）

**`tests/unit/v460/test_157_regime_features.py`**:
- 同上

## 影響分析

### Parkinson σ の挙動（修正後）

| 5分窓の H-L 幅 | σ | ATR (mult=1.2) | cap 3.0bps |
|----------------|---|-----------------|------------|
| 1,000 JPY | 0.000057 | 721 | 721 |
| 2,000 JPY | 0.000113 | 1,441 | 1,441 |
| 3,000 JPY | 0.000170 | 2,162 | 2,162 |
| 5,000 JPY | 0.000283 | 3,603 | **3,180** (cap) |
| 8,000 JPY | 0.000453 | 5,765 | **3,180** (cap) |
| 14,000 JPY | 0.000793 | 10,089 | **3,180** (cap) |

修正後：5分窓リセット時に Parkinson は H==L でフォールバック →
Roll proxy σ = spread/(2·mid) → ATR = spread×0.6 < spread → 必ず通過。
次の窓内で市場が落ち着けば Parkinson σ も低下 → ATR floor 低下 → 取引再開。

### Roll proxy の自己整合性

Roll proxy: ATR = spread × mult/2 = spread × 0.6 (@mult=1.2)
→ ATR は常に spread 未満 → spread guard は Roll proxy では**絶対にブロックしない**。
Parkinson 窓リセット直後は Roll proxy がフォールバックに使われるため、
ワーストケースでも **300 秒以内に取引可能状態に復帰**。

## Part 2: 在庫デッドロック検出 + ATR cap 検証

### 在庫デッドロック検出

03/28 インシデントで判明した**buy preflight_insufficient と sell no_feasible_quote の
クロスチャネル検出不在**を解消。

**既存メカニズムの限界:**
- `_preflight_skip_count`: 両側不足 → balance_shrink → SAFE_STOP のみ
- `_consecutive_no_feasible`: 片側毎の追跡、反対側との交差検出なし
- buy 残高不足 + sell no_feasible が同時発生しても、別チャネルでカウントされ
  エスカレーションされない

**新メカニズム: `_inventory_deadlock_counter`**

```
non-fill cycle (preflight skip / unfilled / 片側残高不足) → counter++
fill success → counter = 0

if counter >= threshold(10) AND opposite_no_feasible >= 2:
    → WARNING ログ + _inc_guard_fire("inventory_deadlock")
    → 300秒インターバルでスロットル
```

### ATR cap 検証結果

σ stale fix が根本原因であり、**ATR cap=3.0bps の変更は不要**。

| シナリオ | H-L 幅 | ATR | cap 3,180 | 判定 |
|----------|--------|-----|-----------|------|
| 通常市場 | 1,500 | 1,081 | 不作動 | spread > ATR → 通過 |
| 中ボラ | 3,000 | 2,162 | 不作動 | spread > ATR → 通過 |
| 高ボラ | 5,300+ | 3,200+ | **作動** | cap で 3,180 に抑制 |
| 窓リセット | H==L | Roll proxy | 不作動 | ATR = spread×0.6 < spread |

cap=3.0bps は Parkinson H-L > ~5,300 JPY の**真の高ボラ時のみ**作動する
安全弁として適切。σ stale fix により循環ブロックは解消されているため変更不要。

### 潜在課題調査

網羅的調査により Critical/High バグは検出されず：
- Alert multiplier compound → 毎サイクルリセット + MCB_SAD_ESCALATION でスキップ。問題なし
- MCB/SAD hot-reload gap → 607# で対応済み
- Guard fire counts unbounded → 固定 ~15 キー、bounded
- Counter timing race → 単一スレッドの async loop、race condition なし

**低優先度の改善候補:**
- `_preflight_pause_count` 日替わりリセット未実装（SAFE_STOP がカバー）
- Parkinson σ 窓境界での一瞬の Roll proxy フォールバック（設計制限）
- 新 deadlock config の `_HOT_RELOADABLE_FIELDS` 未登録（alert 閾値のみ）

## 残存課題

| 優先度 | 課題 | 状態 |
|--------|------|------|
| ~~P1~~ | ~~在庫デッドロック検出~~ | ✅ 実装済み |
| ~~P2~~ | ~~ATR cap チューニング~~ | ✅ 変更不要と結論 |
| P2 | preflight_insufficient 継続検出 | 648# deadlock detection でカバー |
| P3 | sell timeout 分析 | timeout 売注文の fill 失敗原因 |

## 変更ファイル一覧

### Part 1 (SHA f35ef8ee5)

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/maker_price.py` | `compute()`: σ refresh を spread guard 前に移動 |
| `tests/unit/v460/test_239_feasible_quote.py` | ATR テスト修正 + σ stale テスト追加 |
| `tests/unit/v460/test_143_regime_utilization.py` | mock 修正 (`last_volatility_ratio`) |
| `tests/unit/v460/test_157_regime_features.py` | mock 修正 (`last_volatility_ratio`) |

### Part 2

| ファイル | 変更内容 |
|----------|----------|
| `ztb/trading/common/cancel_reasons.py` | `INVENTORY_DEADLOCK` 定数 + AUDIT set + Literal |
| `scripts/v460/lib/fill_config.py` | `inventory_deadlock_threshold/alert_interval_sec` 追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | クラス属性 `_inventory_deadlock_counter`, `_last_inventory_deadlock_alert_time` |
| `scripts/v460/lib/orchestrator_balance.py` | `_check_inventory_deadlock()` + カウンタ増分 |
| `scripts/v460/lib/orchestrator_post_cycle.py` | fill/unfill 時のカウンタリセット/増分 |
| `tests/unit/v460/test_145_structural_fixes.py` | AUDIT frozenset に INVENTORY_DEADLOCK 追加 |
| `tests/unit/v460/test_648_inventory_deadlock.py` | 新テスト 15件 |
| `docs/v460/648_cplt_sigma_stale_feedback_loop_fix.md` | Part 2 追記 |
