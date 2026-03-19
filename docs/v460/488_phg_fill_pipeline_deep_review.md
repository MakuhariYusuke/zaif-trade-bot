# 488# Fill Test パイプライン総合コードレビュー

> 種別: review / audit
> 対象: fill_cycle_executor, orchestrator_mid/post_cycle, fill_loop_orchestrator, maker_price, maker_risk_guards, daily_drawdown_guard, sidecar_signal_io, fill_test_config, config_hot_reload, side_selector
> 日付: 2026-03-19
> 目的: Codex レビュー用 — 487# §7 ログ分析で判明した PF 0.883 / fill rate 30% / balance insufficient 5,703 件の根本原因を多角的に調査

---

## 0. 背景と動機

487# §7 の fill test ログ分析（33 日間）で以下の深刻な問題が確認された:

| 指標 | 値 | 問題 |
|---|---|---|
| Profit Factor | 0.883 | 赤字運用 |
| cumPnL | -986 bps | 累積損失 |
| Fill rate | 86% → 30% | 急激な劣化 |
| Balance insufficient | 5,703 件 | 最大のフリクション |
| sell_dynamic_kill | 979 件 | 過剰ゲートブロック |
| NameError (10.5h outage) | 260 件 | §6 で修正済 |

本ドキュメントでは **ログ出力・エラー処理・設定検証・アーキテクチャ・パフォーマンス** の5カテゴリで問題を洗い出し、Codex レビューに供する。

---

## 1. ログ出力の改善点

### 1.1 サイクル結果ログの欠落フィールド

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py` L563-589

`_log_cycle_result()` で以下の重要情報がログに出力されていない:

| 欠落フィールド | 影響 | 備考 |
|---|---|---|
| `cancel_reason` | unfilled 時の原因特定不能 | filled=False パスで省略 |
| `_queue_fill_prob_est` | キュー位置の推定精度検証不能 | L898-910 で計算されるが FillRecord に格納のみ、ログには出ない |
| regime 情報 | 環境条件との相関分析不能 | progress log にはあるがサイクル単位では欠落 |
| SkipGate スコア | gate 判定の透明性なし | gate_result にあるがログ出力なし |
| spread 実測値 | quote の妥当性分析不能 | offset と mid_price はあるが bid-ask spread なし |

### 1.2 プログレスログの欠落

**ファイル**: `scripts/v460/lib/orchestrator_post_cycle.py` L247-261

現在のプログレスログ出力:
```
Progress: {cycles} cycles, fill rate=X/Y (Z%), cumPnL=NJPY, btcDelta=+MBTC, lot=L, regime=R, none_regime=A/B, unsaved_batch=C
```

**欠落している情報**:

| 欠落項目 | 理由 |
|---|---|
| cancel_reason 分布 | `cancel_reason_counts` は `RunSessionState` L91 に蓄積されているがログ出力されない |
| per-side fill 統計 | buy/sell 別の fill 数・PnL が見えない → 片側偏りの検出遅延 |
| sidecar 活動統計 | `sidecar_fresh/stale/missing_count` は state にあるがプログレスでは非表示 |
| キュー健全性指標 | wait time 中央値、stale reprice 回数 |
| slippage メトリクス | 実行価格と理論価格の差 |

### 1.3 ログフォーマットの不統一

| 問題箇所 | 現状 | 提案 |
|---|---|---|
| bracket tag | `[cross_venue]`, `[toxicity]` vs `Progress:` (bracket なし) | 全タグを `[tag]` 統一 |
| 数値精度 | .1f / .2f / .3f / .4f / .6f / .8f が混在 | JPY は .1f, BTC は .8f, % は .2f に統一 |
| パーセント表記 | `85.3%` vs `0.853` が混在 | 可読性重視で `%` に統一 |
| NO_FEASIBLE_QUOTE | spread threshold 値がログに出ない | 判定に使った spread 閾値を明記 |
| Order attempt error | 残高・circuit breaker 状態がログにない | 注文失敗時のコンテキスト出力追加 |

---

## 2. エラー処理・例外安全性

### 2.1 [P0] Cross-venue hint 例外の飲み込み

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py` L291

```python
except Exception as exc:
    logger.warning("cross-venue hint update error: %s", exc, exc_info=True)
    self._maker_price.set_cross_venue_lead_lag_hint(None)
```

**問題**: bare `Exception` で全エラーをキャッチし、hint を None にフォールバック。API 障害・認証失敗・タイムアウトが全て同じ処理になり、原因分類が不可能。

**提案**: 例外型ごとに分類 (`ConnectionError`, `TimeoutError`, `ValueError`) し、型別カウンタをログ出力。

### 2.2 [P0] `_execute_and_track_cycle` の広範 Exception キャッチ

**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py` L448

```python
except Exception as e:
    logger.error(f"Cycle execution error: {e}", exc_info=True)
    if _recovery_scale < 1.0:
        self._daily_drawdown_guard.restore_recovery_counter(next_side)
    return
```

**問題**:
- 全例外をキャッチし `return` → 呼び出し元が成功と区別不能
- partial state restore: `_recovery_scale < 1.0` の時のみ restore → **条件に合わない場合は state が不整合のまま続行**
- `KeyboardInterrupt` は L441 で別処理されているが、`SystemExit` は捕捉される

**提案**: 再送可能な例外 (`ConnectionError` 等) と致命的例外を明確に分離。return ではなく例外を再送するか、結果オブジェクトで区別。

### 2.3 [P1] 片側エスカレーションの非原子的状態変更

**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py` L476-545

`_track_one_sided_escalation()` で 5+ のインスタンス変数を逐次変更:

```python
self._one_sided_consecutive_count += 1      # L481
self._one_sided_freeze_remaining = _freeze_n # L496
self._one_sided_frozen_side = next_side      # L497
# ... さらに複数フィールド
```

**問題**: 変更途中で例外が発生した場合、一部だけ更新された状態になる。ロールバック機構がない。

**提案**: dataclass にまとめて一括代入、または snapshot/restore パターン。

### 2.4 [P1] batch_persistence 保存失敗後の flush timer リセット

**ファイル**: `scripts/v460/lib/orchestrator_post_cycle.py` L213-218

```python
if self._batch_persistence.try_save_batch(st.batch):
    st.batch = []
    self._batch_persistence.reset_flush_timer()
else:
    st.batch = self._batch_persistence.maybe_flush(st.batch, "run_loop")
```

**問題**: `try_save_batch()` が例外を投げた場合（bool 返却の保証なし）、`reset_flush_timer()` は呼ばれないが `maybe_flush` も呼ばれない → 保存が永久にスキップされるリスク。

### 2.5 [P1] sidecar_signal_io の stat/read 競合

**ファイル**: `scripts/v460/lib/sidecar_signal_io.py` L125, L149

```python
mtime = path.stat().st_mtime       # I/O #1 (L125)
# --- RACE WINDOW: SAC がここでファイルを書き換え可能 ---
raw = path.read_text(encoding="utf-8")  # I/O #2 (L149)
```

**問題**: `stat()` で取得した mtime と `read_text()` で読んだ内容が異なるファイルバージョンの可能性。SAC が高頻度で signal を書き換える場合、キャッシュ判定が破綻する。

**提案**: `read_text()` で読んだ後に再度 `stat()` し、mtime が一致していることを確認する（double-check pattern）、またはファイルロック。

### 2.6 [P2] random.random() による非再現性

**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py` L304

```python
and random.random() > gate_result.participation_rate
```

**問題**: システム global な `random` を使用。シードなしのため backtest 再現性がない。

**提案**: 専用の `random.Random(seed)` インスタンスを使用。

---

## 3. 設定検証・バリデーション欠陥

### 3.1 [P0] offset_ceiling_ratio の論理矛盾

**ファイル**: `ztb/trading/maker_price.py` L595

```python
def _effective_max_ratio(self, side: str) -> float:
    base = cfg.max_offset_ratio  # 0.30
    if side == "sell" and cfg.offset_ceiling_ratio_sell is not None:
        return max(base, cfg.offset_ceiling_ratio_sell)  # max(0.30, 0.20) = 0.30!
```

**問題**: `offset_ceiling_ratio_sell = 0.20` が `max_offset_ratio = 0.30` より小さい場合、`max()` により ceiling が無効化。**ceiling が機能していない**。

**本番影響**: offset = 0.24 ratio（43 JPY on mid=18,000）→ AS loss が常態化 → **PF 0.883 の直接原因の一つ**。

**提案**: `max()` ではなく `min()` を使用すべき。ceiling は上限なので `min(base, ceiling)` が正しい。

### 3.2 [P0] sigma_floor = 0 でゼロ除算

**ファイル**: `scripts/v460/lib/fill_test_config.py` L1107

```python
sigma_floor: float = 1e-6  # バリデーションなし
```

**問題**: 0 に設定された場合、AS δ*, Kyle λ, Amihud 計算でゼロ除算。hot-reload 経由なら実行時に設定可能。

### 3.3 [P1] VPIN 閾値の相互参照チェック欠如

**ファイル**: `scripts/v460/lib/fill_test_config.py` L871-873

```python
volatility_guard_vpin_threshold = 0.70
vg_vpin_continuous_min = 0.40
```

**制約**: `vpin_threshold > vg_vpin_continuous_min` が必要だがバリデーションなし。逆転すると VPIN boost が常に最大。

### 3.4 [P1] microprice_depth > weights 長でクラッシュ

**ファイル**: `scripts/v460/lib/fill_test_config.py` L885

```python
microprice_depth: int = 5  # _MICRO_WEIGHTS は 5 要素
```

**問題**: depth=6 に設定すると `IndexError`。weight 長との整合チェックなし。

### 3.5 [P1] per_side_dd_hard_limit と reanchor_budget の逆転

**ファイル**: `scripts/v460/lib/fill_test_config.py` L674, L682

```python
per_side_dd_hard_limit_bps = -50.0
per_side_dd_reanchor_budget_bps = -25.0
```

**制約**: `hard_limit < reanchor_budget` が必要（回復のために）。逆転すると永久 halt ループ。バリデーションなし。

### 3.6 [P1] sidecar_max_boost_bps / sidecar_dead_zone 未検証

**問題**: 負の `max_boost_bps` や `dead_zone > 1.0` が設定可能。結果として sidecar offset が逆方向に作用。

### 3.7 [P2] hot-reload 後のキャッシュ不整合

**ファイル**: `scripts/v460/lib/config_hot_reload.py`

```python
new_config = type(self._config).from_yaml(new_yaml_cfg)
# maker_price._last_spread / _last_imbalance のキャッシュは無効化されない
```

**問題**: config 変更後も `MakerPriceCalculator` のキャッシュが旧設定の値を保持 → offset 計算が一時的に不整合。

### 3.8 [P2] hot-reload 対象外の重要フィールド

以下のフィールドが `_HOT_RELOADABLE_FIELDS` に含まれていない:

- `sigma_floor` — AS 計算の基盤
- `vol_ratio_floor` — regime scaling の分母
- `micro_timeout_max_requote` — requote 制限
- `stale_reprice_min_delta_jpy` — deadband 閾値

---

## 4. リスクガード・ドローダウン防御の構造的問題

### 4.1 [P0] VPIN continuous ramp の sell_dynamic_kill 助長

**ファイル**: `scripts/v460/lib/maker_risk_guards.py` L123-134

```python
_norm = min((self._last_vpin - _min_vpin) / (_thresh - _min_vpin), 1.0)
vpin_boost = 1.0 + (boost_factor - 1.0) * _norm * _norm  # 二次曲線
```

**問題**:
- VPIN=0.65 で boost=1.69x、VPIN=0.70+ で boost=2.0x（ハードシーリング）
- **VPIN 0.65-0.75 間で勾配なし** → 非危機的状況でも offset が膨張 → quote stale → **979 sell_dynamic_kill の直接原因**
- 二次曲線のみでログ関数やシグモイドのテストなし

**提案**: VPIN 閾値以上のゾーンで段階的な勾配を設ける。boost 上限の config 化。

### 4.2 [P0] inverse skew damping が sell 防御を無効化

**ファイル**: `scripts/v460/lib/maker_risk_guards.py` L326 付近

**問題**:
- inventory short (BTC 枯渇) → `inv_skew_factor = -1.0`（最大 sell discount）
- damping が VG boost を 2.0x → 1.0x に抑制
- **高 vol + short inventory = sell 最脆弱時に offset が最小** → sell 損失拡大
- **979 sell_dynamic_kill blocks** と整合: damping が sell offset を抑制 → quote stale → kill 発動

**提案**: short inventory 時は damping を無効化するか、方向を反転させる。

### 4.3 [P1] per-side halt 後の 25 bps 無防備ウィンドウ

**ファイル**: `scripts/v460/lib/daily_drawdown_guard.py` L247-266

```python
_reanchor = self._state.side_reanchor_pnl_buy  # halt 解除時の PnL
_effective_pnl = side_pnl - _reanchor
_threshold = self._per_side_reanchor_budget_bps  # -25 bps
```

**シナリオ**:
1. daily_pnl_buy = -60 bps → halt 発動
2. 10 サイクル後: halt 解除、`side_reanchor_pnl_buy = -60 bps`
3. 新 fill: daily_pnl_buy = -70 bps → `_effective_pnl = -10 bps > -25 bps` → **再 halt されない**
4. -85 bps (-60 + -25) に達するまで無防備 → **25 bps の損失ウィンドウ**

### 4.4 [P1] cooldown 再 arm の永久ロック

**ファイル**: `scripts/v460/lib/daily_drawdown_guard.py` L272-289

```python
if self._state.cooldown_released and not self._state.cooldown_rearmed:
    self._state.cooldown_rearm_pnl_bps += pnl_bps
    if self._state.cooldown_rearm_pnl_bps <= self._cooldown_rearm_budget_bps:
        self._state.cooldown_rearmed = True  # 一方通行
```

**問題**: `cooldown_rearmed = True` になった後、**日次リセットまで解除されない**。1 fill が budget を超えただけで残りの時間は完全停止。段階的回復機構がない。

### 4.5 [P2] loss_boost の効果が fill タイミング依存

**ファイル**: `ztb/trading/maker_price.py` L712-757

```python
_elapsed = now - self._loss_boost_set_time
_decay = math.exp(-_elapsed / _tau)  # τ=300s
```

**問題**: `set_loss_boost()` と `compute()` が同サイクルで呼ばれる保証なし。halt 中に set されると、再開時には decay ≈ 0 → loss defense 無効化。

### 4.6 [P2] hardcoded magic numbers

**ファイル**: `scripts/v460/lib/daily_drawdown_guard.py` L42

```python
_WARMUP_REPAIR_EPS: float = 0.01   # per-side PnL ≈ 0 判定
_WARMUP_REPAIR_MIN_PNL: float = 1.0  # 合計 PnL 有意判定最小 bps
```

config 化されておらず、チューニングにコード変更が必要。

---

## 5. 構造・DRY・パフォーマンス

### 5.1 [P1] offset clamping の 3 重適用

| 場所 | ファイル | 行 |
|---|---|---|
| `_scale_offset_ratio()` | maker_price.py | L595 |
| `resolve_offset_ceiling()` | fill_test_config.py | 内部 |
| `execution_final_clamp` | fill_cycle_executor.py | L821 |

**問題**: 3 箇所で独立にクランプ → 境界が同期されていない場合、offset がレイヤー間で振動。**fill rate 86% → 30% の一因**。

**提案**: single source of truth としてクランプロジックを一箇所に集約。

### 5.2 [P1] cancel_reason_counts の無限成長

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` L91

```python
cancel_reason_counts: dict[str, int] = field(default_factory=dict)
```

**問題**: key が固定文字列ならリスクは低いが、動的に生成される場合メモリリーク。eviction / 清掃ロジックなし。

### 5.3 [P1] inventory skew の thread safety

**ファイル**: `ztb/trading/maker_price.py` L197-199

```python
self._inv_fill_history: collections.deque[str] = collections.deque(maxlen=_w)
self._inv_buy_count: int = 0
```

**問題**: deque と手動カウンタが同期保証なし。async fill が race すると counter 破損 → imbalance が [-1, 1] 範囲を逸脱。

### 5.4 [P2] side_selector の freeze decrement race

**ファイル**: `scripts/v460/lib/side_selector.py` L142-143, L167

```python
if self._frozen_side is not None and self._frozen_remaining > 0:
    self._frozen_remaining -= 1  # next() 呼び出しごとにデクリメント
```

**問題**: 同一サイクルで `next()` が 2 回呼ばれると freeze が早期解除。atomicity guard なし。

### 5.5 [P2] regime=None 時の microprice override 誤活性化

**ファイル**: `scripts/v460/lib/side_selector.py` L203-204

```python
if _guardrail_pass and _rg and regime and regime not in _rg:
    _guardrail_pass = False
```

**問題**: `regime=None`（warmup/不明）時は `regime not in ["ranging"]` → True → guard PASS → warmup 中に microprice override が活性化。**最も不確実な状況で override が有効になる逆説**。

---

## 6. 本番障害との因果関係マッピング

| 本番症状 | 根本原因候補 | 該当セクション |
|---|---|---|
| **PF 0.883 (赤字)** | offset_ceiling_ratio の `max()`/`min()` 逆転 (§3.1)、VPIN boost 膨張 (§4.1)、loss_boost タイミング依存 (§4.5) | §3.1, §4.1, §4.5 |
| **fill rate 86%→30%** | 3 重 offset clamping (§5.1)、VPIN continuous ramp (§4.1)、offset ceiling 無効化 (§3.1) | §5.1, §4.1, §3.1 |
| **5,703 balance insufficient** | per-side halt 後の 25 bps 無防備ウィンドウ (§4.3)、cooldown 永久ロック (§4.4) | §4.3, §4.4 |
| **979 sell_dynamic_kill** | inverse skew damping の sell 防御無効化 (§4.2)、VPIN boost 上限固定 (§4.1) | §4.2, §4.1 |
| **NameError 10.5h** | §6 (487# で修正済) | 487# §6 |

---

## 7. 改善アクション優先度

### P0 (即時対応 — 収益直結)

| # | タイトル | 対象ファイル | 概要 |
|---|---|---|---|
| P0-1 | offset ceiling `max()` → `min()` 修正 | maker_price.py L595 | ceiling が機能していない根本バグ |
| P0-2 | VPIN boost 上限の段階化 | maker_risk_guards.py L123 | 非危機的 VPIN で offset 膨張を抑制 |
| P0-3 | inverse skew damping の方向修正 | maker_risk_guards.py L326 | sell 防御を逆に抑制しているバグ |
| P0-4 | sigma_floor=0 バリデーション追加 | fill_test_config.py L1107 | ゼロ除算防止 |

### P1 (短期改善 — 安定性)

| # | タイトル | 対象ファイル | 概要 |
|---|---|---|---|
| P1-1 | `_execute_and_track_cycle` 例外分類 | orchestrator_mid_cycle.py L448 | bare Exception を typed catch に |
| P1-2 | サイクル結果ログ拡充 | fill_cycle_executor.py L563 | cancel_reason, queue_fill_prob_est 追加 |
| P1-3 | プログレスログ拡充 | orchestrator_post_cycle.py L247 | cancel 分布, sidecar 統計追加 |
| P1-4 | per-side halt reanchor budget 検証 | daily_drawdown_guard.py L247 | 25 bps 無防備ウィンドウの縮小 |
| P1-5 | cooldown 段階的回復 | daily_drawdown_guard.py L272 | 永久ロック回避 |
| P1-6 | config 相互参照バリデーション | fill_test_config.py | VPIN 閾値逆転, depth/weights 整合 |
| P1-7 | offset clamping 一元化 | maker_price + fill_config + executor | 3 重クランプの single source of truth 化 |
| P1-8 | 片側エスカレーション atomic 化 | orchestrator_mid_cycle.py L476 | dataclass 一括代入パターン |

### P2 (中期改善 — 品質)

| # | タイトル | 対象ファイル | 概要 |
|---|---|---|---|
| P2-1 | ログフォーマット統一 | 全パイプラインファイル | bracket tag, 数値精度, % 表記 |
| P2-2 | sidecar stat/read race fix | sidecar_signal_io.py L125 | double-check pattern |
| P2-3 | random.Random(seed) 化 | orchestrator_mid_cycle.py L304 | backtest 再現性 |
| P2-4 | hot-reload cache invalidation | config_hot_reload.py | maker_price キャッシュクリア |
| P2-5 | hot-reload 対象フィールド追加 | config_hot_reload.py | sigma_floor 等の重要フィールド |
| P2-6 | magic number の config 化 | daily_drawdown_guard.py L42 | _WARMUP_REPAIR_EPS 等 |
| P2-7 | regime=None 時の microprice guard | side_selector.py L203 | warmup 中の override 防止 |
| P2-8 | cancel_reason_counts eviction | fill_loop_orchestrator.py L91 | 長期運用でのメモリ安全性 |
| P2-9 | inventory skew thread safety | maker_price.py L197 | async fill 競合防止 |
| P2-10 | loss_boost の set/consume 同期 | maker_price.py L712 | halt 中の decay 無効化防止 |

---

## 8. 補足: テスト観点

上記の修正にあたり、以下のテストが必要:

1. **P0-1**: offset ceiling が `min()` で正しくクランプされることの単体テスト
2. **P0-2**: VPIN boost の段階的勾配が sell_dynamic_kill を削減することの統計テスト
3. **P0-3**: short inventory + high vol 時に sell offset が増加する (以前は減少) ことの確認
4. **P1-1**: typed exception ごとに正しいハンドリングが行われることの確認
5. **P1-6**: 不正 config (sigma_floor=0, depth=6 等) で例外が投げられることの確認
6. **P1-7**: 単一クランプで旧 3 重と同等の output range を持つことの回帰テスト

---

> 以上、Codex レビュー用。P0 項目は収益直結のため最優先で対応すべき。
