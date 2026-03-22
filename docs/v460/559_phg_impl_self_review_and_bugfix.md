# 559# セルフレビュー: 553#–555# CalibrationMap パイプライン バグ修正

| 項目 | 値 |
|------|-----|
| 番号 | 559# |
| 種別 | impl (bugfix + observability) |
| 対象 | 553#–555# CalibrationMap パイプライン |
| 前提 | 553#–555# 実装完了 + ドキュメント・テスト修正 (46ef1b6c5) |

---

## 1. 背景

553#–555# で構築した CalibrationMap パイプライン（offline batch → runtime integration）の
セルフレビューを実施。3 件の実バグと 1 件の可観測性不足を発見・修正した。

---

## 2. 発見・修正した問題

### 2.1 C2: PnL 単位不整合 (CRITICAL)

**場所**: `orchestrator_post_cycle.py` — CalibrationMap online 更新

**問題**:
- offline batch (`calibration_batch.py`): `record.post_fill_30s_pnl` (bps 単位) を使用
- online update: `compute_record_pnl_jpy(record)` → JPY 単位 を使用

EWMA に bps と JPY が混在すると、avg_win / avg_loss の分布が根本的に崩壊する。

**修正**: online update を `record.post_fill_30s_pnl` (bps) に統一。
`pnl_jpy` は `cumulative_pnl_jpy` 専用に分離。

### 2.2 Action Bin 不整合 (CRITICAL)

**場所**: `calibration_batch.py` `_side_to_action()`

**問題**:
- offline batch: `buy → +1.0, sell → -1.0` → `CalibrationMap._get_bin()` で
  Strong_Buy / Strong_Sell にマッピング
- runtime (mid_cycle / post_cycle): `buy → +0.3, sell → -0.3` → Buy / Sell にマッピング

L1 キーが `{regime}_Strong_Buy` vs `{regime}_Buy` で完全に分離。
offline batch で構築した L1 データが runtime で一切参照されない。
（L2/L3 fallback は機能するが、L1 の粒度が無駄になる）

**修正**: `_side_to_action()` を `±0.3` に変更し、Buy / Sell bin に統一。

### 2.3 C3: CalibrationGate.evaluate() のコスト二重計上 (NOT A BUG)

**評価結果**: CalibrationGate は cost model 付き EV を計算するが、
offline 学習データが `post_fill_30s_pnl`（コスト込み実績 PnL）であるため、
avg_win / avg_loss にはスプレッドコストが含まれている。
mid_cycle の手動 EV 式 `p_win * avg_win - (1-p_win) * avg_loss` は**正しい**。
CalibrationGate の cost 項を追加するとコスト二重計上になるため、現状維持。

### 2.4 H2: Entry Gate 統計のログ未出力 (MEDIUM)

**場所**: `fill_loop_orchestrator.py` で `RunSessionState` に 3 フィールド定義:
- `entry_gate_eval_count`: EV 評価回数
- `entry_gate_block_count`: EV ≤ 0 の回数
- `entry_gate_ev_sum`: EV 加算

これらは `orchestrator_mid_cycle.py` でインクリメントされるが、
進捗ログに一切出力されていなかった。

**修正**: `_log_progress_and_adapt()` の進捗ログセクションに
EntryGate 統計行 (eval/block/avgEV) を追加。

---

## 3. レビューで確認済み・問題なしの項目

| 項目 | 結果 |
|------|------|
| CalibrationMap EWMA 減衰ロジック | 正常: decay = exp(-dt/tau), dt > 0 でのみ適用 |
| Beta prior (α=2, β=2) | 適切: 情報の少ない prior、n_eff 増加で prior 影響が減少 |
| 3 段階 fallback (L1→L2→L3) | 正常: n_eff ≥ n_min で L2 信頼、else L3 |
| null/None ガード (mid_cycle) | 正常: `self._calibration_map is not None` チェック |
| CycleGateResult 生成 | 正常: blocking_reason="entry_gate_ev_negative" |
| gated_regime 使用 (post_cycle) | 適切: 意思決定時点の regime を記録 |

---

## 4. 残課題 (将来対応)

| 優先度 | 課題 | 状態 |
|--------|------|------|
| HIGH | CalibrationMap state persistence (get_state → file) | 555# doc に「将来」記載。online 学習がセッション跨ぎで消失 |
| MEDIUM | Regime source 整合性 (offline: record.regime vs online: record.gated_regime) | L2 fallback で概ね吸収。regime / gated_regime の差異は小さい |
| LOW | CalibrationStats TypedDict 化 | 現状 dict access。型安全性向上余地あり |
| LOW | Hot reload 非対応 | entry_gate_* パラメータ変更時に CalibrationMap 再初期化不要 (ewma_tau 等は変更頻度極低) |

---

## 5. 前セッションの作業記録

### 5.1 ドキュメント作成
- `docs/v460/553_phg_impl_ohlcv_auto_update_pipeline.md` (95 行)
- `docs/v460/554_phg_impl_raw_gap_fill_and_calibration_batch.md` (161 行)
- `docs/v460/index.md` 更新 (553–555 のファイルリンク追加)

### 5.2 テスト修正 (555# 起因で陳腐化した 6 件)
1. `test_166_remaining_tasks.py` — 2500 char window 超過; 分割チェックに変更
2. `test_276_blocking_policy_dry.py` — 同上
3. `test_336_yaml_code_drift_prevention.py` — KNOWN_YAML_OVERRIDES に 6 項目追加 + field cap 470→500
4. `test_239_feasible_quote.py` — sell_guard 移動先 `_enforce_spread_guards()` 追従
5. `test_305_p0_improvements.py` — OB cache "305# S2" → `_resolve_market_snapshot` 追従
6. `test_ml_cache_cleanup.py` — 削除済みモジュール 6 件を _ENTRYPOINTS から除去

### 5.3 既知の既存問題 (未修正)
- `test_enricher_skip_gate`: PnL データ不足でフレーキー
- `test_408_f_series_blindspot`: `inspect.getsource` 行番号ズレ
