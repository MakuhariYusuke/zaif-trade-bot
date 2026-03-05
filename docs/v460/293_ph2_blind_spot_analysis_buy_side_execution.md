# 293# Buy-Side Execution Blind Spot Analysis — 290#/291# 査読補完

> **日付**: 2026-03-06  
> **対象**: v460 maker execution system — buy 側重点  
> **レビュー対象**: 290# (Codex), 291# (Gemini 3.1 Pro), 292# (実装)  
> **手法**: ソースコード静的解析 (fill_cycle_executor, order_monitor, skip_gate_evaluator, fill_config, fill_quality)

---

## 結論サマリ

| # | 盲点 | 深刻度 | 要即時修正 |
|---|---|---|---|
| BS-1 | Reprice deadband: cancel が deadband 判定より先に実行 — queue position 喪失 | **CRITICAL** | **YES** |
| BS-2 | ev_offset パラメータが buy/sell 共通 — sell は暗黙的に同一感度で運用 | **MEDIUM** | NO (計測後) |
| BS-3 | Skip record に 292# 3 フィールドが記録されない — ev_emergency_skip 分析不能 | **HIGH** | YES |
| BS-4 | ev_offset_mult_applied=None の曖昧性: ev 未計算 vs mult=1.0 の区別不能 | **MEDIUM** | YES |
| BS-5 | SkipGate blocked reprice → 注文消滅 (cancel 済み・再発注なし) | **MEDIUM** | NO (設計意図) |
| BS-6 | R-list 中 buy 側直結タスク R-3/R-5 が R-1 完了待ち停滞 | **LOW** | NO (Phase A 凍結中) |

---

## BS-1: Reprice Deadband — Cancel-Then-Check の構造バグ【CRITICAL】

**ファイル**: [order_monitor.py](scripts/v460/lib/order_monitor.py#L470-L545)

### 問題

292# P1 で実装された reprice deadband（`stale_reprice_min_delta_jpy=500`）は、queue position 保護を目的としている。しかし、**既存注文のキャンセルが deadband 判定の前に実行される**ため、queue position は deadband が発動する時点で既に失われている。

**実行順序**:

```
L479: cancel (await _try_cancel_with_fill_recheck)  ← ここで queue 喪失
L488: if not cancel_succeeded: continue
L493: SkipGate reprice guard
L507: new_price = compute_maker_price()
L527: deadband check: if abs(new_price - order_price) < min_delta  ← 手遅れ
L537:   new_price = order_price  (同じ価格で再発注)
L538: place_order(new_price)  ← 最後尾に並び直し
```

コメント（L535）には「Skip reprice to protect queue position」と記載されているが、実際には **L479 のキャンセルで queue priority は不可逆的に消滅済み**。同一価格での再発注は最後尾からの再出発であり、Gemini 291# が指摘した「Cancel & Replace による逆選択の自爆構造」そのもの。

### 修正案

deadband チェックを cancel の **前** に移動する:

```python
# BEFORE cancel
result = await compute_maker_price(side)
new_price = tighten(result[0])  # tighten 適用後
if _min_delta > 0 and abs(new_price - order_price) < _min_delta:
    logger.info("Deadband: skip entire reprice cycle")
    continue  # cancel を実行しない → queue 保持
# THEN cancel
chk = await self._try_cancel_with_fill_recheck(...)
```

### 影響範囲
レンジ相場 (ranging) で favorable drift が `stale_drift_bps` を超えるが、reprice target が `min_delta_jpy` 未満の微小変動の場合に、無意味な queue 再構築が発生し続ける。夜間低 VPIN 環境では約定が Toxic Flow のみとなり、buy 側 PnL リークの一因。

---

## BS-2: ev_offset パラメータの buy/sell 非分離【MEDIUM】

**ファイル**: [fill_config.py](scripts/v460/lib/fill_config.py#L288-L295)

### 問題

`ev_offset` 関連パラメータが buy/sell で完全に共通:

```python
skip_gate_ev_offset_sensitivity: float = 0.05    # 共通
skip_gate_ev_offset_min_mult: float = 0.5         # 共通
skip_gate_ev_offset_max_mult: float = 1.5         # 共通
skip_gate_ev_warning_threshold: float = -4.0      # 共通
skip_gate_ev_warning_offset_factor: float = 0.7   # 共通
```

他の SkipGate パラメータには side 別バリアント（`as_threshold_buy/sell`, `model_path_buy/sell`, `target_skip_rate_buy/sell`）が存在するのに、`ev_offset` には存在しない。

`_apply_offset_multiplier` 内部の delta 適用方向は side-aware（L529-534: buy は `+delta`、sell は `-delta`）なので **方向は正しい**。しかし **感度の大きさ** が同一であり、buy と sell で逆選択の性質が異なる市場（e.g. 下落トレンドでは buy の AS が sell の AS より深刻）では最適化余地がある。

### 判断
計測データ（R-1 完了後）で buy/sell の `ev_offset_mult_applied` 分布を比較し、有意差があれば `skip_gate_ev_offset_sensitivity_buy/sell` を追加。現時点では YAML 変更凍結中のため保留。

---

## BS-3: Skip Record に 292# 3 フィールドが記録されない【HIGH】

**ファイル**: [skip_gate_evaluator.py](scripts/v460/lib/skip_gate_evaluator.py#L307-L340) → `_make_skip_fill_record`

### 問題

292# P0 で追加された 3 フィールド:
- `ev_score_pretrade`
- `ev_offset_mult_applied`
- `decision_path`

これらは `_build_fill_record`（executor 経由の正常フロー）でのみ記録される。`_make_skip_fill_record`（skip_gate_evaluator 内の早期リターン）では **一切渡されていない**。

特に問題なのは **`ev_weighted_emergency_skip`** のケース（[skip_gate_evaluator.py L1225](scripts/v460/lib/skip_gate_evaluator.py#L1225)）:

```python
# ev_score は計算済み (_ev_combined.predicted_pnl_bps に格納)
# しかし _make_skip_fill_record に ev_score_pretrade を渡していない
result.early_return_record = self._make_skip_fill_record(
    cycle_id=cycle_id,
    # ... ev_score_pretrade は不在
)
```

`build_skip_fill_record` は `**extra` kwargs を受け付ける設計（[fill_quality.py L252](ztb/metrics/fill_quality.py#L252)）なので、技術的には `ev_score_pretrade=result.ev_score` を追加するだけで修正可能。

### 影響
- ev_emergency_skip が発動した際の ex-ante ev_score が記録されず、**しきい値チューニングの根拠データが欠落**
- `decision_path` が None となるため、skip 分析時に「ev_emergency_skip かルールスキップか」を `cancel_reason` 文字列から推定する必要がある（冗長・脆弱）

### 修正案

`_make_skip_fill_record` のシグネチャに 3 フィールドを optional で追加し、`build_skip_fill_record` の `**extra` 経由で注入:

```python
@staticmethod
def _make_skip_fill_record(
    *,
    # ... existing params ...
    ev_score_pretrade: float | None = None,        # 292# P0
    ev_offset_mult_applied: float | None = None,   # 292# P0
    decision_path: str | None = None,              # 292# P0
) -> "FillRecord":
    return build_skip_fill_record(
        # ... existing ...
        ev_score_pretrade=ev_score_pretrade,
        ev_offset_mult_applied=ev_offset_mult_applied,
        decision_path=decision_path,
    )
```

---

## BS-4: ev_offset_mult_applied=None の曖昧性【MEDIUM】

**ファイル**: [fill_cycle_executor.py](scripts/v460/lib/fill_cycle_executor.py#L958-L990)

### 問題

`_ev_offset_mult_applied` は以下の 2 ケースで `None`:

1. **ev_score 自体が None**（ev_weighted 未計算）→ `decision_path = "primary_only"`
2. **`compute_ev_offset_multiplier` が `mult=1.0` を返した**（ev_score ≈ 0.0）→ `decision_path = "ev_no_change"`

ケース 2 では ev 経路は active だったが offset 変更なしを意味する。しかし `ev_offset_mult_applied = None` と記録されるため、**分析時に「ev 自体が無効だった」ケースと区別不能**。

根拠コード（[_apply_offset_multiplier L519-520](scripts/v460/lib/fill_cycle_executor.py#L519-L520)):

```python
if offset_mult == 1.0:
    return order_price, effective_offset_ratio, None, None  # ← applied_mult = None
```

### 修正案

`_apply_offset_multiplier` で `mult=1.0` の場合に `(order_price, ratio, 1.0, 0.0)` を返す、もしくは executor 側で `decision_path == "ev_no_change"` のとき `ev_offset_mult_applied = 1.0` をセットする:

```python
if _applied_mult is not None and _delta is not None:
    _ev_offset_applied = True
    _ev_offset_mult_applied = _applied_mult
elif _ev_mult == 1.0:
    _ev_offset_mult_applied = 1.0  # "ev計算済み・変更なし" を明示
```

---

## BS-5: SkipGate Blocked Reprice → 注文消滅【MEDIUM】

**ファイル**: [order_monitor.py](scripts/v460/lib/order_monitor.py#L493-L502)

### 問題

reprice フロー内で SkipGate が reprice を拒否（`reprice_gate_skipped=True`）した場合:

```python
# L479: cancel 実行済み
# L493-498: SkipGate check
if reprice_gate_skipped:
    cancel_reason_poll = "stale_skip_gate_blocked"
    break  # ← 再発注なし。注文はキャンセル済みで板から消滅
```

注文はキャンセル済みだが新規発注されない。サイクルは `cancel_reason="stale_skip_gate_blocked"` で終了。

### 判断

これは **設計意図** と判断する。SkipGate が「現在の市場状態で発注すべきでない」と判断しているため、再発注しないのは合理的。L568 のコメントにも「stale_skip_gate_blocked は既にキャンセル済み」と明記。

ただし、BS-1 と同じ構造（cancel-then-check）のため、**SkipGate check も cancel の前に移動すれば、不要な cancel（≒queue 喪失）を回避できる**。これは BS-1 修正と併せて改善すべき。

---

## BS-6: R-list Buy 側直結タスク状況【LOW】

**ファイル**: [280_ph2_rpt_position_and_remaining_tasks.md](docs/v460/280_ph2_rpt_position_and_remaining_tasks.md#L92-L106)

### 現状

Buy 側 PnL に直結する残タスク:

| ID | 内容 | 状態 | Buy 側インパクト |
|---|---|---|---|
| **R-1** | fill_test 168h 再実測 | 計測中 (~03-11) | ベースライン確立 |
| **R-3** | SkipGate 再訓練 (n≥500) | データ蓄積待ち | Buy AS 精度向上 |
| **R-5** | Volatility Guard 動的ゲーティング | 未実装 | Buy VG 過剰発動の抑制 |
| **R-12** | order_monitor except narrow化 | 未着手 | BS-1 修正時に併せて改善可 |

R-3 は 292# P0（`ev_score_pretrade` 記録）と直結 — 新フィールドのデータが蓄積されれば pretrade features での再訓練が可能になる。R-1 完了を待つのは正しい。

---

## 補足: 290#/291# レビューアが指摘した項目の実装確認

| 指摘 | 実装 (292#) | コード確認 | 残懸念 |
|---|---|---|---|
| `ev_score_pretrade` 記録 | ✅ FillRecord に追加 | [fill_quality.py L143](ztb/metrics/fill_quality.py#L143) | BS-3 (skip record 漏れ) |
| `ev_offset_mult_applied` 記録 | ✅ FillRecord に追加 | [fill_quality.py L144](ztb/metrics/fill_quality.py#L144) | BS-4 (None 曖昧性) |
| `decision_path` 記録 | ✅ FillRecord に追加 | [fill_quality.py L145](ztb/metrics/fill_quality.py#L145) | BS-3 (skip record 漏れ) |
| Reprice deadband | ✅ `stale_reprice_min_delta_jpy` | [order_monitor.py L527](scripts/v460/lib/order_monitor.py#L527) | **BS-1 (cancel 順序バグ)** |
| Queue position 保護 | ⚠️ 意図通り動作していない | — | BS-1 が根本原因 |

---

## 推奨アクション優先度

1. **BS-1 修正** (CRITICAL): deadband check を cancel の前に移動。R-1 計測期間内でも hotfix として許容すべき。
2. **BS-3 修正** (HIGH): `_make_skip_fill_record` に 3 フィールドを追加。FillRecord スキーマ変更なし（既存フィールドの注入漏れ修正のみ）。
3. **BS-4 修正** (MEDIUM): `ev_offset_mult_applied = 1.0` の明示記録。分析基盤の正確性向上。
4. **BS-2 検討** (MEDIUM): R-1 完了後のデータ分析フェーズで buy/sell 分離の必要性を判断。
