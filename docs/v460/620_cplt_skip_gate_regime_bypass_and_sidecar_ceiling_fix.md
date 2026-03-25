# 620# skip_gate regime_thresholds bypass 修正 + sidecar ceiling 適用順序修正

- **日付**: 2026-03-25
- **著者**: Copilot
- **コミット**: `9b9daf5fd`
- **種別**: fix (critical bug × 2)
- **目的**: fill test 損失の根本原因であった 2 件のバグを修正し、即時再起動で本番反映

---

## §1 背景

fill test の週次データ (3/17–3/23) で **avg EV/cycle = -0.12 bps**（負の期待値）が判明。
4741 fills / avg PnL = -0.27 bps、特に **ranging_buy が -734.89 bps（全損失の 54.6%）** と突出。

ユーザ指摘「大体バグだったりするケースも多かった」に従い、パッチ的 IF 文追加 (536# で禁止) ではなく **根本原因のバグ調査** を実施。2 件の critical bug を発見・修正した。

---

## §2 Bug 1: skip_gate regime_thresholds bypass（CRITICAL）

### 症状

`skip_gate` の `adaptive_threshold=true` 時、**regime_thresholds で設定した per-regime 安全閾値がすべて無視**されていた。

### 原因

`_calibrate_pnl_threshold()` は per-side グローバル閾値 (`_pnl_threshold_buy` / `_pnl_threshold_sell`) を EWMA で適応的に更新するが、この値は **全レジームで共有** されていた。

```python
# 修正前: adaptive が regime_thresholds を完全に上書き
new_threshold = ewma(current, target_skip_rate)
self._pnl_threshold_buy = new_threshold  # ranging=0.1 も trending=-0.1 も無視
```

`evaluate()` で `regime_thresholds[regime]` を参照する際、adaptive 値がすでに regime 固有値を下回っていても**そのまま使用**されるため、ranging レジームの厳しい閾値 (`0.1`) が `-0.3` 程度まで緩和され、本来スキップすべき悪条件トレードが通過していた。

### 修正内容

**ファイル**: `ztb/ml/skip_gate.py`

1. `evaluate()` メソッドで `regime_floor` を抽出し、`_calibrate_pnl_threshold()` に渡す
2. `_calibrate_pnl_threshold()` に `regime_floor: float | None` 引数を追加
3. 適応更新後に `max(regime_floor, adaptive_threshold)` を適用

```python
# 修正後: adaptive は regime_floor を下回れない
new_threshold = ewma(current, target_skip_rate)
self._pnl_threshold_buy = new_threshold  # 収束追跡用は raw 値を保持
effective = max(regime_floor, new_threshold) if regime_floor is not None else new_threshold
```

`regime_floor=None`（regime_thresholds 未設定）の場合は従来通り制約なし → 後方互換性を維持。

### 影響

ranging レジームで PnL threshold が `0.1` を下回らなくなり、**低品質 ranging_buy トレードの大半がスキップ対象** になる。週次データの ranging_buy -734.89 bps の大幅改善が期待される。

---

## §3 Bug 2: sidecar ceiling bypass

### 症状

sidecar（SAC RL バイアス注入）が **offset ceiling を事実上バイパス** していた。

### 原因

multiplicative_pipeline / offset_pipeline の両方で、処理順序が：

1. ceiling clamp（ratio 空間で `effective_offset_ratio` を制限）
2. **sidecar injection（JPY 空間で `order_price` を直接変更）** ← ceiling の後に見えるが…

実際のコードでは sidecar が ceiling の **前** に `order_price` を修正していた。ceiling は `effective_offset_ratio` を clamp するが、price 再計算時に「すでに sidecar delta が乗った `order_price`」を基準にするため、**sidecar の JPY delta は ceiling 制限を生き残って** 最終価格に反映されていた。

### 修正内容

**ファイル**:
- `scripts/v460/lib/multiplicative_pipeline.py`
- `scripts/v460/lib/offset_pipeline.py`

両ファイルで sidecar injection ブロックを **ceiling clamp の後** に移動。

```python
# 620# sidecar injection: ceiling clamp の後に適用
# → ceiling で制限された最終 order_price に対して sidecar delta を加算
```

### 影響

sidecar の `max_boost_bps=0.20` が ceiling (`buy=0.35`, `sell=0.40`) を超えて価格を歪めることがなくなった。

---

## §4 検証

| 項目 | 結果 |
|------|------|
| skip_gate ユニットテスト (199件) | ✅ 全パス |
| 全テストスイート (2225件) | ✅ 全パス、127 skipped |
| mypy 型チェック | 既存通り |

---

## §5 Non-bugs（調査で正常と確認）

以下は疑われたが正常と確認:

| 項目 | 結論 |
|------|------|
| EWMA 更新式 | `α·new + (1-α)·old` — 正しい |
| EWMA clamp before update | 更新前 clamp は意図通り |
| `inv_relaxation` 符号 | 正常 |
| PnL 計測 (mark-to-market) | 正常 |
| 乗算チェーン (9段) | 正常 |

---

## §6 536# 原則の遵守

調査過程で「**hard_skip_utc_hours に固定時間帯を追加**」する案が浮上したが、ユーザが即座に「536# と同じ轍ではないか」と指摘。

536# (風水渙) の教訓：
> 固定値 IF 文によるパッチ当ては技術的負債を積み上げるだけ。動的マイクロストラクチャ信号 (OFI, VPIN) で対応すべき。

これに従い、固定時間帯スキップは採用せず、**バグ修正による根本対処** を選択した。

---

## §7 デプロイ

- **コミット**: `9b9daf5fd` (`fix(620#): skip_gate regime_thresholds bypass + sidecar ceiling ordering`)
- **再起動**: `ops/windows/hot_swap_restart.ps1` により PID 38020 で稼働開始
- **retrain_scheduler**: PID 37852 で再起動済み
