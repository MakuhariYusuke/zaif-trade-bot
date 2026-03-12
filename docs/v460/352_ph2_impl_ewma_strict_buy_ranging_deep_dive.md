# 352# EWMA 厳密化実装 + buy_ranging 深堀り分析

**日付**: 2026-03-09  
**対象**: 350# Codex レビュー / 351# Gemini 3.1 Pro レビューの合意事項実装、buy 側強化検討  
**前提**: Bot PID 89776 稼働中（09:51 JST～）、349# EWMA 修正から 7 時間経過  
**コミット**: 70823fd67 → 本セッション

---

## §1 350# / 351# 合意サマリー

350# (Codex) と 351# (Gemini 3.1 Pro) の外部 AI レビューで **共通して P0 と認定** された事項:

| # | 対象 | 350# 指摘 | 351# 指摘 | 合意レベル |
|---|------|-----------|-----------|------------|
| P0-1 | `_rebuild_ewma_from_history()` | F1: 平均 seed + 全 replay は数学的不正確。元の `track()` と一致しない | Q1: Look-ahead bias / Double counting。未来データがシードに混入 | **完全一致** |
| P0-2 | TIME LIMIT `threshold * 0.8` | F4: 固定係数のハードコード、regime/side 非依存 | Q2: `threshold_offset_bps` 無視で即再 kill デッドロック復活リスク | **方向一致**（351# がより具体的） |
| P0-3 | テスト不足 | F2: 旧 state フォールバックのテストが `is not None` 止まり | (暗黙 — Q1 の修正には正確性テスト必須) | **暗黙合意** |

### 350# 独自指摘（F3–F7）

- **F3 (HIGH)**: 349# 因果性主張 — 349# の結論を「liveness 回復」に留めるべき
- **F5 (MED)**: `min_lot=0.001` floor 張り付き — ロット制御の実効性は限定的
- **F6 (MED)**: `balance_forced` 撤廃の裏面 — 在庫偏り長期化の機会損失
- **F7 (MED)**: 主戦場は `buy_ranging` — sell は 7 日集計で中立復帰済

### 351# 独自指摘（盲点 1–3）

- **盲点 1**: Ranging ≠ 対称。ディストリビューション局面の隠蔽。VPIN/OBI 事前退避の必要性
- **盲点 2**: EWMA イベント依存更新の罠。kill 中は更新なし → stale 凍結 → TIME LIMIT 必須化
- **盲点 3**: `balance_forced` 撤廃 → 資本非効率。A-S 型非対称 Skewing が本質的解

---

## §2 P0-1: EWMA rebuild 厳密化

### 問題

`_rebuild_ewma_from_history()` は旧 state からの復元時に呼ばれるが、`track()` と異なる初期化式を使っていた:

```python
# 旧実装 (349#)
ewma = sum(history) / len(history)  # 平均でシード ← Look-ahead bias
for v in history:                    # 全履歴 replay ← Double counting
    ewma = alpha * v + (1 - alpha) * ewma
```

数値例: `[1.0, -1.0, 0.5, -0.5, 0.0, -10.0], α=0.1`
- `track()` 順次適用: `-0.47917`
- 旧 rebuild: `-1.89635`
- **差分: -1.417** (kill 判定を有意に変える)

### 修正

```python
# 352# 修正
ewma = history[0]           # 初回 seed = track() と同一
for v in history[1:]:        # history[1:] のみ replay
    ewma = alpha * v + (1 - alpha) * ewma
```

`track()` の初回が `pnl_bps` 直接代入、2 回目以降が EWMA 更新式であることと完全一致。

### track() 初回シードの修正

349# P1 で導入した history 平均シードも同一の look-ahead bias:

```python
# 349# P1 (撤回)
if len(self._pnl_history) > 1:
    self._ewma_value = sum(self._pnl_history) / len(self._pnl_history)

# 352# 修正（直接代入に戻す）
self._ewma_value = pnl_bps
```

**根拠**: 単一外れ値による毒化防止という 349# P1 の意図は理解できるが、それは kill threshold が担うべき責務。シードに未来データ（の近似としての history 平均）を混入させるのは情報漏洩。

---

## §3 P0-2: TIME LIMIT effective_threshold 修正

### 問題

TIME LIMIT 解除時の EWMA リセット先が base `threshold` のみで計算され、`threshold_offset_bps`（在庫連動緩和）を無視:

```python
# 旧実装
threshold = self._config.threshold_bps
if regime and regime in self._config.regime_thresholds:
    threshold = self._config.regime_thresholds[regime]
reset_target = threshold * 0.8  # ← offset_bps 未反映
```

**デッドロック条件**: `threshold=-0.3`, `offset=0.2` → `effective=-0.5` だが `reset=-0.24`。
`effective < reset` (kill 圏内にリセットされる) → 即再 kill。

### 修正

```python
# 352# 修正
effective_threshold = self._config.threshold_bps
if regime and regime in self._config.regime_thresholds:
    effective_threshold = self._config.regime_thresholds[regime]
if threshold_offset_bps != 0.0:
    effective_threshold -= threshold_offset_bps  # check_kill と同一ロジック
reset_target = effective_threshold * 0.8
```

これにより `reset_target` は常に `effective_threshold` よりも kill 圏から遠い位置にリセットされる。

---

## §4 P0-3: テスト拡充 (13 → 18 tests)

### 新規・修正テスト一覧

| テスト | 種別 | 検証内容 |
|--------|------|----------|
| `test_import_missing_ewma_rebuilds` | 修正 | rebuild 後の EWMA が `track()` 参照値と完全一致（旧: `is not None` のみ） |
| `test_second_track_applies_ewma_update` | 修正 | 初回 seed が直接代入、2 回目が EWMA 更新式（旧: 平均シード検証） |
| `test_time_limit_uses_effective_threshold_with_offset` | 新規 | offset=0.2 反映で reset_target が変化 |
| `test_time_limit_regime_plus_offset` | 新規 | regime threshold + offset の複合 |
| `TestExactRebuild.test_rebuild_matches_track_sequence` | 新規 | 10 要素で rebuild と track() の EWMA 完全一致 |
| `TestExactRebuild.test_rebuild_single_element` | 新規 | 単一要素でも rebuild = track() |
| `TestExactRebuild.test_rebuild_vs_track_long_sequence` | 新規 | 150 要素ランダム列で rebuild = track() |

### テスト結果

```
tests/unit/v460/test_349_ewma_fixes.py: 18 passed in 0.83s
tests/unit/ (kill 関連全体):            30 passed, 7 skipped, 0 failed in 37.90s
```

---

## §5 buy_ranging 深堀り分析

### 5.1 本日の buy/sell 内訳 (09:51 JST～, 352# P0 修正後)

| 指標 | BUY | SELL |
|------|-----|------|
| fills | 24 | 25 |
| wins (>0bps) | 13 (54.2%) | 13 (52.0%) |
| avg PnL | **+0.87bps** | **+0.77bps** |
| total PnL | +20.8bps | +19.2bps |

**注目**: buy が sell と同等以上のパフォーマンス。349# の 7 日集計 (buy=-97.3bps) から劇的改善。

### 5.2 buy 損失パターン分析

| Cycle | wait(s) | pnl(bps) | パターン | 状況 |
|-------|---------|----------|----------|------|
| 8714 | 6.0 | -2.19 | fast fill | ev_score 不明、通常損失 |
| 8729 | 5.8 | -2.98 | fast fill | 連続 buy の 2 連敗 |
| 8731 | 5.8 | -1.62 | fast fill | 軽微、許容範囲 |
| **8743** | **5.7** | **-9.07** | **fast + toxic** | **ev_score=-0.824, offset_mult=0.959。Toxic fill veto 発動(3 cycle)。fast_fill_defense 発動** |
| 8764 | 38.2 | -2.35 | long wait | offset ceiling 0.15 でクランプ、ev_score=-0.432 |
| 8786 | 28.3 | -2.84 | long wait | loss_boost 1.147 適用中、ev_score=0.725（正のスコアだが損失） |
| **8792** | **39.1** | **-4.18** | **long wait + toxic** | **VPIN=0.83, vol_guard=2x offset。Toxic fill veto 発動** |
| 8802 | 5.8 | -3.44 | fast fill | 通常損失 |

### 5.3 損失の構造的分類

**パターン A: fast fill 逆選択 (8743)**
- wait=5.7s は「市場がこちらに向かってきている」= adverse selection の典型
- ev_score=-0.824 にもかかわらず offset_mult=0.959 で微調整止まり
- **根本原因**: ev_score の skip 感度が低すぎる（sens=0.050）、または ev_score が逆選択リスクを十分に反映していない

**パターン B: long wait + VPIN 高値 (8792)**
- VPIN=0.83 で vol_guard が offset を 2x に拡大 → それでも被弾
- 39.1s 待ちで約定 ＝ 市場が自分のレベルまで下がってきた（下落方向）
- **根本原因**: VPIN 高値 + long wait の組み合わせは「ゆっくり下落中の bid 被り」。offset 拡大だけでは不十分

**パターン C: ev_score 偽陽性 (8786)**
- ev_score=+0.725（正 ＝ 好材料）と判定されたが結果は -2.84bps
- **根本原因**: ev_score モデルの accuracy 問題。buy side での精度が sell と異なる可能性

### 5.4 buy 損失の定量的特徴

```
fast fill 損失 (<15s):  5 件, avg=-3.86bps, worst=-9.07bps
long wait 損失 (>=15s): 3 件, avg=-3.12bps, worst=-4.18bps
```

fast fill の方が damage が大きい (逆選択の直接被弾)。

---

## §6 buy 側強化の提案

### 6.1 即着手可能 (config 調整)

| 項目 | 現状 | 提案 | 根拠 |
|------|------|------|------|
| `buy_velocity_skip_threshold_bps` | -6.0 | -4.0 | 8743 (-9.07) は velocity_skip で回避可能だった可能性 |
| `fast_fill_defense` buy threshold | 10.0s | 8.0s | buy の fast fill (<6s) がほぼ全て adverse selection |
| ev_score sensitivity (buy) | 0.050 | 0.075 | buy で ev_score が有効に機能していない |

### 6.2 中期実装 (コード変更必要)

**A. VPIN + long wait の複合 skip**
```
if vpin > 0.70 and wait > 20s → buy skip
```
8764, 8792 のパターンをカバー。VPIN 高値・長待ちの組み合わせは「ゆっくり下落」の強いシグナル。

**B. 非対称 offset (351# 盲点 1 対応)**
ranging 判定下でも buy offset を sell offset より広く取る:
```
buy_offset = base_offset * (1 + vpin_markup)  where vpin_markup = max(0, vpin - 0.5) * 2
```
VPIN=0.7 で 40% 追加、VPIN=0.9 で 80% 追加。

**C. EWMA 時間減衰 (351# 盲点 2 対応)**
kill 中に時間経過で EWMA を中立方向へ自然減衰。TIME LIMIT ハック不要化:
```python
if self._kill_active and elapsed_since_last_fill > decay_start_sec:
    decay = min(1.0, (elapsed - decay_start) / decay_half_life)
    self._ewma_value *= (1.0 - decay * 0.5)
```

### 6.3 長期構造改善

**D. A-S 型在庫 Skewing (351# 盲点 3)**
`maker_price.py` に在庫量に応じた非対称 offset を組み込む。
在庫余剰側（売）は mid に寄せ、不足側（買）は遠ざける。
→ `balance_forced` の代替であり、受動的な在庫均衡メカニズム。

---

## §7 351# 市場理論盲点への評価

| 盲点 | 評価 | 対応 |
|------|------|------|
| 盲点 1: Ranging ≠ 対称 | **同意**。本日データでも long wait buy 損失は ranging 下落局面での被弾。VPIN は 0.79-0.83 と高め。 | §6.2-A/B で対処。VPIN + long wait 複合 skip 実装が最優先 |
| 盲点 2: EWMA 時間減衰 | **方向は同意** だが即実装は危険。TIME LIMIT は現状のフォールバックとして機能中。段階的に移行。 | §6.2-C は次セッション以降。352# では P0-2 の effective_threshold 修正で当面の安全性を確保 |
| 盲点 3: A-S Skewing | **理論的に正しい** が、maker_price.py の大改修が必要。収益インパクトの事前見積もりが困難。 | §6.3-D として長期計画に登録。バックテストでの検証後に着手 |

---

## §8 変更サマリー

### コード変更

| ファイル | 変更箇所 | 内容 |
|----------|----------|------|
| `ztb/risk/sell_dynamic_kill.py` | `track()` L258-268 | 初回 seed を `pnl_bps` 直接代入に戻す (349# P1 history 平均撤回) |
| `ztb/risk/sell_dynamic_kill.py` | `_rebuild_ewma_from_history()` L285-300 | `history[0]` seed + `history[1:]` replay に修正 |
| `ztb/risk/sell_dynamic_kill.py` | `check_kill()` TIME LIMIT L484-500 | `effective_threshold` (regime + offset) でリセット計算 |
| `tests/unit/v460/test_349_ewma_fixes.py` | 全体 | 13 → 18 テスト。exact rebuild 検証、effective_threshold 検証追加 |

### ドキュメント

| ファイル | 状態 |
|----------|------|
| `docs/v460/352_ph2_impl_ewma_strict_buy_ranging_deep_dive.md` | 新規 |

### 未着手（次セッション以降）

- [ ] VPIN + long wait 複合 buy skip (§6.2-A)
- [ ] EWMA 時間減衰 (§6.2-C)
- [ ] A-S 在庫 Skewing (§6.3-D)
- [ ] buy ev_score sensitivity 調整 (§6.1 — config 変更のみ、要検証)
