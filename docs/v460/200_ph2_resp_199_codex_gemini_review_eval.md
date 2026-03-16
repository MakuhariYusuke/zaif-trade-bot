# 200# 199 Codex/Gemini レビュー評価 + P0 実装計画

> **対象**: `199_ph2_rev_198_drawdown_and_hidden_risks.md` (Codex §1-5, Gemini §6)  
> **目的**: 両レビューの指摘を個別評価し、198# 提案 (A–I) との統合優先度を確定する

---

## 1. レビュー概観

### 1.1 Codex (§1–5)

198# の分析を妥当と認めた上で、**198# が見落とした 7 つの運用層・設計層の問題**を追加指摘。
定量ログ検証に基づく具体性が高く、優先度付けも明確。

### 1.2 Gemini (§6)

Codex 指摘を全面支持し、**市場理論** (Avellaneda-Stoikov, 一目均衡表時間論, 酒田五法) からの
構造的批判を追加。パラメータ調整ではなく「出血点の縫合」を最優先と主張。

---

## 2. 指摘事項の個別評価

### 凡例
- **同意**: 評価者が指摘に完全同意
- **条件付**: 方向は正しいが実装時に留意事項あり
- **保留**: 重要だが現フェーズでは対応見送り

---

### 2.1 [Codex §2.1] HALT 中 state 非保存 — 運用監視破綻

| 項目 | 内容 |
|---|---|
| 評価 | **同意 — P0** |
| 198# 対応 | 未記載 (新規指摘) |
| 影響 | `saved_at` が古いまま → 外部監視で HALT/停止が区別不能 |
| 実装方針 | HALT 分岐 (L422-434) で `_state_persistence.save()` を `progress_log_interval` ごとに呼ぶ |
| 工数 | 低 (10行未満の追加) |
| Gemini見解 | 直接言及なし (Codex 全面支持の一環) |

### 2.2 [Codex §2.2] daily_drawdown_halt レコードの fill_records 汚染

| 項目 | 内容 |
|---|---|
| 評価 | **条件付 — P2** |
| 198# 対応 | 未記載 (新規指摘) |
| 影響 | JSONL の 54% が non-trade レコード → 分析指標の歪み |
| 実装方針 | halt レコードを別ファイルに分離、または分析時フィルタの標準化 |
| 判断理由 | 正しい指摘だが、現行分析スクリプトは side="none" を既に除外。緊急度は低い |
| Gemini見解 | 直接言及なし |

### 2.3 [Codex §2.3] VG と velocity_offset の符号逆転

| 項目 | 内容 |
|---|---|
| 評価 | **同意 — P1** |
| 198# 対応 | 未記載 (新規指摘) |
| 影響 | maker_price と skip_gate が逆符号 velocity で独立に offset 調整 → 認知分裂 |
| 実装方針 | velocity SSOT (Single Source of Truth) 化。maker_price.py の VG 計算を正として skip_gate 側を合わせる |
| Gemini見解 | §6.1 で強く支持。「耳を二つ持ち、都合よく解釈して逆選択に突っ込んでいる」 |
| 注釈 | 実装影響範囲が広いため、P0 修正後に慎重に着手 |

### 2.4 [Codex §2.4] ev_as_offset が負 EV を通しすぎ

| 項目 | 内容 |
|---|---|
| 評価 | **条件付 — P2** |
| 198# 対応 | 未記載 (新規指摘) |
| 影響 | 負 EV トレードが offset のみで通過し損失拡大 |
| 実装方針 | 3 段階化 (軽度=offset, 中度=offset+cooldown, 重度=skip) |
| 判断理由 | 193# の方向は正しかった。閾値チューニング依存のため、十分な検証データ蓄積後に実施 |
| Gemini見解 | 直接言及なし (Codex 全面支持) |

### 2.5 [Codex §2.5] Gate 方向ミスアライン (buy 止め / sell 通し)

| 項目 | 内容 |
|---|---|
| 評価 | **条件付 — P1** |
| 198# 対応 | 提案 D (trending sell reprice 禁止) が部分対応 |
| 影響 | 損失 74% が sell なのに hard skip 6 件中 5 件が buy |
| 実装方針 | sell 側 gate の再調整。ただし 198# A (reprice 方向ガード) が主因であり、A 修正後に再評価 |
| Gemini見解 | §6.4 で sell サイド封鎖を提案 |

### 2.6 [Codex §2.6] postonly_guard による価格決定の責務逆転

| 項目 | 内容 |
|---|---|
| 評価 | **同意 — P1** |
| 198# 対応 | 提案 B (offset 保全) |
| 影響 | offset パイプライン全体が postonly_guard の snap で無効化 |
| 実装方針 | Codex 提案: crossing 検出 → 最新板基準で offset 再計算 → post-only 満足価格を導出。Gemini は skip (発注取消) を推奨 |
| 判断差異 | Codex=再計算、Gemini=skip。**Gemini の skip 方式が安全**。crossing 時は skip + cooldown が妥当 |

### 2.7 [Codex §2.7] low_vol_boost 実質定数化

| 項目 | 内容 |
|---|---|
| 評価 | **同意 — P2** |
| 198# 対応 | 提案 C (比例スケーリング) |
| 影響 | vol_ratio 0.15–0.27 << threshold 0.75 → boost x1.4 が常時発動 |
| 実装方針 | 198# 提案 C の通り、比例化 or 閾値再較正 |
| Gemini見解 | 直接言及なし |

---

## 3. 198# 提案 (A–I) × レビュー統合マトリクス

| 198# | 提案内容 | Codex 評価 | Gemini 評価 | 統合優先度 |
|---|---|---|---|---|
| **A** | stale reprice 方向ガード | §4.1 P0 #1 — 全面支持 | §6.2 — MM理論から逆選択特攻阻止 | **P0-1** |
| **B** | postonly_guard offset 保全 | §2.6 — 再計算方式推奨 | §6.3 — skip方式推奨 | **P1** (A後に着手) |
| **C** | low_vol_boost 比例化 | §2.7 — 定数化確認 | 言及なし | P2 |
| **D** | trending sell reprice 禁止 | §2.5 — 部分支持 | 言及なし | P1 (A で大半解消) |
| **E** | balance_forced cooldown | 言及なし | 言及なし | P3 |
| **F** | soft lot 半減バグ | 暗黙支持 (§2.4, §4.2) | §6.4 — 代替策提案 | **P0-2** |
| **G** | sell PnL wait regime連動 | 言及なし | 言及なし | P2 |
| **H** | regime 感度改善 | 言及なし | 言及なし | P3 |
| **I** | reprice offset 再計算 | §2.6 と重複 | §6.3 と重複 | P1 (B に統合) |

### Codex 新規提案

| ID | 提案内容 | 統合優先度 |
|---|---|---|
| **J** | HALT 中 state 保存 | **P0-3** |
| **K** | halt レコード分離 | P2 |
| **L** | velocity SSOT 化 | P1 |
| **M** | ev_as_offset 3段階化 | P2 |

### Gemini 新規提案

| ID | 提案内容 | 統合優先度 |
|---|---|---|
| **N** | soft drawdown 時 interval 3倍 or side封鎖 | **P0-2** (F と統合) |

---

## 4. 統合 P0 実装計画 (200# スコープ)

### P0-1: stale_order reprice 不利方向ガード (198# A)

**ファイル**: `scripts/v460/lib/order_monitor.py` L376-383

**現状**:
```python
is_drifting_away = (
    (side == "buy" and current_mid > mid_at_order)
    or (side == "sell" and current_mid < mid_at_order)
)
if drift_bps >= _stale_drift and is_drifting_away:
    # cancel & reprice
```

**修正**: 不利方向 drift 時は reprice ではなく cancel-only
```python
is_drifting_away = (
    (side == "buy" and current_mid > mid_at_order)
    or (side == "sell" and current_mid < mid_at_order)
)
if drift_bps >= _stale_drift and is_drifting_away:
    # cancel-only: 不利方向への追随は逆選択リスク
    cancel_reason_poll = "stale_adverse_drift"
    break
```

**根拠**: Codex P0 #1, Gemini §6.2 (MM理論: 逆選択特攻阻止)  
**推定効果**: +23bps/day (Cycle 5229 -23.32bps の完全回避)

### P0-2: soft lot 半減バグ修正 + interval 延長代替 (198# F + Gemini N)

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` L875-882

**現状**: `max(order_quantity, current_lot/2)` = `max(0.001, 0.0005)` = 0.001 (不変)

**修正**: lot 半減不可の場合、cycle interval を 3 倍に延長
```python
new_lot = self._current_lot / 2
if new_lot >= self.config.order_quantity:
    self._current_lot = new_lot
else:
    # 最小ロット: lot 半減不可 → interval 延長で exposure 削減
    self._soft_drawdown_interval_multiplier = 3.0
```

**根拠**: 198# F (確認済バグ), Gemini §6.4 (interval延長提案)

### P0-3: HALT 中 state 定期保存 (Codex J)

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` L422-434

**修正**: HALT 分岐内で cycle_count が progress_log_interval の倍数なら state 保存

---

## 5. 両レビューの総合評価

### 5.1 Codex

- **強み**: 実ログ・コードの定量検証に基づく具体性。新規発見 4 件 (J-M) は全て妥当
- **弱み**: 特になし。優先度付けも適切
- **評価**: ★★★★★ — 198# の盲点を正確に突いた高品質レビュー

### 5.2 Gemini

- **強み**: 市場理論フレームワークからの構造批判。「なぜそれが間違いか」の理論的裏付けが強い
- **弱み**: 一部過激 (postonly snap → 即 skip は taker 不利市場で機会損失の可能性あり)
- **評価**: ★★★★☆ — 方向提示として極めて有用。実装時は Codex の漸進的アプローチとの折衷が必要

### 5.3 意見の分岐点

| 論点 | Codex | Gemini | 採択 |
|---|---|---|---|
| postonly crossing 時 | 板基準で offset 再計算 | skip (発注キャンセル) | **Gemini 寄り**: skip + next cycle で再試行 |
| soft drawdown 時 | lot 半減 (min lot 対応なし) | interval 延長 or side 封鎖 | **Gemini**: interval 延長を採用 |
| 全体方針 | 漸進的修正 | 出血点の縫合を最優先 | **折衷**: P0 は出血点縫合、P1 以降は漸進 |
