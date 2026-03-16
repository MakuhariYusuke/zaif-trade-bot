# 431# Clamp Observability + Fill Data 分析

> **種別**: ph2 impl  
> **コミット**: `e63295439` (impl), `fd1ada321` (doc + self-review fix)  
> **前提**: 427#–430#（世界クオンツ文化・Sidecar/Clamp レビュー群）  
> **fill_test PID**: 39220（ホットスワップ適用済み）

---

## 背景

427#–430# の 4 文書を受け、ph2 fill_test への適用可能性を検証。
428# が警告した **"Clamp-Driven Development"**（全注文が ceiling で切り詰められ、offset chain が装飾的になる問題）を定量データで裏付け、430# P3 推奨の clamp observability を実装した。

---

## fill_records データ分析結果（2026-03-12〜14, 1617 レコード）

### サマリ

| 指標 | 値 |
|---|---|
| 総レコード | 1,617 |
| Filled | 505 (31.2%) |
| Skipped | 176 (10.9%) |
| Cancelled | 1,112 (68.8%) |
| **Buy ceiling clamp rate** | **347/347 (100%)** |
| **Sell ceiling clamp rate** | **0/332 (0%)** ※後述の self-review 訂正参照 |
| Buy AS rate | 22.6% |
| Sell AS rate | 28.9% |
| Buy 30s PnL (mean) | **−0.53 bps** |
| Sell 30s PnL (mean) | +0.01 bps |
| EV Score × 30s PnL 相関 | **0.0174** (ほぼゼロ) |
| Sidecar 有効レコード | **0/1,617** |

### 重要所見

1. **Buy Ceiling clamp rate 100%** — buy 側の全非 skip 注文が ceiling (0.20) に切り詰め。offset chain（base→regime→VG→kyle→amihud→EV mult）は buy 側では **完全に装飾的**。
2. **Sell Ceiling clamp rate 0%** — sell ceiling (0.50) は spread_adapt で自然到達 (0.50) し、306# clamp は `> ceil` 条件のため発火せず。EV mult (0.997) で 0.498 に下がるため実質非 clamp。※初回分析 (analyze_deep.py) では `stages.get("ceiling", 0)` のデフォルト=0 により sell も 100% と誤報—self-review で訂正。
2. **EV Score 無効化** — 相関 0.0174。clamp が EV 調整を打ち消すため、EV mult の最適化効果がゼロ。
3. **VG boost パラドックス** — VG boost ≧ 2.0 の PnL: −0.39 bps、< 2.0: +0.03 bps。高 boost は保護意図だが ceiling で効果消失。
4. **Sidecar 完全停止** — retrain_scheduler 未起動 → signal 過期 → 0/1617。430# 指摘通り。
5. **Buy 側損失** — mean −0.53 bps。sell はほぼ breakeven。

### Offset chain 分析（Buy 側平均）

```
base: 0.0500
→ regime:     0.0689 (+38%)
→ vol_guard:  0.2202 (+220%) ← ここで ceiling(0.2000) 超過
→ final:      0.2341 (≫ ceiling)
→ clamped:    0.2000 (ceiling)
```

VG boost が原因で ceiling 超過するため、regime 以降の全調整が無意味。

### 時間帯・レジーム別 PnL

| レジーム | PnL (bps) |
|---|---|
| ranging | −0.26 |
| trending_down | **+0.70** |
| trending_up | **−1.65** |

| 時間帯 | Fill Rate |
|---|---|
| 03:00–08:00 | 56–66% (最良) |
| 10:00, 20:00 | 18–21% (最悪) |

---

## 実装: Clamp Observability (428#/430# P3)

### 変更内容

| ファイル | 変更 |
|---|---|
| `fill_loop_orchestrator.py` | `RunSessionState` に `clamp_fire_count`, `ceiling_check_count` 追加 |
| `orchestrator_post_cycle.py` | `_process_cycle_result`: 非 skip レコードで `effective_offset_used == ceiling` を検出しカウント |
| `orchestrator_post_cycle.py` | `_log_progress_and_adapt`: `[431# clamp] clampFires=X/Y (Z%)` を periodical ログ出力。rate ≧ 90% で WARNING |
| `test_421_final_clamp_deadlock.py` | `TestClampObservability` 3 テスト追加 |

### 検出ロジック（self-review 修正済み）

```python
# 431# SR-1 fix: skip_gate_skipped は bool|None — 明示的に is False で比較
if record.skip_gate_skipped is False and record.effective_offset_used is not None:
    st.ceiling_check_count += 1
    _ceil = self.config.resolve_offset_ceiling(record.side)
    if _ceil > 0 and abs(record.effective_offset_used - _ceil) < 1e-6:
        st.clamp_fire_count += 1
```

306# ceiling と 418# final_clamp の **両方** をカバー（`effective_offset_used` は最終適用値）。

### テスト結果

- `test_421_final_clamp_deadlock.py`: **48 passed** (45 既存 + 3 新規)
- 全回帰テスト: **2,179 passed** (INTERNALERROR は benchmark plugin の既知問題)

---

## 今後の優先事項（427#–430#統合ロードマップより）

| 優先度 | 項目 | 状態 |
|---|---|---|
| ~~P3~~ | ~~Clamp observability~~ | ✅ **431# 完了** |
| P2 | Buy ceiling 引上検討 (0.20 → 0.25) | ⏳ データ蓄積中 |
| P2 | Sidecar 蘇生 (retrain_scheduler 起動) | ⏳ S1/S1' 依存 |
| P2 | Sidecar v1→v2 切替 (比例変換, 1行変更) | ⏳ Sidecar 蘇生後 |
| P3 | `pre_clamp_offset × 30s_pnl` 条件付き相関分析 | ⏳ |
| P3 | EV Score 有効性の再評価（clamp 撤去後） | ⏳ ceiling 変更後 |

---

## 結論

**100% ceiling clamp rate は buy 側の fill_test 最重要構造問題**。buy 側では offset chain の全最適化・EV モデル・VG boost が ceiling で打ち消され、全注文が同一価格で出る。sell 側は ceiling には引っかからないが、spread_adapt で自然に ceiling 近傍 (0.498) に到達するため差別化余地は小さい。431# で clamp 発火率のリアルタイム可視化を実装し、同 self-review で `skip_gate_skipped` 真理値比較修正 + sell 誤報訂正 + 統合テスト追加。

---

## Self-Review 発見事項

### 修正済み

| # | 内容 | 重要度 |
|---|---|---|
| SR-1 | `not record.skip_gate_skipped` → `record.skip_gate_skipped is False` に修正。`bool\|None` 型で `not None = True` となり、ガードブロック records が filter を誤通過する問題。2次条件 (`effective_offset_used is not None`) が安全網として機能していたため実害なし | 中 |
| SR-2 | 「Sell ceiling clamp 100%」は分析スクリプトの誤報。`stages.get("ceiling", 0)` のデフォルト 0 により `final > 0` で常に true。ライブ 431# 検出は `resolve_offset_ceiling(side)` で正しく sell=0.50 を取得 | 高 |
| SR-3 | clamp 検出ロジックの統合テスト追加（カウンター算術のみ → 実レコードでの検出パス検証） | 中 |

### 既知の制約（修正不要）

| # | 内容 |
|---|---|
| KL-1 | `effective_offset_used` vs `offset_stages.final` の乖離 — EV mult が stages 記録後に適用されるため sell で 0.5 vs 0.498 の差異。431# は `effective_offset_used`（実注文価格）を使うため正しい |
| KL-2 | offset_stages の `ceiling` フィールドは 306# 発火時のみ記録。sell 側 (`0.50 == ceil`, not `>`) では未記録。分析スクリプトで要注意 |
| KL-3 | Clamp カウンターはセッション全体で累積（batch reset と独立）。他の RunSessionState カウンター（total_count, filled_count 等）と同一ライフサイクル |
