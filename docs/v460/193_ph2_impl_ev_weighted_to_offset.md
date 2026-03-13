# 193# ev_weighted → offset 修飾子変換

## 概要

192# レビュー + Gemini §9.4 の指摘を受け、**ev_weighted を二値ゲート（PASS/SKIP）から連続的なオフセット修飾子に変換**した。

### 解決した問題

| 問題 | 192# 指摘元 | 対策 |
|---|---|---|
| ev_weighted が 100% 負 → 全て SKIP | §5.2 "distributed ownership" | SKIP しない、代わりに offset を調整 |
| 安全弁が強制 PASS → 大損 | Gemini §9.2 "safety valve is suicide" | 安全弁廃止、EV→offset 連続調整 |
| balance_forced + ev_pass → -18.56bps | Gemini §9.3 "balance loop insanity" | 負 EV 時は保守的な価格で取引 |
| ev_score と price offset に接続なし | **191#/192# 両方の盲点** | 新規実装: EV→offset pipe |

### 191#/192# どちらも見落としていた構造的盲点

**ev_weighted の score は maker_price の offset 計算に一切接続されていなかった。**

- ev_weighted は `predicted_pnl_bps` を計算するが、それは PASS/SKIP の二値判定にしか使われない
- `maker_price.py` の offset パイプライン（regime_boost, spread_adaptive, volatility_guard, imbalance_risk）に EV 情報は流入しない
- つまり「エッジがある/ない」という情報が価格設定にフィードバックされていない

**これが最大の構造的欠陥**：ML が「エッジなし」と判定しても、価格は変わらず同じ場所に発注される。

## 設計

### 新アーキテクチャ

```
maker_price.compute()
  → order_price (base) + spread + effective_offset_ratio
    ↓
skip_gate.evaluate()
  → primary ML PASS/SKIP + ev_score
    ↓
if PASS and ev_score exists:
  ev_offset_mult = clamp(1.0 + sensitivity × ev_score, min, max)
  effective_offset *= ev_offset_mult
  order_price post-hoc adjusted
    ↓
place order at adjusted price
```

### ev_score → offset 乗数の変換

```
ev_offset_mult = 1.0 + sensitivity × ev_score
                 (clamped to [min_mult, max_mult])
```

| ev_score | sensitivity | raw_mult | clamped | 意味 |
|---|---|---|---|---|
| +2.0 | 0.05 | 1.10 | 1.10 | エッジあり → 10%積極的 |
| 0.0 | 0.05 | 1.00 | 1.00 | ニュートラル → 変更なし |
| -3.0 | 0.05 | 0.85 | 0.85 | エッジなし → 15%保守的 |
| -10.0 | 0.05 | 0.50 | 0.50 | 非常に悪い → 50%保守的 (下限) |

- offset が **減少** → 価格が mid_price から離れる → **fill 確率は下がるが AS リスクも下がる**
- offset が **増加** → 価格が mid_price に近づく → **fill 確率は上がるがリスクも上がる**
- **カタストロフィック EV**（< -8.0）→ 依然ハード SKIP（emergency threshold）

### 旧モードとの比較

| 状況 | 旧モード（gate） | 新モード（offset） |
|---|---|---|
| EV = -2.0 | SKIP（取引不可） | PASS + offset ×0.90 |
| EV = -5.0 | SKIP → 安全弁で PASS → 大損の可能性 | PASS + offset ×0.75（保守的） |
| EV = -10.0 | SKIP → 安全弁で PASS → さらなる大損 | EMERGENCY SKIP（< -8.0） |
| EV = +1.0 | PASS（通常価格） | PASS + offset ×1.05（やや積極的） |

## 変更ファイル

### 1. `scripts/v460/lib/fill_config.py`
- `SkipGateResult.ev_score: Optional[float]` 追加
- 新 config フィールド:
  - `skip_gate_ev_as_offset_enabled: bool = False`
  - `skip_gate_ev_offset_sensitivity: float = 0.05`
  - `skip_gate_ev_offset_min_mult: float = 0.5`
  - `skip_gate_ev_offset_max_mult: float = 1.5`
  - `skip_gate_ev_emergency_skip_threshold: float = -8.0`
- YAML skip_gate セクションにマッピング追加

### 2. `scripts/v460/lib/skip_gate_evaluator.py`
- `_try_ev_weighted_decision()`: `ev_as_offset_enabled=True` 時に `_ev_weighted_as_offset()` にディスパッチ
- 新メソッド `_ev_weighted_as_offset()`:
  - `should_skip=False` 固定（emergency 以外）
  - ev_score を `predicted_pnl_bps` に格納
  - 安全弁カウンタ不使用
  - offset 乗数をログ出力
- `evaluate()`: ev_as_offset モード時は primary decision を保持、ev_score のみ抽出

### 3. `scripts/v460/lib/fill_cycle_executor.py`
- SkipGate PASS 後の EV offset 価格調整ブロック追加
- ev_score × sensitivity → offset 乗数 → order_price の delta 計算
- buy/sell 方向に応じた価格調整（buy: delta 加算、sell: delta 減算）

### 4. `configs/v460/fill_test.yaml`
- `ev_as_offset_enabled: true` 有効化
- `ev_offset_sensitivity: 0.05`
- `ev_offset_min_mult: 0.5`
- `ev_offset_max_mult: 1.5`
- `ev_emergency_skip_threshold: -8.0`

### 5. `tests/unit/v460/test_193_ev_offset.py` (新規)
- 23 テスト:
  - `TestEvWeightedAsOffset`: 正/負/emergency/sell/disabled/no_alt
  - `TestEvOffsetMultiplier`: パラメトリックテスト (5 パターン) + ログ確認
  - `TestSkipGateResultEvScore`: フィールド存在確認
  - `TestEvOffsetConfig`: デフォルト値 + YAML parse
  - `TestBackwardCompatibility`: 旧モード挙動保持
  - `TestExecutorEvOffsetAdjustment`: 価格調整計算の正確性

### 6. `tests/unit/v460/test_188_split_evc_macro.py`
- mock config に 193# 新フィールド追加（後方互換性対応）

## テスト結果

```
2490 passed, 0 failed (v460 unit tests)
  - 新規 23 テスト (+23)
  - 既存 2467 テスト 全通過
```

## 192# レビュー指摘への対応状況

| 192# 指摘 | 対応 |
|---|---|
| §5.2 "ev_weighted を offset source に" | ✅ 実装済み |
| §5.3 "安全弁の削除" | ✅ offset モードでは無効化 |
| §5.4 "片側 balance 緩和の削除" | ✅ offset モードでは無効化 |
| Gemini §9.2 "safety valve is suicide" | ✅ offset モードで構造的に解消 |
| Gemini §9.4 "EV as offset source" | ✅ 実装済み |
| §5.1 "distributed ownership 統合" | 🔄 ev_weighted は offset に demote 済み、残る guard 統合は次段 |

## 今後の課題（next session で取り組む内容）

1. **velocity_skip のソフト化**: 現在ハードゲート → offset boost への変換検討
2. **B1' (ranging_buy_low_vol) の統合**: maker_price の low_vol_offset_boost と重複
3. **trending_sell_skip の簡素化**: bypass 条件（HF4, inv_bypass, safety_valve）が複雑
4. **balance_forced 問題**: JPY 枯渇→forced sell→損失パターンの根本対策
5. **sensitivity パラメータの最適化**: バックテストによる最適値探索
