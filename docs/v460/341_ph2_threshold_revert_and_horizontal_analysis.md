# 341# 閾値復元・横展開・設計分析

## 概要

340# で修正した `threshold_offset_bps` 符号逆転バグ (286# 以降) に基づき、
符号バグの存在を前提にキャリブレーションされた閾値パラメータを元の値に復元する。
加えて横展開チェックと設計面・市場理論面からの課題網羅を行う。

---

## §1 売り方閾値の復元 (337# → 341# revert)

### 経緯

| 時系列 | sell_dk base | trending_up | trending_down | ranging | 根拠 |
|--------|-------------|-------------|---------------|---------|------|
| ~336# | -0.3 | -0.3 | -1.0 | -0.5 | データ較正値 |
| 337# | **-1.0** | **-0.8** | **-1.5** | **-1.5** | 売り崩壊対策 (符号バグ未発見時) |
| 341# | **-0.3** | **-0.3** | **-1.0** | **-0.5** | 340# 符号修正により根本原因解消 → 復元 |

### 論理

337# の閾値変更は「売り側 kill が過剰に発動する」問題への対策だった。
しかし 340# で発見された符号逆転バグ (`threshold += offset` → `threshold -= offset`)
が真の根本原因であり、閾値変更は症状への弥縫策にすぎなかった。

**符号バグ下の inv_relaxation の挙動**:
```
threshold = -0.3, offset = +0.3
buggy:  -0.3 + 0.3 = 0.0   ← 全 fill が kill される (壊滅的)
fixed:  -0.3 - 0.3 = -0.6  ← 適切に緩和 (意図通り)
```

符号修正後は inv_relaxation が正しく動作するため、元の較正値 (-0.3) で運用可能。
effective threshold range: **[-0.6, -0.3]** (在庫偏重時に緩和)

---

## §2 買い方閾値の復元 (336# → 341# revert) — 横展開

### 経緯

| 時系列 | buy_dk base | trending_down | high_vol | ranging | inv max_bps | 根拠 |
|--------|-------------|---------------|----------|---------|-------------|------|
| ~335# | -0.8 | -0.5 | -0.5 | (default) | 0.3 | データ較正値 |
| 336# | **-1.5** | **-1.0** | **-1.0** | **-2.0 NEW** | **0.5** | 333# buy 過剰抑制対策 |
| 341# | **-0.8** | **-0.5** | **-0.5** | **(削除)** | **0.3** | 符号修正により根本原因解消 → 復元 |

### 論理

336# の修正は 333# SHA-isolated 分析から buy 側の過剰抑制が #1 問題として特定されたことに基づく。
しかしこの過剰抑制も符号バグに起因していた:

```
threshold = -0.8, offset = +0.3
buggy:  -0.8 + 0.3 = -0.5   ← inv_relaxation が逆に厳格化 (根本原因)
fixed:  -0.8 - 0.3 = -1.1   ← 正しく緩和

# 336# で緩和した値 + 修正後の relaxation
threshold = -1.5, offset = +0.5
fixed:  -1.5 - 0.5 = -2.0   ← kill 機構が事実上死亡
ranging: -2.0 - 0.5 = -2.5   ← 完全に無意味
```

336# のまま運用すると **buy kill が機能しない** 状態になり、
逆選択損失の蓄積に対するガードが失われる。

復元後の effective threshold range: **[-1.1, -0.8]** (適切な kill 機能を維持)

---

## §3 横展開チェック結果

### skip_gate の offset ロジック — ✅ 問題なし

skip_gate (`scripts/v460/ml/skip_gate.py`) は独立した offset 体系を持つ:
- PnL mode: `threshold += offset` (正=厳格化) — kill 条件 `pred_pnl < threshold` に対して正しい
- AS mode: `threshold -= offset` (正=厳格化) — kill 条件 `pred_prob >= threshold` に対して正しい
- 設計意図と実装が一致。符号バグなし。

### DynamicKillManager — ✅ buy/sell 完全分離

- `SellDynamicKillManager` / `BuyDynamicKillManager` は別インスタンス・別 PnL 履歴
- `check_kill()` は単一メソッドで buy/sell 共有 → 340# 修正は両方に適用済み

### orchestrator_guards.py — ✅ offset 生成ロジック正常

- buy: `imbalance < 0` → 正の offset → `threshold -= offset` で緩和 ✓
- sell: `imbalance > 0` → 正の offset → `threshold -= offset` で緩和 ✓
- 方向性の整合性は正しい

---

## §4 設計面・市場理論面の課題発見

### Finding A: forced_switch 完全除外の副作用 — HIGH

**現状**: `_track_side_pnl()` で `balance_forced_switch=True` の fill を rolling PnL から完全除外
**問題**: データポイント欠損 → window が古い fill を含み続ける → kill 判定が遅延
**推奨**: 完全除外 → downweight (0.5) への変更。sell 側にも forced KPI 分離を追加。
**再起動前**: 不要 (337# の除外自体は汚染防止として正常機能)

### Finding B: inv_bypass ステップ関数 — MEDIUM

**現状**: imbalance=0.28 → inv_relaxation のみ、imbalance=0.30 → 完全バイパス (binary)
**問題**: 0.29→0.30 で sell 制御が不連続ジャンプ
**推奨**: 中期的に bypass を廃止し、inv_relaxation に統一 (graduated defense)
**再起動前**: 不要

### Finding C: skip_gate / dynamic_kill 二重抑制 — HIGH

**現状**: 両フィルターが独立動作、互いの状態を認識しない
**データ**: 3/6 (eb24cf4a) 実績で skip_gate=51, sell_dk=42 — multiplicative に sell を抑制
**問題**: kill → fill 減少 → skip_gate adaptive threshold が stale 化 → kill 解除後も抑制継続の可能性
**推奨**: kill 状態を skip_gate に通知するか、kill 中は adaptive threshold をフリーズ
**再起動前**: 不要 (現時点で致命的デッドロックは確認されず)

### Finding D: Count-based rolling window — MEDIUM

**現状**: 50 fill 固定ウィンドウ。fill 頻度が変動すると時間カバレッジが不均一。
**問題**: trending_up (fill 多) → 短時間のデータ、ranging (fill 少) → 長時間のデータ
**推奨**: EWMA (半減期パラメータ) への移行
**再起動前**: 不要

### Finding E: Sell 90s vs Buy 30s post_fill_wait — MEDIUM

**現状**: sell PnL は 90s 後に測定、buy PnL は 30s 後に測定
**問題**: 異なる時間 horizon の PnL を同一 kill 閾値スケールで評価
**推奨**: 168# の設計合理性は認めるが、kill 閾値解釈に影響することを認識
**再起動前**: 不要

### Finding F: Regime 検知のフィードバックループ — OK (対策済み)

**結論**: regime 更新は OB mid 依存で fill 非依存。kill 中も regime は正常更新される。

### Finding G: Daily Drawdown Guard とのリセット不整合 — MEDIUM

**現状**: DD guard は日替わりリセットだが kill cooldown は持ち越し
**推奨**: 意識的に分離されている設計。273# max_kill_duration_sec=1800 で 30分上限があり実害なし
**再起動前**: 不要

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | sell_dk 閾値復元、buy_dk 閾値復元、inv_relaxation max_bps 復元 |
| `scripts/v460/lib/fill_config.py` | buy_dk code default -1.5→-0.8、inv max_bps 0.5→0.3 |
| `tests/unit/v460/test_169_c1_c3_c4_config.py` | sell regime threshold assertions 更新 |
| `tests/unit/v460/test_337_sell_side_countermeasures.py` | max_bps 比較 `<` → `<=`、docstring 更新 |
| `tests/unit/v460/test_157_regime_features.py` | buy threshold assertions -1.5→-0.8 |
| `tests/unit/v460/test_286_comprehensive_resolution.py` | buy inv max_bps assertion 0.5→0.3 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | sell_dk_threshold を override list から除去 |

---

## §7 338#/339# 全 Finding 残課題マトリクス

### 完了済み

| 元番号 | Finding | 対応 | Commit |
|--------|---------|------|--------|
| 338#1 / 339#§2 | **CRITICAL**: `threshold_offset_bps` 符号逆転 | `threshold -= offset` に修正、テスト assertion 修正 | `7b7601758` (340#) |
| 338#5 | -1.0bps は hindsight fit | 閾値を元の較正値に復元 (sell=-0.3, buy=-0.8)。符号修正で根本原因解消 | 341# (本 commit) |
| 338#7 | resume_window の wall-clock 解釈 | サイクル数ベースであることを 341# で明記。273# max_kill_duration_sec=1800 が wall-clock 安全弁 | 341# (文書のみ) |

### 対応不要 (設計として妥当 / 既対策)

| 元番号 | Finding | 判定理由 |
|--------|---------|---------|
| 338#2 | PnL 指標の混同 (post_fill_30s vs ev_weighted) | 337# は kill 制御分析であり post_fill_30s_pnl が正しい指標。metric ラベル注記は有用だが分析結論に影響なし |
| 342#F | Regime Detection Feedback Loop | 158# で対策済み。regime 更新は OB mid 依存で fill 非依存 |

### P1 残課題 (中期: 次回デプロイ前に検討)

| 元番号 | Finding | 現状 | 推奨アクション |
|--------|---------|------|---------------|
| 338#6 / 339#§3.3 / 342#A | forced_switch 完全除外のリスク | 337# で hard exclude 実装済み。buy 側のみ forced KPI 分離あり | downweight=0.5 に変更 + sell 側 forced KPI 分離追加 |
| 338#4 / 342#C | skip_gate / dynamic_kill 二重抑制 | 独立動作。sell pass rate が multiplicative に低下 | kill 状態を skip_gate にコンテキスト伝達。kill 解除後 N cycle は adaptive threshold 緩和 |

### P2 残課題 (中長期)

| 元番号 | Finding | 推奨アクション |
|--------|---------|---------------|
| 338#3 / 339#§3.2 / 342#B | 二重緩和ルート (inv_bypass vs inv_relaxation) | inv_bypass をステップ関数→gradual 化、または inv_relaxation max_bps 拡大で bypass 廃止 |
| 342#D | Count-based rolling window | EWMA 化 (α=0.05, effective window≈20)。regime 遷移応答の高速化 |
| 342#E | Sell 90s vs Buy 30s post_fill_wait | PnL horizon 差を kill 閾値に反映 (σ√(90/30)≈1.73x スケーリング) |
| 342#G | DD Guard / Dynamic Kill のリセット不整合 | 日替わり時の rolling window padding、max_kill_duration と cooldown_release の整合 |
| 338# §6-P2-8 | Gate hierarchy 文書化 | quote widening → participation reduction → duty-cycle → hard kill の責務順序を文書化 |

---

## §8 再起動前チェックリスト

- [x] 340# 符号修正 (`threshold -= offset`)
- [x] 341# sell 閾値復元 (-0.3 / regime: -0.3, -1.0, -0.5)
- [x] 341# buy 閾値復元 (-0.8 / regime: -0.5, -1.5, -0.5 / max_bps: 0.3)
- [x] 337# forced_switch filter 維持 (除外ロジック自体は正常)
- [x] 337# sell_inv_relaxation 維持 (scale=0.4, max_bps=0.3)
- [x] テスト全通過 (4180 passed)
- [ ] Bot 再起動 (ユーザー判断待ち)
