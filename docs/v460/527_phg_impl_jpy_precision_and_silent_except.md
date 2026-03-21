# 527# JPY 精度改善 + silent except 可観測化

> 524# 分析で発覚した JPY 残高表示の精度不足と、
> サイレント例外抑制を修正する。

---

## 1. 背景

Coincheck は JPY を繊（せん）単位、即ち小数点以下 8 桁まで管理する。
実際の残高は `252.34695371` のような値を取るが、ログでは `:.0f` により
`252` と表示され、サブ円残高のコンテキストが失われていた。

また、config hash 計算や VPIN vol_sync の例外が `except Exception: pass`
で完全に無音化されており、障害時の原因特定を困難にしていた。

---

## 2. 変更一覧

### A. JPY 精度修正

| ファイル | 変更内容 |
|---|---|
| `adapter.py` L317 | `str(int(jpy_amount))` → `str(round(jpy_amount, 8))` — market_buy_amount のサブ円精度保持 |
| `balance_checker.py` | JPY 残高ログ `:.0f` → `:.2f` (shrink/insufficient/lot拡大 計4箇所) |
| `adaptation_engine.py` | loss_cap 動的算出ログ `:.0f` → `:.2f` (残高・キャップ計算 計3箇所) |
| `orchestrator_post_cycle.py` | cumulative_pnl_jpy `:.0f` → `:.2f` (SOFT/HARD cap 計2箇所) |
| `orchestrator_lifecycle.py` | resume 時 soft_loss_cap ログ `:.0f` → `:.2f` |

### B. silent except 可観測化

| ファイル | 変更内容 |
|---|---|
| `config_hot_reload.py` L802 | `except Exception: pass` → `logger.debug(...)` で hash 計算失敗を記録 |
| `skip_gate_evaluator.py` L578 | `except Exception: pass` → `logger.debug(...)` で VPIN fallback を記録 |

### C. テスト修正

| ファイル | 変更内容 |
|---|---|
| `test_013_fixes.py` L263 | `int()` → `float()` + `pytest.approx()` — market_buy_amount のサブ円精度に対応 |

---

## 3. JPY 表示 Before/After

### balance_checker.py — JPY 不足

**Before:**
```
Insufficient JPY for buy: free=252 < min=11376 (regime_mult=1.00)
```

**After:**
```
Insufficient JPY for buy: free=252.35 < min=11376.00 (regime_mult=1.00)
```

### adaptation_engine.py — loss_cap

**Before:**
```
動的キャップ算出: 残高=25649 JPY × 30% = 7695 JPY (旧: 7500 JPY)
```

**After:**
```
動的キャップ算出: 残高=25649.35 JPY × 30% = 7694.80 JPY (旧: 7500.00 JPY)
```

---

## 4. 設計判断

| 判断 | 理由 |
|---|---|
| JPY ログは `:.2f` (8桁ではなく2桁) | 可読性とのバランス — 繊単位は内部計算で保持、ログは円の銭単位で十分 |
| adapter.py は `round(..., 8)` | API 仕様に合わせた最大精度保持 — 丸め対象がないケースでも情報ロスなし |
| silent except → `logger.debug` | WARNING ではなくDEBUG — 高頻度発生時のログノイズ回避 |

---

## 5. 525# 残課題ステータス

| Finding | 525# 説明 | 対応 |
|---|---|---|
| 1 | inventory_escape legacy surface | ⏳ 未対応 — 後方互換フィールド (YAML 読み込み側) |
| 2 | _cancel_stale_orders 共有化 | ⏳ 未対応 — 524# の open order cancel 実装時に統合予定 |
| 3 | _apply_final_offset_ceiling dead code | ✅ 526# で削除済 |
| 4 | _check_balance_for_side 命名 (二重否定) | ⏳ 未対応 — リスク低・リファクタ時に改名 |
| 5 | docs 520# vs 521# 重複 | ⏳ 未対応 — prose のみ、低優先度 |

---

## 6. テスト結果

- 全 v460 テスト: 2519 passed, 5 failed (全て既存の既知失敗)
- 新規リグレッション: なし
- 修正テスト: test_013_fixes `test_market_buy_sends_market_buy_amount` (int→float 対応)
