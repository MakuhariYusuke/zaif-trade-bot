# 275# 責務分離 DRY + 市場理論活用拡大

> **フェーズ**: ph2 (G1.1-exec)  
> **種別**: impl (実装 + 理論補強)  
> **日付**: 2026-03-04  
> **前提**: 274# 市場理論補強 → **本 275# で DRY 強化 + 理論活用の最大化**

---

## 1. 背景

274# で 3 モジュールに市場理論 docstring を追加。275# では:

1. **side パラメータ化** — `_is_sell_killed`/`_is_buy_killed` → `_is_side_killed(side)` 統一 (~40行削減)
2. **PnL 追跡統一** — `_track_sell_pnl`/`_track_buy_pnl` → `_track_side_pnl(record)` 統一
3. **toxic veto DRY** — サイクル末尾の inline 重複コード → `_tick_toxic_veto("cycle_end")` に置換
4. **市場理論 docstring 8 モジュール追加** — 理論カバレッジを 6 → 14 モジュールに拡大

---

## 2. side パラメータ化 (DRY)

### 2.1 問題

`_is_sell_killed()` / `_is_buy_killed()` は完全対称コード。差異はインスタンス (`_sell_kill_mgr` vs `_buy_kill_mgr`) と
メトリクス suffix (`"sell"` vs `"buy"`) のみ。同様に `_track_sell_pnl()` / `_track_buy_pnl()` も対称。

### 2.2 理論的根拠

> Glosten & Milgrom (1985) §3: 逆選択リスクは bid/ask 対称であり、  
> kill 判定ロジック自体は side に依存しない。

> Ho & Stoll (1981): PnL 追跡も在庫リスクモデルにおいて対称。

### 2.3 変更

| Before | After | 削減 |
|---|---|---|
| `_is_sell_killed()` (18行) | `_is_side_killed("sell")` | 統一 |
| `_is_buy_killed()` (18行) | `_is_side_killed("buy")` | ~18行削減 |
| `_track_sell_pnl()` (6行) | `_track_side_pnl(record)` | 統一 |
| `_track_buy_pnl()` (6行) | — | ~6行削減 |

呼び出し 2 箇所 (L953: track, L2143-2144: kill check) を新メソッドに更新。

---

## 3. toxic veto DRY (P0 修正)

### 3.1 問題

272# で `_tick_toxic_veto()` ヘルパーを抽出したが、**サイクル末尾 (L2487-2492) に inline 重複コードが残存**。
`_tick_toxic_veto()` の 3 呼出箇所 (both-blocked, inventory_escape, halt_block) とは
別パスで実行されるため二重デクリメントは起きないが、DRY 違反であり保守性を低下させる。

### 3.2 変更

```python
# Before (L2487-2492): 6行の inline コード
if self._toxic_veto:
    for _veto_side in list(self._toxic_veto.keys()):
        self._toxic_veto[_veto_side] -= 1
        if self._toxic_veto[_veto_side] <= 0:
            del self._toxic_veto[_veto_side]
            logger.info(f"[205# §9.2] Toxic veto expired: {_veto_side}")

# After: 1行
self._tick_toxic_veto("cycle_end")
```

ログメッセージも `[226# S2]` に統一。

---

## 4. 市場理論 docstring 追加 (8 モジュール)

### 4.1 追加一覧

| モジュール | 追加理論 | 学術引用 |
|---|---|---|
| `regime_detector.py` | Markov-Switching Model | Hamilton (1989) |
| | Adaptive Market Hypothesis | Lo (2004) |
| | ヒステリシスの Bayes 解釈 | posterior 確定まで待つ離散近似 |
| `side_selector.py` | Inventory Management | Garman (1976), Ho & Stoll (1981) |
| | 在庫中立化の離散実装 | Stoll (1978) §3 → Avellaneda-Stoikov (2008) 簡略版 |
| `param_adapter.py` | Optimal Spread 適応 | Avellaneda & Stoikov (2008) |
| | 逆選択回避 | Glosten & Milgrom (1985) |
| `micro_circuit_breaker.py` | Circuit Breaker 理論 | SEC Rule 80B, Greenwald & Stein (1991) |
| | Liquidity Spiral | Brunnermeier & Pedersen (2009) |
| `spread_anomaly_detector.py` | Effective Spread | Roll (1984) |
| | Information-Based Spread | Copeland & Galai (1983) |
| | Illiquidity | Amihud (2002) |
| `velocity_math.py` | Price Impact (λ) | Kyle (1985) |
| | Information Share | Hasbrouck (1991) |
| `macro_regime.py` | Regime-Switching | Hamilton (1989) |
| | Micro-Macro 矛盾検出 | compose_regimes() による状態整合性検証 |
| `adaptation_engine.py` | Adaptive Market Hypothesis | Lo (2004) |
| | Kelly Criterion 統合 | Kelly (1956) |

### 4.2 市場理論カバレッジ推移

| フェーズ | 理論参照あり | 理論参照なし | カバレッジ |
|---|---|---|---|
| 273# まで | 6 | 41 | 13% |
| 274# 完了 | 9 | 38 | 19% |
| **275# 完了** | **14** | **33** | **30%** |

主要なリスク管理・戦略モジュール (14/14) は全て理論的根拠を持つ状態に到達。
残33ファイルは基盤/ユーティリティ/IO系で理論参照は不要。

---

## 5. _opposite_side 再利用性分析

272# で抽出された `_opposite_side` は orchestrator 内 5 箇所で使用中。
他モジュール (`side_selector`, `cycle_gate_aggregator` 等) では inline で
`"sell" if side == "buy" else "buy"` を書いている箇所は見当たらず、
現時点では orchestrator 内 staticmethod で十分。

将来的に他モジュールで必要になった場合は共通ユーティリティに昇格を検討。

---

## 6. テスト結果

| スコープ | 結果 |
|---|---|
| 275# 新規テスト (29 件) | ✅ 29 passed |
| 240# テスト修正 (1 件) | ✅ `_is_buy_killed` → `_is_side_killed("buy")` 文字列検索更新 |
| v460 全体回帰テスト | ✅ 3793 passed, 0 failed |

### テストファイル内訳

```
test_275_dry_separation_and_theory.py
├── TestSideParameterization           (6 tests)
├── TestToxicVetoDRY                   (2 tests)
├── TestMarketTheoryDocstrings275      (8 tests)
├── TestOppositeSideReuse              (2 tests)
└── TestMarketTheoryCoverage           (11 parametrize tests)
```

---

## 7. 変更ファイル一覧

| ファイル | 行数変化 | 対象 |
|---|---|---|
| `scripts/v460/lib/fill_loop_orchestrator.py` | -35 (4メソッド → 2メソッド + inline DRY) | side パラメータ化, toxic veto DRY |
| `scripts/v460/lib/regime_detector.py` | +15 (docstring) | Hamilton, AMH |
| `scripts/v460/lib/side_selector.py` | +13 (docstring) | Garman, Ho-Stoll, Stoll |
| `scripts/v460/lib/param_adapter.py` | +14 (docstring) | Avellaneda-Stoikov, Glosten-Milgrom |
| `scripts/v460/lib/micro_circuit_breaker.py` | +14 (docstring) | SEC 80B, Brunnermeier-Pedersen |
| `scripts/v460/lib/spread_anomaly_detector.py` | +16 (docstring) | Roll, Copeland-Galai, Amihud |
| `scripts/v460/lib/velocity_math.py` | +12 (docstring) | Kyle, Hasbrouck |
| `scripts/v460/lib/macro_regime.py` | +10 (docstring) | Hamilton, Regime-Switching |
| `scripts/v460/lib/adaptation_engine.py` | +12 (docstring) | AMH (Lo 2004), Kelly |
| `tests/unit/v460/test_240_toxicity_budget.py` | +3 (テスト修正) | 275# DRY 追従 |
| `tests/unit/v460/test_275_*.py` | +200 (新規) | 全テスト |

---

## 8. 残課題 (将来対応)

| 優先度 | 課題 | 出典 | 備考 |
|---|---|---|---|
| P2 | Pattern E: halt 中 veto 時間減衰速度見直し | 269# | 二重デクリメントなし確認済、現状で問題なし |
| P2 | BlockingPolicy 抽出 (veto/balance/inventory ~200行) | 272# | 状態密結合で中規模リファクタ |
| P2 | `_opposite_side` 共通ユーティリティ昇格 | 275# | 他モジュール需要が出たら |
| P3 | `except Exception` 絞り込み (74箇所, SGE 重点) | scan | API/IO 防御が大半、要個別精査 |
