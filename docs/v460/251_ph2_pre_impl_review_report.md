# 251# Pre-Implementation Review Report

> **日付**: 2026-03-03  
> **対象**: 247#/248# 残 P1/P2 項目の実装準備レビュー  
> **前提**: 249# (DD Re-arm, Total Equity MTM, Regime Inv Skew Gate, dual_kill Quiescence, Param Validation)  
>          250# (P/L 3-way split, freeze/cooldown side tracking, quiescence deadlock defense, probe deprecation)

---

## 1. P1-1: Sell Asymmetric Mode — 要件定義

### 出典

| ソース | セクション | 優先度 |
|--------|-----------|--------|
| 247# §2.2 | "Sell defence hardening は延命であって根治ではない" | HIGH |
| 248# §2 Q4/Q5 | "Trending Up での Sell 完全封鎖は是" | P0→P1 |
| 248# §4 [P1] | "Sell 側モデルの Asymmetric Mode 化" | P1 |

### 具体要件

248# の定義が最も明確:

> **現在 Sell 側モデルは統計的エッジを喪失 (DEGRADED)。Trending Up / High Vol 時は「極端な利確 (TP) 目的の片側エスカレーション」以外で Sell を完全 Freeze (Hard Skip) する非対称運用をデフォルトとする。**

247# の補足:

> sell を harder にするのではなく、**sell を inventory target 超過時に限定** + **reversal evidence が出るまで BTC を持つ**

### 実装に必要な変更箇所

1. **`fill_config.py`**: `sell_asymmetric_mode: bool` パラメータ追加（現在このパラメータは存在しない）
2. **`cycle_gate_aggregator.py`**: regime=trending_up 時に sell を Hard Skip する分岐
3. **`sell_dynamic_kill.py`**: asymmetric mode 時の sell 許可条件を TP 目的のみに限定
4. **`maker_price.py`**: asymmetric mode 時の sell offset を極端に広げる (TP レンジ)

### 収益インパクト推定

245# データ: `sell pass PnL = -1.316bps`, `trending_up sell = -0.919bps`  
→ sell 封鎖で最大 **+2.2bps/day** の損失削減が見込める（18日間 -792 JPY の主因）

---

## 2. P1-2/P1-3: PhantomPositionGuard 改善 — 現状と必要変更

### P1-2: 三値化 (ternary reconciliation result)

**現状** ([phantom_position_guard.py](scripts/v460/lib/phantom_position_guard.py)):
- `_reconcile_single()` は `PhantomDetection | None` を返す（二値: detected / clean）
- API 例外は `except Exception` で warning ログのみ → pending は次の `clear()` で消える
- [L298](scripts/v460/lib/phantom_position_guard.py#L298): order recheck 失敗 → warning のみ
- [L332](scripts/v460/lib/phantom_position_guard.py#L332): balance check 失敗 → warning のみ
- [L259](scripts/v460/lib/phantom_position_guard.py#L259): `self._pending.clear()` で全エントリ破棄

**問題**: 取引所 API 不安定時こそ phantom が起きやすいのに、一回の再照合失敗で quarantine を手放す。

**必要変更**:
```python
# 新しい三値 enum
class ReconcileResult(Enum):
    CLEAN = "clean"           # 確実にキャンセル済み
    PHANTOM = "phantom"       # 確実に約定済み
    INCONCLUSIVE = "inconclusive"  # 判定不能 → pending 維持

# _reconcile_single() の戻り型変更
async def _reconcile_single(...) -> tuple[ReconcileResult, PhantomDetection | None]:
```

- `INCONCLUSIVE` な entry は `_pending` に残す
- TTL (`_MAX_PENDING_AGE_SEC=300s`) は既存のまま最終防衛線として機能
- retry 上限を設けて無限保留を防止 (`max_retries=3` 程度)

### P1-3: buy 側残高照合の配線完了

**現状**:
- `BalanceChecker._last_jpy_free` は[L44](scripts/v460/lib/balance_checker.py#L44)で保持されている
- しかし **公開プロパティ `last_jpy_free` が存在しない**（`last_btc_free` はある）
- `_maybe_register_phantom()` ([fill_cycle_executor.py L176](scripts/v460/lib/fill_cycle_executor.py#L176)) は `balance_btc` のみ渡し、`balance_jpy` は未渡し

**必要変更**:
1. `BalanceChecker` に `last_jpy_free` プロパティ追加
2. `_maybe_register_phantom()` で buy 時は `balance_jpy=_jpy_snap` も渡す
3. `_reconcile_single()` の Phase 2 で buy 時は JPY 減少を照合に使用

**工数**: 小（3ファイル, 計 ~30 行の変更）

---

## 3. God Object メトリクス

| ファイル | 行数 | メソッド数 | 247# 時点 | 増減 |
|----------|------|-----------|-----------|------|
| `fill_loop_orchestrator.py` | **2,433** | **31** | 2,356 | +77 |
| `fill_cycle_executor.py` | **1,370** | **19** | 1,369 | +1 |
| `fill_config.py` | **1,617** | **7** | 1,569 | +48 |
| **合計** | **5,420** | **57** | 5,294 | **+126** |

### 評価

- orchestrator は 247# 時点の 2,356 行から **+77 行** 増加（250# の P/L split, freeze side tracking, quiescence deadline 等）
- 247# §1.12 の警告通り「守りの追加が orchestrator に再集中」するパターンが継続
- ただし 249#/250# で追加された責務（P/L 追跡、freeze side、deadlock 防御）は orchestrator のライフサイクル管理と密結合しており、単純な切り出しは非自明
- **次の責務追加（Sell Asymmetric 等）は orchestrator ではなく gate_aggregator / dynamic_kill に持たせるべき**

---

## 4. TODO/FIXME/HACK 検出結果

### scripts/v460/lib/ 配下

| ファイル | 行 | 内容 | 状態 |
|----------|-----|------|------|
| [fill_config.py L438](scripts/v460/lib/fill_config.py#L438) | `TODO(235#)` | "YAML / hot_reload から参照が消えたら削除" | **未対応** — 要確認 |

### ztb/risk/ 配下

検出なし。

### 評価

TODO は 1 件のみ。235# の古い TODO で、hot_reload のパラメータ参照整理に関するもの。低優先度だが長期放置は衛生上よくない。

---

## 5. 型安全性の問題

### Any 型使用

| ファイル | 状況 |
|----------|------|
| `fill_loop_orchestrator.py` | **Any 型なし** ✅ |
| `fill_cycle_executor.py` | **Any 型なし** ✅ (L419 はコメント中の言及のみ) |
| `sell_dynamic_kill.py` | **Any 型なし** ✅ |
| `ztb/utils/cli_common.py` | `Any` 使用あり（CLI argparse の `default: Any` 等） — 許容範囲 |
| `ztb/utils/performance_utils.py` | デコレータの `*args: Any, **kwargs: Any` — 許容範囲 |
| `ztb/utils/fault_injection.py` | `Callable[..., Any]` — 許容範囲 |

### `except Exception` の広範使用

| ファイル | 件数 | 評価 |
|----------|------|------|
| `skip_gate_evaluator.py` | 11 件 | 多い。外部 API/WebSocket の防御的 catch だが、一部は具体例外に絞れる |
| `config_hot_reload.py` | 5 件 | YAML パース等で妥当 |
| `fill_cycle_executor.py` | 5 件 | 注文/キャンセル系で妥当 |
| `phantom_position_guard.py` | 2 件 | **P1-2 三値化で改善予定** |
| `event_logger.py` | 4 件 | うち 2 件は `except Exception:` (変数未束縛) — bare catch 同等で要改善 |
| `balance_checker.py` | 1 件 | 妥当 |

### `getattr` 残留

- [fill_cycle_executor.py L176](scripts/v460/lib/fill_cycle_executor.py#L176): `getattr(self._balance_checker, 'last_btc_free', None)` — Protocol / property が存在するのに文字列ベースアクセス。直接 `self._balance_checker.last_btc_free` に変更すべき。

---

## 6. 251# 推奨施策（優先順位付き）

### Tier 1: 収益直結（今回実装すべき）

| # | 施策 | 根拠 | 工数 | 期待効果 |
|---|------|------|------|----------|
| **A** | **Sell Asymmetric Mode** (P1-1) | 最大の損失源 (-2.2bps/day)。248# P1 指定。gate_aggregator + dynamic_kill で完結し orchestrator 肥大化を回避 | M (中) | **+2.2bps/day** 損失削減 |
| **B** | **PhantomPositionGuard 三値化** (P1-2) | 一回の API 失敗で phantom quarantine を手放すリスク。三値 enum + retry 上限で完結 | S (小) | 安全性向上 |
| **C** | **PhantomPositionGuard buy 側照合** (P1-3) | 3 ファイル ~30 行。P1-2 と同時作業で効率的 | XS (極小) | buy phantom 検出漏れ防止 |

### Tier 2: 構造改善（次回以降）

| # | 施策 | 根拠 | 工数 |
|---|------|------|------|
| **D** | Inventory Target Band (P2-2) | 247# §2.1 の "ゼロ在庫前提" 脱却。249# の regime inv skew gate が第一歩だが、明示的な target band はまだない | L (大) |
| **E** | Feasible Quote 完全計算 (P2-1) | 247# §1.7 "真の制約交点計算"。post-only 非交差制約等が未実装 | M (中) |
| **F** | God Object 抑制 (P2-3) | orchestrator 2,433 行。ただし 249#/250# の追加は密結合で切り出し非自明。sell asymmetric を gate_aggregator に持たせることで間接的に抑制 | L (大) |

### Tier 3: 衛生改善（チケット発行して順次）

| # | 施策 |
|---|------|
| **G** | `event_logger.py` の bare `except Exception:` を具体例外に |
| **H** | `fill_cycle_executor.py` L176 の `getattr` → 直接プロパティ参照 |
| **I** | `fill_config.py` L438 の TODO(235#) 解消 |
| **J** | `skip_gate_evaluator.py` の `except Exception` 11 件の精緻化 |

### 251# 推奨スコープ

> **A + B + C + H を 1 コミットで実装**（推定 +250/-30 行）
>
> 理由:
> - A (Sell Asymmetric) が収益改善の最大レバー
> - B + C (PhantomGuard) は小工数で安全性を大幅に向上
> - H (getattr 修正) は B/C 作業中についでに修正可能
> - D (Inventory Target Band) は設計検討が必要で 252# 送りが妥当

---

## 付録: 248# vs 247# の P1-1 要件差分

| 観点 | 247# | 248# |
|------|------|------|
| 方向性 | sell を inventory target 超過時に限定 | sell を regime=trending_up で Hard Skip |
| TP 例外 | 言及なし | "極端な利確目的の片側エスカレーション"は許可 |
| 実装場所 | 明示なし | "Sell 側モデルの Asymmetric Mode 化" |
| 前提条件 | inventory target band 導入後 | レジーム判定のみで十分 |

**結論**: 248# の方が実装可能性が高い（inventory target band を前提としない）。247# の inventory target band は P2-2 として別途進める。
