# 343# P1 改善: forced downweight / sell KPI 分離 / skip_gate kill 連携

> **種別**: impl  
> **日付**: 2026-03-08  
> **コミット**: `6195f1515`  
> **前提**: 342# 設計・市場理論深掘り調査 (Finding A–G)  
> **テスト**: v460 unit 4206 passed, +25 new test cases  

---

## §1 概要

342# の Finding A / C を中心に P1 タスク 3 件を実装し、
作業過程で発見した P0 修正 2 件を合わせて 5 項目を実施した。

| # | 施策 | 342# 参照 | 概要 |
|---|------|-----------|------|
| A | forced fill PnL downweight | 342#A / 338#6 | 337# の完全除外→0.5 倍重み付け |
| B | sell forced KPI 分離 | 342#A 補完 | buy 側と対称な forced sell KPI |
| C | skip_gate/kill release grace | 342#C / 338#4 | kill 解除直後の skip_gate 過剰抑制防止 |
| D | regime_min_confidence sync | 作業中発見 | コードデフォルト 0.3→0.2 (YAML 一致) |
| E | getattr→直接参照 | 作業中発見 | 型安全性向上 |

---

## §2 施策詳細

### §2.1 (A) forced fill PnL downweight

**問題** (342#A / 338#6): 337# で導入した `balance_forced_switch` PnL の完全除外は、
rolling window から forced fill の情報を完全に遮断し、kill 解除判定を鈍らせる。
kill 中に発生した forced fill が 5 件続いても rolling mean は更新されず、
本来なら kill 解除すべき状況で kill が継続する。

**解決**: 完全除外 (return) → `forced_fill_pnl_downweight` (default 0.5) による重み付け投入。

```python
# Before (337#)
if getattr(record, "balance_forced_switch", False):
    return  # 完全除外

# After (343#)
if getattr(record, "balance_forced_switch", False):
    weight = self.config.forced_fill_pnl_downweight
    if weight <= 0.0:
        return  # 0.0 = 旧挙動 (完全除外)
    pnl = pnl * weight
```

**設計根拠**:
- `0.5` = 「通常取引ほど信頼しないが、情報は活用する」
- forced fill は MM の spread capture を意図せず即時約定するため、PnL の品質は通常取引より劣る
- しかし完全除外は情報の損失であり、kill 解除判定の精度を低下させる
- `0.0` (完全除外) ↔ `1.0` (通常扱い) の連続スペクトルで hot-reload 可能

**DynamicKillManager との相互作用**:
- `track()` は PnL 値をそのまま `_pnl_history` に append
- `is_kill_active()` は直近 `window` 件の算術平均で判定
- downweight=0.5 は **件数はそのまま** だが **PnL 寄与を半分** にする
- 結果: forced fill が連続すると rolling mean が実態より楽観的になる可能性
- → 0.5 は完全除外 (情報遮断) と通常扱い (PnL 汚染) の折衷点

**変更ファイル**: `orchestrator_guards.py`, `fill_config.py`, `fill_config_parser.py`, `config_hot_reload.py`, `fill_test.yaml`

---

### §2.2 (B) sell forced KPI 分離

286# 283# P1-5 で buy 側のみ実装されていた forced/normal KPI 分離を、
sell 側にも対称的に拡張。

**追加フィールド** (RunSessionState):
- `forced_sell_fill_count`, `forced_sell_pnl_sum_bps`
- `normal_sell_fill_count`, `normal_sell_pnl_sum_bps`

**ログ出力例**:
```
[343#] Sell KPI split: forced=12 fills (-3.42bps avg), normal=45 fills (+0.89bps avg)
```

これにより sell 側でも forced fill の品質劣化を定量的に把握可能になった。

**変更ファイル**: `fill_loop_orchestrator.py` (RunSessionState), `orchestrator_post_cycle.py`, `fill_config.py`, `fill_config_parser.py`, `config_hot_reload.py`, `fill_test.yaml`

---

### §2.3 (C) skip_gate/kill release grace window

**問題** (342#C / 338#4): dynamic kill 中は skip_gate が実行されず、
adaptive data (EWMA, velocity 等) が stale になる。
kill 解除直後に skip_gate が古い stale データで判定すると、
**本来 pass すべき取引を過剰に skip してしまう** (二重抑制)。

**解決**: kill 解除直後の N サイクルは skip_gate offset を負方向 (緩和) にシフト。

**実装構造**:
1. **Kill release 検出** (`orchestrator_guards.py`):
   - `_is_side_killed()` 内で kill(True)→非kill(False) 遷移を検出
   - 遷移時に `_kill_released_at_cycle_{side} = self._cycle_count` を記録

2. **Grace offset 計算** (`fill_cycle_executor.py`):
   - `_evaluate_skip_gate()` で grace window 内か判定
   - `self._cycle_count - _rc < grace_cycles` なら offset 加算

3. **Offset 適用** (`skip_gate_evaluator.py`):
   - `evaluate()` の `kill_release_offset` パラメータ経由で受け取り
   - `_total_offset = _hour_offset + _spread_offset + kill_release_offset`
   - **clamp (floor/ceil) の前** に加算されるため、安全に制限される

**パラメータ**:
```yaml
skip_gate:
  kill_release_grace_cycles: 3   # kill 解除後の緩和サイクル数
  kill_release_offset: -0.1     # 緩和 offset (負=緩和)
```

**エッジケース分析**:
- プロセス再起動: `_kill_released_at_cycle_{side}` は `None` にリセット → grace 不発 (安全)
- `_cycle_count` ラップ: 理論上ありえないが、`0 <= diff` ガードで安全
- grace window 内の kill 再発: kill 中は `_evaluate_skip_gate` 未到達 → offset 無適用 (正しい)

**変更ファイル**: `orchestrator_guards.py`, `fill_cycle_executor.py`, `skip_gate_evaluator.py`, `fill_loop_orchestrator.py`, `fill_config.py`, `fill_config_parser.py`, `config_hot_reload.py`, `fill_test.yaml`

---

### §2.4 (D) regime_min_confidence default sync

**問題**: `FillTestConfig.regime_min_confidence` のコードデフォルトが `0.3` だが、
YAML は `0.2` (152# で変更済)。YAML なしで起動すると閾値が 50% 高くなり、
regime 判定が過剰に保守的になる。

**修正**: コードデフォルトを `0.3` → `0.2` に変更。
336# drift prevention test の allowlist から `regime_min_confidence` を除去。

---

### §2.5 (E) getattr→直接参照

`orchestrator_guards.py` の `_is_side_killed()` 内で使われていた
`getattr(self, "_cycle_count", 0)` を `self._cycle_count` に変更。
`_cycle_count` は `FillLoopOrchestratorMixin` にクラスレベル宣言を追加 (= 0)。

`fill_cycle_executor.py` の `_evaluate_skip_gate()` でも
`getattr(self, f"_kill_released_at_cycle_{side}", None)` → 直接属性参照に変更。

---

## §3 セルフレビュー結果

| # | チェック項目 | 結果 | 詳細 |
|---|-------------|------|------|
| 1 | sell KPI `next_side == "sell"` は正しいか | ✅ OK | `next_side` は「このサイクルの side」(docstring で明記)。buy 側と対称。 |
| 2 | kill release grace のリスタート安全性 | ✅ OK | `_kill_released_at_cycle_{side}` は `None` に初期化 → grace 不発 (フェイルセーフ)。 |
| 3 | downweight × DynamicKillManager | ⚠️ 要認識 | 件数は等倍だが PnL 寄与が半分 → rolling mean が楽観に寄る可能性。0.5 は折衷点。 |
| 4 | Hot-reload マッピング | ✅ OK | `sg_map` + `_HOT_RELOADABLE_FIELDS` 正常。 |
| 5 | YAML キー名整合性 | ✅ OK | 止血 + skip_gate セクション双方で正確。 |
| 6 | `balance_forced_switch` アクセス一貫性 | ⚠️ 注意 | guards は `getattr` (防御的)、post_cycle は直接参照 (286# 踏襲)。FillRecord の `bool \| None = None` が保証するため `AttributeError` は発生しないが、パターンの不一致は認識すべき。 |
| 7 | clamp 前の offset 加算 | ✅ OK | `kill_release_offset` は `_total_offset` に加算された後、floor/ceil で安全にクランプされる。 |
| 8 | テストカバレッジ | ✅ OK | 25 new test cases + 337# テスト更新。4206 passed。 |

---

## §4 変更ファイル一覧

| ファイル | 変更行 | 施策 |
|---------|--------|------|
| `configs/v460/fill_test.yaml` | +7 | A, B, C: YAML パラメータ追加 |
| `scripts/v460/lib/config_hot_reload.py` | +5 | A, B, C: hot-reload 対象追加 |
| `scripts/v460/lib/fill_config.py` | +13, -1 | A, B, C, D: config フィールド追加 + default sync |
| `scripts/v460/lib/fill_config_parser.py` | +9 | A, B, C: parser 追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | +7 | C, E: kill release offset 計算 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +12 | B, C, E: クラスレベル宣言 + RunSessionState |
| `scripts/v460/lib/orchestrator_guards.py` | +39, -1 | A, C, E: downweight + kill release 追跡 |
| `scripts/v460/lib/orchestrator_post_cycle.py` | +31 | B: sell KPI トラッキング + ログ |
| `scripts/v460/lib/skip_gate_evaluator.py` | +8, -1 | C: kill_release_offset パラメータ |
| `tests/unit/v460/test_343_p1_improvements.py` | +344 | 新規テスト |
| `tests/unit/v460/test_337_sell_side_countermeasures.py` | +21, -4 | downweight 対応に修正 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | -1 | allowlist 除去 |

---

## §5 残課題 (342# アクションリストの残り)

### P2 (中期検討)

| 342# | 内容 | 状態 |
|------|------|------|
| B | inv_bypass 不連続性 (0→full jump) | 未着手。テーパリング導入の検討が必要 |
| D | EWMA 化 (rolling mean → EWMA) | 未着手。全体的なリファクタが必要 |
| E | sell post_fill_wait_sec 非対称化 | 未着手。YAML パラメータ追加のみ |
| F | velocity ルールの AS-aware 化 | 未着手。skip_gate 制御の複雑さ増加リスク |
| G | preflight 差分対称性 | 未着手。低優先度 |

### 作業中に発見した改善候補

| 優先度 | 内容 | 詳細 |
|--------|------|------|
| P1 | `velocity_ema_alpha=1.0` (無効) | 0.3推奨。bid-ask bounce による誤 velocity spike を抑制可能 |
| P1 | `ranging_obi_asymmetry_factor=0.0` (無効) | 0.3推奨。ranging 市場での OBI 方向シグナル活用 |
| P2 | `inv_decay_tau_sec=0.0` (無効) | 1800推奨。古い fill 履歴の影響を時間減衰 |
| P2 | `skip_gate_score_calibration=False` | Isotonic regression による score 校正 |
