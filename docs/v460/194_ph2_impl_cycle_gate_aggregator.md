# 194# CycleGateAggregator per-cycle skip 判定一元化

## 概要

192# §3.1 の指摘「同じ判断が 4 箇所に分散」を解消し、**per-cycle skip 判定を `CycleGateAggregator` に一元集約**した。

### 解決した問題

| 問題 | 192# 指摘元 | 対策 |
|---|---|---|
| skip 判定が orchestrator/executor/skip_gate/maker_price に分散 | §3.1 "distributed ownership" | CycleGateAggregator に集約 |
| 条件分岐の if/continue が ~137 行散在 | 同上 | `evaluate()` → `CycleGateResult` 単一フロー |
| 判定理由のトレーサビリティ欠如 | §3.2 "audit trail needed" | `GateCheckResult` チェーン + `audit_summary` |
| skip 優先順位が暗黙的 | 同上 | Gate 1–7 の明示的優先順位チェーン |

### 旧アーキテクチャ → 新アーキテクチャ

```
旧: 判定ロジックが 4 レイヤに散在
  orchestrator → A10-A14 (scattered if/continue)
  executor     → B3 (narrow_spread_pause)
  skip_gate    → C2, C4-C5 (unknown_sell, velocity_skip)
  maker_price  → D1-D3 (ValueError raise)

新: CycleGateAggregator が一元管理
  orchestrator → cycle_gate.evaluate(context) → CycleGateResult
    └ blocked=True  → continue (skip this cycle)
    └ blocked=False → executor/skip_gate/maker_price に進む
```

## 設計

### Gate 判定チェーン (優先順位順)

| Gate # | ゲート名 | 旧配置 | 条件 |
|---|---|---|---|
| 1 | unknown_regime_buy | A10 | unknown regime + buy + !balance_forced |
| 2 | ranging_buy_low_vol (B1') | A11 | ranging + buy + vol_ratio < threshold |
| 3 | trending_sell | A12 | trending + sell + 安全弁群 |
| 4 | buy_dynamic_kill | A13 | buy + is_buy_killed + !balance_forced |
| 5 | sell_dynamic_kill | A14 | sell + is_sell_killed + inv check |
| 6 | velocity_skip | C4-C5 | velocity > sell_threshold / < buy_threshold |
| 7 | unknown_regime_sell | C2 | unknown regime + sell |

**早期終了**: 最初の Hard Blocker (blocked=True) で即座にリターン。
後続ゲートは未評価のまま `CycleGateResult` を返す。

### データフロー

```python
@dataclass
class GateCheckResult:
    gate_name: str
    blocked: bool
    reason: str = ""
    detail: str = ""

@dataclass
class CycleGateResult:
    blocked: bool = False
    blocking_reason: str = ""
    checks: list[GateCheckResult]  # all evaluated gates (audit trail)

    @property
    def audit_summary(self) -> str:
        """✓gate1 → ✓gate2 → ✗gate3 (ワンライン)"""
        ...

    @property
    def cancel_reason(self) -> str:
        """blocking_reason → cancel_reasons 定数"""
        ...
```

### orchestrator 側の変更

```python
# 旧 (~137 行の散在する if/continue)
if skip_buy_unknown_regime and side == "buy" and regime == "unknown" and not balance_forced:
    continue
if skip_ranging_buy_low_vol and side == "buy" and regime == "ranging" and vol_ratio < ...:
    continue
# ... 他にも多数

# 新 (CycleGateAggregator に委譲)
gate_result = self._cycle_gate.evaluate(
    side=side, regime=regime, vol_ratio=vol_ratio,
    balance_forced=balance_forced, ...
)
if gate_result.blocked:
    logger.info(f"[cycle_gate] {gate_result.blocking_reason}: {gate_result.audit_summary}")
    continue
```

### trending_sell 安全弁の統合 (Gate 3 詳細)

trending_sell_skip は最も複雑な判定で、以下の bypass 条件を統合管理:

| 安全弁 | 条件 | 効果 |
|---|---|---|
| HF4 (連続スキップ) | trending_sell_skip_count ≥ 5 | bypass → sell 許可 |
| inv_bypass | inv_net_imbalance > 0.3 | bypass → sell 許可 (在庫偏重解消) |
| buy_side_insufficient | BTC/JPY 残高不足 | bypass → sell 許可 (balance 維持) |

### 責務境界

| CycleGateAggregator の責務 | NOT 責務 |
|---|---|
| per-cycle の「この side で取引すべきか」判定 | ループ制御 (while loop) |
| 判定理由の audit trail 記録 | balance check |
| Hard Blocker の優先順位管理 | system-level halt |
| blocking_reason → cancel_reason マッピング | ML skip gate 判定 |

## 変更ファイル

### 1. `scripts/v460/lib/cycle_gate_aggregator.py` (新規: 414 行)
- `GateCheckResult`: 個別ゲートの判定結果
- `CycleGateResult`: 全ゲート評価の統合結果 + audit trail
- `CycleGateAggregator`: 7 ゲートの順次評価 + 早期リターン
- `_GATE_TO_CANCEL_REASON`: blocking_reason → cancel_reasons 定数変換

### 2. `scripts/v460/lib/fill_loop_orchestrator.py` (-137 行, +46 行)
- ~137 行の散在する if/continue ブロックを削除
- `self._cycle_gate = CycleGateAggregator(config)` を初期化
- `gate_result = self._cycle_gate.evaluate(...)` に置換
- audit_summary をログ出力

### 3. `scripts/v460/run_fill_test.py`
- `CycleGateAggregator` import + 初期化コード追加

### 4. `tests/unit/v460/test_194_cycle_gate.py` (新規: 398 行)
- 40 テスト:
  - `TestGateCheckResult`: 個別ゲート判定のデータ構造
  - `TestCycleGateResult`: 統合結果、audit_summary、cancel_reason
  - `TestUnknownRegimeBuySkip`: Gate 1 の buy/sell/balanced_forced 分岐
  - `TestRangingBuyLowVol`: Gate 2 の B1' ハードスキップ判定
  - `TestTrendingSellSkip`: Gate 3 の安全弁群 (HF4, inv_bypass, buy_insufficient)
  - `TestBuyDynamicKill`: Gate 4 の buy kill 判定
  - `TestSellDynamicKill`: Gate 5 の sell kill + inv_net_imbalance
  - `TestVelocitySkip`: Gate 6 の sell/buy velocity 閾値
  - `TestUnknownRegimeSellSkip`: Gate 7 の unknown sell
  - `TestGatePriority`: 全ゲートの優先順位検証
  - `TestAuditTrail`: 全判定のチェーン記録確認
  - `TestIntegration`: orchestrator 統合テスト

### 5. 既存テスト修正 (6 ファイル)
- `test_139_review_fixes.py`: mock config に cycle_gate 関連フィールド追加
- `test_155_hindsight_review.py`: CycleGateAggregator mock 対応
- `test_158_regime_deadlock_fix.py`: 同上
- `test_166_hotfixes.py`: 同上
- `test_166_remaining_tasks.py`: 同上
- `test_169_ranging_buy_skip_and_metrics.py`: 同上
- `test_176_trending_offset_asymmetry.py`: 同上

## テスト結果

```
2490 → 2530 passed (+40), 0 failed
  - 新規 40 テスト (test_194_cycle_gate.py)
  - 既存テスト 全通過 (6 ファイルの mock 更新で後方互換維持)
```

## 設計判断とトレードオフ

### なぜクラスベースか (関数ではなく)

- `config` を保持する必要がある → ステートフル
- テストで個別ゲートを独立検証したい → メソッド分離
- 将来の拡張 (ゲート追加/削除) が容易 → OCP 原則

### ML skip_gate を含めなかった理由

- ML 判定は async + model load + feature extraction が必要
- per-cycle の高速パス (ルールベース) と ML パスは責務が異なる
- ML 判定は executor 内の `skip_gate_evaluator.evaluate()` に残留

### 193# ev_offset との関係

- CycleGateAggregator は「取引するかしないか」の Hard Blocker を管理
- 193# ev_offset は「取引するが価格をどう調整するか」の Soft Modifier
- 両者は直交: Gate PASS → ev_offset 適用 → 発注

## 今後の課題 (195# 以降で取り組み)

1. **velocity_skip のソフト化**: Gate 6 を hard skip → offset boost に変換 → ✅ 195# で実装済み
2. **B1' ranging_buy_low_vol のソフト化**: Gate 2 を maker_price 委譲に変換 → ✅ 195# で実装済み
3. **trending_sell_skip の簡素化**: Gate 3 の安全弁群 (HF4, inv_bypass) が依然複雑
4. **narrow_spread_pause の統合**: executor の B3 判定をここに移管検討
5. **maker_price ValueError の事前チェック**: D1-D3 を Gate 化して例外回避

## コミット

```
fb5cce9a4 feat(194#): CycleGateAggregator per-cycle skip 判定一元化 (192# §3 対応)
```

- 12 files changed, 1022 insertions(+), 303 deletions(-)
- net: +719 行 (aggregator 新設 414 行 + テスト 398 行 - orchestrator 削減 137 行)
