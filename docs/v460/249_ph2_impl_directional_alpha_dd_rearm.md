# 249# Directional Alpha + DD Re-arm + Quiescence

## 概要

247# (Codex Review) と 248# (Gemini Review) の P0 勧告を実装。
**パラダイムシフト: 在庫中立への固執を止め、トレンド方向のポジション蓄積を許容する。**

BTC 価格上昇中に bot が BTC を買い続けるのは正しいポジショニング。
「BTC がどれだけ増えたか」も含めた Total Equity MTM で収益性を評価する。

## 変更一覧

### 1. DD Cooldown Re-arm (247# CRITICAL 1.5)

**問題**: 246# の cooldown release で halt 解除後、追加損失が青天井になるリスク。
**対策**: release 後の累積 PnL を追跡し、予算超過で再 halt (re-arm)。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `cooldown_rearm_budget_bps` | -10.0 | release 後にこの bps 超過で再 halt |

**動作フロー**:
1. DD halt → cooldown release (246#)
2. release 後の各 fill で `cooldown_rearm_pnl_bps` を累積追跡
3. `cooldown_rearm_pnl_bps <= cooldown_rearm_budget_bps` で再 halt
4. `cooldown_rearmed=True` → 二度目の cooldown release は発生しない
5. rearm_budget_bps=0 で無効化 (条件: `< 0` でのみ発動)

**変更ファイル**:
- `daily_drawdown_guard.py`: `DailyDrawdownState` に rearm フィールド追加、
  `update_pnl()` に追跡ロジック、`is_halted()` に rearm チェック
- `fill_config.py`: `dd_cooldown_rearm_budget_bps` フィールド + YAML パース
- `run_fill_test.py`: 両コンストラクタにパラメータ配線

### 2. Total Equity MTM Tracking (248# P0)

**問題**: cumPnL が JPY スプレッド損益のみを追跡 → BTC 蓄積の価値を見逃す。
**対策**: `cumulative_btc_delta` を追跡し、MTM ベースの Total Equity Δ をログ出力。

```
totalEquityΔ = spreadPnL(JPY) + btcDelta × (mid - entryMid)
```

**変更ファイル**:
- `fill_loop_orchestrator.py`:
  - `cumulative_btc_delta` 変数追加 (buy=+qty, sell=-qty)
  - resume 時に既存レコードから再構成
  - progress log に `btcDelta=` と `[249# MTM] totalEquityΔ=` 行追加
  - hard loss cap は JPY-only のまま (保守的安全策)

### 3. Regime-aware Inventory Skewing (248# P0)

**問題**: トレンド中に inventory skewing が逆方向の清算を促進 →
directional alpha (トレンド追従利益) を阻害。
**対策**: trending regime 時に inv_skew を自動無効化。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `inv_skew_regime_gate_enabled` | False | True でトレンド時に inv_skew 無効化 |

**変更ファイル**:
- `maker_price.py`: `compute()` 内 inv_skew ブロック前に regime gate チェック追加
- `fill_config.py`: `inv_skew_regime_gate_enabled` フィールド + YAML パース

### 4. dual_kill_bypass → Quiescence (247# P0-3)

**問題**: buy/sell 両方 kill 時の dual_kill_bypass が「両方 toxic なのに全通過」
させてしまい、損失を拡大する。
**対策**: quiescence モードで bypass を無効化 → 各 kill gate が個別にブロック判定。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `dual_kill_quiescence_enabled` | False | True で dual kill 時に静観 |

**変更ファイル**:
- `cycle_gate_aggregator.py`: dual_kill ブロックに quiescence 分岐追加
- `fill_config.py`: `dual_kill_quiescence_enabled` フィールド + YAML パース

### 5. Parameter Validation Hardening (247# 1.11)

**問題**: 不正なパラメータ値がランタイムまで検出されない。
**対策**: `__post_init__()` に境界チェック追加。

| パラメータ | 制約 |
|-----------|------|
| `degraded_liquidation_lot_mult` | [0.01, 1.0] |
| `degraded_liquidation_offset_mult` | >= 1.0 |
| `degraded_liquidation_duty_cycle` | >= 2 |
| `dd_cooldown_release_lot_scale` | [0.01, 1.0] |
| `dd_cooldown_release_sec` | >= 0 |
| `dd_cooldown_rearm_budget_bps` | <= 0 |

## YAML 設定 (fill_test.yaml)

```yaml
loss_control:
  daily_drawdown:
    cooldown_rearm_budget_bps: -10.0  # 249# re-arm
  inventory_skewing:
    regime_gate_enabled: true          # 249# directional alpha
  dual_kill_quiescence_enabled: true   # 249# quiescence
```

## テスト

- `test_249_directional_alpha.py`: 29 テスト (5 クラス)
  - `TestCooldownRearm249`: 7 テスト — re-arm 発動、二重解除防止、lot scale、export/import
  - `TestInvSkewRegimeGate249`: 4 テスト — trending 時 block、ranging 時 active、gate disabled
  - `TestDualKillQuiescence249`: 5 テスト — quiescence block、legacy bypass、single kill
  - `TestParameterValidation249`: 9 テスト — 各パラメータ境界違反
  - `TestConfigWiring249`: 5 テスト — デフォルト値、YAML パース
- 既存テスト修正:
  - `test_168_daily_drawdown_guard.py`: metrics_keys に rearm フィールド追加
  - `test_234_gate_bypass_removal.py`: duty_cycle テストを ValueError 期待に変更

**全 3449 テスト PASS** (29 新規 + 3420 既存)

## レビュー元

- 247# Codex Review: `docs/v460/247_ph2_rev_234_246_functionality_market_theory_review.md`
- 248# Gemini Review: `docs/v460/248_ph2_gemini_31_pro_review_234_246_directional_alpha.md`
