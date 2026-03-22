# 535# Pre-emptive CV Kill + min_spread_jpy 引き下げ

> **種別**: impl  
> **日付**: 2026-03-23  
> **起源**: 532# §8 残課題 (P1-5, P1-6)  
> **前提**: 534# (P0 完了), 533# (P1-4 時間帯ベース却下)

---

## §1 概要

532# §8 のアクションアイテムから残存していた P1 項目を実装した。

| # | 施策 | 状態 |
|---|------|------|
| P0-1/2/3 | veto deadlock, BTC=0 緩和, log_cycle_no | ✅ 534# で完了 |
| P1-4 | 12h/15h sell offset (時間帯ハードコード) | ❌ 533# で却下 |
| **P1-5** | **sell_dynamic_kill 事前指標導入** | ✅ **本実装** |
| **P1-6** | **min_spread_jpy 700→500** | ✅ **本実装** |
| P2-7 | ceiling 0.25→0.30 | ⏸ 据置 (§5 参照) |
| P2-8 | pipeline 活性段監視 | ⏸ 据置 (§5 参照) |

---

## §2 P1-5: Pre-emptive CV Kill

### 問題 (532# §4)

`sell_dynamic_kill` は rolling PnL が悪化した**事後**に反応する。
CV (cross-venue) lead-lag が adverse（BitFlyer 価格下落が先行）でも、Coincheck の sell が約定してから損失が確定するまでブロックされない。

### 設計

CV hint の adverse velocity が連続して高い場合、rolling PnL の悪化を**待たず**に sell を pre-emptive にブロックする。

```
if cv_hint.adverse_side == "sell"
   AND |velocity| >= threshold (default 2.0 bps/s)
   AND confidence >= floor (default 0.5)
   for N consecutive cycles (default 3)
→ sell kill 発動 + cooldown (default 5 cycles)
```

GlostenMilgrom (1985) の逆選択モデルに基づく: informed trader が先に参照市場で行動することで生じる情報の非対称性を、参照市場の velocity として検出し、マーケットメーカーとして不利な約定を回避する。

### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_config.py` | 5 config fields 追加 |
| `scripts/v460/lib/fill_config_parser.py` | cv_map に 5 YAML mapping 追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 2 状態変数追加 (`_preemptive_cv_sell_adverse_count`, `_preemptive_cv_sell_cooldown`) |
| `scripts/v460/lib/orchestrator_guards.py` | `_check_preemptive_cv_kill()` 新設 + `_is_side_killed()` 冒頭で呼出 + `_apply_kill_release_tracking()` 抽出 |
| `scripts/v460/lib/config_hot_reload.py` | 5 fields hot-reloadable 化 |
| `configs/v460/fill_test.yaml` | `preemptive_sell_kill` ブロック追加 |

### 設定パラメータ

| YAML キー | config field | デフォルト | 説明 |
|-----------|-------------|-----------|------|
| `preemptive_sell_kill_enabled` | `sell_preemptive_cv_kill_enabled` | `False` | 機能有効化（YAML で `true`） |
| `preemptive_sell_kill_velocity_threshold` | `sell_preemptive_cv_velocity_threshold` | `2.0` | adverse velocity 閾値 (bps/s) |
| `preemptive_sell_kill_confidence_floor` | `sell_preemptive_cv_confidence_floor` | `0.5` | CV confidence 下限 |
| `preemptive_sell_kill_consecutive_threshold` | `sell_preemptive_cv_consecutive_threshold` | `3` | 連続検出回数 |
| `preemptive_sell_kill_cooldown_cycles` | `sell_preemptive_cv_cooldown_cycles` | `5` | kill 後のクールダウン |

### リファクタリング

`_is_side_killed()` 内の 343# kill release 追跡ロジックを `_apply_kill_release_tracking(side, killed)` メソッドに抽出した。pre-emptive kill と通常 kill の両方で統一的に呼び出す。

---

## §3 P1-6: min_spread_jpy 引き下げ (700→500)

### 問題 (532# §8)

buy アクセス率 33.5% の改善余地。`min_spread_jpy=700` は BTC/JPY 11M 基準で約 6.4bps に相当し、タイトなスプレッド環境で fill 機会を逃す。

### 変更

`configs/v460/fill_test.yaml`: `min_spread_jpy: 700` → `min_spread_jpy: 500`

500 JPY ≈ 4.5bps (BTC/JPY 11M)。spread_adapt + pipeline が生成する offset が 500 JPY を下回ることは稀であり、floor としての実効性は維持される。

---

## §4 テスト

### 新規 (16 件)

`tests/unit/v460/test_535_preemptive_cv_kill.py`:

| クラス | テスト数 | 内容 |
|--------|---------|------|
| `TestPreemptiveCvKillActivation` | 4 | 単発不発火、連続発火、cooldown 持続、非adverse リセット |
| `TestPreemptiveCvKillFilters` | 4 | 低velocity、低confidence、disabled、buy側無影響 |
| `TestPreemptiveCvKillGuardFire` | 1 | guard_fire カウンタ増分 |
| `TestKillReleaseTrackingWithPreemptive` | 1 | 343# release 追跡との統合 |
| `TestMinSpreadJpyConfig` | 2 | YAML parse (500)、default (0.0) |
| `TestPreemptiveCvKillConfig` | 4 | field 存在、default disabled、YAML parse、hot-reload |

### 既存テスト修正

| ファイル | 修正 |
|----------|------|
| `test_190_ev_weighted_safety.py` | `min_spread_jpy` 期待値 700→500 |
| `test_336_yaml_code_drift_prevention.py` | `KNOWN_YAML_OVERRIDES` に `sell_preemptive_cv_kill_enabled` 追加 |

---

## §5 P2 据置事項

### P2-7: ceiling 引き上げ (0.25→0.30)

532# が明示的に指摘: ceiling 引き上げは upstream calibration (spread_adapt) が先行すべき。0.25→0.30 で mid からの距離が 20% 拡大し、現在の fill rate 33% からの更なる低下リスクが未定量。spread_adapt パイプラインのキャリブレーション完了後に再評価する。

### P2-8: pipeline 活性段監視

多くの offset pipeline 段が identity (×1.0) を返している現状。活性段の把握は有用だが、現在の P1 実装と直接関連しない。ログ可観測性改善 (534#) で一定の基盤ができたため、次回のパイプライン改善時に併せて対応する。

---

## §6 532# §8 全項目ステータス

| # | 優先 | 施策 | 状態 | 対応先 |
|---|------|------|------|--------|
| 1 | P0 | veto deadlock max_consecutive | ✅ | 534# |
| 2 | P0 | BTC=0 veto 緩和 | ✅ | 534# |
| 3 | P0 | log_cycle_no join key | ✅ | 534# |
| 4 | P1 | 12h/15h sell offset | ❌ 却下 | 533# (時間帯ハードコード拒否) |
| 5 | P1 | sell_dynamic_kill 事前指標 | ✅ | **535#** |
| 6 | P1 | min_spread_jpy 700→500 | ✅ | **535#** |
| 7 | P2 | ceiling 0.25→0.30 | ⏸ 据置 | upstream calibration 依存 |
| 8 | P2 | pipeline 活性段監視 | ⏸ 据置 | 次回 pipeline 改善時 |
