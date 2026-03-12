# 286# 282#–284# 課題包括的解決 + 市場理論補強

**Phase**: ph2 G1.1-exec  
**Type**: fix / enhancement  
**Date**: 2026-03-06  
**Parent**: 285# (`a209d23dc`)  

---

## 概要

283# (Codex レビュー) と 284# (Gemini 3.1 Pro レビュー) で指摘された残課題を
包括的に解決。P0 は 285# で対応済み、本 286# は P1/P2/MEDIUM 全項目を実装。

さらに **市場理論に基づく順張り/逆張り両戦略の補強** を実施:
- Ho & Stoll (1981): 在庫リスク最適化 → buy kill 緩和
- Glosten-Milgrom (1985): 情報非対称性 → buy AS 防御 + 強制買い遅延
- Kyle (1985): 注文分割と情報伝達 → guard dominance 解消

---

## 変更一覧

### 1. Lock Manager portalocker 強化 (283# P0-1 強化)

**ファイル**: `scripts/v460/lib/lock_manager.py`

- **OS レベルロック**: `portalocker` による `LOCK_EX|LOCK_NB` をレイヤー 2 として追加
- **ゾンビ待機**: `_wait_for_pid_exit(pid)` — psutil で PID 消滅を最大 30 秒ポーリング
- **グレースフルフォールバック**: `_HAS_PORTALOCKER=False` 時はレイヤー 1 (O_CREAT|O_EXCL) のみ
- **リリース強化**: OS ロック先行解放 → `.lock` + `.os_lock` 両ファイル削除

**理論的根拠**: Split-Brain は単一ファイル存在チェックでは検出のみ。OS レベル排他制御で
プロセスクラッシュ時のロック残留を防止。

### 2. Split-Brain 事後検出 (283# P0-1 補完)

**ファイル**: `ztb/metrics/fill_quality.py`

- `detect_split_brain(records, overlap_window_sec=300.0) -> list[dict]`
- 隣接レコードの `run_id` / `pid` を走査、time window 内の重複を CRITICAL ログ
- 事後監査用 — Lock Manager が突破された場合のセーフティネット

### 3. Events start/stop ペア保証 (283# P0-3)

**ファイル**: `scripts/v460/lib/fill_test_cli.py`

- `finally` ブロックで crash 時も必ず stop イベントを記録
- 変更: `if stop_reason and not stop_reason.startswith("crash:")` → `if stop_reason:`
- セッション境界の完全なペアリングを保証

### 4. buy_dynamic_kill 在庫連動緩和 (284# P1 + Ho & Stoll 1981)

**ファイル**: `ztb/risk/sell_dynamic_kill.py`, `scripts/v460/lib/fill_config.py`, `scripts/v460/lib/fill_loop_orchestrator.py`

- `check_kill()` に `threshold_offset_bps` パラメータ追加
- `_is_side_killed()` で在庫不均衡 (`inv_net_imbalance`) から offset 計算:
  - BTC 不足 (imbalance < 0) → `offset = min(|imbalance| × scale, max_bps)`
  - kill 閾値を緩和し、在庫補充の buy を過度に抑制しない

**理論的根拠 (Ho & Stoll 1981)**:
マーケットメイカーの最適スプレッドは在庫ポジションの関数。在庫偏りが大きいほど、
偏り解消方向の取引に対するスプレッドを縮小すべき。

**Config**:
- `buy_dynamic_kill_inv_relaxation_enabled: bool = False`
- `buy_dynamic_kill_inv_relaxation_scale: float = 0.5`
- `buy_dynamic_kill_inv_relaxation_max_bps: float = 0.3`

### 5. 強制買い KPI 分離 (284# P1)

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`, `scripts/v460/lib/fill_config.py`

- `RunSessionState` に `forced_buy_fill_count`, `forced_buy_pnl_sum_bps`, `normal_buy_fill_count`, `normal_buy_pnl_sum_bps` 追加
- `_process_post_cycle()` で `record.balance_forced_switch` フラグに基づいて分離集計
  - (**287#**: 初期実装で `record.balance_forced` と誤記。正しい `FillRecord` 属性は `balance_forced_switch`。
    属性名不一致により `AttributeError` でプロセスクラッシュ。`e7d2f50d9` で修正済み。
    詳細: [287_ph2_fix_balance_forced_switch_attribute.md](287_ph2_fix_balance_forced_switch_attribute.md))
- 定期ログ: `[286# P1-5] Buy KPI split: forced=N fills (+X.XXbps avg), normal=M fills (+Y.YYbps avg)`

**Config**: `forced_buy_kpi_tracking_enabled: bool = True`

### 6. Buy 側 AS 防御 — Glosten-Milgrom (1985)

**ファイル**: `scripts/v460/lib/maker_price.py`, `scripts/v460/lib/fill_config.py`

- `_apply_buy_as_guard()` パイプラインステージ追加 (`_apply_imbalance_risk` 後)
- velocity ≤ threshold (価格下落中) → buy offset を拡大 (`_scale_offset_ratio()`)
- 最大拡大率 `max_offset_ratio` でクリップ

**理論的根拠 (Glosten-Milgrom 1985)**:
価格下落時は情報を持つトレーダーが売り注文を出す確率が上昇。
buy maker は逆選択リスクが増大するため、スプレッドを拡大して防御。

**Config**:
- `buy_as_guard_enabled: bool = False`
- `buy_as_guard_velocity_threshold_bps: float = -5.0`
- `buy_as_guard_offset_mult: float = 1.5`
- `buy_as_guard_max_offset_ratio: float = 0.5`

### 7. 強制買い遅延実行 — Glosten-Milgrom (1985)

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`, `scripts/v460/lib/fill_config.py`

- `balance_forced=True` + `next_side=="buy"` + velocity ≤ threshold → delay カウンタ設定
- delay 中はサイクルスキップ (guard reason: `forced_buy_delay` → MARKET 分類)
- 「下落中の強制買いは待つ勇気」を実装

**Config**:
- `forced_buy_delay_enabled: bool = False`
- `forced_buy_delay_velocity_threshold_bps: float = -5.0`
- `forced_buy_delay_cycles: int = 3`

### 8. Guard Dominance 解消 (283# MEDIUM-4)

**ファイル**: `scripts/v460/lib/guard_reason_classifier.py`, `tests/unit/v460/test_244_guard_reason_classification.py`

- 7 件のガードを SYSTEM → RECOVERY に再分類:
  - `balance_forced_halt_block`
  - `one_sided_freeze_skip`, `one_sided_cooldown_skip`
  - `degraded_liquidation_duty_skip`, `degraded_liquidation_active`
  - `inventory_escape_duty_skip`, `inventory_escape_active`
- 新規: `forced_buy_delay` → MARKET
- 効果: SYSTEM ガード発火回数の削減 → MARKET/RECOVERY の比率が上昇し、
  市場起因ガードの分析精度が向上

**理論的根拠 (Kyle 1985)**:
ガード発火パターンの分類精度は、戦略の情報感度分析の基盤。
リカバリー動作をシステムガードとして集計すると、市場適応性の評価にバイアスが生じる。

---

## 順張り/逆張り戦略への影響

### 順張り (Trending)
- **buy_dynamic_kill 緩和** (Todo 4): トレンドに沿った buy が過度に kill されない
- **trending_boost offsets** (既存: up_buy=0.7, up_sell=1.8) との共存で、
  上昇トレンド中の在庫補充が殺されにくくなる

### 逆張り (Ranging/Mean-Reversion)
- **Buy AS 防御** (Todo 6): 急落時の buy offset 拡大で逆選択コストを低減
- **強制買い遅延** (Todo 7): 下落中は GM 理論に基づき「待つ」→ より良い価格で買える
- **KPI 分離** (Todo 5): forced buy と normal buy の品質を分離監視 → 戦略評価精度向上

---

## テスト

- **新規**: `test_286_comprehensive_resolution.py` — 37 テストケース (9 クラス, 287# 回帰テスト 2 件含む)
- **修正**: `test_244_guard_reason_classification.py` — 再分類に対応
- **結果**: v460 suite 3887 passed, 32 skipped, 0 failed

---

## 新規 Config フィールド (全て安全なデフォルト値)

| フィールド | デフォルト | 意味 |
|---|---|---|
| `buy_dynamic_kill_inv_relaxation_enabled` | `False` | 在庫連動 kill 緩和 |
| `buy_dynamic_kill_inv_relaxation_scale` | `0.5` | 不均衡→offset 変換係数 |
| `buy_dynamic_kill_inv_relaxation_max_bps` | `0.3` | offset 上限 (bps) |
| `forced_buy_kpi_tracking_enabled` | `True` | 強制買い KPI 分離 |
| `forced_buy_delay_enabled` | `False` | 強制買い遅延 |
| `forced_buy_delay_velocity_threshold_bps` | `-5.0` | 遅延発動閾値 |
| `forced_buy_delay_cycles` | `3` | 遅延サイクル数 |
| `buy_as_guard_enabled` | `False` | Buy AS 防御 |
| `buy_as_guard_velocity_threshold_bps` | `-5.0` | AS 防御発動閾値 |
| `buy_as_guard_offset_mult` | `1.5` | offset 拡大倍率 |
| `buy_as_guard_max_offset_ratio` | `0.5` | offset 上限比率 |

---

## 変更ファイル一覧

1. `scripts/v460/lib/lock_manager.py` — portalocker + zombie wait
2. `ztb/metrics/fill_quality.py` — detect_split_brain()
3. `scripts/v460/lib/fill_test_cli.py` — start/stop finally
4. `ztb/risk/sell_dynamic_kill.py` — threshold_offset_bps
5. `scripts/v460/lib/fill_config.py` — 11 新規 config フィールド
6. `scripts/v460/lib/fill_loop_orchestrator.py` — inv relaxation, KPI, delay
7. `scripts/v460/lib/maker_price.py` — _apply_buy_as_guard()
8. `scripts/v460/lib/guard_reason_classifier.py` — SYSTEM→RECOVERY 再分類
9. `tests/unit/v460/test_244_guard_reason_classification.py` — 再分類対応
10. `tests/unit/v460/test_286_comprehensive_resolution.py` — 37 テスト (287# 回帰テスト 2 件追加)
11. `docs/v460/286_ph2_fix_282_284_comprehensive_resolution.md` — 本ドキュメント
