# 207# 206堅牢性修正 5件 + 片側連続実行制限

> **日付**: 2026-03-02  
> **前提**: 206# (3 P0 施策) コードレビューで発見された 5 件のバグ修正 + 205# §4.2 追加実装  
> **コミット**: `b4add440d`

---

## 1. 背景

206# で実装した Hard Skip / Toxic Veto / Per-Side DD の 3 施策について、
コードレベルの堅牢性レビューを実施した結果、5 件の不具合を発見。

加えて、205# §4.2 (Codex) で指摘されていた「片側残高枯渇時の連続実行制限」が
202# B (one_sided_balance_rescue_offset) の offset 保護だけでは不十分なため、
interval 延長メカニズムを追加実装した。

---

## 2. バグ修正 (§1–§5b)

### §1: toxic_veto 永続化欠落 (HIGH)

**問題**: `_toxic_veto` dict が `FillTestState` に含まれておらず、プロセス再起動で
veto 状態が消失。再起動直後に toxic fill サイドが再実行される。

**修正**:
- `FillTestState` に `toxic_veto: dict[str, int] | None = None` フィールド追加
- 3 箇所の `FillTestState(...)` 構築で `toxic_veto=dict(self._toxic_veto)` を設定
- 状態復元ロジック (regime_detector 有/無の両パス) で `_toxic_veto` を復元

**ファイル**: `resilience.py`, `fill_loop_orchestrator.py`

### §2: warmup 時の per-side PnL 未計算 (HIGH)

**問題**: `_warmup_daily_drawdown_from_records()` が全体 PnL のみ計算し、
`daily_pnl_bps_buy` / `daily_pnl_bps_sell` を設定しない。warmup 後に per-side halt
が正しく機能しない。

**修正**:
- warmup 内で buy/sell 別に PnL を集計
- 閾値超過時に `update_pnl(side=...)` で per-side halt を発動

**ファイル**: `fill_loop_orchestrator.py`

### §3: toxic veto off-by-one (MEDIUM)

**問題**: veto カウンタの decrement がサイクル開始時に実行されるため、
`toxic_fill_veto_cycles=3` 指定で実際には 2 サイクルしかブロックされない。

**修正**:
- decrement をサイクル末尾 (sleep 直前) に移動
- N サイクル指定 = N サイクル確実にブロック

**ファイル**: `fill_loop_orchestrator.py`

### §4: toxic veto の日替わりリセット欠落 (LOW-MEDIUM)

**問題**: UTC 日境界で `maybe_reset_day()` が呼ばれても `_toxic_veto` がクリアされず、
前日の veto が翌日まで残存する可能性。

**修正**:
- `maybe_reset_day()` が `True` を返した際に `self._toxic_veto = {}` を実行

**ファイル**: `fill_loop_orchestrator.py`

### §5b: veto ↔ per_side_dd bypass (MEDIUM)

**問題**: toxic veto で次サイド候補が切り替わる際、切替先が per_side_dd で halt 中
でも実行されてしまう。

**修正**:
- veto 切替先の判定に `self._daily_drawdown_guard.is_side_halted(alt_side)` を追加
- 両サイドとも封鎖されている場合は skip

**ファイル**: `fill_loop_orchestrator.py`

---

## 3. 片側連続実行制限 (§4 拡張 — 205# §4.2)

### 3.1 課題

202# B で one_sided_balance 時に rescue offset を適用する保護を入れたが、
連続して片側強制実行が繰り返される場合、offset だけでは out-of-balance 側の
資金枯渇を止められない。

### 3.2 設計

| 設定 | デフォルト | 説明 |
|---|---|---|
| `one_sided_consecutive_limit` | 5 | 片側強制実行の連続上限 (0=無制限) |
| `one_sided_consecutive_interval_mult` | 3.0 | 上限到達時の sleep interval 乗数 |

**動作**:
1. `run_single_cycle` 後、`_one_sided_balance` が True の場合にカウンタ increment
2. 正常サイド実行時にカウンタ reset (ログ出力)
3. カウンタが `one_sided_consecutive_limit` に到達すると:
   - WARNING ログ出力
   - sleep interval に `one_sided_consecutive_interval_mult` を乗算
4. 延長 interval により市場に資金リバランスの猶予を与える

### 3.3 既存乗数との共存

sleep interval は以下の全乗数の積として計算:

```
interval × soft_dd_mult × loss_cooldown_mult × one_sided_mult
```

各乗数は独立に 1.0 (無効) → N.0 (延長) で動作し、加算ではなく乗算で適用。

---

## 4. 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/resilience.py` | `FillTestState.toxic_veto` フィールド追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | §1–§5b 修正 + one-sided 制限 (+116/-18) |
| `scripts/v460/lib/fill_config.py` | `one_sided_consecutive_limit/interval_mult` 追加 |
| `configs/v460/fill_test.yaml` | one-sided 制限パラメータ追加 |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | 新規 14 テスト追加 |

---

## 5. テスト結果

```
55 passed (test_168_daily_drawdown_guard.py)
110 passed (test_168 + test_v460_core)
v460 全体: 584 passed, 5 failed (全て既存の無関係な失敗)
```

新規テスト:
- `TestFillTestStateToxicVeto` (3件): toxic_veto フィールドの存在・初期値・データ保持
- `TestPerSideDDWarmup` (2件): warmup per-side PnL 計算・halt 発動
- `TestToxicVetoDayReset` (1件): maybe_reset_day トリガ確認
- `TestPerSideDDAndVetoInteraction` (1件): veto + per_side_dd 両方封鎖時の動作
- `TestOneSidedConsecutiveConfig` (3件): config デフォルト値・カスタム値・無効化
- `TestOneSidedConsecutiveMultLogic` (4件): 乗数計算の under/at/over limit + disabled

---

## 6. 残課題 (次番号以降)

| 優先度 | 課題 | 根拠 |
|---|---|---|
| **P1** | Velocity SSOT 化 (§9.1) | 205# §3.2 / §9.1 — mid_trend_bps と velocity の二重系統解消 |
| **P1** | 204# H/I/J オフライン replay 評価 | 204# 分析で提案された改善施策の検証 |
| **P2** | OFI/PIN toxic flow 検知 (§9.3) | coincheck API 制約から proxy 指標で段階的導入 |
