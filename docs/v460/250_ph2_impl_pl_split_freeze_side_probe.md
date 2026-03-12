# 250# P/L 3分離・freeze side・quiescence deadlock・probe廃止基盤

最終更新: 2026-03-03  
前提: 249# (commit `89dcdfc89`)

## 概要

247#/248# レビューの残課題 (P1-4, P1-5, P1-6) を解消し、
コード品質スイープで発見した edge case を修正。
市場理論 (Glosten-Milgrom) に基づくコメント補強を追加。

## 変更一覧

### 1. P/L 3分離追跡基盤 (248# P1-5)

**ファイル**: `fill_loop_orchestrator.py`

マーケットメイキングの P/L を3要素に分離して進捗ログに表示:
- **spreadPnL**: スプレッド収益 (cumulative_pnl_jpy)
- **btcMTM**: BTC 在庫の MTM 評価損益 (249# で追加済み)
- **adverseFills**: 逆選択による被害件数・累積 bps

逆選択 (adverse selection) はスプレッド収益を侵食する主要因であり、
Glosten-Milgrom の情報非対称モデルにおける情報リスクコストに相当する。

**実装**:
- `cumulative_adverse_count`, `cumulative_adverse_bps` を resume ループとインクリメンタル追跡の両方で計算
- 進捗ログに `[250# AS] adverseFills=N (X.X%), cumASbps=±Y.Ybps` を追加

### 2. freeze/cooldown side 紐付け (247# P1-4)

**ファイル**: `fill_loop_orchestrator.py`

従来の one-sided freeze/cooldown は両 side を無差別にブロックしていた。
247# §1.10 の指摘に基づき、freeze/cooldown を発動した side のみをブロックし、
反対 side は通常実行を許可するよう変更。

**実装**:
- `_one_sided_frozen_side: str | None` クラス変数追加
- Stage 2 (cooldown) / Stage 3 (freeze) エスカレーション時に `next_side` を記録
- skip 判定時に `_frozen_side is None or _frozen_side == next_side` でフィルタ
- streak 終了時に `None` にリセット (従来互換)

### 3. quiescence + balance_forced deadlock 防御 (code sweep finding)

**ファイル**: `cycle_gate_aggregator.py`

249# の quiescence (dual_kill 時に静観) と balance_forced (ポジションリバランス要求) が
同時に発生すると、永久に取引できない deadlock が生じうる edge case を修正。

**実装**:
- `balance_forced=True` かつ `degraded_liquidation_enabled=True` の場合、
  quiescence を緩和して degraded liquidation での縮退清算を許容
- `balance_forced=False` の場合は従来通り pure quiescence (resting)

### 4. probe 廃止コメント補強 (247# P1-6)

**ファイル**: `ztb/risk/sell_dynamic_kill.py`

probe (kill 中の強制取引) が Glosten-Milgrom の情報非対称理論と矛盾することを
DynamicKillConfig と check_kill() の両方にコメントとして明記。

理論的根拠:
- 情報優位者への無防備な流動性供給 → 逆選択コスト増大
- 242# `toxic_kill_stale_multiplier` で実質的に無効化済み
- `max_stale_kill_cycles=0` で明示的に無効化可能
- 将来的にデフォルト `0` (無効) への変更を検討

### 5. rearm halt_triggered_at 防御コメント (code sweep finding)

**ファイル**: `scripts/v460/lib/daily_drawdown_guard.py`

249# rearm 発動時の `halt_triggered_at = time.time()` 更新が機能的には
不要 (`cooldown_rearmed=True` がガード) だが、状態永続化とログ調査用に
防御的に保持している旨をコメントで明記。

## テスト

新規テスト: 23 件 (`test_250_pl_split_freeze_side.py`)

| クラス | テスト数 | 内容 |
|--------|----------|------|
| TestPLSplit250 | 6 | adverse selection 累積ロジック (resume + incremental) |
| TestFreezeSideTracking250 | 9 | freeze/cooldown side 紐付け |
| TestQuiescenceDeadlockDefense250 | 6 | quiescence + balance_forced deadlock 防御 |
| TestProbeDisable250 | 2 | probe 無効化 (max_stale=0) |

全 v460 テスト: **3472 passed** (3449 + 23)

## 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | P/L 3分離追跡 + freeze side 紐付け |
| `scripts/v460/lib/cycle_gate_aggregator.py` | quiescence deadlock 防御 |
| `ztb/risk/sell_dynamic_kill.py` | probe 廃止コメント補強 |
| `scripts/v460/lib/daily_drawdown_guard.py` | rearm コメント補強 |
| `tests/unit/v460/test_250_pl_split_freeze_side.py` | 新規テスト 23 件 |
| `docs/v460/index.md` | ヘッダ更新 + 248#/249# エントリ追加 |
| `docs/v460/250_ph2_impl_pl_split_freeze_side_probe.md` | 本ドキュメント |

## 残課題 (未着手 — 将来イテレーション)

- **P1-1**: Sell Asymmetric Mode (最大収益インパクト)
- **P1-2**: PhantomPositionGuard 三値化
- **P1-3**: PhantomPositionGuard buy側残高照合
- **P2-1**: Feasible Quote 完全計算
- **P2-2**: Inventory Target Band 導入
- **P2-3**: God Object 回帰の抑制
