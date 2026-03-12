# 295# Config Hot-Reload 包括的カバレッジ修正 & セルフレビュー

> **目的**: config_hot_reload.py の `_HOT_RELOADABLE_FIELDS` 未登録フィールドを包括的に追加し、  
> YAML 変更がプロセス再起動なしで反映されるカバレッジを大幅向上  
> **日付**: 2026-03-09  
> **前提**: 292# v3 で 5 フィールド追加 → 残り 157 フィールドが未登録  
> **関連**: 290#/291# 外部レビュー対応漏れ調査 + セルフレビュー

---

## 1. Hot-Reload カバレッジ問題

### 1.1 背景

292# v3 で `stale_reprice_min_delta_jpy` と `forced_buy_delay_*` 4 フィールドを  
`_HOT_RELOADABLE_FIELDS` に追加したが、`FillTestConfig` 全 368 フィールド中、  
155 フィールドのみ登録 → **カバレッジ 42%**。  
運用上変更頻度の高いパラメータが hot-reload 不可能で、YAML 変更のたびにプロセス再起動が必要。

### 1.2 分析結果

全 368 フィールドを以下に分類:

| 分類 | フィールド数 | 対応 |
|---|---|---|
| **登録済み** (292# v3 時点) | 155 | — |
| **運用パラメータ** (hot-reload 候補) | 157 | ✅ 全件追加 |
| **構造・初期化専用** (hot-reload 不適切) | 56 | 除外 |

### 1.3 構造フィールド除外根拠 (56 件)

- **パス/ファイル系** (`results_dir`, `skip_gate_model_path*` 等) — 再構築影響大
- **接続設定** (`cb_*`, `lock_*`, `heartbeat_*` 等) — API クライアント再構築必要
- **ログ設定** (`file_log_level`, `log_*`) — ロガー再構成必要
- **レジーム初期化** (`regime_window`, `regime_min_confidence` 等) — 検出器再構築
- **取引所定数** (`maker_fee_bps`, `taker_fee_bps`, `min_order_btc`, `symbol`) — 不変

### 1.4 最終カバレッジ

| 項目 | 値 |
|---|---|
| 総フィールド数 | 368 |
| Hot-Reload 対象 | **312** (84.8%) |
| 構造フィールド | 56 (15.2%) |

---

## 2. 追加フィールド一覧 (カテゴリ別)

- **AS (Avellaneda-Stoikov / delta-star)**: 9 件
- **Amihud / Kyle**: 6 件
- **Balance / Inventory**: 6 件 + Skewing 6 件
- **Degraded Liquidation**: 4 件
- **Inventory Escape**: 2 件
- **DD Cooldown / Recovery**: 9 件
- **One-Sided Escalation**: 4 件
- **Dual Kill Quiescence**: 3 件
- **MCB (Market Circuit Breaker)**: 6 件
- **SAD (Spread Anomaly Detector)**: 5 件
- **Buy AS Guard**: 4 件
- **SkipGate EV Warning / Adaptive**: 12 件
- **Imbalance / OBI**: 6 件
- **Narrow/Wide Spread**: 7 件
- **Early Exit**: 4 件
- **E3 Measurement**: 2 件
- **Dynamic Kill Advanced**: 12 件
- **VG Advanced**: 5 件
- **Loss / PnL**: 8 件
- **Lot / Adapt**: 8 件
- **Low Vol / Ranging / Regime**: 5 件
- **Smart Side / FFD / Stale / Preflight**: 12 件
- **Misc Operational**: 6 件

### MCB / SAD の _COMPONENT_REBUILD_PREFIXES 不要

MCB・SAD は `maybe_reload` で値変更されたら次サイクルで直接参照される設計。  
`_rebuild_mcb()` 等のメソッドは存在せず、再構築コールバックは不要。

---

## 3. セルフレビュー修正

### 3.1 SR-1 (LOW): `as_deadzone_bps` 重複エントリ

**問題**: 既存の offset セクションと 295# AS セクションの両方に `as_deadzone_bps` が記載。  
`frozenset` なので動作上は無害だが、ソース上の不整合。

**修正**: 295# AS セクション側を削除し、コメントで既存登録を参照。

---

## 4. 290# / 291# 対応漏れ調査結果

全 P0/P1 項目は対応済み。残存は P2/将来課題のみ:

| 出典 | 項目 | 状態 | 対応 |
|---|---|---|---|
| 290# §4.1 | FillRecord 3 fields | ✅ 292# | ev_score_pretrade, offset_mult, decision_path |
| 290# §4.2 | Block bootstrap 評価 | P2/将来 | 分析手法の改善 |
| 290# §4.3 | git_sha 正規化 | ✅ 既存 | `[:12]` で統一済み |
| 290# §6 P0 | Buy KPI alpha/forced 分離 | ✅ 286# | forced_buy_kpi_tracking |
| 290# §6 P1 | Night+ranging 保守化 | ✅ 既存 | ranging_offset_discount, skip_ranging_buy_low_vol 等 |
| 290# §6 P1 | EV path eval model_used 置換 | ✅ 292# | decision_path |
| 290# §6 P2 | Ranging 専用 buy policy | P2/将来 | 要追加分析 |
| 291# P0 | FillRecord 3 columns | ✅ 292# | (is_balance_forced は既存) |
| 291# P1 | Queue Priority 保護 | ✅ 292# | reprice deadband min_delta_jpy=500 |
| 291# P1 | Forced Buy Toxicity Veto | ✅ 286#+292#+294# | forced_buy_delay + regime-aware + max_consecutive |

### セルフレビュー所見

**294# forced_buy_delay deadlock fix**: ロジック追跡で正常性を確認。

- 連続ブロック上限 (`max_consecutive=10`) で永久デッドロック防止
- 突破後は `consecutive` がリセットされ、再度 delay 可能 → 完全 OFF ではなく段階的保護維持
- `if remaining > 0 and buy` チェックは最外ブロックにあり、非 forced buy にも適用される
  - これは 286# 初期設計から存在 (velocity が悪い時は全 buy が危険)
  - 動作上問題なし (delay は 3 サイクルで短い、非 forced buy が連続で来る状況は稀)

---

## 5. テスト

32/32 PASSED (`test_292_observability.py`):
- 既存 22 件 (292# + 294#)
- 新規 10 件 (295# hot-reload 包括カバレッジ):
  - AS, Amihud/Kyle, Inventory, Degraded/Escape, DD Cooldown/Recovery
  - MCB/SAD, SkipGate Advanced, Dynamic Kill Advanced
  - ソース重複検知, 最低カバレッジ数 (>=310)

v460 全体: 3924 passed, 32 skipped (リグレッションなし)

---

## 6. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/config_hot_reload.py` | 157 フィールド追加 + `as_deadzone_bps` 重複除去 |
| `tests/unit/v460/test_292_observability.py` | 295# 包括カバレッジテスト 10 件追加 |
| `docs/v460/295_ph2_hot_reload_comprehensive_coverage.md` | 本ドキュメント |
