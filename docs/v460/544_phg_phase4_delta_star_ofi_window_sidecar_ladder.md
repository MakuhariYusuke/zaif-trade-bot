# 544# Phase 4 初期実装: δ*→spread_adapt 動的バインド / OFI rolling window / sidecar ladder

> 540# §6 Phase 4「動的最適化」への第一歩  
> 日付: 2026-03-23

---

## §1 概要

Phase 3 (543#) で構築した計測基盤を、Phase 4 の動的最適化に接続する初期実装。
**全変更は既存 pipeline の意味を保ちつつ、理論的根拠に基づく動的化を追加**。

| # | 施策 | 種別 | Phase 4 目標との対応 |
|---|------|------|---------------------|
| 4-1 | δ* → spread_adapt 動的 narrow 閾値 | Code | A-S 最適スプレッド live 接続 |
| 4-2 | OFI rolling deque (50 cycles) | Code | Prediction Hub 基盤強化 |
| 4-3 | sidecar ladder step 0.15→0.20 | YAML | SAC 影響力拡大 (377# 計画内) |
| 4-T | δ*_bps / spread_bps テレメトリ | Code | 計測→最適化フィードバック |

---

## §2 Phase 4-1: δ* → spread_adapt 動的 narrow 閾値

### 問題

spread_adapt の `narrow_spread_bps: 2.5` (固定) は市場条件の変化に追従しない。
A-S δ* は理論的に「このスプレッドより狭い場合、逆選択リスクが高い」を教えてくれるが、
spread_adapt とは独立に動作していた。

### 解決

δ*_bps が固定 `narrow_spread_bps` より大きい場合、動的に引き上げ:

$$\text{narrow\_eff} = \min\left(\delta^*_{\text{bps}},\ \text{narrow\_bps}_{\text{config}} \times 3.0\right)$$

- $\delta^*_{\text{bps}} = \delta^*_{\text{ratio}} \times \text{spread\_bps}$
- 上限キャップ: 固定値の 3 倍まで (σ 暴騰時の過剰反応防止)
- δ* 未算出時: 固定値にフォールバック (安全)

### テレメトリ

`offset_stages` に以下を追加:
- `delta_star_bps`: δ* の bps 換算値 (理論スプレッド)
- `spread_bps`: 実測スプレッド bps
- 比率 `delta_star_bps / spread_bps > 1.0` → 理論がより広いスプレッドを推奨

---

## §3 Phase 4-2: OFI Rolling Window

### 問題

543# の OFI-Lite はスカラー値 (`_last_ofi_lite`) のみ。
cycle 間のノイズに弱く、トレンド方向の判定ができない。

### 解決

`collections.deque(maxlen=50)` で直近 50 cycles (≈100 分) の OFI 履歴を保持。
rolling mean を `offset_stages["ofi_mean"]` に記録。

- `ofi_mean > 0`: 直近の買い圧力優位 (bid volume 増加傾向)
- `ofi_mean < 0`: 直近の売り圧力優位 (ask volume 増加傾向)
- `ofi_mean ≈ 0`: 均衡

### 将来接続

- spread_adapt boost の OFI-aware 変調 (Phase 5+)
- SAC 特徴量としての OFI mean/trend 入力

---

## §4 Phase 4-3: Sidecar Ladder Step 0.15→0.20

### 377# Ladder 計画

| Step | max_boost_bps | 期間 | 状態 |
|------|--------------|------|------|
| 1 | 0.10 | 24h | ✅ 完了 |
| 2 | 0.15 | 24h+ | ✅ 完了 (長期運用で安定確認) |
| 3 | 0.20 | — | ✅ **本実装** |

### 安全性

- 0.20 bps は median spread (≈2.2bps) の **9%** → 安全域内
- 375# hard ceiling (0.20) に到達 → 次の拡大 (0.30+) には ceiling 引上げが必要
- dead_zone: 0.10 のまま (|bias| ≤ 0.10 でノイズ除去)

---

## §5 改善点発見 (Explore 調査結果)

### 既存 ztb/ 資産で活用可能なもの

| ztb モジュール | 用途 | Phase 4 統合可能性 |
|---------------|------|-------------------|
| `ztb/trading/signal/calibration_map.py` | EWMA online learning (win_rate, EV) | SAC confidence → sidecar boost 変調 |
| `ztb/features/market_theory.py` | Parkinson σ, VPIN proxy, Kyle λ batch mode | backtest 整合性検証 |
| `ztb/utils/drift_detection.py` | PSI + KS test | OFI/Toxicity 分布ドリフト監視 |
| `ztb/features/scalping.py` | `order_flow_imbalance()` (wick-body ratio OFI) | OFI-Lite との cross-validation |
| `ztb/adaptation/hyperparameter_adaptation_system.py` | 動的ハイパラ調整 | Toxicity warn/caution 閾値の適応 |

### fill_test 未統合の重要発見

- **CalibrationMap / CalibrationGate**: fill_test には未接続。環境側 (`fast_intraday_env_v456.py`) のみで使用。
  → Phase 4-4 (将来): CalibrationMap.get_stats() → sidecar confidence weight
- **Feature drift detection**: OFI/Toxicity の分布シフト無監視
  → `ztb/utils/drift_detection.py` を OFI 時系列に適用可能
- **spread_adapt の narrow/wide boost が YAML 非連動**: 既存 ztb/ 純粋関数は config 依存なし → 正しい設計

---

## §6 テスト結果

- 3806 passed, 9 skipped (pre-existing: test_143/260/336 除外)
- Phase 4 コード変更による新規テスト失敗: なし

---

## §7 次ステップ (Phase 4 継続)

| 優先度 | 施策 | 根拠 |
|--------|------|------|
| P1 | sidecar hard ceiling 引上げ (0.20→0.30) | SAC ±5.0bps への段階的拡大 |
| P1 | CalibrationMap → sidecar confidence 統合 | 538# §6「第三の道」learned calibration |
| P2 | OFI mean → spread_adapt boost 変調 | 買い圧力時に buy aggressive, 売り圧力時に sell defensive |
| P2 | δ* → executor stage 参照 (execution-level floor) | 現在 pre-order のみ。executor にも δ* 情報を伝搬 |
| P3 | drift detection for OFI/Toxicity | 分布シフト監視 (ztb/utils/drift_detection.py 活用) |
