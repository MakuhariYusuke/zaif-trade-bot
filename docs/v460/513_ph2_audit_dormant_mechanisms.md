# 513# 実装済み機構の活用度監査 + SAD/MCB YAML化

## 概要
実装済みだが運用上の活躍が不明確な機構を調査し、活用度を分類。
Dead code の特定と、SAD/MCB の YAML 設定セクション追加、PPG 活用検討を提示する。

## 調査結果サマリ

| # | 機構 | 状態 | 判定 |
|---|------|------|------|
| 1 | Skip Gate ML | enabled=true, 4モデル | ✅ 非常にアクティブ |
| 2 | Macro Trend boost | enabled=true, boost 適用中 | ✅ アクティブ |
| 3 | Regime Policy (C/D/Chase) | 全3機能 enabled=true | ✅ アクティブ |
| 4 | A-S Reservation Price | enabled=true, σ推定つき | ⚠️ アクティブ（効果量不明） |
| 5 | Adaptation Engine | enabled=false (122#) | ❌ 無効化済み |
| 6 | Fast Fill Defense | enabled=true, 2.0-2.5x boost | ✅ アクティブ |
| 7 | Spread Anomaly Detector | YAML化済み, enabled=false | ⚠️ 待機状態 |
| 8 | Phantom Position Guard | 未インスタンス化 | ⚠️ 活用検討中 |
| 9 | Micro Circuit Breaker | YAML化済み, enabled=false | ⚠️ 待機状態 |
| 10 | Dynamic Kill Manager | enabled=true, 広範統合 | ✅ 非常にアクティブ |

## 各機構の詳細

### 1. Skip Gate ML フィルター — ✅ 非常にアクティブ
- 4 モデル（pnl30_buy, pnl30_sell, pnl120_buy, pnl120_sell）
- Adaptive threshold（target skip rate: buy=15%, sell=20%）
- Regime 別閾値、時間帯 hard skip、velocity skip
- EV-weighted scoring（188#/190#/193#）
- **判定**: コア機構。最も YAML チューニング密度が高い。

### 2. Macro Trend boost — ✅ アクティブ
- 5m/15m OLS slope → STRONG_UP/WEAK_UP/NEUTRAL/WEAK_DOWN/STRONG_DOWN
- `conflict_action: log` は regime downgrade にのみ影響（観測モード）
- **offset boost は独立動作**: offset_pipeline 458# で sell/buy に 1.3-1.6x 適用
- timeout 調整: sell_timeout_weak_up=12s, strong_up=6s
- **判定**: `conflict_action: log` は regime 管理の安全策。boost 自体は機能している。

### 3. Regime Policy (C/D/Chase) — ✅ アクティブ
- **YAML で全3機能 enabled=true** (Explore 調査の「disabled」報告は不正確)
  - `dynamic_cycle.enabled: true` → trending 60s, ranging 120s
  - `dynamic_wait.enabled: true` → side×regime 別の post-fill wait
  - `chase.enabled: true` → trending 時 3bps drift で即 reprice
- YAML→RegimePolicyConfig マッピングが `from_dict()` で正しく変換確認済み
- **判定**: アクティブ。ただし chase の実発火率は trending regime の出現頻度に依存。

### 4. A-S Reservation Price — ⚠️ アクティブ（効果量不明）
- `as_reservation.enabled: true`, σ_Parkinson 推定
- Kyle λ (`impact_mult: 0.5`, cap 5%) + Amihud ILLIQ (`max_mult: 1.5`)
- compute() パイプラインのステージとして毎サイクル適用
- **問題**: AS 寄与量（~1-5%）は下流ステージ（regime ×1.8, VG ×2.0）に埋もれる
- **改善案**: ステージ別寄与量の isolated logging 追加で検証可能に

### 5. Adaptation Engine — ❌ 意図的に無効化
- `adaptation.enabled: false` (122# R2: 因果分離のため)
- `try_auto_adapt()` は毎サイクル呼ばれるが即 return
- Kelly ceiling（静的 equity）と dynamic loss cap のみ稼働
- **根拠**: 因果推論が不十分な状態での自動適応は逆効果リスク

### 6. Fast Fill Defense — ✅ アクティブ
- `enabled: true`, threshold buy=8s / sell=15s
- adverse fill 検出時 2.0-2.5x offset boost
- `boost_release_streak: 3` で正常 fill 3 連続後にリリース
- **判定**: シンプルだが高信頼度の adverse selection 防御。

### 7. Spread Anomaly Detector — ⚠️ 待機状態（YAML化済み）
- `spread_anomaly_detector.py`: SADLevel (NORMAL/WIDE/DRY/FROZEN) 全実装済み
- **513# で YAML 設定セクション追加** (`enabled: false`)
- `_sad: SpreadAnomalyDetector | None = None` で宣言のみ（インスタンス化未実装）
- `_feed_mcb_sad()` で `if self._sad is not None` ガード → 常に no-op
- テスト: `test_211_spread_anomaly_detector.py` に 8 テスト存在
- **推定コスト**: ~200 LOC のデッドコード

### 8. Phantom Position Guard — ⚠️ 活用検討中
- `phantom_position_guard.py`: 250+ LOC、TTL ベース残留ポジション検出
- **反省会での活用が有望**: status_unknown 注文の事後分析に有用
- config dataclass なし（クラス定数のみ）、**未インスタンス化**
- `_phantom_guard: PhantomPositionGuard | None = None` 宣言のみ
- TYPE_CHECKING import のみ
- テスト: `test_237_phantom_position_guard.py`, `test_252_*.py` に存在
- **推定コスト**: ~300 LOC のデッドコード

### 9. Micro Circuit Breaker — ⚠️ 待機状態（YAML化済み）
- `micro_circuit_breaker.py`: MCBLevel (NORMAL/CAUTION/WARNING/HALT) 全実装
- **513# で YAML 設定セクション追加** (`enabled: false`)
- `_mcb: MicroCircuitBreaker | None = None` 宣言のみ（インスタンス化未実装）
- `_mcb: MicroCircuitBreaker | None = None` 宣言のみ
- `_feed_mcb_sad()` に `if self._mcb is not None` ガード → 常に no-op
- テスト: `test_211_micro_circuit_breaker.py` に存在
- **推定コスト**: ~250 LOC のデッドコード
- **備考**: `dynamic_cycle_interval` (306#) が一部の役割を代替

## デッドコード回収の判断

### 削除候補（~750 LOC）
| ファイル | LOC | 理由 |
|----------|-----|------|
| `spread_anomaly_detector.py` | ~200 | YAML化済み、インスタンス化待ち |
| `phantom_position_guard.py` | ~300 | 反省会活用を検討中 |
| `micro_circuit_breaker.py` | ~250 | YAML化済み、インスタンス化待ち |

### 削除しない理由（考慮事項）
- 3 機構ともテストが存在し、設計は完了している
- 将来的に市場環境が変化した時 (e.g. 流動性枯渇、API 不安定) に必要になる可能性
- デッドコードだがメンテコストは低い（他コードとの結合がほぼゼロ）

### 推奨
- **SAD/MCB**: YAML 化完了。有効化にはオーケストレータでのインスタンス化実装が必要
- **PPG**: 反省会（事後分析）での status_unknown 注文追跡に活用検討
- **A-S 寄与量の可視化**: 効果量が不明なため、ステージ別 logging で検証
- **adaptation.enabled**: 122# の因果分離理由を YAML コメントに記載（既にある）

## 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | `spread_anomaly_detector:` + `micro_circuit_breaker:` セクション追加 |
