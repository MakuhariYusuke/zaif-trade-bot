# 567# 計測基盤修正 I1–I3

**タイプ**: impl  
**日付**: 2026-03-23  
**前提**: 565# §3.1「即時実施すべきもの（計測基盤の修正）」  

---

## 概要

565# で特定した計測基盤の3つの欠陥を修正。
理論的提案（DRC, AS Risk Score, pipeline 改革）の実装に先立ち、
正確な現状把握を可能にする。

## I1: E3 sell窓崩壊の修正

**ファイル**: `scripts/v460/lib/pnl_measurer.py`

**問題**: E3 (60s/120s PnL) の計算ベースが `cfg.post_fill_wait_sec` (30s) 固定。
sell 側は `wait_sec = 90s` で計測するため、E3 の「60s PnL」が実質 90×2.0 ではなく
30×2.0 = 60s となり、一次計測 (90s) より前のタイミングで計測してしまう。
結果として「60s PnL」が「30s PnL」と同一時点に崩壊していた。

**修正**: `cfg.post_fill_wait_sec` → `wait_sec` に変更。
- buy: 30×2.0=60s, 30×4.0=120s（変更なし）
- sell: 90×2.0=180s, 90×4.0=360s（正しい延長窓に修正）

## I2: Execution Quality 分解セクション追加

**ファイル**: `scripts/v460/analysis/analyze_fill_logs.py`

**追加**: `section_execution_quality()` 関数。
Kissell & Glantz (2003) に基づくPnL分解:
```
PnL = spread_capture + adverse_selection_cost
```

Side×Regime / AS/Non-AS のクロス集計で、offset 戦略の質と市場毒性を独立評価する。

**現状**: `spread_capture_bps` フィールドが fill_recorder で未記録のため、
セクションは「未記録 (0/N fills)」を表示。fill_recorder への実装が次のステップ。

## I3: pre_clamp offset 分布の可視化

**ファイル**: `scripts/v460/analysis/analyze_fill_logs.py`

**追加**: `section_clamp_saturation()` に pre_clamp offset の分布（p50/p75/p90/p99）を追加。

**計測結果** (3/12–3/22):
```
buy:  clamped 397/402 (99%), pre_clamp avg=0.3449
      p50=0.3067, p75=0.3968, p90=0.4999, p99=0.7922
sell: clamped 352/354 (99%), pre_clamp avg=0.3154
      p50=0.2725, p75=0.3339, p90=0.4869, p99=0.7477
```

**重要な発見**:
- buy p50=0.3067 → ceiling 0.30 で中央値すら切られている
- ceiling を 0.35 に引上げれば buy の約半数が uncap される
- 562# P-B（ceiling 引上げ提案）の定量的根拠を提供

## 次の課題

- `spread_capture_bps` の fill_recorder 実装 → I2 セクションの有効化
- pre_clamp 分布に基づく ceiling 引上げ実験（0.35 → 0.40 段階的検証）
- I1 修正後の E3 sell データ蓄積と sell AS 検出精度の再評価
