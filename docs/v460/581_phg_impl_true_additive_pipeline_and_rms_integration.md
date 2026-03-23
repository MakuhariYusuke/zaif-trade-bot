# 581# [phg] [impl] True Additive Pipeline の実装完了と RMS 結合による爆発鎮鎮報告

> **ステータス**: 実装完了・A/Bテスト準備 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 580# (設計刷新), 577# (監査指摘)

---

## 1. 実施内容：執行エンジンの「真の加法化」

577# で指摘された「名前だけの加法化」を脱却し、`offset_pipeline.py` を数理的に刷新した。

### 1.1 RMS (Root Mean Square) 結合の導入
従来の 9 段乗算チェーン（$\prod m_i$）を廃止し、独立したリスク増分（$\Delta R_i$）を幾何学的に統合するモデルへ移行した。
- **数式**: $Offset_{total} = Base\_Offset + \sqrt{\sum (\Delta R_i)^2}$
- **効果**: 単一リスク発生時には従来と同等の回避性能を維持しつつ、複合リスク発生時の指数的な爆発（オフセットが 0.50 を超える現象）を構造的に抑制する。

### 1.2 A/B テスト用フォールバック構造
`experimental_additive_pipeline` フラグにより、新旧ロジックを完全に分離した。
- **True (加法)**: RMS 結合 + ロバスト入力（$\sigma_{robust}$, $OFI_{med}$）。
- **False (乗法)**: 従来の 9 段チェーン。
これにより、稼働中のシステムを破壊することなく、新モデルの優位性を Kissell & Glantz 指標（I2）で検証可能となった。

---

## 2. 測定基盤の正常化

### 2.1 FillRecord スキーマの更新
`ztb/metrics/fill_quality.py` の `FillRecord` dataclass に `spread_capture_bps` および `adverse_selection_cost_bps` を物理的に追加。
- **結果**: 579# まで発生していた「記録されているのに保存されない（sanitize で消える）」バグを根絶。

---

## 3. 期待される執行プロファイルの改善

直近の実測データ（Median 0.57 衝突）に対し、本刷新により以下の変化を予測する。
1. **Median の低下**: 天井衝突（Clamp）率が低下し、意図した通りの「余裕のある指値」が反映される。
2. **AS 回避の質的向上**: `RobustStats` によるノイズ除去により、天井のバタつき（Chutter）が軽減され、約定品質が安定する。

---

## 4. Copilot / Codex への共有

- **実装**: `_apply_offset_pipeline`（新）と `_apply_offset_pipeline_multiplicative`（旧）の 2 系統が `offset_pipeline.py` に共存している。
- **次フェーズ**: 数日間の A/B テスト稼働後、`analyze_fill_logs.py` の `section_execution_quality_comparison` を用いて、RMS 結合のパラメータ $\alpha, \beta$ を再キャリブレーションする。

---
