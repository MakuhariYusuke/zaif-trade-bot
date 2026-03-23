# 575# [cplt] [impl] 574# eDRC パラメータ推定・RobustStats統合・Hard Cap

> **ステータス**: 実装完了  
> **更新日**: 2026-03-23  
> **参照**: 574# (Gemini レポート), 572# (eDRC A/B toggle), 573# (RobustStats+テレメトリ)

---

## 概要

574# Gemini レポートの推奨事項をコードに統合:

1. **eDRC パラメータ設定** — α=0.020, β=0.40 をYAML設定 (enabled: false のまま)
2. **σ → bps 変換** — Parkinson σ (ratio ~0.0001) を bps (×10,000) に変換して eDRC に渡す
3. **edrc_hard_cap** — exp爆発防止のハードキャップ (default=1.0)
4. **get_robust_inputs()** — RobustStats の asymmetric_ema (σ) + median_filter (OFI) を統合
5. **条件分岐** — additive_enabled 時のみ robust inputs を使用、無効時は生値

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `fill_config.py` | `edrc_hard_cap` フィールド追加、`min()` クランプ適用 |
| `fill_config_parser.py` | `edrc_hard_cap` パース追加 |
| `offset_pipeline.py` | σ→bps変換 (×10,000)、robust/raw 条件分岐 |
| `maker_price.py` | `get_robust_inputs(side)` メソッド追加 (asymmetric EMA σ + median OFI) |
| `fill_test.yaml` | α=0.020, β=0.40, hard_cap=1.0 設定 |
| `test_467_*.py` | hard_cap・シミュレーションテーブル検証テスト 3件追加 |
| `test_571_*.py` | TestGetRobustInputs クラス 4件追加 |
| `test_336_*.py` | KNOWN_YAML_OVERRIDES に edrc_alpha/beta 追加 |

## eDRC 動作確認 (574# シミュレーション表)

α=0.020, β=0.40, C_base=0.40:

| σ(bps) | OFI=0.2 | OFI=0.6 | OFI=1.0 |
|--------|---------|---------|---------|
| 5.0    | 0.47    | 0.52    | 0.61    |
| 15.0   | 0.58    | 0.64    | 0.75    |
| 30.0   | 0.78    | 0.86    | 1.00*   |

*hard_cap=1.0 でクランプ

## RobustStats パラメータ

- σ: `asymmetric_ema(alpha_up=0.20, alpha_down=0.05)` — リスク増大に4倍敏感
- OFI: `median_filter_fast(window=10)` — フラッシュ注文耐性

## テスト結果

- v460 テスト: 59/59 パス
- 全テスト: 1318 パス (既存の無関係な1件のみ失敗)
