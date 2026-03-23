# 580# [phg] [impl] スキーマ不整合の解消と「真の加法化」への設計刷新

> **ステータス**: 実装完遂・設計刷新 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 577#-579# (監査レポート), 568# (数理仕様)

---

## 1. 執行品質メトリクスの「真の永続化」 (Task 1)

579# で完了とされていた `spread_capture_bps` の記録が実際には機能していなかった問題を解決した。

### 1.1 バグの原因と修正内容
- **原因**: `FillRecord` dataclass (ztb/metrics/fill_quality.py) にフィールド定義が欠落していたため、シリアライズ時の sanitize 処理で削除されていた。
- **修正**: `FillRecord` に `spread_capture_bps` および `adverse_selection_cost_bps` を正式に追加。
- **結果**: 次回のサイクルより、Kissell & Glantz モデルに基づく損益分解が JSONL に刻まれ、`analyze_fill_logs.py` で可視化可能となる。

---

## 2. eDRC 数理仕様の「現実への回帰」 (Task 2)

577# で指摘された「仕様 vs 実装」の不一致を解消し、数式を一本化する。

### 2.1 eDRC 確定数式
$$C_{dynamic} = C_{base} \cdot \exp\left( \alpha \cdot \sigma_{bps} + \beta \cdot Adverse\_OFI \right)$$
- **変更点**: 568# で提案した `sigma/spread` の正規化を廃止。直感的な `sigma_bps` (bps単位) の直入れを採用する。
- **理由**: 実装の単純化と、ボラティリティに対する感度を直接的に制御するため。

---

## 3. True Additive Pipeline の実装設計 (Task 3)

「名前だけの加法化」を脱却し、`offset_pipeline.py` を以下の加重平均モデルへ刷新する。

### 3.1 結合アルゴリズム：RMS (Root Mean Square) 統合
$$Offset_{total} = Base\_Offset + \sqrt{\sum (\Delta R_i)^2}$$
- **採用理由**: 単純加算（Σ）による過剰退避を防ぎつつ、複数のリスクが重なった際に Ceiling 方向へ滑らかにオフセットを拡大させるため。

### 3.2 構成ステージの加法的定義
従来の Multiplier (1.5x 等) を、ベース 0.05 に対する増分へと再定義する。
- **EV Stage**: `+0.10`
- **Velocity Stage**: `+0.12`
- **Toxicity Stage**: `+0.20`

---

## 4. 今後のアクションプラン

1. **[Gemini]**: `offset_pipeline.py` の `_apply_offset_pipeline` メソッドを RMS 加法型へ書き換える `replace` 命令を発行する。
2. **[Copilot]**: `spread_capture_bps` が実際に JSONL に刻まれていることを、 c164 以降の SHA で最終確認する。

---
