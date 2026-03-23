# 572# eDRC A/Bトグル インフラ実装

**タイプ**: impl
**日付**: 2026-03-23
**前提**: 568# (eDRC数理仕様), 570# (Robust Inputs & Additive Parameters)

---

## 概要

568# で定義された eDRC (Exponential Dynamic Risk Ceiling) と加法スコアパイプラインへの
段階的移行を安全に進めるため、**経路2（A/Bトグル方式）** のインフラを実装した。

## 設計方針

- `experimental_additive_pipeline = false` (デフォルト) → 既存の乗法チェーンが動作
- `experimental_additive_pipeline = true` → eDRC ベースの動的 ceiling に切り替え
- 後方互換性を完全に維持（新引数は全てデフォルト値あり）

## 変更ファイル

### 1. `scripts/v460/lib/fill_config.py`
- `FillTestConfig` に5フィールド追加:
  - `experimental_additive_pipeline: bool = False`
  - `edrc_alpha: float = 0.0` (1分足ボラティリティ感度)
  - `edrc_beta: float = 0.0` (OFI逆選択圧力感度)
  - `edrc_c_base: float = 0.40` (ベース ceiling)
  - `additive_base_bps: float = 0.0` (将来用加法ベース)
- `resolve_offset_ceiling()` に `sigma`, `adverse_ofi` キーワード引数を追加
- `experimental_additive_pipeline = True` 時: $C_{dynamic} = C_{base} \cdot e^{\alpha \sigma + \beta \cdot OFI_{adverse}}$

### 2. `scripts/v460/lib/maker_price.py`
- `get_adverse_ofi(side: str) -> float` メソッド追加
- OFI履歴の平均から、指定サイドにとっての逆選択圧力を正の値として返す

### 3. `scripts/v460/lib/fill_config_parser.py`
- `_parse_trading_features()` に YAML → Config マッピング追加
- `experimental_additive_pipeline:` セクション（`enabled` + 各パラメータ）を解析

### 4. `scripts/v460/lib/offset_pipeline.py`
- Final Clamp 内の `resolve_offset_ceiling()` 呼び出しに `sigma`, `adverse_ofi` を伝播
- `self._maker_price.last_sigma` と `self._maker_price.get_adverse_ofi(side)` を利用

### 5. `configs/v460/fill_test.yaml`
- `experimental_additive_pipeline:` セクション追加 (enabled: false)
- 全パラメータにコメント付き

### 6. `tests/unit/v460/test_467_remaining_issues.py`
- eDRC テスト4件追加:
  - `test_resolve_offset_ceiling_edrc_disabled`: トグル off → 従来動作
  - `test_resolve_offset_ceiling_edrc_enabled_zero_inputs`: sigma=0, ofi=0 → c_base
  - `test_resolve_offset_ceiling_edrc_with_sigma`: sigma > 0 → ceiling 拡大
  - `test_resolve_offset_ceiling_edrc_with_hour`: eDRC + hour_ceiling_mult 併用

## Live Presence 4項目 確認結果

| # | 項目 | 状態 | 詳細 |
|---|------|------|------|
| 1 | `cache/sidecar_signal.json` 更新 | ✅ | 最終更新: 2026-03-22T10:36:26 UTC, directional_bias=0.0 (neutral) |
| 2 | `logs/sac_retrain_history.jsonl` 履歴 | ✅ | 最終エントリ: 2026-03-23T01:11:22 UTC, status=deployed |
| 3 | fill_records に sidecar 値 | ⚠️ | キーは存在するが全て None (signal neutral のため) |
| 4 | fill_test.log に sidecar ログ | ❌ | fill_test.log 未生成 (VPS上の別ログパス？) |

**考察**: sidecar_signal.json は neutral モデルが deployed されており、信号自体は流れている。
ただし directional_bias=0.0 のため fill_records の sidecar_offset_bps は全て None になっている。
これは設計通りの挙動（bias=0 なら offset 加算なし）。

## 次のステップ

- Gemini 571# で `ztb/utils/robust_stats.py` の実装ドラフトが完成次第、eDRC の入力値（σ, OFI）をロバスト化
- `experimental_additive_pipeline: enabled: true` での A/B テスト開始は、Gemini の分析 (spread_capture_bps 評価) 完了後
