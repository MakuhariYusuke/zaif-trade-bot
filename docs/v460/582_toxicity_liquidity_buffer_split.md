# 582# [Task 2.2] Toxicity & Liquidity Buffer Split 完了報告

> **作成日**: 2026-03-23
> **ステータス**: 実装完了（A/B テストフラグ `experimental_additive_pipeline` で制御）

---

## 1. 実施内容：Toxicity vs Liquidity の完全分離 (Task 2.2)

「God Object」として一元化されていた `effective_offset_ratio` を、ビジネス要因ごとに分離・集計できるようにリファクタリング。
`scripts/v460/lib/offset_pipeline.py` において、RMS (二乗平均平方根) を用いた Additive バッファ管理モデルを `_apply_offset_pipeline_additive` として新規追加した。

### アーキテクチャ: A/B Dispatcher パターン
- `_apply_offset_pipeline()` — A/B dispatcher。`config.experimental_additive_pipeline` フラグで分岐
  - `True` → `_apply_offset_pipeline_additive()` (582# 新方式)
  - `False` (default) → `_apply_offset_pipeline_multiplicative()` (460# 従来方式)
- 従来の乗算チェーンは `_apply_offset_pipeline_multiplicative` にリネームし完全保持

### 1.1 バッファの分類
- **Liquidity Factors (流動性リスク)**
  - `ev`: スプレッドと枯渇リスクを含む Base factor
  - `macro`: 流動性プレミアムのシフト
- **Toxicity Factors (毒性リスク)**
  - `velocity`: 市場急変
  - `trending`: 強い片道トレンド
  - `toxicity`: Amihud / Kyle 由来の非対称オフセット
  - `vg_supp`: Volatility Guard 補完
  - `alert`: 緊急退避倍率

### 1.2 計算モデル (真の Additive Pipeline 化)
従来の乗算チェーン（`A * B * C`）から、以下のアルゴリズムに移行した。
1. 各項について、`multiplier` が 1.0 を超える場合はベースとなる `effective_offset_ratio` に掛けて差分（$\Delta R_i$）を算出。
2. Toxicity カテゴリの差分をまとめた配列 `tox_deltas` と、Liquidity カテゴリの配列 `liq_deltas` に仕分け。
3. `tox_rms = sqrt(sum(d^2 for d in tox_deltas))`
4. `liq_rms = sqrt(sum(d^2 for d in liq_deltas))`
5. `final_offset = base_ratio + tox_rms + liq_rms`
上記により、要因ごとに独立したバッファを構築しつつ、極端な相乗爆発を回避する仕組みを確立した。

### 1.3 可観測性の向上
- `executor_offset_stages_json` に `tox_buffer` と `liq_buffer` を追加出力。
- 各ステージの multiplier 値 (ev, velocity, trending, toxicity, vg_supp, macro, alert) も記録。
- これにより、後続のデータ分析プロセスで「どちらの要因がスプレッドを広げたか」が直ちに判別できる。

## 2. 状態・今後のタスク
- 本修正は `scripts/v460/lib/offset_pipeline.py` に反映済み。
- デフォルトは従来の乗算パイプライン（`experimental_additive_pipeline: false`）。
- A/B テスト期間中に additive 方式のパフォーマンスを検証し、優位性が確認されれば `true` に切り替える。
- テストファイル (`test_196`) の `inspect.getsource` 参照先を `_apply_offset_pipeline_multiplicative` に更新済み。
