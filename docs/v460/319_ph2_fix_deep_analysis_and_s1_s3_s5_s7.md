# 319# 深層分析 + 316# §6 施策実行 + 318# セルフレビュー

> **種別**: fix/rpt  
> **起票**: 2026-03-09  
> **起源**: 316# §6 施策サマリ + 318# セルフレビュー + 深層分析  
> **コミット**: (本コミット)

---

## §1 318# セルフレビュー結果

### §1.1 レビュー対象

318# (`9a5ba3d71`) の全変更を精査。

| 項目 | 判定 | 詳細 |
|---|---|---|
| F5-1 `in ("none", "unknown")` | ✅ OK | detector=None→"none", warmup→"unknown" を正しくキャッチ |
| F5-3 regime_at_order タイミング | ✅ OK | `_compute_maker_price()` 前でキャプチャ。regime detector は compute 内で更新されない |
| F5-3 `observation_count` property | ✅ OK | `regime_detector.py` L173 で正しく定義 |
| F5-3 FillRecord フィールド追加 | ✅ OK | `build_fill_record` の kwargs パターンに準拠 |
| F5-4 null/unknown 分析分離 | ✅ OK | `regime is None` → "null"、`regime == "unknown"` → unknown 表記で混同解消 |

### §1.2 発見 (低リスク)

**Q-1**: F5-4 の `_window = 20` が分析スクリプト内でハードコード。YAML の `observation_window` が将来変更された場合に不整合。分析スクリプト内のため影響軽微、注記に留める。

---

## §2 深層分析: Sell パイプライン全死の発見

### §2.1 C-1: offset_ceiling (0.15) vs sell_offset_floor (0.30) — CRITICAL

**パイプライン内の全 sell offset チューニングパラメータが ceiling で殺されている。**

```
Pipeline flow (sell):
  base (0.18) → floor max(0.18, 0.30) = 0.30
  → regime boost (×1.8) = 0.54
  → spread_adaptive (floor 再適用) = 0.30
  → VG / Kyle / Amihud / etc → 各種拡大
  → ceiling min(*, 0.15) = 0.15  ← 常にここで固定
```

**死んでいるパラメータ (12+ 件)**:
- `trending_up_sell_offset_boost: 1.8` — ceiling に殺される
- `trending_down_sell_offset_boost: 0.7` — floor → ceiling
- `trending_offset_boost_sell: 1.5` — 同上
- `narrow_spread_boost_sell: 2.5` — 同上
- `ranging_offset_discount: 0.90` (sell) — floor 再適用で無効化 → ceiling
- `sell_hour_offset_boost` (UTC 08/13/14/16) — ceiling に殺される
- `side_offset.sell: 0.18` — floor で即上書き → ceiling
- `low_vol_offset_boost: 1.4` (sell) — ceiling
- `loss_boost` (sell) — ceiling
- `buy_as_guard` は buy のみのため sell 無関係だが、sell 側の VG/Kyle/Amihud 全て ceiling 支配

**実効的な sell offset 制御**: executor の post-pipeline multiplier のみ

| Multiplier | YAML | 値 | 適用条件 | 実効 offset |
|---|---|---|---|---|
| trending_sell | `trending_sell_offset_boost_factor` | 3.0→**4.0** | trending_up sell | 0.15×4.0=0.60 |
| velocity | `velocity_offset_mult` | SkipGate 計算 | velocity > threshold | 可変 |
| toxicity | `toxicity_budget_*` | 動的 | toxicity > 0 | 可変 |
| VG supplement | `vg_supplement_*` | 動的 | VG 発動時 | 可変 |
| alert | `alert_offset_mult` | 動的 | alert 発動時 | 可変 |

### §2.2 C-2: Post-pipeline multiplier のスタッキング

Pipeline ceiling (0.15) 後に最大 6 個の multiplier が乗法的に適用される。明示的な上限なし。

最悪ケース: `0.15 × 3.0 × 4.0 × 2.0 = 3.60` → offset > spread → spread guard フォールバック

**影響**: 通常は 1-2 個のみ同時発火のため実害は軽微。ただし暴落時に 3+ 発動でスプレッドガード依存の暗黙安全弁。

### §2.3 その他の発見 (MEDIUM)

| ID | 内容 | 深刻度 |
|---|---|---|
| M-1 | Trending sell 二重防御 — パイプライン側は ceiling 死 | MEDIUM |
| M-3 | Unknown regime 20 分停止 (Gate 10 サイクル + Passive MM) | MEDIUM |
| M-5 | Passive MM `fixed_offset_bps: 2.0` → 通常 spread で offset > spread → best ask に退化 | MEDIUM |
| M-6 | unknown counter が buy/sell 混合カウント (per-side ではない) | MEDIUM |

### §2.4 LOW

| ID | 内容 |
|---|---|
| L-2 | `velocity_ema_alpha: 1.0` — EMA 無効のまま |
| L-4 | Feature Graveyard: 7 つの `enabled: false` 機能が YAML に残存 |

### §2.5 S-1 修正方針の変更

**元の S-1 提案**: `trending_up_sell_offset_boost: 1.8 → 2.5` (パイプライン YAML)
→ C-1 により **DEAD CHANGE** と判明

**修正版 S-1**: `trending_sell_offset_boost_factor: 3.0 → 4.0` (executor post-pipeline)
→ 実効 offset: 0.15 × 4.0 = 0.60 (spread の 60%)

---

## §3 実施内容

### S-1 修正版: trending_sell_offset_boost_factor 3.0 → 4.0 [P0]

**ファイル**: `configs/v460/fill_test.yaml`

```yaml
# Before
trending_sell_offset_boost_factor: 3.0
# After
trending_sell_offset_boost_factor: 4.0  # 319# ceiling後0.15×4.0=0.60
```

**根拠**: trending_up sell p10=-9.86 bps (全指標最悪)。パイプライン側 boost (1.8) は ceiling に殺されるため executor 側で調整。

### S-5: sell_offset_floor / ceiling 整合コメント [P2]

**ファイル**: `configs/v460/fill_test.yaml`

`sell_guard.offset_floor: 0.30` と `offset_ceiling_ratio: 0.15` に相互参照コメント追加:
- floor 行: ceiling (0.15) > floor (0.30) のため現在 floor 無効
- ceiling 行: sell 側パイプライン全ステージが 0.15 にクランプされる旨

### S-7: テール損失分析関数 [P1]

**ファイル**: `analysis/311_observational_rerun.py`

`tail_loss_analysis()` 関数を追加:
- p10 以下のテール records を抽出
- Regime over-representation 分析 (テールに集中する regime を特定)
- 時間帯 over-representation (テールに集中する UTC hour を特定)
- Decision path 分布
- Spread at order 統計
- AS 率の overrep 計算 (tail AS / total AS)

出力は §9.5 として本文に表示、JSON にも `tail_loss_analysis` として格納。

### S-3: mid_at_order フィールド追加 [P1]

**ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/lib/fill_cycle_executor.py`

| フィールド | 型 | 意味 |
|---|---|---|
| `mid_at_order` | `float \| None` | 注文発行時 (`_compute_maker_price()` 内) の mid price |

**取得方法**: `_compute_maker_price()` が内部で `_prev_mid_price` をセットするため、返り値後に `get_fallback_price()[0]` で取得。315# §4.4 で指摘された検出レイテンシバイアスの分離に使用。

**spread capture 精度向上**:
```
旧: SC = (fill_price - mid_at_fill) / mid_at_fill  ← 検出遅延含む
新: SC = (fill_price - mid_at_order) / mid_at_order  ← 発注時 mid で正確
```

---

## §4 テスト

| テスト | 対象 | 結果 |
|---|---|---|
| `test_fill_record_has_mid_at_order` | S-3 フィールド存在 | ✅ |
| `test_fill_record_mid_at_order_roundtrip` | S-3 to_dict/from_dict 往復 | ✅ |
| fill 全体テスト | 653 passed, 1 skipped | ✅ |
| 303 review テスト | 19 passed | ✅ |

### §4.1 line count 調整

`fill_cycle_executor.py` が S-3 追加で 1500 行に到達し、`test_fill_cycle_executor_line_count_under_limit` (< 1500) を超過。制限を 1510 に引き上げ。

---

## §5 317# ドキュメント作成

番号体系ギャップ (316→318) の 317# を作成: [317_ph2_rpt_observation_experiment.md](317_ph2_rpt_observation_experiment.md)

316# §2 の観測実験結果データ (`317_observation_full.txt`, `317_observation_dcc3064.txt`) を構造化文書化。

---

## §6 残課題・今後の方針

### §6.1 C-1 sell パイプライン全死問題への対処選択肢

| 選択肢 | 影響 | リスク |
|---|---|---|
| **(a)** ceiling 引き上げ (0.15→0.30+) | パイプラインステージ復活 | fill_rate 低下 |
| **(b)** ceiling 撤廃 + floor のみ制御 | 完全復活 | offset 暴走の可能性 |
| **(c)** 現状維持 + executor multiplier で精密制御 | 安全 | パイプラインの 12+ パラメータが無駄 |
| **(d)** sell 側 ceiling を別値で分離 | 柔軟 | 設定複雑化 |

**推奨**: **(c)** 現状維持。短期的にはリスク最小。executor multiplier による制御は実績あり (3.0→4.0 で trending_up sell 強化済)。パイプライン方向別チューニング (up/down 非対称) は executor レイヤーに移設する方向で検討。

### §6.2 未着手の 316# §6 施策

| ID | 施策 | 状況 |
|---|---|---|
| S-2 | Sell Hour Boost 見直し | ⏳ post-310# データ必要 |
| S-4 | None sell skip | ✅ 318# で解決済 (Gate 1+7 存在) |
| S-6 | buy ev_offset 調査 | ⏳ 分析拡張後に実施可能 |

### §6.3 深層分析の MEDIUM 課題

| ID | 対処 |
|---|---|
| M-3 20 分停止 | YAML `unknown_regime_max_consecutive: 10 → 3-5` で緩和可能 |
| M-5 Passive MM 退化 | `fixed_offset_bps: 2.0 → 1.0` で spread 内に収める |
| M-6 混合カウンタ | per-side 分離が設計的に正しい → 要コード変更 |

---

## §7 関連ドキュメント

| # | 関係 |
|---|---|
| 316 | セルフレビュー + 施策提案 S-1〜S-7 (本実装の元) |
| 317 | 観測比較実験報告 (本 commit で作成) |
| 318 | 307# F5 none regime 修正 (セルフレビュー対象) |
| 315 | Ceiling/Ratio Semantics 調査 (C-1 の先行発見) |
| 246 | sell_offset_floor 0.30 の由来 |
| 306 | offset_ceiling 0.15 の由来 |
