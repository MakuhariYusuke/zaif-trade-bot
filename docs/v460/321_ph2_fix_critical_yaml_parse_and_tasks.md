# 321# CRITICAL 修正 + 319# 残タスク処理 + God Object 分割検討

> **種別**: fix  
> **起票**: 2026-03-07  
> **起源**: 320# セルフレビュー + 319# §6 残課題  
> **前提**: 320# (`044e687a9`), 319# (`21bcd6885`)

---

## §1 CRITICAL: 320# sell ceiling YAML 未パース

### §1.1 問題

320# で `FillTestConfig` に `offset_ceiling_ratio_buy` / `offset_ceiling_ratio_sell` フィールドを追加し、`maker_price.py` の ceiling ロジックを修正したが、**`from_yaml()` にパース処理を追加し忘れた**。

```python
# from_yaml() の offset ceiling セクション (320# 時点)
if "offset_ceiling_ratio" in yaml_cfg:
    kwargs["offset_ceiling_ratio"] = float(yaml_cfg["offset_ceiling_ratio"])
# ← offset_ceiling_ratio_buy/sell のパースが欠落！
```

### §1.2 影響

- YAML に `offset_ceiling_ratio_sell: 0.50` を設定しても、`FillTestConfig` のデフォルト `None` のまま
- `maker_price.py` は `None` の場合共通値 `offset_ceiling_ratio: 0.15` にフォールバック
- **sell ceiling = 0.15 のまま → 320# の C-1 修正が完全に無効**

さらに深刻：320# で `trending_sell_offset_boost_factor` を 4.0→1.5 に変更。

| 状態 | 実効 sell offset (trending_up) |
|---|---|
| 319# | `0.15 × 4.0 = 0.60` |
| **320# (壊れた状態)** | **`0.15 × 1.5 = 0.225`** |
| 320# 意図 | `0.50 × 1.5 = 0.75` |

**320# は売り防御力を 319# より 62.5% 悪化させていた。**

### §1.3 修正

`from_yaml()` に 4 行追加:

```python
if "offset_ceiling_ratio_buy" in yaml_cfg:
    kwargs["offset_ceiling_ratio_buy"] = float(yaml_cfg["offset_ceiling_ratio_buy"])
if "offset_ceiling_ratio_sell" in yaml_cfg:
    kwargs["offset_ceiling_ratio_sell"] = float(yaml_cfg["offset_ceiling_ratio_sell"])
```

`config_hot_reload.py` の `_HOT_RELOADABLE_FIELDS` にも追加:

```python
"offset_ceiling_ratio_buy",
"offset_ceiling_ratio_sell",
```

### §1.4 テスト追加

- `test_side_specific_ceiling_yaml_parse` — from_yaml() が buy/sell を正しくパース
- `test_side_specific_ceiling_yaml_absent_stays_none` — YAML 未設定時は None 維持
- `test_side_specific_ceiling_hot_reloadable` — hot-reload 対象に登録済み

---

## §2 319# 残タスク処理

### §2.1 M-3: unknown_regime_max_consecutive 10→5

**問題**: 10 サイクル × 120s = 20 分間の取引停止は過剰。さらに buy/sell 混合カウント (M-6) のため、交互評価で実質 5 サイクル/side で bypass 発動。

**修正**: `fill_test.yaml` で `10 → 5` に変更。

```yaml
# 321# M-3: 10→5 (10=20分停止は過剰、buy/sell混合カウントで実質2.5サイクル/side)
unknown_regime_max_consecutive: 5
```

実効的に ~5 分で bypass 発動、unknown regime 離脱の高速化。

### §2.2 M-5: fixed_offset_bps 2.0→1.0

**問題**: Passive MM モードの固定 offset 2.0 bps はスプレッド超過で spread guard が常時発動。

```
BTC_JPY mid ≈ 14,000,000 JPY
2.0 bps = 2,800 JPY → typical spread (1,000-2,000 JPY) 超過
→ spread guard → best ask/bid 配置 (effective_offset_ratio = 0.0)
```

**修正**: `1.0 bps = 1,400 JPY` に変更。spread 2,000+ JPY 時にスプレッド内配置が可能に。

### §2.3 H-1: target_skip_rate_sell 値・コメント矛盾

**問題**: YAML 値 `0.250` だがコメントに `0.25→0.20` と記載。retrain セクション (L830) では `0.20`。

**修正**: 値を `0.20` に統一。sell 側の逆選択防御を意図通りに強化。

---

## §3 追加バグ修正

### §3.1 M-5b: macro downgrade ログの上書き前値消失

**ファイル**: `fill_cycle_executor.py`

```python
# Before (バグ):
if _action == "downgrade":
    regime_str = "ranging"  # 上書き
    logger.info("... micro=%s ...", regime_str)  # ← "ranging" が出力される

# After (修正):
if _action == "downgrade":
    _original_regime = regime_str  # 上書き前を保存
    regime_str = "ranging"
    logger.info("... micro=%s ...", _original_regime)  # ← 元の regime が出力
```

### §3.2 L-1: velocity_math.py docstring 重複

「アーキテクチャ:」セクションが完全に2回重複していたのを1回に修正。

---

## §4 God Object 分割検討

### §4.1 現状

| ファイル | 行数 | 自己宣言上限 | 超過率 |
|---|---|---|---|
| `maker_price.py` | 1,692 | 850 | **199%** |
| `fill_cycle_executor.py` | 1,502 | — | — |

| メソッド | 行数 | テスト上限 | テスト名 |
|---|---|---|---|
| `compute()` | 294 | 295 | `test_compute_line_count_reduced` |
| `run_single_cycle()` | 739 | 740 | `test_run_single_cycle_under_400_lines` |

`run_single_cycle()` のテスト名は `under_400_lines` だが実際の上限は **740 行** — 当初目標の 1.85 倍。

### §4.2 maker_price.py 分割計画

| 新モジュール | 抽出対象 | 推定行数 | 削減行数 |
|---|---|---|---|
| `maker_microstructure.py` | `_estimate_sigma`, `_dynamic_tau`, `_apply_as_reservation_shift`, `_apply_kyle_lambda`, `_apply_amihud_illiq`, `_get_depth` | ~260 | -260 |
| `maker_regime_boost.py` | `_apply_regime_boosts`, 5 × `_regime_boost_*`, `_resolve_trending_boost` | ~200 | -200 |
| `maker_risk_guards.py` | `_apply_volatility_guard`, `_apply_imbalance_risk`, `_apply_buy_as_guard`, `_apply_sell_hour_boost` | ~215 | -215 |
| `maker_inventory.py` | `update_inventory`, `_decayed_imbalance`, `inv_net_imbalance`, 関連 slots | ~90 | -90 |
| **合計** | | | **~765** |

**結果**: 1,692 → ~927 行 (850 上限に近い)

### §4.3 fill_cycle_executor.py 分割計画

| 新モジュール | 抽出対象 | 推定行数 | 削減行数 |
|---|---|---|---|
| `order_placer.py` | 発注 for-loop, postonly guard, retry, error classification | ~200 | -200 |
| `pre_order_adjustments.py` | EV/velocity/trending/toxicity offset, VG supplement | ~180 | -180 |
| `fill_record_builder.py` | `_build_fill_record`, `_build_fill_*_fields` | ~285 | -285 |
| **合計** | | | **~665** |

**結果**: `run_single_cycle()` 本体は ~100-150 行の dispatcher に縮退可能。

### §4.4 実行方針

- **優先度**: P1 (次回セッション以降)
- **理由**: リファクタリングは大規模で、テスト影響が広範。CRITICAL 修正と機能改善の方が短期的には high value。
- **リスク軽減**: 段階的に 1 モジュールずつ抽出、各ステップでテスト全通過を確認。

---

## §5 その他の発見事項 (未修正)

| ID | 深刻度 | 内容 | 状態 |
|---|---|---|---|
| M-2 | MEDIUM | `_consecutive_unknown_blocks` が buy/sell 混合カウント (M-6 と同一) | ⏳ per-side 分離は影響範囲広大 |
| M-3b | MEDIUM | `_apply_offset_multiplier` で `offset_mult < 1.0` + `aggressive=False` 時に無視 | ⏳ 意図的設計の可能性、要確認 |
| M-4 | MEDIUM | Feature Graveyard: 7つの `enabled: false` 機能が YAML に残存 | ⏳ 実行時影響なし |
| M-1 | MEDIUM | `velocity_ema_alpha: 1.0` で EMA 無効のまま (227# C3) | ⏳ 有効化 (`0.6`) は性能変化、データ蓄積後 |
| L-3 | LOW | `regime_policy.dynamic_cycle` と `dynamic_cycle_interval` の機能重複 | ⏳ 優先度低 |

---

## §6 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_config.py` | CRITICAL: from_yaml() にサイド別 ceiling パース追加 |
| `scripts/v460/lib/config_hot_reload.py` | サイド別 ceiling を hot-reload 対象に追加 |
| `configs/v460/fill_test.yaml` | M-3 (consecutive 10→5), M-5 (offset_bps 2→1), H-1 (skip_rate 0.25→0.20) |
| `scripts/v460/lib/fill_cycle_executor.py` | M-5b macro downgrade ログ修正 |
| `scripts/v460/lib/velocity_math.py` | L-1 docstring 重複除去 |
| `tests/unit/v460/test_303_review_implementations.py` | YAML パース + hot-reload テスト 3 件追加 |
| `tests/unit/v460/test_277_magic_number_grounding.py` | consecutive YAML 値 10→5 |
| `tests/unit/v460/test_fill_quality.py` | target_skip_rate_sell 0.25→0.20 |

---

## §7 関連ドキュメント

| # | 関係 |
|---|---|
| 320 | C-1 サイド別 ceiling (YAML パース欠落がここで発見) |
| 319 | C-1 発見 + M-3/M-5/M-6 課題特定 |
| 277 | unknown_regime_max_consecutive config 化 |
| 303 | Passive MM fixed_offset_bps 導入 |
| 316 | §6 先行施策提案 |
