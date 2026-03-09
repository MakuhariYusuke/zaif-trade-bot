# 353# VPIN 非対称 buy boost 実装

> **種別**: impl  
> **フェーズ**: ph2 (G1.1-exec)  
> **前提**: 352# §6 次ステップ, 351# 盲点1 (Ranging ≠ 対称), 164# SHAP 分析  
> **日付**: 2026-03-09

---

## §1 プロジェクト立ち位置 (000# 参照)

本セッションは **ph2 G1.1-exec** フェーズの実装改善。  
000# §3.3 Gate 判定基準:

| 指標 | Quick (72h) | Full (168h) |
|---|---|---|
| attempted_fill_rate | ≥ 60% | ≥ 70% |
| PnL30 | 有意に負でない | 有意に負でない |
| AS_ratio | ≤ 30% | ≤ 30% |
| 累積損失上限 | — | < 10,000 JPY |

今回の改善は **buy 側 AS 率低減 → AS_ratio ≤ 30% 維持** と **PnL 改善** に直結。

---

## §2 根拠と過去資産の活用

### 351# 盲点 1: Ranging ≠ 対称

> ranging レジームで「方向なし = 対称」と仮定しているが、buy 側の構造的逆選択リスクは
> sell 側より高い。VPIN boost は buy/sell 均一だが、buy にはより強い防御が必要。

### 164# SHAP 分析 — buy 側 VPIN 依存性の非対称

| 特徴量 | buy SHAP | sell SHAP | 比率 |
|---|---|---|---|
| `vpin_60s` | 0.548 | 0.316 | **1.73x** |

buy のスキップ判定は VPIN に 1.73 倍依存 → **buy 側は VPIN シグナルに対してより敏感に反応すべき**。

### 289# C-1 提案: VPIN < 0.3 で buy offset conservative

289# の microstructure 改善ロードマップ Phase C-1 の方向性と整合。
ただし C-1 は binary threshold ベースだったのに対し、本実装は 257# continuous scaling
のインクリメンタル boost に extra multiplier を適用する形で、既存インフラを最大限活用。

### 352# §6.1 具体提案との対応

| 352# 提案 | 本セッション |
|---|---|
| §6.1-A: VPIN boost buy 計算追加 | ✅ **実装完了** |
| §6.1-B: buy_velocity_skip 閾値 | ⏳ 次回以降 (YAML only) |
| §6.2-C: EWMA 時間減衰 | ⏳ 次回以降 |
| §6.3-D: A-S inventory skewing | ⏳ 次回以降 (大改修) |

---

## §3 実装詳細

### 設計思想

- 既存の VPIN continuous scaling (257#) のアウトプット boost 値に対して、
  buy 側のみ追加 multiplier を適用
- boost 増分 `(vpin_boost - 1.0)` に `vg_vpin_buy_extra_mult` を乗算
- sell 側は完全に無影響 → 後方互換性 100%
- default = 1.0 で無効化 (既存動作と完全一致)

### 数式

```
# 標準 VPIN continuous (257#)
norm = clamp((vpin - min) / (thresh - min), 0, 1)
vpin_boost = 1 + (factor - 1) * norm²

# 353# buy 非対称
if side == "buy" and vpin_boost > 1.0 and buy_extra_mult > 1.0:
    vpin_boost = 1 + (vpin_boost - 1) * buy_extra_mult
```

### 数値例

| VPIN | norm | base boost | buy_extra=1.5 時の buy boost | 差分 |
|---|---|---|---|---|
| 0.40 | 0.00 | 1.000 | 1.000 | 0.000 |
| 0.45 | 0.17 | 1.028 | 1.042 | +0.014 |
| 0.50 | 0.33 | 1.111 | 1.167 | +0.056 |
| 0.55 | 0.50 | 1.250 | 1.375 | +0.125 |
| 0.60 | 0.67 | 1.444 | 1.667 | +0.222 |
| 0.65 | 0.83 | 1.694 | 2.042 | +0.347 |
| 0.70 | 1.00 | 2.000 | 2.500 | +0.500 |

> ※ min=0.40, thresh=0.70, factor=2.0 前提

### 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_config.py` | `vg_vpin_buy_extra_mult: float = 1.0` フィールド追加 |
| `scripts/v460/lib/maker_risk_guards.py` | buy 側 VPIN boost 追加増幅ロジック (8行) |
| `scripts/v460/lib/fill_config_parser.py` | YAML → config マッピング追加 |
| `scripts/v460/lib/config_hot_reload.py` | ホットリロード whitelist 追加 |
| `configs/v460/fill_test.yaml` | `vpin_buy_extra_mult: 1.5` 設定 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | drift prevention 対象に追加 |
| `tests/unit/v460/test_258_as_reservation_vpin_continuous_protocol.py` | テスト 4 件追加 |

---

## §4 テスト結果

### test_258 (VPIN buy boost + 既存テスト)

```
29 passed in 1.45s
```

新規テスト 4 件:

| テスト | 検証内容 |
|---|---|
| `test_default_mult_1_no_change` | mult=1.0 で buy == sell (後方互換) |
| `test_buy_extra_mult_increases_buy_boost` | mult=1.5 で buy > sell |
| `test_buy_extra_mult_math_correctness` | 数式検証 (VPIN=0.55 → buy boost 1.375) |
| `test_sell_unaffected_by_buy_extra_mult` | sell は mult 変更に無関係 |

### test_336 (YAML-コード drift prevention)

```
4 passed in 0.87s
```

### 型エラー

変更対象ファイル: **エラーなし**  
(config_hot_reload.py L638/L651 の既存型エラーは今回の変更と無関係)

---

## §5 運用設定

```yaml
# configs/v460/fill_test.yaml
volatility_guard:
  vpin_buy_extra_mult: 1.5  # buy 側 VPIN boost を 1.5 倍増幅
```

**ホットリロード対応**: YAML 書き換えで即時反映可能。  
**推奨チューニング範囲**: 1.0 (無効) 〜 2.0 (aggressive)

---

## §6 今後の改善提案

### 即座に着手可能 (YAML 調整のみ)

| 項目 | 現在値 | 提案値 | 期待効果 |
|---|---|---|---|
| `buy_velocity_skip_threshold_bps` | -6.0 | -4.0 | buy 時の逆走許容を絞る |
| `fast_fill_defense.buy_seconds` | 10 | 8 | buy fast fill 検出を敏感に |

### 中期 (コード変更必要)

| 項目 | 概要 | 優先度 |
|---|---|---|
| EWMA 時間減衰 | kill 中の stale EWMA 問題 (351# §4.3) | P1 |
| A-S inventory skewing | maker_price.py 大改修、gamma パラメータ | P2 |
| VPIN×hour interaction | 夜間帯 VPIN 感度上昇 | P2 |
| spread-conditional buy mult | spread < 3bps 時のみ extra mult 適用 | P3 |

---

## §7 Bot 稼働状況メモ

- PID 89776, 起動 09:51 JST 2026-03-09
- 直近観測: Cycle 8818, buy unfilled (stale adverse drift 4.2bps cancel)
- 本変更反映は次回 Bot 再起動時 (ホットリロードで即時も可)
