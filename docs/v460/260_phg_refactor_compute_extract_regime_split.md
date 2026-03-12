# 260# compute() extract method + regime_boosts 5-split

## 概要

`maker_price.py` の God Object 削減を目的とした構造的リファクタリング。
259# Sweep Report の P2-2 / P2-3 を実施。

- **P2-2**: `compute()` から 2 つのパイプラインステージを抽出
- **P2-3**: `_apply_regime_boosts()` を 5 つの独立 sub-method に分割
- **改番**: 253 merged, 257 impl→258, 258→259 (ユニーク番号化)

---

## P2-2: compute() extract method

### 問題

`compute()` は 214 行で、GOD OBJECT 警告ボックス (L85-98) に記載の 150 行上限を超過。
ただしパイプライン構造のため、独立ブロックの抽出は安全。

### 抽出メソッド

#### `_apply_loss_boost(side, now, effective_offset_ratio) -> float`

```
226# T1: 損失発生後の指数減衰オフセットブースト
mult(t) = 1 + (M - 1) · exp(-t / τ)
```

- AS (Avellaneda-Stoikov) 理論に基づく: 損失後のリスク回避パラメータ γ 一時増大
- `_loss_boost_mult`, `_loss_boost_set_time` 状態を参照
- 約 33 行

#### `_apply_ffd_boost(side, spread, effective_offset_ratio, offset) -> tuple[float, float]`

```
FastFillDefense per-side boost
boost_mult = _fast_fill_defense._scale_offset_ratio(side, ...)
max_ratio clamp 適用
```

- 連続約定リスク検知時の offset 拡大
- `max_offset_ratio` クランプで上界保証
- 約 22 行

### 結果

| 指標 | Before | After |
|---|---|---|
| `compute()` 行数 | 214 | 180 |
| 抽出メソッド数 | 0 | 2 |

---

## P2-3: _apply_regime_boosts() 5-split

### 問題

`_apply_regime_boosts()` は 153 行の単一メソッドに 5 つの独立 if 分岐が直列配置。
各分岐は相互独立であり、テスト・保守性向上のために分割が有効。

### 分割アーキテクチャ

```
_apply_regime_boosts()        ← 12 行のディスパッチャー
  ├── _regime_boost_trending()    ← 052# 156# 176# trending up/down × buy/sell 非対称
  ├── _regime_boost_high_vol()    ← 143# R-1a HIGH_VOL offset 拡大
  ├── _regime_boost_ranging()     ← 143# R-1a + 227# C1 OBI 非対称 ranging discount
  ├── _regime_boost_low_vol()     ← 168# low vol proportional boost
  └── _regime_boost_unknown_buy() ← 130# unknown regime buy guard
```

### 各メソッドの市場理論的根拠

| メソッド | 行数 | 理論 |
|---|---|---|
| `_regime_boost_trending` | ~30 | Momentum effect — trend 方向の注文は fill 確率が高いため offset 縮小 |
| `_regime_boost_high_vol` | ~19 | AS σ² 増大 → 最適 spread 拡大 → offset 拡大で約定リスク低減 |
| `_regime_boost_ranging` | ~30 | Mean-reversion regime → narrow spread で利幅確保。OBI で非対称化 |
| `_regime_boost_low_vol` | ~27 | 低ボラ時のスプレッド過剰圧縮防止 → 最低限の利幅保証 |
| `_regime_boost_unknown_buy` | ~20 | 不確実性下での防御的買い — 情報非対称リスク回避 |

### 結果

| 指標 | Before | After |
|---|---|---|
| `_apply_regime_boosts()` 行数 | 153 | 12 (dispatcher) |
| sub-method 数 | 0 | 5 |
| sub-method 平均行数 | — | ~25 |

---

## 改番

| 旧番号 | 新番号 | 理由 |
|---|---|---|
| 253a + 253b | 253 (merged) | rev と impl を 1 行に統合 |
| 257 impl | 258 | 257 rpt との重複回避 |
| 258 (旧) | 259 | 258→258 impl 採番のため |

---

## GOD OBJECT 警告更新

`maker_price.py` の GOD OBJECT 警告ボックスに新パイプラインステージを追記:

```
Stage 11: _apply_loss_boost  (226# T1 指数減衰)
Stage 12: _apply_ffd_boost   (FastFillDefense per-side)
```

---

## テスト

- `test_260_compute_extract_regime_split.py` — 16 テスト
  - `TestComputeExtractMethod` (8): 抽出確認 + 行数検証
  - `TestRegimeBoostsSplit` (8): 5-split 確認 + ディスパッチャー行数検証
- 既存テスト修正 2 件 (source-inspection が抽出先メソッドを参照するよう更新)
- 全 3601 テスト pass

## コミット

`d578b9441` — `260# compute() extract method + regime_boosts 5-split + renumber 253/258/259`
