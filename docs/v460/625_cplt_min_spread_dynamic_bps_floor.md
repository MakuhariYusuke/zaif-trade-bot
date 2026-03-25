# 625# min_spread 動的 BPS フロア + 624# バグ修正

## 概要

固定値 `min_spread_jpy: 500` を Stoll (1978) order processing cost に基づく **BPS 動的フロア** に置換。
同時に 624# で混入した 3 件のバグを修正。

## 理論的背景

### Glosten-Milgrom (1985) スプレッド分解

マーケットメイカーのスプレッドは 3 成分に分解できる:

$$S = S_{\text{processing}} + S_{\text{inventory}} + S_{\text{adverse\_selection}}$$

| 成分 | 理論 | 実装 |
|------|------|------|
| **Order Processing Cost** | Stoll (1978) / Amihud-Mendelson (1986) | `min_spread_floor_bps` |
| **Inventory Holding Cost** | Ho-Stoll (1981) | (offset pipeline が吸収) |
| **Adverse Selection Cost** | Glosten-Milgrom (1985) / Roll (1984) | `min_spread_atr_*` (σ連動) |

### Stoll (1978) Order Processing Cost

order processing cost は資産価格に比例する:

$$c = P_{\text{mid}} \times \frac{b}{10000}$$

ここで $b$ は BPS 単位のコスト率。手数料ゼロ (Coincheck maker) でも latency risk・tick cushion・最低利益要件がこのコストを構成する。

### 3-tier min spread

$$\text{effective\_min} = \max\left(\underbrace{S_{\text{abs}}}_{\text{Tier 1}},\; \underbrace{P_{\text{mid}} \cdot \frac{b}{10000}}_{\text{Tier 2}},\; \underbrace{\sigma \cdot P_{\text{mid}} \cdot m}_{\text{Tier 3}}\right)$$

| Tier | パラメータ | 役割 | 値 |
|------|-----------|------|-----|
| 1 | `min_spread_jpy: 100` | 絶対安全ネット (mid 不正時) | 100 JPY |
| 2 | `min_spread_floor_bps: 3.8` | 価格連動フロア | BTC 13M → 494 JPY |
| 3 | `min_spread_atr_mult: 2.0` | ボラティリティ連動 | σ × mid × 2.0 |

## 624# バグ修正 (3 件)

### Bug 1: `_enforce_spread_guards` シグネチャ欠落

- **問題**: 呼び出し元が `mid_price=mid_price` を渡しているが、関数シグネチャに `mid_price` パラメータ未定義
- **影響**: ランタイム TypeError (テストは該当パスを通過しないため未検出)
- **修正**: keyword-only パラメータ `mid_price: float` を追加

### Bug 2: `self._last_mid` — 存在しない属性参照

- **問題**: `mid = self._last_mid or 0.0` だが `_last_mid` 属性は未定義 (`_prev_mid_price` が正しい)
- **影響**: Bug 1 で到達不能だったため実質無害 → Bug 1 修正で顕在化する
- **修正**: 引数 `mid_price` を直接使用

### Bug 3: `min_spread_atr_*` が flat_keys 未登録

- **問題**: `fill_config_parser.py` の `flat_keys` に `min_spread_atr_enabled` / `min_spread_atr_mult` が含まれておらず、YAML の値が読み込まれない
- **影響**: ATR 機能が `enabled=False` (デフォルト) のまま → 624# の機能が実質無効
- **修正**: `flat_keys` に追加

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | `min_spread_jpy: 500→100`, `min_spread_floor_bps: 3.8` 追加 |
| `scripts/v460/lib/fill_config.py` | `min_spread_floor_bps` フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | `flat_keys` に 3 フィールド追加 |
| `scripts/v460/lib/maker_price.py` | 3-tier min spread 実装 + Bug 1/2 修正 |
| `tests/unit/v460/test_190_ev_weighted_safety.py` | `min_spread_jpy` アサーション 500→100 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | `KNOWN_YAML_OVERRIDES` に 2 フィールド追加 |

## 価格帯別 BPS フロアの変化

| BTC/JPY | 固定 500 JPY (旧) | BPS × 3.8 (新) | 差分 |
|---------|:---:|:---:|:---:|
| 10,000,000 | 500 | 380 | -24% |
| 13,000,000 | 500 | 494 | -1% |
| 15,000,000 | 500 | 570 | +14% |
| 20,000,000 | 500 | 760 | +52% |

BTC 上昇時に自動でフロアが拡大し、注文処理コスト比率を一定に保つ。

## コミット

- `4ca3bc7f7` — feat(625#): min_spread動的BPSフロア + 624#バグ修正3件
