# 624# kill duration drift 修正 + min_spread ATR 連動

- **日付**: 2026-03-25
- **著者**: Copilot
- **コミット**: `fa3f8d218`
- **種別**: fix / impl
- **目的**: 605#-608# 監査で検出された kill duration drift を解消し、536# シナリオ A の ATR 連動最小スプレッドを実装

---

## §1 kill duration drift 修正

### 問題

| フィールド | コードデフォルト | YAML 値 | 導入元 |
|-----------|:--------------:|:-------:|:------:|
| `sell_dynamic_kill_max_duration_sec` | 1800s | 600s | 540# 短縮 |
| `buy_dynamic_kill_max_duration_sec` | 1800s | 900s | 370# 短縮 |

コメントは `# 336# drift fix: YAML=1800 (273#)` と記載されていたが、540# / 370# で YAML 値が短縮された際にコードデフォルトとコメントが更新されなかった。

YAML が常に優先されるため実害なし（production は YAML 値で動作）だが、コードデフォルトのみで FillTestConfig を構築するテスト（test_273）が 1800.0 を期待しており、実運用値との乖離が混乱の元になっていた。

### 修正

| フィールド | 変更前 | 変更後 |
|-----------|:------:|:------:|
| `sell_dynamic_kill_max_duration_sec` | `1800.0` | `600.0` |
| `buy_dynamic_kill_max_duration_sec` | `1800.0` | `900.0` |

- test_273 アサーション値を更新
- test_336 `KNOWN_YAML_OVERRIDES` から両フィールドを除去（drift 解消済み）

---

## §2 min_spread ATR 連動

### 背景

536# シナリオ A で提案:
> 「`ceiling` という固定の蓋を撤廃し、ベーススプレッド自体を『ATR（直近ボラティリティ）』の係数として動的に設定（例: ATR × 0.5）」

`min_spread_jpy` は固定 500 JPY で、ボラティリティに関わらず一律。低ボラ時はスプレッド過小でも通過し、高ボラ時は不十分な防御となる。

### 実装

**新規設定（fill_config.py）**:
```python
min_spread_atr_enabled: bool = False  # σ×mid×mult を min_spread に加算
min_spread_atr_mult: float = 2.0      # σ(fractional) × mid_price × mult
```

**ロジック（maker_price.py `_enforce_spread_guards`）**:
```python
effective_min = cfg.min_spread_jpy  # 固定フロア (500 JPY)
if cfg.min_spread_atr_enabled and self._last_sigma > 0 and mid_price > 0:
    atr_min = self._last_sigma * mid_price * cfg.min_spread_atr_mult
    effective_min = max(effective_min, atr_min)
```

**動作**:
- `effective_min = max(固定フロア, σ × mid × mult)`
- σ は Parkinson / Roll estimator（maker_microstructure.py）で推定済み
- 固定フロア 500 JPY は常に最低保証（σ = 0 のウォームアップ期間対策）

### 数値例

| 状態 | σ (fractional) | mid_price | σ × mid × 2.0 | effective_min |
|------|:--------------:|:---------:|:--------------:|:-------------:|
| 低ボラ | 0.0002 | 14,000,000 | 5,600 JPY | 5,600 (ATR) |
| 中ボラ | 0.0005 | 14,000,000 | 14,000 JPY | 14,000 (ATR) |
| 高ボラ | 0.001 | 14,000,000 | 28,000 JPY | 28,000 (ATR) |
| σ = 0 | 0.0 | 14,000,000 | 0 | 500 (固定フロア) |

### YAML 設定

```yaml
min_spread_atr_enabled: true
min_spread_atr_mult: 2.0
```

hot-reload 対応: `min_spread_atr_enabled` / `min_spread_atr_mult` は fill_config フィールドとして再読み込み可能。

---

## §3 536# 渙原則との整合

| 536# 指摘 | 624# 対応 |
|-----------|----------|
| 「固定値 `min_spread_jpy (700等)` を散らす」 | 固定フロアは残しつつ ATR 動的化を上積み |
| 「ATR × 係数 でベーススプレッドを動的設定」 | `σ × mid × mult` で実装 |
| 「廟（最終防波堤）は残す」 | 固定フロア 500 JPY は最低保証として残存 |

---

## §4 広範囲残課題サーベイ

| # | P | 項目 | 状態 |
|---|---|------|------|
| 1 | - | sell_dynamic_kill 操作性 | ✅ 稼働中。536# 議論あるが 540# + 543# で Toxicity Budget 段階応答に進化済み |
| 2 | - | trending_sell_skip | ✅ 稼働中（max_consecutive=10）。事後ガードとして残存 |
| 3 | P3 | time_filter 形骸化 | `enabled: true` だが全リスト空（169# 全廃）。実害なし・ゾンビ |
| 4 | P3 | eDRC | 前提条件未達（621# T3） |
| 5 | P3 | entry_gate 本稼働 | observe モード、CalibrationMap データ蓄積待ち |

P1/P2 レベルの残課題は本 624# で解消。P3 はデータ蓄積依存のため現時点では対応不要。

---

## §5 テスト

全 2237 テスト pass（127 skipped, 81 warnings）。
