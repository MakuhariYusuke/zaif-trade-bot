# CX6: OBI 絶対値ベース非対称ファクター (U字型対応)

## 背景

707# は `ranging_obi_asymmetry_factor` の引上げ (0.3→0.6) と `ranging_obi_threshold` の引上げ (0.1→0.25) を提案。
しかし 708# 検証で OBI-PnL 関係は **U 字型** であることが判明:

```
OBI 帯            |  n  | avg buy PnL (bps)
[-1.0, -0.25)     |  55 | -0.545  (sell_heavy → 悪い)
[-0.25, 0.0)      |  52 | -0.059  (mild → OK)
[0.0, 0.25)       |  70 | -0.096  (mild → OK)
[0.25, 1.0)       |  65 | -0.602  (buy_heavy → 悪い)
```

現行の `offset += OBI * factor` は線形モデルで、buy_heavy 時に offset を **下げる** → 最も危険な帯域で防御が薄くなる。

## 現在の実装

```yaml
ranging_obi_asymmetry_factor: 0.3  # offset に OBI × factor を加算
ranging_obi_threshold: 0.1         # |OBI| > threshold で発動
```

コード: `scripts/v460/lib/` 内の offset pipeline で `OBI * factor` を additive に適用。

## 要件

### 1. U 字型対応のファクター設計

3 候補を実装し、CLI で比較可能にする:

**案1: 絶対値ベース**
```python
if abs(obi) > threshold:
    offset_boost = abs(obi) * factor  # 常に正 → 常にオフセット増
```

**案2: 二次関数ベース**
```python
if abs(obi) > threshold:
    offset_boost = (obi ** 2) * factor  # 端ほど急激に増
```

**案3: 中立帯 + 両端ガード**
```python
if abs(obi) > threshold:
    offset_boost = (abs(obi) - threshold) * factor  # 超過分のみ
else:
    offset_boost = 0.0  # 中立帯はゼロ
```

### 2. config 拡張

```yaml
ranging_obi_asymmetry_factor: 0.3
ranging_obi_threshold: 0.1
ranging_obi_mode: "absolute"  # 新規: "linear" (現行) | "absolute" | "quadratic" | "excess"
```

- `linear`: 現行互換 (`OBI * factor`)
- `absolute`: 案1 (`|OBI| * factor`)
- `quadratic`: 案2 (`OBI² * factor`)
- `excess`: 案3 (`(|OBI| - threshold) * factor`)

### 3. counterfactual CLI

`scripts/v460/analysis/obi_mode_comparison.py` を作成:

```bash
.venv/Scripts/python.exe -m scripts.v460.analysis.obi_mode_comparison \
  --results-dir results/v460/fill_test \
  --date-from 2026-04-04 --date-to 2026-04-06 \
  --modes linear,absolute,quadratic,excess \
  --json
```

出力:
- mode 別 × OBI 帯別の offset boost 統計
- PnL-weighted impact 推定 (各 fill に適用した場合の net PnL 変化)
- buy/sell 別の分解

### 4. テスト

- `tests/unit/v460/test_710_obi_modes.py`:
  - 各 mode の offset 計算検証 (positive/negative/zero OBI)
  - threshold 境界のテスト
  - 既存 `ranging_obi_mode` 未設定時は `linear` (後方互換)
  - `absolute` mode で buy_heavy/sell_heavy 両方に positive boost

### 5. 安全制約

- `ranging_obi_mode` を設定しない場合は `linear` にフォールバック（後方互換）
- FillTestConfig, parser, validation, hot-reload に追随
- mode 変更は hot-reload 対応

## 影響範囲

| ファイル | 変更 |
|---------|------|
| `scripts/v460/lib/offset_pipeline.py` or `multiplicative_pipeline.py` | OBI 計算ロジック mode 分岐 |
| `scripts/v460/lib/fill_config.py` | `ranging_obi_mode` フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | parser 追随 |
| `scripts/v460/lib/fill_config_validation.py` | valid modes チェック |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload 追随 |
| `configs/v460/fill_test.yaml` | `ranging_obi_mode: linear` 追加 |
| `scripts/v460/analysis/obi_mode_comparison.py` | 新規 |
| `tests/unit/v460/test_710_obi_modes.py` | 新規テスト |

## 成功基準

1. `absolute` mode で OBI 両端に positive boost が適用される
2. `linear` mode で既存動作が完全再現される
3. counterfactual CLI で `absolute` が `linear` より net PnL impact で優位
4. 既存テストの regression なし
