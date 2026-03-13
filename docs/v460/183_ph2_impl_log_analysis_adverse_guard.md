# 183# ログ分析ベース逆選択防御強化

## 概要
fill_test.log (47,404行, 15日間) と fill_records (4,671件, 1,991 fills) をデータ分析し、
逆選択 (Adverse Selection) を最大のロス要因として特定。5つの改善を実施。

## ログ分析の主要発見

### 全体パフォーマンス
| 指標 | 値 |
|------|-----|
| 総Fill数 | 1,991 (fill rate 42.6%) |
| 30s PnL | -0.30 bps/fill (累計 -595.5 bps) |
| WR | 46.3% |
| 逆選択率 | 28.2% (561/1,991) |

### 逆選択の影響
| グループ | n | Mean 30s PnL | WR | 累計 |
|----------|---|-------------|-----|------|
| Adverse | 561 | **-5.90 bps** | 0.0% | -3,310 bps |
| Non-adverse | 1,430 | **+1.90 bps** | 64.4% | +2,715 bps |

→ **逆選択を除けばシステムは黒字** (核心的発見)

### 逆選択の最強予測因子
| Feature | Adverse | Non-adverse | 差分 |
|---------|---------|-------------|------|
| VG velocity (bps) | med=-0.95 | med=+0.83 | **1.78** |
| Spread <2k rate | 32% | 25-28% | 4-7pt |
| 01h JST rate | 64% | - | 36pt vs avg |

### 時間帯別 Adverse 率 (JST)
| Hour | AS% | Sum PnL | 判定 |
|------|-----|---------|------|
| 01h | **64%** | -79.8 | 最悪 |
| 03h | **50%** | -54.8 | 悪い |
| 06h | 36% | **-125.8** | PnL最悪 |
| 17h | **42%** | -63.1 | 悪い |
| 23h | **43%** | **-134.5** | 両方悪い |

## 改善内容 (5項目)

### 1. 時間帯別 skip_gate 閾値オフセット (YAML)
```yaml
skip_gate:
  hour_offsets:
    14: 0.3    # 23h JST
    16: 0.5    # 01h JST (最厳格)
    18: 0.3    # 03h JST
    21: 0.3    # 06h JST
    23: 0.2    # 08h JST
```
既存の `skip_gate_hour_offsets` 機構 (158# P1-6) を活用。
PnL モードで正 = 厳格化 (予測PnLが高くないとskip)。

### 2. Buy velocity skip 有効化 + 閾値保守化 (YAML)
```yaml
skip_gate:
  buy_velocity_skip_enabled: true    # false → true
  buy_velocity_skip_threshold_bps: -6.0   # -8.0 → -6.0
  sell_velocity_skip_threshold_bps: 6.0   # 8.0 → 6.0
```
- Buy velocity skip: 急落局面での買い参入を防御
- 閾値を 8→6 bps に保守化し捕捉率向上

### 3. 狭スプレッド逆選択防御 (コード + YAML)
**新規コード**: `skip_gate_evaluator.py` にnarrow spread adverse guard追加

```python
# spread < threshold → skip_gate 閾値を厳格化
_ns_thr = self._config.skip_gate_narrow_spread_threshold_jpy
if _ns_thr > 0 and spread_at_order < _ns_thr:
    _spread_offset = self._config.skip_gate_narrow_spread_offset
```

```yaml
skip_gate:
  skip_gate_narrow_spread_threshold_jpy: 2000.0
  skip_gate_narrow_spread_offset: 0.2
```

**新規Config**: `fill_config.py` に2フィールド追加、`config_hot_reload.py` に登録。
hour_offset と加算されて `threshold_offset` に反映。

### 4. Volatility Guard 感度引上げ (YAML)
```yaml
volatility_guard:
  velocity_threshold_bps: 12.0   # 15.0 → 12.0
  vpin_threshold: 0.60           # 0.63 → 0.60
```
VG velocity が逆選択の最強因子 (adv med=-0.95 vs non-adv med=+0.83)。
感度を引き上げて offset boost 発動率を向上。

### 5. Narrow spread boost 強化 (YAML)
```yaml
spread_adaptive:
  narrow_spread_boost_buy: 2.0    # 1.5 → 2.0
  narrow_spread_boost_sell: 2.5   # 2.0 → 2.5
```
spread <2k での AS 32% に対応。offset 拡大で逆選択回避。

## 期待効果 (反実仮想推計)

| 改善項目 | 対象Fill数 | 推定PnL改善 |
|----------|-----------|-------------|
| 時間帯フィルタ | ~300 fills/15d | +50-100 bps |
| Velocity skip | ~30 fills/15d | +20-40 bps |
| 狭スプレッドguard | ~475 fills/15d | +30-60 bps |
| VG感度引上げ | 全fill | +20-50 bps |
| Narrow boost | ~475 fills/15d | +10-30 bps |
| **合計** | | **+130-280 bps/15d** |

→ -595.5 bps が -315 〜 -465 bps に改善見込み (22-55% 損失削減)

## 変更ファイル
| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | 5項目のパラメータ調整 |
| `scripts/v460/lib/skip_gate_evaluator.py` | narrow spread offset 追加 |
| `scripts/v460/lib/fill_config.py` | 2フィールド追加 |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload 登録 |
| `tests/unit/v460/test_183_log_analysis_improvements.py` | 16テスト新規 |
| `tests/unit/v460/test_093_side_params.py` | YAML値更新 |
| `tests/unit/v460/test_fill_quality.py` | VG閾値更新 |

## テスト結果
- 183# テスト: **16 passed**
- v460 回帰: **2330 passed, 0 failed**
