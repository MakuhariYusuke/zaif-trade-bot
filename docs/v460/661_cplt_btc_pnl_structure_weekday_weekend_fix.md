# 661# BTC建て収益構造分析・曜日効果・B-3結論・構造改善

## 概要

660# パラメータチューニング後、BTC建て視点で全期間 (2/13–3/30) を再分析。
**全46日間で -5.44 mBTC** という構造的損失を特定。
根本原因の特定と、2つの大型修正を実施。

## 1. BTC建て収益構造 (全期間)

| 指標 | 値 |
|------|-----|
| 全fills | 5,438 (46日) |
| Buy VWAP | 10,854,935 JPY |
| Sell VWAP | 10,841,683 JPY |
| VWAP spread | **-13,252 JPY (-0.12%)** |
| Net BTC delta | +0.140 BTC (net buyer) |
| Net JPY delta | -1,554,795 JPY |
| **BTC PnL** | **-5.44 mBTC** |
| JPY PnL | -58,185 JPY |

### Sell VWAP < Buy VWAP = 逆選択

matched 2.649 BTC に対し spread PnL = **-3.28 mBTC**（逆鞘）。
0.140 BTC の unmatched long position は **-2.16 mBTC**（含み損）。

## 2. 根本原因

### 原因1: ranging_buy_priority による買いバイアス (60%)

`side_selector.py` の `ranging_buy_priority_max_consecutive: 3` が、
ranging レジーム（全fills の74%）で sell → buy に上書き。

- Buy 2,789 fills vs Sell 2,649 fills = **+140 net buy surplus**
- W11 (最悪週): 519 buys vs 434 sells = +85 surplus, **-2.0 mBTC**
- W11 price rally +9.3% → 高値 buy → 反落で全損

```
=== Cumulative Side Imbalance ===
W06: +11, W07: +0, W08: +11, W09: +1,
W10: -4, W11: +85, W12: +35, W13: +1 → Total: +140
```

### 原因2: 週末トレード (55%)

| 曜日 | BTC PnL | Mean bps | Fills |
|------|---------|----------|-------|
| Mon | -0.38 mBTC | +0.14 | 522 |
| Tue | -0.23 mBTC | -0.24 | 517 |
| Wed | +0.32 mBTC | -0.04 | 629 |
| Thu | -0.32 mBTC | -0.27 | 859 |
| Fri | -0.51 mBTC | -0.29 | 1,067 |
| **Sat** | **-1.84 mBTC** | **-0.43** | 1,065 |
| **Sun** | **-1.17 mBTC** | **-0.54** | 779 |

**Sat + Sun = -3.01 mBTC** (全損失の55%)。

## 3. T2-4: 曜日×時間帯効果分析

### Best Windows (≥5 fills, PnL30 bps mean)
```
Mon 23:00  +3.03 bps (35 fills)
Tue 10:00  +3.01 bps (26 fills)
Mon 16:00  +2.93 bps (22 fills)
Tue 00:00  +2.56 bps (27 fills)
Fri 20:00  +1.78 bps (38 fills)
```

### Worst Windows
```
Thu 23:00  -4.34 bps (29 fills)
Thu 22:00  -3.36 bps (45 fills)
Mon 12:00  -3.34 bps (14 fills)
Sun 06:00  -2.89 bps (15 fills)
Sat 01:00  -2.68 bps (28 fills)
```

## 4. B-3 効果観測 (結論)

### inv_skew
- 1,034 samples 中 441 (42.6%) で非ゼロ補正
- **corrected PnL30: -0.42 bps vs uncorrected: -0.24 bps**
- ranging で 428/813 fills が補正対象 → regimeは最も active
- trending では regime_gate_enabled=false 移行後、max_factor_trending=0.15 で低活性

**判定**: inv_skew は稼働しているが、ranging_buy_priority の買いバイアスを相殺しきれていない。
buy_priority 撤廃後は inv_skew が本来の機能を発揮する見込み。

### toxic_veto
- toxic_veto_set: 446回（検知は活発）
- toxic_veto_block: **61回**（注文ブロック実績あり）
- toxic_veto_as_offset: 0回（offset 調整は未発動）
- fill_records 上のブロック: 0件（cancel_reason に記録されない仕様の可能性）

**判定**: 検知・ブロックは機能しているが、fill_records の可視性が不足。
as_offset 未発動は調査余地あり。重大な問題ではないため observe 継続。

## 5. 変更内容

### 5-1. ranging_buy_priority 無効化 (config)

```yaml
ranging_buy_priority_max_consecutive: 0   # 661# 3→0
```

**影響**: ranging レジームでの sell → buy 上書きを完全停止。
buy/sell が自然な交互選択に戻り、+140 surplus の根本原因を解消。

### 5-2. 週末スキップ (code + config)

**コード変更**:
- `fill_config.py`: `skip_days_of_week: list[int]` フィールド追加
- `fill_config_parser.py`: YAML `time_filter.skip_days_of_week` パーサー追加
- `time_filter.py`: `is_filtered()` 内で曜日判定追加

```yaml
time_filter:
  skip_days_of_week: [5, 6]    # 661# Sat/Sun全停止
```

**影響**: 土曜・日曜は全サイクルをスキップ。**-3.01 mBTC/46日** の損失源を遮断。

### テスト
- `test_661_skip_days_of_week.py`: 新規テスト 7 件（全パス）
- `test_336`: KNOWN_YAML_OVERRIDES 追加
- `test_634`: YAML round-trip 値更新
- 全 4,467 テストパス

## 6. 期待される効果

| 変更 | 期待削減 | 根拠 |
|------|----------|------|
| buy_priority=0 | ~-2.0 mBTC/46日 | VWAP逆鞘 + unmatched position |
| 週末停止 | ~-3.0 mBTC/46日 | Sat+Sun BTC PnL |
| **合計** | **~-5.0 mBTC/46日** | 全損失の92% |

## 7. 残課題更新 (661# 時点)

### クローズ
- **T2-4**: 曜日効果分析 → 完了（本ドキュメント）
- **B-3**: inv_skew + toxic_veto 効果観測 → 結論出し完了

### 計測待ち
- **661# 効果観測**: buy_priority=0 + 週末停止の実運用効果 → 1週間観測
- **inv_skew 再評価**: buy_priority 撤廃後の inv_skew 単体効果を再測定

### Tier 1 (短期・保留中)
| ID | タスク | 備考 |
|----|--------|------|
| T1-2 | RT 主 KPI 化 | 分析基盤整理 |
| T1-3 | Regime 遷移 AS セクション | section 未実装 |
| T1-4 | AS burst 自己相関 | section 未実装 |
| T1-5 | PnL 計測窓正規化 | 命名・実態確認 |

### Tier 2 (中期)
| ID | タスク | 前提条件 |
|----|--------|----------|
| T2-1 | eDRC α/β 再推定 | 661# 安定後 |
| T2-3 | sell ceiling → VG 連動 | 中期改善 |
| T2-5 | Asymmetric RT exit | position tracking |
| T2-6 | Regime-drift exit | regime 遷移検知 |
| T2-8 | sell_dynamic_kill 存廃 | C-4 ARL 計測後 |
| T2-9 | preflight バッファ | 661# buy bias 解消で改善見込み |
