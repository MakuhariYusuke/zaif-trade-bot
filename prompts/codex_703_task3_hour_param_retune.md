# 703# Task 3: 12-17h UTC 時間帯パラメータ再調整

## 背景
Protocol 688 再分析 (702#) で 12-17h UTC (JST 21-02h) に損失が集中:
- 12-17h: 86 fills, total=-174.2bps, avg=-2.03bps
- その他: 324 fills, total=+16.5bps, avg=+0.05bps (黒字)

既存の sell_hour_offset_boost / skip_gate_hour_offsets / hour_ceiling_mult は設定済みだが、
P688 最新データに基づく再調整が必要。

## 修正箇所

### `configs/v460/fill_test.yaml` のみ (コード変更なし)

#### 1. sell_hour_offset_boost (L814付近)
現行値 → 新値:
```yaml
sell_hour_offset_boost:
  # ... 既存エントリは維持 ...
  14: 2.5    # 702# 1.5→2.5: P688 avg=-6.05bps (n=4). 最悪時間帯ペナルティ強化
  16: 2.5    # 702# 1.5→2.5: P688 avg=-3.41bps (n=22). 損失集中帯
```

#### 2. skip_gate_hour_offsets (L555付近)
追加:
```yaml
skip_gate_hour_offsets:
  # ... 既存エントリは維持 ...
  12: 0.3    # 702# 新規: P688 avg=-1.73bps (n=20). JST21h skip_gate 閾値厳格化
```

#### 3. hour_ceiling_mult (L564付近)
追加/変更:
```yaml
hour_ceiling_mult:
  # ... 既存エントリは維持 ...
  12: 1.5    # 702# 新規: P688 avg=-1.73bps (n=20). JST21h ceiling 拡大で防御許容
  16: 2.5    # 702# 2.0→2.5: P688 avg=-3.41bps (n=22). ceiling 強化
```

## テスト
- `tests/unit/v460/test_702_hour_param_update.py`
  - test_sell_hour_offset_boost_updated: YAML 値が計画通りか
  - test_skip_gate_hour_offsets_12h_added: UTC 12h に 0.3 が設定されている
  - test_hour_ceiling_mult_updated: 12h=1.5, 16h=2.5 が設定されている
  - test_existing_hour_params_unchanged: 既存の 2h, 4h, 8h, 9h 等が変更されていない

## 制約
- 既存の時間帯テスト (`sell_hour_offset_boost`, `hour_ceiling_mult` 参照テスト) が引き続きパス
- YAML コメントに必ず 702# 根拠 (P688 avg_pnl, n=) を記載
- 新規コードファイルは不要（YAML config 変更のみ）

## 設計根拠
- sell_hour_offset_boost 14h: n=4 は smallだが -6.05bps は極端。2.5× で sell offset を 2.5倍に拡大
- sell_hour_offset_boost 16h: n=22 で -3.41bps は統計的に信頼できる。2.5× に強化
- skip_gate_hour_offsets 12h: n=20 で -1.73bps。0.3 offset で ML skip 判定を厳格化
- hour_ceiling_mult 16h: 現行 2.0 → 2.5。AS64% 時間帯で ceiling 通過を許容
