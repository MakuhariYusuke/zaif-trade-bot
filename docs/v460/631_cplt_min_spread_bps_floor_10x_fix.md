# 631# min_spread BPS フロア 10× 計算ミス修正 (625# リグレッション)

## 概要

625# で導入した `min_spread_floor_bps: 3.8` が **10× の計算ミス** であったことを修正。
正しい値は `0.38`。630# 再起動後「取引ゼロ」の直接原因。

---

## 障害経緯

| 時刻 | イベント |
|------|---------|
| 625# | min_spread BPS フロア導入: `3.8` bps |
| 630# 19:55 | P1 閾値変更 + hot_swap restart |
| 630# 19:55– | **29 連続 infeasible quote** (`spread_too_narrow`) |
| 631# | 根本原因特定 → `3.8` → `0.38` 修正 |

## 根本原因

625# のドキュメントには:

> `min_spread_floor_bps: 3.8  # BTC 13M時 ≈ 494JPY`

と記載されているが、実際の計算:

$$13{,}000{,}000 \times \frac{3.8}{10{,}000} = 4{,}940 \text{ JPY}$$

ドキュメント内の価格帯別テーブル (380, 494, 570, 760 JPY) は **0.38 bps** で計算された正しい値。
YAML に入力された `3.8` が10倍間違い。

$$13{,}000{,}000 \times \frac{0.38}{10{,}000} = 494 \text{ JPY} \approx \text{旧} 500 \text{ JPY}$$

## ログ証跡

```
[158# §20-D] Spread too narrow: 2844 JPY < min 4342 (abs=100, bps=4342, atr=0)
[234#] NO_FEASIBLE_QUOTE: 29 consecutive infeasible quotes (buy) -- last_reason=spread_too_narrow
```

- coincheck BTC/JPY スプレッド: ~1,700–2,800 JPY (正常)
- BPS フロア (3.8bps): ~4,340 JPY → **常に spread < min → 全棄却**
- BPS フロア (0.38bps): ~434 JPY → spread > min で正常通過

## 副次発見: 重複プロセス

hot_swap 再起動時に PID 36852 (venv Python) と PID 37536 (system Python) の重複を検出。
Lock ファイルは PID 37536 を保持。PID 36852 を手動停止。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | `min_spread_floor_bps: 3.8` → `0.38` |
| `docs/v460/625_cplt_min_spread_dynamic_bps_floor.md` | 3 箇所の `3.8` → `0.38` 修正 (記録修正) |

## 教訓

- **AI 生成の計算値は必ず検算すべき** (592#/605# の教訓の再現)
- BPS 計算: `価格 × bps / 10000`。bps=3.8 → 0.038%、bps=0.38 → 0.0038%
- 625# のドキュメントでは計算結果は正しかった (0.38 で計算) が、YAML の入力値が 10× 誤っていた
