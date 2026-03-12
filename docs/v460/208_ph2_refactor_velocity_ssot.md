# 208# Velocity SSOT 強化 — instant velocity 計算の一元化

> **日付**: 2026-03-02  
> **前提**: 205# §3.2 (Codex) / §9.1 (Gemini) で指摘された velocity SSOT 未達への対応  
> **コミット**: `6b9c0842d`

---

## 1. 背景

205# §3.2 (Codex):
> 201# の `velocity_math.py` は offset multiplier の計算式を共通化しただけ。
> maker_price.py は `mid_trend_bps`、skip_gate_evaluator.py は `price_velocity_60s` を
> 別々に使っている。これは「SSOT」というより「二つの異なる velocity に同じ multiplier 計算式を使うようにした」に留まる。

205# §9.1 (Gemini):
> mid_trend_bps を即刻破棄し price_velocity_60s に強制的一本化

## 2. 分析

二つの velocity は **異なるデータソース・異なるタイムウィンドウ** から計測されており、
それぞれ固有の用途がある。

| 信号名 | データソース | 窓 | 用途 |
|---|---|---|---|
| instant_vel_bps | orderbook mid-price (point-to-point) | ~5s | VG offset boost (瞬間急変) |
| trade_vel_60s | 約定履歴 (first↔last) | 60s | SG skip/offset + ML feature |

Gemini §9.1 の「強制一本化」は VG の即応性を犠牲にするため採用しない。
代わりに、**計算・符号規約・文書を velocity_math.py に集約** する SSOT アプローチを取る。

## 3. 実装

### 3.1 `compute_instant_velocity_bps()` 追加

maker_price.py 内の inline 計算 (054#) を `velocity_math.py` に関数として抽出:

```python
def compute_instant_velocity_bps(
    *, current_mid, prev_mid, dt, max_dt,
) -> float | None:
```

- 0 除算防止: `prev_mid <= 0` → None
- stale 防止: `dt >= max_dt` → None
- 符号規約: 正=上昇, 負=下降 (trade_vel_60s と同一)

### 3.2 maker_price.py 変更

```python
# Before (inline):
mid_trend_bps = (mid_price - prev_mid) / prev_mid * _BPS_FACTOR

# After (SSOT):
mid_trend_bps = compute_instant_velocity_bps(
    current_mid=mid_price, prev_mid=..., dt=..., max_dt=...,
)
```

### 3.3 ドキュメント強化

`velocity_math.py` の module docstring を全面改訂:
- 二重信号アーキテクチャの明示 (テーブル形式)
- 共有要素の定義 (符号規約, bps 単位, offset 乗数計算)
- 205# §3.2 / §9.1 へのトレーサビリティ

## 4. テスト

新規 6 テスト (`TestInstantVelocityBps`):
- 上昇: 100bps 正確
- 下降: -50bps 正確
- stale (dt > max_dt): None
- ゼロ dt: None
- ゼロ prev_mid (0除算防止): None
- 符号規約一致: 上昇=正, 下降=負

既存テスト: 584 passed (変更なし)

## 5. 残課題

| 課題 | 状態 | 備考 |
|---|---|---|
| trade_vel_60s の velocity_math 移動 | 未着手 | gate_features パイプライン内で密結合。将来的に extract 可能 |
| VG boost の velocity_math 統合 | 見送り | VG は binary trigger + inv_skew damping で固有ロジックが多い。無理な統合は複雑化を招く |
