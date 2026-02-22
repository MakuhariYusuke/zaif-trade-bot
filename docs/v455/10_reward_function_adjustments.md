# v455 Fast Intraday Reward 改善案（Fee/Vol/TTL）

## 目的
- **手数料負け回避**: 0.1% 手数料 + スリッページを上回る期待値が無いと入らない。
- **在庫リスク抑制**: 長時間の保有を避ける（TTL=60分想定）。
- **低ボラ抑制**: ATR/Price が小さい局面では取引を控える。

## 変更点（実装済み）
- `compute_hft_reward` に **Edge Penalty / Low-Vol Penalty / Time-Decay Penalty** を追加。
- `scripts/v455/train_hft.py` と `scripts/v455/tune_hft.py` に新パラメータを反映。

## 追加したロジック（数式）
### 1) 手数料負け回避（Edge Penalty）
定義:
- `delta = |position_now - position_prev|`
- `trade_cost = fee_paid + slippage_paid` (JPY)
- `expected_move = ATR * delta` (JPY)
- `required_edge = trade_cost * min_edge_mult`

ペナルティ:
- `edge_shortfall = max(0, required_edge - expected_move)`
- `edge_penalty = edge_penalty_rate * (edge_shortfall / denom)`

狙い:
- **ATRが十分でない（低ボラ）かつコストが高い局面**のトレードを強く抑制。

### 2) 低ボラ抑制（Low-Vol Penalty）
定義:
- `vol_ratio = ATR / price`
- `low_vol_shortfall = max(0, vol_floor - vol_ratio)`

ペナルティ:
- `low_vol_penalty = vol_floor_penalty * low_vol_shortfall * (|position_now| / max_position)`

狙い:
- **ATR/Price < vol_floor** のときは、保有を維持するほど不利になる。

### 3) 長期保有抑制（Time-Decay Penalty）
定義:
- `extra_hold = max(0, holding_steps - hold_grace)`

ペナルティ:
- `time_decay_penalty = hold_ramp * extra_hold * (|position_now| / max_position)`

狙い:
- **一定時間を超えた保有**にのみ追加コストを課す。

## 反映済みパラメータ（初期案）
`scripts/v455/train_hft.py` / `scripts/v455/tune_hft.py` に反映済みの初期値:
```python
reward_params = {
    "alpha": 0.5,
    "beta": 0.02,
    "min_edge_mult": 1.2,
    "edge_penalty_rate": 1.0,
    "vol_floor": 0.001,         # 0.1% ATR/Price
    "vol_floor_penalty": 50.0,
    "hold_grace": 10,
    "hold_ramp": 0.01
}
```

### 推奨チューニングレンジ（目安）
- `min_edge_mult`: **1.0 – 1.5**
- `edge_penalty_rate`: **0.5 – 2.0**
- `vol_floor`: **0.0005 – 0.002**（0.05% – 0.2%）
- `vol_floor_penalty`: **10 – 80**
- `hold_grace`: **5 – 20**
- `hold_ramp`: **0.005 – 0.02**

## 期待される挙動
- **取引回数が減少**し、低ボラ局面ではフラットを維持。
- **損益分岐に届かないエントリー**が減る（手数料負けの抑制）。
- **長期保有の罰則強化**により、短期決済を促進。

## 補足アイデア（次の一手）
- **No-Trade Bonus**: 低ボラ時に「何もしない」行動へ微小ボーナス。
- **エッジ可視化**: `edge_shortfall` / `vol_ratio` / `trade_cost` をログに出す。
- **Gate連携**: Reward側だけでなく EV Gate と共通のコスト/ボラ基準を使う。

## 変更ファイル
- `ztb/trading/rewards/fast_intraday.py`
- `scripts/v455/train_hft.py`
- `scripts/v455/tune_hft.py`
