# 673# cap_bps 引下げ + max_lot デフォルト同期

- **日付**: 2026-03-31
- **前提**: 672# 深堀り分析で cap_bps が拘束パラメータと判明、669# max_lot コード未同期

---

## §1 背景: cap_bps が真の拘束パラメータ

### §1.1 624# 以降の ATR floor 構造

`_enforce_spread_guards` (maker_price.py) の 3-tier 最小スプレッド:

```
effective_min = max(S_abs, BPS_floor, min(ATR_floor_raw, ATR_cap))
```

| パラメータ | 式 | 現行値 (σ=0.000417, mid=12.5M) |
|---|---|---|
| S_abs | `min_spread_jpy` | 100 JPY |
| BPS_floor | `mid × 0.38 / 10000` | 475 JPY |
| ATR_floor_raw | `σ × mid × 1.2` | 6,255 JPY |
| ATR_cap | `mid × cap_bps / 10000` | **cap_bps=3.0 → 3,750 JPY** |

ATR_floor_raw (6,255) >> ATR_cap (3,750) なので **cap_bps が拘束**。
672# 提案の「atr_mult を 20% 引下げ」は mult が非拘束であり **効果なし**。

### §1.2 672# 分析結果 — 1500-2500 帯が最良

| 帯 (JPY) | α (Glosten-Milgrom) | AS cost | realized_hs 平均 |
|---|---|---|---|
| 0-1500 | 21.2% | +0.07 | +1.40 |
| **1500-2500** | **21.6%** | **-0.09** | **+2.11** |
| 2500-5000 | 22.0% | +0.15 | -0.34 |
| 5000+ | 25.5% | +1.23 | -2.67 |

1500-2500 帯: 情報非対称性が最低、AS コスト負（MM 有利）、PnL 最高。

---

## §2 実装

### §2.1 YAML 変更: cap_bps 3.0 → 2.0

**`configs/v460/fill_test.yaml`** L51:
```yaml
min_spread_atr_cap_bps: 2.0    # 673# 3.0→2.0: 672#分析で1500-2500帯α最低・realized_hs黒字。cap拘束下で3750→2500JPY
```

効果: effective_min が 3,750 JPY → 2,500 JPY に低下。1500-2500 帯の上端が約定可能に。

### §2.2 fill_config.py max_lot デフォルト同期

**`scripts/v460/lib/fill_config.py`** L142:
```python
max_lot: float = 0.001  # 669# 0.005→0.001: 1mBTC cap
```

669# で YAML を 0.001 に変更済みだったが、コード側デフォルトが 0.005 のまま不整合。
YAML 未指定時やテスト環境で旧値が使われるリスクを解消。

### §2.3 テスト修正

1. **test_253 行数上限**: 1545 → 1560 (671# NFQ 構造フィールド +11 行分)
2. **test_145 regime_mult テスト**: `_make_checker()` に `max_lot=1.0` を追加。
   669# max_lot clamp が regime_mult テストに干渉するのを防止。

---

## §3 σ レベル別の拘束パラメータ

cap_bps=2.0 適用後の拘束マップ:

| σ | ATR_floor_raw | ATR_cap (2.0bps) | 拘束側 | effective_min |
|---|---|---|---|---|
| 0.000100 | 1,500 | 2,500 | **mult** | 1,500 JPY |
| 0.000167 | 2,500 | 2,500 | 均衡点 | 2,500 JPY |
| 0.000200 | 3,000 | 2,500 | **cap** | 2,500 JPY |
| 0.000300 | 4,500 | 2,500 | **cap** | 2,500 JPY |
| 0.000417 | 6,255 | 2,500 | **cap** | 2,500 JPY |

σ < 0.000167 では mult が拘束に切替わる → 低ボラ環境では自然にスプレッド縮小。
現行 σ ≈ 0.000417 では cap が拘束。

---

## §4 リスク・観察ポイント

1. **3 日間観察**: cap_bps=2.0 デプロイ後、fill rate / avg_pnl30 / α を追跡
2. **次ステップ**: 効果確認後 cap_bps=1.5 (effective 1,875 JPY) への段階的引下げ検討
3. **deadlock_escape との相互作用**: 664# の `deadlock_escape_spread_mult: 0.5` は effective_min の半減 → cap_bps=2.0 時は 1,250 JPY まで緩和。1500-2500 帯の中心に到達可能
4. **672# atr_mult 提案の却下記録**: mult は非拘束のため引下げ無効。cap_bps が唯一の実効パラメータ

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | `min_spread_atr_cap_bps: 3.0 → 2.0` |
| `scripts/v460/lib/fill_config.py` | `max_lot` デフォルト `0.005 → 0.001` |
| `tests/.../test_253_...py` | 行数上限 `1545 → 1560` |
| `tests/.../test_145_...py` | `_make_checker` に `max_lot=1.0` 追加 |
| `docs/v460/673_...md` | 本ドキュメント |
