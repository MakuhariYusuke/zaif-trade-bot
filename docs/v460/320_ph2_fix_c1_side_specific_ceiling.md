# 320# C-1 根本対策: サイド別 Offset Ceiling + dcc3064 暫定評価

> **種別**: fix  
> **起票**: 2026-03-07  
> **起源**: 319# C-1 sell パイプライン全死の根本解消  
> **前提**: 319# (`21bcd6885`), 306# E1 (ceiling 導入), 246# (floor 0.30)

---

## §1 C-1 問題の定量的深堀り

### §1.1 問題構造

```
[maker_price.py パイプライン]
base(0.18) → floor(0.30) → 13段ステージ → ceiling(0.15) = 常に 0.15
                                                ↑ 12+ パラメータが全死

[fill_cycle_executor.py ポスト処理]
ceiling結果(0.15) → trending ×4.0 = 0.60  ← 唯一の実効制御
```

**根本原因**: `sell_guard.offset_floor: 0.30` (246#) と `offset_ceiling_ratio: 0.15` (306# E1) が独立に設定され、**floor > ceiling** の論理矛盾が未検出だった。

### §1.2 定量証拠 (offset_stages 分析)

全 fill_record の `offset_stages` フィールドを解析した結果:

| 指標 | SELL | BUY |
|---|---|---|
| offset_stages 保有レコード | 30 | 30 |
| **ceiling hit 率** | **100.0% (30/30)** | 46.7% (14/30) |
| pre-ceiling offset (mean) | 0.2990 | 0.2596 |
| pre-ceiling offset (max) | 0.3000 | 0.3796 |
| ceiling による削減量 (mean) | **0.1490** | 0.1096 |

**SELL 側は 100% ceiling ヒット** — パイプライン出力に分化なし。

### §1.3 Sell パイプライン各ステージの実態

| ステージ | n | mean | min | max | p50 | 評価 |
|---|---|---|---|---|---|---|
| base | 30 | 0.1800 | 0.1800 | 0.1800 | 0.1800 | 定数 |
| as_shift (=floor) | 30 | **0.3000** | 0.3000 | 0.3000 | 0.3000 | **floor が支配** |
| regime | 30 | 0.2605 | 0.2100 | 0.3000 | 0.2700 | 分化あり (0.21-0.30) |
| spread_adapt | 30 | 0.2930 | 0.2100 | 0.3000 | 0.3000 | floor 近辺に復帰 |
| kyle | 30 | 0.2930 | 0.2100 | 0.3000 | 0.3000 | 同上 |
| amihud | 30 | 0.2930 | 0.2100 | 0.3000 | 0.3000 | 同上 |
| vol_guard | 30 | 0.2990 | 0.2700 | 0.3000 | 0.3000 | ほぼ定数 |
| imb_risk | 30 | 0.2990 | 0.2700 | 0.3000 | 0.3000 | 同上 |
| loss_boost | 30 | 0.2990 | 0.2700 | 0.3000 | 0.3000 | 同上 |
| ffd | 30 | 0.2990 | 0.2700 | 0.3000 | 0.3000 | 同上 |
| **ceiling** | **30** | **0.1500** | **0.1500** | **0.1500** | **0.1500** | **全削除** |

**floor(0.30) → regime 分化(0.21-0.30) → 各ステージで 0.30 に復帰 → ceiling(0.15) で一律クランプ**

regime ステージは trending_down sell で 0.21 (×0.7 discount) まで下がるが、次段 spread_adapt で 0.30 に戻される。結果として全ステージが 0.27-0.30 に収束し、ceiling で全て 0.15 にクランプ。

### §1.4 Buy パイプラインとの比較

Buy 側は floor がないため、パイプラインが正常に機能:

| ステージ | n | mean | min | max | p50 |
|---|---|---|---|---|---|
| base | 30 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| as_shift | 30 | 0.1082 | 0.0471 | 0.3000 | 0.0500 |
| regime | 30 | 0.1217 | 0.0350 | 0.5400 | 0.0459 |
| ffd | 30 | 0.1682 | 0.0450 | 0.3796 | 0.1423 |
| ceiling | 14 | 0.1500 | 0.1500 | 0.1500 | 0.1500 |

Buy 側: 46.7% が ceiling ヒット、53.3% はパイプラインがそのまま有効。 0.045-0.38 の広い分化範囲。

### §1.5 死亡していた YAML パラメータ一覧

以下の sell 側パラメータは ceiling(0.15) により全て無効:

1. `side_offset.sell: 0.18` → floor で 0.30 に上書き
2. `trending_up_sell_offset_boost: 1.8` → floor(0.30)×1.8=0.54 → ceiling(0.15)
3. `trending_down_sell_offset_boost: 0.7` → floor(0.30)×0.7=0.21 → ceiling(0.15)
4. `ranging_offset_discount: 0.90` → 0.30×0.9=0.27 → ceiling(0.15)
5. `sell_hour_offset_boost: {8:1.5, 13:1.3, 14:1.3, 16:1.5}` → ceiling(0.15)
6. `volatility_guard_*` → ceiling(0.15)
7. `kyle_lambda_*` → ceiling(0.15)
8. `amihud_illiq_*` → ceiling(0.15)
9. `narrow_spread_boost_sell` → ceiling(0.15)
10. `fast_fill_defense.offset_boost_sell: 2.5` → ceiling(0.15)
11. `loss_control.loss_boost_offset_mult: 1.3` → ceiling(0.15)
12. `inventory_skewing` (sell) → floor(0.30) 支配、ceiling(0.15)

---

## §2 解決策: Side-specific Ceiling (案 A)

### §2.1 設計方針

5案を検討した結果、**案 A (Side-specific Ceiling)** を採用:

| 案 | 変更規模 | リスク | 即時実行 | 採否 |
|---|---|---|---|---|
| A: サイド別 ceiling | 小 | 低 | ○ | **✅ 採用** |
| B: Floor-aware ceiling | 極小 | 低 | ○ | △ floor に支配される |
| C: Per-stage max | 大 | 中 | × | △ 12+ 新パラメータ |
| D: Pipeline 統合 | 大 | 高 | × | × リスク過大 |
| E: 段階実行 (A→D) | 段階 | 低→高 | ○ | △ Phase 1 = 案 A |

### §2.2 パラメータ設計

```yaml
# 320# C-1 根本対策
offset_ceiling_ratio: 0.15          # 共通デフォルト (後方互換)
offset_ceiling_ratio_buy: 0.15      # buy: 据置 (53.3% 通過で十分)
offset_ceiling_ratio_sell: 0.50     # sell: 0.15→0.50 (pipeline 復活)
trending_sell_offset_boost_factor: 1.5  # 4.0→1.5 (pipeline + executor 協調)
```

### §2.3 新旧フロー比較

**旧フロー (全 sell 一律)**:
```
base(0.18) → floor(0.30) → stages(~0.30) → ceiling(0.15) → executor ×4.0 = 0.60
                                              ↑ 分化なし
```

**新フロー (パイプライン復活)**:
```
[trending_up sell]
base(0.18) → floor(0.30) → regime_boost(×1.8=0.54) → ceil(0.50) → executor ×1.5 = 0.75

[ranging sell]  
base(0.18) → floor(0.30) → regime_discount(×0.9=0.27) → stages(~0.30) → executor ×1.0 = 0.30

[trending_down sell (順方向)]
base(0.18) → floor(0.30) → regime_discount(×0.7=0.21) → stages(~0.27) → executor ×1.0 = 0.27
```

**比較表**:

| 条件 | 旧 effective | 新 effective | 変化 |
|---|---|---|---|
| trending_up sell | 0.60 | 0.75 | +25% 防御強化 |
| ranging sell | 0.15 | 0.30 | +100% 防御強化 |
| trending_down sell | 0.15 | 0.27 | +80% (順方向は攻撃的) |
| buy (ceiling hit) | 0.15 | 0.15 | 変化なし |
| buy (通過) | 0.05-0.14 | 0.05-0.14 | 変化なし |

---

## §3 実装詳細

### §3.1 fill_config.py

```python
# 320# C-1: サイド別 ceiling
offset_ceiling_ratio_buy: float | None = None   # None=共通値使用
offset_ceiling_ratio_sell: float | None = None   # None=共通値使用
```

### §3.2 maker_price.py (compute())

```python
# 320# C-1: サイド別 ceiling
_ceil = cfg.offset_ceiling_ratio
if side == "buy" and cfg.offset_ceiling_ratio_buy is not None:
    _ceil = cfg.offset_ceiling_ratio_buy
elif side == "sell" and cfg.offset_ceiling_ratio_sell is not None:
    _ceil = cfg.offset_ceiling_ratio_sell
```

### §3.3 fill_test.yaml

- `offset_ceiling_ratio_buy: 0.15` (据置)
- `offset_ceiling_ratio_sell: 0.50` (floor(0.30) + regime boost(×1.8=0.54) を 0.50 で受容)
- `trending_sell_offset_boost_factor: 4.0 → 1.5` (pipeline 復活に伴い縮小)
- `sell_guard.offset_floor: 0.30` コメント更新 (矛盾解消を反映)

### §3.4 テスト

| テスト | 内容 |
|---|---|
| `test_config_has_side_specific_ceiling_fields` | フィールド存在 + デフォルト None |
| `test_sell_ceiling_higher_than_buy` | sell ceiling > sell floor 整合性 |
| `test_sell_ceiling_none_falls_back_to_common` | None 時の共通値フォールバック |
| `test_yaml_trending_sell_soft` | YAML 値 1.5 確認 (196# テスト更新) |
| `test_yaml_trending_boost_is_3_0` | YAML 値 1.5 確認 (197# テスト更新) |
| compute line count | 290→295 (3行増分) |
| run_single_cycle line count | 725→740 (319# S-3 分) |

### §3.5 全テスト結果

```
3968 passed, 33 skipped, 0 failed (exit=1: coverage threshold only)
```

---

## §4 dcc3064 暫定評価 (n=21, insufficient)

### §4.1 データ範囲

| 項目 | 値 |
|---|---|
| 稼働コード | `dcc3064a8` (310# 改善) |
| 稼働期間 | 2026-03-06 23:44 〜 2026-03-07 04:14 (4.5h) |
| 全レコード | 111 |
| filled | 21 (sell=11, buy=10) |
| fill rate | sell 36.7% (11/30), buy 12.5% (10/80) |
| fills/h | 4.7 |

**n=21 は A/B 判定に不足 (min=50)**。以下は方向性参考値。

### §4.2 PnL 分布

**Sell** (n=11, mean=-0.962):
```
-13.28, -10.59, -5.21, -1.81, -0.81, -0.37, -0.36, +3.01, +3.19, +6.75, +8.90
```
AS 率: 64% (7/11)、p10=-10.59

**Buy** (n=10, mean=-0.841):
```
-12.63, -6.91, -5.31, -2.29, -1.83, +1.51, +1.55, +2.69, +3.80, +11.01
```
AS 率: 50% (5/10)、p10=-12.63

### §4.3 全データとの比較

| 指標 | 全データ sell (n=1292) | dcc3064 sell (n=11) | 全 buy (n=1315) | dcc3064 buy (n=10) |
|---|---|---|---|---|
| mean PnL | -0.350 | -0.962 | -0.305 | -0.841 |
| AS 率 | 55% | 64% | 52% | 50% |

dcc3064 は全データ平均より悪い方向だが、n=11/10 では統計的有意性なし。
tail イベント (-13.28, -12.63) の影響が大きい。

### §4.4 Regime 分布 (dcc3064 filled)

| Regime | sell | buy |
|---|---|---|
| ranging | 4 | 8 |
| trending_down | 4 | 1 |
| trending_up | 3 | 1 |
| unknown/none | 0 | 0 |

unknown/none が 0 件 — 318# F5 修正 (Passive MM バイパス) が有効に機能し、unknown regime での fills を回避。

### §4.5 Spread / AS 分解

| Side | spread_capture | realized_pnl | AS cost | efficiency |
|---|---|---|---|---|
| SELL | -1.200 bps | -0.962 bps | -0.239 bps | 0.801 |
| BUY | -1.318 bps | -0.841 bps | -0.477 bps | 0.638 |

spread_capture が両サイドで大きく負 (-1.2, -1.3 bps)。全データ (-0.50, -0.49) より悪化。検出レイテンシバイアスが amplify された可能性。319# S-3 `mid_at_order` によるバイアス分離が重要。

### §4.6 C-1 影響の直接証拠 (dcc3064 offset_stages)

dcc3064 sell 全 11 件の pipeline 出力:

```
全件: base=0.18 → floor(as_shift=0.30) → (stages~0.30) → ceiling=0.15 → final=0.30
```

**sell 側は 11/11 (100%) が ceiling ヒット**。パイプライン全ステージが一律 0.15 にクランプ — C-1 が dcc3064 稼働中も完全に再現。

### §4.7 評価結論

1. **n 不足**: A/B 判定不可能。50+ fills まで蓄積必要。
2. **none/unknown 排除**: 318# F5 修正の効果確認 (0 件)。
3. **C-1 再現**: sell offset 分化なし (100% ceiling hit)。
4. **spread_capture 劣化**: 全データ比で 2.4× 悪化。要 mid_at_order での追跡。
5. **次回評価**: bot を 320# コードに更新後、50+ fills 蓄積で再評価。

---

## §5 sell floor 削減の検討

### §5.1 Floor (0.30) の影響

sell ceiling を 0.50 に引上げても、floor(0.30) がパイプライン初期段階で offset を 0.30 にクランプするため:
- regime_discount (trending_down ×0.7 → 0.21) は一時的に 0.30 未満になるが、後段ステージで復帰
- floor 以下の offset は表現不可能
- sell 側の「攻撃的な offset」（0.10-0.30 レンジ）は出せない

### §5.2 Floor 削減の損益

- **削減メリット**: 順方向 sell (trending_down) で offset を 0.20-0.25 に下げ、fill rate 向上
- **削減リスク**: 逆方向 sell (trending_up) で offset が 0.18-0.30 に下がりやすく、AS 被害拡大
- **現時点の判断**: ceiling 分離の効果を先に確認。floor 削減は Phase 2 で検討。

---

## §6 残課題

| ID | 優先度 | 内容 | 状態 |
|---|---|---|---|
| C-1-P2 | P2 | floor 0.30 削減の可否検討 (§5) | データ蓄積後 |
| S-2 | P1 | Sell Hour Boost post-310# 効果検証 | n 不足 → 蓄積待ち |
| S-6 | P2 | buy ev_offset 逆効果調査 | 316# §6 残 |
| M-3 | P2 | unknown_regime_max_consecutive: 10→3-5 | 319# §2 残 |
| M-5 | P3 | fixed_offset_bps: 2.0→1.0 | 319# §2 残 |
| bot 更新 | P0 | 320# を bot に hot-swap | 即時 |

---

## §7 関連ドキュメント

| # | 関係 |
|---|---|
| 306 | E1 — ceiling 導入 |
| 246 | sell floor 0.20→0.30 |
| 316 | §6 先行施策提案 S-1〜S-7 |
| 319 | C-1 発見 + S-1 暫定対策 (boost 4.0) |
| 317 | dcc3064 観測実験報告 |
| 318 | F5 none regime 修正 |
