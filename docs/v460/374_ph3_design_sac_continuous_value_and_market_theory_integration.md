# 374# ph3 設計: SAC 連続値活用 + 市場理論システム統合

| 項目 | 値 |
|---|---|
| 文書番号 | 374# |
| フェーズ | ph3 (G2-train → live injection) |
| 前提 | 365# (Sidecar 設計), 366# (M1-M5), 371# (配線), 373# (安全性監査) |
| 作成 | Copilot |
| ステータス | **v3.0 — 375#/376# レビュー反映済 (軌道修正版)** |

---

## §1 エグゼクティブサマリ

### 問題提起

365# で設計した SAC Sidecar は全 P1-P8 ブロッカーを解消済みだが、**連続値の利点が構造的に失われている**。

| 現状 | 問題 |
|---|---|
| SAC 出力: `[-1.0, +1.0]` 連続値 | ✅ 豊富な勾配情報 |
| `classify_bias()` で ±0.3 閾値離散化 | ❌ BUY/NEUTRAL/SELL の 3値に劣化 |
| `compute_sidecar_offset_bps()` で固定 ±0.3bps ブースト | ❌ bias=0.31 と bias=1.0 が同一出力 |
| 市場理論 M1-M5 は SAC と独立動作 | ❌ 相互フィードバックなし |

**核心**: SAC Actor が学習した「確信度を含む連続方向性情報」のうち、67% の情報 (magnitude) が classify_bias() で捨てられている。

### 結論: 4段階戦略

> **⚠️ 375#/376# レビュー判定 (v3.0)**: Phase 3.1 のみ条件付き GO。3.2 は HOLD (データパリティ未達)。3.3/3.4 は NO-GO。

| Phase | 名称 | 方針 | リスク | 工数 | 375#/376# 判定 |
|---|---|---|---|---|---|
| **3.1** | Proportional Boost | magnitude × boost (離散化廃止) | ★☆☆ | 2-4h | **条件付き GO** — `max_boost_bps=0.15` (最大 0.2)。0.1/0.15/0.2 ladder 検証 |
| **3.2** | Regime-Aware Observation | M2 Bayesian 事後確率を SAC obs に注入 | ★★☆ | 4-8h | **HOLD** — parquet に M2-M5 列が無い。build_features.py はゼロ行。データパリティ復元が先決 |
| **3.3** | Parameter Modulation | SAC が AS γ / offset scale を連続制御 | ★★★ | 1-2d | **NO-GO** — α 未証明段階で action space 拡張は時期尚早 |
| **3.4** | Closed-Loop Reward | fill_records 実績 PnL で SAC 報酬を補正 | ★★★ | 2-3d | **NO-GO** — 因果混乱 (causal confusion) リスクが高すぎる |

---

## §2 現状アーキテクチャの精密分析

### §2.1 データフロー全体像

```
SAC Actor ──→ action[0] ∈ [-1,+1] (continuous)
                  │
                  ▼
         ┌── classify_bias(bias, ±0.3) ──┐
         │   BUY_BIAS / NEUTRAL / SELL   │   ← ★ 情報ロス: magnitude 消失
         └───────────┬───────────────────┘
                     ▼
         compute_sidecar_offset_bps()
         │  direction × boost_bps × confidence
         │  = ±0.3 bps (定数)                  ← ★ 情報ロス: 一律 flat boost
         ▼
  cycle_gate_aggregator._apply_sidecar_offset()
         │  result.sidecar_offset_bps = offset
         ▼
  orchestrator_mid_cycle → run_single_cycle(sidecar_offset_bps=…)
         │
         ▼
  fill_cycle_executor L730:
         │  _sidecar_delta = bps / 10000 * price
         │  order_price ± _sidecar_delta            ← 最終注入ポイント
         ▼
  Zaif API 発注
```

### §2.2 Offset Pipeline 全段階 (maker_price.compute_effective_offset)

```
① base_offset_ratio (config: spread_offset_ratio)
② inv_skew (inventory skewing: imbalance × factor)
③ AS reservation_shift (Avellaneda-Stoikov γσ²τ)    ← M4 GLFT k 連動
④ regime_boosts (regime → offset multiplier)
⑤ spread_adaptive (spread/mid ratio)
⑥ kyle_lambda (Kyle 1985 λ × lot / mid)             ← M2 相関
⑦ amihud_illiq (Amihud 2002 非流動性)
⑧ volatility_guard (VG: velocity/VPIN boost)         ← M5 VPIN 連動
⑨ imbalance_risk (OB imbalance → offset widen)
⑩ buy_as_guard (microprice 急落 → buy offset widen)  ← M1 microprice 連動
```

Post-pipeline (fill_cycle_executor):
```
⑪ skip_gate ev_offset (EV-weighted 調整)
⑫ VG sell supplement
⑬ alert_mode offset_mult
⑭ sidecar_offset_bps ← ★ SAC はここ (最後の 1 段)
⑮ lot 計算 (regime_lot × confidence × alert × recovery × DD × cooldown)
```

### §2.3 情報ロスの定量評価

| bias 値 | 直感的意味 | 現在の出力 | 理想的出力 (v3.0 修正) |
|---|---|---|---|
| +1.00 | 極めて強い上昇確信 | +0.3 bps | +0.15 bps (全力だが安全範囲内) |
| +0.70 | 強い上昇確信 | +0.3 bps | +0.10 bps |
| +0.31 | 弱い上昇確信 | +0.3 bps | +0.035 bps (ほぼ中立) |
| +0.29 | 弱い上昇確信 | **0.0 bps** | +0.032 bps (微小ブースト) |
| +0.10 | ノイズ水準 | 0.0 bps | 0.0 bps (dead zone) |
| -0.50 | 中程度の下落確信 | -0.3 bps | -0.067 bps |

**closed-form**: 現在の方式は $f(b, s) = \text{sgn}(b) \cdot H(|b| - 0.3) \cdot c \cdot \delta_{s}$ (ステップ関数)。
理想的には $f(b, s) = g(b) \cdot c \cdot \delta_{s}$ で $g$ は連続かつ $g(0) = 0$ の活性化関数。

---

## §3 提案A: Proportional Boost (Phase 3.1)

### §3.1 設計

連続値 bias の **magnitude 全体をブーストに反映**する最小変更。

```python
def compute_sidecar_offset_bps_v2(
    bias: float,
    side: str,
    max_boost_bps: float = 0.15,   # 最大ブースト (375#/376# 修正: 3.0→0.15)
    dead_zone: float = 0.10,       # ノイズ抑制 dead zone
    confidence: float = 1.0,
    shaping: str = "linear",       # "linear" | "quadratic" | "sigmoid"
) -> float:
    """bias の magnitude に比例したオフセット調整 (bps).

    375#: max_boost_bps=3.0 は median spread 2,282 JPY を超える 3,274 JPY 相当 → 自殺的。
    376#: 0.15 bps を絶対上限とし、ladder 検証 (0.1 / 0.15 / 0.2) で最適値を探索。
    """
    abs_bias = abs(bias)
    if abs_bias <= dead_zone:
        return 0.0

    # Normalize: [dead_zone, 1.0] → [0.0, 1.0]
    t = (abs_bias - dead_zone) / (1.0 - dead_zone)

    if shaping == "quadratic":
        magnitude = t * t
    elif shaping == "sigmoid":
        import math
        magnitude = math.tanh(3.0 * t)  # tanh(3)≈0.995, smooth saturation
    else:  # linear
        magnitude = t

    effective_boost = max_boost_bps * magnitude * min(max(confidence, 0.0), 1.0)

    # 方向性: BUY bias → buy 攻撃的, sell 保守的
    if bias > 0:
        return effective_boost if side == "buy" else -effective_boost
    else:
        return effective_boost if side == "sell" else -effective_boost
```

### §3.2 各 Shaping 関数の特性

| 関数 | 特性 | 適用場面 |
|---|---|---|
| **Linear** | magnitude に正比例。シンプル。勾配一定 | 初期検証に最適。挙動が予測しやすい |
| **Quadratic** | 低 bias で抑制的、高 bias で加速 | confidence が低い序盤。慎重な市場 |
| **Sigmoid (tanh)** | 低 bias で急速立上、高 bias で飽和 | 明確な方向性がある市場。飽和で暴走防止 |

**推奨**: 初期は **Linear** で開始。配信実績を見て quadratic/sigmoid に切替可能な config 化。

### §3.3 パラメータ設計根拠

| パラメータ | 値 | 根拠 |
|---|---|---|
| `max_boost_bps` | **0.15** | 375#: 3.0bps は median spread 超過で自殺的。376#: 0.15 を絶対上限。ladder 検証 0.1/0.15/0.2 |
| `dead_zone` | 0.10 | SAC `ent_coef=auto` で温度調整中のノイズ振幅 ≈ ±0.05-0.15。0.10 で十分なノイズカット |
| `confidence` | OOS ROI ベース (372# 動的計算) | 既存の confidence 計算をそのまま活用 |

### §3.4 既存コードへの影響

| 変更ファイル | 変更内容 | 影響範囲 |
|---|---|---|
| `sidecar_types.py` | `compute_sidecar_offset_bps` → v2 置換 | 呼出元: `cycle_gate_aggregator.py` のみ |
| `fill_config.py` | `sidecar_max_boost_bps`, `sidecar_dead_zone`, `sidecar_shaping` 追加 | config 追加のみ |
| `fill_config_parser.py` | YAML `sidecar:` セクション追加 | 解析追加 |
| `config_hot_reload.py` | hot-reload 可能キーに追加 | live 調整可能に |

**API 変更**: `compute_sidecar_offset_bps()` の signature 変更。ただし呼出元は 1箇所のみ (`cycle_gate_aggregator._apply_sidecar_offset`)。

### §3.5 自己批判 (v3.0 更新)

1. ~~**3.0 bps で十分か?**~~ → **375#/376# 指摘により完全否定**。3.0bps は base_offset の 28.7 倍に達し、median spread (2,282 JPY) を超える。config 可変にしても初期値が 0.15 を超えてはならない。
2. **Dead zone 0.10 は最適か?** — SAC の exploration noise 水準次第。ent_coef=auto で温度が下がり deterministic に近づくと dead zone が不要になる可能性。→ config 化で対応。
3. **Proportional boost だけで Alpha は出るか?** — 375# §6 が推奨する **maker uplift 指標** (fill_rate, post_fill_30s_pnl, adverse_selected, postonly_crossing_skip) で検証すべき。return correlation は不適切。
4. **v3.0 追加: 環境の選択** — 376# は `FastIntradayEnvV456` (88-dim, GroupedFeatureScaler, Ichimoku) を発掘。LiteTradingEnv (12-dim) との二者択一ではなく、Phase 3.1 は LiteTradingEnv で実施し、データパリティ復元後に FastIntradayEnvV456 を検討する段階的アプローチを採用。

---

## §4 提案B: Regime-Aware Observation (Phase 3.2)

### §4.1 動機

現在の SAC 特徴量は OHLCV ベース 12 次元 (price_velocity, micro_trend, ...) のみ。**市場理論システムの出力が SAC の観測に含まれていない**。

M2 Bayesian Regime (371# 配線済) は 4状態事後確率を毎ステップ更新しているが、この情報は SAC に渡されていない。

### §4.2 設計

```
SAC Observation (現在: 12 dim)
  + M2 Bayesian regime posterior: 4 dim  (trending_up, trending_down, ranging, volatile)
  + M3 σ-cluster id: 1 dim              (LOW=0, MID=1, HIGH=2, EXTREME=3、正規化)
  + M5 VPIN: 1 dim                       (current VPIN [0, 1])
  + M4 GLFT fill_prob: 1 dim             (current fill probability [0, 1])
  ─────────────────────────────────────
  = 19 dim (or 22 dim with action_masks)
```

### §4.3 実装ポイント

**LiteTradingEnv 拡張**:
```python
class LiteEnvConfig:
    # 既存
    feature_columns: list[str] | None = None
    # 追加
    market_theory_features: bool = False  # True: M2-M5 cols を obs に追加
```

**方法**: `feature_columns` に M2-M5 由来の列を OHLCV parquet に併載する (前処理パイプラインで追加)。SAC は追加列もそのまま観測する。

- Bayesian regime は `BayesianRegimeFilter.update(ret)` で得られる posterior を列として前計算
- σ-cluster は `VolatilityRegimeClassifier.classify(vol_ratio)` 結果を列化
- VPIN は `compute_vpin_volume_sync()` で列化
- fill_prob は GLFT `A·exp(-k·δ)` で列化

### §4.4 期待効果

SAC は「現在がどの regime にいるか」を observation から学習でき、**regime 遷移時に先行して bias を変化**させられる。

| 現在 | 改善後 |
|---|---|
| SAC は OHLCV からレジームを暗黙的に推定 | M2 の事後確率を直接観測 → 明示的な regime 情報 |
| VPIN の毒性を知らずに方向性を予測 | VPIN を観測 → toxic flow 回を bias に反映 |
| fill probability を知らずにオフセットを示唆 | fill_prob を観測 → 実現可能性を考慮した bias |

### §4.5 自己批判

1. **次元の呪い**: 12→19 dim は SB3 SAC (MLP 256×256) で十分処理可能。ただし sample efficiency は低下する → buffer_size を 100K→200K に増加検討。
2. **情報の冗長性**: Bayesian posterior と OHLCV derived features (micro_trend, momentum) には相関がある。PCA 等で次元削減すべきか → 初期は raw で投入し、後から冗長特徴量を pruning。
3. **タイミング**: M2-M5 は 371# で配線済みだが **default disabled (enabled=false)**。SAC obs に追加する前に M2-M5 を enabled にして live 運用実績を確認すべきか → SAC 訓練は offline なので、訓練時に M2-M5 を計算すれば live 有効化不要。

---

## §5 提案C: Parameter Modulation (Phase 3.3)

### §5.1 動機 — 「方向性予測」から「パラメータ制御」への転換

現在の SAC は「価格が上がるか下がるか」を予測し、その予測を offset bps に変 する。しかし、fill_test の offset pipeline は既に 10+ 段の市場理論ベース調整を持っている。SAC が学ぶべきは **「既存パラメータのどれをどの程度調整するか」** ではないか。

### §5.2 Multi-Dimensional Action Space 設計

```python
# LiteTradingEnv v2 (Phase 3.3)
action_space = Box([-1, -1, -1], [+1, +1, +1], shape=(3,))

action[0]: directional_bias      ∈ [-1, +1]  # 方向性 (既存)
action[1]: offset_scale_factor   ∈ [-1, +1]  # offset 攻撃性/保守性
action[2]: lot_confidence_factor ∈ [-1, +1]  # ロットサイジング確信度
```

| Dimension | 意味 | fill_test での注入先 |
|---|---|---|
| `action[0]` | 方向性 bias | `sidecar_offset_bps` (Phase 3.1 の proportional boost) |
| `action[1]` | offset scale | AS γ の動的スケーリング: γ_eff = γ_base × (1 + action[1] × scale_range) |
| `action[2]` | lot confidence | `lot × (0.5 + 0.5 × (1 + action[2]))` → [0.5×lot, 1.0×lot] |

### §5.3 市場理論パラメータへの直接接続

```
action[1] (offset_scale) の注入先:

① AS γ (risk aversion):
   γ_eff = γ_base × clip(1 + action[1] × 0.5, 0.5, 2.0)
   → action[1] = -1: γ 半減 (攻撃的オフセット)
   → action[1] = +1: γ 倍増 (保守的オフセット)

② Kyle λ scaling:
   kyle_lambda_impact_mult_eff = base_mult × clip(1 + action[1] × 0.3, 0.7, 1.3)

③ VG boost ceiling:
   volatility_guard_offset_boost_factor_eff = base_factor × clip(1 + action[1] × 0.3, 0.7, 1.5)
```

### §5.4 報酬関数の修正

LiteTradingEnv の報酬を「パラメータ制御」に適合させる:

```python
# 現在: step_pnl = position × price_change - cost
# Phase 3.3: 市場メイキングの指標を追加

reward = (
    step_pnl                          # ベース PnL
    - risk_penalty * position**2      # 在庫リスクペナルティ (AS 理論)
    - spread_cost * abs(position_delta)  # スプレッドコスト
    + fill_reward * was_filled         # Fill 報酬 (仮想)
)
```

### §5.5 自己批判

1. **3次元 action space は overshoot リスク**: 1 dim でも SAC の学習が不安定な可能性がある中、3 dim はさらに困難。**Phase 3.1 で 1 dim の安定性を確認してから拡張すべき。**
2. **env と fill_test の乖離拡大**: SAC は LiteTradingEnv で「AS γ を制御」して学習するが、live では offset pipeline の 10 段のうち 1 段のみに注入。学習 env と live の乖離が大きい。
3. **「パラメータを学習で制御」は本当に有利か?** — 手動チューニングで M1-M5 パラメータは既に調整済み。SAC が上回れるかは疑問。ただし **regime 遷移時のパラメータ切替** は手動では困難であり、SAC の優位性はここにある。
4. **Zaif の流動性**: BTC/JPY market on Zaif は流動性が薄く、理論が想定する「十分な流動性のある市場」とは異なる。AS や Kyle の理論パラメータの最適値が学術的知見と大きく乖離する可能性。

---

## §6 提案D: Closed-Loop Reward (Phase 3.4)

### §6.1 動機 — 環境と現実の乖離

SAC は LiteTradingEnv で「OHLCV 価格変動 × ポジション」の PnL で学習するが、live では:

| 要素 | Env | Live |
|---|---|---|
| 執行 | 即座に target_position に到達 | 指値 → キューイング → fill/timeout |
| コスト | `position_delta × price × cost_rate` | スプレッド + slippage + AS risk |
| Fill rate | 100% | ~60-80% (offset/regime 依存) |
| 遅延 | なし | OB fetch, SkipGate eval (~0.5-1s) |

### §6.2 設計: fill_records → SAC Reward Bridge

```
┌──────────────────────┐     ┌───────────────────────────────────┐
│ Live fill_test       │     │ SAC Retrain (sac_retrain_scheduler)│
│                      │     │                                   │
│  cycle → fill/cancel │     │  OHLCV → env → SAC.learn()       │
│      ↓               │     │                  ↑                │
│  fill_record.jsonl   │────→│  Reward Bridge:                   │
│  (pnl, AS, fill rate)│     │    "at this timestamp,            │
│                      │     │     with this bias,               │
│                      │     │     actual PnL was X,             │
│                      │     │     fill_rate was Y"              │
│                      │     │                                   │
└──────────────────────┘     └───────────────────────────────────┘
```

### §6.3 報酬の 2 段構成

```python
# SAC 報酬 = env_pnl + α × hindsight_correction

# hindsight_correction:
#   この timestamp の sidecar_bias で live は実際にどの程度の PnL を得たか？
#   fill_records から retrospective に correction term を算出

def compute_hindsight_reward(
    timestamp: str,
    sidecar_bias: float,
    fill_records: list[dict],
    lookback_minutes: int = 10,
) -> float:
    """fill_records から事後的な報酬補正を計算."""
    relevant = [r for r in fill_records
                if within_window(r["timestamp"], timestamp, lookback_minutes)]
    if not relevant:
        return 0.0
    actual_pnl = sum(r.get("pnl", 0.0) for r in relevant)
    fill_rate = sum(1 for r in relevant if r.get("filled")) / len(relevant)
    # bias と 実績 PnL の alignment を報酬に
    alignment = sidecar_bias * actual_pnl  # 正なら方向一致
    return alignment * fill_rate  # fill rate で重み付け
```

### §6.4 自己批判

1. **タイミングの非整合**: fill_records は cycle 単位 (5-30s)、SAC step は 1 分。alignment の粒度が合わない → fill_records を 1 分バケットに集約。
2. **因果関係の混乱**: sidecar_bias が PnL に影響を与えている可能性があるため、「バイアスが正しかったから PnL が良かった」のか「バイアスの影響で PnL が変わった」のか分離困難。→ confidence-weighted で sidecar の影響を除外する試みが必要。
3. **データ量**: fill_records は cycle 1 回あたり 1 レコード。1 日で ~1000-2000 レコード。7 日 rolling で ~7K-14K サンプル。SAC の replay buffer 100K に対して十分か → env PnL をベースにし、hindsight は**補正項**にとどめる (α << 1.0)。

---

## §7 市場理論×SAC 統合マトリクス

### §7.1 接続設計

| M# | 市場理論 | → SAC Obs (Phase 3.2) | → SAC Action (Phase 3.3) | → SAC Reward (Phase 3.4) |
|---|---|---|---|---|
| M1 | Microprice L5 | microprice_bias_bps (1 dim) | — | — |
| M2 | Bayesian Regime | posterior: 4 dim (P(trending↑), P(trending↓), P(ranging), P(volatile)) | — | — |
| M3 | σ-Clustering | vol_cluster: 1 dim (0-3 normalized) | — | — |
| M4 | GLFT Fill Prob | fill_prob: 1 dim (A·e^{-kδ}) | action[1] → AS γ / GLFT k | fill_rate alignment |
| M5 | Volume-Sync VPIN | vpin_vol_sync: 1 dim | — | toxic flow avoidance |

### §7.2 情報フローの双方向化

**現在** (一方向):
```
M1-M5 → offset pipeline → order price
SAC → bias → offset pipeline → order price  (独立)
```

**Phase 3.2 後** (M→SAC):
```
M1-M5 output → SAC observation → SAC bias → offset pipeline
```

**Phase 3.3 後** (M←→SAC 双方向):
```
M1-M5 output → SAC observation → SAC action → M パラメータ変調 → offset pipeline
                                                       ↑
                                                  fill_records (Phase 3.4)
```

---

## §8 多角的評価

### §8.1 収益インパクト推定

| Phase | メカニズム | 期待効果 | 最悪ケース |
|---|---|---|---|
| 3.1 | magnitude 比例 → 強確信時の攻撃性↑ | +0.5-2.0 bps/fill | Dead zone で中立回帰 → ±0 |
| 3.2 | regime 認知 → 遷移時の先行調整 | DD -5-15% | 冗長特徴量 → SAC 性能不変 |
| 3.3 | AS γ 動的制御 → regime 適応 offset | +1-5 bps/fill (最大) | γ 暴走 → clamp で ±50% 範囲内 |
| 3.4 | 環境-現実ギャップ圧縮 | fill_rate +5-10% | α が小さすぎ → 影響なし |

### §8.2 リスクマトリクス

| リスク | 深刻度 | Phase | 緩和策 |
|---|---|---|---|
| Proportional boost で AS 悪化 | ★★☆ | 3.1 | `max_boost_bps=0.15` (375#/376# 修正) + hot-reload で即時調整。ladder 0.1/0.15/0.2 |
| Obs 次元増加 → SAC 学習効率低下 | ★★☆ | 3.2 | buffer_size 増加 + 段階的特徴量追加 |
| Multi-dim action → 学習不安定 | ★★★ | 3.3 | Phase 3.1 安定性確認後。clamp で値域制限 |
| Hindsight reward → 因果混乱 | ★★★ | 3.4 | α ≪ 1.0 で補正項扱い。ablation study |
| SAC 全体の方向性予測精度不足 | ★★★★ | 全体 | Phase 3.1 の dead zone + confidence weighting で worst-case = 中立 |

### §8.3 東洋的/西洋的思考のバランス

**西洋的バイアス (注意)**:
- 「連続値 → 比例変換」は制御理論/最適化から来る考え方。市場の非線形性に対して線形な応答関数が最適とは限らない
- Sidecar パターン自体が Kubernetes 文脈。金融では co-location (直接統合) が主流

**東洋的視座**:
- 「無為」(何もしないのが最善) の可能性。SAC の方向性予測が無価値なら、**連続値を活用してもゼロの連続値が返るだけ**。まず Phase 3.1 で SAC の基本的 Alpha を検証することが最優先
- 「陰陽」の均衡。攻撃的 offset (低 dead zone, 高 max_boost) と保守的 offset (高 dead zone, 低 max_boost) の**両極を config で表現可能**にしておく
- 「漸進主義」。Phase 3.1→3.2→3.3→3.4 の段階的アプローチは、各ステップで**エビデンスを蓄積してから次に進む**

### §8.4 HFT/マーケットメイキング文脈での批判的検討

1. **SAC の推論レイテンシ**: 現設計では retrain_scheduler が「数分に 1 回シグナル更新」。しかし市場は秒単位で動く。**方向性バイアスが stale になるリスク**。
   - 対策: TTL を 7800s (372# fix) → 600s に短縮し、推論頻度を上げる (毎サイクル推論は CPU 的に非現実的だが、30s-60s 周期なら可能)
   - 長期的: ONNX Runtime 推論で MLP Actor を 1ms 以下で推論可能に

2. **市場微細構造の非線形性**: 板が極端に薄い Zaif BTC/JPY では、0.15 bps でも fill probability に影響する (指数減衰 A·e^{-kδ} の k が大きい)。~~Phase 3.1 の max_boost_bps は過大かもしれない~~ → **375#/376# により 0.15 bps に是正済み**。
   - 対策: ladder テスト (0.1/0.15/0.2 bps) で最適値を探索

3. **相関構造**: price_velocity と directional_bias の相関が高ければ、**SAC は micro_trend のラグ付きコピーを出力しているだけ**の可能性。
   - 検証: 訓練後に `corr(directional_bias, price_velocity_t+k)` を k=1,2,...,10 で計算。高相関なら SAC は**独自の Alpha を持たない**

---

## §9 実装優先順位

| 優先度 | タスク | 工数 | 前提 | 根拠 |
|---|---|---|---|---|
| **P0** | Phase 3.1: Proportional boost 実装 | 2-4h | なし | 最小変更、即時効果、rollback 容易 |
| **P0.5** | SAC α 検証実験 | 1-2h | P0 完了 | SAC が無価値なら Phase 3.2+ は無意味。先に検証 |
| **P1** | Phase 3.2: M2-M5 obs 注入 (データ前処理) | 4-8h | P0.5 でα確認 | 特徴量追加のみ。env 変更最小 |
| **P2** | Phase 3.3: Multi-dim action (設計のみ) | 4h | P1 | 設計文書 + プロトタイプ |
| **P3** | Phase 3.3: Multi-dim action (実装) | 1-2d | P2 | env + sidecar_types + pipeline 変更 |
| **P4** | Phase 3.4: fill_records bridge (設計) | 4h | P1 | データ解析 + reward 設計 |
| **P5** | Phase 3.4: fill_records bridge (実装) | 2-3d | P4 | 最大効果だが最大リスク |

### クリティカルパス

```
P0 (proportional boost)
  ↓ 検証
P0.5 (α検証: SAC bias と future PnL の相関)
  ↓ α > 0 確認
P1 (M2-M5 obs) ────→ P2 (multi-dim設計) ────→ P3 (multi-dim実装)
                                                    ↓
                      P4 (hindsight設計) ────→ P5 (hindsight実装)
```

**α ≈ 0 の場合**: Phase 3.2+ の前に**訓練 env 自体の改善**が必要。gamma 調整、報酬関数改修、データ拡張 (domain randomization for price data)。

---

## §10 Phase 3.1 最速実装計画

Phase 3.1 は次のセッションで着手可能。変更量は ~100 行以内。

### §10.1 変更ファイル一覧

| # | ファイル | 変更 |
|---|---|---|
| 1 | `scripts/v460/lib/sidecar_types.py` | `compute_sidecar_offset_bps_v2()` 追加。旧関数は `_legacy` suffix を付けて保持 |
| 2 | `scripts/v460/lib/fill_config.py` | `sidecar_max_boost_bps` (**0.15**), `sidecar_dead_zone`, `sidecar_shaping` 追加 |
| 3 | `scripts/v460/lib/fill_config_parser.py` | YAML `sidecar:` セクション解析 |
| 4 | `scripts/v460/lib/config_hot_reload.py` | hot-reload 可能キーに追加 |
| 5 | `scripts/v460/lib/cycle_gate_aggregator.py` | `_apply_sidecar_offset` で `v2` を呼出。config 参照 |
| 6 | `configs/v460/fill_test.yaml` | `sidecar:` セクション追加 |
| 7 | `tests/unit/v460/test_374_proportional_boost.py` | 新規テスト (20-30 tests) |

### §10.2 テスト計画

| テストカテゴリ | 内容 | 件数 |
|---|---|---|
| Dead zone | bias ∈ [-0.10, +0.10] → 0.0 bps | 5 |
| Linear proportional | bias=0.5 → 50% × max_boost | 4 |
| Quadratic/Sigmoid | 各 shaping 関数の特性 | 6 |
| Direction × side | buy/sell × BUY_BIAS/SELL_BIAS の 4 通り | 4 |
| Confidence weighting | confidence = 0.5 → half boost | 3 |
| Edge cases | bias=±1.0, bias=0.0, confidence=0.0 | 4 |
| backward compat | 旧関数の signature 維持確認 | 2 |

---

## §11 SAC α 検証実験 (P0.5) の具体手順

Phase 3.2+ に進む前に、**SAC が実際に有用な方向性情報を出力しているか**を検証する。

### §11.1 方法

```python
# 検証スクリプト: analysis/sac_alpha_validation.py
#
# 1. 訓練済み SAC モデル (sac_sidecar.zip) をロード
# 2. OOS データ (直近 1 日分) で全 step の directional_bias を推論
# 3. 将来 k-step リターンとの相関を計算
#
# r_k = corr(bias_t, return_{t+k})  for k = 1, 2, ..., 10
#
# 判定:
#   r_1 > 0.05 かつ p < 0.01 → "弱い Alpha あり" → Phase 3.2 に進む
#   r_1 > 0.10               → "有意な Alpha"    → Phase 3.3 を加速
#   r_1 ≈ 0                  → "Alpha なし"      → 訓練改善が先
```

### §11.2 追加検証

- **Regime 条件付き相関**: trending 時と ranging 時で r_k が異なるか → regime-conditional bias の有用性
- **Turnover 分析**: bias の符号反転頻度 → 高すぎればノイズ、低すぎれば stale
- **Drawdown 回避力**: bias が大きく負の直前にボラティリティが高かった時、bias が保守的方向にシフトしていたか

---

---

## §12 既存 SAC 実装インベントリと再利用分析

### §12.1 既存実装マップ (27+ ファイル)

v395–v460 にわたる SAC 関連コードは 27 ファイル以上に分散している。以下は Phase 3.1–3.4 設計で **直接再利用すべき** 資産の棚卸しである。

#### A. コアアルゴリズム層

| ファイル | 行数 | 主要機能 | Phase 3.x 再利用 |
|---|---|---|---|
| `ztb/training/algorithms/sac/sac_algorithm.py` | 1224 | SACAlgorithm: SB3 SAC ラッパー。**4 種のネットワークアーキテクチャ** (MLP/LSTM/Transformer/Efficient)、転移学習、モデル圧縮 (量子化/プルーニング/蒸留)、SHAP 説明可能性、replay buffer 永続化 | **3.2**: LSTM/Transformer で temporal 特徴量の活用検討<br>**3.1**: replay buffer save/load で warm-start (365# P1 と連携)<br>**全体**: explain_decision() で α 検証の補助 |
| `ztb/training/unified_trainer/algorithms/sac_trainer.py` | 1973 | SACTrainer: VecNormalize、分散訓練、CheckpointManager (lz4 async)、OptimizerFeatureTracker、**market_regime_adaptation** (enabled=false) | **3.2**: market_regime_adaptation ブロックが既存。Phase 3.2 の regime obs 注入と連携可能<br>**3.3**: reward_settings 検証ロジック再利用 |
| `ztb/training/trainers/sac_trainer.py` | 748 | SACAlgorithmTrainer (deprecated): SACMetricsCallback (Critic/Actor Loss + Entropy のリアルタイム CSV/TB 記録) | **P0.5**: α 検証時のメトリクス記録に SACMetricsCallback を再利用 |
| `ztb/training/sac_trainer.py` | 401 | SACTrainer ファサード (deprecated): **RegimeAdaptiveTrainerMixin**、curriculum training (multi-stage)、hyperparameter adaptation | **3.2**: RegimeAdaptiveTrainerMixin の `apply_hyperparameter_adaptation()` は Phase 3.3 のパラメータ変調の雛形<br>**3.2**: `run_curriculum_training()` は段階的訓練に直接転用可能 |

#### B. コールバック・ユーティリティ層

| ファイル | 行数 | 主要機能 | Phase 3.x 再利用 |
|---|---|---|---|
| `ztb/training/callbacks/reinforcement/sac/sac_callbacks.py` | 512 | **SACTemperatureScheduler** (適応的エントロピー温度制御)、**SACValueFunctionMonitor** (Q/V 安定性監視)、**SACTargetNetworkUpdater** (適応的 τ 制御)、**SACExplorationAnalyzer** (探索状態診断) | **3.1–3.3**: 全 callback を sac_retrain_scheduler の retrain_once() に組込み可能。特に TemperatureScheduler は v430 の HOLD/SELL 偏重問題の再発防止に直結 |
| `ztb/training/utils/sac_utils.py` | 482 | SACUtilities: config 整合性チェック、コマンド実行、メンテナンスタスク | **全体**: `check_config_consistency()` を g2_sac_train.yaml 検証に転用 |

#### C. 特徴量・分析層

| ファイル | 行数 | 主要機能 | Phase 3.x 再利用 |
|---|---|---|---|
| `ztb/features/models/sac/sac_v427_feature_engineering.py` | 2699 | SACv427FeatureEngineer: 品質フィルタリング (NaN率/分散/ゼロ率/外れ値率/相関)、複数 window_sizes、feature_set 切替 (default/high_quality/minimal/full) | **3.2**: Phase 3.2 で M2-M5 obs を追加する際、SACv427FeatureEngineer の品質フィルタリングロジックを再利用して冗長特徴量を自動 pruning |
| `ztb/analysis/core/model/sac_analyzer.py` | 443 | SACAnalyzer: アクション分布分析、バイアス検出・補正、パフォーマンス評価 | **P0.5**: α 検証で directional_bias の分布偏り検出に活用 |
| `ztb/analysis/sac/sac_types.py` | 206 | SACHyperparameters, EnvironmentConfig, RewardSettings 等の TypedDict 定義 | **全体**: 型安全な config 参照に利用 |
| `ztb/analysis/core/training/sac_v423_analyzer.py` | 128 | v423 訓練結果分析 (メトリクス抽出、アクション分布、速度計算) | **P0.5**: training_report.json の標準的な解析パターンとして参照 |

#### D. SAC–Trading 統合層

| ファイル | 行数 | 主要機能 | Phase 3.x 再利用 |
|---|---|---|---|
| `ztb/trading/strategies/action_signal_guide/components/sac_integration.py` | 651 | **SACSignalValidator**: SAC 決定と ActionSignal の相関検証 (action alignment, confidence correlation, timing alignment, market alignment)。**SACDecisionIntegrator**: signal-SAC 統合意思決定 | **3.3**: SACSignalValidator の correlation_score 計算ロジックは、Phase 3.3 で multi-dim action の品質評価に応用可能<br>**3.4**: PerformanceDecisionRecord は fill_records → reward bridge のデータ構造の参考 |
| `scripts/v460/ml/sac_retrain_scheduler.py` | 883 | warm-start retrain、OOS gate、_update_sidecar_signal() | **3.1**: Phase 3.1 の注入点。proportional boost 関数を呼ぶのはここ |

#### E. 実験・設定層

| バージョン | 主要ファイル | 教訓 |
|---|---|---|
| v427 | `experiments/sac_v427_training_executor.py` (対称アクション変換 ±0.3333)、`features/sac_v427_feature_engineering.py` (150+ 次元) | 高次元特徴量 → **過学習リスク**。Phase 3.2 で obs 拡張時は 19 dim に抑制 |
| v430 | `analysis/sac/sac_v430_analysis_report.py` (報酬関数設計ミス分析)、`sac_v430_reward_fix_experiment.py` | **sell_penalty / buy_bonus が負 → 全停止**。Phase 3.4 reward 設計の反面教師 |
| v434 | docs/SAC_v434_2_DEVELOPMENT_PLAN.md | **過剰取引 92.4% → 取引コスト 10 倍強化**。Phase 3.3 の lot_confidence_factor 設計参考 |
| v444 | docs/SAC_v444_DEBUG_GUIDE.md (balance_penalty_scale=1000 → 毎 step -488) | **ペナルティスケールの暴走**。Phase 3.4 で回避すべきパターン |
| v446 | docs/SAC_V446_5M_STATUS_ANALYSIS.md (reward=-80, BUY 偏重 52.8%) | マルチタイムフレーム学習の困難さ |
| v459 | 015# (SAC ≈ Random, p=0.64。88 dim 訓練 → 5 dim 推論の次元不一致) | **致命的教訓**: 訓練と推論の env が一致しなければ全く無意味 |

### §12.2 ネットワークアーキテクチャ選定: MLP vs LSTM vs Transformer

SACAlgorithm は 4 種のネットワークをサポート済み:

| アーキテクチャ | 推論速度 | 時系列活用 | Sidecar 適性 | 検討 |
|---|---|---|---|---|
| **MLP** (現行) | ◎ <1ms | ✗ (各 step 独立) | ◎ | 現行 LiteTradingEnv は 1-step obs。MLP で十分 |
| **LSTM** | ○ ~5ms | ◎ (sequence) | △ | sequence_length=10 なら直近 10 分の時系列を考慮。ただし LiteTradingEnv を frame-stacking 対応にする改修が Phase 3.2 と連動 |
| **Transformer** | △ ~20ms | ◎ (attention) | ✗ | 推論レイテンシが大。Sidecar は 30-60s 周期なので許容範囲だが、MLP/LSTM で十分な性能が出るなら不要 |
| **Efficient** | ○ ~3ms | ○ (depthwise conv + linformer) | ○ | MLP と Transformer の中間。探索的に試す価値あり |

**推奨**: Phase 3.1–3.2 は **MLP で開始**。Phase 3.2 で obs に M2-M5 時系列を追加した後、**LSTM に切替えて比較実験**。SACAlgorithm の `network_type` config 変更のみで切替可能 (コード変更不要)。

### §12.3 既存コールバックの Phase 3.x 組込み計画

sac_callbacks.py の 4 つのコールバックは訓練品質の監視に不可欠:

| Callback | 目的 | retrain_scheduler 統合案 |
|---|---|---|
| `SACTemperatureScheduler` | エントロピー温度の適応制御。reward_trend が正 & entropy 低 → 温度↑ (探索促進) | retrain_once() の callback_list に追加。**v430 の HOLD 偏重問題 (temperature 未制御) の再発防止** |
| `SACValueFunctionMonitor` | Q/V の安定性監視、収束判定、発散検出 | divergence_threshold 超過時に retrain を中断。OOS gate 前の品質フィルタ |
| `SACTargetNetworkUpdater` | τ の適応制御 (Q-loss 安定時 τ↓, 不安定時 τ↑) | retrain_once() の callback に組込み。**warm-start 時の target network 不安定性を緩和** |
| `SACExplorationAnalyzer` | アクション分布の偏り検出 | HOLD/SELL 偏重の早期検出 → 訓練中断 + アラート |

### §12.4 RegimeAdaptiveTrainerMixin の活用

`ztb/training/sac_trainer.py` の ファサード SACTrainer は deprecated だが、**RegimeAdaptiveTrainerMixin** のインターフェースは Phase 3.2–3.3 に直接活用可能:

```python
# RegimeAdaptiveTrainerMixin.apply_hyperparameter_adaptation(adapted_params)
# Phase 3.3 で以下のように活用:
adapted_params = {
    "as_gamma": gamma_base * clip(1 + action[1] * 0.5, 0.5, 2.0),
    "kyle_lambda_mult": base_mult * clip(1 + action[1] * 0.3, 0.7, 1.3),
}
trainer.apply_hyperparameter_adaptation(adapted_params)
```

ただし、現在のファサードは UnifiedTrainer に委譲しており、`update_hyperparameters()` の実装は UnifiedTrainer 側で保証されていない。**Phase 3.3 着手時に UnifiedTrainer のインターフェースを確認する必要あり**。

---

## §13 vXXX シリーズからの構造的教訓

### §13.1 報酬関数の歴史的失敗パターン (再発防止)

vXXX シリーズの **繰り返し発生した問題** を Phase 3.4 (Closed-Loop Reward) の設計制約として組込む:

| 問題パターン | 発生版 | 根本原因 | 復旧策 | Phase 3.4 への反映 |
|---|---|---|---|---|
| **全停止 (0%取引)** | v430, v439 | sell_penalty=-0.35, buy_bonus=-0.43 → 全アクションにペナルティ | penalty→bonus 転換 | hindsight reward の `alignment` 項が全負にならないよう `abs(alignment)` 下限を設ける |
| **HOLD 90%+** | v397h, v395 | 閾値 0.10 高すぎ + hold_penalty 不在 + entropy 不足 | 閾値 0.05 + 即時ボーナス + entropy↑ | Phase 3.1 の dead_zone=0.10 も同様の閾値。検証後に 0.05 に下げる柔軟性を config 化 |
| **SELL 66%+** | v444 | balance_penalty_scale=1000 → 毎 step -488 | scale 大幅削減 | Phase 3.4 で risk_penalty に上限値を設ける (max_penalty_per_step) |
| **過剰取引 92%+** | v434 | 取引コスト 0.005% → 手数料が安すぎ | コスト 10 倍 | Phase 3.3 の lot_confidence_factor に**最小ロット制約**を追加済み (373# F1/F2) |
| **In-sample 過学習** | v459 (p=0.64) | train=eval 同一 env | Train/Test 時系列ホールドアウト | 363# A3 で修正済み。Phase 3.2+ の訓練では OOS 検証を必須とする G2 gate |
| **次元不一致** | v459→v460 | 88 dim 訓練 → 5 dim 推論 | LiteTradingEnv 統一 | Phase 3.2 で obs 拡張時は **LiteTradingEnv と sac_retrain_scheduler の obs 構成を一致させること**が絶対条件 |

### §13.2 ハイパーパラメータの収斂

v395–v460 で 10 以上のバージョンにわたりハイパラを探索した結果、以下の値で収斂:

| パラメータ | 収斂値 | 探索範囲 | 根拠 |
|---|---|---|---|
| learning_rate | **3e-4** | 1e-5–1e-3 | 全バージョンで一貫。SB3 default と一致 |
| batch_size | **256** | 128–512 | 256 が安定。v446 で 512 を試したが改善なし |
| gamma | **0.99** (通常) / **0.80** (短期 v459) | 0.80–0.99 | Sidecar 推論は長期パターン → 0.99 推奨 |
| tau | **0.005** | 0.001–0.01 | 全バージョンで一貫 |
| ent_coef | **"auto"** | 0.01–1.0 / "auto" | "auto" + SACTemperatureScheduler の組合せが最も安定 |
| buffer_size | **50K–100K** | 50K–1M | v446 の 1M は冗長。50K→100K が Phase 3.2 obs 拡張時の推奨 |

### §13.3 カリキュラム学習の知見

v397i で導入された 3 段階カリキュラム:
```
forced_balance → balanced_transition → pnl_focused
```

v431 で 5 段階に拡張:
```
stage1: 簡単な市場 (低ボラ) → stage2: 通常市場 → stage3: 高ボラ市場
→ stage4: regime 混合 → stage5: full complexity
```

**Phase 3.2 への示唆**: LiteTradingEnv に M2-M5 obs を追加する際、**最初は M2 regime のみ (1 dim) → 次に全 7 dim** と段階的に追加するカリキュラムを SACTrainer.run_curriculum_training() で実現可能。

---

## §14 改善点の特定と追加提案

### §14.1 Phase 3.1 への改善

| # | 改善案 | 根拠 | 工数 |
|---|---|---|---|
| **I1** | `compute_sidecar_offset_bps_v2()` に **ONNX Runtime** 推論を前提とした型注釈を追加 | 将来的に MLP Actor を ONNX 化すれば推論 <1ms。現行 PyTorch predict は ~10ms | SACAlgorithm に ONNX export メソッド追加: 2h |
| **I2** | proportional boost の **hot-reload** を config_hot_reload.py に追加し、live で max_boost_bps/dead_zone を変更可能にする | v448 emergency_fix.json のような緊急対応パターン | §10.1 に含む |
| **I3** | **SACTemperatureScheduler** を retrain_once() に統合 | v430 HOLD 偏重の再発防止 | callback 追加: 1h |

### §14.2 Phase 3.2 への改善

| # | 改善案 | 根拠 | 工数 |
|---|---|---|---|
| **I4** | SACv427FeatureEngineer の **品質フィルタリング** を M2-M5 追加特徴量にも適用 | NaN 率 > 10% や分散 < 閾値の特徴量を自動除外 | SACv427FeatureEngineer 呼出追加: 2h |
| **I5** | **VecFrameStack** (SACTrainer で既に import 済み) を活用し、LSTM なしで直近 N-step の obs を MLP に渡す | MLP のまま時系列情報を活用。LSTM 切替前の中間アプローチ | LiteTradingEnv + VecFrameStack 連携: 4h |
| **I6** | LiteTradingEnv に **domain randomization** を追加 (price noise, spread jitter) | v459 の in-sample 過学習問題 (362#) への対策 | env 拡張: 4h |

### §14.3 Phase 3.3 への改善

| # | 改善案 | 根拠 | 工数 |
|---|---|---|---|
| **I7** | multi-dim action の Phase 3.3 を **2段階に分割**: action[0]+action[1] → action[0]+action[1]+action[2] | 1 dim → 3 dim の跳躍は学習不安定。2 dim で安定性確認後に 3 dim | スケジュール変更のみ |
| **I8** | SACAlgorithm の **モデル圧縮 (quantization)** を Phase 3.3 後に適用 | 推論速度向上 → 推論頻度増加 (30s→10s) → stale bias 問題の緩和 | SACAlgorithm.create_model(compression_enabled=True): 2h |

### §14.4 Phase 3.4 への改善

| # | 改善案 | 根拠 | 工数 |
|---|---|---|---|
| **I9** | SACSignalValidator の **correlation_score 計算** を hindsight reward の alignment quality 評価に転用 | "SAC bias と fill_record 実績の alignment" = "SAC decision と signal の alignment" の構造が同一 | sac_integration.py 参照: 2h |
| **I10** | **SACMetricsCallback** (CSV 記録) を fill_records bridge のデータロギングに活用 | 既存の CSV 書込みインフラを再利用 | callback 拡張: 1h |

### §14.5 構造的改善 (全 Phase 横断)

| # | 改善案 | 根拠 | 工数 |
|---|---|---|---|
| **I11** | sac_dependency_graph.py を **CI に組込み** (import 整合性自動チェック) | 015# で発覚した 6系統並存問題の再発防止 | CI yaml 追加: 1h |
| **I12** | **zt/analysis/sac/sac_types.py** の TypedDict を sidecar_types.py の型定義と統合 | SACHyperparameters, EnvironmentConfig 等が sac_types.py と fill_config.py で二重定義 | 型統合: 3h |
| **I13** | SACAlgorithm の **explain_decision()** を α 検証 (P0.5) で活用 | SHAP で "どの obs 特徴量が bias に寄与しているか" を可視化 → SAC が price_velocity のコピーなのか独自 Alpha なのかを判別 | explain_decision() 呼出 + 可視化: 3h |

---

## §15 更新された実装優先順位

v1.0 の §9 を既存実装の再利用と改善点で拡充:

| 優先度 | タスク | 工数 | 再利用する既存資産 |
|---|---|---|---|
| **P0** | Phase 3.1: Proportional boost 実装 | 2-4h | `sidecar_types.py`, `config_hot_reload.py` |
| **P0+** | **I3: SACTemperatureScheduler を retrain_once() に統合** | 1h | `sac_callbacks.py` |
| **P0.5** | SAC α 検証実験 + **I13: SHAP 説明** | 3-5h | `sac_analyzer.py`, `SACAlgorithm.explain_decision()` |
| **P1** | Phase 3.2: M2-M5 obs 注入 + **I4: 品質フィルタ** + **I5: VecFrameStack** | 8-12h | `SACv427FeatureEngineer`, `VecFrameStack`, `market_regime_adaptation` ブロック |
| **P1.5** | **LSTM 比較実験**: `network_type: lstm` に切替えて Phase 3.2 obs で訓練比較 | 4h | `SACAlgorithm._resolve_policy_kwargs()` (変更なし、config のみ) |
| **P2** | Phase 3.3 設計 + **I7: 2段階分割** | 4h | `RegimeAdaptiveTrainerMixin`, `SACSignalValidator` |
| **P3** | Phase 3.3 実装 (2-dim → 3-dim) | 1-2d | `SACAlgorithm.create_model()`, `run_curriculum_training()` |
| **P3.5** | **I8: モデル圧縮** (quantization) | 2h | `SACAlgorithm._apply_model_compression()` |
| **P4** | Phase 3.4 設計 + **I9: correlation_score 転用** | 4h | `SACSignalValidator`, `SACMetricsCallback` |
| **P5** | Phase 3.4 実装 | 2-3d | `sac_integration.py`, `fill_records` |

### 更新達成基準

| Phase | Go 条件 | 検証手段 |
|---|---|---|
| 3.1 | `corr(proportional_boost, future_return) > corr(flat_boost, future_return)` | backtest 比較 |
| P0.5 | `r_1(bias, return_{t+1}) > 0.05, p < 0.01` | OOS 相関分析 + SHAP |
| 3.2 | `IC_19dim > IC_12dim` | 4-seed OOS 比較訓練 |
| 3.3 | `PnL(2-dim) > PnL(1-dim)` | backtest + paper trade |
| 3.4 | `fill_rate(hindsight) > fill_rate(baseline)` | 7日 rolling 比較 |

---

## §16 375#/376# レビュー反映 — 軌道修正 (v3.0)

### §16.1 レビュー判定サマリ

| レビュー | レビュア | 核心指摘 |
|---|---|---|
| 375# | Codex | `max_boost_bps=3.0` は自殺的 (median spread 超過)。3.2-3.4 は NO-GO/HOLD。SAC は live に存在すらしていない |
| 376# | Gemini 3.1 Pro | build_features.py に M2-M5 がゼロ行。FastIntradayEnvV456 (88-dim) が埋もれている。max_boost_bps は 0.15 絶対上限 |

### §16.2 是正した設計値

| 項目 | v2.0 (旧) | v3.0 (修正後) | 根拠 |
|---|---|---|---|
| `max_boost_bps` | 3.0 | **0.15** (ladder: 0.1 / 0.15 / 0.2) | 375# §2.1: 3.0×base=28.7倍→median spread超過。376# §3: 0.15 絶対上限 |
| Phase 3.2 判定 | GO (4-8h) | **HOLD** | 375# §2.2: parquet に M2-M5 列なし。376# §2.1: build_features.py にゼロ行 |
| Phase 3.3 判定 | GO (1-2d) | **NO-GO** | 375# §3: α 未証明段階で action space 拡張は時期尚早 |
| Phase 3.4 判定 | GO (2-3d) | **NO-GO** | 375# §2.4: causal confusion。自己強化バイアスで学習が崩壊する |
| α 検証指標 | `corr(bias, return_{t+k})` | **maker uplift** | 375# §2.3: fill_rate, post_fill_30s_pnl, adverse_selected, postonly_crossing_skip |
| 対象 Env | LiteTradingEnv v2 | LiteTradingEnv (3.1) → **FastIntradayEnvV456 復活** (3.2+) | 376# §2.2: 88-dim, GroupedFeatureScaler, Ichimoku, regime 13D one-hot |

### §16.3 発見されたブロッカー

| ブロッカー | 分類 | 影響範囲 | 解消要件 |
|---|---|---|---|
| **build_features.py に M2-M5 がゼロ行** | CRITICAL | Phase 3.2 全体 | build_features.py にM2 (Bayesian regime posterior), M3 (σ-cluster), M4 (GLFT fill_prob), M5 (VPIN) の計算ロジックを追加 |
| **FastIntradayEnvV456 が放置** | HIGH | Phase 3.2+ の env 選択 | 1062 行, 88-dim env を再評価し、LiteTradingEnv との機能差分を整理 |
| **v454 MarketRegimeClassifier 未活用** | HIGH | 訓練データの regime annotation | `ztb.analysis.regime.market_regime_classifier` を offline 訓練パイプラインに統合 |
| **SAC の live 存在が未確認** | HIGH | Phase 3.1 の前提 | sidecar_signal.json 常時更新、retrain 履歴、fill_records sidecar_offset_bps non-null を確認 |
| **training_metrics.total_timesteps 誤ラベル** | LOW | 運用診断 | ✅ 377# で修正済み (trade_count にリネーム) |

### §16.4 Phase 3.2 に必要なデータパリティ復元ロードマップ

376# が提起した「3 ステップロードマップ」を Phase 3.2 の前提条件として明文化:

```
Step 1: build_features.py に M2-M5 計算ロジック追加
        ├── M2: BayesianRegimeFilter.update(ret) → 4-dim posterior
        ├── M3: VolatilityRegimeClassifier.classify() → 1-dim
        ├── M4: GLFT fill_prob A·exp(-k·δ) → 1-dim
        └── M5: compute_vpin_volume_sync() → 1-dim
        → parquet に 7 列追加

Step 2: FastIntradayEnvV456 の復活評価
        ├── 88-dim obs → 要求カラムの整合確認
        ├── GroupedFeatureScaler の互換性
        └── LiteTradingEnv 12-dim からの段階的移行パス

Step 3: Phase 3.1 Proportional Boost (max_boost_bps=0.15)
        ├── LiteTradingEnv で先行実施 (3.2 データパリティ不要)
        ├── maker uplift 指標で検証
        └── ladder: 0.1 / 0.15 / 0.2 bps
```

### §16.5 α 検証の修正 (§11 補遺)

375# §2.3 の指摘により、§11 の α 検証を以下のように修正する:

**旧 (v2.0)**: `r_k = corr(bias, return_{t+k})` → 方向性の return correlation
**新 (v3.0)**: maker uplift metrics による検証

```python
# 検証指標 (375# §6 准拠):
# 1. fill_rate: sidecar 有/無での fill 率差分
# 2. post_fill_30s_pnl: fill 後 30 秒の PnL (bps)
# 3. adverse_selected: AS 率差分
# 4. postonly_crossing_skip: post-only 違反回避率
#
# 判定:
#   fill_rate(sidecar) > fill_rate(baseline) → "maker quality 改善"
#   post_fill_30s_pnl(sidecar) > baseline   → "有効な α"
#   adverse_selected(sidecar) <= baseline    → "AS 悪化なし"
```

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-16 | 1.0 | 初版 (4 Phase 設計 + 市場理論統合マトリクス + α 検証) |
| 2026-03-17 | 2.0 | §12-§15 追加: 既存 SAC 実装インベントリ (27 ファイル)、vXXX 歴史的教訓 (v395-v460)、改善点 13 件 (I1-I13)、ネットワークアーキテクチャ選定、コールバック統合計画、更新された優先順位 |
| 2026-03-18 | 3.0 | §16 追加 + 全体修正: **375#/376# レビュー反映**。max_boost_bps 3.0→0.15、Phase 3.2 HOLD / 3.3-3.4 NO-GO、α検証指標を maker uplift に変更、FastIntradayEnvV456 復活パス追記、データパリティブロッカー明文化、training_metrics 誤ラベル修正 |
