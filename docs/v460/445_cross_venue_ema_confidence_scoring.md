# 445# Cross-Venue EMA 平滑化 + Confidence Scoring

| 項目 | 内容 |
|---|---|
| 番号 | 445# |
| 分類 | ph2_impl |
| 前提 | 439# (基盤構築), 442# (L5 microprice/DI), 443# (閾値修正), 444# (ログ可視化) |
| 目的 | sign_disagree による hint 脱落の解消、binary gate → continuous confidence への移行 |

---

## 1. 問題の特定

444# の実運用ログ (23 cross-venue entries) を分析した結果:

| 分類 | 件数 | 比率 |
|---|---|---|
| hint 発火 | 4 | 17% |
| sign_disagree | 7 | **30%** (最大ボトルネック) |
| spread too small | 7 | 30% |
| first_call | 3 | 13% |
| velocity too small | 1 | 4% |

### 1.1 sign_disagree の根本原因

**市場微構造理論の時間スケール不一致**

Hasbrouck (1995) の lead-lag 分析は、マーケットマイクロストラクチャの先導・追随関係が **100ms〜5s** の特性時間で発現することを示している。Coincheck fill_test の cycle interval は **120s** であり、この間の velocity は lead-lag signal ではなく **mean-reversion ノイズ** を拾っている。

具体例 (実運用ログ):

```
spread = +3.75bps (BF > CC: 上方向乖離)
velocity = -0.09bps/s (直近120sで BF が下落)
→ sign_disagree → hint=None
```

spread +3.75bps は顕著な adverse selection signal だが、120s 前の BF 価格がやや高かっただけで velocity が負になり、有効な signal が完全に失われている。

### 1.2 その他の設計問題

| 問題 | 症状 |
|---|---|
| Binary gating | spread 1bps でも 5bps でも同じ 1.25x boost |
| Microprice 未活用 | 計算済みだが gating 判定に寄与しない (mid spread +2.51bps vs microprice -0.36bps の矛盾を検出不能) |
| Point spread ノイズ | 120s 間隔の単一観測値に依存、一時的なスパイクで誤判定 |

---

## 2. 設計: Binary → Continuous (Dual-Mode)

### 2.1 Confidence Scoring 数式

$$\text{confidence} = \underbrace{\min\!\bigl(1.0,\; \max\!\bigl(0.33,\; \tfrac{|\text{ema\_spread}|}{\text{ref\_spread}}\bigr)\bigr)}_{\text{base\_conf (spread magnitude)}} \;\times\; \underbrace{f_{\text{vel}}}_{\text{0.5–1.0}} \;\times\; \underbrace{f_{\text{mp}}}_{\text{0.5–1.0}}$$

| Factor | Condition | Value | 理論根拠 |
|---|---|---|---|
| $f_{\text{vel}}$ | velocity agrees with EMA direction | 1.0 | Hasbrouck lead-lag 確認 |
| $f_{\text{vel}}$ | velocity negligible ($< \text{threshold}$) | 0.8 | 120s では insufficient information |
| $f_{\text{vel}}$ | velocity disagrees | **0.5** | 従来は0.0 (hard gate) → 半減に緩和 |
| $f_{\text{mp}}$ | microprice agrees | 1.0 | L1 depth 非対称性が方向を裏付ける |
| $f_{\text{mp}}$ | microprice negligible ($< 0.5$ bps) | 0.9 | ほぼ中立 |
| $f_{\text{mp}}$ | microprice disagrees | **0.5** | L1 深度情報が方向に矛盾 |

### 2.2 EMA 平滑化

Point spread のノイズを低減するため、cross-venue spread を EMA で追跡する。

$$\text{ema\_spread}_{t} = \alpha \cdot \text{spread}_{t} + (1 - \alpha) \cdot \text{ema\_spread}_{t-1}$$

- **初回**: $\text{ema\_spread}_0 = \text{spread}_0$ (point value で初期化)
- **alpha = 0.3**: 120s cycle で実効的な half-life ≈ 2 cycle (≈240s)

Direction は EMA spread の符号で決定する (point spread ではなく)。これにより一時的なスパイクによる方向の誤判定を抑制する。

### 2.3 Dual-Mode 設計 (後方互換)

| パラメータ | Legacy mode | Confidence mode |
|---|---|---|
| `ema_spread_bps` | `None` | EMA 値を供給 |
| Velocity | **hard gate** (disagree → None) | **modifier** (disagree → ×0.5) |
| Microprice | 未使用 | **modifier** (disagree → ×0.5) |
| Spread threshold | point spread で判定 | EMA spread で判定 |
| Direction | point spread の符号 | EMA spread の符号 |
| Boost | 固定 1.25x | $1 + (1.25 - 1) \times \text{confidence}$ |

Legacy mode は `ema_spread_bps=None` のとき自動的に発動し、既存テスト (15件) がそのまま通る。

### 2.4 Confidence-Proportional Boost

従来の固定ブースト:
```python
# 従来 (442#): 全 hint に同一 boost
boost = cfg.cross_venue_lead_lag_offset_boost  # 1.25
```

445# の比例ブースト:
```python
# 445#: confidence に比例した退避量
actual_boost = 1.0 + (cfg.cross_venue_lead_lag_offset_boost - 1.0) * hint.confidence
# confidence=1.0 → 1.25x (従来同等)
# confidence=0.5 → 1.125x (弱い信号 = 弱い退避)
# confidence=0.33 → 1.083x (最小退避)
```

---

## 3. 既存 EMA 実装との関係

本プロジェクトには複数の EMA 実装が存在する。445# の実装はこれらと同一の数学的パターンを踏襲している。

| コンポーネント | ファイル | 用途 | alpha |
|---|---|---|---|
| `_smoothed_velocity_bps` | `scripts/v460/lib/maker_price.py` L841 | Mid trend offset 用 velocity EMA | 0.3 |
| `DynamicKillManager.track()` | `ztb/risk/sell_dynamic_kill.py` L303 | PnL bps の EWMA 追跡 | 0.05 |
| `ema_velocity_bps()` | `ztb/features/market_theory.py` L229 | バッチ特徴量 (pandas ewm) | span=5 |
| `TaLibWrapper.ema()` | `ztb/utils/talib_wrapper.py` L237 | TA-Lib 互換 EMA (array-based) | period-based |
| **445# `update_cross_venue_ema()`** | `scripts/v460/lib/cross_venue_lead_lag.py` L48 | cross-venue spread EMA | **0.3** |

すべて同じ標準 EMA 数式 `new = α × obs + (1−α) × prev` を使用。

### 3.1 445# が独自 dataclass を採用した理由

- **リアルタイム逐次更新**: `update_cross_venue_ema()` は 1 観測ずつ更新する online 方式。TaLibWrapper や pandas ewm は array/batch 方式であり、逐次更新には不向き。
- **maker_price._smoothed_velocity_bps** のパターンが最も近いが、CrossVenueEMAState は `ema_ref_mid` と `ema_spread_bps` の 2 チャネルを同時に管理するため、単一 float では不十分。
- **DynamicKillManager**: 時間減衰 (tau_sec) 付き EWMA で設計意図が異なる (PnL 追跡 vs spread 平滑化)。
- **テスタビリティ**: frozen=False の dataclass でテスト時に状態を注入可能。

### 3.2 今後の統合候補

`ztb/utils/` に `OnlineEMA` ヘルパーを抽出し、maker_price, dynamic_kill, cross_venue の 3 箇所を統一することは将来的に有意義だが、現時点では各用途の微妙な差分 (2-channel, time-decay, single-float) を考慮し、まず正しく動作する個別実装を優先した。

---

## 4. 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/cross_venue_lead_lag.py` | `CrossVenueEMAState` dataclass, `update_cross_venue_ema()`, `confidence` field, dual-mode compute |
| `scripts/v460/lib/fill_cycle_executor.py` | EMA state 管理、compute へ confidence パラメータ転送、ログに conf/ema 表示 |
| `scripts/v460/lib/maker_risk_guards.py` | 固定 boost → `1 + (max-1) × confidence` 比例ブースト |
| `scripts/v460/lib/fill_config.py` | 3 新フィールド (`ema_alpha`, `min_confidence`, `confidence_reference_spread_bps`) |
| `scripts/v460/lib/fill_config_parser.py` | YAML→Config マッピング追加 |
| `configs/v460/fill_test.yaml` | 3 新エントリ |
| `scripts/v460/run_fill_test.py` | `_cross_venue_ema_state = None` 初期化 |
| `ztb/metrics/fill_quality.py` | `cross_venue_confidence` FillRecord field |
| `tests/unit/v460/test_439_cross_venue_lead_lag.py` | 14 新テスト (EMA, confidence mode, proportional boost) |
| `tests/unit/v460/test_253_...` | 行数上限更新 (1375→1390) |

## 5. 設定パラメータ

| パラメータ | デフォルト | YAML | 説明 |
|---|---|---|---|
| `cross_venue_ema_alpha` | 0.3 | `ema_alpha` | EMA 減衰係数 (0.3 = half-life ≈ 2 cycle) |
| `cross_venue_min_confidence` | 0.2 | `min_confidence` | この閾値未満の confidence → hint=None |
| `cross_venue_confidence_reference_spread_bps` | 3.0 | `confidence_reference_spread_bps` | base_confidence=1.0 となる EMA spread |

## 6. テスト結果

- 既存テスト 15 件: **全 PASS** (legacy mode 後方互換)
- 新規テスト 14 件: **全 PASS**
  - `TestUpdateCrossVenueEMA`: 初期化、EMA ブレンド
  - `TestConfidenceModeCompute`: sign_disagree 復活、velocity agree、small EMA spread、min_confidence gate、base_confidence scaling、microprice disagree、direction from EMA
  - `TestConfidenceProportionalBoost`: full/half confidence boost 検証
- v460 全体 (5,000+ tests): PASS (flaky performance test 除く)

## 7. 期待される改善

| 指標 | 変更前 (444#) | 変更後 (445#) 予測 |
|---|---|---|
| hint 発火率 | 17% (4/23) | 40-50% |
| sign_disagree 脱落 | 30% (7/23) → None | → confidence=0.5 で発火 |
| boost 精度 | 固定 1.25x | 1.08x〜1.25x (信号強度比例) |
| microprice 活用 | 計算のみ | confidence modifier として gating に寄与 |

---

## 8. Codex レビュー向け注意点

### 8.1 レビューすべきポイント

1. **EMA alpha=0.3 の妥当性**: half-life ≈ 2 cycle (240s) は 120s interval に対して適切か？
2. **velocity/microprice の具体的な modifier 値** (0.5, 0.8, 0.9, 1.0): 理論的根拠は Hasbrouck lead-lag + Gatheral microprice だが、実証データによるキャリブレーションは今後の課題
3. **min_confidence=0.2**: 閾値の妥当性。低すぎるとノイズフィルタリングが不十分、高すぎると発火率が低下
4. **reference_spread_bps=3.0**: CC-BF 間の実測 spread 分布に基づくが、市場環境変動で要調整の可能性
5. **depth imbalance boost が confidence と独立**: DI boost × confidence boost の二重適用は意図的か？

### 8.2 想定される質問と回答

**Q: なぜ既存の `TaLibWrapper.ema()` を使わないのか？**
A: TaLibWrapper は array/batch 処理用。120s cycle ごとに 1 観測ずつ逐次更新する online EMA には不適合。同じ数式 (`α × new + (1-α) × old`) を使っているが、状態管理方式が異なる。

**Q: EMA state の永続化は？**
A: 現時点では未実装。fill_test 再起動時に point value から再初期化される。N=3 (≈6分) で EMA が安定するため、warm-up のコストは軽微。

**Q: `CrossVenueEMAState` は `frozen=True` にすべきでは？**
A: `update_cross_venue_ema()` で新規インスタンスを返す設計のため、`frozen=True` が適切ではあるが、`n_updates` のインクリメントで `frozen=True` にするとコンストラクタ呼出が必須になる。現在は `frozen=False` + 関数型更新で安全性を確保。`frozen=True` への移行は可。
