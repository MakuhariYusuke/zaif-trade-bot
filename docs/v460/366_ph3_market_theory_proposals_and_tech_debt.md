# 366# 市場理論システム提案 + 技術課題 + 計算高速化

| 項目 | 値 |
|---|---|
| 文書番号 | 366# |
| フェーズ | ph3 (G2-train / fill_test 改善) |
| 前提 | 365#, 364# C2/G3, 306#, 266#, 054# |
| 作成 | Copilot 038 |
| ステータス | **ACTIVE** |

---

## §1 エグゼクティブサマリ

本文書は 365# SAC Sidecar 設計を補完し、fill_test の**収益向上に直結する市場理論システム**と**技術負債**を体系化する。各提案には既存実装の再利用可否、計算高速化の方針、Codex 委任の適否を明記する。

### 要点

| # | 提案 | 既存基盤 | 追加工数 | 収益インパクト | Codex 委任 | 状態 |
|---|---|---|---|---|---|---|
| M1 | Microprice L1→L5 拡張 | ★★★★★ (90%) | 2-4h | **高** — AS 低減直結 | ✅ 候補 | ✅ **完了** `265e768de` |
| M2 | Bayesian Regime Detection | ★★★☆☆ (50%) | 1-2d | **高** — 遷移予測 | ⚠ 一部可 | ✅ **完了** (Phase A) |
| M3 | σ-Clustering (Vol Regime) | ★★★★☆ (70%) | 4-8h | **中** — offset 適応精度↑ | ✅ 候補 | ✅ **完了** `1b8d8e55f` |
| M4 | GLFT Fill Probability Model | ★★★★☆ (80%) | 4-8h | **高** — fill 最適化 | ⚠ 設計依存 | ✅ **完了** `59a30956c` |
| M5 | Volume-Sync VPIN | ★★★☆☆ (60%) | 4-8h | **中** — toxicity 精度↑ | ✅ 候補 | ✅ **完了** `cf28375b7` |
| T1-T10 | 技術負債 | — | 各1-4h | 安定性向上 | ✅ 最適 | T1✅ T4✅ T5✅ T9✅ |

---

## §2 市場理論提案 (M1-M5)

---

### §2.1 M1: Microprice L1→L5 拡張 (Gatheral 2018)

#### 理論

Gatheral (2018) microprice は板の**多段階**の数量不均衡を反映した公正価格推定:

$$\mu = \frac{\sum_{k=1}^{K} w_k \cdot (P_k^{bid} \cdot Q_k^{ask} + P_k^{ask} \cdot Q_k^{bid})}{\sum_{k=1}^{K} w_k \cdot (Q_k^{ask} + Q_k^{bid})}$$

ここで $w_k = e^{-\alpha k}$ は指数減衰重み ($\alpha \approx 0.5$)。

#### 既存実装の再利用分析

| 既存 | ファイル | 再利用度 |
|---|---|---|
| `compute_microprice_bias_bps()` | `maker_price.py` L444 | **ベース関数** — L1のみ使用中だが、同一メソッド内でL5拡張可能 |
| `_last_ob_snapshot` | `maker_price.py` L431 | **そのまま** — `OrderBookSnapshot.bids[:5]` / `asks[:5]` は**既にメモリ上に存在** |
| `compute_imbalance()` depth=5 | `maker_price.py` L417 | **変更不要** — OB のキャッシュ更新を行っており、microprice はこの後に呼ばれる |
| microprice side selection | `side_selector.py` L111 | **変更不要** — `microprice_bias_bps` 引数経由で自動反映 |
| ガードレール (310#) | `side_selector.py` L128 | **変更不要** — spread/regime gate はそのまま有効 |

**結論**: `compute_microprice_bias_bps()` の**内部15行**を書き換えるだけ。API・インターフェース変更ゼロ。

#### 追加 API 呼出し: **不要**

現在の `compute_imbalance()` が `depth=5` で板を取得し `_last_ob_snapshot` にキャッシュしている。L5 microprice は同じスナップショットを再利用する。

#### 計算高速化

- **現状**: L1 のみ → 乗算2回 + 加算2回。O(1)
- **L5**: 乗算10回 + 加算10回 + 指数重み5回。依然 O(1)
- `exp(-αk)` は固定 K=5 なので**定数テーブル**に展開可能 (`WEIGHTS = [1.0, 0.607, 0.368, 0.223, 0.135]`)
- **ボトルネック**: なし。追加計算コスト ≈ 50ns

#### 実装方針

```python
# maker_price.py compute_microprice_bias_bps() の置換案
_MICRO_WEIGHTS: Final[list[float]] = [
    1.0, 0.6065, 0.3679, 0.2231, 0.1353,  # exp(-0.5*k), k=0..4
]

def compute_microprice_bias_bps(self) -> float:
    ob = self._last_ob_snapshot
    if ob is None or not ob.bids or not ob.asks:
        return 0.0
    depth = min(len(ob.bids), len(ob.asks), 5)
    if depth == 0:
        return 0.0
    num, den = 0.0, 0.0
    for k in range(depth):
        w = _MICRO_WEIGHTS[k]
        pb, qb = ob.bids[k]
        pa, qa = ob.asks[k]
        num += w * (pb * qa + pa * qb)
        den += w * (qa + qb)
    if den <= 0:
        return 0.0
    microprice = num / den
    mid = (ob.bids[0][0] + ob.asks[0][0]) / 2.0
    if mid <= 0:
        return 0.0
    return (microprice - mid) / mid * 10_000.0
```

---

### §2.2 M2: Bayesian Regime Detection (Hamilton 1989 拡張)

#### 理論

Hamilton (1989) Markov-Switching Model の**オンラインベイズ更新**:

$$P(s_t = j \mid y_{1:t}) \propto f(y_t \mid s_t = j) \sum_{i=1}^{S} P(s_t = j \mid s_{t-1} = i) \cdot P(s_{t-1} = i \mid y_{1:t-1})$$

- $f(y_t \mid s_t)$: 状態 $s_t$ からの emission probability (正規分布)
- $P(s_t \mid s_{t-1})$: 遷移確率行列 (A 行列)

#### 既存実装の再利用分析

| 既存 | ファイル | 再利用度 |
|---|---|---|
| `_calculate_transition_matrix()` | `v444_regime_analyzer.py` L380 | **A 行列の計算ロジックがそのまま使える**。regime_labels → 遷移回数 → 確率 |
| `FillTestRegimeDetector._classify()` | `regime_detector.py` L348 | **置換候補**: 閾値ベース → ベイズ事後確率の `argmax` に |
| `_apply_hysteresis()` | `regime_detector.py` L398 | **不要化**: ベイズ filtering 自体がヒステリシスの上位互換 |
| `RegimeClassifier._compute_scores()` | `v444_regime_classifier.py` L600 | **emission probability のヒント**: 12指標スコアが emission の基盤 |
| `_estimate_sigma()` | `maker_microstructure.py` L77 | **σ パラメータ**: emission 正規分布の σ に活用可能 |

**不足**: emission probability モデル (各レジーム状態の return 分布のフィッティング)。hmmlearn は未導入。

#### 計算高速化

- **4状態 Bayesian filter**: 更新は行列ベクトル積 $4 \times 4 \cdot 4 \times 1 = 4 \times 1$。O(1) per step
- `_safe_returns()` の rolling window std が毎回全数再計算 → **Welford online variance** で O(1) 化可能
- A 行列の再推定は低頻度 (1h毎) なので計算コスト問題なし

#### 実装方針

```
Phase A: FillTestRegimeDetector に Bayesian 事後確率を追加 (4次元確率ベクトル)
  - _classify() の戻り値に probabilities: np.ndarray を追加
  - 既存の閾値分類はフォールバックとして残す
  - A 行列は v444_regime_analyzer の計算ロジックを流用
Phase B: fill_test 側で probability を活用
  - regime_boost の連続乗数化: boost = Σ P(s) · boost_coeff(s)
  - offset の確率加重: 現在の離散 regime 切替の代わりに確率混合
Phase C: 定期的な A 行列再推定 (1h毎)
  - fill_records から regime label 列を抽出 → 遷移行列更新
```

#### Welford Online Variance (計算高速化の詳細)

現在の `_safe_returns()` (regime_detector.py L227):
```python
# 現状: O(n) — 毎回全 window を再計算
returns = np.diff(self._prices[-n:]) / self._prices[-n-1:-1]
current_vol = np.std(returns)
```

Welford (1962) online algorithm に置換:
```python
# O(1) per update — 新しい return の追加/古い return の除去
class WelfordOnlineVar:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    def remove(self, x: float) -> None:
        if self.count <= 1:
            self.count = 0; self.mean = 0.0; self.M2 = 0.0
            return
        delta = x - self.mean
        self.count -= 1
        self.mean -= delta / self.count
        self.M2 -= delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
```

---

### §2.3 M3: σ-Clustering (Volatility Regime Classification)

#### 理論

ボラティリティの**離散クラスタ**を K-Means/GMM で自動検出し、regime 依存の offset をキャリブレーションする。

| Cluster | σ 範囲 (例) | fill_test の offset 戦略 |
|---|---|---|
| Low-σ | < 0.05% | tight offset → fill rate 優先 |
| Mid-σ | 0.05%-0.20% | balanced (現状の default) |
| High-σ | > 0.20% | wide offset → AS 防御優先 |
| Extreme-σ | > 0.50% | halt / 超ワイド |

#### 既存実装の再利用分析

| 既存 | ファイル | 再利用度 |
|---|---|---|
| `_estimate_sigma()` Parkinson | `maker_microstructure.py` L77 | **σ 入力そのまま** — Parkinson σ は安定的 |
| `vol_ratio` | `regime_detector.py` L227 | **正規化済みσ** — baseline 比でクラスタリング可能 |
| `realized_volatility()` | `scalping.py` L210 | **RV 時系列** — ⚠ O(n²) → numpy化必須 |
| EWMA パターン | `sell_dynamic_kill.py` L285 | **σ の EWMA** にパターン転用可能 |
| Bollinger squeeze | `bollinger_ext.py` L61 | **低σ 検出** に流用可能 |

#### 計算高速化

1. **`realized_volatility()` O(n²) → O(n)**: 2重ループを numpy vectorize

```python
# 現状 (O(n²)):
for i in range(window, len(close)):
    returns = np.zeros(window - 1)
    for j in range(1, window):      # ← 内側ループ
        idx = i - window + j
        returns[j - 1] = (close[idx] - close[idx - 1]) / close[idx - 1]
    rv[i] = np.sqrt(np.sum(returns ** 2))

# 改善案 (O(n)):
log_returns = np.diff(np.log(close))
squared_returns = log_returns ** 2
cumsum = np.cumsum(squared_returns)
rv[window:] = np.sqrt(cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]]))
```

2. **Online K-Means**: 全データを保持せず、オンラインのミニバッチ K-Means でクラスタ更新 (sklearn `MiniBatchKMeans`)
3. **numba 候補**: σ-clustering の距離計算は `@nb.jit(nopython=True)` 化可能 (既存の numba パターンが `ztb/analysis/optimization.py` にあり)

---

### §2.4 M4: GLFT Fill Probability Model (Guéant-Lehalle-Fernandez-Tapia 2013)

#### 理論

GLFT の核心は**fill probability** (到着率) のモデル化:

$$A(\delta) = A \cdot e^{-k\delta}$$

- $A$: 基準到着強度 (単位時間あたりの fill 確率)
- $k$: decay 定数 (offset 感度)
- $\delta$: offset (mid からの距離)

最適 offset:

$$\delta^* = \frac{1}{k} + \frac{q \cdot \gamma \cdot \sigma^2 \cdot \tau}{k}$$

#### 既存実装の再利用分析

| 既存 | ファイル | 再利用度 |
|---|---|---|
| `_apply_as_reservation_shift()` | `maker_microstructure.py` L165 | **★★★★★** — AS δ* + inventory 項 + τ動的化が**全て実装済み** |
| `_dynamic_tau()` | `maker_microstructure.py` L139 | **τ動的化** — `τ_eff = τ_base / vol_ratio` |
| `estimate_queue_depth()` | `maker_price.py` L490 | **queue position** — fill probability の実証ベース入力 |
| Kyle λ | `maker_microstructure.py` L258 | `λ = spread / (2·depth)` — price impact の第一近似 |
| Amihud ILLIQ | `maker_microstructure.py` L296 | `ILLIQ = spread/mid / depth` — 流動性指標 |
| `adaptation_engine.py` | adaptation_engine L285 | **AS ratio per side** — fill に対する逆選択の実測値 |

**結論**: GLFT の**理論的フレームワークの 80% は既に実装されている**。不足は $A(\delta) = A \cdot e^{-k\delta}$ の**到着率モデル** (fill_records から A, k を回帰推定) のみ。

#### 計算高速化

- fill_records からの A, k 推定: OLS 対数回帰 `log(fill_rate) = log(A) - k·δ` → O(n) で十分高速
- 推定は低頻度 (1h毎、adaptation_engine と同期) で十分
- **numba 候補**: A, k からの δ* 計算は既にスカラー演算で O(1)。高速化不要

#### 実装方針

```
新規: fill_probability_model.py
1. fill_records の offset vs fill/timeout を集約
2. log-linear 回帰で A, k を推定
3. δ* = 1/k + q·γ·σ²·τ/k を計算
4. 既存の _apply_as_reservation_shift() の reservation_delta に注入
```

---

### §2.5 M5: Volume-Synchronized VPIN (Easley-López de Prado 2012)

#### 理論

Time-bucket VPIN (現状) vs Volume-bucket VPIN (提案) の違い:

| | Time-bucket (現状) | Volume-bucket (提案) |
|---|---|---|
| バケット | 60秒毎 | V_bucket 出来高毎 |
| ノイズ | 低出来高時に精度低下 | 出来高正規化で安定 |
| 理論根拠 | 簡易近似 | Easley+ (2012) |
| 感度 | 時間帯依存 | 出来高密度依存 |

#### 既存実装の再利用分析

| 既存 | ファイル | 再利用度 |
|---|---|---|
| `vpin = abs(signed_flow) / total_vol` | `feature_enricher.py` L398 | **core ロジック** — bucket 化部分のみ変更 |
| `cumulative_buy_volume` / `cumulative_total_volume` | `feature_enricher.py` L352 | **累積配列** — volume bucket 境界検出に再利用可能 |
| `_apply_volatility_guard()` VPIN 消費 | `maker_risk_guards.py` L82 | **消費側は成熟** — continuous quadratic + buy asymmetry |
| `vpin_60s`, `vpin_30s`, `vpin_300s` | `feature_enricher.py` L839-854 | **SkipGate 特徴量** — volume-sync 版で精度向上 |
| `TestVPINContinuousModulator` | テスト群 | **テストパターン** — 新 VPIN のテスト基盤 |

#### 計算高速化

- **Volume-bucket 化**: 累積出来高の二分探索 `np.searchsorted` で O(log n) per bucket
- **全体**: n 本の約定 → n/B バケット (B=バケットサイズ)。O(n) で十分
- **numba 候補**: バケット境界検出 + VPIN 計算の一体化 JIT で 5-10x 可能

```python
# Volume-bucket VPIN の numba 化案
@nb.jit(nopython=True)
def compute_vpin_volume_sync(
    prices: np.ndarray,
    volumes: np.ndarray,
    buy_flags: np.ndarray,
    bucket_size: float,
    n_buckets: int = 50,
) -> float:
    cum_vol = 0.0
    bucket_buy = 0.0
    bucket_sell = 0.0
    buckets = np.zeros(n_buckets)
    bucket_idx = 0
    for i in range(len(prices)):
        v = volumes[i]
        if buy_flags[i]:
            bucket_buy += v
        else:
            bucket_sell += v
        cum_vol += v
        if cum_vol >= bucket_size:
            if bucket_idx < n_buckets:
                buckets[bucket_idx] = abs(bucket_buy - bucket_sell) / (bucket_buy + bucket_sell + 1e-10)
                bucket_idx += 1
            bucket_buy = 0.0
            bucket_sell = 0.0
            cum_vol = 0.0
    if bucket_idx == 0:
        return 0.5
    return float(np.mean(buckets[:bucket_idx]))
```

---

## §3 技術負債インベントリ (T1-T10)

| ID | ファイル | 問題 | 深刻度 | 工数 | Codex 委任 |
|---|---|---|---|---|---|
| T1 | `retrain_scheduler.py` L520 | `except:` bare except — SystemExit/KeyboardInterrupt を飲む | ⚠ High | 0.5h | ✅ |
| T2 | `order_monitor.py` | 11箇所の broad `except Exception` — エラー分類不能 | ⚠ Medium | 2h | ✅ |
| T3 | `realtime_optimizer.py` | 7個の TODO — 未完成ロジックが散在 | Medium | 4h | ⚠ 要判断 |
| T4 | `scalping.py` L210 | `realized_volatility()` O(n²) 二重ループ | Medium | 1h | ✅ |
| T5 | `regime_detector.py` L227 | `_safe_returns()` 毎回全 window 再計算 → Welford O(1) 化 | Medium | 2h | ✅ |
| T6 | `ztb/` 全体 | 43個の TODO コメント | Low | 各0.5h | ✅ |
| T7 | `ztb/` 全体 | 50+ 箇所の `# type: ignore` — 型安全性低下 | Low | 各0.5h | ⚠ 一部のみ |
| T8 | `ztb/` 全体 | 80+ 箇所の broad `except` — エラー識別不能 | Medium | 各0.5h | ✅ |
| T9 | `scalping.py` L125 | `order_flow_imbalance()` for ループ → numpy 化 | Low | 1h | ✅ |
| T10 | `v444_regime_classifier.py` | 12レジーム × pandas ベース指標計算 — 粒度過剰 | Low | 4h | ❌ 設計判断 |

---

## §4 計算高速化ロードマップ

### §4.1 既存の高速化基盤

| 基盤 | ファイル | 状態 |
|---|---|---|
| numba `@nb.jit(nopython=True)` | `ztb/analysis/optimization.py` | **実績あり** (KAMA, ADX, Kalman) |
| `CPUParallelProcessor` | `ztb/training/utils/parallel_utils.py` | **実績あり** (ProcessPool / ThreadPool) |
| `CacheCoordinator` (LRU + TTL) | `ztb/utils/cache_coordination.py` | **実績あり** (multiprocessing 対応) |
| `FeatureCache` (DataFrame hash) | `ztb/features/processors/caching/cache.py` | **実績あり** |
| `@cached_with_ttl` decorator | `ztb/cache/parquet_io.py` | **実績あり** |
| `@lru_cache` | 30+ 箇所 | **広範に使用** |

### §4.2 高速化対象一覧

| 対象 | 現状 | 改善 | 手法 | 期待改善率 |
|---|---|---|---|---|
| `realized_volatility()` | O(n²) for ループ | O(n) numpy vectorize | numpy cumsum | **100x** (~10ms→0.1ms) |
| `_safe_returns()` std | O(n) 毎回再計算 | O(1) Welford online | Welford (1962) | **10x** (~1ms→0.1ms) |
| `order_flow_imbalance()` | O(n) for ループ | O(n) numpy vectorize | numpy | **5x** |
| VPIN bucket 化 | O(n) Python loop | O(n) numba JIT | `@nb.jit` | **10x** |
| σ-clustering 距離計算 | 未実装 | numba JIT | `@nb.jit` | N/A (新規) |
| microprice L5 | O(1) 5回ループ | O(1) 定数テーブル | weights precompute | **微小** (50ns) |
| A 行列再推定 | 未実装 | O(n) 1h周期 | numpy | N/A (新規) |

### §4.3 numba テンプレート (既存パターンの転用)

既に `ztb/analysis/optimization.py` に確立された numba パターン:

```python
# optimization.py L40-80 (既存 — KAMA 計算)
@nb.jit(nopython=True)
def _calculate_kama_numba(
    close_prices: np.ndarray,
    period: int,
    fast_span: int,
    slow_span: int,
) -> np.ndarray:
    kama = np.empty_like(close_prices)
    # ... pure numpy/scalar computation ...
    return kama
```

新規の numba 関数もこのパターンに準拠する:
- `nopython=True` 必須
- 入力は `np.ndarray` + スカラーのみ
- pandas 非依存
- 戻り値は `np.ndarray` or scalar

---

## §5 Codex 委任可能タスクの分析

### §5.1 委任適性の判定基準

| 基準 | 適 (✅) | 要注意 (⚠) | 不適 (❌) |
|---|---|---|---|
| スコープ | 単一ファイル/関数の改修 | 複数ファイル連動 | アーキテクチャ設計判断 |
| テスト | 既存テストで検証可能 | テスト新規作成が必要 | テスト不可能 (live のみ) |
| 既存パターン | 同等の実装が codebase に存在 | 類似パターンはあるが応用 | 前例なし |
| リスク | low (revert 容易) | medium | high (収益直結) |

### §5.2 委任推奨タスク

| タスク | 理由 |
|---|---|
| **T4**: `realized_volatility()` numpy 化 | 純粋な計算高速化、テスト既存、パターン明確 |
| **T5**: Welford online variance | アルゴリズム変換、単体テスト容易 |
| **T9**: `order_flow_imbalance()` numpy 化 | T4 と同パターン |
| **T1**: bare except 修正 | 機械的修正 |
| **T2**: broad except 絞り込み | 機械的修正 (ただし 11 箇所で工数大) |
| **M1**: Microprice L5 拡張 | 実装案が §2.1 に明示済み。テスト可能 |
| **M5**: Volume-sync VPIN | 既存 VPIN の生成ロジック差し替え。テスト可能 |

### §5.3 委任非推奨 (Copilot 対話で進めるべき)

| タスク | 理由 |
|---|---|
| **M2**: Bayesian Regime | emission probability のモデル選択に設計判断が必要 |
| **M4**: GLFT Fill Probability | fill_records の解釈 + パイプライン統合に文脈知識必要 |
| **T3**: realtime_optimizer TODO | TODO の妥当性判断が必要 (一部は "do nothing" が正解) |
| **T10**: 12-regime 粒度見直し | 収益影響の大きい設計判断 |

---

## §6 実装優先順位

| 優先度 | タスク | 工数 | 依存 | 理由 |
|---|---|---|---|---|
| **P1** | M1: Microprice L5 | 2-4h | なし | 既存基盤 90%。API 変更ゼロ。AS 低減に直結 |
| **P2** | T4+T9: scalping.py numpy化 | 2h | なし | O(n²)→O(n)。Codex 委任最適 |
| **P3** | T5: Welford variance | 2h | なし | regime_detector の毎サイクル計算を O(1) 化 |
| **P4** | M5: Volume-Sync VPIN | 4-8h | なし | VPIN 精度↑ → offset 適応精度↑ |
| **P5** | M4: GLFT Fill Prob | 4-8h | なし | fill_records 基盤あり |
| **P6** | M3: σ-Clustering | 4-8h | T5 | Welford σ を入力に使用 |
| **P7** | M2: Bayesian Regime | 1-2d | T5, M3 | 設計判断が多い |
| **P8** | T1-T2: except 修正 | 2-4h | なし | 安定性、Codex 最適 |

---

## §7 収益インパクト評価

| 提案 | メカニズム | 期待効果 |
|---|---|---|
| M1 Microprice L5 | AS 低減 → 逆選択 fill 率↓ → 平均 PnL/fill ↑ | **+0.5-2.0 bps/fill** |
| M2 Bayesian Regime | 遷移予測 → regime boost の先行調整 → DD 圧縮 | **DD -10-20%** |
| M3 σ-Clustering | vol regime 精度↑ → offset キャリブレーション↑ | **fill rate +5-10%** |
| M4 GLFT Fill Prob | δ* 最適化 → fill rate / AS トレードオフ改善 | **+0.3-1.0 bps/fill** |
| M5 Volume-Sync VPIN | toxicity 検知精度↑ → toxic flow 回避率↑ | **AS ratio -5-15%** |

---

## §8 自己批判的レビュー

### §8.1 批判点

1. **M1 の L5 重みパラメータ α=0.5 は根拠薄弱** — BTC/JPY の板の厚み分布は銘柄固有。α のキャリブレーションが必要だが、ZaifのBTC/JPY板は常に薄い (L3以降はほぼ空の可能性)。
   - **対策**: L5 の有効段数を動的判定 (qty > min_qty のレベルのみ使用)

2. **Bayesian Regime の emission 分布が正規分布で十分か不明** — crypto は fat tail (レヴィ分布/t分布)。正規近似は tail event の過小評価につながる。
   - **対策**: t分布 (自由度推定) or non-parametric (カーネル密度) も検討

3. **σ-clustering の K=4 は天下り的** — 最適クラスタ数は Elbow 法/BIC で決定すべき。固定は overfitting/underfitting のリスク。
   - **対策**: 初期調査で K=2-6 を比較。BIC 最小の K を採用

4. **GLFT の到着率モデル $A(\delta)=A e^{-k\delta}$ は Zaif BTC/JPY で成立するか未検証** — 流動性の極めて薄い市場では指数減衰より power-law $A(\delta) = A \cdot \delta^{-\beta}$ が適切な場合がある。
   - **対策**: fill_records から指数 vs 冪乗の AIC 比較

5. **Volume-sync VPIN のバケットサイズ選択が収益を左右する** — 過小 → ノイジー、過大 → 遅延。BTC/JPY の日次出来高からバケットサイズを動的調整する必要。
   - **対策**: 日次出来高の 1/50 (Easley 推奨) で初期設定、adaptation_engine で動的調整

### §8.2 計算高速化の注意点

- numba JIT の初回コンパイルは**重い** (~1-5秒)。fill_test 起動時に warm-up 呼び出しが必要
- Welford online variance は**数値安定性**に注意 (catastrophic cancellation)。上記実装は Welford の原論文準拠で安全
- `MiniBatchKMeans` は scikit-learn 依存。fill_test 環境に sklearn が入っているか要確認 (→ 入っている: SkipGate が使用)

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-10 | 1.0 | 初版 (M1-M5 市場理論提案 + T1-T10 技術負債 + 計算高速化) |
| 2026-03-10 | 1.1 | M1✅ T4✅ T5✅ T9✅ 完了。T1 は既に修正済みと判明 |
| 2026-03-10 | 1.2 | M3✅ M4✅ M5✅ 完了。残り M2 のみ (emission分布設計必要) |
| 2026-03-10 | 1.3 | M2✅ Phase A 完了 (BayesianRegimeFilter 単体実装 42テスト)。M1-M5 全完了 |
| 2026-03-15 | 1.4 | 365# SAC ブロッカー P3/P4/P5 完了 (sidecar interface + signal I/O + gate injection) |
| 2026-03-15 | 1.5 | 365# P6 完了 — sac_retrain_scheduler.py (warm-start + OOS gate + atomic deploy, 24 tests) |
| 2026-03-15 | 1.6 | 365# P7 (embed_action_masks) + P8 (LiteTradingEnv) 完了 — P1-P8 全ブロッカー解消 (32 tests) |
| 2026-03-15 | 1.7 | 371# M2-M5 live 配線完了 + GLFT calibration cycle + Bayesian state persistence + VPIN 負amount guard |
| 2026-03-11 | 1.8 | 378# 366#以前の市場理論導入履歴を 377# §10 に体系化。035#→366# の段階的導入タイムライン + offset パイプライン全景図を追記 |
