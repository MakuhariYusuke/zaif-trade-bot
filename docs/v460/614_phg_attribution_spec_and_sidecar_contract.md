# 614# 執行パイプライン寄与度分析仕様および Sidecar Feature Contract 策定

- **日付**: 2026-03-24
- **目的**: 608# の数値誤認を正し、実効的な Attribution 分析および Sidecar インプロセス推論のための数理仕様・データ契約を策定する。
- **役割**: 数理仕様・設計文書の策定（コード・YAML 変更・テスト実装は担当外）。

---

## §1 608# 正誤対照表 (T1)

608# における前提値の誤認と、それが議論に与えた影響を整理する。

| 項目 | 608# の記載 | 実値 (検証済み) | 出典 | 影響 |
|:---|:---|:---|:---|:---|
| **Sidecar TTL** | 600s | **7800.0s** | `sidecar_types.py:45` | 有効率低下の原因を TTL 値そのものと誤認。真因は「更新の retrain 成功依存」にある。 |
| **Ceiling (buy)** | 0.15-0.35 | **0.35** | `fill_test.yaml` | 上限ガードの余裕を過小評価。 |
| **Ceiling (sell)** | 0.15-0.35 | **0.40** | `fill_test.yaml` | 同上。 |
| **max_boost_bps** | ±0.15bps | **0.20bps** | `fill_test.yaml:632` | Sidecar の影響力を過小評価。 |
| **Stage Max Mult** | 未言及 | **2.0 cap 実装済** | 565# P3 | 乗算的膨張が野放しであるという誤った危機感を醸成。 |
| **Pipeline 状態** | 9段乗算のみ | **加法も実装済** | 581# | 加法移行を「新規開発」と誤認。実際には「有効化と較正」のフェーズ。 |
| **Nyquist 定理** | 推論頻度の根拠 | **誤用** | 609# §1.2 | 理論的根拠の不適切。情報理論的・実用的理由に置換が必要。 |

---

## §2 Attribution Analyzer 仕様書 (Phase 1) (T2)

既存の `scripts/v460/analysis/analyze_fill_logs.py` を拡張し、執行パイプラインの透明性を確保する。

### 2.1 計測指標定義
- **clamp_rate [%]**: `execution_pre_clamp_offset > effective_offset_used + 0.0001` となるサイクルの割合。
- **information_loss [bps]**: `(execution_pre_clamp_offset - effective_offset_used) * mid_price / 10000` の累積および平均。
- **stage_saturation [%]**: `executor_offset_stages` 内の各ステージ出力が 2.0 (Max Mult) に到達した頻度。

### 2.2 出力仕様
- **集計粒度**: Daily サマリおよび `run_id` 単位のサマリ。
- **フィルタ条件**: `side`, `regime`, `git_sha` 別のクロス集計。
- **フォーマット**: 既存のテキストレポートへのセクション追加、および詳細分析用の JSON 出力。

---

## §3 σ-unit 正規化の数理仕様 (T3)

オフセット寄与度 $\Delta R_i$ をボラティリティ適応型にするための仕様。

### 3.1 スケーリング公式
各ステージのオフセット寄与度 $\Delta R_i$ を以下のように定義する：

$$\Delta R_i = f_i(\text{signal}_i) \times \frac{\sigma_{current}}{\sigma_{baseline}}$$

- $f_i(\text{signal}_i)$: 信号入力に対する基本オフセット変換関数。
- $\sigma_{current}$: `RobustStats.asymmetric_ema()` (575#) により算出される直近ボラティリティ。
- $\sigma_{baseline}$: 長期的な基準ボラティリティ（例：過去 24 時間の `sigma` の EMA）。初期値は直近 1 時間の平均。

### 3.2 理論的接続
Avellaneda-Stoikov (2008) における最適スプレッド $s = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln(1 + \frac{\gamma}{\kappa})$ に基づき、$\sigma$ の変動をオフセットの「歩幅」に直接反映させることで、リスク量に応じた動的なクォート配置を実現する。

---

## §4 加法パイプライン A/B テスト設計 (T4)

`experimental_additive_pipeline` トグルを用いた検証計画。

### 4.1 仮説
- **帰無仮説 ($H_0$)**: 加法パイプラインと乗法パイプラインの間で PnL および fill_rate に有意な差はない。
- **対立仮説 ($H_1$)**: 加法パイプラインは乗法パイプラインよりも `clamp_rate` を低下させ、`post_fill_30s_pnl` を改善する。

### 4.2 比較指標と判定基準
1.  **PnL (bps)**: 有意水準 5% での改善。
2.  **fill_rate [%]**: 低下幅が 5% 以内であることを確認。
3.  **clamp_rate [%]**: 有意な低下を確認。
4.  **information_loss [bps]**: 削減量を確認。

### 4.3 撤退基準
- `fill_rate` が 10% 以上低下した場合。
- `postonly_crossing_skip` が 2.0% を超えた場合。
- 累積 PnL がドローダウン許容限度（例：-50bps）に達した場合。

---

## §5 Sidecar Feature Contract 仕様 (T5)

インプロセス推論（Option B）を実現するための、訓練・推論間のデータ契約。

### 5.1 共有特徴量リスト (Feature Set)
以下の特徴量を live 側で毎サイクル（あるいは 1 分毎）に計算し、推論モデルへ供給する。

| カラム名 | 型 | 単位 | データソース |
|:---|:---|:---|:---|
| `price_velocity` | float | bps/s | Mid-price (1m window) |
| `order_flow_imbalance` | float | [-1, 1] | Orderbook (L2, 5-level) |
| `micro_volatility` | float | bps | Standard deviation of mid (1m) |
| `parkinson_sigma` | float | bps | High-Low range (1m) |
| `vpin_proxy` | float | [0, 1] | Volume/Price sync (544#) |
| `ema_velocity_bps` | float | bps | `smoothed_velocity_bps` (227#) |

### 5.2 正規化パラメータ (Normalization)
- **管理場所**: `models/v460/sac_sidecar.norm.json`
- **内容**: 特徴量ごとの `mean`, `std`, `min`, `max` (Z-score 補正用)。
- **更新タイミング**: 再学習（Retrain）成功時に訓練側で保存し、推論側で hot-reload する。

### 5.3 供給頻度
- **計算頻度**: `fill_test` の各サイクル開始時。
- **精度**: 特徴量エンジニアリング層と `sac_retrain_scheduler` 側の計算ロジックが Python レベルで同一であることを単体テストで保証する。

---

## §6 Action Range 拡大ロードマップ (T6)

### 6.1 段階的投入ステップ
1.  **Step 1**: ±0.20 bps (現状) → ±0.30 bps。
2.  **Step 2**: ±0.30 bps → ±0.40 bps (基準ハーフスプレッド $\delta^*$ 近傍)。
3.  **Step 3**: ±0.40 bps → **0.50 bps (当面の上限)**。

### 6.2 監視および撤退基準
- **監視指標**: `postonly_crossing_skip` 発生率。
- **撤退条件**:
    - `postonly_crossing_skip` > 1.5% (現状 1.1% からの有意な悪化)。
    - `adverse_selection_cost_bps` の悪化。
- **非対称性の考慮**: `offset_ceiling_ratio_buy` (0.35) と `sell` (0.40) の差異に基づき、buy 側はより早期にクリップが発生することを許容する。

---

*以上。本文書は設計仕様であり、実装は担当エンジニアに委ねる。*
