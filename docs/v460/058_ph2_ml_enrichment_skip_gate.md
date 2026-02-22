# 058# ML 強化: マイクロストラクチャ特徴量 + PnL 回帰 Skip Gate

**日付**: 2026-02-15  
**コミット**: `2f6284113`  
**前提**: 057# ML ベースライン (ROC-AUC 0.528, AS 分類ほぼランダム)  
**ステータス**: 完了 — Ridge skip gate がベースラインPnLを +0.53 bps 改善  

---

## §0 エグゼクティブサマリ

057# で構築した AS 分類器は ROC-AUC 0.528 でほぼランダム。
fill record のメタデータ (10 特徴量) だけでは情報が足りない。

本 058# では、raw orderbook snapshots (15,258 件) と trades (866,830 件)
からマイクロストラクチャ特徴量を抽出し、AS 分類器の強化と、
**PnL 直接予測ベースの skip gate** の2つのアプローチを実装した。

| アプローチ | 結果 | 判定 |
|---|---|---|
| AS 分類器 (enriched) | ROC-AUC 0.537 (+0.01) | 微改善だが不十分 |
| **PnL Ridge skip gate** | -0.51 → +0.03 bps | **有効** ⭐ |

**結論**: 二値分類 (AS/non-AS) より連続値回帰 (PnL bps) が少サンプルで有効。
Ridge の L2 正則化が 373 サンプルの小データに適合。

---

## §1 問題設定

### §1.1 なぜ AS 分類では足りないか

| 課題 | 詳細 |
|---|---|
| ラベリング品質 | `adverse_selection` ラベルは fill 後の mid_price 変化で定義。閾値依存で明確な境界がない |
| クラス不均衡 | AS=52.1%, non-AS=47.9% — ほぼ均等だが、境界が曖昧 |
| サンプル数 | AS ラベル付き: 284 件。小サンプルで高次元特徴量は過学習しやすい |
| 情報量 | fill record メタ (queue_wait, edge_bps, spread) だけでは市場の状態を十分に捉えられない |

### §1.2 方針転換: AS → PnL 直接予測

二値分類から連続値回帰にシフトする理由:

1. **サンプル効率**: PnL は `filled` な全レコード (373 件) で使え、AS ラベル (284) より多い
2. **情報保存**: PnL=+3bps と PnL=+15bps は同じ non-AS だが、経済的には大きく異なる
3. **判定の自然性**: skip 判定に PnL 予測値を直接使えるため、中間層 (分類器) が不要
4. **Ridge の適合性**: PnL は概ね正規分布に近く、線形回帰の仮定に合う

---

## §2 データ棚卸し

### §2.1 利用可能データ

| データ | 件数/行数 | 期間 | 格納場所 | 形式 |
|---|---|---|---|---|
| fill_records | 491 (373 filled, 284 AS-labeled) | Feb 13-15 | `data/v460/results/` | jsonl |
| raw orderbook | 15,258 snapshots | Feb 13-15 | `data/v460/raw/orderbook/` | jsonl.gz |
| raw trades | 866,830 trades | Feb 13-15 | `data/v460/raw/trades/` | jsonl.gz |
| medium parquet | 1,223,371 rows × 31 cols | 2022-01 ~ Feb 15 | `data/` | parquet |
| v460 real parquet | 187 rows × 22 cols | Feb 13 (3h) | `data/` | parquet |

### §2.2 medium parquet vs raw data

medium parquet は OHLCV + テクニカル指標で、fill test 期間 (Feb 13-15) のデータは
Feb 10 時点で 3 日分の空白があった (058# 作業中に yfinance で更新済)。

しかし、fill record へのマッチングには **raw orderbook/trades が直接必要**。
medium parquet の RSI 等テクニカル指標は fill test のタイムスケール (秒) に対して
粒度が粗すぎる (1分足ベース)。

**設計判断**: medium parquet を経由せず、raw data を直接 fill record にマッチングする。

### §2.3 データマッチング統計

| マッチ種別 | 成功 | 失敗 | 率 |
|---|---|---|---|
| orderbook → fill (±5秒) | 347 / 491 | 144 | 70.7% |
| trades (60秒 window) | ~全件 | ~0 | ~100% |

orderbook のマッチ失敗は、スナップショット取得間隔 (5-10秒) の間に発生した fill。
NaN は median 補完で処理。

---

## §3 特徴量設計

### §3.1 特徴量体系 (21 特徴量)

```
                    ┌─────────────────────────────────┐
                    │      Fill Record (491)          │
                    │  timestamp, side, price, ...    │
                    └───────┬─────────────────────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
    ┌────────────┐  ┌──────────────┐  ┌────────────────┐
    │ Base (10)  │  │  Micro (8)   │  │ Interaction (3)│
    │ メタデータ  │  │ raw data 由来│  │ side × market  │
    └────────────┘  └──────────────┘  └────────────────┘
```

#### Base 特徴量 (10) — fill record メタデータ

| # | 特徴量 | 説明 | 値域 |
|---|---|---|---|
| 1 | `log_queue_wait` | log(待ち時間+1) | [0, ~5] |
| 2 | `side_buy` | buy=1, sell=0 | {0, 1} |
| 3 | `hour_sin` | sin(2π·h/24) | [-1, 1] |
| 4 | `hour_cos` | cos(2π·h/24) | [-1, 1] |
| 5 | `edge_bps` | エッジ (bps) | [-30, +30] |
| 6 | `spread_jpy` | 注文時スプレッド (JPY) | [0, ~5000] |
| 7 | `offset_ratio` | スプレッド内オフセット | [0, 1] |
| 8 | `regime_trending` | トレンド相場=1 | {0, 1} |
| 9 | `regime_ranging` | レンジ相場=1 | {0, 1} |
| 10 | `regime_high_vol` | 高ボラ相場=1 | {0, 1} |

#### Micro 特徴量 (8) — raw orderbook / trades 由来

| # | 特徴量 | ソース | 計算方法 | 意味 |
|---|---|---|---|---|
| 11 | `spread_bps_ob` | 板 | (ask-bid)/mid×10000 | 注文時の実スプレッド |
| 12 | `depth_imbalance_ob` | 板 | (bid_vol-ask_vol)/total | bid/ask 5level 厚みの不均衡 |
| 13 | `trade_count_60s` | 約定 | count(ts-60s ~ ts) | 直近60秒の約定件数 (流動性) |
| 14 | `buy_ratio` | 約定 | buy_vol/total_vol | 買い約定の体積比率 |
| 15 | `trade_flow_imbalance_60s` | 約定 | (buy-sell)/total | 買い/売りフローの不均衡 |
| 16 | `avg_trade_size` | 約定 | total_vol/n_trades | 平均約定サイズ |
| 17 | `price_velocity_60s` | 約定 | (last-first)/first×10000 | 直近60秒の価格変化 (bps) |
| 18 | `vpin_60s` | 約定 | |buy-sell|/total | VPIN (informed trading 指標) |

#### Interaction 特徴量 (3) — side × market state

| # | 特徴量 | 計算 | 経済的意味 |
|---|---|---|---|
| 19 | `side_aligned_imbalance` | depth_imbalance × side_sign | 自分の注文と同方向の板厚有利度 |
| 20 | `side_aligned_tfi` | trade_flow × side_sign | 自分の注文と同方向の約定フロー有利度 |
| 21 | `side_aligned_velocity` | price_velocity × side_sign | 自分の注文と同方向の値動き有利度 |

`side_sign` = buy→+1, sell→-1。正の値は「自分に有利な市場状態」を意味する。

### §3.2 PnL 回帰用の特徴量 (19)

AS 分類用 (21) から `log_queue_wait`, `edge_bps` を除き（fill 後しかわからない）、
代わりに上記の base+micro+interaction の全 19 特徴量を使用。

PnL 回帰では **注文前に取得可能な情報のみ** を使う設計。
`log_queue_wait` は「いつ fill されたか」の事後情報であり、リアルタイムでは使えない。

### §3.3 特徴量の統計量

| 特徴量 (上位) | mean | std | min | max |
|---|---|---|---|---|
| `spread_bps_ob` | ~3.2 | ~1.8 | 0.5 | 15.0 |
| `depth_imbalance_ob` | ~0.05 | ~0.35 | -0.8 | 0.9 |
| `trade_count_60s` | ~85 | ~60 | 0 | 400+ |
| `vpin_60s` | ~0.25 | ~0.15 | 0 | 1.0 |
| `price_velocity_60s` | ~0.1 | ~5.0 | -30 | +30 |

---

## §4 実装アーキテクチャ

### §4.1 ファイル構成

| ファイル | 行数 | 責務 | 依存 |
|---|---|---|---|
| `scripts/v460/ml/feature_enricher.py` | 448 | raw data → fill record enrichment | data_loader |
| `scripts/v460/ml/skip_gate.py` | 375 | PnL 予測 skip gate (Ridge) | enricher, sklearn |
| `scripts/v460/ml/run_ml_pipeline.py` | 392 | パイプライン実行 CLI | enricher, skip_gate, as/fill classifiers |
| `tests/unit/v460/test_enricher_skip_gate.py` | 511 | 23 テスト | enricher, skip_gate |
| `models/v460/skip_gate.pkl` | - | 学習済みモデル (gitignored) | - |

### §4.2 データフローパイプライン

```
 raw orderbook (jsonl.gz)    raw trades (jsonl.gz)
       │                           │
       ▼                           ▼
  load_raw_orderbook()        load_raw_trades()
       │                           │
       ▼                           ▼
  _find_nearest_ob()         _compute_trade_features()
  (binary search, ±5s)       (60s rolling window)
       │                           │
       └───────────┬───────────────┘
                   ▼
         enrich_fill_records()
                   │
           ┌───────┴───────┐
           ▼               ▼
  build_enriched_     build_pnl_
  as_features()       features()
  (21 features)       (19 features)
           │               │
           ▼               ▼
     AS Classifier     Ridge PnL
     (LogReg/GB)       Regressor
           │               │
           ▼               ▼
     ROC-AUC 0.537    SkipGate
                      (threshold=0)
```

### §4.3 SkipGate クラス設計

```python
class SkipGate:
    model: Ridge           # PnL 予測モデル
    scaler: StandardScaler # 特徴量正規化
    feature_cols: list[str] # 19 features (順序固定)
    config: SkipGateConfig  # threshold_bps, max_skip_rate
    _recent_skips: list[bool]  # レート制限用バッファ (直近20件)

    def evaluate(features: dict) -> SkipDecision:
        # 1. 特徴量ベクトル構築 (n_used >= 3 必須)
        # 2. StandardScaler 変換
        # 3. Ridge 予測 → predicted_pnl_bps
        # 4. Skip 判定: pred < threshold → skip
        # 5. レート制限: 直近20件中 skip率 > max_skip_rate → force pass

    def save(path) -> Path  # pickle
    def load(path) -> SkipGate  # pickle
```

**レート制限の設計意図**: 市場環境の急変時にモデルが全注文をスキップしてしまうと
機会損失が発生する。`max_skip_rate=0.7` は直近20件中 70% 以上のスキップを防ぎ、
最低限の約定機会を維持する安全弁。

### §4.4 リアルタイム統合インターフェース

`build_features_from_market_state()` は `run_fill_test.py` からの
リアルタイム呼び出しを想定:

```python
features = build_features_from_market_state(
    side="buy",
    spread_jpy=1500,
    offset_ratio=0.3,
    regime="ranging",
    best_bid=14500000,
    best_ask=14501500,
    bid_vol_5=0.5,
    ask_vol_5=0.3,
    recent_trades=[...],
)
decision = gate.evaluate(features)
if decision.should_skip:
    logger.info(f"Skipping: pred_pnl={decision.predicted_pnl_bps:.2f} bps")
```

---

## §5 評価結果

### §5.1 AS 分類器 (enriched vs baseline)

| モデル | baseline ROC-AUC | enriched ROC-AUC | Δ | PR-AUC |
|---|---|---|---|---|
| GB | 0.528 | 0.502 | -0.026 | 0.541 |
| LR | 0.525 | **0.537** | **+0.012** | 0.578 |

微改善。GB の ROC-AUC 低下は過学習による可能性 (21 特徴量で 284 サンプル)。

**GB Top-5 特徴量重要度 (enriched)**:

| rank | 特徴量 | importance | カテゴリ |
|---|---|---|---|
| 1 | `log_queue_wait` | 0.16 | base |
| 2 | `edge_bps` | 0.15 | base |
| 3 | `spread_jpy` | 0.11 | base |
| 4 | **`side_aligned_tfi`** | **0.07** | **interaction** |
| 5 | **`side_aligned_imbalance`** | **0.06** | **interaction** |

Interaction 特徴量が top-5 に入り、マイクロストラクチャ情報の寄与を確認。

### §5.2 PnL Ridge 回帰 ⭐

**評価方法**: TimeSeriesSplit CV (5 splits) + OOF (Out-of-Fold) skip simulation

| モデル | IC (Spearman) | MAE | Baseline PnL | Skip PnL | Δ |
|---|---|---|---|---|---|
| **Ridge (α=10)** | - | - | **-0.51 bps** | **+0.03 bps** | **+0.53 bps** |
| GBR | -0.054 | 3.23 bps | -0.51 bps | -0.54 bps | -0.03 bps |

Ridge は 62% をスキップし (310 → 117 keep)、平均 PnL を正に転換。
GBR は過学習 (IC 負値) で改善なし。

**Ridge Top-5 特徴量 (|coefficient| after scaling)**:

| rank | 特徴量 | |coef| | 解釈 |
|---|---|---|---|
| 1 | `offset_ratio` | 0.97 | スプレッド内でより中央寄り → 有利 |
| 2 | `regime_ranging` | 0.62 | レンジ相場で maker 有利 |
| 3 | `price_velocity_60s` | 0.61 | 直近の値動き大 → AS リスク ↑ |
| 4 | `vpin_60s` | 0.50 | VPIN 高 → informed trader 活動 |
| 5 | `hour_cos` | 0.50 | 時間帯効果 (深夜は有利?) |

### §5.3 GBR がダメな理由

| 問題 | 詳細 |
|---|---|
| サンプル数 | 373 サンプルで木ベース手法は過学習しやすい |
| IC 負値 | OOF 予測と実際の PnL が逆相関 → 有害な予測 |
| 非線形性の限界 | PnL の分布は概ね線形的。非線形モデルの利点がない |

### §5.4 経済的意義

```
 Before (057#)        After (058#)
 ┌──────────┐        ┌──────────┐
 │ Mean PnL │        │ Mean PnL │
 │ -0.51 bps│   →    │ +0.03 bps│
 │          │        │          │
 │ All 310  │        │ Keep 117 │
 │ orders   │        │ Skip 193 │
 └──────────┘        └──────────┘

 年間影響推計 (仮に 1 日 300 注文):
 Before: 300 注文 × -0.51 bps = -153 bps/day
 After:  117 注文 × +0.03 bps =   +3.5 bps/day
 Δ = +156.5 bps/day = 15.65 JPY/day (仮に 1 BTC × ¥10M)
```

> ⚠️ 上記は fill test 期間 (3日間) のデータに基づく推計。
> サンプル外安定性は未検証。データ蓄積が進めばより信頼性の高い評価が可能。

---

## §6 批判的検証

### §6.1 楽観的バイアスの懸念

| リスク | 対策 | 残存リスク |
|---|---|---|
| 小サンプル (373) で過学習 | Ridge L2, TimeSeriesSplit CV | あり — 7日以上のデータが必要 |
| PnL の定義 (30秒) | 短すぎて最終損益を反映しない可能性 | PnL 窓の最適化が今後の課題 |
| Skip による機会損失 | max_skip_rate=0.7 で上限制約 | 最適レート未探索 |
| yfinance 由来データの品質 | Coincheck raw data を使用 (fill test) | medium parquet のみ yfinance |
| threshold=0 の最適性 | 閾値を変えた検証なし | 閾値チューニングが必要 |

### §6.2 AS 分類器がうまくいかない根本原因

1. **外因性情報の欠如**: AS の発生は大口注文やニュースイベントに起因。
   板情報やフロー情報だけでは捉えきれない
2. **ラベルの曖昧さ**: AS ラベルは fill 後の mid_price 変化で定義されるが、
   市場ノイズと AS を区別できない
3. **時間的ずれ**: 板スナップショット (5秒間隔) と fill の間にマーケットが動くため、
   特徴量が実際の状態を反映していない可能性

### §6.3 Ridge が勝つ構造的理由

1. **低次元**: 19 特徴量 × 373 サンプル → p/n = 0.05。Ridge の正則化が有効に機能
2. **線形仮説**: PnL は offset_ratio, regime, price_velocity 等と概ね線形関係
3. **連続性**: PnL は [-50, +50] bps の連続値。二値化するより情報が豊かい
4. **L2 の安定性**: α=10 で重みが適度に縮小され、ノイズへのロバスト性が確保

---

## §7 テスト構成

23 テスト (total 608 passed):

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestTradeFeatures` | 4 | empty data, no-window, computed, all-buy |
| `TestNearestOB` | 3 | empty, exact match, tolerance-exceeded |
| `TestEnrichFillRecords` | 4 | columns/shape/pnl/full-pipeline |
| `TestSkipGate` | 5 | evaluate/disabled/insufficient/rate-limit/save-load |
| `TestMarketStateFeatures` | 5 | basic/interaction/trades/no-trades/all-present |
| `TestIntegration` | 2 | real-data enrichment, real skip gate training |

### テスト設計のポイント

- **Synthetic fixtures**: 100 件の合成 fill record で再現可能なテスト
- **Edge cases**: empty data, tolerance exceeded, insufficient features
- **Rate limiter**: 連続スキップ率の上限テスト
- **Pickle round-trip**: save → load → evaluate の一貫性

---

## §8 v459 との対比

| 項目 | v459 K2 特徴量ゲート | v460 058# Skip Gate |
|---|---|---|
| 目的 | 方向予測 (next bar up/down) | **PnL 予測 (30s post-fill bps)** |
| 特徴量 | RSI×7 + ReturnStdDev (8) | micro×8 + interaction×3 + base (19) |
| データ | 1.2M bars (medium parquet) | **373 fills + 867K trades** |
| モデル | 未実装 (計画のみ) | **Ridge (α=10, 学習済み)** |
| 評価 | Walk-forward IC (計画) | **TimeSeriesSplit CV + OOF skip sim** |
| 使い道 | 情報量ゲート判定 (概念) | **実運用 skip gate (pkl 保存済み)** |
| skip 実績 | なし | 62% skip → mean PnL: -0.51→+0.03 |

---

## §9 コード品質・設計原則

| 原則 | 遵守状況 |
|---|---|
| DRY | `data_loader.build_as_features` を enricher 内で再利用 |
| 単一責任 | enricher=特徴量, skip_gate=判定, run_ml=CLI |
| 型安全 | `from __future__ import annotations`, dataclass, type hints |
| テスト可能性 | synthetic fixtures, dependency injection (raw_dir param) |
| 保守性 | MICRO_FEATURE_COLS, GATE_FEATURE_COLS 定数で特徴量名一元管理 |

---

## §10 次ステップ

### §10.1 データ蓄積 (最優先)

- **現状**: 3 日間 (Feb 13-15) のみ。最低 7 日間は必要
- **方法**: fill test 再開 (残高追加後) 、または dry-run による疑似 fill 蓄積
- **目標**: 1,000+ fill records で Ridge の安定性を検証

### §10.2 Skip gate ライブ統合

- `run_fill_test.py` に `SkipGate.load()` + `build_features_from_market_state()` を組み込み
- 判定ログの記録 (predicted_pnl, actual_pnl の対比)
- A/B テスト: gate ON vs OFF の PnL 比較

### §10.3 054# 新規フィールドの活用

054# で `MicrostructureManager` に追加した `orderbook_imbalance`, 
`mid_price_trend_5s` が実データとして蓄積され次第、特徴量に追加可能。
これらは fill record に直接保存されるため、raw data マッチングの精度問題を回避できる。

### §10.4 閾値チューニング

- `threshold_bps = 0`: 現在は PnL < 0 でスキップ
- Grid search: threshold = [-1, -0.5, 0, 0.5, 1, 2] で PnL/機会損失の Pareto フロンティア探索
- max_skip_rate のチューニング: 0.5 ~ 0.9 の範囲

### §10.5 モデル改善

- **オンライン学習**: Ridge の逐次更新 (SGDRegressor + partial_fit)
- **特徴量追加**: medium parquet の RSI/ATR を fill timestamp にマッピング (秒間隔で最近傍)
- **アンサンブル**: Ridge + ElasticNet の平均予測

---

## §11 Codex レビュー依頼事項

以下のコードファイルを中心にレビューを依頼:

1. **`scripts/v460/ml/feature_enricher.py`** (448行)
   - マイクロストラクチャ特徴量の計算ロジックは正しいか
   - binary search (板マッチング) のエッジケース処理
   - NaN 処理 (median 補完) は適切か

2. **`scripts/v460/ml/skip_gate.py`** (375行)
   - Ridge PnL 回帰の仕様は妥当か
   - レート制限 (max_skip_rate) の実装
   - pickle シリアライズのセキュリティ (信頼済みモデルのみ)

3. **`tests/unit/v460/test_enricher_skip_gate.py`** (511行)
   - テストカバレッジは十分か
   - 見落としているエッジケースはあるか

4. **設計判断の検証**
   - AS 二値分類 → PnL 連続回帰への方針転換は妥当か
   - Ridge α=10.0 の根拠 (default に近い値、グリッドサーチ未実施)
   - 30 秒 PnL ウィンドウの適切性
   - 373 サンプルでの Ridge の統計的有意性
