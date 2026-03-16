# 001# 技術仕様: データ契約・アーキテクチャ・特徴量候補・実験基盤

| 項目 | 内容 |
|------|------|
| 対象 | 000# §4 の詳細展開、および 000# から委譲された技術仕様 |
| 依拠 | 000# §3–§4 / v459 119# §3–§5 / v459 116# §16–§17 |

---

## §1 データ契約

### §1.1 データソース

| ソース | 取得方法 | 粒度 | 用途 |
|--------|---------|------|------|
| **OHLCV** | 取引所 REST API / 既存 Parquet | 1min | ベースライン特徴量、学習環境の価格参照 |
| **板情報 (Orderbook)** | 取引所 REST / WebSocket | tick (数百ms) | マイクロストラクチャ特徴量 |
| **約定フロー (Trades)** | 取引所 REST / WebSocket | tick | 売買圧力・VWAP 計算 |

対象取引所: Coincheck（主）/ Bitflyer / Zaif。API 品質・流動性に応じて切替可能。Zaif は API 反応品質の低下により優先度を下げた経緯がある。

**二層保存方針**: 板情報・約定フローは **tick raw（REST ポーリング / WebSocket）を一次保存** し、学習用 **1 分集約は二次生成** する。一次保存により、集約粒度の事後変更やキュー変化・瞬間不均衡等の HFT 向け情報を保持できる。

```
data/v460/
├── raw/                    # 一次保存 (tick raw)
│   ├── orderbook/          # OrderBookSnapshot JSONL (gzip)
│   └── trades/             # TradeRecord JSONL (gzip)
└── features/               # 二次生成 (1分集約 Parquet)
    └── btc_jpy_1m_v460_features.parquet
```

### §1.2 データスキーマ

#### OHLCV (既存)

```
timestamp, open, high, low, close, volume
```

- 既存 Parquet: `data/btc_jpy_1m_v451_optimized_features.parquet`（1,216,930 行）
- v460 用: `data/btc_jpy_1m_v460_features.parquet`（OHLCV + マイクロストラクチャ派生列を追加）

#### Orderbook snapshot

```
timestamp, bid_price_1..N, bid_size_1..N, ask_price_1..N, ask_size_1..N
```

- 板深度: 上位 N=10 レベルを記録
- **一次保存 (raw)**: REST ポーリング間隔 5–10 秒。JSONL (gzip) で全スナップショットを記録
- **二次生成 (1min)**: 各分の最終スナップショット + 分内統計（spread 変動幅, depth 変化率等）
- 保持期間: raw は最低 1 ヶ月、1min 集約は最低 3 ヶ月

#### Trades (約定履歴)

```
timestamp, price, amount, side (buy/sell)
```

- **一次保存 (raw)**: 全約定を JSONL (gzip) で記録。REST ポーリング + lastID 追跡
- **二次生成 (1min)**: buy_volume, sell_volume, trade_count, vwap を算出
- 保持期間: raw は最低 1 ヶ月、1min 集約は OHLCV と同一

### §1.3 欠損処理

| ケース | 処理 |
|--------|------|
| OHLCV 欠損 (夜間等) | forward fill (最大 5 分)。5 分超は前値 fill + `is_missing_ohlcv=1` フラグ設定 |
| 板情報取得失敗 | 直前スナップショットで fill。連続 3 分以上は `is_missing_book=1` フラグ設定 |
| 約定ゼロ (閑散時) | volume=0, vwap=前値 fill。`is_thin_market=1` フラグ設定 |

**方針**: 行除外ではなく**マスク + フラグ特徴量**方式を採用する。欠損フラグ列自体を特徴量として学習器に渡し、閑散相場を系統的に捨てることを防ぐ。学習時の損失計算ではフラグ付き行を含めるが、評価指標算出時には `is_missing_*` 行を分離集計する。

### §1.4 データハッシュ

G0-data 準拠。学習に使用する Parquet ファイルの SHA-256 を manifest に記録。データ更新時はハッシュ変更により自動検知。

### §1.5 板情報・約定フロー取得計画

#### 既存実装の現状

| コンポーネント | パス | 現状 |
|-------------|------|------|
| `IBroker` | `ztb/trading/live/exchanges/base/broker_interfaces.py` | `get_current_price()` のみ。板取得 API なし |
| `BaseExchangeAdapter` | `ztb/trading/live/exchanges/base/adapter.py` | dry-run / rate limiter 基盤あり |
| `CoincheckAdapter` | `ztb/trading/live/exchanges/coincheck/adapter.py` | `/api/ticker` のみ実装。板・約定なし |
| `BitflyerAdapter` | `ztb/trading/live/exchanges/bitflyer/adapter.py` | ticker のみ。板・約定なし |

#### 改修計画

**Step 1: `IBroker` インターフェース拡張（互換レイヤ付き）**

`IBroker` への抽象メソッド追加は **破壊的変更** となる（`SimBroker`, `ZaifAdapter`, 各 Adapter が instantiation 不能になる）。これを回避するため、**デフォルト実装付きメソッド**（`NotImplementedError` ではなく `MarketDataNotSupported` 例外）による段階移行を採用する。

```python
# broker_interfaces.py に追加
@dataclass
class OrderBookSnapshot:
    timestamp: float
    bids: list[tuple[float, float]]  # [(price, size), ...] 降順
    asks: list[tuple[float, float]]  # [(price, size), ...] 昇順
    exchange: str

@dataclass
class TradeRecord:
    timestamp: float
    price: float
    amount: float
    side: str  # 'buy' or 'sell'

class MarketDataNotSupported(Exception):
    """Raised when adapter does not support market data collection."""

# IBroker に追加 — @abstractmethod ではなくデフォルト実装
async def get_orderbook(self, symbol: str, depth: int = 10) -> OrderBookSnapshot:
    """Get orderbook snapshot (top N levels).
    
    デフォルト実装は MarketDataNotSupported を送出。
    対応アダプタのみオーバーライドする。
    """
    raise MarketDataNotSupported(f"{self.__class__.__name__} does not support orderbook")

async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[TradeRecord]:
    """Get recent trades.
    
    デフォルト実装は MarketDataNotSupported を送出。
    対応アダプタのみオーバーライドする。
    """
    raise MarketDataNotSupported(f"{self.__class__.__name__} does not support trades")
```

**互換性**: `SimBroker`, `ZaifAdapter` 等はデフォルト実装を継承するため変更不要。`MarketDataCollector` は呼び出し前に `try/except MarketDataNotSupported` でケイパビリティ判定する。

**Step 2: 取引所アダプタ実装**

| 取引所 | 板 API | 約定 API | 備考 |
|---------|-------|---------|------|
| Coincheck | `GET /api/order_books` | `GET /api/trades?pair=btc_jpy` | Public API、認証不要 |
| Bitflyer | `GET /v1/board?product_code=BTC_JPY` | `GET /v1/executions?product_code=BTC_JPY` | Public API |

**Step 3: データ収集サービス**

```python
# ztb/data/market_data_collector.py (新規)
class MarketDataCollector:
    """tick raw 収集 + 1分集約の二層保存を行うデータ収集サービス。
    
    - raw: JSONL (gzip) で全 tick を保存 (data/v460/raw/)
    - aggregated: 1分集約を Parquet で保存 (data/v460/features/)
    
    NOTE: DataAcquisitionScheduler (ztb/data/scheduler.py) は Binance 専用実装のため
    流用せず、スケジュール思想（APScheduler cron パターン）のみ参考とする。
    実装は完全新規。
    """
    def __init__(self, adapter: IBroker, raw_dir: str, agg_dir: str,
                 poll_interval_sec: int = 5): ...
    async def collect_tick(self) -> tuple[OrderBookSnapshot, list[TradeRecord]]: ...
    async def run_continuous(self, duration_hours: int = 24): ...
    def aggregate_to_1min(self, raw_path: str, output_path: str): ...
    def save_raw_jsonl(self, data: dict, path: str): ...
```

**async/sync 整合性に関する注意**: 現状の `CoincheckAdapter`, `BitflyerAdapter` は `IBroker` の `async def` メソッドを宣言しながら、内部で同期 `requests` ライブラリを使用している。`MarketDataCollector` が高頻度ポーリング (5秒間隔) を行う場合、イベントループブロックが問題になる。対策として以下のいずれかを Phase 0 で適用する:

1. **推奨**: `httpx.AsyncClient` への移行（Coincheck/Bitflyer 両方）
2. **暫定**: `asyncio.to_thread()` で同期 `requests` 呼び出しを executor に隔離

**シンボル名の正規化**: Coincheck は `btc_jpy`（小文字）、Bitflyer は `BTC_JPY`（大文字）と API 仕様が異なる。`IBroker` レベルで内部シンボルを小文字に正規化し、各 Adapter が API 呼び出し時に変換する設計とする。

**実装順序**: Step 1 (IBroker 拡張 + 互換レイヤ, 1 日) → Step 2 (Coincheck 板 API + httpx 移行, 1.5 日) → Step 3 (Collector + 二層保存, 1.5 日) → 3 日間収集テスト → G0-data 判定

---

## §2 マイクロストラクチャ特徴量候補

### §2.1 v459 特徴量との対比

| 観点 | v459 (OHLCV 派生) | v460 (マイクロストラクチャ) |
|------|-------------------|---------------------------|
| 情報ソース | 価格のみ (OHLCV) | 価格 + 板 + 約定フロー |
| 特徴量数 | 8 (RSI×7 + ReturnStdDev) | 8–12 (候補から選定) |
| K2 結果 | IC = 0.000 (OOS) | **未検証 → G1-info で判定** |
| 理論的根拠 | テクニカル指標 (遅行) | 需給不均衡 (先行的) |

### §2.2 候補一覧

| # | 特徴量 | 算出方法 | 期待される情報 |
|---|--------|---------|---------------|
| 1 | **bid_ask_spread** | (best_ask - best_bid) / mid_price | 流動性・ボラティリティの即時指標 |
| 2 | **depth_imbalance** | (bid_volume_top5 - ask_volume_top5) / (bid + ask) | 売買圧力の方向。[-1, +1] |
| 3 | **trade_flow_imbalance** | (buy_volume - sell_volume) / total_volume | 実際の売買の方向性 |
| 4 | **vwap_deviation** | (close - vwap) / close | VWAP からの乖離。回帰圧力 |
| 5 | **trade_intensity** | trade_count / (前 N 分の平均 trade_count) | 取引活性度の変化率 |
| 6 | **order_flow_toxicity** | VPIN (Volume-synchronized PIN) 近似 | 情報トレーダーの存在推定 |
| 7 | **price_impact** | Δprice / trade_volume の移動平均 | 板の厚さ・レジリエンス |
| 8 | **micro_return_vol** | 1min リターンの直近 N 分 stdev | 短期ボラティリティ |
| 9 | **bid_depth_slope** | 板の depth-price 勾配 (bid 側) | 板の形状・サポート強度 |
| 10 | **ask_depth_slope** | 板の depth-price 勾配 (ask 側) | 板の形状・レジスタンス強度 |

### §2.3 選定方針

- G1-info (Phase 1) で全候補を XGBoost に投入し、IC > 0.02 の horizon を特定
- feature importance で上位 6–8 を採用
- 採用後にも定期的に importance を再検証（特徴量の劣化検知）

---

## §3 アーキテクチャ

### §3.1 システム構成

```
データ取得層
├── 取引所 REST API (OHLCV, 板スナップショット)
├── 取引所 WebSocket (リアルタイム板・約定)
└── data/v460/ (Parquet 保存)

特徴量生成層
├── ztb/features/ (既存基盤を拡張)
└── configs/v460/base.yaml (特徴量定義)

学習・評価層
├── scripts/v460/run_experiment.py (唯一のランナー)
├── scripts/v460/lib/ (データ読込・評価・manifest)
├── configs/v460/experiments/*.yaml (実験定義)
└── results/v460/ (結果 + manifest.jsonl)

執行層
├── ztb/trading/ (既存基盤)
├── ztb/utils/fee_model.py (ExchangeFeeModel: 取引所別手数料)
└── Paper Trader → Live Trader
```

### §3.2 既存コード活用

| 既存モジュール | パス | v460 での用途 |
|--------------|------|-------------|
| SACTrainer | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | G2-train 以降で使用 |
| FixedFeeModel / ExchangeFeeModel | `ztb/utils/fee_model.py` | 取引所別手数料設定。Coincheck maker 0% がデフォルト |
| p_mean_method | `ztb/metrics/metrics.py` | p平均法による統合 p 値判定。G1/G2/G3 の Gate 検定で使用 |
| StatisticalValidator | `ztb/metrics/statistical_validator.py` | Holm-Bonferroni 補正 |
| IBroker / BaseExchangeAdapter | `ztb/trading/live/exchanges/base/` | 取引所抽象化層。板/約定 API 拡張のベース |
| CoincheckAdapter | `ztb/trading/live/exchanges/coincheck/adapter.py` | 板/約定 API を追加実装予定 |
| DataScheduler | `ztb/data/scheduler.py` | スケジュール思想のみ参考（実体は Binance 専用。実装は流用せず新規） |
| Circuit Breaker | `ztb/trading/production/circuit_breaker.py` | G4-live で使用 |
| Paper Trader | `ztb/trading/live/simulation/paper_trader.py` | G4-live で使用 |
| Walk-Forward | `ztb/evaluation/walk_forward/` | G1-info の評価基盤として参考 |

### §3.3 新規実装

| モジュール | 配置先 | 目的 |
|-----------|-------|------|
| Orderbook collector | `ztb/data/market_data_collector.py` | tick raw 収集 + 1分集約の二層保存。既存 IBroker を活用 |
| Microstructure features | `ztb/features/microstructure.py` | §2 の特徴量算出 |
| Run manifest | `scripts/v460/lib/manifest.py` | 実験再現性の自動記録 |
| Config merger | `scripts/v460/lib/config_loader.py` | base.yaml + experiment.yaml のマージ |

---

## §4 実験基盤

### §4.1 ランナー設計

```
scripts/v460/
├── run_experiment.py      # 唯一のランナー (orchestrator 専任)
├── run_gate_check.py      # Gate 閾値照合ユーティリティ
└── lib/
    ├── config_loader.py   # YAML マージ、スキーマバリデーション
    ├── data_loader.py     # Parquet 読込、train/eval 分割
    ├── evaluator.py       # IC / accuracy / PF / Sharpe 算出
    └── manifest.py        # JSONL 自動記録
```

ランナーの責務: **orchestrator 専任**（config 読込 → task ディスパッチ → 結果保存）。task (feature_info / sac_train / backtest) のロジックは lib/ に委譲。行数制約ではなく、「orchestratorがビジネスロジックを持たない」が守るべき境界。task 追加によるランナー肥大化は lib/ への委譲で解消する。

### §4.2 Config 設計

```
configs/v460/
├── base.yaml              # 全実験の共通ベース
├── experiments/            # 実験差分 (override のみ)
│   ├── g1_xgb_h1_direction.yaml
│   ├── g1_xgb_h5_direction.yaml
│   └── ...
└── gate_thresholds.yaml   # 000# §3 の実装表現
```

**Override 規則**:
1. 実験 YAML は `_base` で base.yaml を指定
2. 記載したキーのみ上書き、他は base を継承
3. `features` と `train_end_index` は必須（base で null のため、未指定はエラー）
4. `_gate` フィールドで対応 Gate を明示

### §4.3 Manifest スキーマ

```jsonl
{
  "run_id": "v460_g1_xgb_h5_20260214_143022",
  "config_path": "configs/v460/experiments/g1_xgb_h5_direction.yaml",
  "config_hash": "sha256:...",
  "data_hash": "sha256:...",
  "git_commit": "abc1234",
  "gate": "G1-info",
  "seed": 42,
  "python_version": "3.11.9",
  "deps_hash": "sha256:...",
  "cuda_version": "12.1",
  "started_at": "2026-02-14T14:30:22+09:00",
  "finished_at": "2026-02-14T15:12:45+09:00",
  "status": "completed",
  "metrics": {"ic_h5": 0.031, "accuracy": 0.523},
  "gate_result": "PASS",
  "artifacts": ["results/v460/g1_xgb_h5_seed42.json"]
}
```

- `deps_hash`: `pip freeze` 出力の SHA-256。依存関係の再現性を保証
- `python_version`: 実行時の Python バージョン (sys.version)
- `cuda_version`: GPU 学習時の CUDA バージョン。CPU のみの場合は null

保存先: `results/v460/manifest.jsonl`（追記専用）。`run_experiment.py` が開始・終了時に自動記録。

---

## §5 統計検定コード参照

000# §3.7 の実装。v459 00# §5.6 踏襲に加え、G1 multi-target 判定用の Holm-Bonferroni 補正と p平均法を統合:

### §5.1 Holm-Bonferroni + Cliff's Delta

```python
from scipy.stats import mannwhitneyu


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's Delta effect size."""
    n1, n2 = len(x), len(y)
    dominance = sum(
        (1 if a > b else -1 if a < b else 0)
        for a in x for b in y
    )
    return dominance / (n1 * n2)


def compare_single(
    model: list[float], baseline: list[float],
) -> tuple[float, float]:
    """単一比較: Mann-Whitney U p値 + Cliff's Delta."""
    _, p = mannwhitneyu(model, baseline, alternative="greater")
    return p, cliffs_delta(model, baseline)


def holm_bonferroni_gate(
    results: dict[str, tuple[list[float], list[float]]],
    alpha: float = 0.05,
    min_effect: float = 0.33,
) -> dict[str, dict]:
    """G1 multi-target 判定: Holm-Bonferroni 補正付き.

    Args:
        results: {target_name: (model_scores, baseline_scores)}
        alpha: family-wise error rate
        min_effect: Cliff's Delta 最小閾値

    Returns:
        {target_name: {"pass": bool, "p_raw": float, "p_holm": float, "d": float}}
    """
    # 1. 全 target の raw p + effect size
    raw = {k: compare_single(*v) for k, v in results.items()}

    # 2. p 昇順ソート → Holm 補正
    sorted_keys = sorted(raw, key=lambda k: raw[k][0])
    m = len(sorted_keys)
    out: dict[str, dict] = {}
    gate_open = True  # Holm: 最初の非棄却で以降すべて非棄却

    for rank, key in enumerate(sorted_keys):
        p_raw, d = raw[key]
        holm_alpha = alpha / (m - rank)
        rejected = gate_open and (p_raw < holm_alpha) and (abs(d) > min_effect)
        if not (p_raw < holm_alpha):
            gate_open = False
        out[key] = {
            "pass": rejected,
            "p_raw": round(p_raw, 6),
            "p_holm": round(min(p_raw * (m - rank), 1.0), 6),
            "d": round(d, 4),
        }

    return out
```

**使用例** (G1 判定):

```python
# 9 組合せの結果を渡す
results = {
    "h1_direction": (model_ic_h1, baseline_ic_h1),
    "h5_direction": (model_ic_h5, baseline_ic_h5),
    # ... 他 7 組合せ
}
gate = holm_bonferroni_gate(results)
g1_pass = any(v["pass"] for v in gate.values())
```

### §5.2 p平均法（既存実装活用）

既存 `ztb.metrics.metrics.p_mean_method()` を Gate 判定に統合。複数 fold/seed の p 値を幾何平均で統合する。

```python
from ztb.metrics.metrics import p_mean_method


def p_mean_gate(
    fold_p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """p平均法による Gate 判定.

    データを N 分割し各分割で Mann-Whitney U を実行、
    得られた p 値の幾何平均で統合判定する。
    richmanbtc 氏のオリジナル手法。

    Args:
        fold_p_values: 各 fold の Mann-Whitney U p 値
        alpha: 有意水準

    Returns:
        {"p_geometric": float, "p_arithmetic": float, "pass": bool}

    参照:
        - docs/guides/P_MEAN_METHOD_README.md
        - docs/v459/24_phase3_specification.md §2.1.2
    """
    p_geo = p_mean_method(fold_p_values, method="geometric")
    p_arith = p_mean_method(fold_p_values, method="arithmetic")
    return {
        "p_geometric": round(p_geo, 6),
        "p_arithmetic": round(p_arith, 6),
        "n_folds": len(fold_p_values),
        "pass": p_geo < alpha,
    }
```

**Gate 判定での併用**: Holm-Bonferroni (§5.1) と p平均法 (§5.2) の両方を実行し、**両判定の AND** で Gate 通過とする。Holm は多重比較補正、p平均法は fold 間安定性の保証という異なる側面を担保する。

### §5.3 G1 判定アルゴリズム（厳密仕様）

G1 判定は以下の 3 ステップで行う。実装者による解釈差を排除する。

```
Step A: target 単位の fold/seed p値統合
  • 9 target (× 5 folds) それぞれについて、fold 別 Mann-Whitney U p値を算出
  • 各 target の fold p値を p_mean_gate() で幾何平均に統合 → target 単位の統合 p値 (p_geo_t)

Step B: Holm-Bonferroni 補正
  • 9 個の p_geo_t を Holm-Bonferroni 補正 (family = 9)
  • 補正後 p値 < α かつ Cliff's Delta |d| > 0.33 の target を PASS 候補とする

Step C: AND 判定
  • Step B で PASS 候補が 1 つ以上あり、
    かつ Step A の p_mean_gate も PASS (幾何平均 p < α) である target が
    1 つ以上存在する場合にのみ G1 PASS

判定式:
  G1_PASS = ∃ target: (holm_pass(target) ∧ pmean_pass(target) ∧ |d(target)| > 0.33)
```

```python
def g1_judgment(
    fold_results: dict[str, list[tuple[list[float], list[float]]]],
    alpha: float = 0.05,
    min_effect: float = 0.33,
) -> dict:
    """G1 判定: p-mean (target単位) → Holm 補正 → AND.

    Args:
        fold_results: {target_name: [(model_fold1, baseline_fold1), ...]}
    """
    from scipy.stats import mannwhitneyu
    from ztb.metrics.metrics import p_mean_method

    # Step A: target 単位で fold p値を統合
    target_p_geo: dict[str, float] = {}
    target_effects: dict[str, float] = {}
    target_pmean_pass: dict[str, bool] = {}
    for tgt, folds in fold_results.items():
        fold_ps = []
        all_model, all_baseline = [], []
        for model_scores, baseline_scores in folds:
            _, p = mannwhitneyu(model_scores, baseline_scores, alternative="greater")
            fold_ps.append(p)
            all_model.extend(model_scores)
            all_baseline.extend(baseline_scores)
        p_geo = p_mean_method(fold_ps, method="geometric")
        target_p_geo[tgt] = p_geo
        target_effects[tgt] = cliffs_delta(all_model, all_baseline)
        target_pmean_pass[tgt] = p_geo < alpha

    # Step B: Holm-Bonferroni 補正
    sorted_targets = sorted(target_p_geo, key=lambda t: target_p_geo[t])
    m = len(sorted_targets)
    holm_pass: dict[str, bool] = {}
    gate_open = True
    for rank, tgt in enumerate(sorted_targets):
        holm_alpha = alpha / (m - rank)
        rejected = gate_open and (target_p_geo[tgt] < holm_alpha)
        if not (target_p_geo[tgt] < holm_alpha):
            gate_open = False
        holm_pass[tgt] = rejected

    # Step C: AND 判定
    passed_targets = [
        tgt for tgt in sorted_targets
        if holm_pass[tgt] and target_pmean_pass[tgt]
        and abs(target_effects[tgt]) > min_effect
    ]
    return {
        "g1_pass": len(passed_targets) > 0,
        "passed_targets": passed_targets,
        "details": {
            tgt: {
                "p_geo": round(target_p_geo[tgt], 6),
                "pmean_pass": target_pmean_pass[tgt],
                "holm_pass": holm_pass[tgt],
                "cliff_d": round(target_effects[tgt], 4),
            }
            for tgt in sorted_targets
        },
    }
```

---

## §6 実装ロードマップ

000# / 001# の全仕様を実現するために必要な実装作業を、既存コードとのギャップ分析に基づき整理する。

### §6.1 ギャップ分析サマリ

| カテゴリ | そのまま使える | 要拡張 | 新規作成 |
|---------|--------------|-------|---------|
| データ収集層 | — | IBroker, CoincheckAdapter, BitflyerAdapter, DataScheduler | market_data_collector.py |
| 特徴量生成 | hft_proxies.py (OHLCV proxy) | — | microstructure.py |
| 実験基盤 | — | run_manifest.py (JSONL化) | scripts/v460/ 一式, configs/v460/ |
| 統計 Gate | p_mean_method(), StatisticalValidator | StatisticalValidator (Holm明示化) | holm_bonferroni_gate(), p_mean_gate() |
| Walk-Forward | WalkForwardSplitter | — | XGBoost Walk-Forward 評価器 (K2 流用) |
| 執行層 | CircuitBreaker, PaperTrader | — | — |

### §6.2 Phase 別実装計画

#### Phase 0 (データ取得基盤・仕様確定) — G0-data まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P0-1 | `IBroker` に `OrderBookSnapshot`, `TradeRecord`, `get_orderbook()`, `get_recent_trades()` 追加。互換レイヤ（デフォルト実装 + `MarketDataNotSupported`）で既存 `SimBroker`/`ZaifAdapter` を破壊しない。`httpx.AsyncClient` への移行も含む | `ztb/trading/live/exchanges/base/broker_interfaces.py`, `coincheck/adapter.py`, `bitflyer/adapter.py` | — | 1d |
| P0-2 | `CoincheckAdapter` に板 (`/api/order_books`) ・約定 (`/api/trades`) API 実装 | `ztb/trading/live/exchanges/coincheck/adapter.py` | P0-1 | 1d |
| P0-3 | `BitflyerAdapter` に板 (`/v1/board`) ・約定 (`/v1/executions`) API 実装 | `ztb/trading/live/exchanges/bitflyer/adapter.py` | P0-1 | 1d |
| P0-4 | `MarketDataCollector` 新規作成。tick raw 収集 (5–10秒間隔) + 1分集約の二層保存。スケジューラは完全新規（`DataAcquisitionScheduler` は Binance 専用のため流用せず、思想のみ参考） | `ztb/data/market_data_collector.py` | P0-1,2 | 1.5d |
| P0-5 | configs/v460/ 初期構造作成 (`base.yaml`, `gate_thresholds.yaml`) | `configs/v460/` | — | 0.5d |
| P0-6 | `scripts/v460/lib/manifest.py` — JSONL 追記型 manifest（既存 `run_manifest.py` をベースに拡張） | `scripts/v460/lib/manifest.py` | — | 0.5d |
| P0-7 | `scripts/v460/lib/config_loader.py` — base.yaml + experiment.yaml マージ | `scripts/v460/lib/config_loader.py` | P0-5 | 0.5d |
| P0-8 | `scripts/v460/lib/data_loader.py` — Parquet 読込・train/eval 分割 | `scripts/v460/lib/data_loader.py` | — | 0.5d |
| P0-9 | G0-data チェッカー（ハッシュ検証・NaN 比率・カラム数） | `scripts/v460/run_gate_check.py` (G0 部分) | P0-6 | 0.5d |
| P0-10 | 板データ 3 日間収集テスト → G0 判定 | 運用タスク | P0-4 | 3d (実時間) |

**Phase 0 合計**: 実装 ~7d + 収集テスト 3d

#### Phase 1 (特徴量情報量検証) — G1-info まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P1-1 | `microstructure.py` — §2 の 10 候補特徴量算出モジュール | `ztb/features/microstructure.py` | P0-10 (データ) | 1.5d |
| P1-2 | XGBoost Walk-Forward 評価器 — v459 `run_k2_nonrl_upper_bound.py` のライブラリ化 | `scripts/v460/lib/evaluator.py` | P1-1 | 1d |
| P1-3 | `holm_bonferroni_gate()` 実装 | `ztb/metrics/gate_checks.py` (新規) | — | 0.5d |
| P1-4 | `p_mean_gate()` 実装（既存 `p_mean_method()` 利用） | `ztb/metrics/gate_checks.py` | P1-3 | 0.3d |
| P1-5 | `run_experiment.py` — 唯一のランナー (task: feature_info) | `scripts/v460/run_experiment.py` | P1-2, P0-7 | 1d |
| P1-6 | G1 実験 YAML 作成 (3 horizon × 3 target = 9 組合せ) | `configs/v460/experiments/g1_*.yaml` | P0-5 | 0.5d |
| P1-7 | G1 判定: target 単位で fold p値を p-mean 幾何平均統合 → Holm 補正 → AND 判定（§5.3 準拠） | `scripts/v460/run_gate_check.py` (G1 部分) | P1-3,4,5 | 0.5d |

**Phase 1 合計**: ~5.3d

#### Phase 2 (maker 執行可能性検証) — G1.1-exec まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P2-1 | maker 注文発注・fill rate 計測スクリプト | `scripts/v460/run_fill_test.py` | P0-2 | 1d |
| P2-2 | post_fill_30s_pnl / adverse_selection 計算ロジック | `ztb/metrics/fill_quality.py` (新規) | P2-1 | 1d |
| P2-3 | 1 週間 fill rate 実測 → G1.1 判定 | 運用タスク | P2-1,2 | 7d (実時間) |

**Phase 2 合計**: 実装 ~2d + 実測 7d（Phase 1 と並行可能）

#### Phase 3 (SAC 学習安定性検証) — G2-train まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P3-1 | `run_experiment.py` に task: sac_train 追加 | `scripts/v460/run_experiment.py` | G1 PASS | 1d |
| P3-2 | 4-seed 並列訓練 YAML + worst-seed 判定 | `configs/v460/experiments/g2_*.yaml` | P3-1 | 0.5d |
| P3-3 | G2 判定 (seed 比率 + IC 分散 + worst-seed ROI) | `scripts/v460/run_gate_check.py` (G2 部分) | P3-1 | 0.5d |

**Phase 3 合計**: ~2d

#### Phase 4 (収益性検証) — G3-pnl まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P4-1 | `run_experiment.py` に task: backtest 追加 | `scripts/v460/run_experiment.py` | G2 PASS | 0.5d |
| P4-2 | PF/Sharpe/DD 評価 (ExchangeFeeModel 込み) | `scripts/v460/lib/evaluator.py` (拡張) | P4-1 | 0.5d |
| P4-3 | G3 判定 (median + worst-seed PF/Sharpe/DD) | `scripts/v460/run_gate_check.py` (G3 部分) | P4-2 | 0.5d |

**Phase 4 合計**: ~1.5d

#### Phase 5 (Paper trading) — G4-live まで

| # | タスク | 対象ファイル | 依存 | 工数 |
|---|-------|------------|------|------|
| P5-1 | 既存 `PaperTrader` にマイクロストラクチャ特徴量フィード統合 | `ztb/trading/live/simulation/paper_trader.py` | G3 PASS | 1d |
| P5-2 | CircuitBreaker テスト (手動) | 運用タスク | P5-1 | 0.5d |
| P5-3 | 7 日間 paper trading → G4 判定 | 運用タスク | P5-1,2 | 7d (実時間) |

**Phase 5 合計**: 実装 ~1.5d + 運用 7d

### §6.3 クリティカルパス

```
P0-1 ─→ P0-2 ─→ P0-4 ─→ P0-10 (3d収集) ─→ P1-1 ─→ P1-2 ─→ P1-5 ─→ P1-7 (G1判定)
                                                                              │
                                                                              ↓ G1 PASS
P0-2 ─→ P2-1 ─→ P2-3 (7d実測,並行) ──────────────────────────────→ G1.1判定
                                                                              │
                                                                              ↓ G1+G1.1 PASS
                                                              P3-1 ─→ P3-3 (G2判定)
                                                                              │
                                                                              ↓ G2 PASS
                                                              P4-1 ─→ P4-3 (G3判定)
                                                                              │
                                                                              ↓ G3 PASS
                                                              P5-1 ─→ P5-3 (G4判定)
```

**最短経路**: ~13.3d 実装 + 10d 実時間（P0 収集 3d + G1.1 実測 7d を並行）+ 7d Paper

### §6.4 既存コード流用一覧

| 既存モジュール | 流用方法 | 変更度 |
|-------------|---------|-------|
| `run_k2_nonrl_upper_bound.py` (373行) | `walk_forward_eval()` を `evaluator.py` にライブラリ化。XGBoost + blocked time split 5 窓のロジックをそのまま抽出 | 中 (リファクタ) |
| `run_manifest.py` (448行) | `generate_manifest()` のフィールド定義を流用。出力を JSON → JSONL に変更、`deps_hash`/`python_version`/`cuda_version` フィールド追加 | 小 (拡張) |
| `DataAcquisitionScheduler` | Binance 専用実装のためコードは流用せず、APScheduler cron パターンの**思想のみ参考**。`MarketDataCollector` は完全新規 | 小 (参考のみ) |
| `StatisticalValidator` | `_apply_multiple_testing_correction()` は statsmodels `multipletests(method="holm")` で Holm 対応可。`gate_checks.py` から呼ぶか、validator 自体を拡張 | 小 (設定変更) |
| `hft_proxies.py` (82行) | `add_hft_features()` の CLV/vol_pressure/impact_proxy は OHLCV proxy として `microstructure.py` に統合可能。ただしリアル板データ版に置換予定 | 中 (段階的置換) |

### §6.5 テスト方針

| Phase | テスト対象 | 手法 |
|-------|----------|------|
| P0 | IBroker 拡張、Adapter API 呼び出し、MarketDataCollector | pytest + mock (API) + 実 API 疎通テスト |
| P1 | microstructure.py 特徴量算出、evaluator.py IC/accuracy | pytest (計算結果の数値検証) |
| P1 | gate_checks.py (Holm, p-mean, g1_judgment) | pytest (既知の p 値での PASS/FAIL 判定) |
| P2 | fill_quality.py | pytest (模擬約定データ) |
| P3-P5 | 統合テスト | G0→G1→G2 の自動パイプライン実行確認 |

### §6.6 技術的負債の事前解消

002# レビューで指摘された既存コードベースの問題を、実装前に解消する。

| # | 対象 | 問題 | 対応 | タイミング |
|---|------|------|------|-----------|
| D1 | `ztb/metrics/metrics.py` | `p_mean_method()` が L1394 と L2069 で**二重定義**（後者が上書き、型ヒントも不一致） | 関数を一本化。L1394 の `Union[list[float], NDArray]` 版を残し、L2069 を削除 | P0 開始前 |
| D2 | `ztb/trading/live/simulation/paper_trader.py` | `_load_data_feed()` が合成ランダムウォーク (`np.random.seed(42)`, `base_price=30000`) で BTC/JPY に不適切 | G4 までに market replay 実データ固定パスを実装。合成は fallback のみ | P5-1 |
| D3 | Coincheck/Bitflyer adapters | `async def` メソッド内で同期 `requests` を使用（event loop ブロック） | `httpx.AsyncClient` へ移行。P0-1 と同時実施 | P0-1 |
| D4 | 各 Adapter | シンボル名不統一 (`btc_jpy` vs `BTC_JPY`) | IBroker レベルで内部シンボルを小文字正規化。Adapter が API 仕様に合わせて変換 | P0-1 |

---

## §7 バージョン枝番規則

v460.1 が必要になるケース:

| ケース | 例 |
|--------|----|
| 特徴量ソース分岐 | 板情報 (v460) vs 約定フローのみ (v460.1) で並行評価 |
| 取引所分岐 | Coincheck (v460) vs Bitflyer (v460.1) で並行評価 |
| 中間リリース | G3 PASS 後のスナップショットタグ |

**規則**: N は整数連番。再帰 (v460.1.1) 禁止。最大 2 本。3 本目が必要なら v461 を立てる。全 Gate FAIL → v460.1 ではなく v461 へ移行。

詳細: [v459/119# §8.3](../v459/119_v460_launch_integrated_policy.md)
