# 009# Phase 2 計画: G1.1-exec (Maker 執行可能性検証)

|  | 値 |
|---|----|
| Doc ID | 009 |
| Phase | ph2 |
| Type | plan |
| 前提 | 000# §3.3, 001# §6.2 Phase 2, 005# §5, 006# §4, 008# |
| 状態 | DRAFT |
| Date | 2026-02-13 |

---

## §1 目的と位置づけ

### §1.1 G1.1-exec の目的

maker-only 戦略の大前提 — **実際に注文が約定するか** — を取引所 API 実測で検証する。

v460 の収益モデルは maker 手数料 0% に依存するため、fill rate が十分でなければ戦略自体が成立しない。G1.1 はこの **執行実現性リスク** (000# §6 ⭐⭐⭐) を定量化するゲートである。

### §1.2 G1 との関係

| 項目 | G1-info | G1.1-exec |
|------|---------|-----------|
| Phase | ph1 | ph2 |
| 検証対象 | 情報量 (特徴量の予測力) | 執行品質 (maker 約定特性) |
| 結果 | **FAIL** (cliff_d 未達) | 未実施 |
| 依存 | G0 PASS | P0-2 (CoincheckAdapter) |
| 並行可否 | — | **G1 と並行可** (001# §6.3) |

005# §5.1 の判定で **選択肢C (G1.1-exec 並行 + 実データ収集)** が推奨されており、006# §4 でも「約定品質（fill/adverse selection）は独立価値がある」と確認済み。

**G1 FAIL でも G1.1 は独立に実行可能かつ有用。** 理由:

1. **maker 約定特性のデータ蓄積** — G1 再検証 (実板データ由来の特徴量) に必要な raw data の収集基盤になる
2. **戦略クラス判定** — fill rate / adverse selection の結果次第で maker-only を維持するか taker 併用に切り替えるかを判断
3. **G2 以降のブロッキング解消** — 001# §6.3 のクリティカルパスでは G1 + G1.1 の **両方** が PASS して初めて G2 に進む

---

## §2 Gate 仕様 (000# §3.3 準拠)

### §2.1 判定基準

**116# 改訂**: 二段階ゲート化により、旧 E1-E5 は G1.1-quick (K1-K6) + G1.2-full (F1-F8) に再編。
詳細は 000# §3.3 (改訂版) を参照。

#### 旧基準 (参考: SkipGate 導入前)

| # | 指標 | 旧閾値 | 備考 |
|---|------|------|------|
| E1 | fill rate (90th percentile) | ≥ 90% | → K1/F1 (attempted ベース 60%/70%) に置換 |
| E2 | cancel ratio | ≤ 30% | → K2/F2 (attempted ベース) に置換 |
| E3 | queue wait time (median) | ≤ 60 sec | → K3/F3 に維持 |
| E4 | post_fill_30s_pnl (mean) | ≥ 0 (p<0.05) | → K4 (複合条件) / F4 に再設計 |
| E5 | adverse selection ratio | ≤ 20% | → F5 (30%) に緩和 |

#### 新基準 (116#)

G1.1-quick (72h Kill) / G1.2-full (168h Qualification) の詳細は **000# §3.3** に一元化。

### §2.2 gate_thresholds.yaml

```yaml
# 旧 (後方互換維持)
g1_1_exec:
  min_fill_rate_p90: 0.85
  max_cancel_ratio: 0.30
  ...

# 新 (116#)
g1_1_quick_exec:
  min_attempted_fill_rate: 0.60
  max_attempted_cancel_ratio: 0.40
  max_queue_wait_median_sec: 120
  pnl_kill_p_threshold: 0.02
  pnl_kill_mean_threshold: -0.8
  max_cumulative_loss_jpy: 10000
  max_skip_gate_ratio: 0.25

g1_2_full_exec:
  min_attempted_fill_rate: 0.70
  min_overall_fill_rate: 0.62
  max_attempted_cancel_ratio: 0.30
  max_queue_wait_median_sec: 60
  pnl_alpha: 0.05
  max_adverse_selection_ratio: 0.30
  max_skip_gate_ratio: 0.20
  min_calendar_days: 7
  min_attempted_samples: 500
```

### §2.3 FAIL 時のアクション (000# §3.3)

- G1.1-quick FAIL → fill_test 即時停止。戦略クラスの変更を検討 (aggressive maker, IOC 併用等)
- G1.2-full FAIL → 個別指標により対応分岐 (000# §3.3 参照)
- G1.2 FAIL → G2 には進まない (001# §6.3)

### §2.4 統計的補足

**Kill Gate (K4)** は複合条件 (115# 反映):
- `p < 0.02` **かつ** `mean ≤ -0.8 bps` の同時成立で FAIL
- Watch 層: `p < 0.05` かつ `mean < -0.3 bps` → パラメータ凍結推奨

**Qualification Gate (F4)** は原初 E4 を維持:
- mean ≥ 0 → PASS
- mean < 0 かつ片側 t 検定 p ≥ 0.05 → PASS (統計的に有意でない)
- mean < 0 かつ片側 t 検定 p < 0.05 → FAIL (系統的な逆選択)

---

## §3 既存インフラ棚卸し

### §3.1 利用可能な既存実装

| コンポーネント | パス | 状態 | G1.1 での役割 |
|--------------|------|------|-------------|
| `IBroker` | `ztb/trading/live/exchanges/base/broker_interfaces.py` | ✅ 完成 | 抽象インターフェース (v460 で板/約定メソッド追加済み) |
| `Order` dataclass | 同上 | ✅ 完成 | 注文表現。`order_id`, `status`, `price`, `quantity` |
| `OrderBookSnapshot` | 同上 | ✅ 完成 | 板スナップショット (v460 新規追加) |
| `TradeRecord` | 同上 | ✅ 完成 | 約定記録 (v460 新規追加) |
| `CoincheckAdapter.place_order()` | `coincheck/adapter.py` L230-370 | ✅ Real API 対応 | maker limit 発注 |
| `CoincheckAdapter.cancel_order()` | 同上 L375-420 | ✅ Real API 対応 | 未約定キャンセル |
| `CoincheckAdapter.get_orderbook()` | 同上 L584-618 | ✅ Real API 対応 | 板スナップショット取得 (public) |
| `CoincheckAdapter.get_recent_trades()` | 同上 L619-659 | ✅ Real API 対応 | 直近約定取得 (public) |
| `CoincheckAdapter.get_balance()` | 同上 L458-527 | ✅ Real API 対応 | 残高確認 |
| `OrderState` / `OrderRecord` | `ztb/trading/live/order_state.py` | ✅ 完成 | 注文状態管理 |
| `MarketDataCollector` | `ztb/data/market_data_collector.py` | ✅ 完成 | tick raw 収集 + 1 分集約 |
| `gate_thresholds.yaml` | `configs/v460/gate_thresholds.yaml` | ✅ 定義済み | G1.1 閾値 |

### §3.2 不足しているコンポーネント

| # | コンポーネント | 計画パス | 状態 | 説明 |
|---|--------------|---------|------|------|
| M1 | `CoincheckAdapter.get_order_status()` (real) | `coincheck/adapter.py` | ⚠️ dry-run only | Real API 用に `GET /api/exchange/orders/opens` or 個別注文照会が必要 |
| M2 | `CoincheckAdapter.get_open_orders()` (real) | `coincheck/adapter.py` | ⚠️ dry-run only | 同上 |
| M3 | `run_fill_test.py` | `scripts/v460/run_fill_test.py` | ❌ 未作成 | P2-1: maker 注文発注・fill rate 計測スクリプト |
| M4 | `fill_quality.py` | `ztb/metrics/fill_quality.py` | ❌ 未作成 | P2-2: post_fill_30s_pnl / adverse_selection 計算 |
| M5 | G1.1 Gate judge | `scripts/v460/run_gate_check.py` (拡張) | ❌ 未実装 | G1.1 閾値照合ロジック |
| M6 | trade dedup 数値比較化 | `market_data_collector.py` | ⚠️ F8 残件 | 007# §6 F8 で G1.1 実装時に対応と決定 |

### §3.3 Coincheck API リアルモード課題

`CoincheckAdapter` は以下の API 呼出しにおいて real / dry-run の実装が非対称:

```
place_order()     : real ✅ / dry-run ✅
cancel_order()    : real ✅ / dry-run ✅  
get_order_status(): real ❌ / dry-run ✅  ← G1.1 で必要
get_open_orders() : real ❌ / dry-run ✅  ← G1.1 で必要
get_balance()     : real ✅ / dry-run ✅
get_orderbook()   : real ✅ (public)
get_recent_trades(): real ✅ (public)
get_current_price(): dry-run ✅ (but real raises NotImplementedError — 要修正)
```

---

## §4 実装計画

### §4.1 Phase 2 タスク一覧 (001# §6.2 準拠)

| # | タスク | 対象ファイル | 依存 | 工数 | 内容 |
|---|-------|------------|------|------|------|
| P2-0 | CoincheckAdapter real mode 補完 | `coincheck/adapter.py` | — | 0.5d | M1/M2: `get_order_status()`, `get_open_orders()`, `get_current_price()` の real mode 実装 |
| P2-1 | maker 注文発注・fill rate 計測スクリプト | `scripts/v460/run_fill_test.py` (新規) | P2-0 | 1d | fill test runner: 小額 maker limit → 監視 → metrics 記録 |
| P2-2 | fill quality 計算ロジック | `ztb/metrics/fill_quality.py` (新規) | P2-1 | 1d | E1-E5 指標計算 + G1.1 Gate 判定 |
| P2-2.1 | F8 trade dedup 修正 | `market_data_collector.py` | — | 0.2d | 007# §6 F8 残件 |
| P2-3 | run_gate_check.py G1.1 対応 | `run_gate_check.py` (拡張) | P2-2 | 0.3d | G1.1 閾値照合追加 |
| P2-4 | 単体テスト | `tests/unit/v460/test_fill_quality.py` (新規) | P2-2 | 0.5d | 模擬約定データでの指標検証 |
| P2-5 | 1 週間 fill rate 実測 | 運用タスク | P2-1,2 | 7d (実時間) | → G1.1 判定 |

**合計**: 実装 ~3.5d + 実測 7d

### §4.2 P2-1: run_fill_test.py 設計概要

```
scripts/v460/run_fill_test.py
├── FillTestRunner
│   ├── __init__(adapter, config)
│   ├── run_single_cycle()      # 1 サイクル: 発注 → 監視 → 結果記録
│   ├── run_continuous(hours)    # 連続実行 (指定時間)
│   └── save_results(path)      # JSONL 形式で保存
└── CLI entry point (argparse)
```

#### 1 サイクルの流れ

```
1. get_orderbook() → best bid/ask 取得
2. maker limit 発注 (best bid + 1 JPY / best ask - 1 JPY)
   - 発注数量: 最小ロット (0.001 BTC ≈ 15,000 JPY) 
   - ※ Coincheck BTC 最小注文: 0.001 BTC
3. 発注時刻 t_submit 記録
4. ポーリング監視 (5 秒間隔, 最大 300 秒)
   - get_order_status() で fill/cancel/open を確認
5. 約定 (filled):
   - t_fill = 約定時刻
   - queue_wait = t_fill - t_submit
   - 30 秒後: get_orderbook() → mid price → post_fill_30s_pnl 計算
6. 未約定 (timeout):
   - cancel_order() → cancel ratio にカウント
7. 結果を FillRecord に記録
```

#### FillRecord スキーマ

```python
@dataclass
class FillRecord:
    cycle_id: str
    timestamp: float         # t_submit
    side: str                # 'buy' or 'sell'
    order_price: float       # 発注価格
    order_quantity: float    # 発注数量
    fill_price: float | None # 約定価格 (未約定は None)
    filled: bool             # 約定したか
    cancelled: bool          # キャンセルしたか
    queue_wait_sec: float    # 発注→約定 (or cancel) の秒数
    mid_at_fill: float | None     # 約定時の mid price
    mid_30s_after: float | None   # 約定 30 秒後の mid price
    post_fill_30s_pnl: float | None  # 30 秒後 PnL (bps)
    adverse_selected: bool | None    # 30 秒後に逆行したか
```

#### 安全設計

| リスク | 対策 |
|-------|------|
| 資金リスク | 最小ロット (0.001 BTC) のみ使用。片側ポジション蓄積禁止 (buy→sell 交互) |
| API 制限 | RateLimiter (5 req/sec) 使用。サイクル間隔 ≥ 120 秒 |
| 異常時 | 残存注文の一括キャンセル (`atexit` + signal handler) |
| コスト | maker 0% 手数料。スプレッド損失は最小ロット × spread ≈ 数十円/サイクル |
| ポジションリーク | 各サイクル終了時にポジション確認、残があれば即時反対売買 |

### §4.3 P2-2: fill_quality.py 設計概要

```python
# ztb/metrics/fill_quality.py

def compute_fill_metrics(records: list[FillRecord]) -> FillMetrics:
    """FillRecord のリストから G1.1 Gate 指標を算出."""

def g1_1_judgment(metrics: FillMetrics, thresholds: dict) -> dict:
    """G1.1 Gate 合否判定."""
```

#### FillMetrics

```python
@dataclass
class FillMetrics:
    total_orders: int
    filled_orders: int
    cancelled_orders: int
    fill_rate_p90: float          # E1: 日別 fill rate の 90th percentile
    cancel_ratio: float           # E2: cancelled / total
    queue_wait_median_sec: float  # E3: filled orders の待ち時間中央値
    post_fill_30s_pnl_mean: float # E4: 30 秒後 PnL の平均 (bps)
    post_fill_30s_pnl_pvalue: float  # E4: 片側 t 検定 p 値
    adverse_selection_ratio: float   # E5: 逆行注文の割合
```

### §4.4 P2-3: 実測プロトコル

| 項目 | 値 |
|------|------|
| 取引所 | Coincheck (BTC/JPY) |
| 期間 | 7 日間 (連続) |
| サイクル間隔 | 120 秒 (最短) |
| 1 日あたりサイクル数 | ~720 (= 86400/120) |
| 7 日間合計 | ~5,040 サイクル |
| 発注数量 | 0.001 BTC / サイクル |
| 最大タイムアウト | 300 秒 / サイクル |
| buy/sell | 交互に発行 |
| データ保存 | `results/v460/fill_test/fill_records_YYYYMMDD.jsonl` |

---

## §5 実装順序とクリティカルパス

```
P2-0 (Adapter補完, 0.5d)
  │
  ├── P2-2.1 (F8 fix, 0.2d, 並行)
  │
  ↓
P2-1 (run_fill_test.py, 1d)
  │
  ↓
P2-2 (fill_quality.py, 1d)
  │
  ├── P2-3 (run_gate_check G1.1, 0.3d, P2-2 後)  
  ├── P2-4 (テスト, 0.5d, P2-2 後)
  │
  ↓
P2-5 (7d 実測, P2-1+P2-2 完了後)
  │
  ↓
G1.1 判定
```

**最短実装日数**: 3.5d (P2-0 → P2-1 → P2-2 → P2-3/P2-4)  
**実測期間**: 7d (P2-5)  
**G1.1 判定までの合計**: ~10.5d

---

## §6 リスク評価

### §6.1 G1.1 固有リスク

| # | リスク | 影響度 | 発生確率 | 対策 |
|---|-------|-------|---------|------|
| R1 | 板が薄く fill rate ≪ 90% | ⭐⭐⭐ | 中 | aggressive maker (mid-1) or IOC 併用を検討 (000# §3.3 FAIL action) |
| R2 | adverse selection が支配的 | ⭐⭐⭐ | 中 | fill しても負ける → maker-only 戦略自体の見直し |
| R3 | API 障害で実測期間が途切れる | ⭐⭐ | 低 | 自動リトライ + 日別集計でギャップ許容 |
| R4 | 最小ロットでも 7 日間でスプレッド損失が蓄積 | ⭐ | 低 | 推定: ~720 cycles/day × 7d × ~50 JPY/cycle ≈ 252,000 JPY (最大) |
| R5 | Coincheck API の rate limit 超過 | ⭐ | 低 | RateLimiter (5 req/sec) で制御済み |

### §6.2 R4 コスト推定の詳細

```
1 サイクルの最大コスト:
  maker 手数料: 0 JPY (maker 0%)
  スプレッド損失: bid-ask spread の半分 × 0.001 BTC
  BTC/JPY spread ≈ 500-2000 JPY → 損失 ≈ 0.25-1.0 JPY/cycle
  (※ best bid+1/best ask-1 で発注するため、市場スプレッドの内側)

7 日間推定:
  保守的: 5040 cycles × 1.0 JPY = 5,040 JPY
  楽観的: 5040 cycles × 0.1 JPY = 504 JPY
  
  fill しなかった注文は cancel → コスト 0
  fill した注文も maker 0% → 手数料 0
  実コストはスプレッド内側に置いた maker 注文の逆行分のみ
```

**結論**: 7 日間の fill test コストは推定 1,000〜5,000 JPY 程度で許容範囲。

---

## §7 G1 FAIL との整合性

### §7.1 000# ルールとの適合

000# §3.2 の G1 FAIL 時アクション:
> 「特徴量再設計へ戻る。RL には進まない。」

この制約は **G2-train (RL) への遷移禁止** を意味するが、G1.1 は RL とは独立した **執行品質検証** であるため、G1 FAIL 状態でも実行可能。001# §6.3 のクリティカルパスでも G1.1 は G1 と並行で配置されている。

### §7.2 G1.1 結果の利用シナリオ

| G1.1 結果 | G1 再検証 | 次のアクション |
|-----------|----------|-------------|
| **PASS** | 実板データで G1 再検証 → PASS | G2-train へ進む |
| **PASS** | 実板データで G1 再検証 → FAIL | 特徴量を根本的に再設計 (maker 断念ではなく情報源の問題) |
| **FAIL** (fill rate) | — | maker-only → aggressive maker or taker 併用 |
| **FAIL** (adverse selection) | — | maker 戦略自体を再検討。fill しても負ける問題 |

### §7.3 実板データ収集との相乗効果

G1.1 の fill test 実行中に蓄積される以下のデータは、G1 再検証の特徴量ソースとして再利用可能:

- `get_orderbook()` のスナップショット → `MarketDataCollector` と同じ形式
- 約定タイミングと板の状態 → microstructure 特徴量の検証データ
- post_fill 価格変動 → 方向予測モデルの教師データ候補

---

## §8 成果物一覧 (計画)

| # | 成果物 | パス | Phase |
|---|-------|------|-------|
| S1 | CoincheckAdapter real mode 補完 | `ztb/trading/live/exchanges/coincheck/adapter.py` | P2-0 |
| S2 | fill test runner | `scripts/v460/run_fill_test.py` | P2-1 |
| S3 | fill quality metrics | `ztb/metrics/fill_quality.py` | P2-2 |
| S4 | trade dedup 修正 | `ztb/data/market_data_collector.py` | P2-2.1 |
| S5 | G1.1 gate judge (拡張) | `scripts/v460/run_gate_check.py` | P2-3 |
| S6 | 単体テスト | `tests/unit/v460/test_fill_quality.py` | P2-4 |
| S7 | fill test 結果 | `results/v460/fill_test/*.jsonl` | P2-5 |
| S8 | G1.1 判定レポート | `docs/v460/0XX_ph2_rpt_g1_1_exec.md` | P2-5 完了後 |

---

## §9 実装優先度

本ドキュメントは計画書であり、実装は以下の優先度で進める:

1. **P2-0 + P2-2.1** (前提条件整備): CoincheckAdapter 補完 + F8 修正
2. **P2-1** (fill test runner): G1.1 データ収集の核
3. **P2-2 + P2-3 + P2-4** (metrics + gate + test): 判定ロジック
4. **P2-5** (7d 実測): 実時間投資

P2-5 実測開始後は Phase 1 の再検証 (実板データによる G1 リトライ) を並行で進めることが可能。

---

## Appendix A: 改訂履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-02-14 | 初版作成 — G1.1-exec Phase 2 計画書 |
| 2026-02-14 | Appendix B 追加 — fill test 確認方法 (040# より転記) |
| 2026-02-19 | §2 二段階ゲート化反映: E1-E5 → K1-K6 / F1-F8 に再編。旧基準は参考として残す。116# / 115# レビュー |

---

## Appendix B: fill test 確認方法

### B.1 状況確認コマンド

```powershell
# 1. プロセス稼働確認
Get-Process python* | Format-Table Id, ProcessName, StartTime, @{N='CPU_s';E={[math]::Round($_.CPU,1)}}, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}} -AutoSize

# 2. レコード件数確認
Get-ChildItem results/v460/fill_test/fill_records_*.jsonl | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName | Measure-Object -Line).Lines) lines" }

# 3. メトリクス＋Gate判定 (results-only モード)
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --results-only --results-dir results/v460/fill_test

# 4. 結果JSON出力
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --results-only --results-dir results/v460/fill_test --output results/v460/fill_test/g1_1_judgment.json
```

### B.2 ログ確認

```powershell
# 最新ログの末尾
Get-Content results/v460/fill_test/logs/fill_test.log -Tail 20

# 方策A/B の適応ログ
Select-String -Path results/v460/fill_test/logs/fill_test.log -Pattern "方策" | Select-Object -Last 10
```

### B.3 fill test 再起動

```powershell
# 停止
Stop-Process -Id <PID>

# 起動 (レジューム自動対応、レジーム検知は enabled: true で自動有効化)
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 168

# dry-run 確認
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 1 --dry-run
```
