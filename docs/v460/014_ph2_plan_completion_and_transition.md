# 014# ph2 完遂計画 + ph3 移行条件

|  | 値 |
|---|----|
| Doc ID | 014 |
| Phase | ph2 |
| Type | plan |
| 前提 | 000# §2–§3, 010#–013# |
| 状態 | ACTIVE |
| Date | 2026-02-13 |

---

## §1 現在地の確認

### §1.1 Phase 進捗

| Phase | Gate | 状態 | 備考 |
|-------|------|------|------|
| ph0 | G0-data | **PASS** | データパイプライン構築済み |
| ph1 | G1-info | **FAIL (要再検証)** | OHLCV proxy で FAIL。7/9 ターゲット個別 PASS だが Cliff's Delta 未達。Real data で G1 再検証が必要 |
| **ph2** | G1.1-exec | **進行中** (n=35) | fill_test 35 件。API 基盤修正 (013# D.6) 完了。n≥200 必要 |
| ph3 | G2-train | 未着手 | G1 + G1.1 両方 PASS が前提 |
| ph4 | G3-pnl | 未着手 | |
| ph5 | G4-live | 未着手 | |

### §1.2 ph2 完了済みタスク

| 済 | 内容 | 文書 |
|----|------|------|
| ✅ | G1.1 計画策定 | 009# |
| ✅ | 収益性批判的評価 (IC≈0 → 月 -28,000 JPY) | 010# |
| ✅ | レビュー: 迷走要因特定 + 方針修正 | 011# |
| ✅ | レスポンス: 指摘検証 + 補完 | 012# |
| ✅ | Exchange API 監査 + 全修正実装 (C-3~C-9, D-1~D-5) | 013# |
| ✅ | テスト 97/97 pass | — |
| ✅ | 観測データ収集基盤 (MarketDataCollector) 稼働中 | run_observation.py |
| ✅ | 000# 最小改訂 (G1.1 暫定判定条件 n≥200, 3暦日) | 000# §3.3 |

### §1.3 ph2 未完了タスク (本計画の対象)

| # | 項目 | 優先度 | 工数見積 | 根拠 |
|---|------|--------|---------|------|
| T1 | fill_test n≥200 収集 | **P0** | 3日 (実行時間) | 009# §2, 000# §3.3 |
| T2 | real 特徴量設計 + G1 再検証 | **P0** | 5-7日 | 010# §9 R3-R4 |
| T3 | fill_test の .env 自動読込対応 | P1 | 30min | fill_test CLI に --api-key 必須は運用上不便 |
| T4 | WebSocket API 検討 (D-4) | ~~P2~~ **DONE** | 1-2日 | 013# D.5、ポーリング解像度改善。✅ 実装完了 |
| T5 | PnL モンテカルロ化 | ~~P2~~ **DONE** | 1日 | 012# §3 #4。✅ 実装完了 + テスト 34/34 PASS |

### §1.4 現在のデータ蓄積状況

| データ種別 | ファイル | サイズ | 状態 |
|-----------|---------|--------|------|
| Orderbook raw | `data/v460/raw/orderbook/20260213.jsonl.gz` | ~94 KB (1,806 snapshots) | 蓄積中 (run_observation.py) |
| Trades raw | `data/v460/raw/trades/20260213.jsonl.gz` | ~68 KB (158,107 records) | 蓄積中 |
| Real features | `data/v460/features/btc_jpy_1m_v460_real_features.parquet` | 48.7 KB (187 rows × 22 cols) | ✅ 生成済み (1日分) |
| Fill records | `results/v460/fill_test/fill_records_20260213.jsonl` | 15 件 (5.1 KB) | n=15、統計的に不十分 (n≥200 必要) |

---

## §2 3 トラック並走計画

012# §5.4 の「最小改訂 + 並行着手」方針に基づき、以下の 3 トラックを並走する。

### §2.1 Track B: データ収集・実測 (即時開始)

**目的**: G1.1 判定に必要な n≥200 サイクルを収集する。

| Day | アクション | 成果物 |
|-----|----------|--------|
| 0 | T3: `run_fill_test.py` に .env 自動読込を追加 | コード修正 |
| 0 | fill_test を dry-run で動作確認 | 正常動作の確認 |
| 0-3 | fill_test を 0.001 BTC で本番実行開始 (--hours 72) | fill_records の蓄積 |
| 並行 | run_observation.py は継続稼働 (板・約定 raw data 蓄積) | orderbook/trades raw |

**安全設計** (009# §4.2 準拠):
- 最小ロット: 0.001 BTC (≈10,300 JPY)
- buy/sell 交互 (片側蓄積禁止)
- サイクル間隔: 120秒 (1日 720サイクル → 3日で n≈2,160)
- 推定コスト: 1,000-5,000 JPY (009# §6.2)

### §2.2 Track A: Edge 検証 (Week 2)

**目的**: Real 板・約定データから新特徴量を設計し、G1 を再検証する。

**前提**: Track B で 3 日分以上の real data が蓄積されていること。

| Day | アクション | 成果物 |
|-----|----------|--------|
| 4-6 | R3: 実板データから新特徴量候補を設計・実装 | `ztb/features/` への追加 |
| 7-8 | R4: G1 再検証 (XGBoost Walk-Forward, real 特徴量) | G1 判定結果 |

**既存活用** (011# §4 原則):
- 特徴量探索: 既存 `ztb/features/microstructure.py` を拡張
- G1 再実行: 既存 `scripts/v460/run_experiment.py` + `run_gate_check.py`

### §2.3 Track C: 品質向上 (隙間タスク)

| 項目 | タイミング | 工数 |
|------|----------|------|
| T4: WebSocket API 検討 | ~~Day 2-3~~ **Day 0 DONE** | ~~1-2日~~ 0.5日 |
| T5: PnL モンテカルロ化 | ~~Day 4-5~~ **Day 0 DONE** | ~~1日~~ 0.5日 |

---

## §3 Ph2→Ph3 移行条件

000# §3.2/§3.3 に基づく移行判定:

**116# 改訂**: G1.1 を G1.1-quick + G1.2-full の二段階に分割 (115# レビュー反映)。

```
G1 再検証 PASS  ──AND──  G1.2-full PASS  ──→  ph3 (G2-train)
     │                      │
     FAIL                   FAIL
     │                      │
     ↓                      ↓
  v460 縮退           戦略変更検討
  or v461 移行       (aggressive maker,
                      IOC 併用等)
```

**G1.1-quick (72h Kill Gate)**: fill_test 開始 72h で最初の判定。FAIL → 即時停止。WATCH → パラメータ凍結・監視強化で継続。PASS → G1.2-full まで実行継続。

**G1.2-full (168h Qualification Gate)**: 7 暦日の全データで最終判定。PASS → ph3 移行可。

### §3.1 G1.2-full 判定基準 (000# §3.3 / 116#)

| 指標 | 閾値 |
|------|------|
| attempted_fill_rate (F1) | ≥ 70% |
| overall_fill_rate (F1b) | ≥ 62% |
| attempted_cancel_ratio (F2) | ≤ 30% |
| queue_wait_median (F3) | ≤ 60s |
| PnL30 (F4) | 有意に負でないこと (p ≥ 0.05) |
| AS_ratio (F5) | ≤ 30% |
| skip_gate_ratio (F6) | ≤ 20% |
| calendar_days (F7) | ≥ 7 |
| n_attempted (F8) | ≥ 500 |

### §3.2 継続中止ルール (012# §3 #5)

| 条件 | 判断 |
|------|------|
| n≥200 かつ fill_rate < 70% | **中止** — maker 戦略不成立 |
| n≥500 かつ AS > spread/2 が継続 | **中止** — 逆選択コスト過大 |
| G1 再検証で全 9 ターゲット FAIL | **v461 移行** |
| G1 再検証で方向 IC > 0.04 (BE ライン) | **続行** — 収益性の可能性あり |

---

## §4 実施ログ

### §4.1 Day 0 (2026-02-13) — 本計画策定

| 時刻 | アクション | 結果 |
|------|----------|------|
| 21:30 | 000# 改訂状況確認 | ✅ §3.3 暫定判定条件は追記済み (Appendix A 6 行目) |
| 21:30 | MarketDataCollector 稼働確認 | ✅ PID 24148/30024, 19:40 から稼働中。orderbook 94KB, trades 68KB |
| 21:30 | fill_records 確認 | n=3 (fill 2, cancel 1)。統計判断不可 |
| 21:30 | API 資格情報確認 | ✅ .env に COINCHECK_API_KEY/SECRET 設定済み |
| 21:30 | T3: run_fill_test.py に .env 自動読込追加 | ✅ 実装完了 |
| 21:35 | fill_test dry-run 動作確認 | ✅ 1 cycle 正常完了 (sell 開始対応含む) |
| 21:36 | fill_test 本番 72h 開始 (--start-side sell) | ✅ Cycle 1 sell 約定 5.8s |
| 21:36 | 残高問題発見: JPY 1,465 < 買い必要額 10,300 | → sell 開始で自己資金循環解決 |
| 21:41 | Cycle 2 buy 正常発注確認 | ✅ 10,296,737 JPY で発注成功 |
| 21:42 | コミット ae6f3f313 | 014# + .env 自動読込 + --start-side + エラーログ改善 |

### §4.2 Day 0 cont. — T4 WebSocket API 実装

| 時刻 | アクション | 結果 |
|------|----------|------|
| — | Coincheck WebSocket API 仕様調査 | ✅ 公式ドキュメントから完全仕様取得 |
| — | Public WS 接続テスト (`wss://ws-api.coincheck.com`) | ✅ 10秒で trades 23件 + orderbook 26件 (4.8 msg/s) |
| — | `websocket_client.py` 新規作成 | ✅ `CoincheckPublicWS` + `CoincheckPrivateWS` 実装 |
| — | MarketDataCollector に `run_continuous_ws()` 追加 | ✅ WS ストリーミング ⇔ REST ポーリング 切替可能 |
| — | `run_observation.py` に `--ws` オプション追加 | ✅ `--ws` で WS モード起動 |
| — | ユニットテスト 23 件作成 | ✅ 23/23 PASS |
| — | conftest.py 修正 (pytest_ignore_collect 引数 + websockets stub 条件) | ✅ 既存テスト回帰なし |

**T4 成果サマリ**:
- REST 5秒ポーリング → WS 0.1秒ストリーミング (情報密度 **50x** 向上)
- Public チャネル: `btc_jpy-trades` + `btc_jpy-orderbook` (認証不要)
- Private チャネル: `order-events` + `execution-events` (HMAC-SHA256 認証)
- 自動再接続 (exponential backoff) + 統計モニタリング内蔵
- 新ファイル: `ztb/trading/live/exchanges/coincheck/websocket_client.py`
- テスト: `tests/trading/test_websocket_client.py`

### §4.3 T5: PnL モンテカルロシミュレータ実装

| 時刻 | アクション | 結果 |
|------|----------|------|
| — | fill_records 分析 (n=15) | fill_rate 66.7%, PnL mean +0.603 bps, AS 40%, queue wait 5.7s |
| — | 既存 MonteCarloSimulator 確認 | archived — backtest return ベース、fill_test 非対応 |
| — | `ztb/risk/pnl_monte_carlo.py` 新規作成 | ✅ Bootstrap resampling + Bernoulli fill model |
| — | `scripts/v460/run_pnl_monte_carlo.py` 新規作成 | ✅ CLI (--input, --sensitivity, --output) |
| — | モンテカルロ試行実行 (n=15, 10,000 sims) | ✅ E[PnL] = +8,940 JPY/mo, σ = 304 JPY/mo, P(loss) = 0% |
| — | 感度分析実行 | ✅ fill_rate × pnl_adjustment 7×5 グリッド |
| — | ユニットテスト 34 件作成 | ✅ 34/34 PASS (4.2s) |

**T5 成果サマリ**:
- fill_test JSONL から月次 PnL 信頼区間を Bootstrap Monte Carlo で推定
- n=15 でも動作、n 増加で自動精度向上
- 10,000 paths × 21,600 cycles/month のシミュレーション
- G1.1 判定指標を同時出力 (000# §3.3 準拠)
- VaR/CVaR リスク指標 + break-even 分析
- 感度分析: fill_rate × PnL 調整のグリッドスキャン
- 新ファイル: `ztb/risk/pnl_monte_carlo.py`, `scripts/v460/run_pnl_monte_carlo.py`
- テスト: `tests/unit/v460/test_pnl_monte_carlo.py` (34 テスト)

**n=15 時点の G1.1 暫定結果** (統計的意味は n≥200 で確定):
| 指標 | 現状 | 閾値 | 判定 |
|------|------|------|------|
| fill_rate | 66.7% | ≥90% | ❌ |
| cancel_ratio | 33.3% | ≤30% | ❌ |
| queue_wait | 5.7s | ≤60s | ✅ |
| pnl_mean | +0.603 bps | ≥0 | ✅ |
| AS_ratio | 40.0% | ≤20% | ❌ |

### §4.4 A1-A3: G1 再検証パイプライン先行準備

| アクション | 結果 |
|----------|------|
| 000# 分析 → 先行着手可能な A1-A3 タスク特定 | fill_test 蓄積待ち中に G1 再検証パイプラインを先行構築 |
| A1: `build_features.py` に `--mode real` 追加 | ✅ raw JSONL.gz → aggregate_to_1min → microstructure → Parquet |
| A1: `aggregate_to_1min()` VWAP lambda バグ修正 | ✅ pandas resample 内 numpy shapes mismatch → sum(pv)/sum(amount) |
| A1: `_discover_dates()` stem バグ修正 | ✅ `.jsonl.gz` 二重拡張子対応 |
| A1: real 特徴量パイプライン試行実行 | ✅ 187 rows × 10 features × 22 cols、NaN 0%、48.7 KB |
| A2: `microstructure.py` ユニットテスト 29 件作成 | ✅ 29/29 PASS (1.0s) |
| A3: G1 再検証用 real config 作成 | ✅ `g1_real_full_9targets.yaml` (9 target 組合せ) |
| v460 テスト全体リグレッション | ✅ 160/160 PASS (7.2s) |

**A1-A3 成果サマリ**:
- **real 特徴量パイプライン完全動作**: `build_features.py --mode real` で raw → 1min agg → 10 microstructure features → Parquet 一気通貫
- **proxy vs real**: proxy は OHLCV 近似 (高値-安値 range)、real は実際の bid/ask/depth/trade flow から算出。spread は 2.4 bps (real) と精度が桁違い
- **VWAP 集約バグ修正**: `aggregate_to_1min()` 内の lambda での numpy shapes mismatch を sum(price\*amount)/sum(amount) 形式に修正
- **G1 再検証準備済**: 3日分以上の data 蓄積後に `run_experiment.py --config g1_real_full_9targets.yaml` で即実行可能
- 新テスト: `tests/unit/v460/test_microstructure_features.py` (29 テスト)
- 新 config: `configs/v460/experiments/g1_real_full_9targets.yaml`

| データ | 統計 |
|--------|------|
| orderbook raw | 1,806 snapshots (10-level depth) |
| trades raw | 158,107 records |
| 1min bars | 187 rows |
| bid_ask_spread | mean 2.4 bps |
| depth_imbalance | [-0.97, +0.90] |

### §4.5 B1-B3 + 課題探索・修正

| アクション | 結果 |
|----------|------|
| B1: `aggregate_to_1min()` ユニットテスト 26 件作成 | ✅ 26/26 PASS — OB-only, Trades-only, merge, edge cases |
| B2: 000# §3.9 継続中止ルール追記 | ✅ fill_rate<70% 中止、AS>spread/2 中止、実損キャップ 10,000 JPY |
| B3: mypy strict — v460 新規コード全ファイル | ✅ 0 エラー (legacy 1,162 エラーは pre-existing) |
| #1: `ztb/risk/__init__.py` エクスポート修正 | ✅ PnLMonteCarloSimulator, MonteCarloConfig, MonteCarloResult 追加 |
| #2: `ztb/features/__init__.py` エクスポート修正 | ✅ add_microstructure_features, MICROSTRUCTURE_FEATURES 追加 |
| #3: CHANGELOG.md に v460 セクション追加 | ✅ Added/Changed/Fixed/Documentation 全項目記載 |
| #5: fill_test レジューム機構追加 | ✅ `resume_from_existing()` で cycle_count + last_side 復元 |
| 課題探索 (コードベース全体スキャン) | 17 件発見 — Critical 4, High 4, Medium 4, Low 5 |
| テスト全体リグレッション | ✅ 186/186 PASS (14.4s) |

**課題スキャン結果サマリ** (未対応分):

| 優先度 | 内容 | 影響 |
|--------|------|------|
| 🟠 #4 | sac_utils.py L251: メトリクス読み取り仮値 | G2-train フェーズで要修正 |
| 🟠 #6 | run_experiment.py: SAC/backtest タスク未実装 | G2-train フェーズで実装 |
| 🟠 #7 | bridge.py: Zaif API スタブ | v460 では Coincheck 使用のため低影響 |
| 🟠 #8 | broker_registry.py: 全 API NotImplementedError | v460 では直接 adapter 使用 |
| 🟡 #9 | テスト skip/xfail 多数 | legacy、v460 テストは全 PASS |
| 🟡 #10-12 | Coincheck reconciliation/order_state/paper_trader スタブ | G4-live で要実装 |
| 🟢 #13-17 | 移行 shim、ops スタブ、型定義 | 技術的負債、優先度低 |

---

### §4.6 テストカバレッジ強化 + モニタリング基盤

| アクション | 結果 |
|----------|------|
| fill_test モニタリングスクリプト作成 | ✅ `scripts/v460/monitor_fill_test.py` — §3.9 自動判定、G1.1 指標表示、`--watch` モード |
| WebSocket client テスト 44 件作成 | ✅ 44/44 PASS — パーサー、Public/Private WS、認証、ディスパッチ |
| Config validation テスト 28 件作成 | ✅ 28/28 PASS — `_deep_merge`、`_validate`、`gate_thresholds.yaml` 整合性 |
| LFS 問題発見 + 作業コピー復元 | ✅ `.gitattributes` の `ztb/analysis/**`, `ztb/evaluation/**`, `docs/**` がソースコードをLFS化。`git lfs pull` で復元 |
| mypy strict — monitor_fill_test.py | ✅ 0 エラー (StopRule dataclass 型安全化) |
| テスト全体リグレッション | ✅ 258/258 PASS (14.5s) — 前回 186 から +72 テスト |

**fill_test 中間レポート (n=35)**:

| 指標 | 値 | G1.1 閾値 | 判定 |
|------|------|-----------|------|
| fill_rate | 77.1% | ≥90% | ❌ 未達 |
| cancel_ratio | 22.9% | ≤30% | ✅ |
| queue_wait | 11.8s (median) | ≤60s | ✅ |
| pnl_mean | -1.166 bps | ≥0 | ❌ (モデル未完成のため想定内) |
| pnl_median | +0.042 bps | — | ✅ (中央値はプラス) |
| AS_ratio | 48.1% | ≤20% | ❌ (要監視) |
| 経過 | 4.8h | 72h+ | 🔄 収集中 |
| n=200 推定 | ~23h後 | — | — |

---

## Appendix A: 改訂履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-02-13 | 初版作成 — ph2 完遂計画 + 3 トラック並走 + 移行条件 |
| 2026-02-13 | T4 WebSocket API 実装完了。§4.2 追記。T4 ステータス DONE |
| 2026-02-14 | T5 PnL モンテカルロ実装完了。§4.3 追記。T5 ステータス DONE。fill_test n=15 に更新 |
| 2026-02-14 | A1-A3 G1 再検証パイプライン先行準備完了。§4.4 追記。テスト 160/160 PASS |
| 2026-02-14 | B1-B3 + 課題修正。§4.5 追記。aggregate_to_1min テスト 26件、000# §3.9 継続中止ルール、__init__.py エクスポート修正、CHANGELOG 更新、fill_test レジューム機構。テスト 186/186 PASS |
| 2026-02-14 | テストカバレッジ強化 + モニタリング。§4.6 追記。WS テスト 44件、config テスト 28件、monitor_fill_test.py、LFS 復元。テスト 258/258 PASS |
| 2026-02-19 | §3 移行条件を二段階ゲートに改訂: G1.1 PASS → G1.2-full PASS に更新。116# / 115# レビュー反映 |
