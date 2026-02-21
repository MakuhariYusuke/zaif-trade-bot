# 133# ph2 rev: 132ログ分析の再検証 + 高収益化最大化プラン

| key | value |
|---|---|
| 番号 | 133 |
| フェーズ | ph2 |
| 種別 | rev |
| 対象 | `132_fill_test_log_analysis.md` |
| 作成日 | 2026-02-22 |
| 参照 | `docs/v460/000_ph0_plan_project_proposal.md`, `docs/v460/118_phg_rpt_backlog_deep_analysis.md`, `docs/v460/132_fill_test_log_analysis.md`, `results/v460/fill_test/fill_records_20260213.jsonl`, `results/v460/fill_test/fill_records_20260214.jsonl`, `results/v460/fill_test/fill_records_20260215.jsonl`, `results/v460/fill_test/fill_records_20260216.jsonl`, `results/v460/fill_test/fill_records_20260217.jsonl`, `results/v460/fill_test/fill_records_20260218.jsonl`, `results/v460/fill_test/fill_records_20260219.jsonl`, `results/v460/fill_test/fill_records_20260220.jsonl`, `results/v460/fill_test/fill_records_20260221.jsonl`, `results/v460/fill_test/logs/fill_test.log`, `logs/retrain_scheduler.log`, `logs/retrain_history.jsonl`, `configs/v460/fill_test.yaml`, `scripts/v460/ml/retrain_scheduler.py`, `scripts/v460/ml/feature_enricher.py`, `scripts/v460/analysis/analyze_fill_records.py`, `scripts/v460/analysis/analyze_fill_detail.py`, `scripts/v460/gate_judgment.py` |
| 結論 | **132の方向性は概ね妥当。ただし最新ログ時点で「全体WATCHでも最新runはFAIL」の乖離が拡大しており、今は“改善”より先に“悪化ドリフト停止”を優先すべき。** |

---

## §0 エグゼクティブサマリ

1. `132` の主要仮説（sell劣後、unknown劣後、retrain停滞、trades I/O問題）は概ね正しい。  
2. ただし最新データ反映で件数は `1722→1726`、filled は `1096→1097` に更新。  
3. 全体 clean 判定は `G1.2=WATCH` だが、**最新 run_id `1771669596_481369d6` 単体は G1.1/G1.2 とも FAIL**。  
4. 実運用の最重要ボトルネックは次の3つ。  
   - `new_samples` 負値で retrain 永久スキップ (`scripts/v460/ml/retrain_scheduler.py:1008`)  
   - trades 欠損により全量フォールバック常態 (`scripts/v460/ml/feature_enricher.py:421`)  
   - 残高制約由来の強制切替でエッジ悪化（`balance_forced_switch=True` の fill は平均 `-1.98bps`）  
5. 月次期待値（Monte Carlo, 2000回）は `-3,327 JPY/month`。`000#` の大義（短期高収益）に対して現状は未達。

---

## §1 132の妥当性検証（最新ログ反映）

| 項目 | 132の主張 | 最新確認 | 判定 |
|---|---|---|---|
| 全体収益 | `PnL30 < 0` | `-0.292bps`（raw）, `-0.218bps`（clean） | ✅ |
| sell劣後 | sellが主損失源 | sell平均 `-0.552bps`、buy `-0.036bps` | ✅ |
| unknown劣後 | unknownが最悪 | 全体では `unknown=-0.891bps`、ただし最新runではunknown寄与は相対低下 | ✅/⚠️ |
| retrain停滞 | deploy 不能 | `retrain_history`: 26件すべて skipped。`-831` 新規数も再発 | ✅ |
| trades全量ロード | 4.4M行ロード | 継続発生（`Loaded 4396171 trades`） | ✅ |
| SkipGate/TimeFilter記録欠損 | skip種別見えない | SkipGateは記録済み。TimeFilter/Preflight由来は依然欠損 | ⚠️（部分修正済） |
| latest run 劣化 | 直近run悪化 | 最新run `FR=44.9%, PnL30=-1.20bps` | ✅ |

---

## §2 追加で見えた重要課題（132未記載または弱い）

### C1. 最新run FAILを全体WATCHが隠している（Simpson型リスク） [CRITICAL]

- 全体 clean (`n=1486`) では `G1.2=WATCH`。  
- しかし最新 run (`n=78 clean`) は `overall_fill_rate=0.513`, `skip_gate_ratio=0.308`, `pnl30=-1.20bps` で FAIL。  
- 現状の意思決定は「全期間混合」に寄りすぎている。

### C2. retrain one-shot の監査ログ欠落 [HIGH]

- `--once` 経路は `logs/retrain_history.jsonl` に書かない (`scripts/v460/ml/retrain_scheduler.py:1491`)。  
- 実際に one-shot rejected を実行しても履歴上は「deploy/reject の痕跡なし」になる。  
- 「0/26 skipped」は定期ループ限定の値で、運用観測として不完全。

### C3. `analyze_fill_detail.py` が途中クラッシュ [HIGH]

- `cancel_reason=None` で `startswith` 呼び出しクラッシュ (`scripts/v460/analysis/analyze_fill_detail.py:92`)。  
- さらに `offset_ratio` / `spread_jpy` 参照は現行FillRecord列と非整合 (`scripts/v460/analysis/analyze_fill_detail.py:80`, `scripts/v460/analysis/analyze_fill_detail.py:86`)。  
- 分析結果の一部は誤差または未出力。

### C4. trades欠損が構造化（データ供給停止） [CRITICAL]

- `data/v460/raw/trades` は `20260213..20260219` のみ。`20260220/20260221` が欠落。  
- retrain は当日 `date_filter` で空になり、最終的に全件 fallback。  
- 最新runに古い約定分布を混ぜるため、特徴量の時間整合が崩れる。

### C5. 残高制約時の強制side切替が損失を増幅 [HIGH]

- `balance_forced_switch=True` fill 13件の平均は `-1.98bps`。  
- 非強制fill 1084件は `-0.272bps`。  
- 最新runの悪化に直接寄与（`scripts/v460/run_fill_test.py:1133` 以降のpreflight分岐）。

---

## §3 損失寄与の分解（何を止めれば早く効くか）

### 3.1 時間帯×side（寄与 = mean×件数）

- sell-UTC14: `n=10, mean=-6.659bps, contrib=-66.6bps`
- sell-UTC08: `n=8, mean=-6.725bps, contrib=-53.8bps`
- buy-UTC18: `n=12, mean=-3.598bps, contrib=-43.2bps`
- sell-UTC04: `n=15, mean=-2.854bps, contrib=-42.8bps`
- sell-UTC13: `n=19, mean=-1.629bps, contrib=-30.9bps`

### 3.2 レジーム×side

- ranging-sell: `n=266, mean=-0.399bps, contrib=-106.2bps`
- n/a-sell: `n=128, mean=-0.803bps, contrib=-102.7bps`
- trending-sell: `n=104, mean=-0.706bps, contrib=-73.5bps`
- unknown-buy: `n=47, mean=-1.384bps, contrib=-65.0bps`

### 3.3 逆にプラス寄与

- trending-buy: `n=100, mean=+0.538bps, contrib=+53.8bps`
- ranging-buy: `n=267, mean=+0.044bps, contrib=+11.9bps`

**示唆**: まず「sell損失帯」と「unknown-buy」を止血するのが最短。

---

## §4 改善案（貪欲版・実行順つき）

| ID | 優先 | 施策 | 根拠 | 期待効果 |
|---|---|---|---|---|
| P0-01 | P0 | `new_samples` を run_id 同一基準で算出し負値を禁止 | `-831` 再発 (`retrain_scheduler.py:1008`) | retrain再始動 |
| P0-02 | P0 | `--once` でも history追記 | 監査欠落 (`retrain_scheduler.py:1491`) | 判断の再現性向上 |
| P0-03 | P0 | trades欠損日の原因特定（collector起動漏れ/失敗） | `data/v460/raw/trades` 20/21欠落 | 特徴量の時系列整合回復 |
| P0-04 | P0 | fill_test内に TradesRecorder 追加（OBRecorder対称） | OBのみ常時記録 (`ob_recorder.py`) | retrain I/O暴走抑制 |
| P0-05 | P0 | TimeFilter/Balance preflight skip も FillRecord化 | 現在は可観測性欠損 (`run_fill_test.py:1052`) | 分析盲点解消 |
| P0-06 | P0 | `analyze_fill_detail.py` の None/列名バグ修正 | `startswith(None)` crash | 分析再信頼化 |
| P0-07 | P0 | run単位ゲートを標準化（all-runと分離） | 最新run FAIL隠蔽 | ドリフト早期検出 |
| P0-08 | P0 | `balance_forced_switch` 時は発注抑制（hard skip） | 強制fill平均 `-1.98bps` | 即時損失削減 |
| P0-09 | P0 | unknown-buy を一時 skip（または更に強いboost） | unknown-buy寄与 `-65bps` | 止血 |
| P0-10 | P0 | sellの動的kill: rolling50で mean<-0.5bpsならsell停止 | sell全体負け | 損失片側切断 |
| P0-11 | P0 | hot-reload失敗時に自動ハッシュ再生成 + 再試行上限設定 | 失敗46回/成功1回 | 復旧時間短縮 |
| P0-12 | P0 | `run_gate_check.py` を G1.2 呼び出し可能に統一 | CLIは `G1.1` まで | 運用コマンド統一 |
| P1-01 | P1 | buy/sell 完全分離モデル（別閾値でなく別学習器） | sellだけ構造負け | side特化改善 |
| P1-02 | P1 | target二層化（buy=pnl30, sell=pnl120） | 120sで回復、sell短期弱い | side別最適化 |
| P1-03 | P1 | score校正（isotonic/quantile） | score-pnl相関ほぼ0 | SkipGate有効性改善 |
| P1-04 | P1 | regime別閾値（ranging/trending/unknown） | 一律閾値で混線 | 過剰/過少skip抑制 |
| P1-05 | P1 | `skip_gate_ratio` 上限超過時の自動degrade（model→rule） | 最新run 30.8% | 収集量確保 |
| P1-06 | P1 | stale reprice売側上限縮小（2→1）AB | reprice=2 平均 `-3.44bps` | tail損失圧縮 |
| P1-07 | P1 | timeout群の再見積（90s固定から動的） | timeout 195件 | fill率/機会損失改善 |
| P1-08 | P1 | spread狭小時の「休む」判定追加 | too narrow 61回 | 無駄サイクル削減 |
| P1-09 | P1 | balance margin (`1.01`) の動的化 | Insufficient大量 | preflight失敗削減 |
| P1-10 | P1 | preflight失敗連続時は side freeze ではなく run pause | dead-cycle抑止 | 資本毀損回避 |
| P1-11 | P1 | PnL評価を fee/slippage控除後で統一 | 現在は楽観寄り | 実収益一致 |
| P1-12 | P1 | 直近N fillのみでonline比較（全履歴平均を廃止） | 非定常で旧run混入 | 反応速度向上 |
| P2-01 | P2 | v458 WalkForwardSplitter を retrain判定へ移植 | `ztb/evaluation/walk_forward/splitter.py` | 過学習耐性 |
| P2-02 | P2 | v459統計gate（Holm/p-mean/cliff）を deploy判定に常時化 | `ztb/metrics/gate_checks.py` | false deploy抑制 |
| P2-03 | P2 | `run_observation.py` を fill_test と同時運転（監視プロセス化） | raw trades欠損対策 | 学習データ連続化 |
| P2-04 | P2 | oracle上限を毎日再計測し「理論上限との差」をKPI化 | v459教訓③ | 改善の方向誤り防止 |
| P2-05 | P2 | inventory-aware quoting（在庫偏りでside優先） | JPY/BTC不足頻発 | 強制切替減 |
| P2-06 | P2 | worst hour-side に限定した局所ルール学習 | 寄与分解で集中 | 小改修で効く |
| P2-07 | P2 | execution traceの因果ログ（decision→fill→pnl）標準化 | 事後検証工数高 | 速度向上 |
| P2-08 | P2 | shadow model配信（本番未反映でA/B採点） | hot-reload事故回避 | 安全な比較 |
| P2-09 | P2 | run開始時の前提健全性チェック（trades当日ファイル必須） | 欠損でall-trades fallback | 異常早期停止 |
| P2-10 | P2 | Gateに「最新run hard floor」を追加 | 全体WATCHでも現場FAIL | 運用事故防止 |
| P3-01 | P3 | v458 `hft_proxies` を boardless fallback 特徴量として再活用 | `ztb/features/hft_proxies.py` | 板欠損耐性 |
| P3-02 | P3 | `advanced_regime_detector` の導入AB | `ztb/analysis/regime/advanced_regime_detector.py` | non-stationary対応 |
| P3-03 | P3 | `dynamic_position_sizer` を lot_sizingへ橋渡し | `ztb/risk/dynamic_position_sizer.py` | 資本効率向上 |
| P3-04 | P3 | `pnl_monte_carlo` を日次自動実行し赤信号通知 | `ztb/risk/pnl_monte_carlo.py` | 期待値悪化の早期検知 |
| P3-05 | P3 | venue横断の同一ロジック比較（Coincheck/Bitflyer） | 000# 前提（取引所切替可能設計） | 実行品質改善余地 |

---

## §5 vXXX資産の再利用提案（具体ファイル）

| 系統 | 再利用先 | 候補 |
|---|---|---|
| v458 Walk-Forward | retrain品質判定 | `scripts/v458/run_walk_forward_v458.py`, `ztb/evaluation/walk_forward/splitter.py`, `ztb/evaluation/walk_forward/evaluator.py` |
| v459 統計ゲート | deploy判定 | `ztb/metrics/gate_checks.py` |
| v458 boardless代理特徴量 | trades欠損時 fallback | `ztb/features/hft_proxies.py` |
| v45x レジーム資産 | unknown削減 | `ztb/analysis/regime/advanced_regime_detector.py`, `ztb/analysis/regime/regime_eval.py` |
| v45x リスク資産 | 強制side切替抑制 | `ztb/risk/dynamic_position_sizer.py`, `ztb/risk/drawdown_controller.py` |

---

## §6 直近の推奨実行順（48時間）

1. **止血**: P0-01〜P0-06 を先に実施（学習・分析・監査の土台を修復）。  
2. **即効改善**: P0-08〜P0-10 で損失寄与の大きい条件を短期遮断。  
3. **再計測**: 同一 Git SHA / 同一 YAML で 24h 連続runし、run単位 Gate を判定。  
4. **再学習再開**: trades欠損を直した後に retrain を再起動し、one-shot でなく定期ループで deploy 可否を確認。  
5. **上限引き上げ**: P1 群（side分離・score校正）に進む。

---

## §7 最終判断

`132` は「壊れている箇所」を正しく捉えているが、最新ログまで含めると優先順位はさらに明確。  
いま最短で儲けに近づく道は、**モデル高度化より先に運用ドリフト停止（latest-run FAIL封じ）**。  
そのうえで、sell/unknown/balance_forced の3点を同時に潰すのが最短経路。

## 追記: 133# レビュー結果に対するセカンドオピニオン (Gemini 3.1 Pro)

### 1. 133# プランへの全体評価：圧倒的に支持
133# で指摘された「全体WATCHが最新runのFAILを隠蔽している（Simpson型リスク）」「残高制約による強制side切替の損失増幅」は、実運用における致命的な盲点を突いており極めて優秀な分析である。「モデル高度化より先に運用ドリフト停止（止血）」という結論は、本プロジェクトの大義（短期間での高収益性）に完全に合致する。P0施策は直ちに実行に移すべきである。

### 2. Gemini 3.1 Pro からの追加の批判的考察と警告

#### A. 在庫管理（Inventory Risk）の軽視は致命的
133# の C5 で指摘された「残高制約時の強制side切替による損失（-1.98bps）」は、単なるエッジケースではなく**アーキテクチャの根本的欠陥**を示している。マーケットメイカーにおいて在庫の偏り（片張りによる資金枯渇）は死を意味する。
133# では `inventory-aware quoting` (P2-05) が P2 に設定されているが、これは **P0 に格上げ** すべきである。予測スコアがいくら高くても、在庫が限界に達している場合は逆サイドのクオートを優遇（または同サイドをスキップ）するロジックを最優先で組み込む必要がある。

#### B. 「止血」でマイナスは消せても、プラス（アルファ）は生まれない
P0施策（sellの停止、unknownの停止など）は「負け戦を避ける」ための止血としては完璧だが、これらを全て実行した後に「何で勝つのか（エッジの源泉）」が依然として不透明である。
現状のプラス寄与（trending-buy等）が、単なる地合い（BTCの全体的な上昇トレンド）によるベータなのか、モデルの真の予測力（アルファ）なのかを切り分ける必要がある。ベースライン（単純なランダムエントリーや常時クオート）との厳密な比較（Oracle PnLとの乖離分析）を急ぐべき。

#### C. データパイプラインの脆弱性（GIGOの危機）
C4で指摘された「tradesデータの欠損による全量フォールバック」は、MLシステムとして最も恐れるべき事態（Garbage In, Garbage Out）を引き起こしている。特徴量の時間整合性が崩れた状態で retrain を回すことは、モデルを意図的に破壊しているに等しい。P0-03/P0-04 のデータ供給網の修復は、いかなるロジック改修よりも優先されるべき「インフラの復旧」である。

#### D. Phase 3 (SAC/強化学習) への移行に対する最終警告
133# では言及されていないが、以前のレビューでも警告した通り、**現状のマイナス期待値（または止血直後の不安定なゼロ近傍）の環境下で Phase 3 (SAC) に突入することは絶対に避けるべき**である。
133# の P0/P1 施策を完遂し、十分なサンプルサイズ（n > 2000）で統計的に有意なプラスの PnL (G1.2 FULL PASS) を証明するまでは、強化学習の封印を継続すること。

### 3. 結論とネクストアクション
133# の提案する「48時間の推奨実行順」を全面的に支持する。ただし、以下の修正を加える。
1. **インフラ復旧**: tradesデータ欠損の修復 (P0-03, P0-04) を最優先の「Day 0」タスクとする。
2. **在庫管理のP0化**: 残高制約時の強制切替 (P0-08) に加え、在庫偏りを加味したクオート制御 (P2-05相当) を P0 に引き上げる。
3. **止血と観測**: P0施策実装後、最低24時間はモデルの改修を止め、純粋なデータ収集とベースライン測定に徹する。
