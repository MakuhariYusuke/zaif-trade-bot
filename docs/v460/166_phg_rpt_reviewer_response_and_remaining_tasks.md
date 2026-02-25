# 166# レビュー対応 + 162/163 残課題消化

## 概要

165# 末尾 7 のレビュー指摘 (P02, P12) への対応と、162#/163# 残課題の消化を実施。

## 1 レビュー指摘事項 (165# 7.5)

| ID | 優先度 | 指摘内容 | 対応状況 |
|----|--------|----------|----------|
| R-1 | P0 | 再現性固定 (run_id/period/files を1セット) |  stopgap_health.py に apply_filters() 追加、CLI 引数対応 |
| R-2 | P0 | 163出口判定の運用化 (閾値逸脱自動アラート) |  generate_alerts() + AlertItem (critical/warning/info) |
| R-3 | P1 | AS-R1 閾値校正 (8.0bps, 機会費用測定) |  velocity log 蓄積待ち (fill_test SHA 955a7818842a で2件skip確認) |
| R-4 | P1 | model_used 経路別レポート (side_sell/side_buy/unified 分離) |  section_model_used() + compute_model_used_metrics() |

## 2 実施内容

### 2.1 stopgap_health.py (584  762 lines)

**追加機能:**
- apply_filters(records, *, run_id, git_sha, date_from, date_to)  analyze_fill_logs.py と同一ロジック
- ModelUsedMetrics dataclass  経路別 AS率/PnL メトリクス
- AlertItem dataclass  severity (critical/warning/info), stopgap_id, message
- compute_model_used_metrics()  model_used 経路別の AS率/PnL 算出
- generate_alerts()  退出基準閾値逸脱の自動アラート生成
- generate_health_report() に filters_applied 引数追加
- print_health_summary() に Model Used 表 + Alerts セクション追加

### 2.2 stopgap_daily_report.py (95  115 lines)

**追加:**
- --run-id, --git-sha, --date-from, --date-to CLI 引数
- apply_filters() 呼び出し + フィルタ後空チェック

`
# 使用例: 再現性固定分析
.venv\Scripts\python.exe scripts/v460/analysis/stopgap_daily_report.py \
  --git-sha 955a78 --date-from 2026-02-25 --date-to 2026-02-28
`

### 2.3 analyze_fill_logs.py (675  709 lines)

**追加:**
- section_model_used()  model_used (skip_gate_model_used) 経路別分析セクション
  - 各経路: N (約定数), AS# (AS 件数), AS% (AS 率), PnL30 (平均PnL30s), AS_Loss (AS平均損失)
  - 対象経路: 
one, primary:side_buy, primary:side_sell, primary:unified 等

### 2.4 テスト

| ファイル | 追加テスト数 | 合計 |
|----------|-------------|------|
| 	est_stopgap_health.py | +20 (6 apply_filters + 5 model_used + 4 alerts + 4 report + 1 intent) | 52 |
| 	est_analyze_fill_logs.py | +3 (section_model_used) | 21 |
| **合計** | +23 | 73 passed |

## 3 162/163 残課題ステータス

### 162# 4 改善提案

| ID | 内容 | 状態 |
|----|------|------|
| P0-A1 | SkipGate AS 検知改善 |  AS-R1 velocity rule (165#) |
| P0-A2 | balance_forced_skip 根因 |  IS enabled=true (5a5b9ba42) |
| P0-A3 | sell guard 緩和 |  offset_floor 0.100.20 (164#) |
| P1-B1 | 時間帯フィルタ |  107# dynamic gating |
| P1-B2 | null regime fallback |  unknown regime skip |
| P1-B3 | Retrain data 蓄積 |  quality gate モデル拒否中 |
| P1-B4 | git_sha 8ba101953 diff |  新SHA で supersede |
| P2-C1 | orderbook_error sell_guard 閾値 |  未着手 |
| P2-C2 | Reprice logic tuning |  未着手 |
| P2-C3 | stale_skip_gate_blocked 閾値 |  未着手 |

### 162# 7.3 レビュー提案

| 提案 | 状態 |
|------|------|
| --run-id/--git-sha/--date CLI |  完了 (analyze + stopgap 両方) |
| Stopgap exit 表 |  163# + 165# stopgap_health.py |
| 160# 判定ロジック統合 |  ab_judgment.py + stopgap_health.py |
| IS staging plan |  163# 文書化済 (S0S1 完了, S2 待機) |

### 163# ロードマップ

| 項目 | 状態 |
|------|------|
| IS enabled=true |  S1 (5a5b9ba42) |
| Dynamic gating |  regime_adaptive_enabled=true |
| Sell offset 動的最適化 | SO-1 , SO-2/SO-3  |
| AS 根因 + モデル改善 |  AS-R1 (165#) + model_used (本166#) |
| sell_guard 閾値動的化 |  P2 未着手 |

## 4 165# 受入条件チェック (7.6)

| 条件 | 状態 | 備考 |
|------|------|------|
| 同一母集団の再現検証 |  | apply_filters で固定可能 |
| model_used 経路別 AS 源泉説明 |  | section_model_used + compute_model_used_metrics |
| Lock-free 連続 run log で AS-R1 効果/副作用評価 |  | SHA 955a7818842a 稼働中 (2 velocity skip/29 cycles) |
| 162/163 stopgap exit 判断と整合 |  | generate_alerts で自動化 |

## 5 現在のシステム状態

- **Fill Test**: PIDs 139024/139880 (SHA 955a7818842a), 安定稼働中
- **Retrain**: PIDs 120328/147464
- **Velocity Skip**: 29 cycles 中 2 件 skip (7.3%)
- **テスト**: 73 passed (関連テストのみ), PPO integration test は無関係の既知障害
- **Git HEAD**: 955a78188

## 6 次ステップ

1. **R-3 AS-R1 閾値校正**: velocity log が 100+ records 蓄積次第、price_velocity_60s 分布 vs AS outcome を分析
2. **P2-C1/C2/C3**: orderbook_error 閾値、reprice tuning、stale_skip_gate 閾値  IS S2 移行後に着手
3. **IS S1S2 移行判定**: nalyze_fill_logs.py --git-sha 5a5b9ba --date-from 2026-02-26 で評価

---

## 7 Reviewer追記（118#以降の未消化 + 実ログ改善提案）

### 7.1 000# 基準からの逸脱警告

`000` の大義/運用基準（短期高収益、Gate判定、run_id/git_sha固定再現）に照らすと、現状は**大きく逸脱はしていない**。
ただし次の2点は逸脱予兆として警告する。

1. **再現性の汚染源が残存**  
  `fill_test.log` に `blank git_sha` quarantine が継続出現（312件除外ログ）。run固定分析の信頼性を削る。
2. **索引の運用遅延**  
  `index.md` が「最終更新: 162#」のままで、165#/166# が反映されていない。運用判断の参照点が古くなる。

### 7.2 118#以降で現時点も未消化の項目（要指摘）

実装済み扱いではなく、**待ち/未着手/運用未検証**として残っているものを抽出。

| 区分 | 項目 | 現状 | 根拠文書 |
|------|------|------|----------|
| P1 | AS-R1 閾値校正（R-3） | 速度ログ蓄積待ち | 166# §1 |
| P2 | orderbook_error sell_guard 閾値 (P2-C1) | 未着手 | 166# §3 |
| P2 | Reprice logic tuning (P2-C2) | 未着手 | 166# §3 |
| P2 | stale_skip_gate_blocked 閾値 (P2-C3) | 未着手 | 166# §3 |
| P1 | 同side内 variant 比較への段階移行 | データ蓄積待ち | 160# §11.5 / 161# §9 |
| P2 | execute_trade 品質検証 | 未着手（実装済みだが運用検証未了） | 159# §9.3 / 160# §11.5 |

補足: 166# で「対応済み」とした項目（R-1/R-2/R-4）は実装確認ベースで妥当。

### 7.3 実ログ直読での改善可能箇所（優先順）

#### A. 再現性・運用安定（P0相当）

1. **lock競合/回収イベントの継続発生**  
  stale lock reclaim ログが複数回出現。加えて events に lock競合起因 crash 履歴あり。  
  → 単一起動ガードを watchdog 側で先に判定し、`run_fill_test` 側は「既存稼働検知時に即終了（再起動抑止）」を明示化すべき。

2. **blank git_sha quarantine 継続（312件）**  
  再現性固定方針に反する継続ノイズ。  
  → `git_sha` 欠損レコードの生成経路を特定し、記録時必須化（欠損なら保存拒否 or 即時補完）を推奨。

#### B. 収益直結（P1相当）

3. **Insufficient JPY/BTC 警告の反復**  
  同一条件で短時間に連続発火（buy/sell 両側）。  
  → side別 cooldown/hysteresis を導入し、同理由スキップの高頻度再試行を抑制（機会損失・ログノイズ削減）。

4. **trending_up 連続 sell skip が長連鎖**  
  `consecutive=19/30` 付近まで継続し、片側機会損失リスクが高い。  
  → 162/163 の stopgap出口判定に「連続skip上限時の緩和条件（時間窓/AS悪化なし）」を追加して運用化。

5. **AS-R1 velocity skip は発火件数がまだ少ない**  
  skipログは確認できるが、閾値校正には不足。  
  → R-3 継続（100件+ の速度サンプル蓄積後に閾値再推定）は妥当。

#### C. 品質・安全（P1-P2相当）

6. **sklearn feature-name warning が大量反復**  
  `X does not have valid feature names` が数百回発生。  
  → 推論入力をカラム名付き DataFrame で統一し、特徴量順序ズレの潜在リスクを除去。

7. **cancel race / status unknown 系の実運用ノイズ**  
  `order not found -> cancel failed(400)` が発生。  
  → 既存の「likely filled during cancel」処理は有効だが、KPIに `cancel_failed_likely_filled` を分離集計し、誤判定率を監視すべき。

### 7.4 判定

- 166# はレビュー対応報告として概ね妥当。  
- ただし「残りは待ち作業」の表現は過度に楽観的で、**P2-C1/C2/C3・execute_trade品質検証・再現性ノイズ除去**は能動タスクとして残存。  
- 000# の方針に沿うなら、次の意思決定は「待機」ではなく**再現性ノイズの除去完了確認**をゲートに置くのが安全。

---

## 8 追記: 166# に対するセカンドオピニオンとログ直読からのインサイト (Gemini 3.1 Pro)

### 8.1 Codexレビュー (7. Reviewer追記) への評価
Codexレビューの指摘は、実運用ログのノイズや運用上の摩擦を正確に捉えており、非常に的確である。特に「lock競合」と「Insufficient JPY/BTC 警告の反復」は、システムの安定性と収益性に直結する問題であり、P0/P1としての優先度設定に完全に同意する。

### 8.2 ログ直読からの追加インサイトと「致命的デッドロック」の発見

最新の `fill_test.log` および `fill_test_events.jsonl` を直読した結果、Codexレビューの指摘を裏付けるとともに、**さらに深刻な「資金枯渇 × レジームスキップ」のデッドロック状態**が進行していることを確認した。

#### A. 「資金枯渇 × レジームスキップ」による完全な機会損失 (P0)
ログ上で以下のパターンが無限ループしている。
1. `Insufficient JPY for buy` が発生し、`balance_forced` により強制的に `sell` に切り替わる。
2. しかし、現在のレジームが `trending_up` であるため、`sell` がスキップされる (`Skipping sell — trending_up regime`)。
3. 結果として、buy も sell も実行できず、ただサイクルを空回りさせている。

**対策**: `balance_forced` による強制 side 切替が発生した際、その side がレジームフィルタ等でスキップされる場合は、単に待機するのではなく、**「在庫リバランスのための特別緩和モード（例：offset をゼロまたはマイナスにしてでも約定させ、資金を回復させる）」**を発動させるべきである。現状のロジックでは、一度資金が偏るとトレンドが変わるまで一切の取引ができなくなる致命的な欠陥がある。

#### B. Fast Fill Defense の過剰反応リスク (P1)
ログにて `wait=6.0s` で `fast_fill_defense` が発動し、`negative edge detected (L1)` として multiplier が 1.67 や 2.00 に跳ね上がっているのを確認した。
6秒での約定が本当に「Adverse Selection」なのか、単なる通常の流動性消化なのかを区別できていない可能性がある。

**対策**: `fast_fill_defense` の発動条件を、単なる「約定までの秒数」だけでなく、直前の orderbook の imbalance や trade velocity と掛け合わせた複合条件に引き上げるべきである。過剰な防御は fill rate を構造的に押し下げる。

#### C. Lock 競合による Crash ループ (P0)
`fill_test_events.jsonl` にて `RuntimeError: 別のfill_test プロセスが実行中です (PID=128012)` による crash が記録されている。

**対策**: 166# 7.3 A.1 の指摘通り、起動スクリプト側で PID をチェックし、既存プロセスが生きている場合は例外を吐いて crash ログを汚すのではなく、**正常に exit（または silent skip）** するように二重起動防止ロジックを修正すべきである。

### 8.3 結論とネクストアクション
「待機」している余裕はない。以下の3点を直ちに修正（Hotfix）することを強く推奨する。
1. **在庫リバランス特別緩和モードの実装**（資金枯渇時のレジームスキップ貫通）。
2. **二重起動時の silent exit 化**（Lock 競合による crash ログ汚染の停止）。
3. **Insufficient 警告の cooldown 導入**（ログノイズの削減）。

---

## 9 Hotfix 実装レポート (166# 後半)

**着手日時**: 08-02-25 18:37
**対象**: 7.3 A.1 / 8.2A / 8.2C / 8.3 の3点Hotfix

### 9.1 HF1: balance_forced_rescue 有効化 (8.2A 資金枯渇レジームスキップ デッドロック解消)

**根本原因**: 158# P1-1 で実装済の `balance_forced_rescue` 機構が `FillTestConfig` のデフォルト値 `False` のまま YAML に設定が存在せず、**一度も有効化されていなかった**。

**修正内容**:
- `configs/v460/fill_test.yaml` の `loss_control` セクションに以下を追加:
  `yaml
  balance_forced_rescue_enabled: true
  balance_forced_rescue_offset_mult: 2.0
  `
- 効果: balance_forced 発動時、即座に rescue モード（2倍オフセット）で約定を試行。従来の deadlock_limit=3 サイクル空転を回避。

**コードパス**: `fill_loop_orchestrator.py` L600 `elif rescue_enabled`  即時発動

### 9.2 HF2: Lock 競合 silent exit (7.3 A.1 / 8.2C)

**根本原因**: `LockManager.acquire()` が `RuntimeError` を送出  `fill_test_cli.py` の汎用例外ハンドラが crash イベントとして記録。

**修正内容**:
- `scripts/v460/lib/lock_manager.py`:
  - 新規 `LockConflictError(RuntimeError)` 例外クラス追加
  - `raise RuntimeError(...)`  `raise LockConflictError(...)` に変更（2箇所）
  - `import psutil` をモジュールレベルに移動（テスタビリティ向上）
- `scripts/v460/lib/fill_test_cli.py`:
  - `except LockConflictError` ハンドラ追加（`except KeyboardInterrupt` の前）
  - `logger.info()` で記録し `stop_reason = "lock_conflict"` で正常終了

### 9.3 HF3: Insufficient 警告 cooldown (7.3 B.3 / 8.3)

**根本原因**: `BalanceChecker._check_sell/buy` が毎サイクル `logger.warning()` を発行。1分間隔のサイクルで大量のログノイズを生成。

**修正内容**:
- `scripts/v460/lib/balance_checker.py`:
  - `_insufficient_cooldown_sec: float = 120.0` (per-side 2分クールダウン)
  - `_last_insufficient_log: dict[str, float]` (サイド別タイムスタンプ)
  - `_log_insufficient(side, message)` メソッド: 初回は warning、クールダウン内は debug レベルに抑制
  - `_check_sell` / `_check_buy` 内の直接 `logger.warning()` を `_log_insufficient()` に置換

### 9.4 テスト結果

| テストクラス | テスト数 | 結果 |
|---|---|---|
| TestLockConflictError | 4 | ALL PASSED |
| TestLockManagerConflict | 2 | ALL PASSED |
| TestInsufficientCooldown | 5 | ALL PASSED |
| TestRescueConfig | 2 | ALL PASSED |
| **合計** | **13** | **13 passed** |

回帰テスト: **1988 passed, 1 failed** (既知の `test_train_skip_gate_real` pandas重複ラベル問題、HF無関係)

### 9.5 残課題 (7 / 8 からの未着手項目)

| ID | 優先度 | 内容 | ステータス |
|---|---|---|---|
| 7.1 | P1 | index.md 更新 (162# 以降停止) | 待機 |
| 7.3 A.2 | P0 | blank git_sha 312件の生成パス修正 | 待機 |
| 7.3 B.4 | P1 | trending_up consecutive sell skip 上限緩和 (max=30検討) | 待機 |
| 7.3 B.5 | P1 | AS-R1 velocity skip サンプル蓄積 (R-3 継続) | 蓄積中 |
| 7.3 C.6 | P1-P2 | sklearn feature-name warning (DataFrame統一) | 待機 |
| 7.3 C.7 | P1-P2 | cancel_failed_likely_filled KPI分離 | 待機 |
| 8.2B | P1 | Fast Fill Defense 過剰反応チューニング | 待機 |
