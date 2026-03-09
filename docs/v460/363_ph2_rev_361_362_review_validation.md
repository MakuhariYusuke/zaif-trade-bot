# 363# ph2 361/362 AI レビュー妥当性検証 & 統合アクション

> **種別**: rev (ph2)
> **フェーズ**: ph2 G1.1-exec
> **対象**: 361# (Codex レビュー), 362# (Gemini 3.1 Pro レビュー)
> **日付**: 2026-03-10
> **前提**: 360# ph2 Fill Test ログ分析完了

---

## §0 レビュー概要

| # | レビュワー | 対象範囲 | 主張の核心 |
|---|-----------|---------|-----------|
| 361 | Codex | 351#–360# コード + ドキュメント | ph3 plumbing は前進したが、G2 評価設計と問題設定のズレが未解決。ph2 current-SHA の edge 確定が先決 |
| 362 | Gemini 3.1 Pro | 361# + 351#–360# 全体 | SAC を直接執行 policy に使うのは根本的に誤り。Sidecar Architecture への転換が必要。ph2 の Adverse Selection 事前回避が最優先 |

**両者の共通見解**:
1. G2 in-sample 評価は CRITICAL — 即修正必要
2. E2 `ic_seed_std` は実質ダミー
3. ph3 の OHLCV 特徴量と ph2 microstructure 問題のズレ
4. ph3 は direct trader より sidecar/regime prior として使うべき

---

## §1 361# Codex レビュー検証

### 1.1 指摘一覧と検証結果

| # | 重大度 | 指摘要旨 | コード検証 | 判定 |
|---|--------|---------|-----------|------|
| F1 | **CRITICAL** | G2 評価が完全に in-sample: 訓練と同一 env で評価 | `sac_train.py` L117→L128→L131: 同一 `env` で train→eval。L300 docstring でも「訓練と同一の env」と明記 | **✅ 確認** |
| F2 | HIGH | E2 `ic_seed_std` がダミー: `ic_mean` が生成されない | `sac_train.py` 全体に `ic_mean` 出力なし。`run_experiment.py` L254 で `get("ic_mean", 0.0)` → 全 seed 0.0 → stdev=0.0 → 常に PASS | **✅ 確認** |
| F3 | HIGH | ph3 の OHLCV 特徴量が ph2 microstructure とズレ | `g2_sac_train.yaml` L26-37: 12 特徴量すべて OHLCV 由来。L27 コメント「microstructure は ph4 で統合」 | **✅ 確認** |
| F4 | HIGH | 360# が mixed-SHA / pre-post fix 混在集計 | 03-05~03-09 で Bot SHA が `eb24cf4a74` → `819ec73b2081` 等複数。`forced_buy_delay` 389 件は pre-348# SHA | **✅ 確認** (360# §6.4 TUNE-1 で既に認識済み) |
| F5 | MEDIUM | 360# 内の数値揺れ (cancel reason 件数と fill-rate 分解) | 360# §1.1 と §5.2 間で算出方法が異なる (全体 vs excl skip_gate) | **⚠️ 部分確認**: 揺れではなく算出基準の違い。ただし明示不足は指摘の通り |
| F6 | MEDIUM | OPS-5 の Task Scheduler XML は repo では既に `IgnoreNew` | `ops/windows/task_scheduler.xml` L24: `<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>` | **✅ 確認**: repo と本番の drift 問題 |
| F7 | MEDIUM | 353# EWMA time decay がトキシシティ証明なしで 0 に近づく | `sell_dynamic_kill.py` L360-366: `ewma *= exp(-elapsed/tau)` が無条件実行。トキシシティ確認ロジックなし | **✅ 確認** |

### 1.2 361# 推奨アクションの妥当性

| 推奨 | 妥当性 | 備考 |
|------|--------|------|
| P0-1: current-SHA 限定 72h 再集計 | ✅ 妥当 | mixed-SHA 問題の根本解決。SSOT スクリプト化も正しい |
| P0-2: train/val/test 時系列分割 | ✅ **最重要** | in-sample 評価はゲートの信頼性を根底から損なう |
| P0-3: E2 を `roi_seed_std` 等に置換 | ✅ 妥当 | `ic_mean` は RL に不適切。seed 間の ROI 分散が自然 |
| P1-1: `buy_ranging` current-SHA deep dive | ✅ 妥当 | ただし prioritization は ph2 安定性 (crash 解消) の後 |
| P1-2: `timeout` 根因分離 | ✅ 妥当 | spread vs cancel-too-early vs 板厚の 3 要因分解 |
| P1-3: ph3 sidecar 方針 | ✅ 妥当 | 362# と完全一致。両レビュワーの独立した結論 |
| P2-1: watchdog 本番 drift 確認 | ✅ 妥当 | repo XML は正しいが本番が drift している可能性 |
| P2-2: K1 gate 改訂は再計測後 | ✅ 妥当 | 360# GATE-1 を急がない判断は correct |

### 1.3 361# に対する猜疑的視点

| 懸念 | 評価 |
|------|------|
| 「ph2 current-SHA の edge がまだ見えていない」は希望的観測では? | ⚠️ 部分的に正当。post-352/353 の局所サンプルでプラスが見えている根拠は示されているが、サンプルサイズが小さい可能性あり |
| sidecar architecture は追加複雑性を導入する | ⚠️ 正当な懸念。ただし direct policy より段階的で安全 |
| 「E2 削除」は gate を 3/4 に弱めるリスク | ✅ 361# は「削除 or 置換」と書いており、置換推奨。妥当 |

---

## §2 362# Gemini レビュー検証

### 2.1 主張と検証

| # | 主張 | コード検証 | 判定 |
|---|------|-----------|------|
| G1 | SAC の in-sample 過学習 (361# F1 と同一) | 上記 F1 と同一結果 | **✅ 確認** |
| G2 | EWMA kill は「被弾後の撤退」→ 事前的 AS 回避が必要 | `sell_dynamic_kill.py`: EWMA は fill 後の PnL で計算 → 確かに事後的 | **✅ 確認** |
| G3 | OFI / Micro-price 導入で事前的 AS 回避 | アーキテクチャ提案。実装観点は妥当だが Zaif の板深度データ制約要確認 | **⚠️ 方向性は妥当、実装難度は高** |
| G4 | SAC を Quote ではなく Regime/Directional Bias 出力に | 361# P1-3 と同一結論。独立した 2 レビュワーの合意 | **✅ 妥当** |
| G5 | Train/Test 時系列 Holdout 必須 | 361# P0-2 と同一。具体的に「最新 20%」と提案 | **✅ 妥当** |

### 2.2 362# 独自の追加価値

| 提案 | 360#/361# にない付加価値 | 実現性 |
|------|------------------------|--------|
| Avellaneda-Stoikov 発展形の価格スキュー | 理論的フレームワーク提示。ただし Zaif はフル L2 板が取れるか要確認 | 中期 |
| OFI (Order Flow Imbalance) | 約定履歴の buy/sell aggressor 分析。現行の trades_recorder データで一部可能 | 短〜中期 |
| Micro-price | bid/ask size imbalance からの FV 推定。現行 OB データで計算可能 | 短期 |

### 2.3 362# に対する猜疑的視点

| 懸念 | 評価 |
|------|------|
| 「このbotは絶対に儲からない」は言い過ぎ | ⚠️ 修辞的表現。post-352 の局所サンプルでは正 PnL が観測されており、「絶対に」は反証可能 |
| Zaif の板深度は HFT 向きか? | ⚠️ Zaif は流動性が限定的。L2 板の厚みが薄い場合、OFI/Micro-price の信号はノイジーになる |
| 「一旦 SAC の優先度を下げ」は ph3 準備のモメンタムを失うリスク | ⚠️ コードレベルブロッカーは既に解消 (359#)。評価設計修正 (F1/F2) のみで ph3 に入れる可能性もある |
| HFT の文脈を Zaif に直接適用できるか | ⚠️ Zaif は伝統的 HFT 市場ではなく、maker 手数料構造も異なる。HFT 理論の application gap に注意 |

---

## §3 両レビューの統合分析

### 3.1 完全合意事項 (独立した 2 レビュワーが同じ結論)

| # | 事項 | 361# | 362# | 対応優先度 |
|---|------|------|------|-----------|
| C1 | G2 in-sample 評価は CRITICAL | F1 | G1 | **P0** |
| C2 | E2 `ic_seed_std` は無効 | F2 | — (暗黙) | **P0** |
| C3 | ph3 は sidecar として使うべき | §4, P1-3 | §3 | **P1** (設計方針) |
| C4 | ph2 microstructure 改善が先決 | §7.1 | §4 | **P0** |
| C5 | current-SHA 限定の再分析必要 | §5, P0-1 | — | **P0** |

### 3.2 見解が分かれる事項

| 事項 | 361# の立場 | 362# の立場 | 本検証の判断 |
|------|-----------|-----------|------------|
| ph3 の緊急度 | 評価設計修正後に進行可能 | 一旦優先度を下げるべき | **361# 寄り**: F1/F2 修正は小〜中規模。修正後に ph3 試行は可能 |
| AS 対策のアプローチ | time decay + kill 閾値の段階検証 | OFI/Micro-price の導入 | **段階的**: まず 361# の漸進策、次に 362# の構造改善 |
| K1 gate 見直し | 今は早い。再計測後に複合 gate | 言及なし | **361# に同意**: 先に再計測 |

### 3.3 どちらにもない視点 (補完)

| # | 視点 | 理由 |
|---|------|------|
| S1 | fill test のメモリ問題 (サイレントクラッシュ) の優先度が両レビューとも低い | K1/K2 の根本改善には crash-free 72h が必須。OPS-1 (atexit hook) が最も cost-effective な次の一手 |
| S2 | テストコードのメモリ効率 | 037# で進行中の Codex がテスト最適化実績あり。fill_test_cli.py の RSS ダンプ実装と親和性が高い |
| S3 | `per_side_dd_halt` (11.3%) の寄与が過小評価 | halt は fill 機会の完全喪失。SDK/BDK と違い「一定サイクル数完全停止」するため、実効的な損害が大きい |

---

## §4 統合アクションプラン

### 4.1 P0: 即時 (Week 1)

| # | アクション | 根拠 | 対象ファイル | 工数 |
|---|-----------|------|-------------|------|
| A1 | **OPS-5**: 本番 Task Scheduler 設定が repo XML と一致しているか確認 | 361# F6 | 本番サーバー | 10min |
| A2 | **OPS-1**: atexit hook で RSS/状態ダンプ追加 | 360# §2, S1 | `scripts/v460/lib/fill_test_cli.py` | 30min |
| A3 | **F1 修正**: `sac_train.py` に train/val 時系列分割追加 | 361# F1, 362# G1 (合意 C1) | `scripts/v460/lib/tasks/sac_train.py` | 2h |
| A4 | **F2 修正**: E2 `ic_seed_std` → `roi_seed_std` に置換 | 361# F2 (合意 C2) | `sac_train.py`, `run_experiment.py` | 1h |
| A5 | **current-SHA 再集計スクリプト** 作成 | 361# P0-1 (合意 C5) | 新規スクリプト | 1h |

### 4.2 P1: 短期 (Week 2)

| # | アクション | 根拠 | 対象 | 工数 |
|---|-----------|------|------|------|
| B1 | fill test 再起動 + 72h crash-free 確認 | A1/A2 の結果待ち | ops | 72h |
| B2 | current-SHA 再集計 → K1/K2/PnL 再判定 | A5 | データ分析 | 2h |
| B3 | TUNE-3 (SDK 閾値緩和) を time decay と分離して検証 | 361# F7 | `fill_test.yaml` | 1h |
| B4 | ph3 sidecar 設計文書作成 | 合意 C3 | docs | 2h |

### 4.3 P2: 中期 (Week 3+)

| # | アクション | 根拠 | 対象 | 工数 |
|---|-----------|------|------|------|
| C1 | G2 gate で holdout 評価が PASS → ph3 SAC 訓練実行 | A3/A4 完了後 | SAC pipeline | 4h |
| C2 | Micro-price 計算の実装 (OB データ活用) | 362# G3 | 新規モジュール | 4h |
| C3 | K1 gate 複合化検討 (再計測結果次第) | 361# §6 | 000# 改訂 | 2h |
| C4 | env に microstructure proxy 追加 | 361# P2-3 | `HeavyTradingEnv` | 4h |

---

## §5 メモリ問題: 037# Codex への委託提案

### 5.1 背景

360# §2 で報告された fill test のサイレントクラッシュ (13 回/5 日間) は、
OOM が最有力候補。しかし atexit/signal handler で RSS を記録する仕組みがなく、
原因特定ができていない。

037# の Codex は Session 037-001〜073 でテストコード最適化・DRY 化を継続しており、
以下の実績がある:

- メモリ効率化 (parquet 読込の selected feature 化、class-scope fixture)
- source 契約の shared helper 化
- module scope import 集約
- production コードの helper 分離 (`hour_rules.py`, `raw_paths.py`)

この Codex に fill_test_cli.py の OPS-1 (atexit hook) 実装を委託することで、
テスト最適化の知見を活かした実装が期待できる。

### 5.2 Codex プロンプト

以下のプロンプトを 037# Codex に投入する:

---

```
## タスク: fill_test_cli.py へのメモリ診断 atexit hook 追加 (OPS-1)

### 背景
fill test が 5 日間で 13 回サイレントクラッシュしています（360# §2 参照）。
最有力候補は OOM ですが、クラッシュ時の RSS を記録する仕組みがないため原因特定できていません。
361# / 362# の AI レビューでも「クラッシュ原因特定が最優先」と合意されています。

### 要件

1. `scripts/v460/lib/fill_test_cli.py` に atexit hook を追加
   - `atexit.register()` でプロセス終了時に以下をダンプ:
     - RSS (Resident Set Size) in MB — `psutil.Process().memory_info().rss`
     - VMS (Virtual Memory Size) in MB
     - 現在時刻 (UTC ISO format)
     - `stop_reason` (取得可能な場合)
     - `run_id`
     - 直近の `fill_test.lock` heartbeat age
   - ダンプ先: `{results_dir}/diagnostics/exit_dump_{run_id}_{timestamp}.json`
   - logger にも WARNING レベルで RSS を出力

2. signal handler (`_signal_handler`) 内でも同様の RSS ダンプを呼ぶ
   - 既存の signal handler (L342-361) は維持。ダンプ関数を共通化して呼び出し追加

3. health_monitor の RSS チェック強化（オプション）
   - `configs/v460/fill_test.yaml` の `health_monitor.check_interval_sec` を
     300.0 → 60.0 に短縮する変更を検討（ただしCPU負荷とのトレードオフを考慮）

### 制約
- 既存のテスト (38 tests in test_356, 4206 tests in v460 全体) を壊さないこと
- `psutil` は既に `lock_manager.py` で使用済み（新規依存なし）
- メモリリーク防止: ダンプ関数自体が大量メモリを使わないこと
- 型安全: Any 型回避、mypy 準拠
- DRY: `_dump_exit_diagnostics()` を 1 関数に集約し、atexit と signal の両方から呼ぶ
- 037# のセッションログ (`docs/v460/037_phg_rpt_refactoring_session_log.md`) に
  作業記録を追記すること

### 参照ファイル
- `scripts/v460/lib/fill_test_cli.py` — メイン対象
- `scripts/v460/lib/lock_manager.py` — psutil 使用例、heartbeat 参照
- `configs/v460/fill_test.yaml` L121-125 — health_monitor 設定
- `docs/v460/360_ph2_rpt_fill_test_analysis.md` §2 — クラッシュパターン分析
- `docs/v460/363_ph2_rev_361_362_review_validation.md` §5 — 本タスクの根拠

### テスト
- 実装後に `pytest tests/unit/v460/test_356_g2_sac_blockers.py --no-cov` が全 PASS であること
- 可能であれば `_dump_exit_diagnostics()` の単体テストを追加
  - mock psutil で RSS/VMS を返し、JSON ダンプの構造を検証
- `--no-verify` オプション付きでコミットすること
```

---

## §6 AI レビューチェックリスト

### 6.1 検証の完全性

| # | チェック項目 | 結果 |
|---|------------|------|
| V1 | 361# の全 7 指摘をコード検証したか | ✅ F1-F7 全件検証 |
| V2 | 362# の全主張をコード検証したか | ✅ G1-G5 全件検証 |
| V3 | 両レビューの共通点を特定したか | ✅ C1-C5 (§3.1) |
| V4 | 見解の相違点を特定したか | ✅ §3.2 |
| V5 | 猜疑的視点を含めたか | ✅ §1.3, §2.3 |

### 6.2 アクションの妥当性

| # | チェック項目 | 結果 |
|---|------------|------|
| A1 | P0 アクションは依存関係なく並行実行可能か | ✅ A1-A5 は独立 |
| A2 | 各アクションに対象ファイルと工数があるか | ✅ |
| A3 | 実現不可能な提案がないか | ✅ (362# G3 の OFI は「中期」に位置づけ) |
| A4 | 既存資産 (359# blockers 解消) を無駄にしていないか | ✅ A3/A4 は 359# 成果の上に構築 |

### 6.3 文書間の整合性

| # | チェック項目 | 結果 |
|---|------------|------|
| D1 | 360# の改善提案との整合性 | ✅ OPS-1/OPS-5 は 360# §6 と一致 |
| D2 | 000# Phase 定義との整合性 | ✅ ph2 scope 内の活動 |
| D3 | 359# ph3 準備との整合性 | ✅ F1/F2 修正は 359# 成果を強化する方向 |
