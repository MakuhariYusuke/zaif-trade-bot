# 134# ph2 rev: 133# 妥当性評価

| key | value |
|---|---|
| 番号 | 134 |
| フェーズ | ph2 |
| 種別 | rev (評価) |
| 対象 | `docs/v460/133_ph2_rev_132_profitability_max_plan.md` |
| 作成日 | 2026-02-22 |
| 前提 | Git `cc1b1bf97` (133# P0 実装済み: P0-01/02/05/06/08/09/10 + Y1-Y7) |
| 結論 | **133# 本体の分析・提案は高品質。Gemini セカンドオピニオンは一部過剰（P2-05→P0 昇格は不要）。未実装 P0 のうち P0-03/04 が最優先、P0-07/12 は中優先、P0-11 は P1 降格妥当。** |

---

## §0 エグゼクティブサマリ

133# は 132# に対する外部レビューとして、分析精度・網羅性ともに高い。42施策の優先リストは概ね合理的であり、§4 テーブルの P0 群は「止血」として正しい階層付けがなされている。

ただし以下の点で修正・補足が必要:

1. **P0-11 (hot-reload 強化)** は既に本番稼働レベルで実装済みであり、**P1 降格が妥当**
2. **Gemini P2-05→P0 (inventory-aware quoting)** は、P0-08 で `skip_balance_forced` を実装済みのため **P0 昇格不要**。現行アーキテクチャ（交互発注 MM、ポジション非保有設計）では inventory skew の概念自体が限定的
3. **P0-03/04 (trades データパイプライン)** が真の最優先。retrain 品質の根幹であり、133# の「止血」思想とも整合
4. **P1/P2/P3** 群は工数対効果の観点で再分類を推奨

---

## §1 133# 本体の妥当性判定

### §1.1 分析の正確性

| 項目 | 133# の主張 | コード検証結果 | 判定 |
|---|---|---|---|
| C1: Simpson 型リスク | 全体 WATCH が最新 run FAIL を隠蔽 | `gate_judgment.py` は all-run 一括評価のみ。per-run API 不在を確認 | ✅ **正確** |
| C2: --once history 欠落 | `retrain_scheduler.py:1491` で history 未記録 | 実際に `--once` ブランチに history 書込みなし（P0-02 で修正済み） | ✅ **正確** |
| C3: analyze_fill_detail crash | `cancel_reason=None` + 列名不整合 | 3箇所のバグを確認し修正済み（P0-06） | ✅ **正確** |
| C4: trades 欠損構造化 | 20/21 日の trades 欠落 | `data/v460/raw/trades/` は 20260213-19 のみ。20 以降なし。`feature_enricher.py:418-440` で全量 fallback | ✅ **正確** |
| C5: balance_forced_switch 損失 | 平均 -1.98bps | BalanceChecker のフロー、side_selector の freeze ロジックと整合。P0-08 で skip 実装済み | ✅ **正確** |

**分析精度 5/5。** 133# の指摘は全て現行コードベースで再現可能。

### §1.2 提案の妥当性

| ID | 133# 判定 | 検証後判定 | 理由 |
|---|---|---|---|
| P0-01 | P0 | ✅ P0 → **実装済み** | `max(0, ...)` clamp で解決。retrain 再始動の前提 |
| P0-02 | P0 | ✅ P0 → **実装済み** | `--once` ブランチに history 追記。監査完全化 |
| P0-03 | P0 | ✅ **P0 維持** — **最優先** | trades 欠損は retrain 品質に直結。`run_observation.py` 停止が原因 |
| P0-04 | P0 | ✅ **P0 維持** — **最優先** | `OBRecorder` 対称の `TradesRecorder` で fill_test 内蔵化 |
| P0-05 | P0 | ✅ P0 → **実装済み** | P0-08/09/10 の全 skip パスで FillRecord 生成 |
| P0-06 | P0 | ✅ P0 → **実装済み** | 3箇所バグ修正完了 |
| P0-07 | P0 | ⚠️ **P0 だが中優先** | per-run 評価は `gate_judgment.py` に `--run-id` arg 追加で対応可。工数 ~0.3日 |
| P0-08 | P0 | ✅ P0 → **実装済み** | `skip_balance_forced` + FillRecord 記録 |
| P0-09 | P0 | ✅ P0 → **実装済み** | `skip_buy_unknown_regime` + FillRecord 記録 |
| P0-10 | P0 | ✅ P0 → **実装済み** | sell dynamic kill (rolling50 mean < -0.5bps) |
| P0-11 | P0 | ❌ **P1 へ降格** | hot-reload は `skip_gate_evaluator.py` で既に本番稼働。SHA256 変更検出 → 120s 間隔で暗黙リトライ → 失敗時は前モデル維持。致命的問題なし |
| P0-12 | P0 | ⚠️ **P0 だが中優先** | `gate_judgment.py` で既に G1.2 CLI 呼び出し可能。`run_gate_check.py` は deprecated 注記あり。統一化は整理の問題 |

### §1.3 §4 テーブルの全体評価

**P0 群 (12 項目):** 7 項目実装済み、2 項目最優先、2 項目中優先、1 項目 P1 降格。配分は妥当。

**§6 推奨実行順（48h）:** 合理的。ただし Step 1「止血」は今回で大部分完了。次は Step 3「再計測」と Step 4「trades 修復→retrain 再開」を並行すべき。

---

## §2 未実装 P0 の精密評価

### §2.1 P0-03: trades 欠損の原因特定 [最優先]

**現状:**
- `run_observation.py` は fill_test とは完全に別プロセス（observation-only、注文なし）
- OB + trades 両方を `MarketDataCollector` 経由で蓄積
- `data/v460/raw/orderbook/` は 20260221 まで存在（fill_test 内蔵 `OBRecorder` が記録）
- `data/v460/raw/trades/` は 20260219 で途切れ → **`run_observation.py` が停止したと推定**

**対策:**
- 即座に `run_observation.py` を再起動（データ欠損日数を最小化）
- systemd/タスクスケジューラで自動再起動を設定
- P0-04 と併せて fill_test 内蔵化で二重系に

**工数:** 運用対応 ~0.1日 + 自動化 ~0.2日

### §2.2 P0-04: fill_test 内 TradesRecorder 追加 [最優先]

**現状:**
- `OBRecorder` (`scripts/v460/lib/ob_recorder.py`) が板データを JSONL 記録 — **trades 版は存在しない**
- fill_test は毎サイクル `get_recent_trades()` で約定データを取得しているが、記録せず消費

**実装方針:**
- `OBRecorder` をベースに `TradesRecorder` を新規作成（構造はほぼ同一）
- `run_fill_test.py` の `get_recent_trades()` 呼び出し結果を TradesRecorder にも渡す
- 保存先: `data/v460/raw/trades/YYYYMMDD/trades_HHMMSS.jsonl`
- これにより `run_observation.py` 停止時も trades が途切れない（二重系）

**工数:** ~0.5日

### §2.3 P0-07: run 単位ゲート標準化 [中優先]

**現状:**
- `gate_judgment.py:_load_all_records()` は glob で全 `fill_records_*.jsonl` を一括読込
- `g1_2_full_judgment()` は渡された records を一括評価（per-run 分解なし）
- FillRecord には `run_id` フィールドが存在するため、グループ化は容易

**実装方針:**
- `gate_judgment.py` に `--run-id` オプション追加
- `--latest-run` で最新 run_id のみフィルタするショートカット
- 表示: `[ALL] G1.2=WATCH` + `[LATEST] G1.2=FAIL` の対比出力

**工数:** ~0.3日

### §2.4 P0-12: run_gate_check.py の統一 [中優先]

**現状:**
- `run_gate_check.py:L211` に deprecated 注記あり、新規利用は `gate_judgment.py` 推奨
- `gate_judgment.py` は既に G1.1 + G1.2 の統合判定を実装

**判断:** P0-07 と同時に対応すべき。`run_gate_check.py` を `gate_judgment.py` へのラッパーに書き換えるか、deprecation を完了させる。

**工数:** P0-07 と合算で ~0.4日

### §2.5 P0-11: hot-reload 自動ハッシュ再生成 [P1 降格推奨]

**現状:**
- `skip_gate_evaluator.py:L125` — SHA256 ハッシュ比較で変更検出
- `skip_gate_evaluator.py:L176` — 失敗時は `logger.error` + 前モデル維持
- 120s 間隔の暗黙リトライで、次回チェック時に再試行
- `retrain_scheduler.py:L1344-1366` — deploy 時にアトミック rename + `.pkl.sha256` 並行管理

**理由:** 現行動作で致命的問題がない。「失敗46回/成功1回」のログデータは retrain 側が skip → deploy 自体が稀だったことに起因しており、hot-reload メカニズムの故障ではない。明示的リトライ回数の設定化は nice-to-have。

---

## §3 P1 群の評価

| ID | 施策 | 判定 | 根拠 |
|---|---|---|---|
| P1-01 | buy/sell 完全分離モデル | ⚠️ **P1 維持、ただし P0-03/04 後** | sell 構造負けは明確だが、trades 欠損下で retrain しても意味なし。データ修復後に着手 |
| P1-02 | target 二層化 (buy=pnl30, sell=pnl120) | ✅ **P1 妥当** | 120s で回復するなら sell の評価 horizon を伸ばすのは理にかなう。P1-01 と同時検討 |
| P1-03 | score 校正 (isotonic/quantile) | ✅ **P1 妥当** | FillRecord に `skip_gate_score` + `post_fill_30s_pnl_bps` が蓄積済み。事後分析から着手可。`CalibratedClassifierCV` は AS 分類器で使用済みだが、ライブスコアの校正は未実装 |
| P1-04 | regime 別閾値 | ✅ **P1 妥当** | 一律閾値はレジーム毎の勝率差を吸収できない。P0-09 の unknown-buy スキップは応急処置であり、恒久的にはレジーム別閾値が望ましい |
| P1-05 | skip_gate_ratio 上限超過時の自動 degrade | ⚠️ **P2 降格推奨** | 最新 run で 30.8% は高いが、P0-09/10 の skip で改善する可能性あり。再計測後に判断すべき |
| P1-06 | stale reprice 売側上限縮小 (2→1) AB | ✅ **P1 妥当** | reprice=2 平均 -3.44bps は明確な損失源。AB テスト容易 |
| P1-07 | timeout 動的化 (90s→動的) | ⚠️ **P2 降格推奨** | timeout 195件は大きいが、動的化の閾値設計に regime 分析が必要。P1 群の止血完了後 |
| P1-08 | spread 狭小時の「休む」判定 | ✅ **P1 妥当、既存 too_narrow 拡張** | `too_narrow` 61回。既存の spread 判定を拡張するだけで対応可 |
| P1-09 | balance margin 動的化 | ⚠️ **P2 降格推奨** | P0-08 で balance_forced スキップ済み。margin の動的化は二次的改善 |
| P1-10 | preflight 失敗連続→run pause | ✅ **P1 妥当** | dead-cycle 抑止。`SAFE_STOP` (044#) が既にあるが、pause（一時停止→自動再開）は別概念として有用 |
| P1-11 | PnL 評価を fee/slippage 控除後で統一 | ✅ **P1 妥当** | 現行は楽観寄り。実収益一致は信頼性の前提 |
| P1-12 | 直近 N fill のみで online 比較 | ✅ **P1 妥当** | 非定常環境では全履歴平均は過去に引きずられる。P0-10 の sell dynamic kill が部分的に対応 |

**P1 まとめ:** 12 項目中 9 項目は P1 維持妥当、3 項目は P2 降格推奨。P1 の着手順は P1-01/02 (side 分離) → P1-03 (score 校正) → P1-06 (reprice 上限) を推奨。

---

## §4 P2/P3 群の評価

### §4.1 P2 群

| ID | 施策 | 判定 | 根拠 |
|---|---|---|---|
| P2-01 | v458 WalkForward を retrain 判定へ移植 | ⚠️ **実現可能だが接合コスト高** | `ztb/evaluation/walk_forward/` は SAC/PPO 評価用。LGBM retrain 用にはアダプタ層が必要。工数 ~1日 |
| P2-02 | v459 統計 gate を deploy 判定に常時化 | ⚠️ **冗長性あり** | `ztb/metrics/gate_checks.py` の Holm-Bonferroni は既に `g1_2_full_judgment()` 内部で使用済み。追加インパクト小 |
| P2-03 | run_observation.py を fill_test と同時運転 | ✅ **P0-04 と統合推奨** | P0-04 で fill_test 内蔵化すれば、run_observation.py は補助的な二重系に格下げ |
| P2-04 | oracle 上限を毎日再計測し KPI 化 | ✅ **有用** | `oracle_baseline.py` (131# D2) が既に実装済み。cron/タスクスケジューラで日次実行するだけ |
| P2-05 | inventory-aware quoting | ❌ **P0 昇格不要** | 後述 §5.A |
| P2-06 | worst hour-side 局所ルール学習 | ✅ **P1-04 と統合推奨** | レジーム×時間帯の交差分析は有望。P1-04 (regime 別閾値) の拡張として位置づけ |
| P2-07 | execution trace 因果ログ標準化 | ⚠️ **工数大** | decision→fill→pnl の因果チェーンは FillRecord で部分的に追跡可能。完全標準化は大改修 |
| P2-08 | shadow model 配信 (A/B 採点) | ✅ **有用だが P2 妥当** | hot-reload のアトミック性は確保済み。shadow は安全だが工数対効果でP2 |
| P2-09 | run 開始時の前提健全性チェック | ✅ **P1 昇格推奨** | `trades` 当日ファイル必須チェックは P0-03/04 の補完として即効。工数 ~0.1日 |
| P2-10 | Gate に「最新 run hard floor」追加 | ✅ **P0-07 と同時実装** | per-run G1.2 の hard floor は P0-07 の自然な拡張 |

### §4.2 P3 群

| ID | 施策 | 判定 | 根拠 |
|---|---|---|---|
| P3-01 | v458 hft_proxies を boardless fallback | ⚠️ **優先度低** | fill_test は tick レベル板データを直接保有。`hft_proxies.py` は 1分足ベースで精度劣後 |
| P3-02 | advanced_regime_detector AB | ✅ **有望だが P3 妥当** | unknown レジーム削減には寄与するが、P0-09 で応急処置済み |
| P3-03 | dynamic_position_sizer を lot_sizing へ橋渡し | ⚠️ **アーキテクチャ変更要** | 現行の固定ロット + balance_shrink と設計思想が異なる。慎重な検討が必要 |
| P3-04 | pnl_monte_carlo 日次自動実行 | ✅ **P2 昇格推奨** | 期待値悪化の早期検知は運用安全性に直結。`ztb/risk/pnl_monte_carlo.py` 存在済み |
| P3-05 | venue 横断比較 (Coincheck/Bitflyer) | ✅ **P3 妥当** | 000# 前提と整合。現時点ではリソース集中が必要 |

---

## §5 Gemini セカンドオピニオンの評価

### A. 在庫管理 (P2-05→P0 昇格): ❌ 不要

**Gemini の主張:** 「マーケットメイカーにおいて在庫の偏りは死を意味する。inventory-aware quoting を P0 に格上げすべき」

**反論:**
1. **P0-08 で既に対応済み。** `skip_balance_forced` により、強制切替時のサイクル自体をスキップ。平均 -1.98bps の損失源を遮断した
2. **現行アーキテクチャの特性。** 本システムは「ポジションを長期保有しない交互発注 MM」であり、古典的な MM の continuous inventory skew とは構造が異なる。各サイクルは buy→sell の交互で、ポジション累積は BalanceChecker + side_selector freeze (120# A5) で制御済み
3. **工数対効果。** inventory-aware quoting はゼロから新規実装で 2-3日。P0-08 のスキップで主要損失は回避済みであり、残りの改善幅に対して工数が見合わない
4. **ただし long-term では有用。** P1 群でロット最適化を検討する際に、inventory 偏りを考慮するのは合理的。**P2 維持が妥当**

### B. アルファ vs ベータの切分け: ✅ 正当な指摘

**Gemini の主張:** 「プラス寄与（trending-buy 等）が地合いによるベータなのかモデルの真のアルファなのか切り分けるべき」

**評価:**
- これは本質的かつ重要な批判。`oracle_baseline.py` (131# D2) は理論上限を算出するが、「ランダムエントリーとの比較」は未実装
- **対策:** P2-04 (oracle 日次 KPI 化) に加え、ランダムエントリーベースラインとの差分分析を追加すべき
- **優先度:** P1-03 (score 校正) と並行して分析可能。データは FillRecord に既蓄積

### C. データパイプライン GIGO: ✅ 完全に正しい

**Gemini の主張:** 「trades 欠損下で retrain を回すことはモデルを意図的に破壊しているに等しい」

**評価:**
- 133# C4 + Gemini の指摘を合わせて、P0-03/04 は文句なしの最優先。本評価でも §2.1/2.2 で最優先とした
- `feature_enricher.py:418-440` の全量 fallback は時間整合性を完全に破壊する
- **追加対策:** P2-09 (run 開始時の trades 当日ファイル必須チェック) を P1 昇格し、trades 不在時は retrain をブロックするガードを追加すべき

### D. Phase 3 (SAC) 封印: ✅ 支持

**Gemini の主張:** 「現状のマイナス期待値環境下で SAC に突入することは絶対に避けるべき」

**評価:**
- 完全に同意。133# P0/P1 完遂 + n > 2000 で G1.2 FULL PASS まで、SAC/RL は封印を継続
- `oracle_baseline.py` の ph3 進入判定ロジックがこの gate として機能する

---

## §6 §5 (vXXX 資産再利用) の評価

| 候補 | 存在 | 再利用可否 | コメント |
|---|---|---|---|
| `ztb/evaluation/walk_forward/` | ✅ | ⚠️ 接合要 | SAC/PPO 用。LGBM retrain にはアダプタ必要 |
| `ztb/metrics/gate_checks.py` | ✅ | ⚠️ 冗長 | Holm-Bonferroni は既に `fill_quality.py` 内で使用中 |
| `ztb/features/hft_proxies.py` | ✅ | ❌ 優先度低 | 1分足ベース。fill_test は tick レベル板データ直接保有 |
| `ztb/analysis/regime/advanced_regime_detector.py` | ✅ | ✅ 有望 | unknown レジーム削減に寄与。P3-02 として中長期検討 |
| `ztb/risk/dynamic_position_sizer.py` | ✅ | ⚠️ 設計差 | 固定ロット設計との整合が必要 |
| `ztb/risk/pnl_monte_carlo.py` | ✅ | ✅ 即利用可 | P3-04 → P2 昇格推奨。日次 cron で赤信号通知 |

---

## §7 修正後の実行ロードマップ

### Phase A: データインフラ復旧 (Day 0-1) [最優先]

| ID | 施策 | 工数 | 状態 |
|---|---|---|---|
| P0-03 | trades 欠損原因特定 + run_observation 再起動 | 0.3日 | **次に着手** |
| P0-04 | TradesRecorder fill_test 内蔵化 | 0.5日 | **次に着手** |
| P2-09→P1 | run 開始時 trades 当日ファイル必須チェック | 0.1日 | P0-04 と同時 |

### Phase B: 観測性強化 (Day 1-2)

| ID | 施策 | 工数 | 状態 |
|---|---|---|---|
| P0-07 | gate_judgment.py に per-run 評価追加 | 0.3日 | |
| P0-12 + P2-10 | run_gate_check 統一 + latest-run hard floor | 0.2日 | P0-07 と同時 |

### Phase C: 再計測 (Day 2-3)

- 同一 Git SHA / 同一 YAML で **24h 連続 run**
- run 単位 Gate 判定（Phase B で実装したもの）
- Oracle 対比で alpha/beta 切分け初期分析

### Phase D: retrain 再始動 (Day 3-5)

- Phase A で trades パイプライン復旧後
- retrain 定期ループ再開（`--once` ではなく）
- deploy/reject を history で追跡（P0-02 で修正済み）

### Phase E: P1 群着手 (Day 5+)

1. P1-01/02: side 分離モデル + target 二層化
2. P1-03: score 校正（FillRecord データで事後分析→リアルタイム化）
3. P1-06: stale reprice 売側上限縮小 AB

---

## §8 総合判定

| 評価軸 | スコア | コメント |
|---|---|---|
| 分析精度 | **5/5** | C1-C5 全て正確。コードレベルで再現可能 |
| 提案の合理性 | **4/5** | P0 群は妥当。P0-11 のみ P1 降格が適切 |
| 優先順位付け | **4/5** | §6 の 48h プランは概ね合理的。trades インフラ復旧のDay 0化が必要 |
| 網羅性 | **5/5** | 42 施策は v460 の問題空間をほぼカバー |
| Gemini セカンドオピニオン | **3/5** | B (alpha/beta), C (GIGO), D (SAC封印) は正当。A (P2-05→P0) は過剰 |
| 実行可能性 | **4/5** | 大部分は既存コードベースの拡張で対応可能。新規大改修は P3 のみ |

**最終結論:** 133# の「止血優先→再計測→retrain再開→モデル改善」の実行順は正しい。本評価で修正した優先度（P0-11 降格、P2-09 昇格、P2-05 非昇格）を反映し、Phase A (trades インフラ) から着手すべきである。
