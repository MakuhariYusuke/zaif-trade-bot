# 130# ph2 実装報告 — 128#/129# 分析結果の実装と改善

> **ドキュメント番号**: 130#  
> **セッション**: 129.1#–129.3# (ドキュメント 129# を起点とした実装セッション群)  
> **日付**: 2026-02-21  
> **Git 範囲**: `b525b3a8a`..`HEAD`  
> **前提文書**: 128# (ログレビューと方策), 129# (ログ分析・残課題統合レビュー + Appendix D Codex レビュー)  
> **目的**: 128#/129# で特定された改善施策の実装結果を外部 AI コーディングエージェントのレビュー用に集約

---

## §0 エグゼクティブサマリ

### セッション番号の是正

本書作成に至るまでの実装コミットでは、**131#–133#** というドキュメント番号にない番号をセッション識別子として誤用していた。000# §5 の規則に基づき是正:

| 誤用ラベル | 正しいセッション | 内容 | 主コミット |
|-----------|----------------|------|-----------|
| "130#" | 129.1# | 128# 分析結果 6 施策実装 | `b525b3a8a`, `9df2715ea` |
| "131#" | 129.1# (続) | retrain pnl30 target + hot-reload | `2780f80b0`, `7cb39ebb5` |
| "132#" | 129.2# | Codex レビュー (129# Appendix D) 対応 | `0e5dcf71b`, `e8af70aad` |
| "133#" | 129.3# | YAML チューニング + balance filter + trades I/O | `8b466d555` |

コード内のコメントは本書確定に伴い **130#** (本ドキュメント) または **129#** (レビュー起源) に統一済み。

### 大義への距離

000# §0 の大義: **「短期間での高収益性システム」**。  
現在 ph2 G1.1-exec gate の fill_test 168h 実測中。

| 指標 | 全期間 (1609 cycles) | 最新 run (13 cycles) | 傾向 |
|------|---------------------|---------------------|------|
| Fill rate | 65.1% | — (データ蓄積中) | — |
| PnL30 mean | -0.173 bps | — | ⬆️ 改善傾向 (02/21: -0.221→-0.10) |
| AS rate | ~30.8% → 5.0% (前 run 末) | — | ⬆️⬆️ 劇的改善 |
| Cumulative PnL | -168.7 JPY | — | キャップ 10K JPY の 1.7% |
| retrain deploy | 0 回 | bootstrap 13/25 | ❌ 未到達 |

---

## §1 実装一覧

### 1.1 セッション 129.1# — 128# 分析結果の実装

**コミット**: `b525b3a8a` (6 施策), `9df2715ea` (DRY リファクタ), `2780f80b0` (retrain 改善), `7cb39ebb5` (trades fallback)

| # | 施策 | 出典 | ファイル | 詳細 |
|---|------|------|---------|------|
| S1 | UTC21 sell ブロック追加 | 128# §4.2 | `fill_test.yaml` | `skip_utc_hours_sell` に 21 追加。JST06 sell -1.136bps n=42 |
| S2 | Unknown regime buy guard | 128# §4.3 | `maker_price.py`, `fill_config.py` | `unknown_buy_offset_boost: 2.0` で offset を VG 相当に引上げ |
| S3 | Postonly 二重確認 (E1) | 128# §4.5 | `run_fill_test.py` | 発注直前に mid price 再取得、best_bid/ask と比較 |
| S4 | Orderbook error 細分化 | 128# §4.6 | `run_fill_test.py` | timeout/rate_limit/empty/guard_reject の 4 区分 |
| S5 | Bootstrap 2 段化 | 128# §5 | `retrain_scheduler.py`, `fill_test.yaml` | `bootstrap_min_total=30, min_new=10` |
| S6 | I/O 日付限定ロード | 128# §5 | `feature_enricher.py` | date_filter で OB 99.9% 削減 |
| S7 | Gate 判定統一 (K1-K6 + F1-F8) | 129# §4.2 | `monitor_fill_test.py` | 二段階ゲート表示 |
| S8 | DRY: `_read_jsonl_gz` 統合 | — | `feature_enricher.py` | `market_data_collector` から import |
| S9 | retrain target: pnl30 | 118# §4 | `retrain_scheduler.py`, `fill_test.yaml` | pnl120 coverage 40% → pnl30 100%。データ量 2.5x |
| S10 | Config hot-reload | 118# §8.3 | `retrain_scheduler.py` | per-cycle YAML 再読込。YAML 変更で再起動不要 |
| S11 | Trades date_filter fallback | 130# S6 副作用 | `feature_enricher.py` | 当日 trades ファイル不在時に `date_filter=None` でフォールバック |

### 1.2 セッション 129.2# — 129# Appendix D (Codex レビュー) 対応

**コミット**: `0e5dcf71b` (コード), `e8af70aad` (ドキュメント)

| # | 施策 | 出典 | ファイル | 詳細 |
|---|------|------|---------|------|
| R1 | Lock heartbeat 強化 | 129# D.3 | `run_fill_test.py` | ロックファイルに heartbeat タイムスタンプ追加。形式: `PID\|created_ts\|run_id\|heartbeat_ts` |
| R2 | Stale lock 自動回収 | 129# D.3 | `run_fill_test.py` | PID alive でも heartbeat_age > 1800s なら stale 判定・回収 |
| R3 | `balance_forced_switch` フラグ | 129# D.2 | `fill_quality.py`, `run_fill_test.py` | FillRecord に残高制約フラグ追加。サイクルごとに自動記録 |
| R4 | 129# Appendix E 追記 | 129# D.1-D.3 | `129_ph2_rpt_*.md` | 事実照合・重要論点・アクション全件回答 |

### 1.3 セッション 129.3# — YAML チューニング + 残施策

**コミット**: `8b466d555`

| ID | 施策 | 変更 | 根拠 |
|----|------|------|------|
| Y1 | `bootstrap_min_total_samples` | 30→**25** | 初回 deploy 加速。短期高収益の大義 |
| Y2 | `min_spread_jpy` | 1500→**1200** | 129# D.3 Q4: 1200≈1.14bps で安全マージン維持。fill_rate +10~15pt |
| Y4 | `vpin_threshold` | 0.70→**0.63** | VG 有効性確認済 (118# §9-A4)。-10% で AS -3~5pt 期待 |
| Y5 | balance filter | 新規 | retrain_scheduler で `balance_forced_switch=True` レコード除外 |
| Y7 | trades I/O 最適化 | 全量→7日 window→全量の 3 段階 | 4.4M 行 → ~50 万行。retrain cycle I/O 30s→数秒 |

---

## §2 retrain 運用状況と課題

### 2.1 現在の retrain 状態

| 項目 | 値 |
|------|-----|
| 稼働プロセス | PID 17716/99272 (retrain_scheduler) |
| Git SHA | `7cb39ebb5` |
| 最新 run records | **13 件** (run_id `1771663547_044513cf`) |
| Bootstrap 閾値 | 25 (YAML hot-reload で 30→25 に反映済予定) |
| Bootstrap 状態 | 13/25 → **あと 12 records で初回 deploy** |
| `latest_run_only` | `true` → 旧 1048 filled は **学習から除外** |
| trades fallback | 全量 4.4M 行ロード (~27 秒/cycle) |

### 2.2 コード変更の反映状況

| 変更 | hot-reload? | 稼働中プロセスへの反映 |
|------|------------|---------------------|
| Y1 (bootstrap 25) | ✅ YAML | 次 retrain cycle で反映 |
| Y2 (spread 1200) | ✅ YAML | 次 fill cycle で反映 |
| Y4 (vpin 0.63) | ✅ YAML | 次 fill cycle で反映 |
| **Y5 (balance filter)** | ❌ コード | **再起動が必要** |
| **Y7 (trades 7日window)** | ❌ コード | **再起動が必要** |
| 番号修正 (130#/129#) | ❌ コメントのみ | 影響なし |

### 2.3 Y3 (SkipGate 再訓練) の検討

129# §8 Y3 は残課題中の最重要施策: SkipGate AUC≈0.5 (ランダム分類器) の改善。

**現状の制約**:

1. **`latest_run_only: true`** — 現行 retrain_scheduler は最新 run (13 records) のみ使用。過去 1048 filled は無視される
2. **Bootstrap 段階** — 13/25 で deploy 未到達。初回 deploy は ~25 サンプルの低品質モデル
3. **コード未反映** — Y5 (balance filter) と Y7 (trades I/O) は稼働中プロセスに反映されていない

**選択肢の分析**:

| 選択肢 | メリット | デメリット | リスク |
|--------|---------|-----------|--------|
| A. 現行 run 続行 (再起動なし) | 中断ゼロ。YAML hot-reload で Y1/Y2/Y4 は反映 | Y5/Y7 未反映。初回 deploy は ~25 sample の低品質。旧 1048 件は未活用 | 低品質 bootstrap deploy がスキップ判定を歪める |
| B. fill_test 再起動 (新コード) | Y5/Y7 反映。7 日 trades I/O で retrain 高速化 | 現 run 13 records 破棄。run_id 切替で再び bootstrap 0 からスタート | ゼロリセットのデータ蓄積遅延 |
| C. `latest_run_only: false` に一時変更 | 旧 1048 + 現行 13 = 大量データで即 deploy 可能 | 旧 run は異なる設定 (UTC21 なし、postonly 旧式)。設定混在 ("因果崩壊" 127# H2) | **学習データの設定混在汚染** |
| D. オフライン一括再訓練スクリプト | 設定フィルタを適用しつつ全データ活用。稼働中プロセスに非干渉 | 実装コスト中。retrain_scheduler と重複 | スクリプト品質管理 |
| **E. 現行 run 25 到達を待ち → bootstrap deploy → 再起動** | bootstrap deploy で初回モデルを早期取得。再起動で Y5/Y7 反映 | bootstrap model は低品質 (25 sample) | 低品質モデルだが「何もないより良い」 |

### 2.4 推奨: 選択肢 E (段階的移行)

**理由**:

1. **Bootstrap deploy (25 samples) を先に取得** — AUC≈0.5 のランダム分類器よりは改善する可能性がある。最悪でも quality_gate で棄却されるため harm は限定的
2. **Deploy 後に再起動** — 新コード (Y5/Y7) を反映。新 run は最初から balance filter + trades 高速 I/O の恩恵を受ける
3. **再起動後のデータ蓄積** — `latest_run_only: true` のまま、新設定ベースのクリーンなデータで本格再訓練

**タイムライン**:

```
現在 (13/25 records)
  │ time_filter 非ブロック時間帯: ~120s/cycle
  │ 12 cycles ≈ 24 分 (最短) ~ 数時間 (time_filter zone 次第)
  ▼
Bootstrap deploy (25 samples)
  │ quality_gate で 棄却 or 採用
  ▼
fill_test 再起動 (新コード: Y5/Y7/番号修正)
  │ 新 run_id で 0 からデータ蓄積
  │ min_spread=1200, vpin=0.63 が最初から適用
  ▼
25 samples → bootstrap deploy (第2世代)
  │ balance_forced_switch 除外、trades 高速 I/O
  ▼
100 samples → 安定期 retrain → deploy (第3世代以降)
```

**Y3 (本格 SkipGate 再訓練) について**: 1048 旧データの活用は **選択肢 D (オフラインスクリプト)** で行うのが設定混在リスクを回避しつつ最善。ただし優先度は中 — まず上記タイムラインで bootstrap→再起動を完了させ、新データ蓄積が 200+ に達した段階でオフライン再訓練の要否を再判断する。

---

## §3 現行 YAML パラメータ (本書時点)

| パラメータ | 値 | 変更 | 根拠 |
|-----------|-----|------|------|
| order_quantity | 0.001 BTC | — | Coincheck 最小 |
| cycle_interval_sec | 120.0 | — | |
| min_spread_jpy | **1200** | 1500→1200 | Y2: fill_rate 改善 |
| side_offset.sell | 0.18 | — | 121# |
| adaptation.enabled | false | — | 因果分離 |
| skip_utc_hours_sell | [4,8,14,15,16,21] | +21 | S1: 128# 分析 |
| unknown_buy_offset_boost | 2.0 | 新規 | S2: unknown regime |
| volatility_guard.vpin_threshold | **0.63** | 0.70→0.63 | Y4: 感度 -10% |
| retrain.target | pnl30 | pnl120→pnl30 | S9: coverage 100% |
| retrain.bootstrap_min_total_samples | **25** | 30→25 | Y1: deploy 加速 |
| retrain.bootstrap_threshold | 100 | — | |

---

## §4 残課題 (Phase Y/Z)

### 4.1 Phase Y 残留

| # | 施策 | 状態 | コスト | 備考 |
|---|------|------|--------|------|
| Y0/Y0b | lock heartbeat + balance flag | ✅ | — | R1-R3 |
| Y1/Y2/Y4 | YAML tuning 3件 | ✅ | — | hot-reload 反映可 |
| Y5 | balance filter in retrain | ✅ (コード) | — | 再起動で反映 |
| Y6 | Appendix B キー名修正 | ✅ | — | 129# Appendix E |
| Y7 | trades I/O 7日 window | ✅ (コード) | — | 再起動で反映 |
| **Y3** | **SkipGate 再訓練** | ⬜ | **高** | AUC≈0.5。§2.4 で段階的移行を推奨 |

### 4.2 Phase Z — ph3 準備 (Gate 判定後)

| # | 施策 | 状態 | ph3 ブロッカー? |
|---|------|------|---------------|
| Z1 | v458 Walk-Forward バグ 6 件修正 | ⬜ | **Yes** |
| Z2 | Oracle テスト (maker 0% 理論上限) | ⬜ | **Yes** |
| Z3 | ph3 Stop 条件明文化 | ⬜ | Yes |
| Z4 | `execute_trade()` 実装 | ⬜ | ph4 ブロッカー |
| Z5 | `skip_gate.py` → `ztb/models/` 移動 | ⬜ | No |

### 4.3 118# との整合

118# §9 の行動計画に対する現在の進捗:

| Phase | 状態 | 詳細 |
|-------|------|------|
| A (即時) | ✅ 全完了 | A1-A4: fill_test 再起動, warm_start, sell SG, VG 効果 |
| B (Gate 判定) | ✅ 全完了 | B1-B4: 自動判定, Holm, t検定, 日別 AS |
| **C (ph3 準備)** | ⬜ **全件未着手** | C1-C4: WF バグ, Oracle, Stop 条件, execute_trade |
| D (Gate FAIL 施策) | 🔄 部分進行 | D1/D3/D5 解決済。D2/D4/D6 は Gate 結果待ち |

---

## §5 テスト結果

| テストスイート | 件数 | 結果 |
|--------------|------|------|
| test_enricher_skip_gate.py | 81 | ✅ all passed |
| test_retrain_hot_reload.py | 20 | ✅ all passed |
| test_fill_quality.py | 176 | ✅ all passed (1 fix: vpin_threshold assertion) |
| **新規テスト** | 3 | ✅ Y5 balance filter (2) + F7 trades I/O fallback (1) |
| **合計** | **260** | ✅ **0 failed** |

---

## §6 コミット履歴

| SHA | 日時 | セッション | 内容 |
|-----|------|-----------|------|
| `b525b3a8a` | 02/21 16:39 | 129.1# | 128# 分析結果 6 施策実装 (S1-S6) |
| `9df2715ea` | 02/21 16:53 | 129.1# | DRY: feature_enricher `_read_jsonl_gz` 統合 (S8) |
| `2780f80b0` | 02/21 17:10 | 129.1# | retrain pnl30 target + config hot-reload (S9-S10) |
| `f8ee8f7fb` | 02/21 17:27 | 129.1# | 129# ドキュメント作成 |
| `7cb39ebb5` | 02/21 17:42 | 129.1# | trades date_filter fallback (S11) |
| `9b4481f47` | 02/21 17:48 | 129.1# | 129# 更新: X1/X2 解決 |
| `0e5dcf71b` | 02/21 18:02 | 129.2# | lock heartbeat + balance flag (R1-R3) |
| `e8af70aad` | 02/21 18:05 | 129.2# | 129# Appendix E (Codex レビュー回答) |
| `8b466d555` | 02/21 18:28 | 129.3# | YAML tuning + balance filter + trades I/O (Y1-Y7) |
| *(本書)* | 02/21 | 129.3# | 130# 作成 + 番号是正 |

---

## §7 外部レビュー向け質問事項

以下の項目について外部 AI コーディングエージェントの見解を求める:

### Q1: retrain 段階的移行 (§2.4 選択肢 E) の妥当性

Bootstrap 25 samples で一旦 deploy → 再起動 → 新データで本格 retrain、というアプローチは妥当か。代替として `latest_run_only: false` (旧データ併用) のリスク/ベネフィットをどう評価するか。

### Q2: balance_forced_switch 除外 (Y5) の学習への影響

全 1609 cycles 中 705 件 (43.8%) が balance 制約の影響を受けていた。これを除外することで学習データが大幅に減少する。除外すべきか、それとも「balance_forced_switch」を特徴量として組み込むべきか。

### Q3: trades I/O 7 日 window (Y7) の妥当性

直近 7 日に限定するフォールバックは、trades の時系列依存性を考慮した窓として適切か。特徴量計算は 60s/300s window なので 7 日は過剰に広い可能性。一方で trades 蓄積の偏り（一部の日のみ密）への頑健性は。

### Q4: Phase C (ph3 準備) の先行着手タイミング

Gate 判定結果を待たず Z1 (WF バグ 6 件) と Z2 (Oracle テスト) に着手すべきか。Gate FAIL でも ph3 準備コードの品質改善は無駄にならない一方、Gate PASS が確定するまで ph3 準備にリソースを割くのはサンクコストリスクがある。

### Q5: コード内コメントの番号付与ポリシー

実装時のコード内コメントにドキュメント番号を付与する現在のプラクティスは有用だが、番号が欠番になるリスクがある。代替案:
- A. コミット SHA を使用（不変だが可読性低い）
- B. ドキュメント番号のみ使用（本書のようにまとめ文書が前提）
- C. `IMPL-YYYYMMDD` 形式（日付ベース）

### Q6: 残高持続可能性 (129# F6) の設計的対策

JPY 12,749 円 + BTC 0.001 での極限運用。入金以外の技術的解決策はあるか。例: side 固定 (buy のみ)、仮想残高による学習データ分離、lot 0.0005 BTC (Coincheck 最低未満だが API では通る可能性)。

---

## Appendix A: 変更ファイル一覧

| ファイル | 変更種別 | セクション |
|---------|---------|-----------|
| `scripts/v460/run_fill_test.py` | S3, S4, R1-R3 | §1.1, §1.2 |
| `scripts/v460/ml/feature_enricher.py` | S6, S8, S11, Y7 | §1.1, §1.3 |
| `scripts/v460/ml/retrain_scheduler.py` | S5, S9, S10, Y5 | §1.1, §1.3 |
| `scripts/v460/lib/fill_config.py` | S2, R2 | §1.1, §1.2 |
| `scripts/v460/lib/maker_price.py` | S2 | §1.1 |
| `scripts/v460/monitor_fill_test.py` | S7 | §1.1 |
| `ztb/metrics/fill_quality.py` | R3 | §1.2 |
| `configs/v460/fill_test.yaml` | S1, Y1, Y2, Y4 | §1.1, §1.3 |
| `tests/unit/v460/test_fill_quality.py` | Y4 assertion | §5 |
| `tests/unit/v460/test_retrain_hot_reload.py` | Y5, Y7 テスト | §5 |

## Appendix B: 番号是正 diff サマリ

コード内 36 箇所のコメント・ログ文字列を修正:

| 旧ラベル | 新ラベル | 件数 | 理由 |
|---------|---------|------|------|
| "131#" | "130#" | 6 | ドキュメント 131 は存在しない → 本書 130# に統合 |
| "132#" | "129#" | 13 | Codex レビュー起源 → 129# Appendix D/E |
| "133#" | "130#" | 17 | YAML/実装変更 → 本書 130# に統合 |
| **合計** | | **36** | 8 ファイル |

## Appendix C: 000# §5 番号規則の再確認

> - **ドキュメント番号** (例: 123#): `docs/v460/` 配下の通し番号。正式な記録単位。
> - **セッション番号** (例: 125.1#): 対話セッションの識別子。**枝番号は小数点**で表記する。
> - ドキュメント番号とセッション番号は独立。1セッションで複数ドキュメントを生成可能、逆も然り。
> - **原則**: 参照はドキュメント番号を使用する。セッション番号は経緯の追跡が必要な場合のみ記載。

**教訓**: コミットメッセージでセッション番号を使用する場合、ドキュメント番号と混同しないよう `S129.1#` (セッション) vs `D130#` (ドキュメント) のような接頭辞を付けるか、ドキュメント番号のみを使用する。
