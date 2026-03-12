# 112# 111# レビュー結果 + 追加提案 + 見落とし補完

| key | value |
|-----|-------|
| type | rev (レビュー) |
| target | `docs/v460/111_phg_rpt_legacy_asset_research.md` |
| date | 2026-02-19 |
| reviewer | GitHub Copilot (GPT-5.3-Codex) |
| purpose | 111# の妥当性検証、実コードとの差分指摘、v460 ph2/ph3向け追加提案の提示 |

---

## §0 総評（先に要点）

111# は、**v456–v459 の教訓を v460 fill_test に接続する観点として非常に有用**であり、方針の骨子（手数料制約・報酬純度・Gate順序・過学習警戒）は妥当。

一方で、実コード照合により以下を確認:

1. **高評価**: 優先度設計（安全基盤→監視→データ品質→状態管理）は実運用上合理的
2. **要修正**: 一部モジュールパスが実体と不一致（`ztb/data/...` の記載など）
3. **要修正**: `ztb/adaptation/` を「完全Dead」とする断定は不正確（import/テスト参照が複数存在）
4. **要追加**: `run_fill_test.py` は 3k 行超の巨大化が進行しており、統合前に責務分離計画を同時に置くべき
5. **要追加**: 168h 運用の「何をもって PASS とするか」の**定量SLO**が文書上まだ弱い

---

## §1 妥当だった点（111# の強み）

### §1.1 根本制約の認識は正しい

- 「手数料構造 × 取引頻度」が最上位制約という結論は、v459 の Oracle/counterfactual 系教訓と整合
- maker 0% 前提で fill quality を実測する v460 ph2 の方向性は妥当

### §1.2 v460 での再利用候補選定は概ね妥当

- `ztb/risk/pnl_monte_carlo.py`
- `ztb/risk/advanced_auto_stop.py`
- `ztb/risk/checks.py`, `ztb/risk/rules.py`, `ztb/risk/profiles.py`
- `ztb/ops/monitoring/watch_1m.py`
- `ztb/ops/alerts/notifications.py`, `ztb/ops/alerts/gates_to_alerts.py`
- `ztb/cache/memory_cache.py`

これらは実在し、fill_test 強化の候補として妥当。

### §1.3 「Fix First」思想は現在フェーズに一致

- 086/110 系のデッドロック修正履歴や、状態整合バグの経緯から見ても、拡張より先に信頼性確保を優先する方針は適切

---

## §2 要修正ポイント（111# と実コードの差分）

### §2.1 パス誤記・存在不一致

111# §9 で記載の以下は、現行ツリー上の実体と不一致:

- `ztb/data/data_validation.py` → 実体は `ztb/utils/data_validation.py`
- `ztb/data/streaming_pipeline.py`（該当なし）
- `ztb/data/stream_buffer.py`（該当なし）
- `ztb/data/anomaly_detection.py`（該当なし）
- `ztb/data/integrity_checker.py`（該当なし）

**提案**: 111# の §9.1/§9.2 は「存在確認済み/未確認」を明示して再版し、誤記を除去する。

### §2.2 「完全Dead」判定の過剰断定

111# §4.1 の `ztb/adaptation/` 系は、現行コードで import 参照が複数存在。

- 例: `ztb/training/unified_trainer/trainer.py`
- 例: `ztb/training/algorithms/sac/sac_algorithm.py`
- 例: `tests/unit/algorithms/test_ab_test_framework.py`

よって、現状は **「fill_test 経路では未使用」** が正確であり、**「参照ゼロの完全Dead」** とは言い切れない。

### §2.3 run_fill_test 統合状況の表現補足が必要

111# は「未統合モジュール候補」の提示は適切だが、現実には `scripts/v460/run_fill_test.py` が巨大化しており、

- 直接統合を重ねると God Object 再発リスクが高い
- `lib/` 分割済みでもオーケストレーション責務が集中

**参照**: [106_ph2_fix_refactoring_r1_r10.md](106_ph2_fix_refactoring_r1_r10.md) R1 で `run_single_cycle` (~750行) の分割が既に特定されており、「大規模リファクタで再起動前にやるにはリスキー」として後日判定済。統合前に R1 を先行実施すべき。

**提案**: 「導入対象」だけでなく「導入手順（分離前提）」をセットで記述すべき。

---

## §3 追加提案（111# への上積み）

### §3.1 導入優先順位の再定義（実装リスク込み）

111# §10 の優先度は方向性正しいが、実装難易度を反映して再配置を推奨:

**Tier-1（今すぐ、低侵襲）**
1. `CircuitBreaker`（API呼出ラッパ）
2. `HealthMonitor` または `watch_1m` のどちらか一方を先行採用
3. `GatesToAlerts`（Gate結果通知）

**Tier-2（短期、状態一貫性に効く）**
4. `StatePersistence`（再起動復元）
5. `Reconciliation`（定期照合）

**Tier-3（中期、設計負荷高）**
6. `RiskRuleEngine + Profiles + AdvancedAutoStop`（ルール競合整理が必要）
7. `PnL Monte Carlo` の定時運用化（データ欠損時のロバスト性設計込み）

### §3.2 168h fill_test の SLO/Gate 指標を固定化

111# は候補を網羅しているが、判定閾値の統一表が不足。

最低限、以下を文書化して Gate と接続すべき:

- 可用性: `uptime_ratio`, `downtime_minutes/day`
- 注文健全性: `api_error_rate`, `cancel_timeout_rate`, `orphan_order_count`
- 執行品質: `fill_rate`, `median_queue_wait_sec`, `adverse_selection_bps`
- 経済性: `avg_net_bps`, `gross_fee_ratio`, `daily_loss_cap_breach_count`
- 再現性: `restart_recovery_success_rate`

### §3.3 「運用失敗モード」ベースの試験追加

現行ドキュメントは資産棚卸し中心で、障害注入観点が薄い。

提案する最小試験:

1. API 429/5xx burst（CircuitBreaker 開閉確認）
2. 再起動復元（StatePersistence + position/offset 整合）
3. 長時間メモリ圧迫（watch_1m/メモリ警告連動）
4. 板薄化・急変時（volatility guard + skip gate の挙動）

### §3.4 ph3 着手前の「Stop条件」を先に明文化

v456–v459 の再発防止として、開始条件だけでなく中止条件を固定化すべき。

- 例: 4-seed worst ROI が N 回連続で閾値未満なら探索停止
- 例: Oracle/counterfactual で fee 優位が確認できないなら ph3 拡張停止
- 例: Walk-Forward P0/P1 未解消なら学習施策を凍結

---

## §4 見落とし（111# に追記推奨）

1. **`context.txt` 参照指示への紐付け不足**: 現ワークスペースでは `context.txt` を確認できず、参照元明示が必要
2. **導入コストの明記不足**: 「~xx行」見積りはあるが、テスト改修/運用Runbook更新コストを未計上
3. **競合リスク記述不足**: `DrawdownController`, `AdvancedAutoStop`, `RiskRuleEngine` の責務重複に関する統合方針が未定義
4. **命名・配置の整合性**: `ztb/data` と `ztb/utils` の責務境界が文書側で混線

---

## §5 改訂推奨（111# の最小修正案）

111# を高品質化する最小修正:

1. §9 のパス不一致を修正（`ztb/utils/data_validation.py` へ）
2. §4.1 の dead 判定を「fill_test経路では未活用」に言い換え
3. §10 に SLO/Gate 閾値表を1枚追加
4. §10 に「統合順序（Tier-1/2/3）」を追記
5. `run_fill_test.py` の責務分割方針（小さく導入→抽象化）を追記

---

## §6 最終コメント

111# は、過去資産の再利用を「教訓→実装候補→優先度」まで落とし込めており、**戦略文書としては強い**。

今回の差分修正（パス整合・dead判定精緻化・SLO明文化）を加えれば、
**ph2運用の実行計画としてそのまま使える実務レベル**になる。

次アクションとしては、Tier-1 の 3 点（CircuitBreaker / 監視 / Gate通知）を先に入れ、168h の再実測で「安定性ゲート」を先に通すのが、短期収益化の最短経路。