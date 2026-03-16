# 177# 176レビュー: 大値動き取り逃し是正の追加方策 + 根本解決案

> **種別**: rev  
> **フェーズ**: ph2  
> **日付**: 2026-02-27  
> **レビュー対象**: 170#〜176#（特に 176# A/B 実装）  
> **目的**: 「損しないが大値動きを取り逃す」状態の是正。追加施策と根本解決、vXXX 資産の再利用計画を整理する。

---

## 0. エグゼクティブサマリ

結論は以下。

1. **176# の方向性は妥当**。  
   施策A（`trending_up` のみ sell skip 対象化）と施策B（方向×サイド別 offset）は、170#〜175# で顕在化した「sell 過剰ブロック」を是正する正しい一手。
2. **ただし、根本問題は未解消**。  
   現在の主問題は「トレンド局面での意思決定ホライズン不一致（30s 判定と 120s 収益機会のズレ）」と「120s 固定ループの遅さ」。
3. **次の勝ち筋は “Trend Mode 化”**。  
   レンジ時は防御、トレンド時は実行密度を上げる二相制御に切り替えるべき。  
   176# の C/D（動的 cycle interval / post_fill_wait）は、このための中核であり優先度は高い。
4. **vXXX の再利用余地は大きい**。  
   `v459` の Oracle/K2/リーク検査、`v458` の walk-forward 基盤、`ztb/risk` 群（CircuitBreaker/Drawdown/RuleEngine）は、今の課題に直接効く。

---

## 1. 事実確認（170#〜176# と実データの整合）

`results/v460/fill_test/fill_records_20260213..20260227.jsonl` の再集計（filled + `post_fill_30s_pnl`）では、以下が確認できる。

| regime × side | total | filled | fill_rate | avg pnl30 (bps) | 含意 |
|---|---:|---:|---:|---:|---|
| trending_up × buy | 69 | 36 | 52.2% | +0.2042 | 順張り buy は有効 |
| trending_up × sell | 253 | 29 | 11.5% | -1.6185 | sell は低約定かつ負寄与 |
| trending × sell | 360 | 118 | 32.8% | -0.6596 | undirected trending sell も弱い |
| trending_down × sell | 79 | 43 | 54.4% | -0.2503 | 改善余地あり |
| ranging × buy | 1057 | 599 | 56.7% | -0.4567 | 依然として主要損失源 |

2/23 の cancel_reason は `balance_forced_skip=246`, `trending_sell_skip=220` が支配的で、171#/176# の「sell 過剰抑制カスケード」仮説と整合。

補足: 2/25 の `trending_up × sell` は **30s では負**だが、120s の一部標本では正転。  
この差は「早すぎる評価窓で勝てるトレードを捨てる」構造を示す。

---

## 2. 176# 実装レビュー（A/B）

### 2.1 施策A: `trending_up` 以外の sell 通過

**評価**: ✅ 妥当（高優先で正しい修正）

- `TRENDING`（方向不明）まで一律ブロックしていた挙動は明確な実装不整合。
- 2/23 の `trending_sell_skip` 多発と `balance_forced_skip` カスケードを説明できる。
- ガード除去ではなく「対象の厳密化」なので副作用管理もしやすい。

### 2.2 施策B: 方向×サイド別 offset boost

**評価**: ✅ 有効な方向だが、静的倍率の固定運用はリスクあり

- `trending_up buy` を取りに行く意図は正しい。
- 一方で `trending_up sell` は fill_quality が悪く、倍率固定（例: 1.8/0.7）だけでは regime 変化に追従しきれない。
- 係数そのものより、**発動条件（regime confidence / volatility / AS prob）** の設計が重要。

---

## 3. 追加方策（短期）

### 3.1 P0: 176# C/D を「安全柵付き」で先行実装

### C. Dynamic Cycle Interval（トレンド時短縮）

- `trending_up/down`: `cycle_interval_sec = 60`
- `ranging`: `120`（現状維持）
- ガード: API エラー率しきい値超過時は自動で `120` にフォールバック

**狙い**: 大値動き時の取り逃し削減（実行機会密度を上げる）

### D. Regime-linked Post-Fill Wait

- `trending_up`: buy `15`, sell `45`
- `trending_down`: buy `45`, sell `15`
- `ranging`: 現状維持

**狙い**: 順張り側の再参入速度を上げる。逆張り側は待機を長めにして逆選択を抑える。

### C/D 共通の停止条件（必須）

- `api_error_rate > 3%` が 2 時間継続 → 即 rollback
- `attempted_fill_rate < 35%` が 6 時間継続 → 即 rollback
- `avg pnl30 < -0.8 bps` が 6 時間継続 → 即 rollback

### 3.2 P0: Trend Mode 発動条件の厳格化

方向別 offset の適用条件を `regime == trending_*` のみでなく、以下の AND 条件へ。

- `regime_confidence >= 0.55`
- `|price_velocity_60s| >= v_min`
- `spread >= s_min`

これで「なんとなく trending」時の誤作動を減らす。

### 3.3 P1: 評価窓の二重化（30s 単独判定の卒業）

大値動き取り込み評価を以下 2 指標で同時判定する。

- `pnl30`（逆選択検知）
- `pnl120`（トレンド追随の最終回収）

採否判定は `EV_weighted = 0.4*pnl30 + 0.6*pnl120` を主指標に変更。  
30s だけで切ると、176# が狙う「伸びる局面」を誤って殺しやすい。

---

## 4. 根本解決案（中期）

### 4.1 根本原因モデル

現象を 1 行で定義すると以下。

`遅いループ × 短い評価窓 × sell抑制ループ` により、トレンド局面の正の期待値を取り切れない。

### 4.2 アーキテクチャ提案: Trend Mode State Machine

`ranging_mode` と `trend_mode` を明示的に分離し、制御量を切り替える。

| 制御量 | ranging_mode | trend_mode |
|---|---|---|
| cycle interval | 120s | 45〜60s |
| post_fill_wait | 現状値 | side/regime 別短縮 |
| offset boost | 防御寄り | 方向別非対称 |
| skip_gate threshold | 厳格 | やや緩和（順方向のみ） |
| lot sizing | 低〜中 | 上限付きで増加 |

遷移条件にはヒステリシス（連続判定）を入れ、モードのフリップフロップを防ぐ。

### 4.3 ロギング契約の修正（必須）

171# で指摘済みの通り、skip の反実仮想精度が欠損している。  
`fill_records` に以下を常時記録し、ガード精度をオンライン監査可能にする。

- `counterfactual_pnl_30s`, `counterfactual_pnl_120s`
- `regime_confidence`
- `guard_decision_trace`（どのガードが通した/止めたか）

---

## 5. vXXX 再利用候補（片っ端から抽出）

以下は「今の課題（大値動き取り逃し + sell 偏重問題）」に効く順で列挙。

| 優先 | 資産 | パス | 用途 | 統合ポイント |
|---|---|---|---|---|
| A | Oracle baseline / oracle test | `scripts/v460/analysis/oracle_baseline.py`, `scripts/v460/analysis/oracle_test.py` | 理論上限との差分管理 | C/D 前後で必ず再測定 |
| A | hindsight_filter + EV | `scripts/v460/analysis/hindsight_filter.py` | guard有害性の定量化 | run ごとの自動レポート化 |
| A | CircuitBreaker | `ztb/utils/circuit_breaker.py`, `ztb/risk/circuit_breakers.py` | 高頻度化時の API 障害耐性 | cycle 短縮時の必須安全弁 |
| A | DrawdownController | `ztb/risk/drawdown_controller.py` | 取り逃し改善中の暴走抑制 | DailyDrawdownGuard と役割分離して併用 |
| A | RiskRuleEngine | `ztb/risk/checks.py`, `ztb/risk/rules.py`, `ztb/risk/profiles.py` | ガード条件の宣言的管理 | if文乱立の抑制 |
| B | AdvancedAutoStop | `ztb/risk/advanced_auto_stop.py` | 変動急増時の段階停止 | trend_mode の fail-safe |
| B | DynamicPositionSizer | `ztb/risk/dynamic_position_sizer.py` | トレンド時ロット最適化 | trend_mode ロット制御 |
| B | Reconciliation | `ztb/trading/live/core/reconciliation.py` | 在庫/約定の不整合検知 | balance_forced_skip の再発防止 |
| B | HealthMonitor / watch_1m | `ztb/trading/live/core/health_monitor.py`, `ztb/ops/monitoring/watch_1m.py` | 長時間運用の健全性監視 | 168h 監視自動化 |
| B | GatesToAlerts | `ztb/ops/alerts/gates_to_alerts.py` | Gate逸脱の通知自動化 | rollback 早期化 |
| C | K2 Non-RL upper bound | `scripts/v459/run_k2_nonrl_upper_bound.py` | 特徴量情報量の上限確認 | trend専用特徴量追加時に実施 |
| C | Data leakage checker | `scripts/v459/check_data_leakage.py` | lookahead 汚染防止 | C/D 実装後の検証 |
| C | E0 diagnostic | `scripts/v459/run_phase_e0_diagnostic.py` | multi-horizon IC 診断 | 30s/120s 乖離の定量化 |
| C | E1 counterfactual | `scripts/v459/run_phase_e1_counterfactual.py` | 完全予測時の費用構造確認 | 「改善の理論余地」再確認 |
| C | Walk-forward evaluator | `ztb/evaluation/walk_forward/evaluator.py` | 時系列頑健性検証 | C/D 後の一般化検証 |

---

## 6. 実行順（提案）

| Step | 変更 | 期間 | 判定指標 |
|---|---|---|---|
| 1 | 176# A/B のみで 24h 観測（凍結） | 24h | fill_rate, pnl30, pnl120, sell比率 |
| 2 | C のみ投入（Dなし） | 24h | trending時 fill 数増加、API健全性 |
| 3 | D のみ投入（Cなし） | 24h | pnl120 改善、在庫偏り悪化なし |
| 4 | C+D 併用 | 48h | `EV_weighted` 正転、DD制御内 |
| 5 | Side別 SkipGate 再訓練 | 以降 | sell 側誤判定率低下 |

重要: **1 run = 1収益系変更** を維持する。  
（170#/168# で繰り返し出た因果分離崩壊の再発防止）

---

## 7. 最終判断

176# は「守りすぎて機会を失う」状態を崩すための正しい転換点。  
次の勝負は、C/D を実装するかどうかではなく、**実装後の評価軸を 30s 単独から EV（30/120）へ更新できるか** にある。  
ここを変えない限り、「損しないが伸びない」状態から抜けにくい。

---

## 8. 追記: 177# に対するセカンドオピニオンと「利益至上主義」に基づく抜本的批判 (Gemini 3.1 Pro)

### 8.1 「悠長な検証」の全否定 — 利益こそが大正義
ユーザーの指摘通り、現在「儲かっていない」状態において、Codexが提案する「1 run = 1変更で24時間ずつ観測（§6）」などという悠長なプロセスは**即刻破棄すべき**である。
利益が出ていないシステムで「綺麗な因果分離」を求めて5日間も時間を浪費するのは、研究者の自己満足に過ぎない。今は「余裕（バッファ）」を生み出すことが最優先であり、**「勝てる可能性が高い施策（C/D）は同時投入し、ダメなら即ロールバック」**というアジャイルな実弾検証に切り替えるべきである。

### 8.2 Codexが見落としている「Makerの限界」と「トレンドの真理」
Codexは「トレンド時にサイクルを短縮し、待機時間を減らす（C/D）」ことで大値動きを取れると主張しているが、これは**Maker戦略の根本的な限界を見落としている**。
強いトレンドが発生している時、順張り（例: 上昇トレンドでのBuy）の指値は置いてけぼりにされ（約定しない）、逆張り（Sell）の指値は轢かれる（Adverse Selection）。サイクルをいくら短くしても、指値（Maker）である以上、この非対称性は覆らない。
本当に大値動き（トレンド）を取りたいのであれば、以下のいずれかの「非対称な執行」が不可欠である。
1. **順張り方向のTaker（IOC）許可**: トレンド確信度（`regime_confidence`）が極めて高い局面に限り、スプレッドを叩いてでも（Taker）ポジションを取りに行く。
2. **Chase（追従）ロジックの導入**: 置いてけぼりにされた順張り指値を、Timeoutを待たずに即座にキャンセルし、現在価格に追従して置き直す（Tick-by-TickのChase）。

### 8.3 「在庫の偏り」はトレンド時の「正解」である
Codexは `balance_forced_skip` を問題視しつつも、在庫偏りを防ごうとする既存の枠組みに囚われている。
トレンド相場において、**「トレンド方向への在庫の偏り」はリスクではなく「利益の源泉」である**。上昇トレンド（`trending_up`）においてBTCを過剰に保有することは、まさに大値動きを享受するための正しい状態である。
したがって、トレンド時には `balance_forced_skip` の閾値を**トレンド方向にのみ大幅に緩和（または無効化）**し、意図的にポジションを傾ける（Inventory Skewing）ロジックを導入すべきである。

### 8.4 評価窓の二重化（EV_weighted）への賛同と補足
Codexの「30s単独判定からの卒業（`EV_weighted = 0.4*pnl30 + 0.6*pnl120`）」には**全面的に賛同**する。
大値動きを狙う場合、30秒という短すぎる評価窓は「ノイズ」や「一時的なスプレッド拡大」を損失と誤認し、勝てるはずのトレードを殺してしまう。利益を伸ばす（Let profits run）ためには、120秒、あるいはそれ以上のホライズンでの評価を主軸に据えるべきである。

### 8.5 結論と「利益を生むため」の即時アクション
「損をしない」フェーズは終わった。今は「リスクを取って利益をもぎ取る」フェーズである。
1. **C/Dの同時投入とChaseロジックの追加**: 悠長なStep検証を捨て、C/Dを同時投入せよ。さらに、順張り方向には指値のChase（追従）を許可せよ。
2. **トレンド方向への在庫許容**: `trending_up` 時のBuy上限、`trending_down` 時のSell上限を動的に引き上げ、トレンドに乗ることを許可せよ。
3. **絶対的敗北ルートの完全遮断**: `trending_up` 時のSellなど、構造的に不利なエントリーは「offset調整」などという甘い対応ではなく、**「完全ハードスキップ」**して無駄な出血をゼロにせよ。
