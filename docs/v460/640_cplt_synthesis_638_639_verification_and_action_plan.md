# 640# 638/639 レビュー検証・総合判定・アクションプラン

## 0. 本文書の位置づけ

638# (PHG) と 639# (Copilot) の2つのレビューを **コードとデータの一次ソースに照合** し、
538# (風水感三爻)・605# (Grand Retrospective)・606# (Unfreeze)・000# (Project Proposal) の
アーキテクチャ的文脈を踏まえて、何が正しく、何が誤りで、何をすべきかを判定する。

**606# の最大の教訓: "AI生成の数値は必ずソースコード + 実ログで検証せよ"** を本文書でも徹底する。

---

## 1. 638# (PHG) 検証結果

### 1.1 ✅ buy/ranging PnL 悪化 — **検証済み**

| 項目 | 638# 主張 | 実データ | 判定 |
|------|-----------|----------|------|
| buy/ranging avg PnL | -1.03bps | **-1.07bps** (n=23, total=-24.71bps) | ✅ 一致 |
| min (catastrophe) | — | **-28.02bps** | — |

buy/ranging はもはや安全バケットではない。-28.02bps の単一 fill が支配的。

### 1.2 ✅ CV widen buy-side 損害 — **正確に検証済み**

| 分類 | n | avg PnL | total PnL |
|------|--:|--------:|----------:|
| CV WIDEN (post_offset > pre_offset) | 14 | **-2.20bps** | -30.82bps |
| CV SAME (ceiling で clamp) | 7 | **+2.60bps** | +18.18bps |
| CV 未適用 | 16 | +0.06bps | +0.90bps |

638# の "buy side CV widen PnL avg=-2.20bps" は **寸分違わず正確**。
ceiling が widen を阻止した 7 fills は +2.60bps と黒字 — **ceiling が利益を守っている**。
逆に widen された 14 fills は -30.82bps の損害。**CV widen は buy 側で有害**。

### 1.3 ✅ skip_rate_limit 分布 — **検証済み**

| side/regime | n | total PnL |
|-------------|--:|----------:|
| sell/ranging | 15 | **-15.38bps** |
| buy/ranging | 3 | -1.29bps |
| sell/trending_down | 3 | +0.42bps |
| sell/trending_up | 1 | +5.39bps |
| **合計** | **22** | **-9.56bps** |

sell/ranging が 22件中15件 (68%) を占め、損失の主因。638# の売り特化 skip budget 提案は方向性として正しい。

### 1.4 ❌ final_clamp_hard_skip: 7件 → **事実誤認 (実際は2件)**

| 項目 | 638# 主張 | 実データ | 判定 |
|------|-----------|----------|------|
| hard_skip 件数 | 7 | **2** | ❌ |
| 内 buy/trending_down | 6 | **2** (100%が buy/trending_down) | 件数は❌、比率は✅ |

実ログ (2026-03-26 07:02–15:26) での HARD SKIP は:
- 15:12:14 — buy/trending_down, pre_clamp_offset=**1.6650** > ceiling(0.35)×2.5=0.875
- 15:21:21 — buy/trending_down, pre_clamp_offset=**1.0607** > ceiling(0.35)×2.5=0.875

**638# の「7件中6件が buy/trending_down」は 606# パターンのデータ幻覚。実際は2件。**
ただし方向性（buy/trending_down が hard_skip で殺されている）自体は正しい。

### 1.5 ⚠️ State汚染 (preflight → side 選択) — **部分的に正確**

638# は `_execute_skip(update_last_side=True)` が side 選択を汚染すると主張。

**実際のアーキテクチャ** (コード検証済み):
- `orchestrator._last_side`: skip 時に更新される。086# deadlock 検出 (`orchestrator_pre_cycle.py:554`) に使用。**SideSelector の交互選択には影響しない**
- `SideSelector._last_side`: `update_after_decision()` でのみ更新。このメソッドは `fill_cycle_executor.py:727` で **実際に約定した場合のみ** 呼ばれる
- `SideSelector._frozen_side`: `freeze_side()` で設定。**これが真の汚染ベクトル**

```
orchestrator_balance.py での preflight 不足時の処理:
  1. freeze_side(next_side, cycles=3)  ← SideSelector に直接影響
  2. _execute_skip(update_last_side=True) ← orchestrator._last_side のみ
```

**結論**: 638# の「State汚染」は方向性として正しいが、メカニズムの理解に誤りがある。
汚染経路は `_last_side` ではなく `freeze_side(cycles=3)` による 3サイクル凍結。
`balance_freeze_cycles: 3 → 1` で影響は大幅に軽減される。

---

## 2. 639# (Copilot) 検証結果

### 2.1 ⚠️ EV ハードフロア — **既存メカニズムを認識していない**

639# は「ev_score < -2.0 で強制 Skip する無敵の IF 文を追加」を提案。

**現実**: 既に3段階のEV防御が存在:
- `ev_emergency_skip_threshold: -8.0` → 強制 Skip (ultimate floor)
- `ev_toxic_skip_threshold: -5.0` → 強制 Skip (593# 追加)
- `ev_warning_threshold: -4.0` → offset 拡大 (保守化)

639# の提案は「新メカニズムの追加」ではなく「既存閾値の調整」で実現可能。
ただし -2.0 まで tighten すると正常な取引も大量に Skip するため **過剰防衛**。
現実的な最適化は `ev_toxic_skip_threshold: -5.0 → -3.5` 程度への段階的調整。

### 2.2 ⚠️ Inventory Skew — **既存実装を知らない**

639# は「Freeze の代わりに在庫偏重によるスプレッド非対称化」を提案。

**現実**: 226# で `inventory_skewing` は既に実装済み:
- `inventory_skewing_enabled: False` (現在無効)
- `inventory_skewing_window: 100`
- `inventory_skewing_max_factor: 0.4`
- `inventory_skewing_neutral_band: 0.1`
- `maker_price.py:751` に `_apply_inventory_skew()` メソッドが存在

可能化のリスク: 605# で「226# inventory_skewing は functional」と確認されているが、
現在は balance_freeze が同じ問題空間を別アプローチで解決中。**両方を同時に有効化すると二重補正** になる。

### 2.3 ✅ buy/trending_down 保護 — **データが強く支持**

| side/regime | n | avg PnL | total PnL |
|-------------|--:|--------:|----------:|
| buy/trending_down | 13 | **+1.48bps** | **+19.20bps** |
| (他の全バケット合計) | 62 | -0.88bps | -54.73bps |

**buy/trending_down は唯一の黒字バケット** であり、2件の HARD SKIP で機会が殺されている。
639# の「ホワイトリストパス」提案は方向性として正しい。

### 2.4 ⚠️ 動的タイムアウト — **概念は良いがインフラ不足**

637# の wait_sec 層別化データが根拠:
- <10秒: 赤字 (adverse selection)
- 10-20秒: 黒字
- >20秒: 大赤字

しかし `same_side_depth_ahead` のリアルタイム追跡は現在実装されておらず、
板情報のストリーミング粒度の問題もある。中長期目標としては有効。

### 2.5 ⚠️ CV cancel-only モード — **急進的すぎる**

CV widen が buy 側で有害なのは検証済み (§1.2)。しかし:
- CV は buy/sell 両方に適用されているが、**sell 側は未分析** (lead_lag_direction が buy=down で全件一致)
- cancel-only に切り替えると「危険検出 → 退場」のみになり、「安全確認 → 参入」シグナルを完全に失う
- 000# の三層アーキテクチャ (Alpha/Execution/Safety) において、CV は Execution 層の情報源。丸ごと kill は過剰

**より穏当な対応**: buy 側の CV widen を無効化 or cap (§4 で具体案)

---

## 3. 風水感三爻 (538#) からの文脈整理

538# の核心的フレームワーク:
1. **Reduction Scorecard**: 各コンポーネントがどれだけ損失を減らしているか計測してから触る
2. **Current-State Matrix**: 現在の各パラメータの実効値を把握し、提案が二重補正にならないか確認
3. **「天井を上げる前に、各段階の寄与を理解せよ」**

### 538# をどう適用するか

| 提案 | 538# 適合性 | 判定 |
|------|------------|------|
| skip_rate_limit 引き上げ | scorecard: -15.38bps sell/ranging 損失を計測済み | ✅ 適合 |
| CV buy widen 無効化 | scorecard: -30.82bps widen 損失 vs +18.18bps ceiling 保護を計測済み | ✅ 適合 |
| buy/trending_down 天井緩和 | **天井を上げる前に理由を理解する必要あり**: なぜ offset が 1.0607/1.6650 まで膨張するか | ⚠️ 要事前分析 |
| EV 閾値調整 | current-state matrix: 既存3段階を把握したうえでの微調整 | ✅ 適合 |
| inventory_skewing 有効化 | current-state matrix: freeze との二重補正リスク | ❌ 事前整理必要 |

---

## 4. アクションプラン (Phase G3→G3.1 に即した段階的改善)

### 605# Tier 分類に準拠した優先度

**605# Tier 0: 安全性を壊さない** — ここから逸脱する変更は不可

---

### P0-A: CV buy-side widen 無効化 (YAML 変更のみ)

**根拠**: 検証済みの -30.82bps 損害。ceiling clamp された fills は +2.60bps → widen 自体が有害。

**対応案**:
- `cross_venue_lead_lag.buy_widen_enabled: false` を追加 (新規 config key)
- または `cross_venue_lead_lag.buy_widen_cap_bps: 0.0` で widen 幅をゼロに制限

**538# 適合**: ✅ scorecard で損益が明確に計測済み

**リスク**: Low — buy 側の widen を止めるだけで、sell 側やその他の CV シグナルは維持

---

### P0-B: max_skip_rate 0.30 → 0.40 (YAML 変更のみ)

**根拠**: 22件の強制実行 → -9.56bps。sell/ranging 15件 → -15.38bps。

**対応案**: `skip_gate.max_skip_rate: 0.40`
- 638# の「まず 0.40 で確認」に合意
- sell 特化 skip budget は将来検討 (config 拡張が必要)

**538# 適合**: ✅ scorecard 計測済み、current-state の 0.30 からの段階的引き上げ

---

### P0-C: balance_freeze_cycles 3 → 1

**根拠**: freeze_side(cycles=3) が SideSelector を 3 サイクル凍結し、ranging_buy_priority を実質無効化。

**対応案**: `balance_freeze_cycles: 1`

**538# 適合**: ✅ 影響範囲が限定的、自然な交互選択を阻害する期間を最小化

---

### P1-A: buy/trending_down hard_skip 緩和 (コード変更)

**根拠**: buy/trending_down は唯一の黒字バケット (+19.20bps)。2件の HARD SKIP がこれを殺している。

**対応案**: `execution_final_clamp_hard_skip_mult` を regime 別に設定可能にする:
- ranging/trending_up: 2.5 (現行維持)
- trending_down の buy: 4.0 (緩和)

**注意 (538# 「天井を上げる前に理解」)**: offset が 1.0607/1.6650 に膨張する根因を先に調査すべき。
これは ev_offset + velocity_offset + regime_boost の積み重ねで発生している可能性が高い。
膨張原因が健全（本当に market extreme）なら緩和は正当。ML の暴走なら別の修正が必要。

---

### P2 以降 (将来検討)

| 優先度 | 項目 | 理由 |
|--------|------|------|
| P2 | sell 特化 skip budget (`max_skip_rate_sell`) | config 拡張 + テスト必要 |
| P2 | ev_toxic_skip_threshold -5.0 → -3.5 | 影響範囲要調査 |
| P3 | 動的タイムアウト (depth tracking) | インフラ整備が前提 |
| P3 | inventory_skewing 有効化 | freeze との統合設計が必要 |
| P3 | CV cancel-only モード | シミュレーション必要 |

---

## 5. 両レビューの総合評価

### 638# (PHG)
- **精度**: データ検証 4/5 正確。final_clamp_hard_skip の件数 (7→2) に 606# パターンの幻覚あり
- **独自価値**: CV widen の -2.20bps 計測は完全に正確で高価値。sell 特化 skip budget の発想は秀逸
- **State 分析**: 方向性は正しいが `_last_side` vs `freeze_side` のメカニズム理解に混乱あり
- **全体判定**: 高品質。データ幻覚を除けば信頼に足る分析

### 639# (Copilot)
- **精度**: 概念的提案が中心で、具体的数値の誤りは少ない（数値自体をほぼ出していない）
- **独自価値**: buy/trending_down 保護の強調が重要。動的タイムアウトの概念は将来有望
- **弱点**: 既存メカニズム（EV 3段階防御、inventory_skewing）の存在を把握していない
- **過剰提案**: CV cancel-only は急進的、EV -2.0 は過剰防衛
- **全体判定**: 方向性は良いが、codebase awareness が不足。「新規追加」として提案しているものの多くが「既存の有効化 or 調整」で済む

### 000# (Project Proposal) との整合

三層アーキテクチャ (Alpha/Execution/Safety) の観点からの整理:
- **Alpha 層** (SAC sidecar): stale 66.7% — 現状 sidecar は判断に使えない。638# の指摘通り
- **Execution 層** (offset/clamp/CV): P0-A (CV widen 無効化)、P1-A (clamp 緩和) がここ
- **Safety 層** (SAD/MCB/skip_gate): P0-B (max_skip_rate)、P0-C (freeze_cycles) がここ

SAC が stale な現状では、**Execution 層と Safety 層の最適化が利益改善の主戦場**。
これは 000# の「SAC = sidecar navigator, not driver」設計思想に沿う。

---

## 6. 実行順序

```
Step 1: P0-A (CV buy widen 無効化) + P0-B (max_skip_rate 0.40) + P0-C (freeze 1)
        → YAML 変更のみ、リスク低、即効性あり
        → 期待改善: CV widen -30.82bps の回避 + skip_rate -9.56bps の軽減

Step 2: P1-A 事前調査 — offset 膨張原因の解析
        → buy/trending_down で pre_clamp_offset > 0.875 になる pipeline 段階を特定

Step 3: P1-A 実装 — regime 別 hard_skip_mult (調査結果に基づく)

Step 4: 効果計測 → P2 判断
```
