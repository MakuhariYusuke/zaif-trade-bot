# 342# Design & Market Theory Deep Investigation

**Date**: 2026-03-08  
**Scope**: 340# 符号修正完了後の設計・市場理論上の問題点の深掘り調査  
**Investigator**: GitHub Copilot (Claude Opus 4.6)

---

## Summary Table

| ID | Finding | Severity | Fix Before Restart? |
|----|---------|----------|---------------------|
| A | Forced Switch PnL 完全除外 | **HIGH** | No (P1 中期) |
| B | inv_bypass ステップ関数不連続 | **MEDIUM** | No (P2) |
| C | skip_gate / dynamic_kill 独立抑制 | **HIGH** | No (P1 中期) |
| D | Count-Based Rolling Window | **MEDIUM** | No (P2) |
| E | Sell Post-Fill Wait 非対称性 | **MEDIUM** | No (informational) |
| F | Regime Detection Feedback Loop | **OK** | N/A (既に対策済み) |
| G | Daily Drawdown / Dynamic Kill 相互作用 | **MEDIUM** | No (P2) |

---

## A. Forced Switch PnL Treatment — **HIGH**

### コード参照
- [orchestrator_guards.py](scripts/v460/lib/orchestrator_guards.py#L108-L117) `_track_side_pnl()`
- [orchestrator_post_cycle.py](scripts/v460/lib/orchestrator_post_cycle.py#L107-L117) forced_buy KPI 分離

### 現状の挙動
```python
# orchestrator_guards.py L112-113
if getattr(record, "balance_forced_switch", False):
    return  # ← 完全除外: rolling PnL に一切反映しない
```

`balance_forced_switch` の fill は `_sell_kill_mgr.track()` / `_buy_kill_mgr.track()` のどちらにも PnL を投入しない。

### 問題点

1. **データ欠損 → rolling window の「年齢」が増加**: window=50 で forced fill が 10 個除外されると、有効 window の最古データは 60 fill 前に遡る。120s サイクルで fill rate 50% なら残り 40 fill ÷ (1 fill/240s) ≈ 2.7 時間分。forced fill を含めれば 50/(1/240s) ≈ 3.3 時間。差分の 0.6 時間分だけ「古い情報」で判断する。

2. **Forced fill は実コストを伴う**: balance_forced_switch は即時約定（ask/bid への即座のクロス）であり、スプレッド全幅のコストを払う。このコストが rolling PnL に反映されないと、kill 閾値が「実態より良く見える」→ kill が遅れる可能性がある。一方で 337# の指摘通り、forced fill は MM 能力の指標ではないため、**full weight で含めるのも不適切**。

3. **Sell 側に forced KPI 分離がない**: `orchestrator_post_cycle.py` L107-117 では `forced_buy_kpi_tracking_enabled` で buy のみ forced/normal 分離トラッキングを実施。sell 側の `balance_forced_switch` fill は KPI 分離されておらず、forced sell の品質が不可視。

### 市場理論的根拠
Ho & Stoll (1981) の在庫リスクモデルでは、強制取引は在庫リバランスコストそのもの。完全除外はこのコストを無視し、kill 判定の精度を低下させる。一方、MM のスキル（=スプレッド capture 能力）の指標に forced fill を含めると Glosten-Milgrom の逆選択コスト推定が汚染される。**最適解は downweight (例: 0.5)**。

### 推奨アクション
1. `_track_side_pnl()` で forced fill を weight=0.5 で track に投入（`track()` メソッドに weight パラメータを追加するか、PnL を 50% にスケールして投入）
2. Sell 側にも `forced_sell_kpi_tracking` を追加
3. 中期で `DynamicKillManager.track(pnl_bps, weight=1.0)` に拡張し、weight 付き移動平均を実装

**再起動前に修正が必要か**: No（現状の完全除外は conservative 方向のバイアスであり、kill が遅れる方向。直ちに損失を拡大するリスクは低い）

---

## B. inv_bypass Jump Discontinuity — **MEDIUM**

### コード参照
- [cycle_gate_aggregator.py](scripts/v460/lib/cycle_gate_aggregator.py#L575-L582) `_check_trending_sell()` 内 inv_bypass
- [cycle_gate_aggregator.py](scripts/v460/lib/cycle_gate_aggregator.py#L626-L636) `_check_sell_dynamic_kill()` 内 inv_bypass
- [fill_test.yaml](configs/v460/fill_test.yaml#L596) `sell_guard_inv_bypass_threshold: 0.3`

### 現状の挙動
```python
# cycle_gate_aggregator.py L575-582 (_check_trending_sell)
_inv_bypass_th = self._config.sell_guard_inv_bypass_threshold
if _inv_bypass_th > 0 and inv_net_imbalance >= _inv_bypass_th:
    return GateCheckResult(blocked=False, ...)  # 完全バイパス

# cycle_gate_aggregator.py L626-636 (_check_sell_dynamic_kill)
_inv_bypass = (
    self._config.sell_guard_inv_bypass_threshold > 0
    and inv_net_imbalance >= self._config.sell_guard_inv_bypass_threshold
)
```

### 問題点

**imbalance が 0.29 → 0.30 に遷移する瞬間に、sell の挙動が不連続的に変化する**：

| imbalance | trending_sell gate | sell_dynamic_kill gate |
|-----------|--------------------|-----------------------|
| 0.28 | blocked (hard skip) | blocked (if killed) |
| 0.29 | blocked | blocked |
| **0.30** | **bypassed** (全面許可) | **bypassed** (kill 無視) |
| 0.31 | bypassed | bypassed |

一方、`inv_relaxation`（sell_dynamic_kill_inv_relaxation）は gradual：
```
imbalance=0.3 → offset = min(0.3 * 0.4, 0.3) = 0.12 bps → threshold = -0.3 - 0.12 = -0.42
```

つまり **Gate 3 (trending_sell) と Gate 5 (sell_dynamic_kill) ではステップ関数**で、**kill 閾値の inv_relaxation は連続関数**。2 つの異なる応答関数が同じ imbalance 入力に対して矛盾した制御を行っている。

### 市場理論
Garman (1976) / Ho & Stoll (1981) の在庫管理理論では、在庫偏重に対する応答は連続的であるべき。ステップ関数は以下のリスクを生む：
- imbalance が閾値付近で oscillate すると、sell が on/off を繰り返す（chattering）
- bypass 直後に大量の sell fill → inventory が急減 → bypass 解除 → 再び積み上がり → 振動

### 推奨アクション
1. **Gate 3 (trending_sell)**: `trending_sell_as_offset_enabled: true` が既に有効（現状 YAML 値）なので、実質的にハードブロックではなく offset boost (×1.5) で応答。**この Gate では inv_bypass のステップ関数問題は軽微**。
2. **Gate 5 (sell_dynamic_kill)**: inv_bypass が `is_sell_killed` を完全無視する。代替案：inv_bypass を廃止し、`inv_relaxation` の `max_bps` を引き上げて gradual に閾値を緩和する方向を推奨。例えば `max_bps: 0.5` にすれば、imbalance=0.3 で effective threshold = -0.3 - 0.2 = -0.5 となり、ranging と同等の寛容さを連続的に実現できる。
3. 中期: `sell_guard_inv_bypass_threshold` を 0 (無効化) にして、inv_relaxation のみに統一

**再起動前に修正が必要か**: No（chattering リスクはあるが、trending_sell は既に soft mode。kill bypass は在庫過剰解消に寄与しており、即時撤廃はリバランス障害のリスク）

---

## C. skip_gate / dynamic_kill 独立抑制 — **HIGH**

### コード参照
- [cycle_gate_aggregator.py](scripts/v460/lib/cycle_gate_aggregator.py#L283-L303) Gate 5: sell_dynamic_kill
- [skip_gate_evaluator.py](scripts/v460/lib/skip_gate_evaluator.py#L936-L1050) `evaluate()` — kill 状態への参照なし
- [fill_cycle_executor.py](scripts/v460/lib/fill_cycle_executor.py#L970-L990) executor 内での regime_detector.update()

### 現状の挙動

```
cycle_gate_aggregator.evaluate()
  ├── Gate 4: buy_dynamic_kill  → blocked?
  ├── Gate 5: sell_dynamic_kill → blocked?
  └── (gate 通過後)
        fill_cycle_executor.run_single_cycle()
          └── skip_gate_evaluator.evaluate()  → ML モデルによる独立判定
```

**2 つのフィルターが直列で独立動作**：
- `dynamic_kill`: rolling 50 fill の平均 PnL < threshold → kill
- `skip_gate`: ML モデルが予測 PnL < threshold → skip

### 問題点

1. **二重抑制による sell 機会の過剰喪失**: 338# のデータが示す通り、`eb24cf4a` 期間で `skip_gate=51`, `sell_dynamic_kill=42`。両方が独立に sell を抑制することで、sell pass rate は multiplicative に低下する。これは 337# §3.4 "Sell double-filter" の指摘と一致。

2. **skip_gate は kill 状態を知らない**: `skip_gate_evaluator.evaluate()` は `is_sell_killed` / `is_buy_killed` パラメータを受け取らない。kill 解除直後に skip_gate が抑制し続けると、新しい PnL データが rolling window に入らず、kill が再発動しやすい。

3. **カスケード**: kill → fill なし → rolling window freeze → skip_gate の特徴量（取引頻度、OB imbalance 変化等）が stale 化 → skip_gate の予測精度低下 → kill 解除後も過剰 skip → fill 減少 → 再 kill。

### 市場理論
Amihud & Mendelson (1980) の流動性提供者モデルでは、参加率の低下は情報の陳腐化（information decay）を招く。2 つの独立した抑制メカニズムの直列展開は、個々の false positive rate が掛け算される：
- dynamic_kill が 30% false positive で sell を正当に kill しない
- skip_gate が 20% false positive で sell を過剰 skip
- 実効 false positive = 1 - (1-0.3)(1-0.2) = 44%

### 推奨アクション
1. **短期**: skip_gate に kill 状態をコンテキストとして渡す（特徴量として `is_side_killed` を追加するか、kill 解除直後 N サイクルは skip_gate threshold を緩和）
2. **中期**: skip_gate と dynamic_kill を「参加度」という単一のメトリクスに統合（Toxicity Budget の拡張）。participation_rate を skip_gate の threshold_offset に反映。
3. **計測**: kill 解除後の skip_gate skip rate を独立にトラッキングし、過剰抑制を定量化

**再起動前に修正が必要か**: No（実装変更が大きい。ただし最も収益に影響する可能性が高いため、P1 優先度で対応すべき）

---

## D. Count-Based Rolling Window — **MEDIUM**

### コード参照
- [sell_dynamic_kill.py](ztb/risk/sell_dynamic_kill.py#L247-L249) `track()`
- [sell_dynamic_kill.py](ztb/risk/sell_dynamic_kill.py#L527-L533) `check_kill()` の rolling mean 計算
- [fill_test.yaml](configs/v460/fill_test.yaml#L599-L600) `window: 50`

### 現状の挙動
```python
# sell_dynamic_kill.py L527-533
recent = self._pnl_history[-window:]
rolling_mean = sum(recent) / len(recent)
```
均等加重の count-based moving average。window=50 は「直近 50 fill」を意味する。

### 問題点

1. **時間的カバレッジの不安定性**:
   - trending_up (sell fill 多): 50 fill ≈ 50 × 240s ≈ 3.3 時間
   - ranging (fill rate 低下): 50 fill ≈ 50 × 480s ≈ 6.7 時間
   - kill 中: fill なし → window freeze（最後の 50 fill 分がそのまま残存）
   
2. **kill 期間中の window freeze**: kill 発動 → fill なし → `track()` 未呼出 → rolling mean 不変 → kill 継続。273# の `max_kill_duration_sec: 1800` で 30 分上限を設けて対策済みだが、これは根本解決ではなく safety valve。

3. **古い fill に過剰な重み付け**: ranging から trending に遷移した場合、window の前半は ranging 期の（高 PnL の）fill で、後半は trending 期の（低 PnL の）fill。均等加重だとレジーム遷移への反応が遅れる。

### 市場理論
EWMA (Exponentially Weighted Moving Average) は RiskMetrics (J.P.Morgan, 1996) で提唱された手法で、「新しい情報ほど重要」という直観を数学的に反映する。金融リスク管理の標準手法。

```
EWMA_t = α × pnl_t + (1-α) × EWMA_{t-1}
```

α = 0.05 (decay factor) で effective window ≈ 20 に相当するが、時間的な重み付けにより regime 遷移への反応が速い。

### 推奨アクション
1. **P2**: EWMA への移行を検討。`DynamicKillManager` に EWMA モードを追加（config toggle で count-based と切替可能）
2. **即時**: 現状の均等加重 window=50 は、273# の `max_kill_duration_sec: 1800` と probe mechanism (現在無効) によりデッドロックは防止されている。致命的ではない。
3. 代替案: time-weighted window（過去 N 時間の fill のみを使用、count 上限も設定）

**再起動前に修正が必要か**: No（`max_kill_duration_sec` により最悪ケースが制限されている）

---

## E. Sell Post-Fill Wait Asymmetry — **MEDIUM**

### コード参照
- [fill_test.yaml](configs/v460/fill_test.yaml#L32-L33)
  ```yaml
  post_fill_wait_sec: 30.0       # buy
  post_fill_wait_sec_sell: 90.0   # sell (168# §4.1: 30→90s)
  ```

### 現状の挙動
- Buy fill 後: 30 秒待って PnL 計測 → `post_fill_30s_pnl` として記録
- Sell fill 後: 90 秒待って PnL 計測 → 同じ `post_fill_30s_pnl` フィールドに記録

### 問題点

1. **計測時間の非対称性が kill 判定にバイアスを与える**:
   - Buy: 30 秒後の PnL → 短期の価格変動のみ反映
   - Sell: 90 秒後の PnL → 中期のトレンド成分も反映
   - Sell の PnL は buy と比べて分散が大きくなる（時間とともに価格のランダムウォーク成分が σ√t で成長）
   
2. **rolling window に混在する異なる horizon の PnL**: `DynamicKillManager` の rolling mean は buy/sell で別々に管理されるため、直接の cross-contamination はないが、**kill 閾値が同じ尺度で設定されている** （sell: -0.3bps, buy: -0.8bps）。90 秒の PnL は 30 秒の PnL より本質的に分散が大きく、同じ閾値でも sell の方が false positive (不必要な kill) が多くなる可能性がある。

3. **ただし**: 168# §4.1 の根拠は「PnL120 > 0 が多い」ことから、sell は長く保持するほど有利という実証データに基づく。sell threshold が -0.3 (buy の -0.8 より厳しい) なのは、この非対称性を暗黙に考慮している可能性がある。

### 市場理論
Maker 戦略における sell の PnL は「(fill 価格 - Δt 後の mid) / mid × 10000 bps」で計算される。Δt が大きいほど：
- **有利**: トレンド反転（sell 後に下落 → 利益）を捕捉できる
- **不利**: トレンド継続（sell 後にさらに上昇）をより大きな損失として計上

BTC/JPY はトレンドフォロー傾向（autocorrelation > 0）があるため、sell 後 90 秒のほうが trending_up での損失計上が大きくなり、dynamic_kill が正当に発動しやすい。**これは意図的な設計の可能性が高い**。

### 推奨アクション
1. **確認**: sell の `post_fill_30s_pnl` フィールド名が誤解を招く（実際は 90s）。ドキュメントまたはフィールド名の clarification を推奨
2. **P2**: kill 閾値を PnL horizon に合わせてスケーリングすることを検討（例: sell 閾値を σ√(90/30) ≈ 1.73 倍に緩和して horizon 差を補正）
3. **即時**: 現状は sell threshold -0.3 が buy の -0.8 より厳しい設定であり、この非対称性と 90s wait の相互作用を意識した調整が必要だが、340# の符号修正 + 341# の revert で基本パラメータは健全

**再起動前に修正が必要か**: No（informational — 現状のパラメータ設定は 168# の実証データに基づいている）

---

## F. Regime Detection Feedback Loop — **OK** (既に対策済み)

### コード参照
- [orchestrator_pre_cycle.py](scripts/v460/lib/orchestrator_pre_cycle.py#L649-L660) メインループでの regime 更新
- [fill_cycle_executor.py](scripts/v460/lib/fill_cycle_executor.py#L970-L990) executor 内での regime 更新
- [regime_detector.py](scripts/v460/lib/regime_detector.py) `update()` — `mid_price` ベース（fill 非依存）

### 現状の挙動

**Regime detector は fill に依存しない**。`update(timestamp, mid_price)` は OB の mid price で更新され、fill の有無に関係なく毎サイクル呼び出される：

1. **メインループ先頭** (`orchestrator_pre_cycle.py` L649): fallback price (OB mid) で毎サイクル更新
2. **Executor 内** (`fill_cycle_executor.py` L980): fill 時は `mid_at_fill`、未 fill 時は fallback price で更新

158# で「skip パスでも regime_detector.update() が呼ばれる」修正が完了済み。kill 中でもサイクルは回り続け（sleep して次のサイクルに進む）、regime は毎回更新される。

### 結論
**このリスクは 158# で完全に対策済み**。regime detection は市場データ（OB mid price）のみに依存し、fill activity には依存しない。kill 中でもレジーム遷移は正常に検出される。

**再起動前に修正が必要か**: N/A（対策済み）

---

## G. Daily Drawdown Guard / Dynamic Kill 相互作用 — **MEDIUM**

### コード参照
- [daily_drawdown_guard.py](scripts/v460/lib/daily_drawdown_guard.py#L87-L300) `DailyDrawdownGuard`
- [orchestrator_pre_cycle.py](scripts/v460/lib/orchestrator_pre_cycle.py#L159) `is_halted()` チェック
- [orchestrator_post_cycle.py](scripts/v460/lib/orchestrator_post_cycle.py#L119) `update_pnl()` 呼出し

### 現状の挙動

Daily Drawdown Guard と Dynamic Kill は**独立に動作**する：

| 制御 | データソース | リセット | スコープ |
|------|-------------|---------|---------|
| Dynamic Kill | rolling 50 fill PnL (bps) | 新データで自動更新 / `max_kill_duration_sec` | 日をまたいでも rolling window 維持 |
| DD Guard | 日次累積 PnL (bps) | TZ 設定による日替わり | 1 日限定 |

### 問題点

1. **日替わり境界での状態不整合**: 223# で指摘された通り、DD Guard は日替わりで全リセット（per-side halt も解消）されるが、Dynamic Kill の rolling window は日をまたいで持ち越される。日替わり直後：
   - DD Guard: クリーンスタート（halted=False, pnl=0）
   - Dynamic Kill: 前日の悪い PnL を 50 fill 分保持 → kill 状態が継続
   - **結果**: DD Guard は「取引可」だが Dynamic Kill が「kill 中」→ 取引不可。ユーザーから見ると「日が変わったのに回復しない」

2. **DD Guard が kill 中に蓄積しないデータ**: Dynamic Kill 中に fill がゼロ → DD Guard の `update_pnl()` も呼ばれない → DD Guard の daily_pnl_bps は実態より良く見える（fill していないので損失も発生しない）。kill 解除後のバースト的な fill で DD Guard が急激に悪化する可能性。

3. **per-side halt と dynamic kill の重複**: DD Guard の per-side halt（例: sell halt）と sell_dynamic_kill が同時に発動する場合、解除タイミングが異なる（DD は cycle count、kill は rolling PnL 改善）。片方が解除されても他方が継続 → 見かけ上の回復がない。

### 市場理論
リスク管理の多層化（defense in depth）自体は正しいが、各レイヤーが独立に同じ side を抑制すると、**解除条件の AND 結合**になり、回復が遅れる。MM 理論的には、損失後の再参入は「市場が変わった」ことの検証であり、段階的に行うべき（273# halt recovery grace period はこの思想）。

### 推奨アクション
1. **P2**: DD Guard halt 解除時に、対応する DynamicKillManager の rolling window に「ニュートラル PnL」をパディングして、kill → halt 相互ロックを緩和
2. **P2**: Dynamic Kill の `max_kill_duration_sec` を DD Guard の `cooldown_release_sec` と整合させる（両方 30 分など）
3. **計測**: 日替わり直後の kill 継続頻度をログから確認し、impact を定量化

**再起動前に修正が必要か**: No（273# の `max_kill_duration_sec: 1800` が safety valve として機能。268# のデッドロック事例は既に対策済み）

---

## 優先度整理

### P1（中期: 次回デプロイまでに）
| ID | 施策 | 期待効果 |
|----|------|---------|
| C | skip_gate に kill 状態コンテキストを渡す | sell pass rate 改善 → 収益機会回復 |
| A | forced fill を downweight 0.5 で track に投入 | kill 判定精度向上 |
| A | sell 側にも forced KPI 分離追加 | 可視性向上 |

### P2（中長期）
| ID | 施策 | 期待効果 |
|----|------|---------|
| B | inv_bypass をステップ関数 → gradual 化 (inv_relaxation max_bps 拡大 + bypass 廃止) | chattering 防止 |
| D | EWMA への移行検討 | regime 遷移への高速応答 |
| E | PnL horizon 差の閾値スケーリング | sell kill の false positive 低減 |
| G | DD Guard / Dynamic Kill のリセット整合 | 日替わり回復の改善 |

### 対策済み（追加作業不要）
| ID | 状況 |
|----|------|
| F | 158# で regime 更新が fill 非依存化済み |

---

## 総合評価

340# の符号修正後、**パラメータレベルのバグは解消された**。残る課題は設計レベルの構造的問題であり、特に **Finding C（skip_gate / dynamic_kill 二重抑制）** が収益性に最も大きな影響を与えている可能性が高い。338# のデータ（skip_gate=51 vs sell_dynamic_kill=42）が示す通り、2 つの独立フィルターが同等の頻度で sell を抑制しており、合計の sell 抑制率は個別の和ではなく積（multiplicative）に近い。

ただし、これらの修正はいずれも**再起動前に必須ではない**。340# + 341# の修正（符号修正 + パラメータ revert）により、bot は正常なパラメータ空間で動作する。上記の設計改善は中期的な収益性向上のための施策。
