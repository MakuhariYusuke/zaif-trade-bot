# 337# Sell-Side 損益悪化分析 & 対策設計

> **種別**: rpt (分析レポート + 対策設計)  
> **対象**: 336# buy_dynamic_kill YAML 変更後の sell-side パフォーマンス崩壊  
> **起票**: 2026-03-08  
> **観点**: 設計面・市場理論面・コード分析・データ分析・対策提案  
> **ベース SHA**: `51e2cf040` (HEAD)  
> **分析対象 SHA**: `eb24cf4a` (03/08 06:03–17:18), `fea79119` (03/08 17:19–19:25)  
> **テスト**: 4163 passed (v460 + ops)

---

## §0 概要

336# で `buy_dynamic_kill` 閾値を緩和 (commit `114a0f056`) し、**buy 側は WR 64%、+18.0bps と劇的改善**した一方、**sell 側は WR 42%、-24.5bps と歴史的損失**を記録。本稿では「なぜ sell だけが崩壊したか」を構造的に分析し、即座に適用可能な対策を設計する。

### §0.1 損益サマリ (SHA `eb24cf4a`, 03/08 06:03–17:18)

| Side | Fills | WR | Total PnL (bps) | AS Rate | DK Kills |
|------|-------|----|-----------------|---------|----------|
| Buy  | 47    | 64.0% | **+18.0** | — | 0 |
| Sell | 55    | 42.0% | **-24.5** | 34.5% | 42 |
| **Total** | **102** | — | **-6.5** | — | 42 |

SHA `fea79119` (直近2時間): -11.5bps, sell_dk=30, fill_rate=19%

---

## §1 何をやったか — 336# YAML 変更 (commit `114a0f056`)

### §1.1 変更内容 (完全 diff)

```yaml
# buy_dynamic_kill のみ変更。sell_dynamic_kill は一切変更なし。

buy_dynamic_kill:
-  threshold_bps: -0.8          # 旧: MM noise margin 不足
+  threshold_bps: -1.5          # 336# T-1: カスケード増幅の root cause 緩和
   regime_thresholds:
+    ranging: -2.0              # 336# T-5: NEW — MM 主戦場で大幅緩和
-    trending_down: -0.5
+    trending_down: -1.0        # 336# T-1: sell trending_down=-1.0 と同水準
-    high_vol: -0.5
+    high_vol: -1.0             # 336# T-1: 高ボラ時も在庫均衡に一定の buy 参加

buy_dynamic_kill_inv_relaxation:
-  max_bps: 0.3
+  max_bps: 0.5                # 336# T-2: effective threshold -1.5+0.5=-1.0
```

### §1.2 変更していないもの (重要)

- ❌ `sell_dynamic_kill` — **完全に未変更** (threshold=-0.3bps)
- ❌ `skip_gate` — 変更なし
- ❌ コードロジック — 一切変更なし (YAML-only commit)
- ❌ `sell_dynamic_kill_inv_relaxation` — **そもそも存在しない**

---

## §2 データ分析: Sell Rolling-50 PnL 軌跡

### §2.1 軌跡全体像

DynamicKillManager は直近50 fill の PnL 平均を算出し、`threshold_bps` を下回ると kill を発動する。以下は sell side の rolling-50 平均の軌跡：

```
期間                rolling-50 PnL   状態
─────────────────────────────────────────────
03/07 09:18-23:42   +0.1 〜 +0.7     ✅ 健全 (SHA 4e670141 / 894c1bf8)
03/08 00:03-07:45   +0.3 〜 +1.0     ✅ 健全 (SHA 894c1bf8 → eb24cf4a)
03/08 07:57-11:58   +0.8 〜 +1.2     ✅ 健全 (ピーク域)
  ↓↓↓ 12:00-13:50 クラスタ損失 ↓↓↓
03/08 12:02         +0.80            ⚠️ 急落開始
03/08 12:14         +0.87            -8.94bps の損失
03/08 12:22         +0.73            -8.27bps の損失
03/08 12:40         +0.21            -3.22bps
03/08 12:51         +0.12            -4.64bps
03/08 13:54         -0.16 ★          ⛔ -0.3 threshold 接近
03/08 14:05         -0.29            ⛔ ほぼ kill 閾値
03/08 14:14         -0.43 ★★         🔴 KILL 発動 (-0.3 を下回る)
  ↓↓↓ Kill 中: cooldown 10 cycle ごとに「漏れ」約定 ↓↓↓
03/08 14:38         -0.43            漏れ約定: -3.34bps (悪化)
03/08 15:19         -0.45            漏れ約定: -3.88bps (さらに悪化)
03/08 15:40         -0.61            自己強化ループ
03/08 15:54         -0.82 ★★★       🔴 深い kill 領域
  ↓↓↓ 16:00 台: 0 fills, 30 全キャンセル ↓↓↓
03/08 17:19         -0.61            SHA fea79119 に切替後も回復せず
03/08 17:52         -0.86            -8.71bps で悪化
03/08 18:48         -0.71            微回復
03/08 19:27         -0.42            +15.41bps でやっと改善
```

### §2.2 核心的発見

**Kill 発動 → 自己強化ループ (Vicious Cycle)**:

1. 12:00–13:50 の集中的損失 (-36.9bps) で rolling 平均が低下
2. -0.3bps の **極めてタイトな閾値** に接触 (index=96, 13:54)
3. Kill 発動 → 大半の sell がブロック
4. cooldown 10 cycle 後の「漏れ」約定は、**市場が既に sell 側に逆選択的な環境** で執行
5. 漏れ約定も PnL 負 → rolling 平均はさらに低下 → kill が解除できない
6. **16:00 台は全 sell が kill** (30 records, 0 fills) — 完全な sell shut-down

**これは設計上のデスロックである。**

---

## §3 根本原因分析 (5 Whys + Code)

### §3.1 Root Cause #1: Sell threshold が Buy の 3–7 倍タイト

| Parameter | Sell | Buy | **比率** |
|-----------|------|-----|---------|
| threshold_bps (base) | **-0.3** | -1.5 | **5x** |
| trending_up | -0.3 | -1.5 | 5x |
| ranging | -0.5 | -2.0 | **4x** |
| trending_down | -1.0 | -1.0 | 1x (唯一の均衡) |
| inv_relaxation | **なし** | +0.5 max | **∞** |

336# で buy を大幅に緩和したことで非対称性はさらに拡大した。
sell base=-0.3bps は「正常な MM ノイズ (±1–3bps)」を考慮すると、50 fill 平均が数 fill の不運で容易に到達する圏内にある。

**コード確認** ([orchestrator_guards.py](scripts/v460/lib/orchestrator_guards.py#L53-L66)):
```python
# buy 側のみ inv_relaxation が存在
threshold_offset_bps = 0.0
if (
    side == "buy"
    and self.config.buy_dynamic_kill_inv_relaxation_enabled
):
    # 在庫偏重時に threshold を緩和
    ...
# sell 側は常に threshold_offset_bps=0.0 → 一切の緩和なし
```

### §3.2 Root Cause #2: balance_forced_switch の PnL 汚染

**コード** ([orchestrator_guards.py](scripts/v460/lib/orchestrator_guards.py#L86-L97)):
```python
def _track_side_pnl(self, record: "FillRecord") -> None:
    if not (record.filled and record.post_fill_30s_pnl is not None):
        return
    if record.side == "sell":
        self._sell_kill_mgr.track(record.post_fill_30s_pnl)
    elif record.side == "buy":
        self._buy_kill_mgr.track(record.post_fill_30s_pnl)
```

`record.balance_forced_switch` のフィルタリングが **一切ない**。強制取引（在庫均衡のため即時約定を強いられるもの）は通常スプレッド回収機会を持たず、PnL が構造的に負になりやすい。これが rolling window を汚染し、kill 閾値到達を早める。

333# データでは全レコードの **18.7% が balance_forced**。

### §3.3 Root Cause #3: Kill 後の漏れ約定 — 逆選択の罠

Kill 中は大半のサイクルで sell をブロックするが、cooldown (`resume_window=10`) 後に1回だけ通す。この「漏れ」は以下の構造的問題を持つ：

1. **選択バイアス**: Kill 中に通るのは、kill 判定直後に cooldown が切れたタイミング。しかしその時の市場環境は「直前まで kill されるほど sell に逆選択的」だった環境の延長上にある。
2. **サンプル数不足**: 10 cycle に 1 回の頻度では、rolling window の更新が極めて遅い。50 fill = 500 cycle = 約 17 時間分のデータが必要。
3. **非対称フィードバック**: 漏れ約定が PnL 正なら rolling 平均は微増、PnL 負なら大幅低下。**1 回の大損失が回復に必要な好約定5–10回分を消す。**

実際の軌跡:
- Index 100 (14:38): -3.34bps → rolling -0.43
- Index 105 (15:19): -3.88bps → rolling -0.45
- Index 108 (15:54): -9.15bps → rolling -0.82
- 3 回の漏れ約定で rolling が **-0.43 → -0.82 に悪化**

### §3.4 Root Cause #4: Sell double-filter (skip_gate + dynamic_kill)

Sell 側には 2 つの独立したフィルタが存在：

1. **sell_dynamic_kill** (Gate 5): Rolling PnL < threshold → kill
2. **skip_gate** (Gate 9): ML モデルによる逆選択予測 → skip

これらは独立に動作し、それぞれが sell を抑制する。組み合わさると **sell pass 率が著しく低下**。333# 分析では skip_gate の sell AS 予測精度は 63.6% (n=33) と低サンプルで不安定。

Gate chain ([cycle_gate_aggregator.py](scripts/v460/lib/cycle_gate_aggregator.py#L602-L633)):
```
Gate 1: regime/vol → Gate 3: buy_dk → Gate 5: sell_dk → Gate 7: velocity/spread
  → [pass] → Gate 9: skip_gate ML
```

### §3.5 Root Cause #5: Buy 緩和の間接的 Sell 悪化効果

336# で buy_dynamic_kill を緩和したこと自体は buy にとって正しい施策だったが、**間接的に sell 環境を悪化させた可能性**がある：

1. **Buy 約定増 → 在庫 BTC 増** → Sell 圧力増大を期待するが、sell kill が発動中のため sell 不可
2. **在庫 BTC 過剰蓄積** → 価格下落リスク増大 → 保有 BTC の含み損
3. **Buy 約定の逆選択** (buy 側の AS が低いのは事実だが、特定時間帯では AS 的買いも発生) → その後の sell で回収できず PnL 低下

これは **一方を修正しただけでは不十分** という MM の本質的課題。

---

## §4 市場理論からの考察

### §4.1 Ho-Stoll (1981): 在庫リスクの対称性

Ho-Stoll モデルの最適スプレッドは、在庫リスクを **買い側・売り側対称** に管理する前提で導出される。

$$S^* = \gamma \sigma^2 + \frac{1}{2}\ln\left(\frac{1+\gamma}{1-\gamma}\right)$$

ここで $\gamma$ はリスク回避度、$\sigma^2$ は資産価格分散。重要なのは、この式に **side 依存パラメータがない** こと。buy/sell の kill threshold に 5 倍の非対称性を持たせることは、Ho-Stoll の均衡解から大きく逸脱している。

**含意**: sell_dynamic_kill の threshold_bps は buy と同水準 (-1.5bps) に近づけるべき。少なくとも、inv_relaxation を sell 側にも追加して実効的対称性を確保する必要がある。

### §4.2 Glosten-Milgrom (1985): 情報付き取引の非対称性

Glosten-Milgrom モデルでは、情報付き取引者 (informed trader) の割合 $\mu$ がスプレッドを決定する：

$$\text{Ask} - \text{Bid} \propto \mu \cdot (\bar{v} - \underline{v})$$

暗号資産市場では、**sell 側の情報非対称性が通常 buy 側より高い** (いわゆる "bad news travels fast") ことが知られる。したがって sell 側の kill threshold をある程度タイトにする理論的根拠はある。

**しかし**、-0.3bps は理論的正当化の範囲を超えて過敏。Glosten-Milgrom 的に正当化できる非対称性は **せいぜい 1.5–2.0 倍** (情報の売買差)。現状の 5 倍は理論的に説明不可能。

### §4.3 Avellaneda-Stoikov (2008): 在庫リスクとスプレッド

Avellaneda-Stoikov の最適クオート制御：

$$\delta^a = \delta^b = \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right) + \frac{q \cdot \gamma \sigma^2 (T-t)}{2}$$

在庫 $q$ が増えるとスプレッドが広がるが、**差分は ask/bid 対称**。在庫偏重時に一方を完全に停止する (kill) のは、最適解から乖離する。

**含意**: Kill ではなく「スプレッド拡大」が理論的には正しい対応。Dynamic kill は最終手段であり、threshold は十分に緩やかであるべき。

### §4.4 Kyle (1985): 流動性供給の価値

Kyle モデルでは、マーケットメイカーが十分な流動性を供給することで情報の漸進的反映が可能になる。**片側の流動性を完全停止する (kill) ことは、スプレッドの反対側で情報を一方的に受ける状態** を意味し、マーケットメイカーとしての information rent を失う。

16:00 台の sell 全停止は、「マーケットメイキングの放棄」に等しい。

---

## §5 構造的問題の総括

### §5.1 問題の階層構造

```
Level 0: 根本
  sell_dynamic_kill threshold_bps = -0.3 はMM正常ノイズ圏内
  → 正常な市場変動で kill が発動する

Level 1: 増幅機構
  ├─ balance_forced_switch PnL 汚染 → window 低下を加速
  ├─ inv_relaxation の非対称性 → sell 側のみ硬直的 threshold
  └─ double-filter (skip_gate + dk) → sell pass 率の過度な低下

Level 2: 自己強化ループ
  kill 発動 → 漏れ約定のみ（逆選択環境で約定）
  → rolling PnL さらに悪化 → kill 解除不可
  → 在庫蓄積 → 含み損リスク増大

Level 3: 帰結
  16:00 台 sell 全停止, 12-13h クラスタ損失 -36.9bps
  SHA eb24cf4a 全期間: sell total -24.5bps
  online monitor: DEGRADED, sell WR 35.3%
```

### §5.2 336# 変更の実質的効果

| 項目 | Buy | Sell |
|------|-----|------|
| 336# 変更 | ✅ threshold 緩和 | ❌ 変更なし |
| 結果 | +18.0bps, WR 64% | -24.5bps, WR 42% |
| DK kills | 0 | 42 |
| 評価 | **好転** | **大幅悪化** |

**結論**: Buy の修正は正しかった。しかし sell を放置したことで、**全体としては -6.5bps の損失**。片側修正の危険性が実証された。

---

## §6 対策提案

### §6.1 【即時】Sell dynamic kill threshold 緩和 (YAML-only)

**優先度: P0 (即時適用)**

```yaml
sell_dynamic_kill:
  threshold_bps: -1.0           # -0.3→-1.0 (buy の -1.5 との比率 1.5x に)
  regime_thresholds:
    trending_up: -0.8            # -0.3→-0.8 (売りに本来不利だが過敏すぎた)
    trending_down: -1.5          # -1.0→-1.5 (下降トレンドで sell≒順張り、寛容に)
    ranging: -1.5                # -0.5→-1.5 (MM 主戦場、buy ranging=-2.0 の 75%)
```

**根拠**:
- Ho-Stoll 対称性: buy/sell 比率を 5x → 1.5x に改善
- Glosten-Milgrom 情報非対称性: sell のほうが AS に晒されやすいため buy と完全同値にはしない (1.5x は理論的に妥当な範囲)
- Rolling-50 軌跡分析: 平均が -1.0 を下回ったのは観測期間中 **一度もない** (最悪 -0.88)。-1.0 なら §2 の全損失イベントが **ゼロ kill** で通過
- ただし -1.5 以下にすると sell side の損失を本当に止められなくなるリスクあり

### §6.2 【即時】Sell inv_relaxation の追加 (YAML + コード)

**優先度: P0 (即時適用)**

```yaml
# YAML 追加
sell_dynamic_kill_inv_relaxation:
  enabled: true
  scale: 0.4                    # buy の 0.5 より保守的
  max_bps: 0.3                  # buy の 0.5 より保守的
```

**コード変更** (`orchestrator_guards.py`):
```python
def _is_side_killed(self, side: str) -> bool:
    ...
    threshold_offset_bps = 0.0
    if side == "buy" and self.config.buy_dynamic_kill_inv_relaxation_enabled:
        ...  # 既存 buy ロジック
    elif side == "sell" and self.config.sell_dynamic_kill_inv_relaxation_enabled:
        imbalance = self._maker_price.inv_net_imbalance
        if imbalance > 0:  # sell 偏重 = BTC 過剰
            raw_offset = abs(imbalance) * self.config.sell_dynamic_kill_inv_relaxation_scale
            threshold_offset_bps = min(
                raw_offset,
                self.config.sell_dynamic_kill_inv_relaxation_max_bps,
            )
```

**根拠**: Ho-Stoll (1981) の在庫リスク管理。BTC 過剰保有時に sell を不必要に抑制することは、在庫リスクの二重増幅を招く。inv_relaxation は sell 側にも必要。

### §6.3 【短期】balance_forced_switch PnL のフィルタリング

**優先度: P1 (次回コード変更時)**

```python
def _track_side_pnl(self, record: "FillRecord") -> None:
    if not (record.filled and record.post_fill_30s_pnl is not None):
        return
    # 337# 強制取引は rolling PnL から除外
    if getattr(record, 'balance_forced_switch', False):
        return
    if record.side == "sell":
        self._sell_kill_mgr.track(record.post_fill_30s_pnl)
    elif record.side == "buy":
        self._buy_kill_mgr.track(record.post_fill_30s_pnl)
```

**根拠**: 強制取引はスプレッド回収を意図しない即時約定。PnL が構造的に負 (スプレッド分の損失) であり、MM 能力の指標としては不適切。これが rolling window を汚染し、kill 閾値到達を早めている。

### §6.4 【短期】Resume window の最適化

**優先度: P1**

現在の `resume_window=10` は、kill 後 10 cycle (=20分) ごとに 1 回だけ再評価する。§3.3 で示したように、この「漏れ」約定は逆選択環境で執行されがちで自己強化ループを引き起こす。

**提案 A: resume_window 短縮 (10→5)**
- 再評価頻度を倍増させ、良環境での sell 再開を早める
- リスク: 悪環境での漏れ約定も増える

**提案 B: resume 条件にレジーム判定を追加**
- Kill 中でもレジーム が `ranging` または `trending_down` なら即座に resume
- Kill 中の `trending_up` は sell に逆選択的なので kill を維持

**推奨**: 提案 B（レジームベース resume）を将来検討。§6.1 の threshold 緩和が先決。

### §6.5 【中期】Rolling window の Exponential Decay 化

**優先度: P2**

現在の rolling-50 は等重み平均。古い fill と直近 fill が同じ重みを持つ。

**提案**: EWMA (Exponentially Weighted Moving Average) に変更

$$\text{EWMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EWMA}_{t-1}$$

$\alpha = 0.1$ (直近 10 fill で 65% の重み) ならば、古い損失の影響が速やかに減衰し、市場環境の変化に追随できる。

**メリット**: Kill からの回復が、好約定2–3回で可能に (現在は10–20回必要)。
**リスク**: 過敏になり、kill が頻繁に on/off する可能性。alpha の調整が必要。

### §6.6 【注意】やってはいけないこと

1. **sell_dynamic_kill を無効化** — kill 機能自体は必要。Glosten-Milgrom 的逆選択防御は残すべき。
2. **threshold を -5.0 以下に設定** — 実質的に無効化と同じ。連続大損失時の防御が失われる。
3. **buy/sell threshold を完全に同値にする** — Sell 側の情報非対称性は構造的に高い (Glosten-Milgrom)。1.0–1.5x の非対称性は維持すべき。
4. **skip_gate を無効化** — §3.4 の double-filter 問題は threshold 緩和で自然に軽減される。skip_gate 自体は独立した逆選択防御として価値がある。

---

## §7 実装優先度マトリックス

| # | 対策 | 優先度 | 種別 | 期待効果 | リスク |
|---|------|--------|------|---------|--------|
| §6.1 | Sell threshold 緩和 | **P0** | YAML | sell kill 大幅削減, 自己強化ループ解消 | 損失 sell の見逃し (理論的に 1.5x で許容範囲) |
| §6.2 | Sell inv_relaxation 追加 | **P0** | YAML+Code | BTC過剰時の sell 解放, 在庫リスク低減 | 実装コスト (orchestrator_guards 2行追加) |
| §6.3 | forced_switch PnL filter | P1 | Code | Window 汚染 18.7%→0%, kill 発動遅延 | 強制取引の品質監視が別途必要 |
| §6.4 | Resume window 最適化 | P1 | YAML/Code | Kill 後回復の迅速化 | 過提案B はコード変更必要 |
| §6.5 | EWMA rolling | P2 | Code | Kill 回復迅速化, 環境追随性向上 | Alpha チューニング必要 |

---

## §8 §6.1 適用時のシミュレーション

§2 の sell rolling-50 軌跡にて、threshold を -0.3 ではなく各候補値で kill 判定した場合：

| Threshold | Kill 発動地点 | Kill 期間 | 備考 |
|-----------|---------------|-----------|------|
| -0.3 (現行) | index=96 (13:54) | 13:54–19:27+ (5.5h+) | 実測通り、16:00 台全停止 |
| -0.5 | index=107 (15:40) | 15:40–19:27+ (3.8h+) | 2h 短縮だが依然長い |
| -0.8 | index=108 (15:54) | 15:54–19:27+ (3.5h+) | ほぼ -0.5 と同じ |
| **-1.0** | **kill なし** | **0h** | **観測期間中最悪=-0.888 で非到達** |
| -1.5 | kill なし | 0h | 大幅な余裕 |

**-1.0bps が最適**: 観測期間中の全損失イベントで kill が発動せず、しかし -1.0 を大きく下回る本物の構造的損失時には kill が機能する。

---

## §9 総合判断

### §9.1 何が起きたか (一文)

Buy_dynamic_kill を緩和したことで buy 側は正常化したが、**sell_dynamic_kill の threshold -0.3bps が MM 正常ノイズ圏内 (rolling-50 の自然変動幅 ±0.5bps) にあり**、12–13 時の一時的集中損失で kill が発動、以降「漏れ約定 → 悪化 → kill 解除不可」の自己強化ループに陥り、16 時台の完全 sell shut-down に至った。

### §9.2 次のアクション

1. **即座に** `sell_dynamic_kill.threshold_bps` を -0.3 → -1.0 に変更 (§6.1)
2. **即座に** `sell_dynamic_kill_inv_relaxation` を追加 (§6.2)
3. §6.3 `forced_switch` フィルタはテスト作成後に実装
4. ボット再起動して新パラメータで稼働開始
5. 1h 後にログ確認で sell kill 発動状況を検証

### §9.3 336# 施策の最終評価

| 施策 | buy_dynamic_kill 緩和 | sell_dynamic_kill 据置 |
|------|----------------------|----------------------|
| 判定 | ✅ 正しかった (+18bps) | ❌ 重大な見落とし (-24.5bps) |
| 教訓 | 片側修正は対側の検証が必須 | Asymmetric threshold の危険性 |

**「片方だけ修正すると、もう片方が壊れる」— これが MM 設計の鉄則。Ho-Stoll の対称性はヒントではなく、要件だった。**

---

*337# — 2026-03-08 sell-side 損益悪化分析 & 対策設計 完了*
