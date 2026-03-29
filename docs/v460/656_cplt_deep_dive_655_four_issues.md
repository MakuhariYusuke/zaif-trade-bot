# 656# 655# 新規4課題の深堀り — 金融工学・市場理論・設計面からの多角的分析

- **日付**: 2026-03-30
- **目的**: 655# §2 で発見した4課題について、コード実地検証を踏まえた **前提の修正** と、金融工学・市場微構造理論・情報理論・制御工学の知見に基づく **代替アプローチの網羅的探索** を行う。
- **入力**: 655#, 654#, 650#, 536#, コードベース実地検証結果
- **方針**: 単一解の提示ではなく、各課題に対して複数の解法を理論的根拠とともに列挙し、トレードオフを明示する。

---

## §0 前提修正 — 655# §2.1 は誤りを含む

### 0.1 max_skip_rate と toxic_sell_veto の独立性（前提修正）

コード検証の結果、**655# §2.1 の前提は部分的に誤っていた**。

**事実**: rule-based skip (toxic_sell_veto, velocity_skip 等) は `SkipGateEvaluator.evaluate()` の L671-718 で **early return** する。この場合:
- `SkipGate.evaluate()` (ztb/ml/skip_gate.py L763) は **呼ばれない**
- `_recent_skips_buy/sell` バッファへの記録も **行われない**
- したがって **max_skip_rate (0.4) のバジェットを消費しない**

**残存リスク**: 完全に独立ではなく、以下の間接的干渉が存在する:
1. toxic_sell_veto が発動しない (条件不充足) → ML 判定に到達 → ML skip → max_skip_rate バジェット消費 — これは正常
2. toxic_sell_veto が連続発動 → sell サイクル自体が skip 扱いで FillRecord に記録 → **fill_rate 低下** → 収益機会の逸失
3. `_primary_consecutive_skip_count` (L790-807) は ML decision のみカウントするが、rule-based skip 後にも別経路でカウンタが進む可能性 → 要精査

**結論**: 655# で懸念した「ML skip が強制 fill に転換される」直接的リスクは **存在しない**。ただし fill_rate 低下という間接的リスクは残る。以下ではこの修正された前提の上で深堀りを行う。

---

## §1 課題 A: skip ガバナンス — rule vs ML の設計哲学

### 1.1 問題の再定義

max_skip_rate のバジェット消費問題は解消されたが、**skip の総量をどう管理するかという上位設計問題** は残存する。現行アーキテクチャには 3種類の skip メカニズムが並存:

| 層 | メカニズム | 性質 | 制限機構 |
|---|---|---|---|
| Rule-based | toxic_sell_veto, velocity_skip | 演繹的・確定的 | **なし** (無制限) |
| ML-based | SkipGate (LightGBM) | 帰納的・確率的 | max_skip_rate (0.4) |
| Evaluator-level | primary_consecutive_skip | 安全弁 | max_consecutive (configurable) |

**問題**: rule-based skip には rate limit がない。理論的には、低スプレッド+高ボラ+高OBI+高VPIN が持続する局面（闪崩后の回復期など）で toxic_sell_veto が **100% 発動** し続け、sell fill がゼロになる。

### 1.2 理論的フレームワーク

#### A) Glosten-Milgrom (1985) の情報非対称性モデル

toxic_sell_veto の理論的基盤。informed trader の存在確率 $\mu$ が高い (VPIN > threshold) とき、マーケットメイカーの期待損失は:

$$E[\text{loss}] = \mu \cdot \delta - (1-\mu) \cdot s/2$$

ここで $\delta$ は情報優位者による逆選択コスト、$s$ はスプレッド。**$\mu$ が十分大きければ取引しないのが最適** — toxic_sell_veto はこの non-participation 条件の実装。

**含意**: Glosten-Milgrom の non-participation は **一時的** であるべき。情報イベント後は $\mu$ が低下する（情報が価格に反映される）。**恒久的な non-participation は流動性供給の放棄** であり、マーケットメイカーの存在意義に反する。

#### B) Avellaneda-Stoikov (2008) の最適参加率

AS モデルでは、在庫リスク回避パラメータ $\gamma$ とポジション $q$ に対して最適なクオート幅が:

$$\delta^* = \gamma \sigma^2 T + \frac{2}{\gamma} \ln(1 + \gamma/k)$$

**skip すべきか、offset を広げるべきか** は $\gamma$ と $q$ の関数。skip (= $\delta \to \infty$) は $\gamma \to \infty$（無限リスク回避）の極限に対応し、**理論的には skip よりも offset 拡張が一般に優れる**。

#### C) 情報理論: Kelly 基準と参加率

Kelly (1956) の最適賭け比率 $f^* = p - q/b$ は、$p < q/b$（負の期待値）のとき $f^* = 0$（不参加）を指示する。しかし **$p$ の推定に不確実性がある場合、fractional Kelly ($f = \alpha \cdot f^*$, $\alpha < 1$) が最適**。

**含意**: toxic_sell_veto のバイナリ判定（skip/pass）は Kelly の 0/1 判定に相当。条件が閾値ギリギリのとき、**参加量を縮小する fractional approach** が情報理論的に優れる。

### 1.3 代替アプローチ一覧

| # | アプローチ | 理論基盤 | 実装コスト | リスク |
|---|-----------|---------|-----------|--------|
| **A-1** | 現状維持 (rule skip 無制限) | G-M non-participation | — | sell fill ゼロ局面の持続 |
| **A-2** | rule-based にも独立 rate limit | 参加率下限保証 | 低 | rule の安全性が rate limit で毀損 |
| **A-3** | skip → offset 拡張に変換 | A-S δ* 拡張 | 中 | 拡張しても fill される → 損失リスク |
| **A-4** | 条件強度に応じた段階的応答 | Kelly fractional | 中 | パラメータ増加 |
| **A-5** | 時間減衰付き veto | 情報半減期 | 低 | 減衰速度の最適化が困難 |
| **A-6** | Bayesian skip probability | 事後確率ベース判定 | 高 | モデル複雑化 |

### 1.4 推奨: A-4 (段階的応答) + A-5 (時間減衰)

**理由**:
- A-3 は既存の `velocity_skip_as_offset_enabled` と同じ設計思想。toxic_sell_veto にも offset 変換モードを追加するのは一貫性がある
- A-4 は toxicity_budget の YELLOW/ORANGE/KILL 3段階応答と同構造。toxic_sell_veto にも同様の段階を導入:
  - YELLOW (条件 3/4 成立): offset ×1.5
  - ORANGE (条件 4/4 成立、閾値ギリギリ): offset ×2.0
  - RED (条件 4/4 成立、閾値大幅超過): hard skip (現行動作)
- A-5 は Glosten-Milgrom の $\mu$ 時間減衰に対応。情報イベント後 $\mu$ は指数減衰するため、連続 veto 時に要求水準を段階的に緩和:

$$\text{threshold}_{\text{effective}}(n) = \text{threshold}_{\text{base}} + \alpha \cdot \ln(1 + n)$$

ここで $n$ は連続 veto 回数、$\alpha$ は緩和速度。対数的緩和により初回は厳格、連続時は徐々に通過を許容。

### 1.5 実装への示唆

```
[既存] velocity_skip → velocity_skip_as_offset (195# で変換済み)
[未着手] toxic_sell_veto → toxic_sell_veto_as_offset (同パターン適用可能)
```

既存の `velocity_skip_as_offset_enabled` パターンが参考実装として完備されている。toxic_sell_veto にも `toxic_sell_veto_as_offset_enabled` を追加し、hard skip / offset boost / 段階的応答の3モードから選択可能にする設計が最も一貫的。

---

## §2 課題 B: inventory skewing と regime_gate — 在庫管理の理論再考

### 2.1 問題の本質

regime_gate_enabled=true は 249# で「trending 中の方向性 α を inv_skew が妨害しないため」に導入された。これは **Gueant-Lehalle-Fernandez-Tapia (2012) の方向性在庫管理** の応用:

$$q^*(t) = -\frac{\mu}{2\gamma\sigma^2} + (q_{\max} + \frac{\mu}{2\gamma\sigma^2}) e^{-\gamma\sigma^2(T-t)}$$

trending 時に $\mu > 0$ (正のドリフト) であれば、最適在庫は **正** であり、在庫中立 ($q \to 0$) に回帰させるのは次善策。

**しかし**: この議論は **在庫が適切な範囲にある場合** にのみ成立する。650# の実測で JPY:BTC = 1:99 という偏重が確認されている。これは $|q| \gg q_{\max}$ の状態であり、Gueant-Lehalle の前提を逸脱している。

### 2.2 理論的背景

#### A) Ho-Stoll (1981) の在庫モデル

マーケットメイカーの最適気配値は在庫 $I$ に対して:

$$p^{\text{ask}} = V + s/2 + \alpha I, \quad p^{\text{bid}} = V - s/2 + \alpha I$$

ここで $\alpha = A\sigma^2$ (リスク回避度 × 分散)。**$I$ が大きくなるほど気配値全体がシフト** し、在庫を減らす方向に誘導する。regime に関係なく在庫水準に応じた応答が必要。

#### B) Amihud-Mendelson (1980) の preferred habitat

マーケットメイカーは「好ましい在庫水準」($I^*$) を持つ。現在在庫 $I$ が $I^*$ から乖離するほど、取引コストを在庫調整に振り向ける。

**含意**: regime_gate が trending 時に inv_skew をゼロにするのは、$I^*$ を「trending 方向に無限大」と設定したことに等しい。これは理論的に不合理で、$|I - I^*|$ に基づく応答が正しい。

#### C) Cartea-Jaimungal (2015) の regime-switching 在庫管理

2-regime (trending/ranging) モデルでは、regime 別の最適在庫水準と切替え確率を同時推定する。重要な結論は:

> **trending 中でも在庫上限は有限**。regime 切替え確率 $\lambda$ が正である限り、trending が終了するリスクをヘッジするための在庫圧縮が最適。

### 2.3 代替アプローチ一覧

| # | アプローチ | 理論基盤 | 実装コスト | リスク |
|---|-----------|---------|-----------|--------|
| **B-1** | regime_gate_enabled: false | 全廃 | 設定変更のみ | trending α 阻害 (249# 懸念) |
| **B-2** | extreme threshold ガード | Ho-Stoll inventory limit | 低 | 閾値設計 |
| **B-3** | regime 別 max_factor | Cartea-Jaimungal | 低 | パラメータ増加 |
| **B-4** | 在庫水準連続関数化 | Amihud-Mendelson preferred habitat | 中 | 非線形 skew |
| **B-5** | regime 切替え確率加味 | Cartea-Jaimungal λ | 高 | λ 推定困難 |
| **B-6** | Kelly × inv_skew ハイブリッド | Kelly lot 調整 + skew | 中 | 二重制御の複雑性 |

### 2.4 推奨: B-3 (regime 別 max_factor) — 段階的に B-4 へ

**B-3 の詳細**:
```yaml
inventory_skewing:
  regime_gate_enabled: false   # gate 廃止
  max_factor: 0.4              # ranging 時の現行値
  max_factor_trending: 0.15    # trending 時は抑制的に skew
```

**理論的根拠**: Cartea-Jaimungal の regime-switching モデルでは、trending 中の在庫最適値は ranging 時よりも方向側に偏るが、**ゼロではない**。max_factor を trending 時に 0.15 に抑制することで:
- 正常在庫範囲 (|imbalance| < 0.3) では skew ≈ 0 (α 非阻害)
- extreme 偏重 (|imbalance| > 0.3) では skew ≈ ±0.15 × imbalance (安全弁)
- ranging 時 max_factor=0.4 の 37.5% の制限で、249# の方向性保全懸念にも対応

**B-4 への発展**: 将来的には在庫水準を連続関数化:

$$\text{skew}(I) = \text{max\_factor} \cdot \tanh(\beta \cdot I)$$

tanh により $|I|$ 小では線形近似 (穏やかな skew)、$|I|$ 大で飽和 (max_factor 付近で ceiling) — Amihud-Mendelson の preferred habitat を smooth に実装。但し、先にまず B-3 で効果を計測してからが妥当。

### 2.5 655# §2.2 対策案への批判と修正

655# 案 (b)「extreme 偏重時のみ regime_gate 無視」は **binary threshold** の追加であり、threshold 前後で不連続なジャンプが発生する。B-3/B-4 は **連続的な応答** を提供し、threshold 設計の恣意性を回避できる。

---

## §3 課題 C: sell_dynamic_kill の理論的位置づけ

### 3.1 現行メカニズムの精査

コード検証で判明した sell_dynamic_kill の実態:

| パラメータ | 値 | 意味 |
|---|---|---|
| ewma_alpha | 0.05 | effective window ≈ 20 fills (RiskMetrics) |
| threshold_bps (trending_up) | -0.5 | EWMA PnL < -0.5bps で kill |
| threshold_bps (ranging) | -0.7 | ranging ではより寛容 |
| max_kill_duration_sec | 600 | 最大 10分 kill |
| toxicity_budget_enabled | true | YELLOW/ORANGE/KILL 3段階 |
| inv_relaxation_max_bps | 0.5 | BTC long 時 threshold 緩和 |

**重要**: 655# で「EWMA alpha=0、toxicity_budget=false」と記述したが、**実際には alpha=0.05、toxicity_budget=true** である。sell_dynamic_kill は 451# 以降で大幅に進化しており、536# 時点の「遅行サンドバッグ」という評価は **現行実装には当てはまらない** 可能性がある。

### 3.2 理論的フレームワーク

#### A) CUSUM 変化点検出 (Page 1954)

sell_dynamic_kill の EWMA は **指数重み付き CUSUM** の一種。cumulative sum が threshold を超えたら regime change (= kill) を宣言。統計的品質管理 (SQC) における管理図の思想。

**CUSUM の特性**:
- Type I error (false alarm): threshold が低すぎると正常局面で kill → 機会損失
- Type II error (missed detection): threshold が高すぎると損失局面を見逃す → 実損
- **ARL (Average Run Length)**: 変化点からkill発動までの平均遅延

現行設定 (threshold=-0.5, alpha=0.05) の ARL を推定すると:
- σ(PnL per fill) ≈ 3-5bps (650# 実測) のとき、mean shift -2bps → ARL ≈ 5-8 fills
- fill 間隔 ≈ 3-30分 → **検出遅延 15-240分** — 536# の「遅行」批判は ARL で裏付けられる

#### B) Optimal Stopping (Shiryaev 1978)

「いつ kill すべきか」は最適停止問題。事前 (kill せず取引継続) のコスト $c$ と事後 (kill して機会損失) のコスト $d$ のトレードオフ:

$$\tau^* = \inf\{t : \Pi_t \geq A^*\}$$

ここで $\Pi_t$ は事後確率（市場が adverse regime に入った確率）、$A^*$ は最適停止境界。**EWMA threshold はこの $A^*$ の近似**。

**問題**: $A^*$ は $c/d$ 比に依存するが、現行設定はこの比率を明示的にモデル化していない。

#### C) 536# の批判再検討 — 予測型 vs 反応型

536# は sell_dynamic_kill を「損失を被ってから発動する遅行指標」と批判し、OFI/VPIN 等の「予測型指標」への移行を提案した。

**しかし**: 654# で toxic_sell_veto (VPIN+OBI 予測型) を導入した今、**sell_dynamic_kill は予測型が見逃した場合の二次防衛** として機能する。defense-in-depth（多層防御）の観点から、予測型と反応型の共存は理にかなっている。

問題は **二重遮断** ではなく、**二重防衛の制御が独立** であること。toxic_sell_veto と sell_dynamic_kill は互いの状態を知らず、協調しない。

### 3.3 代替アプローチ一覧

| # | アプローチ | 理論基盤 | 実装コスト | リスク |
|---|-----------|---------|-----------|--------|
| **C-1** | 現状維持 (独立二重防衛) | defense-in-depth | — | 過剰抑制局面の持続 |
| **C-2** | 廃止 (536# シナリオ A) | 予測型一元化 | 設定変更 | 予測型の盲点が無防備 |
| **C-3** | toxic_sell_veto 発動をEWMAフィードバック | 協調制御 | 中 | 結合度増加 |
| **C-4** | ARL 最適化 (alpha/threshold 再推定) | CUSUM 最適 ARL | 低 | 推定に履歴データ必要 |
| **C-5** | Bayesian online change detection | Shiryaev 最適停止 | 高 | モデル複雑化 |
| **C-6** | toxic_sell_veto → EWMA cooldown 統合 | 統合状態機械 | 中 | 設計複雑化 |
| **C-7** | kill を offset escalation に変換 | A-S δ* 漸増 | 中 | offset 上限 (ceiling) 制約 |

### 3.4 推奨: C-4 (ARL 最適化) + C-3 の軽量版

**C-4**: 650#+ のデータ蓄積後に EWMA のパラメータを最適化:
- **目標**: ARL₁ (変化点→検出) ≤ 5 fills、ARL₀ (正常時の false alarm 間隔) ≥ 100 fills
- **方法**: alpha と threshold を grid search。650# の fill_records が入力データ
- data_freshness_check_interval_sec (649# 分離済み) でデータは利用可能

**C-3 軽量版**: toxic_sell_veto 発動時に sell_dynamic_kill の EWMA に **仮想的な negative PnL** を注入:

```python
if toxic_sell_veto_triggered:
    kill_mgr.track(pnl_bps=-0.5)  # 仮想 AS 損失を EWMA に反映
```

これにより toxic_sell_veto → sell_dynamic_kill の「予測→反応」パイプラインが成立。toxic_sell_veto の発動が kill 判定を先行的に引き上げ、**予測型が見落とした場合も kill の感度が維持される**。

### 3.5 第三の視点: regret minimization

sell_dynamic_kill の存廃を **後悔最小化** フレームワークで再考:

- **Kill して機会を逃す後悔**: $R_{\text{miss}} = E[\text{profit} | \text{market recovers}] \times P(\text{recovery})$
- **Kill せず損失を食らう後悔**: $R_{\text{loss}} = E[\text{loss} | \text{adverse continues}] \times P(\text{adverse})$

**最適ポリシー**: $R_{\text{miss}} = R_{\text{loss}}$ の均衡点で kill を判定。現行の EWMA threshold はこの均衡点の近似だが、$P(\text{recovery})$ がデータ不足で推定困難な現段階では、**kill を残しつつ ARL を改善する C-4 が最善**。

---

## §4 課題 D: preflight_insufficient の理論的解析

### 4.1 問題の構造化

650# で 46.2% という異常に高い preflight_insufficient 率は、**在庫管理の失敗** と **残高管理の静的設計** の複合問題。

#### 現行の balance check フロー

```
cycle_start → get_balance(JPY) → compare vs lot×price×regime_mult
  → insufficient? → try opposite side → both insufficient? → skip cycle
```

**問題点**:
1. `get_balance()` は API 呼出し — 帰値は **直近の確定残高** であり、pending order を含まない
2. buy 注文発注 → 約定前に次サイクル → 残高は前回約定時のまま → insufficient
3. 特に **高頻度 cycling** (3-10秒間隔) で顕著。API の残高反映遅延が支配的

### 4.2 理論的フレームワーク

#### A) 在庫-残高の二重制約問題 (Garman 1976)

Garman (1976) の dealer model では、マーケットメイカーは **在庫制約** (BTC残高) と **現金制約** (JPY残高) の両方に直面する。いずれかがゼロになるとマーケットメイクが停止 (ruin)。

$$P(\text{ruin}) = 1 - \exp\left(-\frac{2\mu x_0}{\sigma^2}\right) \qquad (\mu < 0)$$

ここで $x_0$ は初期残高、$\mu$ は残高ドリフト、$\sigma^2$ はボラティリティ。**buy 注文が集中すると JPY 残高が片側に移動** し、ruin 確率が急上昇。

**含意**: 現行の静的 check は **Garman ruin を検出** しているに過ぎない。根本対策は ruin 確率自体を制御すること。

#### B) Optimal Execution (Almgren-Chriss 2001)

大口注文の最適執行理論では、**残高 constraint の下での最適分割** を扱う:

$$x_j = x_0 \cdot \frac{1 - e^{-\kappa(T-t_j)}}{1 - e^{-\kappa T}}$$

各時点の発注量 $x_j$ は残存期間と残高に応じて決まる。**lot_size を残高水準に応じて連続的に調整する** のは、Almgren-Chriss の最適分割の離散近似。

#### C) Stochastic Control: Hamilton-Jacobi-Bellman

残高管理を連続時間の確率制御問題として定式化:

$$\max_{u_t} E\left[\int_0^T r(x_t, u_t) dt\right] \quad \text{s.t.} \quad dx_t = f(x_t, u_t) dt + \sigma dW_t$$

ここで $x_t$ は残高状態、$u_t$ は制御変数 (lot_size, side 選択)、$r$ は即時報酬。**完全な解は HJB 方程式** だが、実装上は近似ヒューリスティクスで十分。

### 4.3 代替アプローチ一覧

| # | アプローチ | 理論基盤 | 実装コスト | リスク |
|---|-----------|---------|-----------|--------|
| **D-1** | pending order 推定消費の control | 会計原則 (committed balance) | 中 | API 非同期性 |
| **D-2** | lot_size 動的縮小 (balance-proportional) | Almgren-Chriss 分割 | 低 | 極小 lot → 手数料比率悪化 |
| **D-3** | balance に基づく side 選択バイアス | Garman ruin 回避 | 低 | buy 側の機会減少 |
| **D-4** | 残高予測 (EWMA/linear projection) | 時系列予測 | 中 | 予測精度の限界 |
| **D-5** | cycle 間隔の適応的調整 | キューイング理論 | 低 | throughput 低下 |
| **D-6** | virtual balance (shadow accounting) | 複式簿記 | 中 | 実残高との乖離リスク |
| **D-7** | balance_margin_ratio の regime 別最適化 | Garman ruin x regime | 低 | パラメータ増加 |

### 4.4 推奨: D-2 (動的 lot 縮小) + D-6 (shadow balance)

**D-2 の詳細**:

balance_checker に既に auto-shrink ロジックがあるが、**shrink が発動してから insufficient を返す** という順序の問題がある。改善案:

```python
# 現行: insufficient → try shrink → still insufficient → skip
# 改善: pre-compute available_lot → adjust lot BEFORE check
available_jpy = last_jpy_free - pending_commitment
effective_lot = min(base_lot, available_jpy / (price * margin_ratio))
if effective_lot >= min_order_btc:
    # 縮小 lot で続行 (preflight 成功)
```

**期待効果**: Almgren-Chriss の最適分割に基づき、残高が減少しても **lot を縮小して参加を継続**。preflight_insufficient → cycle skip という binary 判定を、**lot 連続調整** に変換。

**D-6 の詳細**:

pending order の推定コストを「仮想残高」として管理:

```python
class ShadowBalance:
    """pending order のコストを追跡する仮想残高."""
    committed_jpy: float = 0.0
    committed_btc: float = 0.0

    def reserve(self, side: str, amount: float, price: float) -> None:
        if side == "buy":
            self.committed_jpy += amount * price
        else:
            self.committed_btc += amount

    def release(self, side: str, amount: float, price: float) -> None:
        # 約定 or キャンセル時に解放
        ...

    def available_jpy(self, actual_free: float) -> float:
        return max(0.0, actual_free - self.committed_jpy)
```

**理論的根拠**: 複式簿記の「引当金」概念。pending order は将来の支出義務であり、free balance から控除すべき。API の残高反映遅延を shadow balance で先行吸収。

### 4.5 根本的な問い: 46.2% は本当に「問題」か？

批判的視点として: preflight_insufficient は **JPY 枯渇の正しい検出** であり、問題は insufficient 率ではなく **JPY 枯渇自体**。

JPY 枯渇の原因:
1. BTC 売却不足 (inv_skew 三重無効化 → buy 偏重 → JPY 消費)
2. sell fill → BTC 取得 → JPY 回復しない (sell_entry は BTC を売らない)
3. 利益確定の JPY 還元がない

**因果**: inv_skew 無効化 → buy 偏重 → JPY 枯渇 → preflight_insufficient。**根本原因は inv_skew であり、preflight は症状**。654# P0-1 (inv_skew 調整) が効けば、preflight_insufficient も自然に減少する可能性がある。

D-2/D-6 は対症療法として有効だが、**inv_skew (課題B) の改善効果を先に計測** し、それでもなお insufficient が高い場合に着手するのが因果関係に忠実な順序。

---

## §5 課題間の相互作用マトリクス

4課題は独立ではなく、以下の相互作用がある:

```
                    [A: skip ガバナンス]
                         |
                    toxic_sell_veto 発動率
                         |
                         v
[B: inv_skew regime_gate] ←──────→ [D: preflight_insufficient]
  在庫偏重 ──→ JPY枯渇 ──→ buy 遮断
  │                                     ↑
  │                                     │
  └──→ sell_entry 集中 ──→ 損失蓄積 ──→ │
                    |                    │
                    v                    │
            [C: sell_dynamic_kill] ──────┘
              kill 中は sell 不能 → buy のみ → JPY 消費加速
```

**最も危険なフィードバックループ**:
1. inv_skew 無効 → BTC 偏重
2. → buy で JPY 消費 → preflight_insufficient
3. → sell_dynamic_kill が sell を止める → buy のみ続行
4. → さらに JPY 枯渇 → 1 に戻る

**ループの遮断点**: **B (inv_skew) の改善が最もレバレッジが高い**。在庫が適正化されれば、sell/buy のバランスが回復し、JPY 枯渇・sell 集中損失の両方が緩和される。

---

## §6 action priority の再評価

655# §7 の優先順位を、本分析の結果に基づき修正する:

### Priority 1: 計測 (24h)
- 654# 投入 (inv_skew + toxic_sell_veto) の効果観測
- fill_rate, inv_skew 発動率, toxic_sell_veto 発動率, preflight_insufficient 率を計測
- **判断**: inv_skew 改善で preflight_insufficient が自然減少するか？

### Priority 2: B-3 (regime 別 max_factor)
- 計測結果に関わらず「trending 中の inv_skew = 0」は理論的に不合理
- regime_gate_enabled: false + max_factor_trending: 0.15 を投入
- **理論根拠**: Cartea-Jaimungal regime-switching model

### Priority 3: A-4/A-5 (toxic_sell_veto 段階化)
- toxic_sell_veto の発動率が高い場合のみ着手
- velocity_skip_as_offset パターンを流用して offset 変換モードを追加
- **理論根拠**: Kelly fractional participation

### Priority 4: C-4 (sell_dynamic_kill ARL 最適化)
- 50+ RT 蓄積後に alpha/threshold を grid search
- **理論根拠**: CUSUM 最適 ARL

### Priority 5: D-2/D-6 (balance 動的管理)
- inv_skew 改善後もなお preflight_insufficient > 30% の場合のみ
- **理論根拠**: Garman ruin + Almgren-Chriss 分割

---

## §7 自己批判

1. **655# §2.1 の前提誤り**: rule-based skip が max_skip_rate を消費するという前提でバジェット枯渇問題を提起したが、コード検証で否定された。**文書作成前のコード検証の不足** が原因。「猜疑心を以て」の原則を文書そのものにも適用すべき。

2. **理論的フレームワークの過剰適用リスク**: Glosten-Milgrom、Avellaneda-Stoikov、Cartea-Jaimungal 等を引用したが、これらの最適解は **無限流動性・連続時間・正規分布** を前提とする。BTC/JPY の Coincheck 板は **流動性限定・離散時間・厚尾分布** であり、理論値と実装値のギャップは常に存在する。理論は **方向性の指針** として使い、パラメータは **必ず実測で較正** すべき。

3. **相互作用マトリクスの不完全性**: §5 のフィードバックループ分析は定性的。ループゲイン (各ステップの増幅率) の定量化にはシミュレーションが必要だが、現時点ではデータ不足。

4. **sell_dynamic_kill への再評価**: 655# で「536# が廃止を提言した遅行メカニズム」と批判的に紹介したが、実際には 451# で toxicity_budget 3段階化、549# で Winsorization、EWMA化 が既に施されており、536# 時点とは別物。**過去の批判をそのまま現在に適用する危険性**。

---

*以上*
