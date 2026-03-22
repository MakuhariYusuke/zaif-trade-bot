# 562# 560番・561番レビュー: 実データ検証と統合改善提案

- **日付**: 2026-03-23
- **目的**: 560#（Fill Test 再起動後パフォーマンス分析）と 561#（Sell 側テールリスク・DRC 提案）を現行コードベース・設定値と照合検証し、両著者への具体的アクション提案を行う
- **入力**: 560#, 561#, 現行 fill_test.yaml, 531#–555# の実装履歴, コードベース

---

## §0 総合所見

560# と 561# は同じ問題（Sell 側 AS による累積損失）を**異なる視座**から分析している：

| | 560# | 561# |
|--|------|------|
| **視座** | 実データ駆動の統計分析 | 理論駆動の構造設計 |
| **強み** | 定量的ファクト、再現可能なコマンド付き | Glosten-Milgrom/A-S 理論的根拠、DRC 数式 |
| **弱み** | 改善提案が「既に実装済み」のものを含む | 実装の現在地（537#–555#）を踏まえていない |
| **著者への方針** | 実装済み状態の棚卸し → 「未着手の真の P0」の再定義 | DRC を既存 OFI-Lite + δ* 天井との整合性で精密化 |

**共通の最大発見**: offset ceiling 0.30 でもなお 99% が飽和 → 537# の 0.25→0.35 提案は 542# で 0.30 に留まり、不十分。

---

## §1 560# ファクトチェック

### 1.1 数値の検証

| 560# の主張 | 検証結果 | 判定 |
|------------|---------|------|
| ceiling 0.25 で 99% 飽和 | 現行 ceiling は **0.30**（542# で引上げ済み）。99% 飽和は 0.30 下での値 | ⚠️ 文中に「0.25」との混在あり — ceiling 値の明示が必要 |
| sell_dynamic_kill が 653 回発動 | 現行 YAML では `sell_dynamic_kill.enabled: false` | ⚠️ **データ期間は旧設定（enabled=true）で稼動していた可能性高い**。現行とのギャップを明記すべき |
| Non-AS +2.31 bps / AS -7.24 bps | 理論的に整合。AS 自体の存在は market microstructure 的に不可避 | ✅ |
| fill rate 26.5% | spread_too_narrow(10.5%) + preflight(12.8%) が主因 | ✅ 妥当な分析 |
| CV sell widen -1.10 bps | CV favorable_tighten が sell 側で逆効果 | ✅ コード確認済み — basis skew が tighten→widen を反転させている |

### 1.2 560# の改善提案の現在地

560# が挙げる P0–P3 のうち、**既に実装済みまたは進行中**のものが多い：

| 560# 提案 | 実装状況 | 現在の値/状態 |
|----------|---------|-------------|
| P0: Ceiling 引上げ | ✅ 542# で 0.25→0.30 | 0.30。但し 99% 飽和が残存 → **さらなる引上げが必要** |
| P1: JST 22-23 時対策 | ✅ 467# で `hour_ceiling_mult` 設定済み | UTC 13: ×1.5 (→0.45), UTC 14: ×2.0 (→0.60) |
| P2: Sell AS 率改善 | ⚠️ 部分的 | OFI-Lite(543#), Toxicity Budget 独立(543#), sell_dynamic_kill disabled |
| P3: Fill Rate 回復 | ⚠️ 部分的 | min_spread 700→500(535#), composite_risk enabled(540#) |

**560# 著者へ**: P0–P3 は「未対策」ではなく「対策済みだが効果不十分」。次のアクションは「対策の calibration」であり、新規施策の提案ではない。

### 1.3 560# が見落としている重要データ

#### 1.3.1 SAC Sidecar が全期間で死亡

552# により SAC retrain が OHLCV データ停止で失敗 → 2026-03-19 16:34 以降 sidecar offset = **0 bps**。つまり 560# の分析期間中、372# Sidecar 段は**完全に無効**。これは PnL への影響が不明な交絡因子。

#### 1.3.2 OFI-Lite は稼動しているが 560# で言及なし

543# で OFI-Lite が maker_price.py に統合済み（545# で動的 boost 化、546# で感度パラメータ化）。OFI boost の効果計測が 560# に含まれていない。

#### 1.3.3 Composite Risk の効果計測なし

540# で `composite_risk_enabled: true`, `threshold: 1.0` が設定済み。560# の cancel reason 分布に composite risk の影響が反映されているはずだが、分析されていない。

---

## §2 561# ファクトチェック

### 2.1 理論的主張の検証

| 561# の主張 | 検証結果 | 判定 |
|------------|---------|------|
| Sell 100% ceiling 衝突 | ceiling は 0.30（561# は「0.25」と記載） | ⚠️ 値が古い。但し 0.30 でも 99% 飽和は事実 |
| AS は VPIN/Spread/Imbalance と低相関 | 543# OFI-Lite、545# δ* が追加された後のデータか不明 | ⚠️ **OFI-Lite 導入後の AS 相関を再計測すべき** |
| 報酬厳罰化の失敗（SNR 低下） | v459 以前の教訓として妥当 | ✅ |
| Glosten-Milgrom の生存境界線 | 理論的に正確 | ✅ |
| DRC 式の妥当性 | 後述（§2.2） | ⚠️ |

### 2.2 DRC 式の精密批評

561# の DRC 式：

$$Ceiling_{dynamic} = Base\_Ceiling \times \left( 1 + \alpha \cdot \frac{\sigma_{1min}}{Spread_{bps}} + \beta \cdot \Delta OFI \right)$$

**批評点 A: σ/Spread は次元的に不安定**

$\frac{\sigma_{1min}}{Spread_{bps}}$ は「ボラティリティ対スプレッド比」で無次元化されているが、Spread が極端に狭い（2 bps 未満）場合に発散する。分母に floor が必要：

$$\frac{\sigma_{1min}}{\max(Spread_{bps}, \epsilon)}$$

**批評点 B: ΔOFI の符号定義が曖昧**

ΔOFI > 0 が ceiling を**引き上げる**（退避方向）のか**引き下げる**（積極方向）のか不明。売り圧力（OFI < 0）時に sell ceiling を引き上げるべきだが、buy ceiling は逆。**side 依存の符号反転**が必要：

```python
# Sell: 売り圧力(OFI<0)時に ceiling↑ → β × (-ΔOFI) (正になる)
# Buy:  買い圧力(OFI>0)時に ceiling↑ → β × (+ΔOFI) (正になる)
adverse_ofi = -ofi_mean if side == "sell" else ofi_mean
```

これは OFI boost（545#）と同一の符号規約であり、コード整合性のために統一すべき。

**批評点 C: 既存 δ* 天井との重複**

545# で `delta_star_ratio` による sidecar 動的天井が既に実装されている：

```python
if delta_star_ratio > 1.0:
    ceiling_scalar = min(delta_star_ratio, 2.0)
    effective_max_boost_bps = config.sidecar_max_boost_bps × ceiling_scalar
```

DRC と δ* 天井は**同じ問題（固定天井の動的化）を異なるレイヤーで解こうとしている**。統合しなければ二重適用のリスクがある。

**批評点 D: 遅行性の自己矛盾**

561# は「追従遅れの罠」を自ら懸念しているが、$\sigma_{1min}$ 自体が 1 分間の遅行指標。$\Delta OFI$ を先行指標として入れることで部分的に補償されるが、**急騰の最初の数秒**（まさに AS が発生する瞬間）には対応できない。

### 2.3 SSG（Selective Skip-Gate）強化提案の検証

561# の「AS Probability を SAC 観測空間に入力する」提案は理論的に正しいが、**SAC が現在死亡中**（552# OHLCV 停止）である現実を考慮すると、即効性はない。SAC 復活が先決条件。

---

## §3 両著者への具体的アクション提案

### 3.1 560# 著者へ（データ分析担当）

| # | アクション | 優先度 | 理由 |
|---|----------|--------|------|
| A1 | **現行設定値の棚卸し表を追記** | P0 | 560# は ceiling=0.25 と記載するが実際は 0.30。sell_dynamic_kill=653 回だが現在 disabled。読者が混乱する |
| A2 | **OFI-Lite boost 効果の計測** | P0 | 543#–546# で導入済み。OFI boost 有無別の PnL 比較がない |
| A3 | **SAC dead 期間の明示** | P1 | 552# によりデータ期間中 sidecar=0bps。交絡因子として注記必要 |
| A4 | **Composite Risk 効果の分析** | P1 | 540# で threshold=1.0 に変更。soft gate block 率の変化を計測 |
| A5 | **unclamped +1.35 bps の分解** | P2 | ceiling 引上げの効果予測のため ceiling=[0.30, 0.35, 0.40] 別の反実仮想分析 |
| A6 | **CV favorable_tighten の sell 無効化検証** | P1 | sell widen -1.10 bps は構造的損失。sell 側のみ tighten 無効化した場合の影響試算 |

### 3.2 561# 著者へ（理論設計担当）

| # | アクション | 優先度 | 理由 |
|---|----------|--------|------|
| B1 | **DRC 式の修正**（§2.2 の A~D 反映） | P0 | 分母 floor、side 依存符号、δ* 統合、遅行性対策 |
| B2 | **既存実装の反映** | P0 | OFI-Lite(543#)、δ* 天井(545#)、OFI boost(546#) は既に稼動。DRC はこれらの**上位互換**として位置付けるべき |
| B3 | **DRC プロトタイプと hour_ceiling_mult の関係整理** | P1 | DRC が稼動すれば hour_ceiling_mult は理論上不要。移行計画を明記 |
| B4 | **SAC 復旧計画との整合** | P1 | SSG 強化は SAC retrain 復旧（553#–555# 修正済み）が前提。OHLCV パイプライン正常化確認後に再評価 |
| B5 | **反実仮想 PnL の定量推計** | P2 | 「0.25 以上の Offset を許容した場合」の試算を 560# のデータで実施 |

---

## §4 両文書の統合的問題認識

560# と 561# が共に示唆しているが、明示的に言語化されていない構造的問題がある。

### 4.1 「防衛の二重計上」問題

現行システムは**同一の AS リスクを複数の層で重複防衛**している：

```
AS リスク検知
├─ [Gate 層]  sell_dynamic_kill (disabled) ← binary kill (撤去済み)
├─ [Gate 層]  trending_sell_skip ← regime が trending_up なら sell skip
├─ [Gate 層]  velocity_skip ← 急変動なら skip
├─ [Offset 層] 196# Trending Sell offset boost
├─ [Offset 層] 202# VG Supplement
├─ [Offset 層] 240# Toxicity Budget
├─ [Offset 層] 306# sell_hour_offset_boost
├─ [Offset 層] OFI boost (546#)
├─ [Offset 層] 458# Macro Trend boost
└─ [Ceiling 層] 421# Final Clamp → 全て 0.30 に切り詰め
```

Gate 層で skip されなかった注文が Offset 層で 8 段もの防衛乗算を受け、最終的に**全て同一の ceiling 価格で発注**される。これは：

1. Gate 層の判断（「通過させた」）を Offset 層が上書きし
2. Offset 層の計算（各段の multiplier）を Ceiling 層が無効化する

**三層が互いを打ち消し合っている**。

### 4.2 根本原因: 各段が独立に AS リスクを推定している

各 offset 段は**自分の入力（VPIN, velocity, regime, hour, OFI, macro）のみ**を見て「AS リスクが高い」と判断し、独立に multiplier を出す。結果として：

- VPIN が高い（×1.5）AND velocity が高い（×1.5）AND trending_up（×1.3）AND macro_strong_up（×1.6）... → 合計乗数 = 4.68 → ceiling 0.30 に clamp

しかし、**VPIN が高い原因が velocity 急変であり、velocity 急変の原因が trending_up であり、trending_up が macro_strong_up の一部**である場合、これは**同一の現象を 4 回カウント**している。

**解決策**: 各段の入力間の**条件付き独立性**を検証し、相関の高い段を統合する。

### 4.3 提案: 「AS Risk Score」の一元化

536# シナリオ A の「ハブ化」に相当するが、より具体的に：

```python
# 各指標を [0, 1] に正規化
vpin_score = normalize(vpin, mu=0.3, sigma=0.15)  # sigmoid
ofi_score  = normalize(-adverse_ofi, threshold=0.5)
vel_score  = normalize(velocity_bps, mu=50, sigma=30)
hour_score = HOUR_RISK_TABLE[utc_hour]  # 0.0–1.0

# 重み付き最大値（OR 的結合: 最も危険な指標が支配）
as_risk = max(
    w_vpin * vpin_score,
    w_ofi  * ofi_score,
    w_vel  * vel_score,
    w_hour * hour_score,
)

# single multiplier
offset_mult = 1.0 + as_risk * (max_mult - 1.0)  # e.g., max_mult=3.0
```

**max 結合**を使う理由：
- 加算型（Σ）は多重計上を招く（現行の問題そのもの）
- 乗算型（Π）は指数的膨張を招く（現行の問題そのもの）
- max 型は「最も危険な 1 指標で決定」→ 多重計上が原理的に不可能

**短所**: max 型は情報を捨てる（VPIN=0.8 かつ OFI=0.7 でも max=0.8）。重み付きルート二乗平均 ($\sqrt{\Sigma w_i s_i^2}$) など準加法的結合も候補。

---

## §5 関係あるなしに関わらない改善提案

### 5.1 P-A: CV Favorable Tighten の Sell 側無効化（即効性 ★★★★★）

560# で sell widen = -1.10 bps と実証。コード上 `cross_venue_favorable_tighten_enabled` は bool で sell/buy 共通。

**提案**: sell 側のみ tighten を無効化するか、あるいは sell 側の tighten direction を反転する。

```yaml
# 新設: side 別制御
cross_venue_favorable_tighten_sell_enabled: false  # sell側は tighten しない
cross_venue_favorable_tighten_buy_enabled: true    # buy側は tighten 維持
```

**期待効果**: sell widen の -1.10 bps 損失が解消 → 全体 avg PnL が +0.05~0.10 bps 改善（fill 数比率から推定）。

### 5.2 P-B: Ceiling の段階的引上げ 0.30 → 0.40（即効性 ★★★★★）

560# で unclamped PnL = +1.35 bps / +0.76 bps。99% が clamped で -0.61 / -0.44 bps。 

現行 ceiling 0.30 は 542# で 0.25 から引き上げたが不十分。**pre_clamp avg = 0.3452（buy）/ 0.3154（sell）** であり、median すら ceiling を超えている。

**提案**: buy: 0.35, sell: 0.40（sell の AS リスクが高いため、sell にはより深い offset を許容）。

**リスク管理**: hour_ceiling_mult により危険時間帯はさらに拡大（UTC 14 で ×2.0 → 0.80）。hard_skip_mult=2.5 が catastrophic guard として機能。

### 5.3 P-C: Stage Multiplier の Max Cap 導入（即効性 ★★★★★）

537# で提案済み、**未実装**。pre_order_adjustments.py の `_apply_offset_multiplier()` に 1 行追加：

```python
if offset_mult is not None and stage_max_mult is not None:
    offset_mult = min(offset_mult, stage_max_mult)
```

各段の max を 2.0 に設定すれば、9 段全 max でも `0.05 × 2.0^8 = 12.8` → ceiling 0.40 で `min(12.8, 0.40) = 0.40`。ceiling の引上げと stage cap の導入は相補的。

### 5.4 P-D: fill rate 向上のための min_spread_jpy 動的化（中期 ★★★★☆）

560# で spread_too_narrow = 10.5%。min_spread_jpy=500 は 535# で引き下げたが、タイトスプレッド環境ではさらに多くの注文をブロック。

**提案**: Parkinson σ（305# 実装済み）を用いた動的最小スプレッド：

```python
min_spread_dynamic = max(
    config.min_spread_floor,           # e.g. 200 JPY
    parkinson_sigma_1h * k_spread,     # e.g. σ × 0.3
)
min_spread_dynamic = min(min_spread_dynamic, config.min_spread_cap)  # e.g. 1000 JPY
```

低ボラ時: min_spread が自然に下がり fill rate 向上。高ボラ時: min_spread が上がり AS 防衛。

### 5.5 P-E: 「テール集中時間帯」での注文量削減（即効性 ★★★☆☆）

560# §7.3 で JST 22-23 時（UTC 13-14）が buy/sell 両方のテール集中帯。hour_ceiling_mult で ceiling は拡大済みだが、ceiling 拡大は「より深い offset で発注」を意味し、fill rate が下がるだけで AS は減らない。

**代替案**: テール集中帯では**lot size を縮小**する。

```yaml
hour_lot_scale:
  13: 0.5   # JST 22h: lot を半分に
  14: 0.3   # JST 23h: lot を 30% に
```

**理論**: Kelly 基準 — edge が低い（or 不明な）ベットでは stake を減らすのが最適。AS 率が高い時間帯は edge が低い → lot 縮小が理論的正解。

### 5.6 P-F: preflight_insufficient 12.8% の削減（中期 ★★★☆☆）

560# で 566 回の preflight 不足。これは残高管理の問題であり AS とは無関係だが、fill opportunity の 12.8% を失っている。

**調査項目**:
- JPY 残高不足か BTC 残高不足か
- 在庫回転率（buy→sell のサイクル速度）
- **lot_size が資本に対して大きすぎないか**

### 5.7 P-G: DRC を hour_ceiling_mult の上位互換として再定義（中期 ★★★★☆）

561# の DRC を新規に作るのではなく、**既存の `resolve_offset_ceiling()` を拡張**する形で実装する：

```python
def resolve_offset_ceiling(self, side: str, *, utc_hour: int | None = None,
                           sigma_1min: float | None = None,
                           ofi_adverse: float | None = None) -> float:
    ceil = self._base_ceiling(side)
    
    # 467# hour_ceiling_mult (既存)
    if utc_hour is not None and self.hour_ceiling_mult:
        mult = self.hour_ceiling_mult.get(utc_hour)
        if mult is not None:
            ceil *= mult
    
    # 561# DRC 拡張 (新規)
    if self.dynamic_ceiling_enabled and sigma_1min is not None:
        spread_bps = max(self.current_spread_bps, 1.0)  # floor
        vol_factor = 1.0 + self.drc_alpha * (sigma_1min / spread_bps)
        ofi_factor = 1.0
        if ofi_adverse is not None:
            ofi_factor += self.drc_beta * max(ofi_adverse, 0.0)
        ceil *= vol_factor * ofi_factor
    
    return min(ceil, self.ceiling_hard_cap)  # catastrophic guard
```

**利点**: 既存コードパスに乗る。hour_ceiling_mult と DRC が自然に共存。テスト容易。

### 5.8 P-H: 「勝てる取引の質」の定量化と最適化（革新的 ★★★★★）

560# の最重要データ: **Non-AS +2.31 bps で健全**。問題は AS の -7.24 bps。

この非対称性は、**AS を 0 にする（不可能）のではなく、AS の損失額を制御**する方が効率的であることを示唆する。

$$E[\text{PnL}] = (1-p_{AS}) \times \bar{G} - p_{AS} \times \bar{L}$$

現在: $(1-0.274) \times 2.31 - 0.274 \times 7.24 = 1.68 - 1.98 = -0.31$

**損益分岐条件**: $\bar{L} < \frac{(1-p_{AS})}{p_{AS}} \times \bar{G} = \frac{0.726}{0.274} \times 2.31 = 6.12$ bps

つまり **AS 損失を平均 7.24 → 6.12 bps に 15% 改善するだけで黒字転換**。

これは ceiling 引上げ（AS 取引の offset を深くする → $\bar{L}$ が下がる）で直接達成できる。AS 率 $p_{AS}$ を下げる（skip/kill を増やす）と fill rate が下がり $\bar{G}$ の機会も減るため、**AS 率を下げるより AS 損失額を下げる**ほうが効率的。

### 5.9 P-I: Avellaneda-Stoikov 参照スプレッドの導入（長期 ★★★★★）

537# §5.6 の提案を再掲。必要パラメータは全て既存実装に存在：

| A-S パラメータ | 現行実装 | 状態 |
|---------------|---------|------|
| σ (ボラティリティ) | Parkinson σ (305# `market_theory.py`) | ✅ |
| q (在庫) | `inventory_imbalance` (226# P5) | ✅ |
| κ (注文到着強度) | fill_rate 計測（560# に raw データあり） | ⚠️ 要追加 |
| γ (リスク回避) | 手動パラメータ | ⚠️ 要設定 |

A-S 最適スプレッドを base_offset の参照値として使えば、パイプラインの出発点が理論的最適値になる。

---

## §6 560# 固有の発見に基づく追加分析

### 6.1 日別パターンの構造的解釈

560# §4 の日別データから、**fill rate と PnL の負の相関**が観察される：

| 日付 | Rate | avg PnL | 解釈 |
|------|------|---------|------|
| 3/15 | 17% | -1.44 | 低 fill + 高損失 = 市場が hostile（全ての fill が AS） |
| 3/16 | 14% | +0.86 | 低 fill + 好 PnL = Gate が正しく機能（AS を回避） |
| 3/19 | 34% | -1.26 | 高 fill + 高損失 = Gate が甘すぎ（AS を通過させた） |
| 3/20 | 53% | +0.03 | 高 fill + 収支均衡 = 市場が benign |

**3/15 vs 3/16**: 同程度の低 fill rate でも結果が真逆 → **fill rate 自体は品質指標ではない**。Gate の判断精度（AS を正しく skip し、non-AS を正しく通過させる F1 スコア）が真の KPI。

**提案**: fill の事後分析で Gate の F1 スコア（precision/recall for AS detection）を算出し、Gate calibration のフィードバックに使う。

### 6.2 Top 5 損失の集中度

Top 5 で -225.69 bps = 全損失 -3,591.83 bps の 6.3%。これは テール分布として**比較的健全**（一般に上位 5 fills で 20% 以上を占めると fat tail 問題）。

ただし最悪 fill = -72.65 bps は異常値。この 1 fill だけで 11 日間の Non-AS 利益 (+2.31 × 1156 non-AS fills ≒ +2670 bps) の 2.7% を消す。

**提案**: 単一 fill の最大損失を制限する **per-fill loss cap** の検討。例えば fill 後の AS-PnL が -30 bps を超えた場合、即座に反対売買で損切り。ただし取引コスト（taker fee + spread cost）との兼ね合いで、-30 bps の損切りが -35 bps の実現損になる可能性あり → 要計算。

---

## §7 結論

### 最優先アクション（両著者共通）

1. **Ceiling 0.30 → sell: 0.40, buy: 0.35** — 99% 飽和を実質的に解消し、AS 損失額を 15% 削減すれば黒字転換（§5.8 の損益分岐分析）
2. **CV favorable_tighten の sell 側無効化** — 構造的に -1.10 bps の損失源
3. **Stage max_mult 導入** — 537# 提案済み・未実装。ceiling 引上げの前提条件

### 560# 著者の次のステップ

- 既存設定値の棚卸し表を追記し、データと現行設定の不整合を解消
- OFI-Lite boost / Composite Risk / SAC dead の影響を定量化

### 561# 著者の次のステップ

- DRC 式の修正（分母 floor、side 依存符号、δ* 統合）
- 既存 `resolve_offset_ceiling()` の拡張として DRC を位置付け

### 中長期のアーキテクチャ

- 多段独立乗算 → AS Risk Score 一元化（§4.3）による多重計上の根絶
- A-S 最適スプレッドの参照値導入
- Gate の F1 スコア計測によるフィードバックループ構築

---

*以上*
