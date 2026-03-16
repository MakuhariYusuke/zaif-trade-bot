# v455 HFTコアロジック設計書: 校正マップと実行モデル (Rev.5)

## 概要
本ドキュメントは、v455における高頻度取引（HFT）化に向けたコアロジックの仕様を定義する。
Rev.5では、Rev.4への指摘（スリッページの片道/往復定義、Fee計算の厳密化、Side定義、パラメータ明記）を反映し、実装に向けた最終仕様とする。

---

## 1. 校正マップ (Calibration Map) 仕様

RLモデルの出力する `action` 値をそのまま信頼せず、過去のパフォーマンスに基づいて補正（Calibration）を行う。
**階層型フォールバック**と**保守的EV（期待値）判定**を採用する。

### 1.1 階層構造とデータ蓄積
データの希薄性（Sparsity）に対処するため、以下の階層順で統計情報を参照する。

1.  **Level 1 (Specific)**: `Regime` × `Action Bin` (例: Trend_Bull + Strong_Buy)
2.  **Level 2 (Regime)**: `Regime` 全体 (例: Trend_Bull)
3.  **Level 3 (Global)**: 全データ

### 1.2 統計量の更新 (EWMA)
全ての統計量は、市場環境の変化に追従するため **EWMA (Exponentially Weighted Moving Average)** で更新する。

- **重み $w_i$**: $\exp(-(t_{now} - t_i)/\tau)$
- **実効サンプル数**: $n_{eff} = (\sum w_i)^2 / \sum w_i^2$

#### 勝率 ($p_{win}$)
Beta分布による下側信頼限界（LCB）を用いる。
$$
p_{win\_LCB} = \text{BetaInv}(\alpha_{post}, \beta_{post}, 0.05)
$$
$$
\alpha_{post} = \alpha_{prior} + \sum w_i \cdot I(win_i)
$$
$$
\beta_{post} = \beta_{prior} + \sum w_i \cdot I(loss_i)
$$
($\alpha_{prior}=2, \beta_{prior}=2$)

#### 平均損益 ($AvgWin, AvgLoss$)
**単位: JPY/BTC (1BTCあたりの価格差)** で統一する。
**Gross（手数料・スリッページ控除前）** の理想約定価格に基づいて計算する。

$$
AvgWin_{gross} = \frac{\sum (w_i \cdot Profit_i \cdot I(win_i))}{\max(\sum (w_i \cdot I(win_i)), \epsilon)}
$$
$$
AvgLoss_{gross} = \frac{\sum (w_i \cdot |Loss_i| \cdot I(loss_i))}{\max(\sum (w_i \cdot I(loss_i)), \epsilon)}
$$
- $Profit_i, Loss_i$: `(ExitPrice - EntryPrice) * Side` (Gross PnL per BTC)
- **Side定義**: Long = +1, Short = -1
- $\epsilon$: ゼロ除算防止用定数（`ztb.trading.environment.constants.EPSILON` を使用）。

### 1.3 コスト定義 (単位: JPY/BTC)
$$
Cost_{total} = Fee_{roundtrip} + Slippage_{roundtrip}
$$

1.  **往復手数料 ($Fee_{roundtrip}$)**:
    $$ Fee_{roundtrip} = (EntryPrice + ExitPrice) \times FeeRate $$
    ※ ExitPriceが不明なエントリー時点では `EntryPrice * 2 * FeeRate` で近似する。

2.  **往復スリッページ ($Slippage_{roundtrip}$)**:
    $$ Slippage_{roundtrip} = Slippage_{one\_way} \times 2 $$
    $$ Slippage_{one\_way} = Spread_{proxy} + Impact + Vol_{risk} $$
    ※ 各項目の定義は後述。

### 1.4 エントリー判定: 保守的EV (Conservative EV)
単位を **JPY/BTC** に統一して計算する。

$$
EV_{conservative} = p_{win\_LCB} \times AvgWin_{gross} - (1 - p_{win\_LCB}) \times AvgLoss_{gross} - Cost_{total}
$$

**階層フォールバックと混合ルール:**
信頼度係数 $\lambda$ を用いて、Level 1 と Level 2 (または 3) のEVをブレンドする。

$$
\lambda = \min(1.0, \frac{n_{eff}}{n_{min}})
$$
$$
EV_{final} = \lambda \cdot EV_{L1} + (1 - \lambda) \cdot EV_{fallback}
$$
- **$n_{min}$**: 信頼に足る最小サンプル数（推奨値: 30）。設定ファイル (`config/hft_config.yaml` 等) で管理する。

**エントリー条件:**
1.  $EV_{final} > 0$

---

## 2. レジーム分類の安定化 (Regime Normalization)

### 2.1 正規化ロジック (オンライン計算前提)
未来の情報をリークさせないため、必ず過去のウィンドウのみを使用する。

1.  **ボラティリティ正規化**:
    $$
    Vol_{norm} = \frac{Vol_{raw}}{\max(\text{RollingMedian}(Vol_{raw}, w_{long}), \epsilon_{vol})}
    $$
    - `RollingMedian`: 直近 $w_{long}$ 期間の中央値。
    - $\epsilon_{vol}$: ゼロ除算防止用定数（`EPSILON`）。

2.  **トレンド正規化**:
    $$
    Trend_{norm} = \frac{Trend_{raw}}{\max(Vol_{raw}, \epsilon_{vol})}
    $$

### 2.2 ヒステリシス制御
- **Enter High Regime**: $Vol_{norm} > Threshold_{high} + \delta$
- **Exit High Regime**: $Vol_{norm} < Threshold_{high} - \delta$
- **最小継続期間**: $k$ ステップ

---

## 3. 擬似HFT実行モデル (Pseudo-HFT Execution Model)

### 3.1 スリッページ推定式 (単位: JPY/BTC, 片道)
Taker（成行）注文前提。以下は**片道分 ($Slippage_{one\_way}$)** の計算式である。

$$
Slippage_{one\_way} = Spread_{proxy} + Vol_{risk} + Impact
$$

1.  **スプレッド代理変数**:
    $$ Spread_{proxy} = c_{spread} \times (High - Low) $$

2.  **ボラティリティ/遅延リスク**:
    $$ Vol_{risk} = c_{vol} \times ATR_{1m} \times \sqrt{\frac{Latency_{sec}}{60}} $$

3.  **マーケットインパクト**:
    $$ Impact = c_{imp} \times ATR_{1m} \times \left( \frac{OrderSize_{BTC}}{\max(Volume_{BTC}, MinVolume)} \right)^\gamma $$
    - **ガード**: $Volume_{BTC}$ が極小または0の場合に備え、$MinVolume$ (例: 0.01 BTC) で除算を保護する。

**パラメータ初期値目安**:
- $c_{spread} = 0.2 \sim 0.4$
- $c_{vol} = 0.1 \sim 0.3$
- $c_{imp} = 0.5$
- $\gamma = 0.5$

### 3.2 Maker（指値）の扱い
- バックテストおよび学習時の報酬計算は**Taker価格**を基準とする。
- Maker約定はボーナス扱いとし、戦略の生存判定には使用しない。

---

## 4. 実装ガイドライン

### 4.1 コード品質と安全性
- **型安全**: `mypy` による型チェックを通過させること。`Any` 型の使用は極力避ける。
- **定数管理**: マジックナンバーを避け、設定ファイル (`config/`) または定数クラスで管理する。
    - ゼロ除算防止には `ztb.trading.environment.constants.EPSILON` を使用する。
- **DRY原則**: 既存の `RiskManager`, `ExecutionModel` 等のクラス・メソッドを再利用し、ロジックの重複を防ぐ。

### 4.2 実装ロードマップ
1.  **データ診断**: `btc_jpy_1m_dataset.csv` を用いて、既存モデルのアクションと次期リターンの相関を分析。
2.  **校正マップクラス実装**:
    - 階層型データ構造 (`Regime` > `Bin`)。
    - EWMAによる $n_{eff}, AvgWin, AvgLoss$ 更新ロジック。
    - フォールバック混合計算。
3.  **実行モデル改修**: `RealisticExecutionModel` に修正版スリッページ式を実装。

