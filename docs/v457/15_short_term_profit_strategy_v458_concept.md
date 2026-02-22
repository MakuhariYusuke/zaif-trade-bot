# 15. Short-Term Profit Maximization Strategy (v458 Concept)

**作成日**: 2026-01-18
**目的**: 安定稼働した v457 をベースに、1分足データの制約下で最大限の利益を追求する「高頻度スイングトレード」戦略の確立

## 1. 現状認識と課題

### 現状 (As-Is)
- **v457**: Config注入修正を経て、Over-Trading（1200回/10k steps）は完全に沈静化。
- **Under-Trading**: 代わりに取引頻度が極端に低下（233回/141k steps = 0.16%）。
- **機会損失**: 慎重すぎて、取れるはずだった小〜中規模の波動（Swing）を見送っている可能性が高い。

### 目標 (To-Be)
- **スタイル**: 「1分足 HFT-like Swing」
    - 数分〜数十分のトレンドを細かく切り取る。
    - 1トレードあたりのpipsは小さくても、回転数で積み上げる。
- **データ制約**:
    - 秒足がないため、執行（Entry/Exit）の精度限界は「次の1分足の始値」となる。
    - 指値戦略はリスクが高いため、成行（Market Order）前提でスリッページに勝てるエッジが必要。

## 2. 戦略コンセプト: "Dynamic Aggression"

過去の資料（v450, v437）を参照し、静的な「閾値」ではなく、市場環境に応じた「動的な攻撃性」を定義します。

### A. Dynamic Thresholds (from v450) の復活と改良
現在 `min_delta: 0.05` (固定) が高すぎる可能性があります。これを市場ボラティリティ（ATR）またはモデルの確信度（Z-Score）に連動させます。

- **静的閾値の弊害**: 低ボラ時になにもできず、高ボラ時にノイズで狩られる。
- **Z-Scoreアプローチ**:
    - モデルの出力（生のアクション値）の分布を監視。
    - 「普段より強いシグナル」が出たら、絶対値が小さくてもGOサインを出す。
    - *v457 Factoryにはこのロジックが未実装なため、v458でEnvラッパーとして再実装が必要。*

### B. ATRベースの可変 min_delta
より実装が容易なアプローチとして、現在のボラティリティ（ATR）を用いた閾値調整を行います。

$$ \text{Threshold}_t = \text{BaseThreshold} \times \frac{\text{CurrentATR}}{\text{AvgATR}} $$

- **狙い**: ボラティリティがある時はノイズ幅も広がるため閾値を上げ、凪（なぎ）の時は閾値を下げて微細な動きを拾う（逆の考え方も可：ボラがある時こそ積極的に入る）。
- **推奨設定**: ボラティリティが高い時こそ「大きな動き」が期待できるため、逆に**閾値を下げて乗りやすくする**（順張り特化）設定も検討。

### C. 利益確定の高速化 (Micro-TakeProfit)
「持ちっぱなしペナルティ（Time-Decay）」は機能しましたが、単に「早く逃げる」だけでなく「利益が出たら即座に確定する」動きを報酬で強化します。

- **Realized PnL Reward の強化**:
    - 含み益（Unrealized）よりも実現益（Realized）の報酬ウェイトを上げる。
    - これにより「利食い」行動が促進され、回転率が上がる。

## 3. 具体的な実装計画 (v458)

### Step 1: パラメータチューニング (即時対応)
既存の v457 環境（Config注入修正済み）で、`min_delta` の最適解を探る。
- **グリッドサーチ**: `min_delta` = [0.01, 0.02, 0.03, 0.04, 0.05]
- **予想**: `0.02` 付近に Profit/Trade のスイートスポットがあると推測される。

### Step 2: "Volatility-Adaptive Threshold" の導入
`EnvironmentFactory` または `Environment` 内で、`min_delta` を動的に変更するロジックを追加。

```python
# 疑似コード
current_volatility = self.monitor.get_current_volatility() # e.g. std of returns
adaptive_delta = self.base_min_delta * (1.0 + coefficient * (ref_vol - current_vol))
# coefficient > 0 なら「低ボラ時に閾値が上がる（慎重）」
# coefficient < 0 なら「低ボラ時に閾値が下がる（敏感）」<- 今回はこちらを狙う？
```

### Step 3: "Sniper Mode" 報酬関数
短期間での高収益を狙うため、報酬関数を「一発のホームラン」ではなく「安打製造機」向けに調整。
- **Win Rate Boost**: 勝率そのものにボーナスを与える（小さい勝ちでもPositive）。
- **Drawdown Penalty**: ドローダウンへの罰則を厳しくし、負けを素早く切る動きを学習させる。

## 4. 期待される効果
- 取引回数が **10〜20回/日** (141k stepsで1000〜2000回) 程度に回復。
- 回転数向上により、複利効果（資金効率）が最大化される。
- 「機会損失」という見えないコストを削減。

## 5. 次のアクション
1. **Config更新**: `config/v458/` ディレクトリを作成。
2. **実験**: `min_delta` を **0.02** に下げて再学習・検証（v457改として実施）。
3. **開発**: 動的閾値の実装（コード改修）。

---
**参照資料**:
- `docs/v450/01_dynamic_thresholding.md`
- `docs/v457/13_training_results_success.md`
