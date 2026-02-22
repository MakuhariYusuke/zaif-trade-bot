# v452 開発に向けた提案書 (Proposals for v452)

## 1. はじめに
v451において「パーマベアの解消」と「レンジ相場での損失回避」を達成し、黒字化に成功したことは大きなマイルストーンです。
v452では、ユーザー様の目標である「スキャルピング並みの高頻度取引」と「さらなる収益性」を追求するため、守りの姿勢（レンジでの取引停止）から、攻めの姿勢（相場環境に応じた戦略の切り替え）への転換を提案します。

## 2. クリティカルな修正 (Immediate Fixes)

### ThresholdManager のバリデーション厳格化
`TODO.md` にある通り、勝手なクランプ（値の強制補正）は設定ミスを隠蔽し、デバッグを困難にします。
`ztb/trading/environment/components/threshold_manager.py` の `_validate_config` を以下のように修正することを推奨します。

```python
def _validate_config(self) -> None:
    # ... (前略) ...
    
    # 修正案: クランプではなく例外をスローする
    if (
        self.base_threshold < self.min_threshold
        or self.base_threshold > self.max_threshold
    ):
        # 以前のコード:
        # logger.warning(...)
        # self.base_threshold = np.clip(...)
        
        # 新しいコード:
        raise ValueError(
            f"Configuration Error: base_threshold ({self.base_threshold}) is outside "
            f"allowed range [{self.min_threshold}, {self.max_threshold}]. "
            "Please update config.py or environment settings."
        )
    # ... (後略) ...
```

## 3. 戦略的改善提案 (Strategic Improvements)

### A. 「レンジ相場＝取引停止」からの脱却 (Range Scalping)
現在のロジックでは、レンジ相場（Ranging）と判定されると閾値を10倍にして取引を抑制しています。これは「トレンドフォロー戦略」をレンジで動かさないためには正しいですが、「スキャルピング」の機会を捨てています。

**提案:** **マルチストラテジー化 (Regime-Adaptive Strategy)**
相場環境に応じて、適用する「戦略（または報酬系）」を切り替えるアプローチです。

*   **Trending (Bull/Bear):** 
    *   戦略: 順張り (Trend Following)
    *   閾値: 通常 (1.0x) または 緩和 (0.5x)
    *   アクション: ブレイクアウトや押し目買いを狙う。
*   **Ranging (Sideways):**
    *   **戦略: 逆張りスキャルピング (Mean Reversion)**
    *   閾値: **緩和 (0.5x - 0.8x)** ※現在は10倍でブロックしている箇所
    *   ロジック: ボリンジャーバンド2σやRSIの買われすぎ/売られすぎでの反転を狙う。
    *   *注意:* これを実現するには、モデルが「トレンドの強さ」だけでなく「反転の可能性」も学習している必要があります。

### B. Market Regime Detector の高度化
現在の `MarketRegimeDetector` は、`trend_threshold = 0.02` などの固定値に依存している部分があります。相場のボラティリティは時期によって大きく異なるため、固定値は機能しなくなるリスクがあります。

**提案:**
*   **完全相対評価への移行:** 過去N期間（例: 1000期間）の分布に基づく「パーセンタイル」で判定するロジック (`use_relative=True`) をデフォルト化し、その精度を向上させる。
*   **Hurst Exponent の導入:** 時系列が「トレンドしやすい」か「平均回帰しやすい」かを測るハースト指数を導入し、レンジ判定の精度を高める。

### C. ActionSignalGuide のレジーム連動 (Signal Weighting)
`ztb/trading/strategies/action_signal_guide/recognizer_factory.py` には `BollingerBandsRecognizer` や `RSIPatternRecognizer` など、レンジ相場で有効なシグナル生成器が既に存在します。
これらを有効活用するため、`ActionSignalGuide` が `MarketRegimeDetector` の結果を受け取り、シグナルの重み付けを動的に変更するロジックを追加します。

*   **Ranging Regime:**
    *   Trend Signals (MACD, Moving Average) -> Weight: 0.2 (Low)
    *   Mean Reversion Signals (RSI, Bollinger, Stochastic) -> Weight: 1.5 (High)
*   **Trending Regime:**
    *   Trend Signals -> Weight: 1.5 (High)
    *   Mean Reversion Signals -> Weight: 0.5 (Low)

これにより、レンジ相場でも「確信度の高い逆張りシグナル」が生成され、緩和された閾値を通過してトレードが実行されるようになります。

## 4. アーキテクチャ改善 (Architecture)

### A. 動的閾値パラメータの自動調整 (Auto-Tuning)
現在の手動での「10倍」設定は経験則に過ぎません。
過去のバックテストデータを用いて、各レジームにおける「最適な閾値倍率」を探索するスクリプトを作成し、定期的にパラメータを更新する仕組みを導入すべきです。

### B. 転移学習 (Fine-tuning) の具体化
「直近の相場」に特化させるため、以下のパイプラインを構築します。
1.  **Base Model:** 長期間のデータで学習した汎用モデル (v451)。
2.  **Fine-tuning:** 直近2週間〜1ヶ月のデータのみを用いて、Base Modelを数エポックだけ追加学習。
3.  **Evaluation:** 直近3日のデータで検証し、Base Modelよりスコアが良ければ採用。

## 5. 結論
v452では、単に閾値を調整するだけでなく、**「レンジ相場を敵ではなく利益の源泉とする」** ようなロジックの組み込みに挑戦すべきです。これにより、トレンドがない時期でも収益を上げられる、真の「高頻度取引ボット」に近づくことができます。
