# v452 バックテスト結果に基づく改善戦略と深掘り分析

## 1. 現状分析 (v452 Optimized)

### 成績概要
*   **総合損益**: +876,868 JPY (非常に優秀)
*   **得意な市場**:
    *   `extreme_volatility` (極端なボラティリティ): +401,832 JPY
    *   `high_volatility_ranging` (高ボラティリティレンジ): +137,680 JPY
    *   `low_volatility_ranging` (低ボラティリティレンジ): +90,161 JPY
*   **苦手な市場**:
    *   `strong_bull_trend` (強い上昇トレンド): -24,177 JPY
    *   `moderate_bull_trend` (緩やかな上昇トレンド): -7,741 JPY
*   **特徴**:
    *   本モデルは「ボラティリティ・ハーベスター（変動収穫型）」としての性質が強い。
    *   レンジ相場や乱高下相場での平均回帰（Mean Reversion）や逆張り戦略が非常にうまく機能している。
    *   一方で、上昇トレンドに対して「上がりすぎ」と判断してショート（売り）を仕掛け、踏み上げられている可能性が高い（逆張りバイアス）。
    *   下降トレンド（Bear Trend）では利益が出ているため、下落に対する順張り、あるいは反発狙いは機能している。
    *   **HOLD率の高さ (83.87%)**: 非常に慎重なモデルであり、機会損失が発生している可能性がある。特に高ボラティリティ相場でのHOLDはもったいない。

---

## 2. 弱点克服：上昇トレンドでの損失対策

上昇トレンドでの損失は、機会損失ではなく実損失（マイナス）であるため、最優先で対処すべき課題です。

### A. 非対称閾値（Asymmetric Thresholds）の導入
**課題**: 現在の `trend_oppose`（トレンド逆張り抑制閾値）は、上昇・下降の両方に一律で適用されています。モデルは下降トレンドには適応できていますが、上昇トレンドで逆張りをしすぎています。
**対策**:
*   `trend_oppose` を `bull_trend_oppose` と `bear_trend_oppose` に分割します。
*   `bull_trend_oppose` を現在の `5.0` からさらに引き上げ（例: `10.0` や `20.0`）、強い上昇トレンド中の「売り」をほぼ禁止に近い形で抑制します。
*   これにより、上昇トレンド中の不用意なショートを防ぎます。

### B. トレンドフォロー・ブースト（Trend Following Boost）
**課題**: 上昇トレンドにおいて、押し目買い（Buy Dip）やブレイクアウト買いが不足しています。
**対策**:
*   レジームが `strong_bull_trend` の場合のみ、`trend_follow`（順張り閾値）を動的に引き下げます（例: `0.5` -> `0.2`）。
*   これにより、わずかな上昇シグナルでも敏感に「買い」エントリーを行うようになり、トレンドに乗り遅れることを防ぎます。

### C. トレーリングストップのレジーム別適用
**課題**: 上昇トレンド中に早すぎる利確を行っている可能性があります。
**対策**:
*   `strong_bull_trend` 判定時は、固定の利確目標（Take Profit）を無効化し、代わりにトレーリングストップのみを有効にします。
*   トレンドが続く限りポジションを保有し続け、利益を最大化します。

---

## 3. 強み強化：ボラティリティ相場での収益最大化

既に利益が出ているボラティリティ相場・レンジ相場において、さらに利益を伸ばす方策です。

### A. レジーム別ポジションサイジング（Dynamic Position Sizing）
**課題**: 勝率と期待値が高い `extreme_volatility` 相場でも、他の相場と同じポジションサイズで戦っています。
**対策**:
*   ケリー基準（Kelly Criterion）の考え方を応用し、期待値の高いレジームではリスク許容度を引き上げます。
*   `extreme_volatility` や `high_volatility_ranging` が検出された場合、基本ロットサイズを 1.2倍 〜 1.5倍 に動的に増加させます。
*   ※ただし、ドローダウンリスクも高まるため、ストップロス設定の厳格化とセットで行います。

### B. リエントリーの高速化（Aggressive Re-entry）
**課題**: 利確後に次のエントリーまで待ちすぎて、機会を逃している可能性があります（特に乱高下時）。
**対策**:
*   高ボラティリティ時は、ポジションクローズ後の「クールダウン期間（Wait time）」を短縮または撤廃します。
*   ドテン（途切れない売買）に近い挙動を許容し、往復ビンタを恐れずに回数をこなす設定にします。

---

## 4. 高頻度取引（HFT）の促進とHOLD率の低減

HOLD率 83.87% は、特に高ボラティリティ相場においては「機会損失」です。これを改善し、よりアクティブに利益を積み上げる戦略です。

### A. 「偽のチョップ（False Chop）」判定の修正
**課題**: 現在の実装では、`HIGH_VOLATILITY` や `EXTREME_VOLATILITY` が「レンジ相場（Ranging）」として扱われ、`range_chop`（閾値倍率 10.0）が適用されている可能性が高いです。
*   つまり、**「相場が荒れているから、手を出さずに静観しよう（HOLD）」** というロジックが働いています。
*   しかし、バックテスト結果は「荒れている時こそ稼げる」ことを示しています。

**対策**:
*   `ThresholdManager` において、`EXTREME_VOLATILITY` と `HIGH_VOLATILITY_RANGING` を `ranging_regimes` から除外、あるいは特別扱いします。
*   これらのレジームには、新しいパラメータ `volatility_scalp`（例: 0.5）を適用し、閾値を**下げる**ことで積極的なスキャルピングを促します。

### B. アクション空間の再解釈（Action Shaping）
**課題**: モデルの出力（連続値）が 0 付近に集中しやすいため、固定の閾値では HOLD になりやすい。
**対策**:
*   高ボラティリティ時のみ、モデルの出力に対する感度を上げます（例: 出力値を 1.5倍する）。
*   これにより、わずかな売買意欲でも閾値を超えやすくなり、トレード回数が増加します。

---

## 5. 実装アーキテクチャとコード構造案

### A. `ThresholdManager` の改修 (`ztb/trading/environment/components/threshold_manager.py`)

現在の `_apply_regime_adjustment` メソッドを拡張し、より詳細なレジーム対応を行います。

```python
# 変更案のイメージ
def _apply_regime_adjustment(self, threshold, regime, base_threshold):
    # ... (既存コード) ...
    
    # 1. ボラティリティ・スキャルピング (HFT促進)
    if regime in ["EXTREME_VOLATILITY", "HIGH_VOLATILITY_RANGING"]:
        # 既存の range_chop (10.0) ではなく、専用の低い倍率を適用
        mult = self.regime_multipliers.get("volatility_scalp", 0.5)
        adjusted_threshold *= mult
        
    # 2. 非対称トレンド抑制 (上昇トレンドでの売り防止)
    elif "BULL" in regime:
        if base_threshold < 0: # SELL Threshold
            # 上昇トレンド中の売りは強く抑制 (例: 20.0倍)
            mult = self.regime_multipliers.get("bull_trend_oppose", 20.0)
            adjusted_threshold *= mult
        else: # BUY Threshold
            # 上昇トレンド中の買いは促進 (例: 0.5倍)
            mult = self.regime_multipliers.get("trend_follow", 0.5)
            adjusted_threshold *= mult
            
    # ... (Bear Trend は既存通り or bear_trend_oppose を使用) ...
```

### B. 設定ファイル (`config/threshold_optimized.json`) の拡張

新しいパラメータを追加します。

```json
{
    "trend_follow": 0.5,
    "trend_oppose": 5.0,      // 後方互換用
    "bull_trend_oppose": 20.0, // 新設: 上昇トレンドでの売り抑制
    "bear_trend_oppose": 5.0,  // 新設: 下降トレンドでの買い抑制
    "range_chop": 10.0,
    "range_scalp": 0.5,
    "volatility_scalp": 0.4    // 新設: 高ボラティリティでのHFT促進
}
```

---

## 6. 具体的な実装ロードマップ

### Phase 1: パラメータロジックの改修（即効性あり）
1.  `ThresholdManager` を改修し、`bull_trend_oppose` / `bear_trend_oppose` / `volatility_scalp` を分離・実装。
2.  `config/threshold_optimized.json` に新しいキーを追加。
3.  バックテストで検証（特にHOLD率の低下と、上昇トレンドでの損失減少を確認）。

### Phase 2: ポジション管理の高度化
1.  `PositionManager` にレジーム情報を渡せるようにする。
2.  レジームに応じたロットサイズ補正係数を実装。
3.  リスク管理（ドローダウン）とのバランスを検証。

### Phase 3: アンサンブル学習（長期的）
1.  学習データをレジーム別に分割。
2.  特化型モデルの学習。
3.  推論時のモデル切り替えロジックの実装。
