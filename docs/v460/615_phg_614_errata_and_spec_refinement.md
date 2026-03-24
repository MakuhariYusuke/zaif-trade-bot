# 615# 614# 補修と実装精度への仕様引き上げ

- **日付**: 2026-03-24
- **目的**: 614# における脱落項目（composite_risk）の補完、および実装担当者が迷いなくコードを記述できるレベルまで各仕様の精度を引き上げる。
- **前提**: 実装担当によるコード裏取り結果（615# 指示プロンプト）に基づく修正。

---

## §1 614# 正誤表の補完 (T1)

614# §1 の正誤対照表に、脱落していた `composite_risk_enabled` を追加する。

| 項目 | 608# での扱い | 実値 (検証済み) | 出典 | 影響 |
|:---|:---|:---|:---|:---|
| **composite_risk** | 未言及 (false 前提) | **true** | `fill_test.yaml:1179` | リスク層の寄与度分析（Skip 理由の分類）において、複合リスクによる遮断が考慮から漏れる。 |

---

## §2 stage_saturation 検出仕様 (T2)

各ステージの 2.0 (Max Mult) キャップ到達を判定するための、`analyze_fill_logs.py` 拡張におけるロジック仕様。

### 2.1 判定ロジック
明示的なフラグがないため、`executor_offset_stages` JSON 内の数値に基づき以下の閾値判定を行う。

1.  **乗法パイプライン**:
    - 各項の multiplier $m_i \ge 1.99$ の場合を **Stage Saturation** とみなす。
2.  **加法パイプライン**:
    - RMS 合成後のバッファ値 $B_{tox}$ または $B_{liq}$ が、論理的な上限（加法化される各項の合計キャップ相当）に対して 99% 以上に達している場合を Saturation とみなす。
    - ただし、加法移行の主目的は「飽和の回避」であるため、Phase 1 では乗法側での $m_i \ge 1.99$ 検出を優先する。

### 2.2 実装方針
- **フラグ追加 vs 推定**: 現時点では JSON スキーマ変更を伴う新フラグ追加は行わず、**既存値からの閾値推定**を採用する。これは過去のログに対しても遡及的に分析可能にするためである。
- **集計式 (疑似コード)**:
  ```python
  saturation_counts = {k: sum(1 for v in stage_values if v >= 1.99) for k in stage_keys}
  saturation_rate = saturation_counts[k] / total_cycles
  ```

---

## §3 σ-unit 正規化の数理仕様（修正版） (T3)

614# §3 における $\sigma$ 取得パスおよび算出定義の正確な仕様。

### 3.1 変数の取得パス
- **$\sigma_{current}$**: `maker_price.py` の `_robust_sigma` 属性から取得する。これは `RobustStats.asymmetric_ema()` によって毎サイクル更新される平滑化ボラティリティである。
- **$\sigma_{baseline}$**: `ztb/trading/signal/regime/regime_detector.py` の `baseline_vol` プロパティを再利用する。これは全履歴 Returns から算出された長期的な基準値である。

### 3.2 修正版スケーリング公式
各ステージのオフセット寄与度 $\Delta R_i$ を以下のように確定する：

$$\Delta R_i = f_i(\text{signal}_i) \times \frac{\text{maker\_price.\_robust\_sigma}}{\text{regime\_detector.baseline\_vol}}$$

- **再利用の理由**: `regime_detector` 側に既に安定した全期間ボラティリティ推定が存在するため、新設による二重管理と計算コスト増を回避する。

---

## §4 composite_risk が Attribution に与える影響 (T4)

### 4.1 リスク層の寄与度分析への影響
`composite_risk_enabled: true` は、個別の Skip Gate (Gate 1, 2, 3, 6, 7) が単独で閾値を超えなくても、それらのリスクの「総和」が閾値（デフォルト 1.5）を超えた場合にサイクルを遮断する。
- **分析の重要性**: 「どのゲートが直接的な原因か」だけでなく、「どのゲートの組み合わせが複合的に遮断を引き起こしたか」を分析対象に含める必要がある。

### 4.2 Attribution Analyzer (Phase 1) への追加考慮
`clamp_rate` および `information_loss` は「発注価格決定プロセス」の指標だが、`composite_risk` は「発注の可否（Gate）」の指標である。
- **追加項目**: Skip 理由の集計において `composite_risk_exceeded` を最重要項目の一つとして分類し、その内訳（どのゲートのウェイト寄与が大きかったか）を `composite_risk_details` 文字列から抽出して可視化する仕様を Phase 1 に追加する。

---

*以上。本文書は 614# を補完・修正するものであり、実装はこの高精度化された仕様に基づいて実行されるべきである。*
