# 404# Review: Gemini Second Opinion on 402# & 403# (Time Guard Analysis & Code Deep Dive)

**Date**: 2026-03-13
**Scope**: docs/v460/402_time_guard_root_cause_and_397_review.md, docs/v460/403_ph3_rev_402_time_guard_fill_test_multifaceted_review.md, and scripts/v460/lib/maker_price.py
**Reviewer**: Gemini (Second Opinion Surrogate)

---

## 0. 総評：多角的視点の完全一致と「真の因果」の特定

402# の「時間帯ガードの考古学的整理とAS率の因果特定」は素晴らしい分析です。しかし分析から導かれたアクション（09hの強化、hard_skipの除外）は、**相関関係を因果関係と取り違えた局所最適なもの**でした。
これに対するCodexの403#レビューは**「分析の粗さ（Mixed-SHA問題）」「AS率（未来情報）への過剰適応」「Offset層のボトルネック（0.30 Ceiling）の真の構造」**を冷徹に指摘しており、完璧です。Geminiとしてもこの監査内容を100%支持します。

Claudeが先の391#で引用した易占『雷火豊（明察と決断）』の示唆が、ここでも完全に一致しています。「時間帯（JST 09h）が悪いから手動で休ませる」という対症療法（局所的な豊）は真の解決ではなく、「AS率予測に基づく動的ガードへの統合」という**本質的な明察**なしにはシステムは破綻します。

本ドキュメントでは、ユーザーの「もう少し深堀りしてほしい」という要望に応え、**なぜパラメータチューニングが一切効かないのか（Offset Ceiling デッドロック）**をコードレベルで証明します。

---

## 1. 致命的指摘のクロスバリデーションとコード実証

### 1.1 dverse_selected_raw のLeakageリスク (HIGH)
*   **【表（402#の視点）】**: AS率（Adverse Selection）と時間帯PnLには明確な負の相関がある。だからAS率が高い時間帯をハードコードで狙い撃ち（Sell Boost等）すれば儲かるはずだ。
*   **【裏（403#＋Geminiの視点）】**: dverse_selected_raw は m.mid_30s_after という**未来の価格**を用いて算出される完全なPost-fill（事後）指標です。
    scripts/v460/lib/pnl_measurer.py:
    m.adverse_selected_raw = (m.mid_30s_after < m.mid_at_fill if side == "buy" ...)
    これを直接的な取引の「言い訳」や「時間帯固定設定の根拠」にすることは、**将来情報のLeakage（データ漏洩）**に近い危うさがあります。時間（09h）をProxyにするのではなく、その時間に起こりやすい事象（VPIN、Spread、OBI）から**事前（Pre-trade）にASを予測するモデル**を構築するのが正しい道です。

### 1.2 THE SMOKING GUN: Offset Ceiling 制約の複雑な全体構造 (CRITICAL)
*   **【表（402#の視点）】**: 397#のMid-confidence boost (×1.2) がSellで無効になっている。SellのCeiling（上限）が0.30だからだ。Ceilingを上げれば解決する。
*   **【裏（403#＋Geminiの深堀り視点）】**: 実態は致命的です。単に上限値を上げれば済む問題ではありません。scripts/v460/lib/maker_price.py 内のロジックが**ブースト処理を各ステップ毎に 0.30 で切り捨てるスパゲティ構造**になっています。

**【コード実証：_scale_offset_ratio の乱用】**
maker_price.py では _scale_offset_ratio 関数が以下のように定義されています。
`python
    def _scale_offset_ratio(
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]:
`
この関数は、引数として受け取った max_ratio (設定値 cfg.max_offset_ratio = 0.30) で**即座にキャップ**を掛けます。

問題は、これが**パイプラインの途中の各ステップで毎回呼ばれていること**です。
1. **Spread Adaptive Layer**: 
   effective_offset_ratio, _ = self._scale_offset_ratio(..., max_ratio=cfg.max_offset_ratio)
2. **Time Decay Layer**:
   effective_offset_ratio, _ = self._scale_offset_ratio(..., max_ratio=cfg.max_offset_ratio)
3. **Fast Fill Defense Layer**:
   effective_offset_ratio, _ = self._scale_offset_ratio(..., max_ratio=cfg.max_offset_ratio)

**【結論】**
もし Spread Adaptive の段階で比率が 0.30 に達してしまった場合、その後の Time Decay や Mid-confidence boost で 1.2 倍、2.5 倍の係数を掛けようとしても、**そのステップ内で再度 max_ratio=0.30 が適用されるため一切値が増えません**。
これが、402# で指摘されていた「ブーストが効かない」真の理由です。YAMLの max_offset_ratio を直列で何度も適用するアーキテクチャ上の欠陥（Offset Ceiling デッドロック）が原因です。「Sell Ceilingだけを上げる」という対処では直りません。制約層の統合整理が必須です。

### 1.3 confidence >= 0.9 最悪問題の解釈 (MEDIUM)
*   **【表（402#の視点）】**: 全データで見ると >=0.9 の帯域がPnL -1.690bps で最も悪い。
*   **【裏（403#＋Geminiの視点）】**: これはHFTや強化学習における典型的な **「The Confident-Wrong Paradox（自信満々に間違えるAI）」** です。
    モデルのConfidence（確信度）が相場の真の不確実性とズレている（Calibration崩れ）ことが根本原因です。これを「時間帯」の問題や「Mid-confidence」だけの問題として矮小化してはいけません。モデルはProfitではなくRewardを最適化しているため、構造的な報酬のズレ（以前の10bpsペナルティ設定など）が残存している証左です。

---

## 2. 実現に向けた統合的アクション（Next Actions）

402#と403#の議論、および今回のコード深掘り結果を踏まえ、以下の順序でシステム改修に「決断（雷）」を下すことを推奨します。これを満たさない限りのパラメータの「手回し（Hard_skip外しなど）」は禁止です。

### Action 1. Offset Ceiling / Floor パイプラインの単一化（CRITICAL）
* 現在地雷と化している max_offset_ratio: 0.30 (複数箇所) の適用方法を修正します。
* maker_price.py において、中間の各ステップでは**一切キャップを掛けず純粋に multiplier を乗算**し、**最終段（returnの直前）で一回だけ全体を Clamp** させる設計に変更してください。これにより、各ブースト係数が正しく機能するようになります。

### Action 2. 時間帯固定ガードからの脱却プロセス（DRY原則の適用）
時間帯（時・分）に依存するハードコードされた閾値を廃止する方向に舵を切ります。
* **却下**: hard_skip_utc_hours[21] の解除や、sell_hour_boost[0] の 1.5 → 2.5 への手動強化は、Mixed-SHAデータに引きずられた一時しのぎであるため**却下（保留）**とします。

### Action 3. AS Prediction (逆選択予測) への昇華
* 時間帯という「雑なProxy」を捨てる代わりとして、VPIN, OBI, spread, ol_ratio を入力として、30秒後の dverse_selected_raw (True/False) を予測する軽量なロジスティック回帰器またはGBDT（**事前に評価可能なAS確率スコア**）を skip_gate に統合してください。これが本質的なHFTの防御になります。

### 結論
402#で「7層にも及ぶ時間帯防御」の病理が可視化されたことは偉大な功績です。
深掘りした結果、「時間帯で無理やり止める」設定も、「ブースト」設定も、**コードの構造的デッドロック（_scale_offset_ratio の過剰適用）により無効化されている**ことが証明されました。対症療法を捨て、「最終段での一括クランプ化」および「AS予測と動的リスク管理」という本道へ統合する時が来ました。Codexの403#の指摘を実装指針のベースラインとして採用し、実装整理へ進んでください。
