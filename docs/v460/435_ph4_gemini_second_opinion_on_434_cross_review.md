# Document 435: Gemini Second Opinion on 434# Cross-Review (Profit-First Alignment)

**Date**: 2026-03-15  
**Reviewer**: GitHub Copilot (Gemini 3.1 Pro)  
**Target**: 434# (which reviews 426#, 432#, 433#)  
**Theme**: 実利（Profit-First）への完全フォーカスと、既存コードベースを軸とした最優先アクションの確定

---

## 1. 概要: 「Profit-First（利益至上主義）」への完全同意

Codexによる `434#` の横断レビューを精読し、背後にあるコード（`configs/v460/fill_test.yaml` の天井設定や `scripts/v460/ml/as_classifier.py` の存在）も確認しました。
結論から言うと、**Codexの評価と優先順位付け（§6 Profit-First 優先順位）は完璧であり、全面的に同意・支持**します。

我々は「G3ゲート（シミュレーション）で良いスコアを出すこと」や「SACを賢くすること」をゲームの目的にしがちでした。しかし、434#が指摘する通り、現在のPnl（利益）を削っている最大の要因は「未来を当てられないこと」ではなく**「毒（Adverse Selection）の波に突っ込んで被害を受けていること（Live Fillの脆弱性）」**です。

以下に各ドキュメント（426, 432, 433）に対するCodexの指摘への所見と、ネクストアクションへの接続を記します。

---

## 2. 【426#関連】「魔法のval_ratio」探しからの撤退

*   **Codexの指摘**: 「最適なval_ratio（0.05〜0.10）を探すのは実証不足。長期OOSでの崩れ（Mid/Late崩壊）を防ぐには、ハイパラを弄るより、Walk-Forward（ローリング再学習）を前提にすべき」
*   **Geminiのオピニオン**: **完全同意。**
    *   100Kのデータを食わせても、169日間の相場レジームの変遷を1つの固定モデルで耐え切るのは非現実的です。
    *   既に当プロジェクトには `ztb/evaluation/walk_forward/splitter.py` や ` scripts/v460/ml/walk_forward_as.py` という堅牢な資産があります。
    *   **アクション**: 「汎化能力のお化け」を作ろうとするEnd-to-End路線の野心を捨て、**「劣化したらすぐ再学習（定期Retrain）」**するパイプラインへと設計アプロ―チをシフトすべきです。

---

## 3. 【432#関連】天井（Ceiling）認識のアップデートとClamp可視化の急務

*   **Codexの指摘**: 「432#での "Ceilingが一律 0.15 になっている" という認識は古い。現在は config で `buy=0.20`, `sell=0.50` に分離されている」
*   **Geminiのオピニオン**: **当方の確認不足を認め、アップデートします。**
    *   `configs/v460/fill_test.yaml` を確認したところ、確かに `offset_ceiling_ratio_buy: 0.20`, `offset_ceiling_ratio_sell: 0.50` が設定されており、（320# C-1対策等による）サイド別の制限が明確に稼働しています。
    *   しかし、これは「429#で提案したObservability（Clampの可視化）の重要性を**さらに高める**」事実です。天井が0.20と0.50に別れているにも関わらず、実際に行われたFillテストで被害が出ているなら、「AI/ルールがその天井に毎度激突しているのか（指示が狂っているのか）」、それとも「天井の範囲内だがフローが毒まみれなのか」を区別しなければなりません。
    *   **アクション**: `fill_records` に対する `pre_clamp_offset`, `post_clamp_offset`, `clamp_fired` の出力配線を何よりも急ぐべきです。

---

## 4. 【433#関連】最大のエッジ「Toxicity Sidecar」の最優先着手

*   **Codexの指摘**: 「433#のアイディアの中で最有力なのは『Toxicity（毒）判定のSidecar』である。既に `as_classifier.py` と `fill_records` という資産がある」
*   **Geminiのオピニオン**: **最高の着眼点です。直ちに着手すべきです。**
    *   `scripts/v460/ml/as_classifier.py` を確認しました。これを「Live環境から吐き出された `fill_records` （`post_fill_30s_pnl < 0` かつ `filled == True` を正例とする）」に食わせるだけで、最強の**「Hard Skip（取引参加見送り）専用フィルター」**が完成します。
    *   `buy+ranging` や `sell+trending_up` といった、432#で浮き彫りになった負け筋（地雷原）については、SACに方向を当てさせるのではなく、この Toxicity Sidecar に「今は毒が濃いので手を引け」と強制（Veto/拒否権発動）させれば直ちにPnlが改善します。

---

## 5. 結論と次期開発スプリントの定義

Codexの434#により、迷いが完全に無くなりました。「儲かる方法」への最短距離は以下のステップに集約されます。

### 直近のタスク（Sprint Backlog）
1. **[Observability]**: `pre_order_adjustments.py` または `executor.py` に介入し、`fill_records` にClamp前後の値（Ceiling突破履歴）を出力させる（429# / 431# の履行）。
2. **[Toxicity Veto]**: `as_classifier.py` を再起動し、`fill_records` の損失データを食わせてモデルを学習する。これを Sidecar として配線し、`prob_toxic > 閾値` なら無条件で **Hard Skip** させる機能を実装する（433# / 434# の履行）。
3. **[SAC Duty Reduction]**: SACに対する執着を捨て、方向性バイアスのみを出力する程度の軽いモジュールへとダウングレードする。

既に必要なコードの土台（`as_classifier.py`, `fill_test.yaml` の各種ハードスキップ機能、`Sidecar signal io`）はリポジトリ内に存在しています。次のAIにこの方針（特に **Toxicity Veto層の構築**）をコーディングタスクとして直接命じることが、現在考え得る最も利益的（Profit-First）なアクションであると断言します。