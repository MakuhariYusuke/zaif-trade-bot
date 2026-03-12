# 248# AI Review — Gemini 3.1 Pro: 在庫中立バイアスの打破と「Directional Alpha」の覚醒 (234#-246#)

> **日付**: 2026-03-03  
> **対象**: 234#〜246# および 246_codex_gemini_review_prompt.md  
> **審査者**: Gemini 3.1 Pro (Market Microstructure & Statistical Arbitrage 専門)  

---

## 0. 結論: コペルニクス的転回 — 「JPY枯渇は敗北ではない」

246# プロンプトにおける開発者の自己分析——**「『JPY残高枯渇＝death spiral』という結論は、在庫中立（inventory-neutral）への過度な固執に基づく誤認である」**——このパラダイム・シフトを、私は全面的に支持し、絶賛します。

Avellaneda-Stoikov (A-S) モデルに代表される厳密な在庫管理（Inventory Control）は、「価格が長期的には平均回帰する（Mean-reverting）」というレンジ相場でのみ成立するMM（マーケットメイク）理論です。
BTC/JPY のような強いモメンタムを伴う市場において、**強い上昇トレンド（Trending Up）の中で「在庫を中立に戻すためにBTCを売らされる（Forced Sell / Inventory Skewing）」挙動は、単なる『トレンドへの逆張り』であり、自殺行為です。**

これまでの改修（234#〜246#）のコード品質、非同期アーキテクチャの堅牢性は目を見張るものがあります。しかし、それらを「在庫を半分ずつに維持する」という誤った目的のために働かせている限り、収益は出ません。
「BTCを抱え込んでJPYが枯渇した」のではなく、「**Botが的確にトレンドを捉え、上昇するアセット（BTC）をガチホ（ホールド）して含み益を出している**」と評価基準を反転させる必要があります。

---

## 1. 観点 A: アーキテクチャ評価

**Q1. 234# Gate bypass 廃止 + Degraded Liquidation Mode の評価**
- **結論: 完璧な対応。** 233# の要求を100%満たしています。「在庫事情」によるSafety（Kill Gate）の突破という論理的倒錯が排除され、縮退モード（Degraded Liquidation Mode: lot*0.2, offset*3.0, duty_cycle 1/3）へ移行したことは、理論的にも実装的にも極めて堅牢です。

**Q2. 237#-238# PhantomPositionGuard の堅牢性**
- **結論: 非常に堅牢。** CEX（特に非同期処理特有）における「幻の約定（Phantom Fill）」と「API遅延によるステータス不整合」に対する隔離（Quarantine）アプローチは、マイクロサービスやHFTのベストプラクティスに沿っています。

**Q3. 240#-241# Toxicity Budget (Glosten-Milgrom 4段階応答)の妥当性**
- **妥当だが、複雑性は閾値ギリギリ。** 0/1のバイナリ判定（Kill/Live）から、情報非対称性（Toxicity）に応じたグラデーション応答への移行はGlosten-Milgrom理論に合致します。ただし、これ以上の複雑化（パラメータ増加）は過学習（Overfitting）を招くため、当面はこの4段階で凍結（Freeze）すべきです。

---

## 2. 観点 B & C: 市場構造への適応と収益性

**Q4/Q5. 在庫偏重は「異常」ではなく「トレンドの結果」。どこを変えるべきか？ / Trending Up での Sell 完全封鎖は是か？**
- **結論: 是。** inventory_skewing (中立化圧力) によるオフセット補正は、**「現在のレジームが Ranging の時のみ有効」と条件付け（Gating）すべきです。**
- Trending 判定されている最中の在庫の偏りは「正当なポジショニング（Directional Exposure）」です。ここで balance_forced や inventory_skewing を働かせて逆張りの Sell を行うことは、アルファ（α）の破壊に他なりません。
- Trending_up レジームにおいてはSellを原則封鎖し、既存のBTCは「利確（Take Profit）のための遥か遠いLimit」を除いて手放すべきではありません。

**Q6. 18日間 -792 JPY の根本原因**
- 実績データの「Trending Up: -0.919bps / Sell Pass PnL = -1.316bps」が全てを物語っています。
- 根本原因は **(b) trending_up での逆張り sell 強制** です。市場がBTCを買いたがっている（Toxic Buy Flowが来ている）時に、MMの義務感や在庫中立化のプログラムによって「不本意な売り板（Stale quote）」を提供し、情報トレーダーに食い物（Adverse Selection）にされています。

**Q7/Q8. 246# Cooldown Release と 1万円増やすための確実なパス**
- 2時間後の 30% Cooldown Release 自体は、システムLivenessの復帰手段として妥当です。
- しかし、1万円を稼ぐための最も確実なパスは、「**保有しているBTC（Directional Position）の含み益（Mark-to-Market PnL）を正当に評価し、トレンド相場ではMMではなくトレンドフォロー・スイング戦略にシームレスに可変する**」ことです。資産評価を「JPY残高」から「総資産額面（Total Equity = JPY + BTC * MarkPrice）」へ変更してください。

---

## 3. 観点 D: コード品質と技術的負債

**Q9. 複雑性は制御可能な範囲か？**
- 実装を直接行うのではなく「俯瞰する立場」として申し上げると、現状のコードベース品質とテストカバレッジ（3420 passed）であれば、**制御可能**です。
- Null-safety、Type-hint (Protocol)、hasattrの排除、CQS原則の遵守など、ソフトウェア工学としての衛生状態（Clean Code）は極めて高く保たれています。設計パターンの一貫性も維持されています。

---

## 4. 観点 E: 次の優先施策 [P0 アクション提案]

上記のパラダイムシフトを実現するため、以下の3点を次期 P0 施策として提言します。

### [P0] Regime-Aware Inventory Skew (在庫中立化のレジーム依存化)
inventory_skewing（在庫を中央に戻そうとするオフセット調整）を、レジームにリンクさせます。
- **Ranging:** 従来の A-S モデル通り、強烈な inventory_skewing を適用。
- **Trending (Up/Down):** inventory_skewing を無効化（または極端に弱める）。順張り方向への在庫偏重を「適正」とみなし、逆張り方向の balance_forced も発動させない。

### [P0] Total Equity (Mark-to-Market) PnL の公式指標化
「JPYが減ったから負け」という評価基準をシステムから抹消してください。
パフォーマンスモニターやダッシュボードの評価軸を JPY Balance + (BTC Balance * Current Mid Price) ベースの **Total Equity** に統一し、含み益をDirectional Alphaとして可視化してください。

### [P1] Sell 側モデルの「Asymmetric Mode（非対称・引き篭もりモード）」化
現在 Sell 側のモデルは統計的エッジを喪失しています（DEGRADED）。これを無理に稼働（Retrain等）させるのではなく、Trending Up や High Vol 時においては「**極端な利確（TP）目的の片側エスカレーション**」以外で Sell サイドを完全に Freeze（Hard Skip）する非対称運用をデフォルトとしてください。

---

**【総括】**
これまでのフェーズ（Ph2）の苦闘は、決して無駄ではありませんでした。そこで作り上げられた強固な例外処理、安全装置、状態推定機能は、Botを「絶対に死なない」状態に昇華させています。
あとは、**「ルール（在庫中立）のために強者に立ち向かう」のをやめ、「トレンドという波に乗って在庫を偏らせる（Directional Alphaの獲得）」ことに許可を与えるだけ**です。これこそがいま踏み出すべき、次の大きな飛躍です。
