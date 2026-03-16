# Document 427: Global AI/ML Trading Cultures and "Sidecar" Architecture Plan

## 1. 概要 (Overview)
これまでの議論で浮かび上がった、世界各地域のクオンツ・アルゴリズムトレード開発文化の特徴を総括し、当プロジェクト（zaif-trade-bot）の抱える課題（100K学習時の過学習、オフセットリーク等）を解決するための次期アーキテクチャ「Sidecar（サイドカー）パターン」及び「The Final Clamp」の導入計画を策定します。
Codexのレビュー（セカンドオピニオン）を前提とした、客観的かつ俯瞰的な設計図として機能させます。

---

## 2. 各国のアルゴリズムトレード開発文化 (Global Algorithmic Trading Cultures)

世界のトップティアHFT（高頻度取引）や個人開発者（Botter）は、地域ごとに独自のアプローチで市場に挑んでいます。

### 🇯🇵 Japan: "Botter" Culture (Feature Engineering & Separation)
- **特徴**: LightGBM等の勾配ブースティングを好み、極めて精緻な特徴量（Feature Engineering）を作り込みます。
- **アーキテクチャ**: 「シグナル予測（AI）」と「注文執行（Rule-based）」を完全に分離する手法が主流です。AIに全ての権限を委ねるEnd-to-Endを避け、人間のドメイン知識による安全装置を必ず挟みます。

### 🇺🇸 USA: Deep RL & End-to-End (Silicon Valley Tech)
- **特徴**: PPOやSACといった最先端の深層強化学習（Deep RL）を積極的に採用し、End-to-End（入力から直接行動を出力）の完全自動化を目指す傾向にあります。
- **課題**: 潤沢な計算資源を前提としているため、市場のレジームチェンジ（環境変化）に対して過学習（Overfitting）を起こしやすく、個人や小規模チームではメンテナンスが破綻しやすい側面があります。（今回の100K問題に酷似）

### 🇨🇳 / 🇰🇷 China & South Korea: StatArb, Grid & "Rigid Clamp"
- **特徴**: 統計的裁定取引（StatArb）やグリッドトレード、マーケットメイクに非常に強く、リスクコントロール（損切りや建玉上限）に対する執着が異常なほど高いです。
- **アーキテクチャ**: AIの予測がどうであれ、数学的に損失が確定するラインに到達すれば、問答無用で執行を停止・逆行させる「Rigid Clamp（絶対的な留め具）」を実装する文化があります。

### 🇫🇷 France: Pure Mathematics & MEV
- **特徴**: 伝統的な金融工学と純粋数学に強く、DeFi（分散型金融）におけるMEV（Maximal Extractable Value）抽出や、複雑なアービトラージの経路探索において世界最高峰のアルゴリズムを構築します。

### 🇪🇸 / LatAm (Spain & Latin America): Survival & Robust Grid
- **特徴**: 計算資源や取引所への物理的距離（レイテンシ）で不利な環境にあるため、高い勝率よりも「絶対に退場しない（Survival）」堅牢なグリッドベースのシステムを構築します。

### 🇷🇺 Russia: Bare-metal Low-Latency & Custom Engines
- **特徴**: 物理レイヤーでの戦い（FPGA、C++によるカスタムエンジン、カーネルチューニング）に圧倒的な強さを持ちます。既存のフレームワーク（SB3等）に頼らず、独自の軽量RLエンジンをC++で自作する文化があります。

---

## 3. 当プロジェクトへの適用: 「日中ハイブリッド」アプローチへの転換

我々の初期設計は**「米国型（完全End-to-End SAC）」**でした。結果として、AIが仕様（0.30オフセット上限）を無視して異常な数値を出し（Offset Leak）、評価期間の罠にハマって過学習を起こしました。

今後は、**「日本型の予測・執行分離（Sidecar）」**と**「中国・韓国型の絶対的リスク管理（The Final Clamp）」**を融合したハイブリッド型へ移行します。

### The "Sidecar" Pattern (予測・執行の分離)
SACモデルは直接「売買のオフセット・価格・数量」を決定する**運転手（Driver）**から、市場の危険度や方向性を0〜1でスコアリングする**助手席のナビゲーター（Sidecar）**へ降格させます。
実際の注文パラメータの決定と執行は、静的かつ堅牢なルールベースの**エグゼキュータ（Driver）**が行います。

### The Final Clamp (最終安全装置のハードコード)
`pre_order_adjustments.py` または `executor.py` の最終段階において、AIや上位ロジックが算出した値に対し、如何なる場合でも物理的に超過できない上限・下限（Hard Ceilings/Floors）を適用します。

---

## 4. 今後の実装計画 (Implementation Plan)

### Phase 1: 防御の網羅 ("The Final Clamp" の実装)
- **対象**: `scripts/v460/lib/pre_order_adjustments.py` または関連執行モジュール。
- **内容**: 注文APIを叩く直前のデータ構造に対し、`offset <= MAX_OFFSET (0.3)` などを強制的に適用するクリップ処理を追加。乗数（Multiplier）処理の後段に配置することで、Offset Leakを物理的に遮断する。

### Phase 2: モデルの責任縮小 (SAC -> Sidecar への移行準備)
- **対象**: Actorネットワークの出力定義。
- **内容**: SACが直接オフセットを出力するのではなく、「買いバイアス（-1.0 to 1.0）」「ボラティリティ推定値」などを出力するように変更。エグゼキュータ側でこれを翻訳し、安全なオフセットに変換する設計へのリファクタリングを目指す。

### Phase 3: 堅牢な評価基盤の構築
- **対象**: 評価プロセス（現在S1/S1'設定でバックグラウンド実行中）。
- **内容**: `5_000` steps のみで best_model を決定する問題を解消し、より長期のOut-of-Sample（OOS）スライスを用いた複数環境評価を行う。環境のレジームチェンジに耐えうるかをG3ゲートの必須要件とする。

---
*Prepared by GitHub Copilot (Gemini 3.1 Pro)*
*※ Codexによる多角的なレビュー（セカンドオピニオン）及び批判的検証を歓迎します。*
