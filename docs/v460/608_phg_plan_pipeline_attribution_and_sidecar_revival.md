# 608# 執行パイプラインの寄与度分析 (Attribution) と Sidecar (SAC) の再活性化計画

- **日付**: 2026-03-24
- **目的**: 605# で発覚した「Ceiling による計算結果の抹消」を解消し、Sidecar (SAC) の有効率を 8% から 80%+ へ引き上げる。
- **前提**: 581# (True Additive Pipeline) および 374# (SAC Sidecar) の既存実装を活用し、不透明な「9段の乗算パイプライン」を「説明可能な加算パイプライン」へ移行する。

---

## §1 現状の課題：透明性の欠如と「TTL パラドックス」

605# の総決算において、以下の 2 つの致命的な盲点が浮き彫りとなった。

1.  **Ceiling 飽和による情報の抹消**: 9段の乗算器（EV, Vel, Trend, Tox, VG, Macro, Alert, Sidecar, Clamp）の計算結果が、最終段の `offset_ceiling_ratio` (0.15~0.35) によって頻繁にクランプされ、Alpha 層や Risk 層の微細な調整が「無効化」されている。
2.  **Sidecar (SAC) の TTL パラドックス**: SAC モデルは 2時間おきに再学習されるが、シグナルの TTL が 600秒（10分）に設定されているため、再学習直後の 8% の時間しかシグナルが有効でない。これは RL プロジェクトとしての本質的な損失である。

---

## §2 プランA：執行パイプラインの寄与度分解 (Attribution)

現在の `MakerPriceCalculator` は `offset_stages` を JSON 形式で記録しているが、これを統計的に解析するツールが不足している。

### 2.1 Attribution Analyzer の構築
`FillRecord` の `offset_stages` をパースし、以下の指標を算出する：
- **Stage Contribution (bps)**: 各ステージが最終オフセットを何 bps 変化させたか。
- **Saturation Rate (%)**: 各ステージが Ceiling（天井）を叩いた頻度。
- **Information Loss (bps)**: Ceiling がなければ到達していたはずのオフセットと、実際のクランプ値の差。

### 2.2 加算パイプライン (581#) の実効化と対数線形較正 (Log-linear Calibration)
乗算器の連鎖による指数関数的な膨張と Ceiling での飽和を避けるため、581# で実装済みの「Additive Pipeline」へ完全に移行し、数理的基礎を整備する。
- **数理モデル**: オフセット比率 $R$ を $R = R_{base} + \sum \Delta R_i$ として定義する。ここで各 $\Delta R_i$ は、**Avellaneda-Stoikov の在庫リスクモデル**における「注文密度関数」のシフト量に相当する。
- **ボラティリティ・ユニット (Volatility Unit)**: 各ステージの寄与度 $\Delta R_i$ は、固定 JPY ではなく、`RobustStats` (575#) で推定された $\sigma$ (Standard Deviation) を単位としてスケーリングする。これにより、ボラティリティが高い局面では自動的にオフセット幅が拡大し、リスク・リワード比を一定に保つ。
- **既存ログの活用**: `MakerPriceCalculator` に既に実装されている `offset_stages` (JSON) を入力ソースとし、新規に「Attribution Analyzer」を開発して、バックテストおよび実運用ログから各項の「実効寄与度」を抽出する。

---

## §3 プランB：Sidecar (SAC) の蘇生と Alpha 層の「実効化」

### 3.1 推論と再学習の分離（Inference-Retrain Separation）
現在、`sac_retrain_scheduler.py` が再学習時（2-4時間周期）にのみシグナルを更新している「TTL パラドックス」を、信号処理理論に基づき解消する。
- **サンプリング定理に基づく高頻度推論**: 1分単位のマイクロストラクチャ変化を捉えるため、サンプリング定理（Nyquist-Shannon）に則り、少なくとも実行サイクル（~120s）よりも短い 30-60秒周期での推論を実行する。
- **Sidecar Inference Agent の導入**: 重い「再学習（Retrain）」プロセスから、軽量な「推論（Inference）」プロセスを分離・独立させる。このエージェントは、最新のモデルウェイトをロードした状態で、直近の板情報を元に `directional_bias` を毎分更新し、`sidecar_signal.json` の鮮度を維持する。これにより、Sidecar 有効率を ~100% へ引き上げる。

### 3.2 Action Range の拡大と物理的制約の統合
現在の ±0.15bps という Action Range は、Coincheck の最小スプレッド（1 tick ≈ 10-15bps）に対して「ノイズ以下」の存在となっている。
- **Action Range 拡大（物理的有効化）**: ±1.0bps 程度まで拡大し、SAC が「次の tick の優先権」を理論的に争える範囲（Microstructure Edge）へ到達させる。
- **適応的平滑化 (Adaptive EMA)**: 推論頻度の向上に伴うシグナルの激しい反転を抑制するため、`confidence` が低いときは強く、高いときは弱く平滑化をかける「信頼度加重 EMA」を導入し、注文価格の安定性と追随性を両立させる。

---

## §4 検証プロセス (Verification)

1.  **Offline Replay**: 過去の `fill_records` を用い、Additive Pipeline に移行した場合の「クランプされなかったはずのアルファ」を再計算する。
2.  **G3.1-stress (Sidecar-ON)**: Sidecar 有効率を上げた状態で、slippage (1 tick) 下での PnL 改善を確認する。
3.  **Attribution Visualizer**: 各サイクルの価格決定要因をサンキー・ダイアグラム、またはスタック・バーチャートで可視化し、AI エージェントが「なぜ負けたか、あるいはなぜ買わなかったか」を物理的に説明できるようにする。

---

## §5 想定 Q&A：レビューへの先回り回答

**Q1: 加算化によってオフセットが累積し、Ceiling を超えやすくなるのでは？**
A1: 加算化の目的は上限を上げることではなく、**「寄与の可視化」**です。加算化により、どの項が Ceiling 到達の主因であるかが明確になり、寄与度の低い項を動的に縮小する等の高度な制御（RMS 連携）が可能になります。

**Q2: Sidecar の高頻度推論は、API 負荷や計算コストを増大させないか？**
A2: Sidecar の推論はローカルの ONNX/PyTorch モデルで行われ、API 取得済みの OB 情報を利用するため、外部通信負荷はゼロです。1分単位の推論であれば、現在のハードウェアリソースで十分余裕があります。

**Q3: Action Range の拡大は、逆選択（Toxicity）を拾うリスクを高めないか？**
A3: その通りです。そのため、拡大は一気に行わず、`directional_bias` と `confidence` の積に加えて、Risk 層（SpreadAnomalyDetector 等）による最終的な「価格の妥当性チェック」を通過することを前提とします。

---

## §6 結論と次の一手

本計画は、v460 の「マイクロストラクチャ・エッジ」を「理論」から「実効的なアルゴリズム」へと昇華させるためのものである。まずは `MakerPriceCalculator` の `offset_stages` ログを Attribution Analyzer で解剖することから着手する。

*風水渙（断捨離）の後は、巽（浸透）である。削ぎ落としたロジックの隙間に、洗練されたアルファを浸透させる。*

以上
