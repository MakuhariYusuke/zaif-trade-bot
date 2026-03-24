# Gemini向け指示プロンプト: Attribution Phase 2 と Feature Contract 詳細化 (616#)

あなたは 614# で Attribution Phase 1 および基本的な Feature Contract の仕様を策定し、615# でその精度を引き上げました。実装担当 (Copilot) は現在、Phase 1 の可視化機能（`clamp_rate`, `information_loss`, `stage_saturation`, `composite_risk 複合分析`）を `analyze_fill_logs.py` に組み込んでいます。

実装が完了しデータが可視化された後、次の打ち手を最速で打つために、以下の「Phase 2 向けの数理仕様」および「Live Feature 構築の詳細仕様」を先行して作成してください。

## あなたの役割
**数理仕様・設計文書の策定に専念してください。コード変更・YAML 変更・テスト実装は担当外です。**

---

## タスク一覧

以下の T1-T3 を順番に実施し、**1つの文書 (616#)** として出力してください。

### T1: Attribution Phase 2 (加法パイプライン向け) の数理仕様

581# の加法パイプライン（`experimental_additive_pipeline=true`）が有効化された環境における、Attribution の仕様を策定してください。

**背景**:
加法パイプラインでは各段の寄与が「和」となるため、Phase 1 のような乗法特有の「Saturation (2.0 cap) 到達判定」よりも、加法的限界（RMS Ceiling）への接近度合いが重要になります。

**策定すべき仕様**:
1. **各 $\Delta R_i$ の実効寄与度**: `tox_buffer` および `liq_buffer` (RMS 合成値) に対する、個別の入力シグナルの「実質的な bps 寄与」をどう一意に逆算するか。
2. **RMS Ceiling 接近率 (%)**: `tox_buffer` や `liq_buffer` が全体 Ceiling に対して占有している割合の計算式。
3. **加法・乗法の比較評価式**: A/B テスト（T4 in 614#）において、「本当に加法の方が Information Loss が小さいか」を定量比較するための統計モデル。

### T2: Live 側 Feature Builder の構造仕様 (Feature Contract 詳解)

614# §5 で定義した 6 つの特徴量について、推論を `fill_test` 内部に組み込む（方式 B）ために、**実運用 (Live) 環境でのデータ構築構造**を設計してください。

**対象特徴量** (`614# §5.1` より):
- `price_velocity`, `order_flow_imbalance`, `micro_volatility`, `parkinson_sigma`, `vpin_proxy`, `ema_velocity_bps`

**策定すべき仕様**:
1. **メモリ構造**: Live 側でこれら 1m/5m/等 のウィンドウ計算を維持するためには、リングバッファ（あるいは DataFrame）をどう保持すべきか。
2. **計算タイミング**: `fill_cycle` は約 120s 周期で発火するが、特徴量は 1m などのより短いエッジを要する。この「非同期的な時間解像度のズレ」をどう同期させるか（WebSocket の L2 スナップショット更新をフックするか、固定周期タイマーか）。
3. **欠損値 (NaN) 取扱**: Live 起動直後（バッファが溜まる前）にモデルに渡す特徴量が欠損している場合のフォールバック戦略（訓練側の Mean で埋める、ゼロ埋め、等）。

### T3: 適応的平滑化 (Adaptive EMA) の数理仕様

608# §3.2 で提案された「信頼度加重 EMA」の仕様を確定させてください。

**背景**:
推論頻度が高まる（例: 毎分）と、シグナル (`directional_bias`) のノイズによる反転が激しくなります。これを抑えるため、モデル出力の `confidence` を利用して EMA の $\alpha$ (平滑化係数) を動的に可変させる案です。

**策定すべき仕様**:
1. 動的 $\alpha$ の計算公式: $\alpha_t = f(\text{confidence}_t, \alpha_{min}, \alpha_{max})$
2. $\text{EMA}_t$ の更新公式（既存の `RobustStats.asymmetric_ema()` を利用できる場合はそのマッピング）
3. 推奨パラメータ: $\alpha_{min}$ と $\alpha_{max}$ の初期提示値とその根拠

---

## 出力形式

- 文書番号: **616#**
- ファイル名: `docs/v460/616_phg_attribution_phase2_and_live_feature_spec.md`
- 各タスクを §1 (T1) 〜 §3 (T3) として構成
- 数式は KaTeX/LaTeX 記法 ($...$, $$...$$) を使用

## 禁止事項
- YAML ファイルの変更提案
- 実装コードの記述（概念レベルの疑似コードのみ許可）
- 新規 `.py` ファイルの作成提案（既存の `analyze_fill_logs.py` 等への拡張方針を維持）
- 614# / 615# で定義された事実（TTL=7800s, composite_risk=true 等）と矛盾する前提の持込