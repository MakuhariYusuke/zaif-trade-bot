# 609# レビュー: 608# 執行パイプライン寄与度分析と Sidecar 再活性化計画

- **日付**: 2026-03-25
- **目的**: 608# の Plan A (Attribution) / Plan B (Sidecar) を現行実装と金融工学理論で検証し、608# 著者と 605#-607# 著者それぞれへのタスクに分離する
- **入力**: 608#, 605#, 606#, 607#, 581#, 537#, 536#, 現行コード・設定値

---

## §0 総合評価

608# は**構造的に正しい問題意識**（パイプライン透明性不足・Sidecar 非稼動）を持つが、**前提となる数値に重大な誤認が複数ある**。これは 605# → 606# で発覚した「ハルシネーション再帰」（AI が生成した「現在値」を検証せず次の AI が引用し、誤った前提の上に計画を構築する）と同種のパターンである。

| 608# の主張 | 実装上の事実 | 判定 |
|------------|------------|------|
| TTL = 600s → 有効率 8% | `DEFAULT_SIGNAL_TTL_SEC = 7800.0`（2h10m） | ❌ **誤り** |
| Ceiling = 0.15〜0.35 | buy=0.35, sell=0.40（565# P1） | ⚠️ 部分的 |
| 「9段の乗算パイプライン」が現行 | 乗法パイプライン稼動中だが、581# 加法パイプラインは実装済み・disabled | ✅ 正確 |
| Sidecar = ±0.15bps | コードデフォルト 0.15、YAML 設定 0.20（544# ladder） | ⚠️ やや古い |
| Stage Max Mult なし（暗示） | 565# で各段 2.0 cap 実装済み | ❌ **言及なし** |

608# の**真の価値**は、表面的な数値ではなく、(1) Attribution Analyzer という可視化ツールの提案、および (2) Inference-Retrain 分離というアーキテクチャ提案にある。以下で各プランを検証する。

---

## §1 608# §1「TTL パラドックス」のファクトチェック

### 1.1 TTL は既に 7800s — ただし問題の本質は別にある

608# が述べる「TTL 600s → 8% しか有効でない」は**数値として誤り**。しかし、**Sidecar 有効率が低い**という問題認識自体は正しい。

**真の原因**は以下の 3 層構造にある：

1. **再学習頻度の制約**: `retrain_interval_sec=7200`（2h）、`retrain_interval_max_sec=14400`（4h）。成功時のみ信号更新。
2. **データ窓ガード (600#)**: 訓練データが 2h 分進まないと retrain をスキップ → 信号更新なし。
3. **OOS 失敗時の fallback**: `gross_roi <= min_gross_roi` → 既存信号を維持（24h 以内なら）or neutral fallback（bias=0.0）。

**結果**: 7800s の TTL があっても、retrain が 4h 以上空くか連続失敗すると信号は stale 化する。有効率 8% の原因は TTL 値ではなく**信号更新が retrain 成功に完全依存している設計**。

608# の「Inference-Retrain 分離」という**解決策**自体は、この真の原因を正しく狙っている。ただし問題の原因記述が誤っているため、「なぜそれが必要か」の説明が読者に誤解を与える。

### 1.2 Nyquist-Shannon 定理の適用について

608# §3.1 で「サンプリング定理に則り 30-60 秒周期での推論」と述べているが、これは信号処理理論の**誤用**である。

- Nyquist-Shannon 定理は「帯域制限されたアナログ信号を完全に再構成するための最小サンプリング周波数」に関するもの
- SAC の directional_bias は離散的な予測値であり、帯域制限信号ではない
- 正しい議論は**Kalman フィルタや Bayesian 更新**の文脈：「新しい観測（OB snapshot）が得られるたびに事後分布を更新する」

推論頻度の正当な根拠は：
- **情報理論的**: OB snapshot のエントロピーが cycle 間（~120s）で有意に変化する → 推論も同等以上の頻度で更新すべき
- **実用的**: 推論コストが無視できる（ローカル ONNX/PyTorch、外部 API 不要）ならば高頻度で損はない

---

## §2 608# Plan A「Attribution Analyzer」の検証

### 2.1 理論的妥当性: ★★★★★

Attribution Analysis は**無条件に正しい**。現在のシステムが抱える最大の問題は「なぜその価格で発注したのか」の説明可能性の欠如である。

- **金融規制的**: MiFID II/MAR における algorithmic trading の best execution 記録義務（本邦では金商法上の最良執行義務）に相当
- **ML 運用的**: SHAP/LIME に相当するモデル説明可能性を執行パイプラインに適用するアプローチ
- **デバッグ的**: 531# で「母集団カウント不一致」が発生した根因は、各段の寄与が追跡不能だったこと

### 2.2 実装の現実性

**既に 14% のデータで `offset_stages` が記録されている**（436# 以降）。Attribution Analyzer は既存ログのパーサーとして構築可能。

ただし以下の注意点がある：

| 項目 | 608# の提案 | 実装上の考慮 |
|------|-----------|-------------|
| Stage Contribution (bps) | 各段の bps 寄与 | 乗法パイプラインでは段間の寄与が**非加法的**（A×B ≠ A+B）。正しい attribution は Shapley 値が必要 |
| Saturation Rate (%) | ceiling hit 頻度 | ✅ 直接計測可能。`_ceiling.clamped` フラグ |
| Information Loss (bps) | clamp 前後の差 | ✅ `pre_clamp - post_clamp` で計算可能 |

**Shapley 値の必要性**:

乗法パイプラインにおいて、各段の「寄与度」を単純な差分で計算すると、段の順序に依存する不公平な attribution になる。例：

$$
\text{Stage 1}: \times 1.5, \quad \text{Stage 2}: \times 2.0
$$
$$
\text{Output}: 0.05 \times 1.5 \times 2.0 = 0.15
$$

- Stage 1 の「寄与」を $0.05 \times 1.5 - 0.05 = 0.025$ とすると
- Stage 2 の「寄与」を $0.075 \times 2.0 - 0.075 = 0.075$ となる

Stage 2 は Stage 1 の出力の上に乗っているため、見かけ上の寄与が大きくなる。Shapley 値はこのバイアスを除去する：

$$
\phi_i = \frac{1}{n!} \sum_{\pi \in \Pi} [\text{marginal contribution of } i \text{ in order } \pi]
$$

2 段なら：$\phi_1 = \frac{1}{2}[(0.05 \times 1.5 - 0.05) + (0.15 - 0.05 \times 2.0)] = \frac{1}{2}[0.025 + 0.05] = 0.0375$

**ただし**、581# の加法パイプラインが有効化されれば、Attribution は単純な $\Delta R_i$ の列挙で済む。これは加法移行の**もう一つの正当化根拠**。

### 2.3 提案: Attribution の 2 段階導入

**Phase 1（即座実装可能）**: clamp_rate と information_loss のみを計測するライトウェイト集計。`offset_stages` JSON をパースし、`pre_clamp` と `post_clamp` の差分を記録。これは Python スクリプト 1 本で可能。

**Phase 2（加法パイプライン有効化後）**: 各 $\Delta R_i$ の実値ベースの attribution。サンキー・ダイアグラムによる可視化。加法パイプラインでは Shapley 値不要（加法的寄与が直接読める）。

---

## §3 608# Plan B「Sidecar 蘇生」の検証

### 3.1 Inference-Retrain 分離: ★★★★☆（方向性正しいが設計要精査）

608# の「重い再学習プロセスから軽量な推論を分離」は正しい。現行の retrain scheduler は「訓練成功時の最終観測値 1 つ」でしか推論しない。

**設計選択肢の比較**:

| 方式 | Pros | Cons |
|------|------|------|
| **A. 独立プロセス（608# 提案）** | 完全な分離、crash isolation | プロセス間通信、モデルファイル競合、メモリ 2 倍 |
| **B. fill_test 内組込み推論** | 既存アーキテクチャ内、共有メモリ | fill cycle に推論レイテンシが加算（~50ms） |
| **C. retrain 間の定期推論 hook** | scheduler に軽量 inference loop 追加 | scheduler の責務肥大 |

**推奨: 方式 B（fill_test 内組込み）**

理由：
- fill_test は**同期ループ**アーキテクチャ（607# で確認）。各 cycle 開始時に OB を取得済み
- 推論は `model.predict(obs, deterministic=True)` の 1 回呼出し（~50ms）
- 既存の `read_sidecar_signal_with_status()` を「ファイル読出し」から「インプロセス推論」に差し替えるだけ
- モデルの hot-reload は retrain 成功時のウェイトファイル更新を検知して行う

```python
# 概念実装
class InProcessSidecarInference:
    def __init__(self, model_path: Path):
        self._model = load_model(model_path)
        self._model_mtime = model_path.stat().st_mtime

    def infer(self, obs: np.ndarray) -> SidecarSignal:
        # Hot-reload check
        current_mtime = self._model_path.stat().st_mtime
        if current_mtime > self._model_mtime:
            self._model = load_model(self._model_path)
            self._model_mtime = current_mtime
        action, _ = self._model.predict(obs, deterministic=True)
        return SidecarSignal(
            directional_bias=float(action[0]),
            confidence=self._last_confidence,
            timestamp=current_iso_timestamp(),
            ...
        )
```

この方式なら：
- 有効率は**理論上 100%**（毎 cycle で推論）
- ファイル I/O のレース条件（488# P1）が消滅
- TTL の概念自体が不要になる

### 3.2 Action Range 拡大: ★★★☆☆（慎重に）

608# の ±0.15bps → ±1.0bps 拡大提案について：

**金融工学的分析**:

Coincheck BTC/JPY の典型的スプレッド ≈ 500-1000 JPY。BTC ≈ 10,000,000 JPY とすると：

$$
\text{1 bps} = 10{,}000{,}000 \times 0.0001 = 1{,}000 \text{ JPY}
$$

- ±0.15bps = ±150 JPY → スプレッド 500 JPY の 30% → **ノイズ域**
- ±1.0bps = ±1,000 JPY → スプレッド 500 JPY の 200% → **mid を超過する可能性**
- ±0.5bps = ±500 JPY → スプレッド 500 JPY の 100% → **ちょうど半スプレッド**

**Avellaneda-Stoikov の最適 half-spread**:

$$
\delta^* = \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right) \approx \frac{1}{\kappa} \quad (\gamma \to 0)
$$

ここで $\kappa$ は注文到着強度。fill rate ≈ 30% かつ cycle ≈ 120s なら $\kappa \approx 0.0025 \text{s}^{-1}$、$\delta^* \approx 400$ — つまり ~0.4bps。

**推奨**: ±0.5bps を上限とし、段階的に拡大（0.20 → 0.30 → 0.50）。±1.0bps は mid 超過リスクが高い。

### 3.3 適応的 EMA: ★★★★☆

608# の「信頼度加重 EMA」は信号処理的に妥当：

$$
\text{EMA}_t = \alpha(c_t) \cdot x_t + (1 - \alpha(c_t)) \cdot \text{EMA}_{t-1}
$$

ここで $\alpha(c) = \alpha_{min} + c \cdot (\alpha_{max} - \alpha_{min})$、$c$ は confidence。

- 高 confidence → 大きな $\alpha$ → 新情報に素早く追随
- 低 confidence → 小さな $\alpha$ → ノイズ耐性

**実装上の注意**: `RobustStats.asymmetric_ema()` が既に存在する（575#）。これをラップして confidence-weighted 版を作るのが最も DRY。

---

## §4 608# §2.2「加算パイプラインと対数線形較正」の数理的検証

### 4.1 ボラティリティ・ユニット正規化: ★★★★★

608# の「各 $\Delta R_i$ を $\sigma$ 単位でスケーリング」は**理論的に最も正しいアプローチ**。

**根拠**: Avellaneda-Stoikov (2008) の最適スプレッドは $\gamma \sigma^2 \tau$ に比例する。各防衛シグナルを $\sigma$ 単位で表現すれば、ボラティリティ regime 変化に対して**自動的にスケール調整**される。

$$
\Delta R_i = f_i(\text{signal}_i) \times \frac{\sigma_{current}}{\sigma_{baseline}}
$$

- 高ボラ時: $\sigma_{current} / \sigma_{baseline} > 1$ → offset 自動拡大
- 低ボラ時: $\sigma_{current} / \sigma_{baseline} < 1$ → offset 自動縮小

**既存の RobustStats (575#) の `asymmetric_ema()` が $\sigma_{current}$ を提供可能**。$\sigma_{baseline}$ は長期 EMA（τ ≈ 24h）で推定。

### 4.2 RMS 集約の数理的妥当性

581# の RMS 集約：

$$
R = R_{base} + \sqrt{\sum_{i \in \text{tox}} (\Delta R_i)^2} + \sqrt{\sum_{j \in \text{liq}} (\Delta R_j)^2}
$$

これは**リスクの独立成分合成**として解釈できる：

- 各 $\Delta R_i$ が独立なリスク要因ならば、合成リスクは $\sqrt{\sum (\Delta R_i)^2}$（ピタゴラス的合成）
- これはポートフォリオ理論における**分散共分散行列の対角成分のみ**を使った VaR 推定と同等

**問題点**: リスク要因間に**正の相関**がある場合（例: volatility guard と toxicity は同時に上昇しやすい）、RMS は**リスクを過小評価**する。

$$
\text{True risk} = \sqrt{\sum_i \Delta R_i^2 + 2\sum_{i<j} \rho_{ij} \Delta R_i \Delta R_j} > \sqrt{\sum \Delta R_i^2}
$$

**推奨**: RMS を初期実装として採用しつつ、$\rho_{ij}$ の推定を行い、将来的に相関行列を組み込む余地を残す。当面は RMS の過小評価を ceiling（上限ガード）で補完。

### 4.3 加法パイプラインにも ceiling が適用されている問題

581# コードを確認したところ、**加法パイプラインも最終段で同じ `clamp_offset_ratio_to_ceiling()` を適用している**。

これは 608# が暗示する「加法移行で ceiling 飽和が解消される」という期待と矛盾する。加法パイプラインが乗法より穏やかな出力を生む（RMS 合成は乗算累積より膨張しない）ため、ceiling hit 頻度は下がるが、**構造的には同じ問題**。

$$
\text{乗法}: 0.05 \times 1.5^8 = 1.28 \quad \text{（ceiling 0.35 で 73% 切捨て）}
$$
$$
\text{加法 RMS}: 0.10 + \sqrt{0.02^2 + 0.01^2} + \sqrt{0.01^2 + 0.005^2} = 0.134 \quad \text{（ceiling 内）}
$$

RMS は乗算的膨張を防ぐため、ceiling に当たりにくい。ただし、これは各 $\Delta R_i$ の**大きさの較正**次第。現行の multiplier 値をそのまま $\Delta R$ に変換すると、較正作業が必要。

---

## §5 608# が見落としている論点

### 5.1 606# 正誤表との整合性

608# は 605# の分析結果を前提としているが、**606# が 605# の複数の前提を否定している**。608# は 606# の正誤表を反映していない。

| 605# の誤り（606# で訂正） | 608# での扱い |
|---------------------------|--------------|
| Ceiling = 0.25（実際は buy=0.35, sell=0.40） | 「0.15〜0.35」と記載（sell=0.40 を見落とし） |
| composite_risk = false（実際は true） | 未言及 |
| Stage Max Mult なし（実際は 2.0 cap） | 未言及 |
| Sidecar TTL = 600s（実際は 7800s） | **そのまま引用して §1 の前提に** |
| sell_dynamic_kill = 1800s（実際は 600s） | 未言及 |

**教訓**: 608# は 605# を検証せずに引用した。これは 592# → 605# で発生した「ハルシネーション連鎖」の 3 回目の再発。

### 5.2 607# アーキテクチャ監査との連携不足

607# は fill_test の**同期ループ構造**を確認し、以下を実証した：
- 次のサイクルに入る前に前サイクルの注文が完了する
- REST API で毎 cycle OB を取得（キャッシュではない）
- レース条件なし（シングルスレッド）

この知見に基づけば、**Sidecar 推論を fill_test の cycle 内に埋め込む（§3.1 方式 B）** が自然な選択。608# が提案する「独立プロセスとしての Inference Agent」は、607# が証明した同期アーキテクチャの利点を活かしていない。

### 5.3 OFI-Lite (543#) との統合

608# は OFI について全く言及していないが、543# で `ofi_lite` が offset_stages に記録されており、RobustStats の `median_filter_fast()` で平滑化された OFI 中央値が maker_price.py で計算されている。

Sidecar の観測空間に OFI を含めることで、SAC が市場の micro-structure を直接学習できる。これは 608# Plan B の「板情報を元に directional_bias を毎分更新」の具体化として最適。

---

## §6 タスク分離: 608# 著者 vs 605#-607# 著者

### 6.1 608# 著者へのタスク

608# 著者は**計画立案と理論的フレームワーク構築**に強みがある。以下のタスクが適切：

| # | タスク | 優先度 | 根拠 |
|---|-------|--------|------|
| **T-A1** | 608# §1 の TTL 記述を 606# 正誤表に基づき修正。「TTL パラドックス」を「Inference-Retrain 結合問題」に改題 | P0 | 誤情報の是正 |
| **T-A2** | Attribution Analyzer の設計仕様書作成（入力: offset_stages JSON、出力: stage_contribution_bps, saturation_rate, information_loss_bps） | P1 | 608# の最大の独自価値 |
| **T-A3** | $\sigma$ ユニット正規化の較正式を定義（$\Delta R_i = f_i(\text{signal}) \times \sigma_{current} / \sigma_{baseline}$）。各段の $f_i$ を既存 multiplier 値から逆算 | P1 | Avellaneda-Stoikov との接続 |
| **T-A4** | Action Range 拡大のバックテスト設計。過去の fill_records を用い、±0.3bps / ±0.5bps での仮想 PnL を再計算 | P2 | リスク管理付き拡大 |
| **T-A5** | Sidecar Inference Agent の OB 特徴量仕様定義（OFI-Lite, microprice, depth_imbalance を SAC 観測空間に追加する場合の次元拡張設計） | P2 | SAC 観測空間の拡充 |

### 6.2 605#-607# 著者へのタスク

605#-607# 著者は**コード実装と検証**に強みがある。以下のタスクが適切：

| # | タスク | 優先度 | 根拠 |
|---|-------|--------|------|
| **T-B1** | 581# 加法パイプラインの A/B テスト実行。`experimental_additive_pipeline.enabled: true` に設定し、24h の fill_rate / AS-PnL / clamp_rate を計測 | P0 | 実装済み・未検証の最大資産 |
| **T-B2** | Sidecar インプロセス推論の実装（§3.1 方式 B）。`read_sidecar_signal_with_status()` を hot-reload 付きモデル推論に差し替え | P1 | 有効率 8% → 100% |
| **T-B3** | Attribution Analyzer Phase 1 実装（clamp_rate + information_loss の集計スクリプト）。既存 offset_stages JSON をパースして CSV 出力 | P1 | T-A2 の設計に基づく |
| **T-B4** | Sidecar boost_bps の段階的拡大（0.20 → 0.30 → 0.50）。各段で 48h の fill_records を記録し AS-PnL 影響を計測 | P2 | T-A4 のバックテスト結果に基づく |
| **T-B5** | 加法パイプラインの ceiling 挙動検証。RMS 出力が ceiling を超える頻度を計測し、乗法パイプラインとの比較データを取得 | P1 | §4.3 の懸念検証 |
| **T-B6** | 607# hot-reload 修正後の SAD/MCB 再起動テスト。`enabled: true → false → true` の YAML 変更で hot-reload が正しく再構築されることを確認 | P0 | 606#/607# の成果物の検証 |

---

## §7 追加提案: 608# のスコープ外だが収益性向上に寄与するもの

### 7.1 提案 P1: Fill-or-Kill の意思決定爆速化 — Decision Latency Budget

**現状**: cycle ≈ 120s は**暗号通貨マーケットメイキングにおいて極めて遅い**。Binance/Bybit のトップ MM は < 100ms で意思決定する。

**金融工学的根拠**: Optimal Execution 理論（Almgren-Chriss 2000）では、execution horizon が長いほど市場インパクトリスクが増大する。120s のうち「注文が板に晒される時間」が長いほど逆選択リスクが高い。

**提案**: cycle 内の意思決定レイテンシを計測し、「推論 50ms + 価格計算 10ms + API 呼出 200ms」のように分解する。推論をインプロセス化（§3.1 方式 B）すれば、ファイル I/O の 20-50ms を節約できる。

### 7.2 提案 P2: 条件付き Ceiling — Risk-Budget-Aware Ceiling

**現状**: ceiling は全条件で一律（buy=0.35, sell=0.40 + hour_ceiling_mult）。

**金融工学的洞察**: 「すべてのリスクシグナルが同時に発火する」状況は、市場の構造変化（regime shift）の兆候であり、防衛を最大化すべき瞬間。しかし ceiling がこれを抑制している。

**提案**: ceiling を**在庫状態に連動**させる：
- inventory_imbalance ≈ 0（中立）→ ceiling = 0.35（通常値）
- inventory_imbalance > 0.5（在庫過多）→ sell ceiling = 0.20（売りは aggressive に）
- inventory_imbalance < -0.5（在庫不足）→ buy ceiling = 0.20（買いは aggressive に）

Ho-Stoll (1981) の inventory aversion モデルに基づく：在庫リスクが高いときは、中立化を最優先し、防衛オフセットを縮小して fill 確率を上げる。

### 7.3 提案 P3: Multi-Horizon SAC — 複数時間軸の統合

**現状**: SAC は単一の時間軸（直近の状態 → 次のアクション）で学習。

**金融工学的根拠**: 市場のダイナミクスは multi-scale（Mandelbrot 1963, Müller et al. 1997 — heterogeneous market hypothesis）。1 分足と 15 分足では異なるパターンが支配的。

**提案**: SAC の観測空間に**複数時間軸の特徴量**を追加：

$$
\mathbf{o}_t = [\underbrace{x_t^{(1m)}}_{\text{1min}}, \underbrace{x_t^{(5m)}}_{\text{5min}}, \underbrace{x_t^{(15m)}}_{\text{15min}}, \underbrace{x_t^{(1h)}}_{\text{1hour}}]
$$

既存の `regime_detector` が 20-bar window の単一スケールで動作しているが、macro_trend (458#) が長期を補完している。これを SAC の特徴に統合する。

### 7.4 提案 P4: Realized Spread Persistence — 「約定後の実質スプレッド」の追跡

**背景**: Adverse Selection は「約定直後の価格変動」で計測される（Roll 1984, Huang-Stoll 1997）。fill_records に `post_fill_30s_pnl` があるが、これをリアルタイム防衛に活用する feedback loop が弱い。

**金融数学**: realized half-spread = $\frac{1}{2}(p_{trade} - m_{t+\Delta})$ where $m_{t+\Delta}$ は約定 $\Delta$ 秒後の midprice。

$$
\text{HS}_t^{realized} = \sum_{k=1}^{N} w_k \cdot \text{HS}_{t-k}^{realized}, \quad w_k = \alpha (1-\alpha)^{k-1}
$$

**提案**: realized half-spread の EWMA を計算し、これが負（= 逆選択コスト > スプレッド利益）の場合に**自動的に base_offset を引き上げる**。240# Toxicity Budget が部分的にこの役割を果たすが、直接的な HS ベースの feedback は未実装。

### 7.5 提案 P5: Adversarial Scenario Testing — 敵対的シナリオでの堅牢性

**現状**: G3.1-stress テストが提案されているが、**敵対的トレーダーのモデリング**が欠如。

**金融工学的根拠**: Hasbrouck (1988) の vector autoregression モデルでは、情報トレーダーは MM の気配値の「予測可能なパターン」を exploit する。

**提案**: 
- MM の過去の注文パターン（時間帯、サイス、方向）を分析し、**予測可能性スコア**を計算
- 例: 「毎正時に必ず buy を出す」→ 情報トレーダーに先回りされるリスク
- cycle 間隔にランダムな jitter を加える（±30s）ことでパターンを破壊

### 7.6 提案 P6: 加法パイプラインの eDRC (Exponential Demand Response Curve)

581# の設定に `edrc_alpha`, `edrc_beta`, `edrc_c_base` が存在するが、現在は `0.0, 0.0, 0.40` で実質無効化。

**eDRC の数理**:

$$
\text{Offset} = c_{base} \cdot e^{-\alpha \cdot \text{demand}} + \beta \cdot \text{demand}
$$

- demand が低い（市場が静か）→ offset ≈ $c_{base}$（ワイド・スプレッド、安全に待つ）
- demand が高い（注文殺到）→ offset ↓（タイト・スプレッド、fill を取りに行く）

これは Cont & de Larrard (2013) の order book dynamics に基づく**逆数需要関数**の実装。proper calibration により、静かな市場で不必要に fill を求めず、活況時に積極化する adaptive behavior が実現される。

---

## §8 結論

### 608# の最大の価値
- **Attribution Analyzer** の概念は無条件に正しく、実装優先度が最も高い
- **Inference-Retrain 分離**の方向性は正しいが、独立プロセスではなく fill_test 内組込みを推奨
- **σ ユニット正規化**は Avellaneda-Stoikov 理論との整合性が高い

### 608# の最大のリスク
- **ハルシネーション連鎖（3 回目）**: 605# の誤った数値を検証なく引用。606# 正誤表の反映なし
- **582# / 606# の既存実装の見落とし**: Stage Max Mult (2.0 cap), composite_risk (true), sell_kill_duration (600s) が既に実装済みであることを認識していない
- **加法パイプラインにも ceiling が適用される**点への言及なし

### 最優先アクション
1. **T-B1**: 581# 加法パイプラインの A/B テスト（YAML 1 行変更で開始可能）
2. **T-B2**: Sidecar インプロセス推論の実装（有効率 8% → 100%）
3. **T-A1**: 608# の誤情報訂正（ハルシネーション連鎖の断ち切り）

---

*以上*
