# Gemini 向け指示プロンプト: 608# 正誤訂正 + 数理仕様策定タスク

あなたは以前 608# (執行パイプラインの寄与度分析と Sidecar 再活性化計画) を執筆しました。その後、609# (技術レビュー)、610# (三者検証)、611# (四者監査)、613# (深堀りレビュー) を通じて、608# の主張に対する検証と補正が行われました。

## あなたの役割

**数理仕様・設計文書の策定に専念してください。コード変更・YAML 変更・テスト実装は担当外です。**

あなたの強みは問題定義力と数理的フレーミングです。一方で、608# では前提数値に複数の致命的誤認がありました。今回はそれを訂正した上で、実装担当者が迷いなく作業できる仕様書群を作成してください。

---

## 前提: 608# で確認された誤り

以下は **実コードから検証済みの正確な値** です。これらを前提に作業してください。

| 項目 | 608# での記載 | 実値 (コード検証済み) | 根拠 |
|------|-------------|---------------------|------|
| Sidecar TTL | 600s | `DEFAULT_SIGNAL_TTL_SEC = 7800.0` | `sidecar_types.py:45` |
| Ceiling (buy) | 0.15-0.35 | `offset_ceiling_ratio_buy: 0.35` | `fill_test.yaml`, `fill_config.py:691` |
| Ceiling (sell) | 0.15-0.35 | `offset_ceiling_ratio_sell: 0.40` | `fill_test.yaml`, `fill_config.py:693` |
| max_boost_bps | ±0.15bps | YAML設定: `0.20` (コードデフォルト: `0.15`) | `fill_test.yaml:632`, `fill_config.py:640` |
| Stage Max Mult | (未言及=なし前提) | **各段 2.0 cap 実装済み** | 565# P3, `offset_pipeline.py` |
| composite_risk | (未言及) | `composite_risk_enabled: true` | `fill_test.yaml` |
| パイプライン状態 | 9段乗算のみ | 乗法=稼働中 / **581# 加法=実装済み・disabled** | `offset_pipeline.py:113-328`, トグル `experimental_additive_pipeline` |
| Nyquist-Shannon 適用 | 推論頻度の根拠 | **誤用** — bias は帯域制限信号ではない | 609# §1.2 で指摘済み |

### TTL についての補足
TTL 値自体は十分(7800s)です。Sidecar 有効率が低い **真の原因** は「信号更新が retrain 成功に完全依存している設計」(609# §1.1)。Inference-Retrain 分離という解決策(608# Plan B)は正しいですが、問題の原因記述を修正してください。

---

## タスク一覧

以下の T1-T6 を順番に実施し、**1つの文書 (614#)** として出力してください。

### T1: 608# 正誤表

608# で使用した全ての前提値について、上記表をベースに完全な正誤対照表を作成してください。単に値を訂正するだけでなく、「その誤認がどの議論をどう歪めたか」を各項目に付記してください。

### T2: Attribution Analyzer 仕様書 (Phase 1)

**制約**: 新規スクリプトではなく、既存の `scripts/v460/analysis/analyze_fill_logs.py` への拡張として設計すること。

同スクリプトは既に以下の基盤を持っています:
- `_load_executor_offset_stages()`: executor_offset_stages JSON パース
- `_is_additive_execution()`: additive/multiplicative 判定ロジック
- 加法/乗法の比較分析フレームワーク

Phase 1 で計測すべき指標:
- **clamp_rate**: ceiling を叩いた頻度 (buy/sell 別)
- **information_loss**: clamp 前後のオフセット差 (bps)
- **stage_saturation**: 各段が 2.0 cap を叩いた頻度

出力フォーマット (JSON/CSV)、集計粒度 (per-cycle / hourly / daily)、フィルタ条件を明示してください。

### T3: σ-unit 正規化の数理仕様

608# §2.2 の「ボラティリティ・ユニット正規化」を正式仕様として整備してください。

含めるべき内容:
1. 各 $\Delta R_i$ のスケーリング式: $\Delta R_i = f_i(\text{signal}_i) \times \frac{\sigma_{current}}{\sigma_{baseline}}$
2. $\sigma_{current}$ の算出: 既存の `RobustStats.asymmetric_ema()` (575#, `robust_stats.py:58`) をどう利用するか
3. $\sigma_{baseline}$ の推定方法: 長期 EMA の窓幅、初期値、更新頻度
4. Avellaneda-Stoikov の $\gamma\sigma^2\tau$ との理論的接続

### T4: 加法パイプライン A/B テスト設計

**重要な前提**: 加法パイプラインも最終段で同じ `clamp_offset_ratio_to_ceiling()` を通ります (`offset_ceiling.py:16`)。加法移行で ceiling 飽和が「なくなる」のではなく「頻度が下がる」ことをテスト設計に反映してください。

含めるべき内容:
1. 仮説: 帰無仮説と対立仮説の明示
2. 比較指標: PnL, fill_rate, clamp_rate, information_loss の最低 4 指標
3. トグル: `experimental_additive_pipeline` (581#) の使用方法
4. サンプルサイズ・期間の根拠
5. 判定基準: 各指標の有意水準、実用的有意差の定義
6. 撤退基準: A/B テスト中に明確な悪化が見られた場合の停止条件

### T5: Sidecar 方式 B (in-process inference) の feature contract 仕様

609# が推す方式 B (fill_test 内組込み推論) の **最大の障壁は feature contract** です。

現状:
- 訓練側: `btc_jpy_1m_full_registry_features.parquet` から `cfg.feature_columns` を選択 → env に注入
- 推論消費側: `orchestrator_mid_cycle.py` は `SidecarSignal.directional_bias` (float 1個) を読むのみ

あなたが策定すべき仕様:
1. training env と live 側で **共有すべき特徴量リスト** (カラム名、型、単位)
2. 各特徴量の **正規化パラメータ** の管理方法 (mean/std の保存場所、更新タイミング)
3. live 側で特徴量を構築するための **データソース** と **計算頻度**

**実装** は実装担当が行います。あなたは「何を」「どの精度で」「どの頻度で」提供すべきかの仕様を書いてください。

### T6: Action Range 拡大ロードマップ

段階: `0.20 → 0.30 → 0.50 bps`

各段階について:
1. **期待効果**: BTC/JPY スプレッド比での影響幅
2. **前提条件**: その段階に進むために満たすべき条件
3. **監視指標**: `postonly_crossing_skip` 発生率の閾値 (現状 ~1.1%, これ以上なら撤退)
4. **撤退基準**: fill_rate, PnL, crossing_skip_rate の各閾値
5. **非対称性**: buy/sell で異なる ceiling (0.35/0.40) との相互作用

**注意**: ±1.0bps は 609#/610#/611#/613# の全者が「過大」と判定しています。0.50bps を当面の上限として設計してください。

---

## 出力形式

- 文書番号: **614#**
- ファイル名: `docs/v460/614_phg_attribution_spec_and_sidecar_contract.md`
- 各タスクを §1 (T1) 〜 §6 (T6) として構成
- 数式は KaTeX/LaTeX 記法 ($...$, $$...$$) を使用
- コード例は概念レベル (Python 疑似コード) のみ。実装可能なコードは書かないでください
- 全ての数値には出典 (ファイルパス:行番号 or 文書番号) を付記

## 禁止事項

- YAML ファイルの変更提案 (実装担当の責務)
- テストコードの記述 (実装担当の責務)
- 新規 .py ファイルの作成提案 (既存拡張のみ)
- 検証なしの数値の引用 (上記の正誤表を必ず参照)
- Nyquist-Shannon 定理への言及 (609# で誤用と指摘済み)
