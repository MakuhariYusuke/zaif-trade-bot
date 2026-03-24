# 613# 609#-611# 深堀りレビューと 608# 著者への課題整理

- **日付**: 2026-03-25
- **目的**: 609#-611# の三文書に対するコード裏取り付き深堀り評価。加えて 608# 著者（Gemini）に課すべきタスクの根拠を明確にする。
- **Author**: GitHub Copilot
- **入力**: 608#-612#, 現行コードベース, 前回検証結果

---

## §0 総括

| 文書 | 信頼度 | 行番号精度 | 論旨の鋭さ | 実装接地性 | 総合 |
|------|--------|-----------|-----------|-----------|------|
| 609# | ★★★★★ | ― (行番号なし) | ★★★★★ | ★★★★☆ | **最優秀レビュー** |
| 610# | ★★★★☆ | ⚠️ 3/10 に 1-4行ズレ | ★★★★☆ | ★★★★★ | **最優秀コーディネーション** |
| 611# | ★★★★☆ | ✅ 6/6 完全一致 | ★★★★★ | ★★★☆☆ | **最優秀着眼点、過大主張あり** |

三文書ともレビューとしての品質は高い。だが各文書には「過信すると危ない箇所」がある。以下で一つずつ裏を取る。

---

## §1 609# 深堀り

### 1.1 支持される主張

**(A) TTL 真因の再定義** — 完全に正しい。

> 「TTL 値ではなく、信号更新が retrain 成功に完全依存している設計が真因」

`sac_retrain_scheduler.py` の `_update_sidecar_signal()` (L1005) は retrain 成功後の最終 obs でのみ呼ばれる。retrain 失敗時は既存信号を維持するか neutral fallback する。TTL=7800s という値自体は十分だが、**更新頻度が retrain 間隔（2-4h）に律速される**点が本質的問題。609# はこれを正確に看破している。

**(B) Shapley 値の必要性** — 乗法パイプラインに対しては理論的に正しい。

乗法的段間の寄与を単純差分で測ると順序バイアスが入る。609# の数式展開 $\phi_i = \frac{1}{n!}\sum_{\pi}[\text{marginal contribution}]$ は正確。

ただし、**581# の加法パイプラインが有効化されれば Shapley 値は不要になる**（加法的寄与は直読可能）。従って Shapley 値の実装は「乗法パイプライン延命中の暫定措置」であり、Phase 2（加法移行後）では不要。

**(C) Phase 1 / Phase 2 の段階分割** — 実務的に正しい。

- Phase 1: clamp_rate + information_loss のみ → 即座実装可能
- Phase 2: 加法パイプライン有効化後に全段の $\Delta R_i$ 実値 attribution

この段階分けは既存コード（`analyze_fill_logs.py` に `_load_executor_offset_stages()` L846, `_is_additive_execution()` L856）との整合性が高い。

**(D) 方式 B（fill_test 内組込み推論）の推奨** — 方向性は正しいが楽観的。後述。

### 1.2 補正が必要な主張

**(A) 「read_sidecar_signal を差し替えるだけ」の過小評価**

609# は方式 B を推す根拠として「ファイル読出しをインプロセス推論に差し替える」簡便さを強調しているが、610# が正しく指摘した通り feature contract の壁がある。

コード裏取りの結果:

| 項目 | 訓練側 (sac_retrain_scheduler) | 消費側 (orchestrator_mid_cycle) |
|------|------|------|
| 特徴量ソース | `btc_jpy_1m_full_registry_features.parquet` (L87) | **なし** — `SidecarSignal.directional_bias` (float) をそのまま使用 |
| 特徴量カラム | `cfg.feature_columns` → `env_config.feature_names` (L844-845) | `features_snapshot` は診断用のみ、cycle gate 未使用 |
| 推論 API | `model.predict(obs, deterministic=True)` (L1023) | `read_sidecar_signal_with_status()` でJSON読取 |
| 正規化 | 訓練 env 内部で処理 | **存在しない** |

**差し替え工数の実態**: JSON 読取 → `model.predict()` 呼出しへの変更自体は簡単だが、**obs を構築するための feature builder を fill_test 側に移植する**必要がある。これは「差し替え」ではなく「feature pipeline の二重化 or 共通化」という設計仕事。

**(B) Action Range 拡大の段階案 (0.20→0.30→0.50) の上限**

609# は ±0.5bps を上限と提案しているが、Avellaneda-Stoikov の $\delta^*\approx 400$ JPY ≈ 0.4bps という式展開はパラメータ推定に依存。$\kappa$（注文到着強度）の推定が fill_rate=30%, cycle=120s から $\kappa\approx 0.0025$ は粗い。

ただし**方向性としては 610# / 611# と一致**しており、±1.0bps は過大、段階的拡大が妥当という結論は全者一致。

### 1.3 609# 総評

三文書中**最も理論と実装のバランスが取れたレビュー**。TTL 真因の看破、Shapley 値の数理的根拠、Phase 分割の実務的判断は信頼に足る。唯一「方式 B の工数過小評価」が弱点だが、これは 610# によって適切に補正されている。

---

## §2 610# 深堀り

### 2.1 支持される主張

**(A) 「606# は設計/変更意図としては良いが、現 working tree の config 実態とは一致していない」**

今回の検証で確認:

- `fill_test.yaml:376` → `spread_anomaly_detector.enabled: false`
- `fill_test.yaml:387` → `micro_circuit_breaker.enabled: false`
- `fill_test.yaml:1186` → `entry_gate_enabled: false`

606# が書く「After: enabled=true」は **YAML への反映が未完了**。この発見は三文書中 610# のみが行っている。

**(B) 「Attribution は analyze_fill_logs.py 拡張でやるべき」**

コード確認で裏付け済み。`analyze_fill_logs.py` は既に:
- `_load_executor_offset_stages()` (L846): executor_offset_stages JSON パース
- `_is_additive_execution()` (L856): additive/multiplicative 判定
- 加法/乗法比較ロジック (L863-869)

を持つ。新規スクリプト乱立より既存分析器拡張が DRY 原則に合致する。

**(C) タスク分割 A/B の構想** — 交通整理として有効。ただし 611# が指摘した「境界責任者 C」の必要性は正しい補足。

### 2.2 補正が必要な主張

**(A) 行番号の精度問題**

| 引用 | 実際 | 差分 |
|------|------|------|
| `fill_test.yaml:375` (SAD enabled) | L376 | +1 |
| `fill_test.yaml:386` (MCB enabled) | L387 | +1 |
| `fill_config_parser.py:1162` | L1166 | +4 |

いずれもセクションヘッダと設定値の取り違え、または微小オフセット。論旨に影響はないが、実装文書としての信頼性を損なう。

**(B) §2.3 inventory-aware ceiling の記述が途中切れ**

610# の §2.3 は `side selection` / `offset adjustment` で文が切れている。inventory_skewing との二重化リスクの指摘自体は正しいが、結論が不明。

### 2.3 610# 総評

**最も実務的な交通整理文書**。config 実態との乖離を発見した点は高く評価。行番号精度と一部記述の不完全さが惜しい。

---

## §3 611# 深堀り

### 3.1 支持される主張

**(A) postonly_crossing_skip 爆発リスク — 三文書中最も鋭い実害指摘**

コード確認:
- `fill_cycle_executor.py:971-987`: BUY 時 `order_price >= best_ask` で crossing skip 発生
- `_postonly_crossing_streak` で連続発生を追跡（L1060-1064、3 連続で警告）
- **非対称 ceiling は既に実装済み**: `offset_ceiling_ratio_buy: 0.35`, `offset_ceiling_ratio_sell: 0.40` (`fill_config.py:691-693`)

608# の ±1.0bps 拡大がなぜ危険かを、**API 拒否 → Fill Rate 0%** という実害チェーンで説明した点は、全文書中最も具体的かつ説得力がある。

**(B) タスク分割における「境界責任者 C」の指摘**

610# の A/B 分割に対し「データコントラクト保証を誰がやるか」を突いた点は的確。feature contract 問題は609# も 610# も認識しているが、**担当を明示的にアサインしていない**。611# のこの指摘は組織設計として正しい。

### 3.2 過大主張・要補正箇所

**(A) 「ZMQ/Redis 等の IPC 通信で完全非同期化が必要」は現状にそぐわない**

611# §2.2 は「同期ブロックの危険」を理由に IPC 分離を主張するが、コード調査の結果:

- **ZMQ/Redis/multiprocessing.Queue**: コードベースに一切存在しない
- **現行 IPC**: 全てファイルベース（JSON/Parquet の tmp→rename アトミック書込み）
- **fill_test は設計上が同期ループ**（607# で確認済み）

現行アーキテクチャは**意図的にファイルベースの疎結合を選択**しており、「IPC に移行しなければ危険」は**現実の障害パターンで裏付けられていない**。SB3 の `model.predict()` は ~50ms であり、120s cycle の中では無視可能。

推論を fill_test 内部に組み込む場合、feature 計算で数百ms かかる可能性は理論上あるが、それは IPC の話ではなく**計算量削減**（feature 数の絞り込み、incremental 更新）で対処すべき。

**(B) 「イベント駆動推論（Hawkes Process 的励起）」は過剰設計**

30-60s の定周期推論が HFT では遅いという指摘は一般論として正しいが、**本システムは HFT ではなく MM（cycle ≈ 120s）**。Hawkes Process ベースのイベント駆動は、tick-by-tick 取引所向けの設計であり、Coincheck の REST/WS ベースの 120s cycle MM には**必要十分条件を大幅に超える**。

**(C) 提案 A「Queue Position 推定」— 面白いが Coincheck L2 の限界**

611# 自身が §4 で「L3 データなしでは完全な把握は不可能」と認めている。到着 Taker 量での近似推定は理論上可能だが:
- Coincheck の WS は全 trade を配信しない（REST で定期取得）
- 自注文の queue 位置を推定するには order_id 追跡 + 板変化の差分計算が必要
- 現段階のボトルネック（Sidecar 有効率、Attribution 不透明性）に対して**優先度が低い**

**(D) 提案 B「Dynamic Holding-Time Penalty」— 既存実装との重複**

**既に存在する inventory penalty 系**:
- `UnrealizedLossPenaltyCalculator`: 含み損ポジションに対する指数関数的ペナルティ `-(base^steps - 1)`
- `AsymmetricRewardScaler`: ロング/ショート/ニュートラルで異なる報酬倍率
- `OpportunityCostPenaltyCalculator`: アイドルポジションへのペナルティ

611# の「在庫保有時間に応じた指数関数的リスクペナルティ」は、`UnrealizedLossPenaltyCalculator` の `unrealized_loss_penalty_base^steps` と**概念的にほぼ同一**。新設する必要はなく、既存の base/max_steps パラメータのチューニングで対応可能。

**(E) 発生率 1.1% の根拠が不透明**

「先日のログ集計で 1.1% 発生」は自己引用であり、ログ解析結果へのリファレンスがない。定量的主張には出典が必要。

### 3.3 611# 総評

**着眼点の鋭さは三文書中最高**。postonly_crossing_skip の実害チェーンと境界責任者の必要性は、他の文書が見落としていた本質を突いている。一方、IPC/Hawkes Process/Queue Position/Holding-Time Penalty の4提案中3つは**既存アーキテクチャとの乖離または既存実装との重複**があり、そのまま採用はできない。

---

## §4 三文書横断の合意点と残存争点

### 4.1 全者が合意している事項

| 項目 | 609# | 610# | 611# | 確度 |
|------|------|------|------|------|
| Attribution は最優先取り組み | ✅ | ✅ | ✅ | **確定** |
| ±1.0bps 拡大は早すぎる | ✅ | ✅ | ✅ | **確定** |
| Sidecar 有効率向上が必要 | ✅ | ✅ | ✅ | **確定** |
| 段階的拡大が安全 | ✅ | ✅ | ✅ | **確定** |
| 608# の前提値に致命的誤認あり | ✅ | ✅ | ✅ | **確定** |
| 加法パイプラインの A/B 評価が先 | ✅ | ✅ | (未言及) | **ほぼ確定** |

### 4.2 意見が割れている事項

| 項目 | 609# | 610# | 611# | 判定 |
|------|------|------|------|------|
| in-process 推論の難易度 | 「差し替えるだけ」 | 「feature contract が必要」 | 「IPC 分離が必要」 | **610# が最も正確** |
| Attribution ツールの新設 vs 拡張 | 新設示唆 | `analyze_fill_logs.py` 拡張 | 新設示唆 | **610# の拡張案を支持** |
| Hawkes/イベント駆動 | 不要 | 言及なし | 必要 | **不要（120s cycle MM）** |
| SAC 報酬の再設計 | 言及なし | 言及なし | 新設提案 | **不要（既存 penalty で対応可）** |

### 4.3 誰も言及していない残存リスク

1. **feature pipeline 共通化の設計責任**: 方式 B（in-process inference）を実装するなら、training env の feature builder を live 側に移植する作業が発生する。この作業を A（実装）と B（数理）のどちらが負うかが未定義。→ 610# のタスク分割に明示的に追加すべき。

2. **加法パイプラインも ceiling を通る**: 609# §4.3 で指摘済みだが、610# / 611# では未言及。加法移行しても ceiling hit 頻度が「下がるだけで構造は同じ」点は A/B テスト設計に影響する。

3. **config parity 問題の根治**: 610# §2.1 が指摘した「effective config の機械的可視化」は、個別の enabled=true/false 反映より先に仕組み化すべき。

---

## §5 608# 著者（Gemini）に課すべきタスク

### 5.1 タスク選定の根拠

608# 著者の強み: 問題定義力、数理的フレーミング、アーキテクチャ設計構想
608# 著者の弱み: 現行コードの正確な把握、前提数値の検証、実装工数の見積もり

従って、**数理仕様・設計文書の策定**に特化させ、**コード変更・YAML 変更・テスト実装は担当外**とする。

### 5.2 課すべきタスク一覧

| # | タスク | 成果物 | 制約 |
|---|--------|--------|------|
| T1 | **608# 正誤表の作成** | 608# で使用した前提値と実値の対照表 | TTL, ceiling, max_boost_bps, Stage Max Mult, composite_risk の全項目 |
| T2 | **Attribution Analyzer 仕様書** | Phase 1 (clamp_rate, info_loss) の入出力仕様 | 既存 `analyze_fill_logs.py` への拡張として設計。新規スクリプト不可 |
| T3 | **σ-unit 正規化の数理仕様** | $\Delta R_i$ のスケーリング式、$\sigma_{baseline}$ の推定方法、既存 `RobustStats.asymmetric_ema()` との接続仕様 | Avellaneda-Stoikov との理論的整合性を明示 |
| T4 | **加法パイプライン A/B テスト設計** | 仮説、比較指標、サンプルサイズ、判定基準 | ceiling が加法にも適用される前提で設計。581# のトグル `experimental_additive_pipeline` を活用 |
| T5 | **Sidecar 方式 B（in-process inference）の feature contract 仕様** | training env と live 側で共有すべき特徴量リスト、正規化パラメータ、更新頻度 | 実装は担当 A。仕様策定のみ |
| T6 | **Action Range 拡大ロードマップ** | 0.20→0.30→0.50 の各段階での期待効果と撤退基準 | postonly_crossing_skip 発生率の閾値を含めること |

### 5.3 各タスクの責務境界

```
608# 著者 (Gemini)          実装担当 (Copilot)
─────────────────────────    ─────────────────────
T1: 正誤表 ────────→───────→ config 修正の根拠入力
T2: Attribution 仕様 ──→───→ analyze_fill_logs.py への実装
T3: σ-unit 数理仕様 ──→───→ offset_pipeline.py への組込み
T4: A/B テスト設計 ──→────→ テストハーネス実装 + 実行
T5: feature contract ─→───→ feature builder 共通化実装
T6: Range ロードマップ ─→──→ YAML 変更 + 段階デプロイ
```

---

## §6 まとめ

- **609#**: 最も信頼性の高い技術レビュー。feature contract の工数のみ過小評価。
- **610#**: 最も実務的な交通整理。config 実態乖離の発見は最重要。行番号精度に改善余地。
- **611#**: 着眼点は最も鋭い。postonly_crossing_skip の実害チェーンは全者にない独自価値。ただし IPC/Hawkes/報酬再設計は過剰設計であり既存実装との重複あり。
- **608# 著者**: T1-T6 のタスクに専念し、数理仕様の品質で貢献すべき。実装は触らない。

以上
