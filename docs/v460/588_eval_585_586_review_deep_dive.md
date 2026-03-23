# 588# 585#/586# レビュー評価 — コード検証に基づく深堀り

> **Date**: 2026-03-24  
> **Scope**: 585# (計画整合レビュー), 586# (セカンドオピニオン & Alpha 提案)  
> **Method**: 両レビューの各指摘を実コードおよび設定ファイルと突合し、正確性・妥当性・実装可能性を評価

---

## 0. エグゼクティブサマリ

| レビュー | 総合評価 | 正確性 | 実行可能性 |
|----------|---------|--------|-----------|
| **585#** | ★★★★☆ 高い — ほぼ全指摘が妥当 | 主張の 90% が実コードと一致 | 推奨アクションの大半が即実行可能 |
| **586#** | ★★★☆☆ 着眼点は鋭いが精度にばらつき | 主張の 60% が正確、40% に過大/事実誤認 | 2 件が即実行可能、2 件は中長期 |

585# は堅実な「監査レビュー」として信頼度が高く、推奨事項もほぼそのまま採用できる。
586# は金融工学の視点から示唆に富むが、コードベースの既存実装を過少評価している箇所があり、提案の採用には個別検証が不可欠。

---

## 1. 585# 評価 — 計画整合レビューと盲点補完

### 1.1 §2.1「ph6 は 000# に存在しない」— ✅ 正確・妥当

**検証結果**: `000_ph0_plan_project_proposal.md` §2 を確認。正式フェーズは `ph0`–`ph5` + `ph3.1` のみ。`ph6` は一切定義されていない。

**評価**: 584# の内容は paper trading 中に見つかった Execution 不全の是正計画であり、性質としては `ph5 remediation` に近い。585# の指摘は完全に正しい。

**対応**: 584# を `ph5.5 remediation` に再ラベルするか、000# Appendix A に ph6 定義を追記するかの二択。前者がローコスト。

### 1.2 §2.2「実装完了と live 実証の混同」— ✅ 正確・重要

**検証結果**:
- `spread_capture_bps` / `adverse_selection_cost_bps`: FillRecord スキーマ (`fill_quality.py` L75) に存在し、`pnl_measurer.py` L149 に計算ロジックが確認できる
- 計算成立条件: `fill_price`, `mid_at_fill`, `mid_30s_after` の 3 値全てが non-None のとき
- 585# が報告した「0/90 fills で未記録」は、`fill_price` が `_build_fill_record` 呼び出し時に正しく渡されていない可能性が高い

**評価**: コード構造は正しいが、fill_price のパススルーに断線があるためlive JSONL に値が出ない。585# の「コード反映済み・live 可観測性は要再確認」という再分類は適切。

### 1.3 §2.3「A/B テレメトリが信用できない」— ✅ 正確・クリティカル

**検証結果 — 三重切断の確認**:

1. **`fill_cycle_executor.py`**: `_build_fill_record(...)` 呼び出し箇所で `execution_additive_enabled` を明示的に渡していない  
2. **`fill_quality.py` FillRecord**: `execution_additive_enabled` フィールドが定義されていない  
3. **`fill_record_builder.py`**: L357 で引数として受け取り L401 で payload に入れるが、呼び出し元から値が来ないため常に None → `_sanitize_fill_record_fields` が除去

**分析スクリプト側**:
```python
# analyze_fill_logs.py L883-890
def _is_additive_execution(record: dict[str, Any]) -> bool:
    _explicit = record.get("execution_additive_enabled")
    if _explicit is not None:
        return bool(_explicit)
    _stages = _load_executor_offset_stages(record)
    return bool(_stages and "tox_buffer" in _stages)
```
二重基準: (1) `execution_additive_enabled` フィールド → 常に None、(2) `executor_offset_stages` に `tox_buffer` 存在 → additive pipeline JSON にのみ存在。結果として **フォールバック基準がかろうじて機能するが、第一基準は完全に死んでいる**。

**評価**: 585# の指摘は本質的に正しく、584# P1 の A/B 比較は現状のテレメトリでは信頼性が低い。587# Codex Task A で修復予定。

### 1.4 §2.4「hot-reload で A/B が切り替わらない」— ✅ 正確

**検証結果**: `config_hot_reload.py` の `_HOT_RELOADABLE_FIELDS` を確認:
- **含まれている**: `execution_additive_enabled`（但し前述の通りテレメトリ用で実ロジック分岐には無関係）
- **含まれていない**: `experimental_additive_pipeline`, `edrc_alpha`, `edrc_beta`, `edrc_c_base`, `edrc_hard_cap`, `entry_gate_enabled` 等すべて

**評価**: 584# の A/B 実施にはプロセス再起動 + run_id 分離が必須。585# の推奨は正しい。587# Codex Task D で拡張予定。

### 1.5 §2.5「P6 は neutral fallback ではなく fresh/stale/error 安定化」— ✅ 正確・鋭い

**検証結果**: `cache/sidecar_signal.json` を確認:
```json
{
  "model_version": "sac_sidecar_20260323_0823",  // ≠ "neutral"
  "confidence": 0.107,                            // 非常に低い
  "regime_hint": "",                              // 空
  "timestamp": "2026-03-23T08:33:00..."           // 約24h前 → stale
}
```

584# は `model_version: "neutral"` 前提で P6 を定義しているが、実際は neutral ではない。問題は「シグナルが無い」ことではなく「fresh で安定供給されない」こと。

**評価**: 585# の再定義は実態に即しており、P6 の定義修正が必要。000# が SAC を **Sidecar** と位置付けていることからも、P6 をクリティカルパスの最上位に置く論拠は弱い。

### 1.6 §2.6「P2 Smart Preflight の設計スケッチが薄い」— ✅ 正確

**検証結果**:
- `get_inventory_skew_score()`: 存在しない
- `smart_preflight_enabled` / `preflight_skip_inv_threshold`: 未定義
- 584# P2 pseudo-code は存在しない API に依存

**評価**: P2 は「小変更」ではなく新規設計タスク。585# の「観測追加からまず始めよ」という推奨は実践的。

### 1.7 §3.4「ベースライン数値の食い違い」— ✅ 正確・重要

**検証結果**:

| 項目 | 584# | 585# 再集計 |
|------|------|------------|
| 分母 | n=3,869 | n=6,034 |
| Fill rate | 25.2% | 26.5% |
| Buy avg_pnl | -0.28 bps | -0.44 bps |
| Sell avg_pnl | +0.21 bps | -0.18 bps |
| git_sha | 不明 | unique=60 |

**原因分析**: `analyze_fill_logs.py` には attempted/clean/raw のフィルタモード切替機能が**存在しない**。差異は CLI の `--git-sha` / `--run-id` フィルタの有無で発生。585# 全データ (60 SHA 混在) では旧コードの悪い成績が含まれ PnL が悪化する。

**評価**: 584# のベースラインは特定条件下のスナップショットであり、再現条件の明記が必要。585# の指摘は妥当。

### 1.8 §3.5「二枚看板の混線」— ✅ 正確

**検証結果**:
- **`experimental_additive_pipeline`**: 実際のロジック分岐フラグ（`fill_config.py`, `offset_pipeline.py`, `multiplicative_pipeline.py`）
- **`execution_additive_enabled`**: テレメトリラベル意図で定義されたが三重切断により死んでいる
- YAMLにも `execution_additive_enabled` は存在しない（設定項目ではない）

**評価**: 587# Codex Task E で整理予定。

### 1.9 §3.6「additive_base_bps は未使用」— ✅ 正確

**検証結果**: `fill_config.py` L364, `fill_config_parser.py` L193, `fill_test.yaml` L702 に定義があるが、`offset_pipeline.py` / `multiplicative_pipeline.py` での参照は**ゼロ**。

**評価**: 完全なデッドコンフィグ。587# Codex Task B で削除予定。

### 1.10 585# 総合評価

| 指摘カテゴリ | 件数 | 正確 | 部分的 | 不正確 |
|-------------|------|------|--------|--------|
| 強い補正（§2） | 6 | 6 | 0 | 0 |
| 中重要度（§3） | 6 | 5 | 1 | 0 |
| 合計 | 12 | 11 | 1 | 0 |

部分的：§3.1 の「True Additive」表現批判は方向として正しいが、実際の 581# / 582# 本文は Execution 層限定を明示しており、過度に強い批判ではある。

**総合判定**: 585# は高信頼度の監査レビューであり、推奨アクションの大半は即採用可能。特に §2.3 (テレメトリ三重切断) と §2.5 (P6 再定義) は実コードに裏付けられた重要な発見。

---

## 2. 586# 評価 — セカンドオピニオンと Alpha 提案

### 2.1 §1.1「eDRC にウィンザライゼーションが欠如」— ⚠️ 方向は正しいが事実誤認あり

**586# の主張**: 「eDRC の入力（sigma, OFI）にクリッピングが無く、exp() がスパイクする」

**検証結果**:

```python
# fill_config.py L385-389
ceiling_dynamic = self.edrc_c_base * exp(
    self.edrc_alpha * sigma + self.edrc_beta * adverse_ofi
)
return min(ceiling_dynamic, self.edrc_hard_cap)
```

586# が「`get_robust_inputs()` に呼び出し元がない」と述べているが、**これは事実誤認**。

実際のコード:
```python
# offset_pipeline.py L268-273
_robust_sigma, _robust_ofi = self._maker_price.get_robust_inputs(side)
_fc_ceil = self.config.resolve_offset_ceiling(
    side, utc_hour=current_utc_hour(),
    sigma=_robust_sigma, adverse_ofi=_robust_ofi,
)
```

```python
# multiplicative_pipeline.py L234-239 (同一パターン)
_robust_sigma, _robust_ofi = self._maker_price.get_robust_inputs(side)
_fc_ceil = self.config.resolve_offset_ceiling(...)
```

**両パイプラインで `get_robust_inputs()` は呼ばれている**。つまり:
- `sigma` は `RobustStats.asymmetric_ema()` で平滑化済み（上方リスクに敏感、下方は鈍感）
- `adverse_ofi` は `RobustStats.median_filter_fast()` でスパイク耐性あり
- 出力側は `min(ceiling_dynamic, edrc_hard_cap)` でクリップ

**ただし**: 入力側の平滑化は winsorization（パーセンタイルクリッピング）ではなく **指数移動平均 + 中央値フィルタ**。2σ超の外れ値は減衰するが完全には除去されない。加えて、現在 `edrc_alpha=0.0`, `edrc_beta=0.0`（YAML確認済み）なので **eDRC は実質無効** — `ceil = c_base * exp(0) = c_base = 0.40` 固定。

**評価**:

| 観点 | 586# 主張 | 実態 |
|------|----------|------|
| get_robust_inputs 未呼出 | ❌ 不正確 | 両パイプラインで呼出済み |
| 入力にクリッピングが無い | △ 部分的に正確 | 平滑化はあるがWinsorization（硬クリップ）は無い |
| exp() スパイクのリスク | △ 理論的には妥当 | 現時点ではα=β=0のため実害なし |
| 対策の必要性 | ✅ 将来有効化時に必要 | α, β を非ゼロにする前にクリップ追加すべき |

**対応方針**: eDRC を有効化する P4 フェーズの前に、`resolve_offset_ceiling` 内で `sigma = min(sigma, 5.0)`, `adverse_ofi = min(adverse_ofi, 50.0)` 等の入力クリップを追加する。587# Codex Task C で対応予定だが、`get_robust_inputs()` が既に呼ばれている事実を反映して、タスク内容の修正が必要。

### 2.2 §1.2「Toxicity × Liquidity 交差項の欠如」— ⚠️ 理論的には正しいが優先度は低い

**586# の主張**: 「Toxicity と Liquidity は動的に相関するため、独立加算ではなくクロスターム（$OFI \times \sigma$）を含めるべき」

**検証結果**:
- 582# additive pipeline: `tox_buffer` と `liq_buffer` を独立加算
- 585# multiplicative pipeline: `toxicity_offset` と別のステージが独立乗算
- 交差項の実装: なし

**評価**: Choi et al. 型のスプレッド設計は理論的に美しいが、現状は eDRC 自体が無効 (α=β=0) かつ additive pipeline も `experimental_additive_pipeline.enabled=false` で本番未使用。理論的整備より先にパイプライン有効化と A/B データ収集が優先。

**判定**: ★★☆ 中長期バックログ。P4 以降で交差項の導入を検討。

### 2.3 §2.1「グローバル取引所リードシグナル」— ⚠️ 既存実装を過小評価

**586# の主張**: 「外部 WebSocket に接続し Lead-Lag シグナルで逆選択を回避せよ」

**検証結果**:
- **BitFlyer REST**: 接続済み、`cross_venue_lead_lag` ガード実装済み
  - `cross_venue_lead_lag.enabled = true` (fill_test.yaml)
  - veto 機能: `veto_enabled = true`, `veto_threshold_bps = 8.0`
  - basis correction, favorable tighten, preemptive sell kill: 全て `true`
- **Binance REST**: ヒストリカルデータ取得のみ
- **WebSocket**: 外部取引所への WebSocket 接続は**未実装**
- **Bybit**: 未接続

**評価**: 586# は「ゼロからの提案」のように書いているが、**BitFlyer 連携は既に稼働中**。ただし REST ベースのため遅延が大きく、WebSocket 化による低遅延化は改善の余地あり。Binance/Bybit WebSocket 追加は中長期課題として妥当。

**586# セルフレビュー（§3）での自己補正**: API Rate Limit について言及しており、この自己認識は適切。

**判定**: ★★★ 既存基盤の拡張として段階的に進めるべき。WebSocket 化は ph6/7 バックログ。

### 2.4 §2.2「Micro-Price 導入」— ✅ 方向は正しいが「1行変更」は過大表現

**586# の主張**: 「(BestBid+BestAsk)/2 の Mid-Price から Micro-Price へ切り替えるだけで AS Cost が劇的に減少」

**検証結果**:

**既存実装**: `compute_microprice_bias_bps()` は L1-L5 加重 Gatheral (2018) multi-level microprice として既に実装済み (`maker_price.py` L570-615)。

**現在の用途**:
- FillRecord のテレメトリ記録用：`microprice_bias_bps` フィールドに記録
- Side 選択時の参考値として `side_selector.py` で使用

**基準価格の実態**:
```python
# maker_price.py L1007
mid_price = (best_bid + best_ask) / 2.0  # 単純算術平均
```

**切り替えの実行可能性**:
- L1007 を `microprice` に置換するのは技術的には可能
- ただし Microprice はスプーフィング（見せ板）に脆弱（586# §3 で自己認識済み）
- Zaif の板の品質（薄板・見せ板の頻度）の事前検証が必要
- 「劇的に減少」は Zaif の市場特性次第で過大な期待

**評価**:

| 観点 | 586# 主張 | 実態 |
|------|----------|------|
| Micro-Price 未実装 | ❌ 不正確 | 366# M1 で L1-L5 加重版が実装済み |
| 基準価格が Mid | ✅ 正確 | L1007 で (bid+ask)/2 |
| 1行変更で導入可能 | △ 技術的には可能だが検証必要 | Spoofing 耐性の事前調査が必須 |
| AS Cost 劇的減少 | △ 市場依存 | Zaif のような薄板市場では効果不確定 |

**判定**: ★★★☆ P1 A/B 完了後に `microprice_basis_enabled` フラグ付きで試験導入の価値あり。ただし段階的に。

### 2.5 §2.3「Hawkes 過程による Toxicity 先行検知」— △ 理論的に正しいが実装コスト高

**586# の主張**: 「VPIN/ATR は集計ウィンドウ依存で検知が遅い。Hawkes 過程で自己励起的な連鎖を検知せよ」

**検証結果**:
- Hawkes 過程の実装: コードベースに**存在しない**
- 依存関係: `tick_tock` (Hawkes ライブラリ) 等の追加パッケージが必要
- リアルタイム推定: Intensity 計算は逐次更新可能だが、パラメータ (μ, α, β) の推定に MLE or EM が必要
- 既存の代替: `FastFillDefense` 機構（高速連続約定検知）が部分的に similar role を担っている

**評価**: 理論的には魅力的だが、「短期間での高収益性」というプロジェクト大義を考えると、実装+検証コストに対する ROI が低い。既存の FastFillDefense / VPIN / OFI の改善が先。

**判定**: ★★☆ 中長期研究課題。

### 2.6 §2.4「Regime-Switching SAC Ensemble」— △ 部品は存在するが接続層がない

**586# の主張**: 「HMM で 3 ステートに分類し、各環境専用の SAC モデルを動的切替」

**検証結果**:
- **RegimeClassifier**: `ztb/ml/` に存在
- **EnsemblePredictor**: `ztb/ml/` に存在
- **Regime 別モデル specialization 定義**: 存在する
- **Regime → Model の推論時切替ロジック**: **未実装**
- 586# 自己認識（§3）: 「複雑性が爆発的に増大し、開発コスト（デバッグコスト）が甚大」と自覚

**評価**: 586# が §3 で自己修正しているとおり、Ensemble 化は overkill。現行 sidecar は fresh/stale/error の安定化が先。

**判定**: ★★☆ 長期アーキテクチャ検討課題。000# の SAC=Sidecar 思想とは整合するが、現段階では premature。

### 2.7 586# 総合評価

| 提案 | 正確性 | ROI | 既存基盤考慮 | 即実行可否 |
|------|--------|-----|-------------|-----------|
| §1.1 eDRC Winsorization | △ 部分的 (呼出元の事実誤認) | ★★★ | get_robust_inputs 見落とし | P4 前に対応 |
| §1.2 交差項 | ✅ | ★★ | — | 中長期 |
| §2.1 Lead-Lag | △ (既存実装見落とし) | ★★★★ | BitFlyer 既接続 | WebSocket 化は中長期 |
| §2.2 Micro-Price | △ (実装済を見落とし) | ★★★ | compute_microprice_bias_bps 既存 | P1 後に試験 |
| §2.3 Hawkes | ✅ | ★★ | FastFillDefense が部分代替 | 長期 |
| §2.4 Ensemble RL | ✅ | ★ | 部品は存在 | 長期 |

**総合判定**: 金融工学の理論的深度は高いが、**コードベースの既存実装を十分に調査していない**。`get_robust_inputs()` の呼出元見落とし、`compute_microprice_bias_bps()` の既存実装見落とし、BitFlyer 連携の既存実装見落とし — いずれもコードリーディング不足に起因。

586# の最大の価値は §3 のセルフレビューにあり、提案の限界を著者自身が正確に認識している点は評価できる。

---

## 3. 両レビュー横断の発見事項

### 3.1 585# と 586# で重複する指摘

| テーマ | 585# | 586# | 合意度 |
|--------|------|------|--------|
| テレメトリの断線 | §2.3 三重切断 | (直接言及なし) | 585# のみ |
| eDRC の脆弱性 | (直接言及なし) | §1.1 Winsorization | 586# のみ |
| 実装 vs 実証の混同 | §2.2 | §1 総評冒頭 | 高い |
| hot-reload の限界 | §2.4 | (直接言及なし) | 585# のみ |
| 基準価格改善 | (直接言及なし) | §2.2 Micro-Price | 586# のみ |

**所見**: 両レビューの視点は相補的。585# は「内部品質・運用規律」、586# は「外部理論・市場効率」に焦点。両方を組み合わせることで blind spot を最小化できる。

### 3.2 両レビューが見落としている問題

1. **`fill_price` の FillRecord 未到達問題**: `spread_capture_bps` が 0/90 fills で未記録なのは、PnlMeasurement の計算条件 (`fill_price is not None and mid_at_fill > 0`) が満たされていないため。`_build_fill_record` で `fill_price` が渡されているかの直接確認が両レビューとも不足。

2. **`sidecar_signal.json` の confidence 低値**: `confidence: 0.107` は実質的にモデル予測が「賭けに値しない」レベル。これは fresh/stale/error 以前の **モデル品質** の問題であり、retrain パイプラインだけでなく報酬設計 (reward shaping) の見直しが必要な可能性がある。

3. **YAML に `entry_gate_enabled` が未定義**: 585# §2.6 でも 586# でも見落とされている。555# CalibrationMap を有効化するフラグ自体が YAML に存在しない。

---

## 4. 即実行可能なアクション一覧

### 4.1 即日対応（コスト低・リスク低）

| # | アクション | 根拠 | 工数 |
|---|-----------|------|------|
| A1 | 584# を `ph5.5 remediation` に再ラベル | 585# §2.1 | 文書のみ |
| A2 | 584# ベースライン数値にフィルタ条件明記 | 585# §3.4 | 文書のみ |
| A3 | 584# P6 を「neutral 解消」→「fresh/stale/error 安定化」に再定義 | 585# §2.5 | 文書のみ |
| A4 | fill_test.yaml の `experimental_additive_pipeline` 現状値を 584# に明記 | 585# §2.4 | 文書のみ |

### 4.2 Short-term（Codex 587# に委託済み）

| # | タスク | Codex 587# Task |
|---|--------|----------------|
| B1 | `execution_additive_enabled` 三重切断修復 | Task A |
| B2 | `additive_base_bps` デッドコンフィグ削除 | Task B |
| B3 | eDRC 入力クリップ追加 | Task C (※ get_robust_inputs 呼出済みの事実を反映して修正必要) |
| B4 | hot-reload スコープ拡張 | Task D |
| B5 | 二枚看板の整理 | Task E |

### 4.3 Medium-term（P1 A/B 完了後）

| # | アクション | 根拠 |
|---|-----------|------|
| C1 | `microprice_basis_enabled` フラグで Micro-Price 基準価格の試験 | 586# §2.2 |
| C2 | V4 成功基準の profit-first 指標への差替え | 585# §3.2 |
| C3 | P2 Smart Preflight の観測追加（skip 影響の可視化） | 585# §2.6 |
| C4 | Lead-Lag WebSocket 化の設計 | 586# §2.1 |

### 4.4 Long-term（バックログ）

| # | アクション | 根拠 |
|---|-----------|------|
| D1 | Toxicity×Liquidity 交差項 | 586# §1.2 |
| D2 | Hawkes 過程の研究プロトタイプ | 586# §2.3 |
| D3 | Regime-Switching SAC Ensemble | 586# §2.4 |

---

## 5. 587# Codex プロンプトへの修正事項

本評価で判明した事実に基づき、587# Codex プロンプトに以下の修正が必要:

### Task C (eDRC Winsorization) の修正

**変更前**: 「`get_robust_inputs()` が呼ばれていないためWinsorization が未適用」  
**変更後**: 「`get_robust_inputs()` は `offset_pipeline.py` L268 および `multiplicative_pipeline.py` L234 で呼出済み。平滑化（asymmetric EMA + median filter）は適用されているが、exp() への入力値の硬クリップ（Winsorization）が無い。`resolve_offset_ceiling` 内で `sigma = min(sigma, σ_cap)`, `adverse_ofi = min(adverse_ofi, ofi_cap)` を追加すること」

---

## 6. 結論

### 585# — 採用度: 高い
- 12 指摘中 11 件がコード検証で裏付けられた
- 特に テレメトリ三重切断 (§2.3)、ph6 位相問題 (§2.1)、P6 再定義 (§2.5) は即対応すべき
- 推奨アクションの実行順序（§5）も現実的

### 586# — 採用度: 選択的
- 金融工学の理論的深度は高く、視野拡大に貢献
- ただしコードベース既存実装の調査不足が目立つ（3 件の見落とし）
- 即効性のある提案は §1.1 (eDRC クリップ) と §2.2 (Micro-Price 試験) の 2 件
- §3 のセルフレビューは誠実であり、提案の限界認識は適切

### 次の一手
585# §5 の推奨順序に概ね従い:  
**A1-A4 文書修正 → B1-B5 Codex 587# 実行 → telemetry parity 確認 → P1 A/B 実施**
