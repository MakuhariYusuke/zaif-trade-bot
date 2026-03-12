# 119# v460 始動統合方針: 118# レビュー評価 + スクリプト・設定改革

| 項目 | 内容 |
|------|------|
| 対象 | 117# (命名・00# 設計), 118# (追加レビュー), v459 スクリプト構造反省 |
| 目的 | v460 の `docs/`, `scripts/`, `configs/` 三位一体での設計方針確定 |
| 依拠 | 117# §3–§4, 118# §2–§3, v459 run_phase_c.py (1,277 行) |

---

## §1 118# レビューへの評価

### §1.1 全面採用 (4 件)

| # | 118# 指摘 | 採用理由 |
|---|----------|---------|
| 1 | Gate 体系を名前付き `G0-data / G1-info / G1.1-exec / G2-train / G3-pnl / G4-live` に | 番号だけでは v459 との混同が発生する。名前付きで意味固定。117# §4.2 §3 を差し替え |
| 2 | K2 結論を過信せず multi-target 化 (h1/h5/h15, direction + magnitude + volatility) | K2 は「次足符号 × 8 特徴」のみの判定。horizon 拡張は v460 G1-info で必須 |
| 3 | Run manifest (JSONL) を実装レベルで強制 | v459 は文書での再現記録しかなく、再検証が事実上不可能だった |
| 4 | `PhX` の表記を小文字 `phX` に統一 | 117# 内で混在していた。運用摩擦を排除 |

### §1.2 条件付き採用 (2 件)

| # | 118# 指摘 | 条件 |
|---|----------|------|
| 5 | `G1.1-exec` (maker fill/latency) | v460 Phase 0 では定義のみ。実測は Zaif API 接続後（Phase 2 以降） |
| 6 | `001` を `data_contract` に変更 | 000 = proposal, 001 = data_contract, 002 = gate_spec の順序は妥当。ただし `architecture` は 003 以降に後退でなく gate_spec 内 §4 に含める |

### §1.3 却下 (1 件)

| # | 118# 指摘 | 却下理由 |
|---|----------|---------|
| 7 | 「ファイル名は英語、本文は日本語可」への緩和 | v459 で `39_review_response_修正計画.md` が規則破壊した前例。**ファイル名英語は堅持**。本文は元々制約していない（117# は「ファイル名の英語」のみ規定） |

### §1.4 118# §5「最初の 7 文書」への修正

118# 提案:
```
000_ph0_plan_project_proposal.md
001_ph0_plan_data_contract.md
002_ph0_plan_gate_spec.md
003_ph0_plan_experiment_manifest.md
004_ph0_rpt_g0_data_validation.md
005_ph0_rpt_g1_feature_info_test.md
006_ph0_rev_000_ref000.md
```

修正版 (スクリプト・設定統合を反映):
```
000_ph0_plan_project_proposal.md       ← 00# 本体 (150行以内)
001_ph0_plan_data_contract.md          ← データ仕様 (板/約定/OHLCV)
002_ph0_plan_gate_spec.md              ← G0–G4 詳細 (117# §4.2 §3 の拡張)
003_ph0_plan_runner_and_config.md      ← 本 119# §3–§4 の転記
004_ph0_rev_000.md                     ← 00# レビュー
```

Gate 実験結果 (004+) は**実験実施後に**発番する。事前に番号を予約しない。

---

## §2 v459 スクリプト構造の反省

### §2.1 数値的事実

| 指標 | 値 |
|------|-----|
| スクリプト総数 | 45 本 |
| 総行数 | 12,458 行 |
| 最大ファイル | `run_phase_c.py` — **1,277 行** |
| 500 行超 | 5 本 (run_phase_c, run_day8_scale_deconfounding, run_ab_reward_experiments, run_day10_comprehensive, run_day7_causal_separation) |
| 実験定義の所在 | Python dict リテラルとしてスクリプト内にハードコード |

### §2.2 God Object: run_phase_c.py の解剖

`run_phase_c.py` (1,277 行) の責務:

1. **実験定義** (L114–400): 30+ 個の config dict を `get_experiment_configs()` に列挙
2. **データ読込・前処理** (L400–500): Parquet ロード、train/eval 分割
3. **SAC 訓練** (L500–700): SACTrainer の呼出し + チェックポイント周りのグルー
4. **評価** (L700–900): IC 計算、Gate 2 KPI、統計検定
5. **結果保存** (L900–1050): JSON 書出し、ログ出力
6. **バッチ実行** (L1050–1277): argparse + 実験ループ

**問題**: 1 と 6 以外の責務 (2–5) は全スクリプトでほぼ同一コードのコピペ。新実験のたびにスクリプトを丸ごと新作していた。

### §2.3 根本原因

> **実験定義が Python コード内にある限り、config 差し替えだけでは実験を変更できず、スクリプト自体に手を入れる必要が生じる → スクリプト肥大化 → コピペ増殖**

v459 の `configs/v459/base/config.yaml` は存在したが、**実験差分は config に書かれずスクリプト内の dict で管理**されていた。config ファイルの意味が形骸化していた。

---

## §3 v460 スクリプトアーキテクチャ

### §3.1 設計原則

| 原則 | 説明 |
|------|------|
| **1 ランナー** | `scripts/v460/run_experiment.py` — 唯一の実験実行スクリプト。200 行以内 |
| **N 設定ファイル** | `configs/v460/experiments/*.yaml` — 実験ごとに 1 ファイル |
| **差分のみ記述** | 各実験 YAML は `base.yaml` からの差分 (override) のみ |
| **ランナー不変** | 新実験追加でランナーを編集しない。YAML だけ追加する |

### §3.2 ディレクトリ構成

```
configs/v460/
├── base.yaml                          # 全実験の共通ベース設定
├── experiments/
│   ├── g1_xgb_h1_direction.yaml       # G1-info: XGBoost h1 符号
│   ├── g1_xgb_h5_direction.yaml       # G1-info: XGBoost h5 符号
│   ├── g1_xgb_h15_magnitude.yaml      # G1-info: XGBoost h15 変動幅
│   ├── g2_sac_seed42.yaml             # G2-train: SAC 学習安定性
│   ├── g2_sac_seed123.yaml
│   └── ...
└── gate_thresholds.yaml               # Gate 判定閾値 (G0–G4)

scripts/v460/
├── run_experiment.py                  # 唯一のランナー (200行以内)
├── run_gate_check.py                  # Gate 判定ユーティリティ
└── lib/                               # v460 固有のヘルパー (必要時のみ)
    ├── data_loader.py                 # データ読込・分割
    ├── evaluator.py                   # 評価・KPI 算出
    └── manifest.py                    # run manifest 記録
```

### §3.3 ランナーの責務分離

```python
# scripts/v460/run_experiment.py (概念設計)
def main():
    args = parse_args()  # --config path/to/experiment.yaml [--seed N]
    
    # 1. base.yaml + experiment.yaml をマージ
    config = load_and_merge(args.config)
    
    # 2. manifest 記録開始
    manifest = Manifest.start(config, data_hash=hash_data(config.data_path))
    
    # 3. 実行 (config.task に応じて分岐)
    if config.task == "feature_info":
        result = run_feature_info_test(config)     # G1-info
    elif config.task == "sac_train":
        result = run_sac_training(config)           # G2-train
    elif config.task == "backtest":
        result = run_backtest(config)               # G3-pnl
    
    # 4. 結果保存 + manifest 完了
    save_result(result, config)
    manifest.finish(result)
```

### §3.4 実験 YAML の例

```yaml
# configs/v460/experiments/g1_xgb_h5_direction.yaml
_base: "../base.yaml"
_description: "G1-info: XGBoost h5 direction prediction"
_gate: "G1-info"

task: "feature_info"

features:
  source: "orderbook"   # v459: "ohlcv_derived" → v460: "orderbook"
  columns: ["bid_ask_spread", "depth_imbalance", "trade_flow", "vwap_deviation"]

target:
  horizon: 5            # h5 (5分先)
  type: "direction"     # 符号 (1: up, 0: down)

model:
  type: "xgboost"
  params:
    n_estimators: 200
    max_depth: 6

evaluation:
  method: "blocked_time_split"
  n_splits: 5
  oos_ratio: 0.2
```

### §3.5 v459 との対比

| 観点 | v459 | v460 |
|------|------|------|
| 実験定義の所在 | Python dict (スクリプト内) | YAML ファイル (configs/) |
| 新実験追加 | スクリプトに dict 追記 or 新ファイル作成 | YAML 1 ファイル追加のみ |
| ランナー数 | 45 本 (うち 13 本が run_*) | **1 本** (run_experiment.py) |
| 最大行数 | 1,277 行 | **200 行以内** |
| 再現性 | 文書 (ドキュメント) に記述 | manifest.jsonl に自動記録 |
| base 設定 | 形骸化 (使われず) | **全実験の起点** として機能 |

---

## §4 configs/v460 の設計

### §4.1 base.yaml の構成

```yaml
# configs/v460/base.yaml
version: "4.6.0"

data:
  path: "data/btc_jpy_1m_v460_features.parquet"
  train_end_index: null   # 実験 YAML で指定必須
  features: []            # 実験 YAML で指定必須

training:
  algorithm: "sac"
  total_timesteps: 50000
  seed: 42
  sac_hyperparameters:
    learning_rate: 0.0003
    buffer_size: 100000
    batch_size: 256
    gamma: 0.80
    ent_coef: "auto"

environment:
  continuous_to_discrete_threshold: 0.70
  min_holding_period: 3

execution:
  fee_model: "maker_only"    # v460 前提: maker 0%
  maker_fee_rate: 0.0
  taker_fee_rate: 0.001      # taker は禁止だが定義は残す

evaluation:
  method: "blocked_time_split"
  n_splits: 5
  oos_ratio: 0.2
  metrics: ["ic", "accuracy", "pf", "sharpe", "max_dd", "win_rate"]
```

### §4.2 差分 Override 規則

1. 実験 YAML は `_base` で base.yaml を指定
2. 記載したキーのみ上書き。記載しないキーは base を継承
3. `features` と `train_end_index` は**必須**（base で null のため、未指定はエラー）
4. `_gate` フィールドで対応 Gate を明示 → manifest に自動記録

### §4.3 gate_thresholds.yaml

```yaml
# configs/v460/gate_thresholds.yaml
# Gate 判定閾値 (00# §3 の実装表現)
G0-data:
  data_hash_match: true
  feature_count_min: 4
  nan_ratio_max: 0.01

G1-info:
  ic_min: 0.02            # OOS で 1 horizon 以上
  accuracy_min: 0.51
  significant_folds_min: 2  # 5 中 2 以上

G1.1-exec:
  fill_rate_90p_min: 0.90
  cancel_ratio_max: 0.30

G2-train:
  positive_seed_ratio: 0.75  # 4 中 3 以上 gross > 0

G3-pnl:
  pf_min: 1.05
  avg_gross_gt_avg_fee: true

G4-live:
  paper_trading_days_min: 7
  circuit_breaker_test: true
```

---

## §5 Run Manifest 仕様

### §5.1 スキーマ

```jsonl
{
  "run_id": "v460_g1_xgb_h5_20260214_143022",
  "config_path": "configs/v460/experiments/g1_xgb_h5_direction.yaml",
  "config_hash": "sha256:a1b2c3...",
  "data_hash": "sha256:d4e5f6...",
  "git_commit": "abc1234",
  "gate": "G1-info",
  "seed": 42,
  "started_at": "2026-02-14T14:30:22+09:00",
  "finished_at": "2026-02-14T15:12:45+09:00",
  "status": "completed",
  "metrics": {"ic_h5": 0.031, "accuracy": 0.523, "pf": null},
  "gate_result": "PASS",
  "artifacts": ["results/v460/g1_xgb_h5_seed42.json"]
}
```

### §5.2 保存先

`results/v460/manifest.jsonl` — 追記専用。全実験の時系列ログ。

### §5.3 自動記録

`run_experiment.py` が実行の開始・終了時に自動で manifest を追記。手動記録は禁止。

---

## §6 v459 → v460 ディレクトリ対比（最終形）

```
v459 (実態)                          v460 (方針)
─────────────────────────────────    ─────────────────────────────────
docs/v459/                           docs/v460/
  117 文書 (上限なし)                   40 文書以内
  00#–116# (2桁混在)                   000–039 (3桁, phX_type 必須)
  命名規則なし                          NNN_phX_type_subject.md

scripts/v459/                        scripts/v460/
  45 本 (計 12,458 行)                 run_experiment.py (200行)
  run_phase_c.py (1,277 行)            run_gate_check.py (100行)
  実験定義=Python dict                  lib/ (必要時のみ)

configs/v459/                        configs/v460/
  base/config.yaml (形骸化)            base.yaml (全実験の起点)
                                      experiments/*.yaml (差分のみ)
                                      gate_thresholds.yaml

prompts/ (なし)                      prompts/v460/ (AI レビュー用)

experiments/ (なし)                   experiments/v460/ (日次ログ)

results/ (雑多)                      results/v460/
                                      manifest.jsonl (自動記録)
                                      *.json (実験結果)
```

---

## §7 批判的検討

### §7.1 ランナー 1 本で全 Gate を賄えるか

G1-info (XGBoost) と G2-train (SAC) は学習器が異なる。1 つの `run_experiment.py` に両方を詰め込むと、結局 2 つの責務を持つ。

**結論**: `config.task` による分岐は 3–4 行の if/elif であり、各 task の実装は `lib/` 内の関数に委譲する。ランナー自体は「config 読込→task 実行→結果保存」のオーケストレータに過ぎず、200 行以内に収まる。task が 5 種を超えた時点でランナー分割を検討する。

### §7.2 YAML config は本当に Python dict より良いか

YAML の利点: diff に残る、Git で変更追跡しやすい、非プログラマにも読める。
YAML の欠点: 型チェックが弱い、動的な値（計算式）が書けない。

**結論**: 動的値は base.yaml の Python ロード時に解決する（`null` は必須入力エラー、`"auto"` は特別扱い）。型チェックは `gate_thresholds.yaml` のスキーマバリデーションで補う。v459 のハードコード dict よりは遥かに管理しやすい。

### §7.3 118# の G0-data / manifest は過剰投資か

v459 では再現性が文書ベースだったため、「同じ実験をもう一度」が不可能だった。manifest の JSONL 1 行分のコストは数行のコードで、投資対効果は極めて高い。

**結論**: 過剰投資ではない。初日に実装すべき。

### §7.4 v459 の 45 スクリプトは全て無駄だったか

`run_k2_nonrl_upper_bound.py` (297 行) は最も重要な知見 (FEATURES_NO_INFO) を生んだ。問題はスクリプトの存在ではなく、**共通化できる処理 (データ読込・評価・保存) が毎回コピペされていたこと**。

**結論**: v460 の `lib/` に共通処理を集約し、ランナーは薄いオーケストレータにする。K2 相当のアドホック分析は `experiments/v460/` に置く。

---

## §8 Gate 枝番・Phase 枝番・バージョン枝番の命名法則

### §8.1 Gate 枝番規則

118# で `G1.5-exec` が提案されたが、`.5` は中間挿入の再帰問題を引き起こす（G1.5 と G2 の間に挟むと G1.75 が必要になる）。

**規則**: 枝番は `.1` 刻みの整数連番とする。

```
G0 → G1 → G2 → G3 → G4          # 初期設計
G0 → G1 → G1.1 → G2 → G3 → G4   # 1回目の挿入
G0 → G1 → G1.1 → G1.2 → G2 → ...# 2回目の挿入（G1.1とG2の間）
```

| 規則 | 説明 |
|------|------|
| 枝番は `.1`, `.2`, `.3`... の連番 | `.5` や `.25` のような非整数禁止 |
| 枝番は親 Gate の**後ろ**にのみ挿入 | G1.1 は G1 と G2 の間に位置 |
| 枝番の再帰禁止 | `G1.1.1` は禁止。2 段以上のネストが必要ならば Gate 体系自体を再定義 |
| 名前は必須 | `G1.1-exec` のように用途名を付ける。番号だけの Gate 禁止 |
| 最大枝番数 | 1 親 Gate あたり **3 枝番**まで。超えたら体系再定義 |

**v460 初期の Gate 体系（確定版）**:

```
G0-data   : データ品質・再現性基盤
G1-info   : 特徴量情報量（非RL上限, multi-target）
G1.1-exec : maker 執行可能性（fill/latency）
G2-train  : 学習安定性（seed分散、再現）
G3-pnl    : コスト込み収益性
G4-live   : Paper trading 運用検証
```

### §8.2 Phase 枝番規則

117# で Phase 小数 (3.5, 4.5) とアルファベット (B, C, D) の混在を禁止した。しかし方向転換時にPhase 間への挿入が必要になるケースは現実に起こり得る。

**規則**: Phase にも Gate と同じ `.1` 刻みを適用する。

```
ph0 → ph1 → ph2 → ph3             # 初期計画
ph0 → ph1 → ph1.1 → ph2 → ph3     # ph1完了後に追加検証が必要になった
```

| 規則 | 説明 |
|------|------|
| `.1`, `.2` の整数連番 | `.5` 禁止。v459 の `Phase 3.5` の再発防止 |
| 再帰禁止 | `ph1.1.1` 禁止 |
| 1 Phase あたり最大 2 枝番 | `.3` が必要になったら Phase 体系を再定義 |
| ファイル名表記 | `ph1` → `ph1-1`（YAML/ファイル名でドットを避ける） |
| 00# §2 に即時反映 | 枝番追加時は 00# の Phase 定義テーブルと Appendix A を更新 |

**ファイル名の実例**:
```
015_ph1-1_plan_additional_validation.md   # Phase 1.1 の計画
016_ph1-1_rpt_validation_results.md       # Phase 1.1 の結果
```

### §8.3 バージョン枝番: v460.1 が必要になるケース

v460 本体が「短期間での高収益性」を目指す以上、v460 内で完結することが原則。しかし以下のケースでは v460.1（マイナー分岐）の発行が妥当:

| ケース | 例 | v460.1 の性質 |
|--------|---|--------------|
| **A. 特徴量ソース分岐** | 板情報 (v460) vs 約定フロー (v460.1) で並行評価が必要 | 同一アーキテクチャ、データソースのみ異なる。短期間で合流予定 |
| **B. 取引所分岐** | Zaif (v460) vs bitFlyer (v460.1) で maker 条件が異なる | 手数料・API が異なるため config 差分では吸収不能 |
| **C. 中間リリース** | G3-pnl PASS 後、G4-live に進む前に一旦安定版をタグ付け | v460 の Phase 3 完了スナップショット |

**v460.1 を発行しない場合**:

| 状況 | 対応 | 理由 |
|------|------|------|
| Phase 計画の変更 | Phase 枝番 (ph1.1) で対応 | バージョン分岐ほど大きな乖離ではない |
| 実験パラメータの追加 | experiments/*.yaml の追加 | config レベルで吸収可能 |
| Gate 追加 | Gate 枝番 (G1.1) で対応 | Gate 体系内で管理 |
| 全 Gate FAIL → 根本再設計 | **v461 へ移行** | v460.1 ではなく新バージョン。v459→v460 と同じ |

**規則**:
- v460.N の N は 1, 2, 3...（整数のみ。v460.1.1 のような再帰禁止）
- v460.1 のディレクトリは `docs/v460.1/`, `configs/v460.1/` etc.
- v460.1 の 00# は v460/00# を**継承・差分記述**し、共通部分は v460/00# へのリンク
- マイナー分岐は**最大 2 本** (v460.1, v460.2) まで。3 本目が必要なら v461 を立てる

---

## §9 00# (000_ph0_plan_project_proposal.md) の分量方針

### §9.1 問題の再整理

117# §4.3 で「150 行以内」を目標としたが、簡潔になりすぎて Gate 定義のエッセンスが読み取れないリスクがある。v459 00# の §5（Gate 定義）が 419 行中 150 行を占めたが、それこそが Phase E まで参照され続けた最重要セクション。

### §9.2 解決策: 目次 (TOC) の設置

00# は**特別扱い**として冒頭に目次を設ける。これにより:
- 分量が 200–250 行になっても、目的セクションへ即座にジャンプ可能
- Gate 定義に必要な詳細を省略せずに書ける
- 「簡潔すぎてエッセンス消失」と「冗長すぎて参照困難」の中庸を実現

### §9.3 00# 構成（改訂版）

```markdown
# v460 Project Proposal: [コードネーム]

## 目次
- [§0 大義と目的](#§0)
- [§1 v459 教訓サマリ](#§1)
- [§2 Phase 定義](#§2)
- [§3 Gate 定義](#§3)
  - [§3.1 G0-data](#§3.1) / [§3.2 G1-info](#§3.2) / [§3.3 G1.1-exec](#§3.3)
  - [§3.4 G2-train](#§3.4) / [§3.5 G3-pnl](#§3.5) / [§3.6 G4-live](#§3.6)
  - [§3.7 統計検定仕様](#§3.7)
- [§4 技術概要](#§4)
- [§5 命名規則・運用規約](#§5)
- [§6 リスク](#§6)
- [Appendix A: 改訂履歴](#appendix-a)
```

### §9.4 分量ガイドライン（改訂）

| セクション | 目標行数 | 方針 |
|-----------|---------|------|
| 目次 | 15 行 | §3 のサブセクションまでリンク |
| §0 大義と目的 | 10 行 | 3 文以内で核心。前提条件 (maker-only) 明記 |
| §1 v459 教訓 | 15 行 | 4 教訓 + v459 index/116# へのリンク。詳細は書かない |
| §2 Phase 定義 | 20 行 | テーブル形式。Phase 枝番規則への参照リンク |
| **§3 Gate 定義** | **80–100 行** | **ここが本体。省略禁止。閾値・統計検定仕様を含む** |
| §4 技術概要 | 15 行 | 決定事項のテーブルのみ。詳細は 001# に委譲 |
| §5 命名・運用 | 20 行 | 119# §8 の転記サマリ + 枝番規則 |
| §6 リスク | 10 行 | Top 3 のみ |
| Appendix A | 5 行 | 初版は空テーブル |
| **合計** | **190–210 行** | 117# の 150 行目標から上方修正。TOC で可読性を担保 |

### §9.5 117# §4.3 との差分

| 指標 | 117# (旧) | 本改訂 (新) | 変更理由 |
|------|----------|-----------|---------|
| 総行数 | 150 行以内 | **190–210 行** | Gate §3 の詳細維持 + TOC 追加 |
| TOC | なし | **あり (15 行)** | 00# は頻繁参照前提。ジャンプ性確保 |
| コード例 | 0 | **0（維持）** | 統計検定コードは別文書 |
| §3 分量 | 指定なし | **80–100 行** | v459 §5 の教訓。省略すると判定が曖昧化 |

---

## §10 結論テーブル

| 領域 | v459 の問題 | v460 の方針 | 根拠 |
|------|-----------|-----------|------|
| **docs/** | 117 文書、命名崩壊 | 40 以内、`NNN_phX_type` | 117# §3 |
| **scripts/** | 45 本・12K 行、God object | 1 ランナー + lib/ | 本 §3 |
| **configs/** | base.yaml 形骸化、dict ハードコード | base + experiments/*.yaml | 本 §4 |
| **再現性** | 文書のみ | manifest.jsonl 自動記録 | 118# §2 |
| **Gate 体系** | 番号のみ、順序不適切 | 名前付き G0–G4、G1 最優先 | 118# §3 |
| **Gate 枝番** | なし (G1.5 が場当たり的に発生) | `.1` 刻み整数連番、最大 3 枝 | 本 §8.1 |
| **Phase 枝番** | 小数/アルファベット混在 | `.1` 刻み整数連番、最大 2 枝 | 本 §8.2 |
| **バージョン枝番** | なし | v460.N、最大 2 本 | 本 §8.3 |
| **00# 分量** | 419 行 (冗長) | **190–210 行 + TOC** | 本 §9 |
| **実験追加** | スクリプト編集 or 新作 | YAML 1 ファイル追加 | 本 §3.5 |
