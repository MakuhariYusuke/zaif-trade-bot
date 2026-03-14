# 430# 428#/429# レビュー妥当性評価 + 既存実装監査 + 理論修正

**Date**: 2026-03-15  
**Phase**: ph4  
**Type**: rev (レビュー妥当性評価 + 深堀り)  
**対象**: `428_ph4_rev_427_global_direction_and_sidecar_review.md` (Codex), `429_ph4_rev_428_actionable_sidecar_roadmap.md` (Gemini)  
**方法**: 実装コード照合 + 既存資産棚卸し + 理論的整合性検証

---

## 1. 428# (Codex) 妥当性評価

### 総合評価: **A- (優)**

428# は 427# の過大な「新規構想」トーンを「既存資産の統合・検証」に補正した、本プロジェクトで最も実務的なレビューの一つ。

### 1.1 正しい指摘

| # | 指摘 | 判定 | コード照合による検証 |
|---|---|---|---|
| F1 | Sidecar/Final Clamp は既に実装済み。新規導入ではなく live 証明が必要 | **◎ 最重要** | `sidecar_types.py` (型・変換完全), `sidecar_signal_io.py` (atomic write/TTL 付き read 完全), `orchestrator_mid_cycle.py:L135` (毎サイクル読込), `fill_cycle_executor.py:L742-760` (offset 適用), `fill_cycle_executor.py:L766-815` (Final Clamp) — 全て実装済み |
| F3 | 国別文化論は比喩に留め、microstructure/観測性/保守性で判断すべき | **◎ 正確** | 設計判断に「文化」は不要。Coincheck BTC/JPY の maker 特性で決める |
| F4 | Phase 名を「実装計画」→「統合・検証計画」に修正 | **◎ 的確** | 下記 §3 の棚卸しが裏付け |
| F5 | Clamp-Driven Development 警告。clamp_fire_rate 可視化が必要 | **◎ 重要** | `fill_record_builder.py:L301` に `execution_pre_clamp_offset` フィールドは既にある。集計層は未実装 |
| F6 | SAC 出力は directional_bias + aggressiveness に限定 | **○ 妥当** | 現行 action_space は既に continuous_1d [-1, 1]。方向は合っている |
| F7 | 422-425 の reward 問題 / corr paradox を 427# が踏まえていない | **◎ 重大な欠落指摘** | 427# は architecture の話に終始し、reward 設計×val_ratio 交互作用 (425#) を無視 |
| F8 | walk-forward は既存資産で済む | **○ 正しい** | `splitter.py`, `walk_forward_as.py` に expanding window + embargo が実装済み |

### 1.2 補完が必要な点

| # | 指摘 | 428# の扱い | 補完 |
|---|---|---|---|
| C1 | 3層分離モデル (Alpha/Execution/Safety) | 図式として提示 | **現実の `fill_cycle_executor.py` では3層が1ファイル 1200行超に混在**。分離は方向として正しいが、refactoring scope の言及が欲しかった |
| C2 | Sidecar の「推論鮮度」問題 | signal stale を指摘 | **根本原因は `sac_retrain_scheduler.py` が起動されていないこと**。428# はこの点を「live presence 証明」で包んでいるが、ops script / scheduled task の欠如を明示すべきだった |
| C3 | `sac_integration.py` の再利用 | SACSignalValidator を言及 | **実態は型定義のみの DEAD コード** (下流 wire なし)。「発想を流用」は正しいが「そのまま使える」ではないことを明記すべき |

### 1.3 見落としている点

| # | 見落とし | 詳細 |
|---|---|---|
| M1 | `sac_retrain_scheduler.py` が **完全実装済み** (L267-447, L730-793) | 428# は live presence の不在を指摘したが、**生産者側 (scheduler) が実装完了済みで起動されていないだけ** という診断に達していない。修正は「コードを書く」ではなく「ops script を作って起動する」のみ |
| M2 | Sidecar v2 実装 (`compute_sidecar_offset_bps_v2`, L174-232) | 離散3値化 (v1) の情報損失 95% 問題を解決する **比例的変換** が既に実装済み。dead zone + shaping (linear/quadratic/sigmoid) + confidence-weighted。428# はこの存在に触れていない |
| M3 | `fill_quality.py` に `sidecar_offset_bps`, `sidecar_bias` フィールドが既存 | fill_records への記録配線は済んでおり、「新しいフィールドを追加する」必要はない。やるべきは **集計** のみ |

---

## 2. 429# (Gemini) 妥当性評価

### 総合評価: **B (良)**

429# は 428# を action plan 化したもので、方向は合っているが、修辞が過剰で実態との乖離がある。

### 2.1 正しい指摘

| # | 指摘 | 判定 |
|---|---|---|
| G1 | 000# (Project Proposal) と copilot-instructions.md への原典回帰 | **◎ 有益**。End-to-End が v459 教訓で否定的に扱われていることの文書整合性チェック |
| G2 | Step 1: Sidecar signal の蘇生 | **○ 方向正しい** |
| G3 | Step 3: Clamp observability フィールド追加 | **○ 具体的** |

### 2.2 不正確・過大な点

| # | 指摘 | 問題 | 正確な実態 |
|---|---|---|---|
| O1 | 「全てのプロトコルに違反」 | **言い過ぎ** | 現行 SAC は End-to-End ではない。action_space = continuous_1d [-1,1] で既に bias 出力構造。fill_cycle_executor がルールベース執行を担当 |
| O2 | 「直ちにそのアプローチを捨ててください」 | **対象が存在しない** | 現在 live で作用する SAC は存在しない (sidecar signal は stale で null)。「捨てる」のではなく「起動する」が正しい |
| O3 | Step 2「SAC出力定義のダウングレード」 | **現状を確認していない** | `sac_train.py` の action_space は既に `continuous_1d` (-1 to 1)。「価格・数量・停止条件まで AI が決定」は v460 の実態と乖離。reward 構造は確かに問題だが output 定義は既にシンプル |
| O4 | 422-425 の val_ratio × reward 交互作用 | **完全にスルー** | 427# と同じく reward 設計問題への言及ゼロ。S1/S1' 実験の文脈も未消化 |
| O5 | Step 4 既存 Walk-Forward 流用 | **方向は正しいが手順が甘い** | `walk_forward_integration_pipeline.py` は skeleton (内部実装なし)。`splitter.py` と `walk_forward_as.py` は functional だが SAC 用に adapter が必要 |

### 2.3 428# → 429# で失われた nuance

429# は 428# を急進化しすぎて、以下の重要な nuance が消えている:

- 428# の「scorecard 5軸判定」(Profit leverage / Controllability / Observability / Data parity / Maintenance cost) → 429# では省略
- 428# の「meta-labeling / supervised overlay も残すべき」→ 429# では「SAC に限定」
- 428# の慎重な「まず proof、それから拡張」→ 429# は「直ちに捨てよ」

---

## 3. ★ 既存実装の包括的棚卸し (深堀り結果)

### 3.1 Sidecar パイプライン全体図

```
[SAC Model] ─── retrain_scheduler.py ──→ [cache/sidecar_signal.json]
                  (L267-447: retrain)        ↑
                  (L730-793: signal更新)       │ atomic write
                  ★ 未起動 ★                   │
                                              │
[fill_test] ── orchestrator_mid_cycle.py ──→ read_sidecar_signal()
                  (L135: 毎サイクル読込)        │ TTL 7800s 検証
                                              │ mtime キャッシュ
                                              ↓
              fill_cycle_executor.py          SidecarSignal | None
                  (L742-760: offset 適用)       │
                  (L766-815: Final Clamp)       │
                                              ↓
              fill_record_builder.py        fill_records_*.jsonl
                  (sidecar_offset_bps)        [記録済み]
                  (sidecar_bias)              [記録済み]
                  (execution_pre_clamp_offset) [記録済み]
```

**診断**: パイプラインは **完全に配線済み**。唯一の断絶は `sac_retrain_scheduler.py` が **ops script なしで起動されていない** こと。

### 3.2 各コンポーネントの稼働状態

| コンポーネント | ファイル | 実装 | 稼働 | 残タスク |
|---|---|---|---|---|
| SidecarSignal 型定義 | `sidecar_types.py` | ✅ 完全 | ✅ | なし |
| Sidecar v1 変換 (3値化) | `sidecar_types.py:L99-145` | ✅ 完全 | ✅ | v2 への移行検討のみ |
| Sidecar v2 変換 (比例) | `sidecar_types.py:L174-232` | ✅ 完全 | ❌ 未使用 | 呼び出し元を v1→v2 に切替 |
| Signal I/O (write) | `sidecar_signal_io.py:L37-75` | ✅ 完全 | ⚠️ stale | scheduler 起動で解決 |
| Signal I/O (read+cache) | `sidecar_signal_io.py:L85-160` | ✅ 完全 | ✅ | なし |
| Signal 読込 (orchestrator) | `orchestrator_mid_cycle.py:L135` | ✅ 完全 | ✅ 毎サイクル | なし |
| Offset 適用 (executor) | `fill_cycle_executor.py:L742-760` | ✅ 完全 | ⚠️ 入力 null | scheduler 起動で解決 |
| Final Clamp | `fill_cycle_executor.py:L766-815` | ✅ 完全 | ✅ 発火中 | なし |
| Clamp 設定 | `fill_config.py:L324-342` | ✅ 完全 | ✅ | なし |
| fill_records 記録 | `fill_record_builder.py:L301` | ✅ 完全 | ✅ | なし |
| **Retrain Scheduler** | `sac_retrain_scheduler.py` | **✅ 完全** | **❌ 未起動** | **ops script 作成のみ** |
| Retrain Trigger | `sac_retrain_scheduler.py:L489-545` | ✅ 完全 | ❌ | scheduler 依存 |
| Signal 更新 (retrain後) | `sac_retrain_scheduler.py:L730-793` | ✅ 完全 | ❌ | scheduler 依存 |
| Neutral fallback | `sac_retrain_scheduler.py:L719-728` | ✅ 完全 | ❌ | scheduler 依存 |
| SAC 統合 (ztb) | `sac_integration.py:L102-230` | 🔴 型のみ | ❌ DEAD | 下流 wire 必要 |
| Walk-Forward splitter | `splitter.py:L49-100+` | ✅ 完全 | ✅ | SAC 用 adapter |
| Walk-Forward AS | `walk_forward_as.py:L62-96` | ✅ 完全 | ✅ | なし |
| Walk-Forward pipeline | `walk_forward_integration_pipeline.py` | 🟡 skeleton | ❌ | 内部実装必要 |
| Guard 分類器 | `guard_reason_classifier.py` | ✅ 完全 | ✅ | clamp metrics 拡張可 |
| Clamp 集計メトリクス | — | ❌ 未実装 | — | `guard_reason_classifier.py` 拡張で対応 |
| Toxicity 推定 | `ztb/features/microstructure.py:L84-137` | ✅ VPIN 実装 | ✅ 特徴量として | contextual 利用は未実装 |

### 3.3 Reward 関連の既存実装

| コンポーネント | ファイル | 状態 |
|---|---|---|
| RewardCalculator (本体) | `ztb/trading/environment/components/calculators/reward_calculator.py:L75-200+` | ✅ 完全。config-driven で A/B テスト可能 |
| BehavioralPenaltyCalculator | `behavioral_penalty_calculator.py` | ✅ consistency/balance/entropy penalty |
| AsymmetricRewardScaler | `asymmetric_reward_scaler.py` | ✅ position-dependent scaling |
| DynamicRewardShaper | `dynamic_reward_shaper.py` | ✅ regime-aware shaping |
| RewardSettings (config) | EnvironmentConfig 内 | ✅ YAML ↔ Python 変換済み |

**reward_clean vs reward_tuned の差分**:

| パラメータ | reward_clean | reward_tuned |
|---|---|---|
| reward_scaling | 100.0 | 1.0 |
| hold_penalty_weight | 0.0 | 0.001 |
| balance_penalty | 0.0 | 0.1 |
| consistency_penalty | 0.0 | 0.01 |
| position_penalty_weight | 0.0 | 0.01 |
| confidence_penalty_threshold | 1.0 | 0.2 |
| ent_coef | 0.01 (fixed) | "auto" |
| gradient_steps | 2 | 1 |

reward_clean は penalty を全て 0 にした「素の PnL reward」。reward_tuned は multiple penalty を有効化。425# で reward_clean のみ G3 PASS という結果は、**penalty が学習を阻害している** 可能性を示唆する。

### 3.4 428#/429# が見落としていた追加の再利用資産

| # | 資産 | ファイル | 活用余地 |
|---|---|---|---|
| A1 | **Sidecar v2 (比例的変換)** | `sidecar_types.py:L174-232` | v1 の 3値化による情報損失 95% を解決。dead zone + shaping で SAC の連続出力を保持。**即座に v1→v2 切替可能** |
| A2 | **SACRetrainConfig の confidence 動的計算** | `sac_retrain_scheduler.py:L750-760` | OOS gross_roi から confidence を導出する仕組みが既にある。Sidecar signal の信頼度を自動で反映 |
| A3 | **Toxicity 特徴量 (VPIN)** | `microstructure.py:L84-137` | SAC の観測空間には入っていない。guard_reason_classifier の `toxic_veto_block` と組み合わせると contextual bandit のコンテキスト変数になる |
| A4 | **Guard 分類器の category totals** | `guard_reason_classifier.py` | market/system/recovery の 3 分類。market 発火率が高い時 = 市場が荒れている → Sidecar bias を NEUTRAL に寄せるヒューリスティクスに使える |
| A5 | **performance_monitor.py の metrics 基盤** | `ztb/ops/health/performance_monitor.py` | 既存の health monitoring framework。clamp_fire_rate/hard_skip_rate の集計を追加する自然な場所 |
| A6 | **fill_records の 30s/60s/120s post-fill PnL** | `fill_quality.py` | Sidecar bias ↔ post_fill_pnl の相関分析に直接利用可能。「Sidecar が利益に寄与したか」の条件付き PnL 測定の基盤 |

---

## 4. 理論的修正: 428#/429# の共通前提への挑戦

### 4.1 「Sidecar 化 = 万能解」ではない

428#/429# とも、Sidecar 化を「正しい方向」として提示している。方向性には同意するが、**以下の前提が暗黙に仮定されておりリスクがある**:

**暗黙の前提**: SAC が directional_bias として有用な信号を出せる。

これは 425# の発見 (**reward_clean のみ G3 PASS、reward_tuned は G3 FAIL**) から自明ではない。  
もし SAC の出力自体が reward 設計に依存して不安定であるなら、Sidecar 化しても **garbage in → (bounded) garbage out** になるだけ。

**修正**: Sidecar 化の前に、まず S1/S1' 実験で **SAC が OOS で安定した directional bias を出せるか** を確認すべき。この順序は 425# の P0 と一貫する。

### 4.2 「retrain_scheduler を起動すれば済む」は半分正しいが半分危険

retrain_scheduler の実装は完全だが、**現在の retrain config がどの reward 設定を使うか** が未確認。  
もし reward_tuned 相当の config で retrain すると、425# で発見された G3 FAIL パターンを live に持ち込む恐れがある。

**確認すべき事項**:
1. `sac_retrain_scheduler.py` が参照する config ファイルはどれか
2. その config の `reward_settings` は reward_clean 相当か
3. retrain の `incremental_timesteps` (デフォルト 15K) が 20K 実験と整合するか

### 4.3 Final Clamp の発火パターンに関する理論的懸念

428# §6.2 の Clamp-Driven Development 警告は正しいが、**もう一段深い問題がある**:

Final Clamp が頻発している場合、2つの解釈が可能:
1. **上流の予測が暴走** → clamp が防波堤として機能 (正常動作)
2. **clamp が tight すぎて妥当な予測も切り捨て** → alpha の機会損失

これを区別するには、`execution_pre_clamp_offset` の分布と `post_fill_30s_pnl` の条件付き相関を見る必要がある:
- clamp fired + pre_clamp_offset が ceiling の 1.5 倍以上 → 暴走 → clamp 正当
- clamp fired + pre_clamp_offset が ceiling の 1.0-1.2 倍 → 境界的 → ceiling 拡大余地

### 4.4 v2 (比例的変換) への移行は low-hanging fruit

428#/429# とも Sidecar v1 (3値化) を前提に議論している。  
しかし **v2 が既に実装済み**で、3値化の情報損失問題を解決している:

| 比較 | v1 (3値化) | v2 (比例的変換) |
|---|---|---|
| SAC 出力利用率 | ~5% (±0.3 境界のみ) | ~90% (dead zone 0.1 以降全領域) |
| 情報損失 | 95% | ~10% |
| shaping | なし | linear/quadratic/sigmoid 選択可 |
| dead zone | なし | 0.10 (ノイズ除去) |
| confidence 反映 | なし | magnitude × confidence |

**v1→v2 切替は `fill_cycle_executor.py` の1行変更** (`compute_sidecar_offset_bps` → `compute_sidecar_offset_bps_v2`)。ただし v2 の max_boost_bps と shaping のパラメータチューニングが必要。

---

## 5. 修正された推奨アクション順

425# (S1/S1' 実験系) と 428#/429# (Sidecar/Clamp 構造系) を統合した全体ロードマップ:

### P0: 即時 (実験 + 評価系修正) — 現在進行中

| # | アクション | 状態 | 根拠 |
|---|---|---|---|
| P0-1 | S1 (reward_clean × 20K × val_ratio=0.20) 実行 | 🔄 実行中 | 425# 合意 |
| P0-2 | S1' (reward_tuned × 20K × val_ratio=0.20) 実行 | ⏳ S1 完了後 | 425# 交互作用分離 |
| P0-3 | F6 multi-slice 実装 | ✅ 済 (`432acd47a`) | 425#/423# 合意 |
| P0-4 | best_model 並行 OOS 評価 | ✅ 済 (`432acd47a`) | 425# |

### P1: S1/S1' 結果待ち → 次方向決定 (426# として文書化)

| S1 結果 | S1' 結果 | 次のアクション |
|---|---|---|
| FAIL | FAIL | val_ratio=0.02 は全般的に楽観 → reward 設計問題ではなく評価系問題 |
| FAIL | PASS | reward_clean 固有の問題 → reward 関数再設計 |
| PASS | FAIL | reward_clean は頑健 → retrain_scheduler に reward_clean config で起動 → P2 |
| PASS | PASS | 100K の問題は timesteps 側 → F6 修正 + best_model で改善可能 → P2 |

### P2: Sidecar live presence (構造系)

| # | アクション | 工数 | 備考 |
|---|---|---|---|
| P2-1 | retrain_scheduler 用 ops script 作成 (systemd or PowerShell scheduled task) | 小 | 実装は完全、起動するだけ |
| P2-2 | retrain config を reward_clean 相当に設定 | 小 | S1 結果に依存 |
| P2-3 | Sidecar v1→v2 切替 | 小 | `compute_sidecar_offset_bps_v2` への呼び出し変更 |
| P2-4 | scheduler 起動 → signal freshness / non-neutral push の確認 | 中 | live 証明の第一歩 |

### P3: Clamp observability

| # | アクション | 工数 | 備考 |
|---|---|---|---|
| P3-1 | `guard_reason_classifier.py` に clamp metrics 集計関数を追加 | 小 | 既存 fill_records フィールドの集計のみ |
| P3-2 | `execution_pre_clamp_offset` × `post_fill_30s_pnl` 条件付き相関分析 | 中 | clamp の tight/loose 判定 |
| P3-3 | clamp_fire_rate / hard_skip_rate を performance_monitor に統合 | 中 | 既存 health framework に追加 |

### P4: 効能検証 + 方向分岐

| # | アクション | 条件 |
|---|---|---|
| P4-1 | Sidecar bias ↔ post_fill_pnl 条件付き分析 | P2 完了後 |
| P4-2 | bias non-neutral 時の fill_rate / AS / timeout 変化計測 | P2 完了後 |
| P4-3 | Walk-forward 評価 (既存 splitter 流用、SAC adapter 追加) | P1 結果が PASS の場合 |
| P4-4 | Sidecar が無効 → meta-labeling / supervised overlay への方向転換検討 | P4-1/P4-2 で効果なしの場合 |

### HOLD

- 国別文化論を設計根拠に使用
- Full end-to-end SAC への回帰
- Sidecar に価格・数量・停止条件を持たせる拡張
- `sac_integration.py` (SACSignalValidator) の直接利用 — 型定義のみで下流 wire なし
- `walk_forward_integration_pipeline.py` — skeleton で内部実装なし
- Bare-metal / 自作 engine

---

## 6. 結論

### 428# (Codex)
方向は正しく、「新規構想」から「既存資産の統合」への補正は的確。欠けていたのは retrain_scheduler が **完全実装済みで起動されていないだけ** という診断と、Sidecar v2 (比例的変換) の存在。

### 429# (Gemini)
アクション化としての価値はあるが、修辞が過剰で実態との乖離がある。「全プロトコル違反」「直ちに捨てよ」は v460 の現実 (SAC は continuous_1d [-1,1]、executor がルールベース) と合わない。428# のトーンと scorecard 方式に戻すべき。

### 両者が共通して見落としていた点

1. **retrain_scheduler (L267-793) が完全実装済み** — ops script 一つで Sidecar pipeline が全通する
2. **Sidecar v2 (比例的変換, L174-232) が実装済み** — v1 の 3値化情報損失 95% 問題の解決策が既にある
3. **fill_records に sidecar/clamp フィールドが既に記録されている** — 新規フィールド追加は不要、集計層のみ必要
4. **S1/S1' 結果を待たずに Sidecar 化を推進するリスク** — reward 設計が不安定なら Sidecar 化しても bounded garbage out
5. **Clamp 発火の 2 面性** (暴走防止 vs alpha 機会損失) の区別方法が未提示

### 橋渡し: 425# → 430# → 次の実験

S1/S1' の結果 (426# として文書化予定) が、Sidecar 化の **是非** と **config** を決定する。両系統 (実験系 + 構造系) は **直列ではなく条件分岐** の関係にある。
