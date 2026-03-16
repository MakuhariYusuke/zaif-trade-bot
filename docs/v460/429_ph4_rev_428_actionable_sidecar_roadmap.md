# Document 429: Actionable Roadmap for Sidecar Revival & "The Final Clamp" Observability (Revised)

**Date**: 2026-03-15  
**Based on**: 428# Codex Review, Project Root Directives (copilot-instructions.md), & v460 Project Proposal (000_ph0_plan_project_proposal.md)  
**Target Audience**: SAC開発・統合を担当するAIエージェント及び開発者

---

## 1. 原典（0番ドキュメント / Proposal）及びプロジェクト大義との整合性

本プロジェクトの真拠とも言える **`docs/v460/000_ph0_plan_project_proposal.md`** には、v460における絶対的な方針が定められています。

1. **大義**: 短期間での高収益性システムの実現（「Microstructure Edge」の追求）
2. **前提条件**: 全取引は maker 注文に限定（taker禁止、手数料負けの回避）
3. **v459からの教訓**: 
   - 「特徴量を先に検証せよ (G1-infoGateでのXGBoost等による実証)」
   - 「単一seedの成功を信じるな (Don't trust single seed)」
   - 「Oracleテスト（完全予測）からの逆算設計による破綻防止」

また、**`copilot-instructions.md`** で定められた「単一責任原則（SOLID）」「西洋的思考（End-to-End・計算資源の暴力）への固執回避」を加味すると、現在の「SAC一辺倒（AIに価格から数量、停止条件まで全てを決定させるEnd-to-End設計）」は**全てのプロトコルに違反**しています。

「儲かるシステム」を最短で作れるアーキテクチャとは何か？
それは、**AIを『勝てそうな風向き（Directional Bias）』の示唆のみに限定（Sidecar）し、複雑なmaker執行や退避判断は数学的に安全な静的ルールとThe Final Clampに委ねる**アプローチです。これは0番ドキュメントの「特徴量を先に検証せよ（G1-info）」の思想をそのままAlphaモジュール（Sidecar）に切り出す構造でもあります。

---

## 2. 潰すべき「駄目」な現状（The Anti-Patterns to Kill）

Codexのレビュー（428番）により、現在のコードベースに潜む重大な「駄目」が浮き彫りになりました。これらを直ちに潰してください。

### ❌ 駄目1: 「Sidecarは実装されているが、息をしていない」
- **実態**: 既に `scripts/v460/lib/fill_cycle_executor.py:742`, `scripts/v460/lib/fill_config.py:324`, `scripts/v460/lib/orchestrator_mid_cycle.py:135`, `scripts/v460/lib/sidecar_types.py` 等にSidecarやClampの骨格はあるが、`cache/sidecar_signal.json` は更新が止まっている。直近の `fill_records` でも Sidecar関連の数値が Null（出力ゼロ）になっている。
- **対処**: 新機能発明は不要です。ゼロから作るのではなく**「死んでいる既存の配線を直して血液（Signal）を流す」**作業を最優先としてください。

### ❌ 駄目2: 「Clamp-Driven Development（安全装置への甘え）」
- **実態**: Final Clamp（0.30オフセット上限など）は既に実装・発火しているものの、「Clampのおかげで損失は防げたが、そもそものAIの予測がどれだけ狂っていたのか」が見えないため、上流の失敗を覆い隠してしまっている。
- **対処**: Clampの発火率（`clamp_fire_rate`）や、Clamp発動前後の期待値（`pre_clamp_offset` / `post_clamp_offset`）を可視化する Observability の層を追加してください。

### ❌ 駄目3: 「SACへの過度な期待（責務過多）」
- **実態**: 1分足のmaker注文というシビアな環境下で、RL（SAC）に相場の「価格」「数量」「ボラティリティ予測」の全てを求めているため、学習が破綻（reward-profit correlation paradox）しています。
- **対処**: SACの責務を劇的に縮小してください。出力は極めてシンプルな **`directional_bias`（方向の偏り: -1.0 〜 1.0）** や、参加の可否を補助する **`aggressiveness_hint`** に限定してください。

---

## 3. 「儲かる方法」への直結アクションプラン（Next Steps for ML Agent）

SAC担当のAIが直ちに取り掛かるべき、利益に直結する具体的なタスクリストです。

### Step 1: Sidecar Signal の蘇生活動 (Live Presence)
- **Target**: `scripts/v460/lib/fill_cycle_executor.py`, `scripts/v460/lib/orchestrator_mid_cycle.py`
- **Action**: 
  - `cache/sidecar_signal.json` を常に最新のAI推論値で更新するループを復旧させること。
  - Executor側で、stale（古すぎる）シグナルを無視する安全機能を担保しつつ、有効なシグナルが `fill_records` の `sidecar_offset_bps` や `sidecar_bias` に記録されるように配線を太く・確実につなぐこと。

### Step 2: SACモデルの出力定義のダウングレード (Duty Reduction)
- **Target**: SAC Action Space, `ztb/trading/strategies/action_signal_guide/components/sac_integration.py`
- **Action**:
  - RL側の出力を `[-1.0, 1.0]` のバイアス値にまで軽量化する。
  - この値を、Executor側で安全なオフセット加減算（例: baseline_offset ± (bias * factor)）に変換するロジックを `pre_order_adjustments.py` に実装する。
  - 複雑な価格決定や数量計算は全てExecutor（静的ルール）側に任せる。

### Step 3: Observability (Clamp発動の計器化)
- **Target**: `scripts/v460/lib/fill_cycle_executor.py`, `scripts/v460/lib/performance_monitor.py`
- **Action**:
  - `fill_records*.jsonl` に、以下のフィールドを確実に保存させる：
    - `pre_clamp_offset`: Clamp適用前のAI/ルールの要求オフセット
    - `post_clamp_offset`: Clamp適用後の最終オフセット
    - `clamp_fired`: 完全に上限・下限にHitして切り捨てられたかを示すフラグ (boolean)
  - `hard_skip_rate` や `clamp_fire_rate` が高い場合は、「上流の予測（Alpha・Sidecar層）が死んでいる」と判定できるアラートの仕組みを入れること。

### Step 4: 既存実証基盤（Walk-Forward・検定）の再利用
- **Target**: `ztb/evaluation/walk_forward/splitter.py`, `scripts/v460/ml/walk_forward_as.py`
- **Action**:
  - 新たな評価基盤を作ることは避ける。プロジェクト内に既に存在する堅牢な Walk-Forward のモジュール（`splitter.py` など）を流用する。
  - 0番ドキュメント（§3.2 G1-info等）に記載されている厳密な評価指標（OOS Spearman ICなど）を用いて、Sidecarから出力されるシグナルの真の有効性が証明できるように配線する。

---

## 4. 担当AI（SACチューナー）へのメッセージ

**「全ての取引をSAC（AI）一発で行う End-to-End は、0番ドキュメントでNo-Goとされたv459の亡霊です。直ちにそのアプローチを捨ててください。」**

短期間で高収益を叩き出す「儲かる方法」の最短ルートは下記の通りです：
1. **アルファ（予測）層**：SACを、方向性（Directional Bias）を超高精度で当てるナビゲーター（Sidecar）に徹させる。
2. **実行層・安全層**：ナビゲーターの指示はあくまで「値幅のヒント」として扱い、実際のmaker注文の複雑な板処理と資金管理、最終的な防波堤（Final Clamp）は、既存の静的ロジック（`fill_cycle_executor.py`）に任せる。

まずは新しいことをするのではなく、既に形だけある配線を直し（Step 1）、AIからのシグナルが「安全な範囲内で」実売買データ（`fill_records`）に流れる『機能するパイプラインの証明（Live Presence）』から着手してください。それがG4（実働運用）への唯一の道です。