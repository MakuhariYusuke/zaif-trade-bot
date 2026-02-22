# v457 Second Opinion Request Prompt

以下のプロンプトは、他のAIコーディングエージェントにプロジェクト全体のアーカイブ解析を依頼し、v457で復活させるべき「失われたAlpha」や「有用な資産」を発掘するためのものです。

---

## Prompt for AI Agent

**Role:**
あなたはアルゴリズムトレード開発の監査役兼アーキオロジスト（考古学者）です。
現在、プロジェクトは `v457` フェーズにあり、過去の複雑化しすぎた実装（v456等）をリセットし、収益性の高かった「シンプルで堅牢なロジック」への原点回帰を目指しています。

**Context:**
- **Current State (v457)**: `scripts/v457` を起点に、v456の安定したインフラ（Factory, Env）は維持しつつ、ロジックと報酬を単純化しようとしています。
- **Problem**: v455/v456での過度な報酬設計や、v454での過剰適合により、本来持っていた収益性（Alpha）が失われている可能性があります。
- **Resources**: `scripts/`, `docs/`, `backtest_results/` ディレクトリには `v444`, `v451`, `v453` などの過去バージョンが残っています。

**Task:**
プロジェクト内の過去のアセット（`scripts/vXXX`, `docs/vXXX`）を横断的に調査し、以下の観点で **「v457で再採用すべき遺産」** を特定して報告してください。

**Investigation Points:**
1.  **Lost Alpha (失われた優位性)**:
    - v455以降で削除・無効化されたが、実は高パフォーマンスだった特徴量やロジックはありませんか？
    - 特に `v444` (Simple SAC) や `v451` (Optimized) 時代に使われていた単純なテクニカル指標やエントリー条件を確認してください。

2.  **Forgotten Successful Configurations**:
    - バックテストログ (`backtest_log.txt` や `backtest_results/`) の中で、高いPnL/Sharpeを出していた具体的なパラメータ設定（`env_config` や `reward_settings`）を特定してください。
    - 「勝率重視」ではなく「最終PnL重視」の設定を探してください。

3.  **Specific Code Assets**:
    - `scripts/v455` や `scripts/v456` 以外で、v457の「シンプル化」の方針に役立つスクリプトツール（分析用、データ処理用など）はありますか？
    - 例: `analyze_bias_root.py` や `inspect_model.py` などの診断ツールの有用性。

**Output Format:**
レポートは以下の形式で出力してください。

```markdown
# Second Opinion Report: Recommended Assets for v457

## 1. 復活推奨のロジック・特徴量 (Lost Alpha)
- **Feature/Logic Name**: [名称]
  - **Source**: [ファイルパス/バージョン]
  - **Reason**: [なぜ有用か、どのログで成果が出ていたか]

## 2. 参照すべき成功コンフィグ
- **Version/Context**: [例: v451 optimized]
  - **Key Variables**: [重要なパラメータ値]
  - **Performance**: [当時の記録数値]

## 3. 有用ツール・スクリプト
- `[Script Path]`: [推奨理由とv457での使い道]

## 4. 警告 (Avoid these)
- [過去の失敗パターンとして避けるべき実装や設定]
```

**Action:**
Please execute this investigation by searching the codebase and reading relevant files.
