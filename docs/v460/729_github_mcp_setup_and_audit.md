# 729# GitHub MCP セットアップ + 残課題監査 + Copilot Agent 委譲

## 概要
GitHub MCP サーバーを VS Code に設定し、Copilot Agent 経由で Issue 作成→自動 PR 生成のワークフローを確立。
併せて残課題の体系的監査を実施し、全 P0/P1 項目が実装済みまたは管理下であることを確認。

## 1. GitHub MCP セットアップ

### 設定ファイル
`.vscode/mcp.json` 作成（HTTP remote 方式、Copilot OAuth 認証）:
```json
{
  "servers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/"
    }
  }
}
```

### 経緯
- 当初 `@modelcontextprotocol/server-github` (npm stdio) を試行 → **deprecated** と判明
- 公式 remote server (`api.githubcopilot.com/mcp/`) に切替 → PAT 不要、Copilot OAuth で認証
- WSL 設定修正: `chatgpt.runCodexInWindowsSubsystemForLinux: false` (Windows native workspace)

## 2. 残課題監査結果

| ID | 内容 | 状態 | 備考 |
|----|------|------|------|
| P0-C | corr gate (G3 E6 reward-profit) | ✅ 409# 実装済み | テスト 7件、test_409_corr_gate.py |
| P0-B | 旧 artifact 再保存 | ⏸ 運用タスク | fill_test 非稼働時に実施 |
| P1-C | Spread Anomaly Detector (SAD) | ✅ 211# 実装済み | - |
| P1-D | MCB↔SAD AND escalation | ✅ 211# 実装済み | - |
| C3-3 | hot-reload field metadata 自動生成 | 📋 構造改善 | 726# CODE_COUPLED_FIELDS で部分対応 |
| C3-4 | YAML unknown-key validation | 📋 構造改善 | test_336 で 200+ override 監視中 |

## 3. Copilot Agent 委譲

### Issue #5 → PR #6
- **タスク**: Mixin `type: ignore[attr-defined]` 一括解消 (34件)
  - `skip_gate_ev_weighted.py`: 13件
  - `skip_gate_model_loader.py`: 21件
  - `sac_train.py`: 2件 (追加)
- **方針**: Option B — Mixin クラスに型付き属性宣言を追加
- **PR**: [#6](https://github.com/MakuhariYusuke/zaif-trade-bot/pull/6) (Draft, branch: `copilot/remove-type-ignore-attr-defined`)
- **状態**: Copilot Agent 作業中（Initial plan commit 済み）

## 4. パフォーマンス確認 (18.4h post-restart)

| 指標 | Pre (04/01-09 avg) | Post (04/10) | 変化 |
|------|-------------------|-------------|------|
| EV-PnL avg | -0.42 bps | +0.97 bps | **+1.39** |
| Fill Rate | 42.9% | 56.4% | **+13.5pp** |
| Win Rate | 46.4% | 49.7% | +3.3pp |
| Sell EV avg | -0.68 bps | +1.26 bps | **+1.94** |

## 変更ファイル
- `.vscode/mcp.json` (新規)
- `docs/v460/729_github_mcp_setup_and_audit.md` (本ドキュメント)
