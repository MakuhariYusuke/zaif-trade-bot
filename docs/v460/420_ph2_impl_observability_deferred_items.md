# 420# 416#/417# 先送り事項: 可観測性改善 + hard_skip_mult 有効化

> **前提**: 421# Execution Final Clamp + Route-to-Kill 実装時に先送りとなった
> 416# (Codex review) / 417# (Gemini second opinion) の指摘事項を整理・実装。

## §1 対象事項と優先度

| # | 項目 | 起源 | 優先度 | 難易度 | 状態 |
|---|------|------|--------|--------|------|
| 1 | `start_git_sha` 固定 | 416# §2-4 | P1 | Easy | ✅ 実装済 |
| 2 | `executor_offset_stages` JSON 記録 | 416# §4.2 / 417# | P1 | Medium | ✅ 実装済 |
| 3 | Skip record side 可観測性 | 416# §4.2 | P1 | Medium | ✅ 実装済 |
| 4 | `hard_skip_mult` 有効化 (0.0→2.5) | 417# Action-2 | P2 | Easy | ✅ 実装済 |
| 5 | sell hour boost vs ceiling 衝突 | 416# §4.1 | P2 | — | ⏳ 要分析 |
| 6 | trending cycle overrun リスク | 417# | P2 | — | ⏳ 資料化のみ |

---

## §2 実装詳細

### §2.1 `start_git_sha` 固定 (Item 1)

**問題**: `hot_reload` のたびに `git_sha` が更新されるため、
ラン開始時のコードベースが不明になり hindsight 分析で SHA 帰属が混乱。

**解決策**: `FillTestRunner.__init__` で `self._start_git_sha = self._git_sha` を保持。
`hot_reload` では `_git_sha` のみ更新、`_start_git_sha` は不変。

**変更ファイル**:
- `ztb/metrics/fill_quality.py` — `start_git_sha: str | None = None` 追加
- `scripts/v460/run_fill_test.py` — `self._start_git_sha = self._git_sha` 追加
- `scripts/v460/lib/fill_record_builder.py` — payload に `start_git_sha` 追加
- `scripts/v460/lib/fill_record_helpers.py` — skip record に `start_git_sha` 追加

### §2.2 `executor_offset_stages` JSON 記録 (Item 2)

**問題**: 6 段階の executor multiplier chain のうち 5 段階が FillRecord に未記録。
EV のみ `ev_offset_mult_applied` があったが、velocity / trending / toxicity /
vg_supplement / alert は不透明で、offset がどの段階で増幅されたか事後追跡不可。

**解決策**: 全 6 段階の multiplier 値を JSON オブジェクトとして 1 フィールドに集約。

```json
{
  "ev": 1.05,
  "velocity": null,
  "trending": 1.10,
  "toxicity": 1.15,
  "vg_supp": null,
  "alert": 1.0
}
```

`null` = 当該段階が条件不成立で未適用。

**変更ファイル**:
- `ztb/metrics/fill_quality.py` — `executor_offset_stages: str | None = None` 追加
- `scripts/v460/lib/fill_cycle_executor.py` — JSON 構築 + `_build_fill_record` 引数追加
  - `_tox_mult`, `_vg_supp_mult`, `_a_mult` のデフォルト初期化追加 (NameError 防止)
- `scripts/v460/lib/fill_record_builder.py` — payload に `executor_offset_stages` 追加

### §2.3 Skip record side 可観測性 (Item 3)

**問題**: SideSelector が選んだ side が balance_switch や Route-to-Kill で
上書きされた場合、最終 FillRecord だけでは元の意図が追跡不可。

**解決策**: `CycleContext` に `requested_side` (SideSelector の初期選択) と
`resolved_side_reason` (切替理由) を追加。FillRecord にも同名フィールドを追加。

**切替理由の値**:
| 値 | 意味 |
|----|------|
| `None` | side 変更なし |
| `"balance_switch"` | 残高不足による side 切替 |
| `"route_to_kill_deadlock"` | Route-to-Kill デッドロック検知による skip |

**変更ファイル**:
- `ztb/metrics/fill_quality.py` — `requested_side`, `resolved_side_reason` 追加
- `scripts/v460/lib/orchestrator_pre_cycle.py` — CycleContext に 2 フィールド追加 + 初期設定
- `scripts/v460/lib/orchestrator_balance.py` — balance_switch / route_to_kill 理由設定
- `scripts/v460/lib/orchestrator_mid_cycle.py` — 事後 record 転写

### §2.4 `hard_skip_mult` 有効化 (Item 4)

**変更**: `configs/v460/fill_test.yaml`
```yaml
# Before
execution_final_clamp_hard_skip_mult: 0.0
# After
execution_final_clamp_hard_skip_mult: 2.5
```

**効果**: Final Clamp offset が ceiling の 2.5 倍を超えた場合、注文自体をスキップ。
- buy: `offset > 0.20 × 2.5 = 0.50` → skip
- sell: `offset > 0.50 × 2.5 = 1.25` → skip

---

## §3 先送り事項

### §3.1 sell hour boost vs ceiling 衝突 (P2)

`sell_hour_offset_boost` がピーク時間帯でフロアを押し上げ、
ceiling と衝突する可能性がある。実運用データから実際の衝突頻度を確認後に対応予定。

### §3.2 trending cycle overrun (P2)

trending レジームの interval=60s に対し、sell worst case (75s timeout + 15s wait)
が 90s の場合、次サイクル開始が遅延する。ただし `run_single_cycle` は await される
ため overlap ではなく throughput 低下のみ。ドキュメント化で完了とする。

## 2026-03-21 補遺

420# 以後、可観測性については追加の前進があった。

- event log 共通メタ:
  - `timestamp_epoch`
  - `utc_day`
  - `utc_hour`
- `cycle_revenue_context` event 追加
- fill test exit/event への `memory_diagnostics` 出力追加
- `cross_venue_hint` の event log 可観測性追加

このため、420# の「先送り事項」は実質的には次の 2 件にかなり絞られている。

1. `sell hour boost vs ceiling` の実データ分析
2. `trending cycle overrun` の扱い整理

観測基盤そのものは、当時よりだいぶ前に進んでいる。

---

## §4 テスト

- `tests/unit/v460/test_421_final_clamp_deadlock.py` に 10 テスト追加
  - `TestStartGitSha` (3 tests)
  - `TestExecutorOffsetStages` (2 tests)
  - `TestSideObservability` (4 tests)
  - `TestHardSkipMultConfig` (1 test)
- 既存 35 テスト + 新規 10 テスト = **45 テスト全パス**
- フルリグレッション: **2179 passed, 125 skipped, 0 failed**

---

## §5 変更ファイル一覧

| ファイル | 変更種別 |
|----------|----------|
| `ztb/metrics/fill_quality.py` | FillRecord に 4 フィールド追加 |
| `scripts/v460/run_fill_test.py` | `_start_git_sha` 保持 |
| `scripts/v460/lib/fill_record_builder.py` | payload に 4 フィールド追加 |
| `scripts/v460/lib/fill_record_helpers.py` | skip record に `start_git_sha` 追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | JSON 構築 + デフォルト初期化 |
| `scripts/v460/lib/orchestrator_pre_cycle.py` | CycleContext 拡張 |
| `scripts/v460/lib/orchestrator_balance.py` | side 切替理由記録 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | record 転写 |
| `configs/v460/fill_test.yaml` | `hard_skip_mult` 0.0→2.5 |
| `tests/unit/v460/test_421_final_clamp_deadlock.py` | 10 テスト追加 |
