# 703# Codex タスク計画: Protocol 688 精密検証に基づく sell 損失対策

## 概要

702# の P688 再分析で特定された 4 つの重大発見に基づく Codex タスク群。
**特に sell/trending_up (-2.01 bps, 統計有意) と 12-17h UTC 損失集中 (-174.2 bps) への対策が最優先。**

## 発見→タスク対応表

| 発見 | 内容 | 対応タスク |
|------|------|----------|
| A: b56771a 交絡 | 1 SHA が全損失の 88.6% | T1 (SHA テレメトリ) |
| B: sell/trending_up | -2.01 bps (有意) | T2 (sell/trending_up ガード) |
| C: 12-17h 損失集中 | -174.2 bps | T3 (時間帯パラメータ再調整) |
| D: NFQ 射程外 71% | ranging 91件未対処 | T2 に統合 (spread_as_guard 連携) |

---

## Task 1: SHA 別 AS テレメトリ (P1)

**目的**: 特定 SHA の異常損失を早期検知するリアルタイムテレメトリ追加

**修正箇所**:
- `scripts/v460/lib/fill_record_builder.py`: SHA 別カウンタ追加
- `scripts/v460/lib/fill_test_runner.py`: SHA 変更検知ログ強化
- `scripts/v460/analysis/protocols/protocol_688.py`: SHA セクションに AS rate 追加

**テスト**:
- `tests/unit/v460/test_702_sha_telemetry.py`
- SHA 別 AS 率計算、SHA 変更検知、空データハンドリング

---

## Task 2: sell/trending_up 損失ガード (P0)

**目的**: 統計有意な sell/trending_up 損失 (-2.01 bps) への防御機構

**修正箇所**:
- `configs/v460/fill_test.yaml`:
  - `sell_trending_up_offset: 0.5` 追加 (sell_ranging_offset と同形式)
  - regime_guard_overrides の trending_up: `ev_threshold_premium_bps: 0.3`, `spread_as_guard_penalty_multiplier: 1.5` 有効化
- `scripts/v460/lib/skip_gate_evaluator.py`: sell_trending_up_offset 適用
- `scripts/v460/lib/fill_config.py`: 新パラメータ定義
- `scripts/v460/lib/fill_config_validation.py`: バリデーション追加

**設計根拠**:
- sell_ranging_offset (0.5) は既存パターン。trending_up でも同等のペナルティを適用
- regime_guard_overrides は enabled=false だが、trending_up 個別有効化で低リスク
- spread_as_guard (今回有効化済) の multiplier=1.5 × ev_penalty=0.5 = 0.75 bps ペナルティ

**テスト**:
- `tests/unit/v460/test_702_sell_trending_up_guard.py`
- offset 適用の数値検証
- regime_guard_overrides との interaction
- 既存 sell_ranging_offset との共存

---

## Task 3: 12-17h UTC 時間帯パラメータ再調整 (P0)

**目的**: 12-17h UTC (JST 21-02h) の損失集中を既存フレームワークで緩和

**修正箇所**:
- `configs/v460/fill_test.yaml`:
  - `skip_gate_hour_offsets`: 12h=0.3, 13-15h=既存維持, 16h=0.5 (AS64% 対応)
  - `sell_hour_offset_boost`: 14h=2.0→2.5 (P688: -6.05bps n=4), 16h=1.5→2.5 (P688: -3.41bps n=22)
  - `hour_ceiling_mult`: 12h=新規1.5, 16h=2.0→2.5 (ceiling 拡大で防御許容)
- 変更なしのファイル: 既存のフレームワークを使うため新規コードは不要

**設計根拠**:
- P688 時間帯別データ:
  - 14h: n=4, avg=-6.05bps → 現行 1.5 では不十分。2.5 に強化
  - 16h: n=22, avg=-3.41bps → 現行 1.5 では不十分。2.5 に強化
  - 12h: n=20, avg=-1.73bps → skip_gate_hour_offsets 0.3 追加
- 他の時間帯 (13h, 15h, 17h) は既存設定で対応可能な範囲

**テスト**:
- `tests/unit/v460/test_702_hour_param_update.py`
- YAML 値の正確性チェック
- sell_hour_offset_boost × hour_ceiling_mult の範囲テスト
- 既存 skip_gate_hour_offsets との整合性

---

## Codex 投入順序

```
Phase 1 (同時投入):
  Task 2 (sell/trending_up ガード) ← 最大効果、低リスク
  Task 3 (時間帯再調整)            ← config のみ、低リスク

Phase 2:
  Task 1 (SHA テレメトリ)          ← 分析基盤改善
```

---

## 見送りタスク

| タスク | 理由 |
|--------|------|
| NFQ ranging 発生量削減 | 根本原因は maker_price の quote 計算。大規模調査が必要 |
| spread bucket cancel 分析 | 62% データ欠損の解消にはログフォーマット変更が必要 |
| b56771a 個別分析 | 既に retired SHA。遡及分析の ROI 低 |

---

*生成: 2026-04-03 by cplt (703#)*
*入力: 702# 重大発見 A-D*

---

## 実装レビュー結果

**ステータス: 実装済み（current runtime に整合する形へ補正）**

### prompt 検証メモ

- `task1` は概ね妥当だったが、現行 runtime の adverse selection フィールドは `is_adverse` ではなく `adverse_selected` が正本
  - 実装では両方を吸収し、旧ログも壊さない形にした
- `task2` はそのまま当てると過剰実装になりやすかった
  - 現行 live path には既に
    - `skip_sell_trending_up_only`
    - `trending_up_sell_offset_boost`
    - `spread_as_guard`
    があるため、今回は不足していた `skip_gate` 側の `sell_trending_up_offset` のみを追加
- `task2` の regime override は「trending_up だけ個別有効化」ではなく、現行 `FillTestConfig` 契約に合わせて
  - global enable
  - trending_up 側の premium / multiplier 調整
  の形に寄せた
- `task3` は YAML 変更だけに見えたが、hidden task として `hour_ceiling_mult` の nested parse drift 修正が必要だった

### hidden task

- `hour_ceiling_mult` は live YAML では `skip_gate` 配下が正本なので、parser を nested 優先へ修正
- hot-reload 対象に
  - `skip_gate_sell_ranging_offset`
  - `skip_gate_sell_trending_up_offset`
  を追加して、値変更が runtime へ反映されるようにした
- YAML drift allowlist に
  - `hour_ceiling_mult`
  - `skip_gate_sell_trending_up_offset`
  - `spread_as_guard_enabled`
  - `regime_guard_overrides_enabled`
  を追記して source-contract を現行 runtime に合わせた

### 横展開

- SHA telemetry は `protocol_688` だけでなく、legacy `is_adverse` を読む履歴にも後方互換を持たせた
- sell/trending_up offset は validation / parser / hot-reload / live YAML / tests まで一気に通した
- hour retune は YAML 値更新だけで終わらせず、parser drift と validation まで揃えた

### 回帰

- focused 703/config/protocol subset:
  - `140 passed in 3.11s`

---

## 703# 最終レビュー (ff245eacb 時点)

**ステータス: 全タスク実装済み + クリティカルバグ修正済み**

### クリティカルバグ: YAML duplicate `hour_offsets` キー

Codex が `hour_offsets:` ブロックを新規追加したが、既存ブロックと **重複キー** になっていた。
YAML 仕様では last-key-wins のため、元の 8 エントリ (13,14,15,16,17,18,21,23) が **サイレントに消失** し、
`{12: 0.3}` のみが残る状態だった。

**修正** (ff245eacb): 2 ブロックを 1 ブロックに統合。全 9 エントリ (12-18,21,23) を正しく保持。

### 実装完全性レビュー

| コンポーネント | ステータス | 詳細 |
|---|---|---|
| YAML config | ✅ | sell_trending_up_offset=0.5, hour_offsets 9 エントリ, ceiling_mult |
| fill_config.py 型定義 | ✅ | 全フィールド型安全、デフォルト値適切 |
| fill_config_validation.py | ✅ | sell_trending_up_offset [0, 2] 範囲検証 |
| skip_gate_evaluator.py | ✅ | sell+trending_up 条件で offset 適用、テレメトリ付 |
| regime_guard_overrides | ✅ | entry_gate_adjustments.py で ev_premium + penalty_multiplier 適用済 |
| SHA テレメトリ | ✅ | protocol_688 に SHA 別 AS 率・PnL 寄与セクション追加 |
| hot-reload 対応 | ✅ | sell_trending_up_offset, regime_guard params を reload 対象に追加 |
| YAML drift allowlist | ✅ | hour_ceiling_mult, sell_trending_up_offset 等を追記 |
| テスト | ✅ | test_703_* 3 ファイル + test_183/336 更新。全 5895 パス |

### regime_guard_overrides 適用フロー (確認済)

```
fill_test.yaml
  └─ regime_guard_overrides.trending_up
       ├─ ev_threshold_premium_bps: 0.3
       └─ spread_as_guard_penalty_multiplier: 1.5
           ↓ (fill_config_parser.py)
fill_config.regime_guard_ev_threshold_premiums["trending_up"] = 0.3
fill_config.regime_guard_spread_as_penalty_multipliers["trending_up"] = 1.5
           ↓ (fill_config.get_regime_guard_override())
RegimeGuardOverride(enabled=True, ev_premium=0.3, penalty_mult=1.5)
           ↓ (entry_gate_adjustments.py)
adjusted_ev = base_ev - (spread_penalty × 1.5) - 0.3
```

### 有効化された全機能一覧 (701#-703#)

| 機能 | 内容 | リスク |
|------|------|--------|
| spread_as_guard | EV penalty 0.5bps × regime_mult (15bps 閾値) | 低: observe→apply 切替のみ |
| regime_exit_strategy | trending_down + buy>10 + imbalance≥0.3 で段階介入 | 低: 3 条件 AND |
| as_trailing_gate | AS trail 追跡 (既存パイプライン) | 低: 699# 検証済 |
| regime_guard_overrides | trending_up: ev+0.3bps, spread_penalty×1.5 | 中: EV 厳格化 |
| sell_trending_up_offset | skip_gate offset +0.5 (sell+trending_up) | 低: sell_ranging 同等 |
| hour retune | 14h/16h boost 2.5, 12h skip 0.3 + ceiling 1.5 | 低: 既存パラメータ調整 |

### 残存リスク

1. **同時有効化**: 5 機能が同時有効化。相互作用の実測データは未取得
2. **b56771a 交絡**: P688 損失の 88.6% を占めた SHA は retired。現行 SHA での再現性不明
3. **sell/trending_up 二重防御**: skip_gate offset + regime_guard premium + spread_as_guard multiplier が重畳するケースあり。過剰防御で機会損失の可能性

*レビュー: 2026-04-03 by cplt (703# final review)*
