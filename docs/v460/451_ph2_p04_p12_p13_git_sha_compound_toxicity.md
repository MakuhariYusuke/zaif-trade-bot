# 451# P0-4 / P1-2 / P1-3 実装

**日付**: 2026-03-16  
**前提**: 450# (`52627ffe1`) — P0-1~P0-3 完了, fill_test PID 41152 稼働中

---

## §1 P0-4: ab_offset_comparison.py git_sha / run_id フィルタ

### 背景
447# F2 → 450# §5 P0-4 繰越。  
`ab_offset_comparison.py` は日付 (`--split-date`) でしか Before/After 分割できず、
mixed-SHA データで A/B 比較が汚染される問題があった。

### 変更内容
| ファイル | 変更 |
|---|---|
| `scripts/v460/analysis/ab_offset_comparison.py` | `apply_fill_record_filters` import 追加。`_load_records()` に `git_sha`/`run_id` kwargs 追加。`_save_baseline`, `_show_baseline`, `_run_comparison` に filters 貫通。CLI に `--git-sha`, `--run-id` 引数追加。JSON 出力に `filters` メタデータ追加。 |
| `tests/unit/v460/test_441_ab_offset_comparison.py` | `TestLoadRecordsFilter` クラス追加 (4テスト: no_filter, git_sha_filter, run_id_filter, combined_filter) |

### 使用例
```bash
# 特定 SHA のレコードのみで比較
python scripts/v460/analysis/ab_offset_comparison.py --compare --git-sha 52627f

# run_id + split-date 複合フィルタ
python scripts/v460/analysis/ab_offset_comparison.py --compare --run-id abc123 --split-date 2026-03-16
```

### テスト結果
- 15/15 passed (既存 11 + 新規 4)

---

## §2 P1-2: ranging_low_vol_skip × buy_dynamic_kill compound suppression 分離

### 背景
450# §5 P1-2: `ranging_low_vol_skip` と `buy_dynamic_kill` が同時に発火する
compound suppression の頻度が不明。Gate 2 (ranging_low_vol) が先にブロックするため、
Gate 4 (buy_dynamic_kill) の共起は可視化されていなかった。

### 設計判断
**Gate 順序 (2→4) は変更しない。** 代わりに:
1. Gate 2 がブロック時、Gate 4 を**投機的に評価**して `speculative_checks` に記録
2. orchestrator が `compound_{gate_name}` カウンタを guard_fire に記録
3. `audit_summary` に投機的チェック結果を `(✗buy_dynamic_kill)` 形式で表示

### 変更内容
| ファイル | 変更 |
|---|---|
| `scripts/v460/lib/cycle_gate_aggregator.py` | `CycleGateResult.speculative_checks` フィールド追加。Gate 2 ブロック時に Gate 4 投機的評価。`audit_summary` に speculative 結果追加。 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | `compound_{gate_name}` guard fire カウンタ記録追加。 |
| `tests/unit/v460/test_194_cycle_gate.py` | `TestCompoundSuppression` クラス追加 (3テスト) |

### 期待される効果
- `guard_fire_counts` に `compound_buy_dynamic_kill` が出現
  → ranging_low_vol と buy_dynamic_kill の共起頻度が定量的に判明
- データ蓄積後、共起率が高ければ Gate 統合 or 条件分岐を検討

### テスト結果
- 38/38 passed (既存 35 + 新規 3)

---

## §3 P1-3: heuristic toxicity 再校正 — toxicity_budget 有効化

### 背景
450# §5 P1-3: 440# で ML toxicity veto を棄却後、既存 heuristic toxicity
(240# Glosten-Milgrom 段階的応答) のパラメータが再校正されていなかった。
`toxicity_budget_enabled: false` (デフォルト) のまま binary kill のみ動作。

### 変更内容: configs/v460/fill_test.yaml
`sell_dynamic_kill` と `buy_dynamic_kill` それぞれに toxicity budget パラメータを追加。

| パラメータ | sell | buy | 根拠 |
|---|---|---|---|
| `toxicity_budget_enabled` | `true` | `true` | 段階的応答の有効化 |
| `toxicity_warn_level` | 0.3 | 0.3 | デフォルト維持 |
| `toxicity_caution_level` | 0.7 | 0.7 | デフォルト維持 |
| `toxicity_warn_offset_mult` | **1.2** | **1.3** | buy は逆選択リスク高→売より厚め |
| `toxicity_caution_offset_mult` | **1.8** | **2.0** | デフォルト(2.0)からsell側をやや緩和 |
| `toxicity_kill_offset_mult` | **2.5** | **2.5** | デフォルト(3.0)→2.5: kill到達時はblock十分 |
| `toxicity_caution_min_participation` | **0.5** | **0.4** | sell: 50%参加, buy: 40%参加 (過抑制防止) |

### side-aware 非対称性の根拠
- **buy (特に ranging)**: PF=0.766 (432# 分析), 逆選択リスク支配的 → offset 厚め, 参加率絞り気味
- **sell**: trending_up 時の順張り方向 → 相対的に寛容

### テスト結果
- 57/57 passed (既存テスト全パス、設定検証は YAML 構文チェック + パース確認で実施)

---

## §4 次のステップ

| 優先度 | 項目 | 状態 |
|---|---|---|
| P1-1 | same-SHA cross-venue metrics 再測定 | 🕐 clean-SHA データ蓄積待ち |
| P2-1 | 447# Micro-Timeout 先行試験 | ✅ 452#/453# で実装済み |
| 観察 | compound suppression (`compound_buy_dynamic_kill`) 頻度 | 🔄 データ蓄積中 |
| 観察 | toxicity budget YELLOW/ORANGE 発火頻度と PnL 影響 | 🔄 データ蓄積中 |
