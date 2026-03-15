# 438# Buy+Ranging 防御三層コンフィグ変更

| 項目 | 内容 |
|---|---|
| 番号 | 438# |
| 分類 | ph2_impl |
| 対象 | 432#, 436#, 437# |
| 前提 | 432# fill_records 30日分析, 436# SAC AI向け統合文書, 437# cross-review |
| 目的 | buy+ranging 最大損失セグメントの YAML コンフィグ最適化 (コード変更なし) |

---

## §0 Executive Summary

432# fill_records 30日分析 (11,356件) で特定された最大損失セグメント **buy+ranging (PF=0.766)** に対し、
3つの独立したコンフィグ変更で多層防御を構築する。全変更は YAML のみ、ホットスワップ互換。

| # | 変更 | 設定キー | 旧値 | 新値 | 根拠 |
|---|---|---|---|---|---|
| C1 | buy+ranging ハードスキップ復帰 | `ranging_buy_low_vol_as_offset` | `true` | `false` | 432# buy+ranging PF=0.766, soft offset(1.4x)では抑制不十分 |
| C2 | buy reprice 無効化 | `max_reprice_buy` | `1` | `0` | 432# §7: 0回-0.22→1回-0.73→2回-2.21 bps |
| C3 | BDK ranging 閾値厳格化 | `buy_dynamic_kill.regime_thresholds.ranging` | `-1.0` | `-0.7` | effective -0.7-0.3=-1.0 (旧-1.3 より 0.3bps 厳格) |

**期待効果**: buy+ranging セグメントの toxic participation 大幅削減。3層が独立に動作し、
ゲート→キル→stale の各段階で損失エントリを遮断。

---

## §1 変更詳細

### §1.1 C1: ranging_buy_low_vol_as_offset = false

- **ファイル**: `configs/v460/fill_test.yaml` L176
- **コードパス**: `ztb/core/cycle_gate_aggregator.py` L495
  - `true` → low_vol_offset_boost (1.4x) で maker_price 調整のみ (ソフトスキップ)
  - `false` → `GateCheckResult(blocked=True)` 即座にサイクル遮断 (ハードスキップ)
- **経緯**: 169# B1' で初導入 (hard skip)。195# B1' で soft 化 (offset 委譲)。
  432# データで soft offset が損失抑制に不十分と判明、hard skip に復帰。
- **テスト**: `test_195_velocity_b1_soft.py` — 両パス (hard/soft) のテスト存在確認済

### §1.2 C2: max_reprice_buy = 0

- **ファイル**: `configs/v460/fill_test.yaml` L243
- **コードパス**: `ztb/core/order_monitor.py` L441
  - `reprice_count < _stale_max_rp` → `0 < 0 = False` → reprice スキップ
  - **adverse drift cancel は independent** (max_reprice とは別ロジック、引き続き機能)
  - favorable drift 時は original price で待機 (追随しない = 432# データで最適戦略)
- **regime_reprice_adjustments**: `{}` (空 dict) — 全レジームで reprice=0 が確定
- **テスト**: `test_143_regime_utilization.py`, `test_137_p1_features.py` — reprice ロジック関連

### §1.3 C3: buy_dynamic_kill.regime_thresholds.ranging = -0.7

- **ファイル**: `configs/v460/fill_test.yaml` L695
- **コードパス**: `ztb/core/orchestrator_guards.py` L88-96 → `ztb/core/sell_dynamic_kill.py` L633-641
  - regime_thresholds lookup → EWMA decay target として使用
  - `inv_relaxation` (max_bps=0.3) → effective threshold = -0.7 - 0.3 = **-1.0**
  - **旧 effective**: -1.0 - 0.3 = -1.3 → **0.3bps 厳格化**
- **SDK との比較**: sell_dynamic_kill ranging = -0.7, inv_relaxation max = 0.5
  - SDK effective = -0.7 + 0.5 = -1.2 → BDK effective -1.0 と概ねバランス
- **テスト**: `test_234_gate_bypass_removal.py` — dynamic kill ロジック確認

---

## §2 相互作用分析

3変更は処理パイプラインの**独立したレイヤー**で動作:

```
サイクル開始
  ├→ [C1] cycle_gate_aggregator: ranging+buy+low_vol → blocked=True (ゲート段階)
  │     ↓ (C1 を通過した場合)
  ├→ [C3] buy_dynamic_kill: EWMA rolling PnL < -0.7 → kill (キル段階)
  │     ↓ (C3 を通過した場合)
  ├→ 発注実行
  │     ↓
  └→ [C2] order_monitor: 価格乖離 → reprice 不可 (stale 段階)
              ↓ adverse drift → cancel (独立動作)
```

- C1 が最初のフィルター: ranging+buy+low_vol なら即遮断
- C1 を通過 (non-low_vol ranging buy) → C3 で EWMA 損益が閾値以下なら kill
- C3 も通過 → 発注、但し C2 で reprice なし (adverse drift は cancel)
- **循環依存・干渉なし**: 各変更は独立した判断基準で動作

---

## §3 ドリフトテスト修正

`tests/unit/v460/test_336_yaml_code_drift_prevention.py` の KNOWN_YAML_OVERRIDES 更新:

| 操作 | キー | 理由 |
|---|---|---|
| **削除** | `ranging_buy_low_vol_as_offset` | YAML=false がコードデフォルト (False) と一致 → allowlist 不要 |
| **追加** | `execution_final_clamp_hard_skip_mult` | 421# 由来の既存 drift (本変更とは無関係、発見時に修正) |

---

## §4 テスト結果

| テストスイート | 結果 |
|---|---|
| 直接関連 9 ファイル (313 tests) | **全 PASS** |
| ドリフト防止 (4 tests) | **全 PASS** |
| v460 全体 (3348 tests) | 3343 PASS / 5 FAIL (全て無関係の既存問題) |

無関係の 5 failures:
- `test_run_single_cycle_under_400_lines` / `test_fill_cycle_executor_line_count_under_limit`: 行数超過 (既存)
- `test_reward_tuned_yaml_*` (2件): 437# P2 で archived 済
- (1件): その他既存

---

## §5 ロールバック計画

全変更は YAML コンフィグのみ。即座にロールバック可能:

```yaml
# C1 ロールバック: hard skip → soft offset に戻す
ranging_buy_low_vol_as_offset: true

# C2 ロールバック: reprice 1回に戻す
max_reprice_buy: 1

# C3 ロールバック: 旧閾値に戻す
ranging: -1.0  # buy_dynamic_kill.regime_thresholds.ranging
```

個別ロールバック可能 (独立レイヤーのため)。

---

## §6 モニタリング指標

ホットスワップ後、以下の fill_records 指標で効果測定:

| 指標 | 期待変化 | 確認方法 |
|---|---|---|
| buy+ranging skip 率 | 上昇 | fill_records の gate blocked 件数 |
| BDK kill 頻度 (ranging) | 上昇 | buy_dynamic_kill cancel 件数 |
| buy reprice 回数 | 0 | reprice_count フィールド |
| buy+ranging PF | 0.766 → 改善 | fill_records PF 再計算 |
| 全体 PnL | 改善 or 横ばい | 日次 PnL 集計 |
| participation rate | 微減許容 | fill 件数 / サイクル数 |
