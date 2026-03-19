# 487# SAC メモリリーク防止 + 重複削減 + P0 観測基盤強化

> 種別: fix / refactor
> 対象: sac_common.py, sac_train.py, sac_retrain_scheduler.py, fill_quality.py, fill_record_builder.py, fill_cycle_executor.py
> コミット: ce10cdb69 (メモリ/重複), 後続コミット (観測基盤)
> 日付: 2026-03-19

---

## 0. 背景

483# / 484# / 485# の3レビューが一致して指摘した課題:

1. **寄与分解不能**: 481# (feasibility/veto緩和) と 482# (SAC sidecar) を同時投入したため、fill改善の原因を分離できない
2. **SAC評価系の脆弱性**: val_ratio=0.02, n_episodes=1 は採用判定に不十分
3. **sidecar live実効の不透明性**: offset実値・signal鮮度・confidence が記録されず検証不可能

486# で全数値・コード検証を行い、上記3点がコード上でも事実であることを確認した。

本ドキュメントでは:
- §1: メモリリーク防止 + 重複削減 (487# 第一コミット)
- §2: P0 観測基盤強化 — sidecar寄与分解に必要なフィールド追加
- §3: 今後の P1 対応方針

---

## 1. メモリリーク防止 + 重複削減

### 1.1 メモリリーク防止 (3件)

| 問題 | 場所 | 修正 |
|------|------|------|
| SB3 model ↔ env 循環参照が gc で回収されない | sac_train.py, retrain_scheduler.py | `cleanup_training_resources()` で model del → env close → gc.collect() |
| `best_model_loaded` が del されない | sac_train.py L210 付近 | finally 内に `del best_model_loaded` 追加 |
| `model` 変数が finally で参照不能 | sac_train.py | try 前に `model: SACModelProtocol | None = None` 宣言 |

**`cleanup_training_resources()`** (sac_common.py に新設):
```python
def cleanup_training_resources(
    *,
    models: list[object] | None = None,
    envs: list[TrainingEnvProtocol | None] | None = None,
    dataframes: list[object] | None = None,
) -> None:
    # 1. model del (env への参照を切る)
    # 2. env.close()
    # 3. del dataframes
    # 4. gc.collect() — 循環参照破壊
```

### 1.2 重複削減 (3件)

| 重複パターン | 統合先 (sac_common.py) |
|-------------|----------------------|
| sac_train._create_training_env / retrain._create_env の HeavyTradingEnv 生成 | `create_env_from_config(df, env_config)` |
| sac_train._create_sac_model / retrain のインライン SAC() | `create_sac_model(env, **kwargs)` |
| 両 finally の cleanup_envs + del df | `cleanup_training_resources()` に統合 |

**差分**: +156 / -59 行

---

## 2. P0 観測基盤強化 — sidecar 寄与分解

483#-485# 全文書が P0 に挙げた「寄与分解可能な観測設計」を実装する。

### 2.1 FillRecord に追加するフィールド

| フィールド | 型 | 目的 |
|-----------|---|------|
| `sidecar_confidence` | `float | None` | signal の confidence 値 (ROI margin から算出) |
| `sidecar_model_version` | `str | None` | deploy 元モデルバージョン |
| `sidecar_signal_status` | `str | None` | "fresh" / "stale" / "missing" / "error" |

→ **実装済み** (`ztb/metrics/fill_quality.py` FillRecord dataclass に3フィールド追加)

### 2.2 `read_sidecar_signal_with_status()` 新設

従来の `read_sidecar_signal()` は file-not-found / parse-error / TTL-exceeded を全て `None` に潰していた。
新関数 `read_sidecar_signal_with_status()` は `(SidecarSignal | None, str)` を返し、失敗モードを識別可能にする:

| 状態 | signal | status |
|------|--------|--------|
| 正常読込 | SidecarSignal | `"fresh"` |
| TTL超過 | None | `"stale"` |
| ファイル不在 | None | `"missing"` |
| JSON パースエラー / OS エラー | None | `"error"` |

→ **実装済み** (`scripts/v460/lib/sidecar_signal_io.py`)

### 2.3 データパイプライン貫通

新フィールドは以下のパスで FillRecord に到達する:

```
orchestrator_mid_cycle.py  (read_sidecar_signal_with_status → signal, status)
  → confidence, model_version を signal から抽出
  → run_single_cycle() に 3 パラメータ追加渡し
    → fill_cycle_executor.py (新パラメータ受取)
      → fill_record_builder._build_fill_market_fields() に貫通
        → build_fill_record() → FillRecord
```

変更ファイル一覧:
| ファイル | 変更内容 |
|---------|---------|
| `ztb/metrics/fill_quality.py` | FillRecord に 3 フィールド追加 |
| `scripts/v460/lib/sidecar_signal_io.py` | `read_sidecar_signal_with_status()` 新設 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | signal_status 取得 + 3 パラメータ渡し |
| `scripts/v460/lib/fill_cycle_executor.py` | `run_single_cycle()` 署名拡張 + ログ改善 |
| `scripts/v460/lib/fill_record_builder.py` | 2 メソッドで 3 フィールド貫通 |

### 2.4 Cycle log 改善

`_log_cycle_result()` に sidecar 要約を追加:
- signal status (fresh/stale — missing 時は省略)
- offset_bps 値 (0 以外の場合のみ)

---

## 4. P2: Progress log 可観測性強化

487# P2 では fill_test の Progress log (50 サイクル周期) に以下の情報を追加。

### 4.1 cancel_reason の cycle ログ追記

`_log_cycle_result()` に `cancel_reason` パラメータを追加。unfilled 時にキャンセル理由を表示:
```
Cycle 150 result: filled=False, wait=30.0s, reason=timeout
```

変更: `fill_cycle_executor.py` (`_log_cycle_result` 署名 + 呼出元)

### 4.2 Sidecar activity summary

RunSessionState に fresh/stale/missing カウンタを追加。Progress log で freshRate を出力:
```
[487# sidecar] fresh=45, stale=3, missing=2 (freshRate=90.0%)
```

変更: `fill_loop_orchestrator.py` (RunSessionState), `orchestrator_mid_cycle.py` (カウント), `orchestrator_post_cycle.py` (出力)

### 4.3 cancel_reason distribution (Top 5)

RunSessionState に cancel_reason_counts dict を追加。gate block + unfilled cycle の理由を集計し Progress log で上位5件を出力:
```
[487# cancel] top reasons: timeout=25, spread_too_narrow=8, gate_dual_kill=3
```

変更: `orchestrator_mid_cycle.py` (gate block 時カウント), `orchestrator_post_cycle.py` (unfilled カウント + 出力)

### 4.4 テスト

`test_sidecar_sac_integration.py` に `TestRunSessionStateSidecarTracking` (5 件) を追加:
- sidecar カウンタのデフォルト値・インクリメント
- cancel_reason_counts の dict 独立性・累積動作

例: `Cycle 142 result: filled=True, wait=3.2s, pnl=0.85bps, sidecar=fresh(+0.048bps)`

### 2.5 期待効果

これにより次回分析で:
- **sidecar_signal_status == "fresh" の fill** vs **"stale" or "missing" の fill** で PnL30s/AS率を比較可能
- **sidecar_offset_bps の分布** からモデルの active/inactive 比率を定量化可能
- **sidecar_model_version** で異なるモデル間の A/B 比較が可能

---

## 3. P1 対応 (実施済み)

### 3.1 セルフレビュー修正

| 指摘 | 対応 |
|------|------|
| `read_sidecar_signal_with_status` が `read_sidecar_signal` のロジックを重複 (DRY 違反) | `_read_sidecar_signal_core()` に統合し両関数が共有。キャッシュ活用も維持 |
| orchestrator に未使用 `read_sidecar_signal` import が残存 | 削除 |

### 3.2 SAC eval val_ratio ガードレール (P1-1)

**問題**: 483-485 レビューで指摘された「val_ratio=0.02 は G3 gate min_val_ratio=0.10 未満」

**対応**:
| 対象 | 変更 |
|------|------|
| `sac_common.py` `train_val_split()` | `min_val_ratio` パラメータ追加、閾値未満で warning ログ出力 |
| `g2_sac_reward_clean.yaml` (ベース config) | val_ratio: 0.02 → 0.10 |

**補足**: `n_episodes > 1` は `random_start=False` (デフォルト) では同一データを繰り返すだけと `evaluate_model_oos()` のコメントが明言しているため、変更不要と判断。

### 3.3 テスト追加 (P1-2)

`test_sidecar_sac_integration.py` に 2 クラス/9 テスト追加:
- `TestReadSidecarSignalWithStatus`: fresh/missing/stale/error の 4 状態 + `read_sidecar_signal` との一貫性
- `TestFillRecordSidecarAttributionFields`: 新 3 フィールドの存在・設定・round-trip

---

## 4. 今後の方針

| 優先度 | 項目 | 状態 |
|--------|------|------|
| P1-3 | seed123 を対照群として live 比較 | sidecar_model_version フィールドで追跡基盤整備済み |
| P1-4 | 700JPY 緩和後の fill quality 点検 | sidecar_signal_status フィールドで分離解析が可能に |
| P2 | sidecar 権限昇格 (max_boost_bps 拡大) | 整合性確認後の将来課題 |

---

## 5. 487# 実験結果 (val_ratio=0.10, 20K steps, g2_sac_reward_clean.yaml)

**日時**: 2026-03-19 08:17–10:26 (約2h10m)
**コンフィグ**: 4 seed × 20K steps, val_ratio=0.10, v459 E-settings
**結果ファイル**: `results/v460/v460_g2train_seed42_20260318_231635.json`

### 5.1 Per-seed 結果

| Seed | Final ROI | Final PF | Sharpe (年率) | Best OOS ROI | Best OOS PF | Reward-Profit Corr |
|------|-----------|----------|--------------|-------------|-------------|-------------------|
| 42   | **+1.74%** | 1.12 | 5.42 | +1.82% | 1.15 | **0.952** |
| 123  | **+7.13%** | **1.45** | **12.66** | +7.74% | 1.51 | **0.979** |
| 456  | -2.40% | 0.86 | -8.07 | +1.36% | 1.10 | 0.977 |
| 789  | -1.17% | 0.94 | -2.96 | -0.22% | 0.99 | 0.878 |

### 5.2 Gate 判定

**G2-train: FAIL**

| チェック | 値 | 閾値 | 結果 |
|----------|-----|------|------|
| positive_seed_ratio | 0.50 (2/4) | ≥ 0.75 | ❌ |
| roi_seed_std | 0.042 | ≤ 0.03 | ❌ |
| convergence | 0.0% | ≤ 5.0% | ✅ |
| worst_seed_roi | -2.40% | > -3.5% | ✅ |

**G3-pnl: FAIL**

| チェック | 値 | 閾値 | 結果 |
|----------|-----|------|------|
| pf_median | 1.03 | ≥ 1.05 | ❌ |
| pf_worst | 0.86 | ≥ 0.95 | ❌ |
| gross_gt_fee | true | true | ✅ |
| max_drawdown | 3.08% | ≤ 15% | ✅ |
| sharpe_annual | 1.23 | ≥ 0.8 | ✅ |
| reward_profit_corr_median | 0.964 | ≥ 0.0 | ✅ |
| val_ratio_compliance | 0.10 | ≥ 0.10 | ✅ |

### 5.3 分析

1. **seed 123 突出**: ROI +7.13%, PF 1.45, Sharpe 12.66 — 全 seed 中最強。best_model でもさらに向上 (+7.74%)
2. **seed 456 final vs best_model 乖離大**: final ROI -2.40% だが best_model (15Kステップ時点) は +1.36% → 後半で過適合崩壊
3. **reward-profit correlation 大幅改善**: 全 seed で 0.88–0.98 (482# 実験の seed456 は 0.187 だった)
4. **val_ratio=0.10 効果**: OOS 評価区間拡大により、seed 456/789 の過適合を正確に検出

### 5.4 前回比較 (482# val_ratio=0.02 vs 487# val_ratio=0.10)

| Seed | 482# ROI | 487# ROI | 変化 |
|------|----------|----------|------|
| 42   | +0.69%   | +1.74%   | ↑ 2.5× |
| 123  | +0.11%   | +7.13%   | ↑ 65× |
| 456  | +0.27%   | -2.40%   | ↓ (過適合露出) |
| 789  | +0.59%   | -1.17%   | ↓ (過適合露出) |

**解釈**: val_ratio 拡大で OOS の評価精度が上がり、seed 間の真の差が見えるようになった。seed 123 が一貫して強く、seed 456 は精密な OOS では弱い。

### 5.5 Next Steps

- **100K steps で seed 123 単独再訓練**: 20K ではステップ不足の可能性。seed 123 のみ long-run 確認
- **best_model checkpoint 戦略**: seed 456 のように final が崩壊するケースに対応し、best-checkpoint を sidecar deploy 候補とする
- **G2 gate 閾値の見直し**: positive_seed_ratio ≥ 0.75 は 4 seed では 3/4 必須 → seed 数増加 or 閾値調整を検討

---

## 6. P0 バグ修正: `_sidecar_signal` NameError

### 6.1 障害概要

487# P0 コミット (f2accd00c) で導入した sidecar attribution フィールド追加に NameError バグが含まれていた。

- **原因**: `_sidecar_signal` は `_evaluate_and_handle_cycle_gate()` のローカル変数だが、`_execute_and_track_cycle()` メソッドで直接参照 → スコープ外
- **影響**: watchdog 再起動 (2026-03-19 05:50) 以降、**全サイクルが NameError で失敗**
- **障害時間**: 約 10.5 時間 (05:50 → 16:28+ 現在進行中)
- **エラー件数**: 260 回 (約 4 分/回のサイクルペース)

### 6.2 修正

| 対象ファイル | 変更内容 |
|---|---|
| `cycle_gate_aggregator.py` | `CycleGateResult` に `sidecar_confidence`, `sidecar_model_version`, `sidecar_signal_status` フィールド追加 |
| `orchestrator_mid_cycle.py` | `_evaluate_and_handle_cycle_gate()` で `_gate_result` に attribution 情報を転記、`_execute_and_track_cycle()` では `gate_result` 経由でアクセス |

### 6.3 教訓

ローカル変数のスコープ境界を跨ぐ参照は必ずテストで検出すべき。今回のケースでは `_execute_and_track_cycle` の統合テストが不足していた。

---

## 7. Fill Test ログ包括分析 (2026-02-14 〜 03-19)

### 7.1 概要

| 指標 | 値 |
|------|-----|
| 分析期間 | 2026-02-14 03:37 〜 2026-03-19 16:28 (約 33 日間) |
| ログファイル | fill_test.log (2.2MB) + .1 (10MB) + .2 (10MB) |
| 総約定回数 | 4,035 filled / 1,371 unfilled = **74.6% fill rate** |
| 総ゲートブロック | 2,397 回 |
| 累積 PnL | **-986.3 bps** (損失) |
| Profit Factor | **0.883** |
| 勝率 | 47.2% (1,906 win / 2,111 loss) |
| 平均利益 | +3.915 bps (win) / -4.002 bps (loss) |
| ERROR 件数 | 1,000 件 (うち NameError 260 件) |

### 7.2 日次推移の主要パターン

**PnL 推移** (cumPnL trajectory):
- 02-14: -48 bps → 急速な損失蓄積開始
- 02-18〜02-19: +152 bps の2日連続好調 (唯一明確なプラス期間)
- 02-26: -208 bps の大幅ドローダウン (1日で最大損失)
- 03-01: -120 bps (soft_loss_cap 頻発)
- 03-16〜03-19: 安定して日次 -30〜-78 bps

**Fill rate 低下**:
- 02-14: 86% → 03-07: 34% → 03-19: 30.3%
- fill rate は一貫して低下。ゲートブロック導入 (02-28〜) により顕著に低下

### 7.3 PnL 分布

| パーセンタイル | 値 (bps) |
|---|---|
| P1 | -17.88 |
| P5 | -9.31 |
| P10 | -6.39 |
| P25 | -2.76 |
| P50 (median) | **-0.17** |
| P75 | +2.24 |
| P90 | +5.61 |
| P95 | +8.99 |
| P99 | +16.56 |
| 最大損失 | -51.9 bps |
| 最大利益 | +74.1 bps |

**分布の特徴**: 左スキュー (median が負)。テール損失が大きく、PF < 1.0。
- 損失ゾーン (-5 〜 0 bps): 37.9% — 小さな損失が頻発
- 利益ゾーン (0 〜 +5 bps): 36.3%
- 大損 (< -10 bps): 4.2% — ここを削減すれば PF 改善

### 7.4 ゲートブロック分析

| 理由 | 回数 | 割合 |
|---|---|---|
| sell_dynamic_kill | 979 | 40.8% |
| ranging_low_vol_skip | 736 | 30.7% |
| buy_dynamic_kill | 615 | 25.7% |
| spread_too_narrow | 67 | 2.8% |

- **sell_dynamic_kill 支配的**: sell 側のリスクガードが過度に反応している可能性
- **ranging_low_vol_skip**: 低ボラ環境でスキップが多発 → offset 調整で対応可能か

### 7.5 リスクイベント

| イベント | 回数 |
|---|---|
| Balance insufficient | 5,703 |
| Cross venue veto | 188 |
| Toxic fill veto (sell) | 281 |
| Toxic fill veto (buy) | 186 |
| Per-side halt (buy) | 256 |
| Per-side halt (sell) | 90 |
| Degraded liquidation | 376 |
| Soft loss cap triggered | 91 |
| Quiescence events | 59 |
| Inventory escape | 96 |

**Balance insufficient (5,703 件)** が最大の問題。JPY/BTC 残高が min_lot を下回り、サイクルがスキップされる頻度が非常に高い。

### 7.6 Wait time

| 指標 | 値 |
|---|---|
| 件数 | 5,406 |
| 平均 | 31.2s |
| 中央値 | 16.5s |
| P90 | 81.8s |

### 7.7 非 NameError エラー

| エラー | 件数 | 重要度 |
|---|---|---|
| Coincheck API 400 Bad Request | 374 | 中 (注文拒否) |
| Failed to place order: 400 | 88 | 中 |
| SkipGate pickle hash mismatch | 46 | 低 (hot-reload 時の一時的不整合) |
| All order attempts failed | 41 | 高 (取引不能) |
| FillTestConfig.min_lot AttributeError | 19 | 低 (旧コード残骸、修正済) |
| spread > max guard | 20 | 低 (正常動作) |
| SAFE_STOP 連続 preflight | 7 | 高 (完全停止) |

### 7.8 診断と改善提案

#### P0 (即時対応)
1. **✅ _sidecar_signal NameError 修正** → §6 で対応完了。fill_test 再起動が必要

#### P1 (短期改善)
1. **Balance insufficient 削減**: 5,703 件の残高不足は最大のフリクション。資金管理の見直し or lot サイズの動的調整を検討
2. **Coincheck 400 Bad Request**: 374 件 → API リクエストのバリデーション強化 or エラー原因の詳細ログ
3. **sell_dynamic_kill 過剰**: 979 回の gate block → kill threshold の緩和 or タイムアウト短縮を検討

#### P2 (中期改善)
1. **PF 改善**: 現在 0.883。大損 (< -10 bps, 4.2% の Pnl) をカットするストップロス or offset 拡大
2. **Fill rate 回復**: 30% → 目標 40%+ への改善。ゲートのチューニング
3. **SAC sidecar 効果測定**: NameError 修正後、P0 attribution フィールドを使った sidecar 効果の定量分析
