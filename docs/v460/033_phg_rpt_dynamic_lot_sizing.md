# 033# 方策B: 動的ロットサイジング + 安全キャップ

| key | value |
|---|---|
| 種別 | phg (cross-gate / governance) |
| 日付 | 2026-02-14 |
| git | (本コミットで付与) |
| テスト | **386 passed** (358→386, +28) |

---

## 1. 概要

fill_test の固定 0.001 BTC (1 mBTC) ロットを、パフォーマンスに応じて段階的に増減する **方策 B: 動的ロットサイジング** を実装した。加えて、000# §3.9 の累積実損 10,000 JPY キャップをランタイムで監視する安全機構を組み込んだ。

## 2. 実装内容

### 2.1 方策 B: lot_sizer.py (新規)

`scripts/v460/lib/lot_sizer.py` — param_adapter.py と同パターンの純関数設計。

**LotSizingConfig:**
| パラメータ | デフォルト | 説明 |
|---|---|---|
| min_lot | 0.001 BTC | Coincheck 最小ロット |
| max_lot | 0.005 BTC | 保守的上限 (5 mBTC) |
| lot_step | 0.001 BTC | 1段階の増減量 |
| min_fill_rate | 0.70 | 増量に必要な約定率 |
| max_as_ratio | 0.30 | 増量に必要なAS上限 |
| min_recent_pnl_bps | 0.0 | 増量に必要な直近PnL |
| loss_cap_jpy | 10,000 | 損失キャップ |
| loss_cap_warning_ratio | 0.7 | 70%で警告・縮小 |
| min_samples | 50 | 最小サンプル数 |

**ロジック:**
1. **損失キャップ優先**: 累積PnL ≤ -7,000 JPY → 強制最小ロット (`cap_shrink`)
2. **サンプル不足**: < 50件 → `hold`
3. **全条件クリア** (fill_rate ≥ 70%, AS ≤ 30%, PnL ≥ 0): → `increase` (+1 step)
4. **いずれか未達**: → `decrease` (-1 step, 最小まで)

**ヘルパー関数:**
- `compute_cumulative_pnl_jpy()` — レコードから累積PnL (JPY) 算出
- `compute_recent_pnl_bps()` — 直近N件の平均PnL (bps) 算出
- `clamp_lot()` — ハードリミット + 小数第4位丸め

### 2.2 run_fill_test.py 統合

- `FillTestConfig` に追加: `enable_dynamic_lot`, `max_lot`, `lot_adapt_interval_cycles`
- `FillTestRunner._current_lot` — 実行時ロット (初期値 = config.order_quantity)
- `run_single_cycle` 内の `self.config.order_quantity` → `self._current_lot` に変更 (5箇所)
- `_try_auto_lot_size()` メソッド追加 — 方策Aと同様に run_continuous ループ内で定期実行
- CLI: `--enable-dynamic-lot`, `--max-lot`

### 2.3 F4: 累積 PnL 安全キャップ

run_continuous ループ内で累積 PnL をインクリメンタルに追跡し、-10,000 JPY を超えた場合に `_shutdown_requested = True` で安全停止。レジューム時にも既存レコードから累積 PnL を復元する。

進捗ログに `cumPnL` と `lot` を追加。

## 3. Fill Test 実績分析 (295 records, ~36h)

| 指標 | 値 |
|---|---|
| Total | 295 |
| Filled | 221 (74.9%) |
| Buy fill | 116/148 (78.4%) |
| Sell fill | 105/147 (71.4%) |
| AS ratio (全体) | 99/221 (44.8%) |
| offset=0.05 AS | 5/14 (35.7%) |
| offset=旧 AS | 94/207 (45.4%) |
| Avg PnL | -0.28 bps |
| Recent 50 PnL | **+0.22 bps** (改善トレンド) |
| 累積 PnL | -64.41 JPY (10K cap内) |
| Pos/Neg | 110/110 (完全均衡) |

### 3.1 発見した課題

| # | カテゴリ | 内容 | 対応 |
|---|---|---|---|
| F1 | SAFETY | 累積PnL 10K JPY キャップがランタイム監視されていない | **本033#で実装済** |
| F2 | DATA | 旧レコード26件に cancel_reason/spread_at_order なし | 031#以前のデータ。影響なし |
| F3 | API | 400 Bad Request 3件 (offset=0.05移行後) | 要観察。retry で回復可能 |
| F4 | PERF | Buy vs Sell fill rate 差 (78.4% vs 71.4%) | 将来: side別offset適応 |
| F5 | PERF | offset=0.05 で AS 35.7% (旧 45.4%比 改善) | サンプル増で確認継続 |
| F6 | PERF | 直近50件 +0.22bps — 改善トレンド | 動的ロットで収益増の好機 |

### 3.2 wait time 分布

| 区間 | 件数 | 割合 |
|---|---|---|
| < 10s | 98 | 44.3% |
| 10-30s | 55 | 24.9% |
| 30-60s | 27 | 12.2% |
| > 60s | 41 | 18.6% |

## 4. テスト

28テスト追加: `tests/unit/v460/test_lot_sizer.py`
- `TestComputeLotSize` (11 cases): increase/decrease/hold/cap_shrink/step_increments
- `TestClampLot` (5 cases): min/max/range/rounding/default
- `TestComputeCumulativePnlJpy` (5 cases): positive/negative/mixed/empty/different_quantities
- `TestComputeRecentPnlBps` (5 cases): window/skip_unfilled/empty
- `TestLotSizingResult` (2 cases): changed true/false

## 5. 使用方法

```powershell
# 動的ロットサイジング有効化
python scripts/v460/run_fill_test.py `
  --hours 168 `
  --spread-offset-ratio 0.05 `
  --start-side buy `
  --enable-dynamic-lot `
  --max-lot 0.005

# 方策A + 方策B 同時有効化
python scripts/v460/run_fill_test.py `
  --hours 168 `
  --enable-auto-adapt `
  --enable-dynamic-lot
```

## 6. ファイル一覧

| ファイル | 変更 |
|---|---|
| `scripts/v460/lib/lot_sizer.py` | **新規**: 方策B ロットサイジングロジック |
| `scripts/v460/run_fill_test.py` | 統合: _current_lot, _try_auto_lot_size, F4安全キャップ |
| `tests/unit/v460/test_lot_sizer.py` | **新規**: 28テスト |
