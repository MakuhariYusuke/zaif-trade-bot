# 032# phg_rpt — Gate スタブ完成・方策 A 実装・品質改善

**種別**: phg (cross-gate / governance)
**Phase**: ph2 (非依存 — 全 Phase 横断)
**日付**: 2026-02-14
**前提**: 028# gap analysis, 031# fill_test 改善

---

## §1 概要

028# gap analysis § 6 で特定された P0/P3 タスクの完了と、
コードベース品質調査に基づく課題修正 6 件の実装。

| コミット | 内容 | テスト結果 |
|---------|------|-----------|
| `9afabd86c` | F7: Gate G2/G3/G4 スタブ + テスト | 341 passed |
| `57987f2ca` | P0: 方策 A core ロジック | 357 passed |
| `e8893d353` | 課題修正 6 件 + 方策 A 統合 | 358 passed |

---

## §2 F7: Gate G2/G3/G4 スタブ (028# P3)

### §2.1 実装

`scripts/v460/run_gate_check.py` に 3 関数を追加:

| 関数 | Gate | チェック項目 |
|------|------|------------|
| `run_g2_judgment()` | G2-train | positive_seed_ratio (≥75%), ic_seed_std (≤0.03), convergence (ROI var ≤5%), worst_seed_roi (>-2%) |
| `run_g3_judgment()` | G3-pnl | pf_median (>1.05), pf_worst (>0.95), gross_gt_fee, max_drawdown (<15%), sharpe_annual (>0.8) |
| `run_g4_judgment()` | G4-live | uptime_days (≥7), downtime_ratio (<1%), circuit_breaker, g3_maintained, emergency_stop (<1s) |

### §2.2 テスト

`tests/unit/v460/test_gate_check.py` に追加:
- TestG2Train: 6 テスト (pass, no_data, low_ratio, high_ic_std, poor_convergence, worst_seed)
- TestG3Pnl: 7 テスト (pass, no_data, low_pf_median, low_pf_worst, fee_gt_gross, high_dd, low_sharpe)
- TestG4Live: 7 テスト (pass, low_uptime, high_downtime, no_cb, g3_not_maintained, slow_stop, multi_fail)
- TestCLI: G2/G3/G4 テスト 3 件追加

### §2.3 ステータス

028# F7 [LOW] Gate G2/G3/G4 未実装 → **完了**。
`gate_thresholds.yaml` は全 6 Gate の閾値が定義済み (変更なし)。

---

## §3 P0: 方策 A パラメータ適応 (028# §3.1)

### §3.1 core ロジック

`scripts/v460/lib/param_adapter.py` を新規作成:

```
compute_adaptation(fill_rate, as_ratio, sample_count, config) → AdaptationResult
```

**適応ルール**:
- fill_rate < 80% → offset 増加 (板の内側へ寄せ、約定率向上)
- AS_ratio > 15% → offset 減少 (板の外側に退避、逆選択回避)
- 両方異常 → **AS 回避優先** (損失抑制 > 約定率)
- 段階調整: step_ratio = 0.01 ずつ
- ハードリミット: [0.01, 0.30] で clamp
- サンプル不足 (< 50) → hold

### §3.2 fill_test 統合

`run_fill_test.py` の `run_continuous()` ループ内に統合:
- `_try_auto_adapt()` メソッド: adapt_interval_cycles サイクルごとにメトリクス評価
- CLI: `--enable-auto-adapt` フラグで有効化 (デフォルト無効 — 安全側)
- `FillTestConfig`: `enable_auto_adapt`, `adapt_interval_cycles` フィールド追加

### §3.3 テスト

`tests/unit/v460/test_param_adapter.py` (16 テスト):
- hold (正常, サンプル不足, 境界値)
- increase (low fill_rate)
- decrease (high AS, 両方異常)
- clamp (min/max)
- repeated_adaptation (連続適応の段階性検証)

---

## §4 品質改善 (コードベース調査)

コードベース全体を調査し、20 件の課題を特定。
本セッションで SECURITY / BUG / DEPRECATION / CONFIG / RESILIENCE の 6 件を修正。

### §4.1 修正済み

| # | カテゴリ | 重要度 | 対象 | 修正内容 |
|---|---------|--------|------|---------|
| 1 | SECURITY | HIGH | `ztb/trading/live_trader/config.py` | health server `0.0.0.0`→`127.0.0.1` + `ZTB_HEALTH_BIND_HOST` 環境変数 |
| 2 | SECURITY | HIGH | `scripts/v460/run_fill_test.py` | `--api-key/--api-secret` 非推奨警告 (.env 推奨) |
| 6 | BUG | MEDIUM | `scripts/v460/run_fill_test.py` | `cancel_reason` 未定義 NameError 防止 (初期値 `"unknown"`) |
| 7 | RESILIENCE | MEDIUM | `ztb/metrics/fill_quality.py` | `FillRecord.from_dict()` 未知フィールド debug ログ |
| 8 | DEPRECATION | MEDIUM | `ztb/metrics/fill_quality.py` | `datetime.utcfromtimestamp()` → `fromtimestamp(tz=utc)` |
| 18 | CONFIG | LOW | `scripts/v460/run_fill_test.py` | `batch_size`, `max_save_retries` を `FillTestConfig` に設定化 |
| 19 | RESILIENCE | MEDIUM | `ztb/metrics/fill_quality.py` | JSONL 破損行スキップ + ログ出力 + テスト追加 |

### §4.2 未修正 (今後の候補)

| # | カテゴリ | 重要度 | 概要 | Phase |
|---|---------|--------|------|-------|
| 3 | ERROR | HIGH | `asyncio.run()` 二重呼出し (live_trader) | ph5 |
| 4 | ERROR | HIGH | trading_loop 重複実装 | ph5 |
| 5 | RESILIENCE | HIGH | ゼロベクトル特徴量フォールバック | ph5 |
| 10 | ERROR | MEDIUM | G2 `stdev` 1要素エラー | ph3 開始前 |
| 11 | TEST | HIGH | FillTestRunner 単体テスト不在 | ph2 継続 |
| 12 | TEST | HIGH | LiveTrader 単体テスト不在 | ph5 |
| 13 | RESILIENCE | MEDIUM | `_cleanup_sync` asyncio 問題 | ph5 |
| 15 | DESIGN | HIGH | LiveTrader god object (1857行) | ph5 |
| 16 | RESILIENCE | MEDIUM | ManifestWriter ディスクフル耐性 | ph3 |
| 17 | RESILIENCE | MEDIUM | save_fill_records アトミック書込み | ph2 継続 |

---

## §5 ルート直下クリーンアップ

`tmp_*.py` 4 ファイル削除 (gitignore 対象、物理削除):
- `tmp_020_verify.py`
- `tmp_check_fill.py`
- `tmp_fill_analysis.py`
- `tmp_syntax_check.py`

---

## §6 000# 改訂提案

| 対象 | 提案 |
|------|------|
| §3.4 G2-train | スタブ実装完了を記録 |
| §3.5 G3-pnl | 同上 |
| §3.6 G4-live | 同上 + health server セキュリティ修正を記録 |
| §3.3 G1.1-exec | 方策 A パラメータ適応の位置づけを追記 |

---

## §7 テスト推移

| 時点 | テスト数 |
|------|---------|
| 031# 完了時 | 318 |
| F7 Gate stubs 追加 | 341 (+23) |
| P0 方策 A core | 357 (+16) |
| 課題修正 + JSONL 耐性 | 358 (+1) |
