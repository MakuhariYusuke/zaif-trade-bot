# 041# 高収益改善施策 — 動的 loss_cap + AS 最適化 5 施策

| key | value |
|---|---|
| 番号 | 041 |
| フェーズ | ph2 (G1.1-exec) |
| 種別 | rpt (実装報告) |
| 前提 | 040# (fill test interim 335 records), 035# (review) |
| 作成日 | 2026-02-14 |
| テスト | 422 passed (v460 全テスト) |

---

## §1 背景

040# の fill test 中間報告で以下が判明:

- **累積 PnL**: -119.61 JPY
- **loss_cap_jpy**: 10,000 JPY (口座残高の 71% — 1 mBTC ≈ 14,000 JPY)
- **E5 AS_ratio**: 43.1% (閾値 20%)
- **E1 fill_rate**: 74.5% (閾値 90%)
- **api_error**: 30 件 (cancel の主因)

ユーザー指摘: 「1 mBTC しか買えない分しか口座に入れてない状況下では、一万円は大きい」  
→ API 残高ベースの動的キャップが必要。

---

## §2 データ分析結果

### §2.1 サイド別

| side | n | filled | fill_rate | AS | mean_pnl |
|------|---|--------|-----------|-----|----------|
| buy | 168 | 133 | 79.2% | 43.6% | -0.15 bps |
| sell | 167 | 122 | 73.1% | 42.6% | -0.77 bps |

sell 側が弱い: 約定率 6pt 低く、PnL 5 倍悪い。

### §2.2 offset 比較

| offset | filled | AS | mean_pnl |
|--------|--------|-----|----------|
| 0.05 (新) | 48 | 33.3% | -1.17 bps |
| unknown (旧) | 207 | 45.4% | -0.28 bps |

offset=0.05 は AS を 12pt 改善したが、PnL は悪化。トレードオフ。

### §2.3 時間帯別 AS (UTC)

| 時間帯 (JST) | UTC | AS 平均 | 備考 |
|---|---|---|---|
| JST 11-16 | UTC 02-07 | **~25%** | 低 AS — 有利 |
| JST 17-04 | UTC 08-19 | **~55%** | 高 AS — 不利 (informed flow) |
| JST 05-10 | UTC 20-01 | **~35%** | 中間 |

→ **時間帯フィルターで AS 43% → 推定 25-30% に低減可能**。

### §2.4 スプレッド

- mean=2,133 JPY, median=2,044 JPY, stdev=955 JPY
- BTC/JPY ~14M で換算すると ~14 bps → `as_deadzone_bps=0.5` はノイズ域

### §2.5 スリッページ

- 56.5% が有利方向（mid より良い位置で約定）
- → maker 戦略自体は機能している

---

## §3 実装した改善施策

### A. 時間帯フィルター (高インパクト)

| 項目 | 内容 |
|------|------|
| ファイル | `run_fill_test.py`, `fill_test.yaml` |
| 設定 | `time_filter.enabled: true`, `skip_utc_hours: [8,9,14,16,17,18,19]` |
| 動作 | サイクル実行前に UTC 時刻をチェック、高 AS 時間帯は `cancel_reason=time_filter` でスキップ |
| 期待効果 | AS 43% → 25-30% (JST 11-16 の低 AS 時間帯に集中) |

### B. AS deadzone 拡大 (YAML 変更のみ)

| 項目 | 内容 |
|------|------|
| 変更 | `as_deadzone_bps: 0.5 → 2.0` |
| 根拠 | BTC/JPY spread ~14 bps に対し 0.5 bps はノイズ域。2.0 bps で AS の過剰判定を抑制 |
| 期待効果 | AS 判定 5-10% 改善 |

### C. 方策 A 有効化 (YAML 変更のみ)

| 項目 | 内容 |
|------|------|
| 変更 | `adaptation.enabled: false → true` |
| 動作 | 50 サイクルごとに fill_rate/AS_ratio を評価し offset を ±0.01 段階調整 |
| 制約 | offset 範囲 [0.01, 0.30]、最小サンプル 50 件 |
| 期待効果 | AS > 15% → offset 減少 (板外側退避)、fill_rate < 80% → offset 増加 |

### D. api_error 削減

| 項目 | 内容 |
|------|------|
| 変更 | `max_order_retries: 1 → 2` |
| 根拠 | 30 件の api_error の大半が一時的障害。リトライ 1 回追加で回復率向上 |
| 期待効果 | api_error 30 件 → 推定 15 件以下 (fill_rate +4-5%) |

### E. 動的 loss_cap (API 残高ベース)

| 項目 | 内容 |
|------|------|
| ファイル | `run_fill_test.py`, `fill_test.yaml` |
| 設定 | `safety.loss_cap_auto: true`, `safety.loss_cap_ratio: 0.05` |
| 動作 | `run_continuous()` 開始時に `get_balance()` + `get_current_price()` で全資産を JPY 換算、残高 × 5% をキャップに設定 |
| フォールバック | API 失敗時は `loss_cap_jpy` (10,000 JPY) を維持 |
| 最低保証 | 50 JPY (極端に小さいキャップは運用不能) |
| 残高 14,000 JPY 時 | キャップ = 700 JPY (旧 10,000 JPY の 7%) |

---

## §4 ファイル変更一覧

| ファイル | 変更種別 | 内容 |
|----------|---------|------|
| `configs/v460/fill_test.yaml` | 変更 | deadzone 2.0, adaptation enabled, retry 2, time_filter 新セクション, safety 拡張 |
| `scripts/v460/run_fill_test.py` | 変更 | FillTestConfig +4 fields, from_yaml +2 sections, time_filter in run_single_cycle, `_update_dynamic_loss_cap()` 新メソッド, main ログ拡張 |
| `tests/unit/v460/test_regime_detector.py` | 変更 | +8 tests (time_filter 3, loss_cap 3, deadzone 2) |
| `tests/unit/v460/test_fill_test_config.py` | 変更 | 期待値更新 (adaptation enabled, time_filter, loss_cap_auto) |

---

## §5 期待される改善効果 (総合)

| 指標 | 現在 | 改善後推定 | 変化 |
|------|------|-----------|------|
| AS_ratio | 43.1% | 20-25% | -18~23pt |
| fill_rate | 74.5% | 80-85% | +5~10pt |
| loss_cap_jpy | 10,000 | 700 | -93% (安全設計適正化) |
| 有効サイクル率 | 100% | ~71% | 7h/24h スキップ |

注: 時間帯フィルターにより有効サイクルが減少するため、FINAL 判定に必要な 3 日は
サイクル数ベースではなくカレンダーベースで判断する。

---

## §6 再起動手順

```powershell
# 1. 現プロセス停止 (Ctrl+C or kill)
Stop-Process -Id (Get-Process python | Where-Object { $_.CommandLine -match "run_fill_test" }).Id

# 2. 再起動 (fill_test.yaml が自動読込される)
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\python.exe scripts\v460\run_fill_test.py `
  --config configs/v460/fill_test.yaml `
  --hours 168
```

再起動後のログで以下を確認:
- `[loss_cap] 動的キャップ算出: 残高=XXXX JPY × 5% = XXX JPY`
- `Config loaded: ... adapt=True, ... time_filter=True, loss_cap_auto=True`
- `[Regime] detector enabled: window=20, hysteresis=3`

---

## §7 変更履歴

| 日付 | 版 | 内容 |
|------|-----|------|
| 2026-02-14 | 初版 | 5 施策実装 + テスト 422 passed |
