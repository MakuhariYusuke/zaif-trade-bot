# 053# — G1.1 暫定判定・Monte Carlo PnL・リポジトリ整理

## §1 概要

000# から全ドキュメントを縦断し、以下のオフラインタスクを実行した。

| タスク | ソース | 結果 |
|--------|--------|------|
| §3.9 継続中止ルール判定 | 000# §3.9 | **CONTINUE** |
| §3.3 G1.1 暫定判定 | 000# §3.3, 005#, 014# | **FAIL** (E1, E4, E5) |
| Monte Carlo PnL推定 | 023# P1-B | Round-trip +645 JPY/日 |
| ドキュメント数監査 | 000# §5 | 54→33 文書 (21 archived) |
| 観測データ収集再開 | 023# P1-D前提, 014# T2 | 再起動済み (168h) |
| リポジトリ整理 | 023# P2 | analysis/ 90件, temp/ 33件 |
| SAC dead code 削除 | 023# P1-A | ~25ファイル削除 |

---

## §2 §3.9 継続中止ルール判定

n=491 で判定要件 (n≥200) を満たす。

| 条件 | 閾値 | 実測 | 判定 |
|------|------|------|------|
| fill_rate ≥ 70% | 70% | 76.0% | **CONTINUE** |
| AS ≤ spread/2 | (n≥500 必要) | AS=39.1%, spread/2=1.02bps | **保留** (n=491) |
| 累積実損 ≤ 10K JPY | -10,000 JPY | -248.6 JPY | **CONTINUE** |
| G1 全9 FAIL→v461 | 全FAIL | 7/9 PASS (005#) | **CONTINUE** |
| 方向 IC > 0.04 (BE) | 0.04 | 0.03-0.04 | **境界** |

**総合: CONTINUE** — 中止条件のいずれにも該当せず。

---

## §3 §3.3 G1.1 暫定判定

暫定判定要件: n=491 (≥200 ✓), 暦日=3 (≥3 ✓)

| 基準 | 閾値 | 実測 | 判定 |
|------|------|------|------|
| E1 fill_rate | ≥ 90% | 76.0% | **FAIL** |
| E2 cancel_ratio | ≤ 30% | 24.0% | PASS |
| E3 queue_wait p50 | ≤ 60s | 17.6s | PASS |
| E4 PnL mean (30s) | ≥ 0 bps | -0.620 bps | **FAIL** |
| E5 AS_ratio | ≤ 20% | 39.1% | **FAIL** |

**G1.1 総合: FAIL** (E1, E4, E5)

### §3.1 FAIL 要因分析

- **E1 fill_rate 76%**: 052# で skip hours を追加して改善余地あり。timeout/api_error が cancel の主因。
  - timeout: 49件、api_error: 34件 → API品質がボトルネック
- **E4 PnL -0.620 bps**: 個別 fill 30s mark-to-market は負だが、Round-trip ベースは正 (§4 参照)
- **E5 AS 39.1%**: sell 側 38.7%, buy 側 39.6%。AS fill と non-AS fill の PnL 差は -7.2 bps と甚大。

### §3.2 G1.1 PASS に必要な改善

| 項目 | 現在 | 目標 | ギャップ |
|------|------|------|----------|
| fill_rate | 76.0% | 90.0% | +14 pp |
| PnL 30s | -0.620 bps | ≥ 0 bps | +0.620 bps |
| AS_ratio | 39.1% | ≤ 20% | -19.1 pp |

---

## §4 Monte Carlo PnL 推定

### §4.1 Round-trip (往復) ベース

n=181 round-trip pairs (3日間) から bootstrap (N=10,000) で日次 PnL を推定。

| 指標 | 値 |
|------|-----|
| RT 平均 | +10.74 JPY/pair |
| RT 中央値 | +2.27 JPY/pair |
| RT 標準偏差 | 52.24 JPY |
| Win rate | 54.7% (99/181) |
| RT total (3日) | +1,944 JPY |

**Monte Carlo 日次分布** (60 RT/day):

| パーセンタイル | JPY |
|---------------|-----|
| p5 | -9 |
| p25 | +367 |
| **p50** | **+642** |
| p75 | +911 |
| p95 | +1,322 |

**月次推定** (×30日):

| パーセンタイル | JPY |
|---------------|-----|
| p5 | -278 |
| **p50** | **+19,265** |
| p95 | +39,673 |

### §4.2 個別 fill (30s mark-to-market) ベース

| パーセンタイル | JPY/day |
|---------------|---------|
| p5 | -197 |
| p50 | -81 |
| p95 | +26 |

### §4.3 Adverse Selection 影響

| 区分 | PnL mean | n |
|------|----------|---|
| AS fill | -5.008 bps | 146 |
| Non-AS fill | +2.202 bps | 227 |
| **AS impact** | **-7.211 bps** | — |

AS 排除時の加重改善: +2.822 bps

### §4.4 解釈

**Round-trip と 30s mark-to-market の乖離が重要な発見。**

- Round-trip (実際のポジション解消) では月次 p50=+19K JPY と正
- 30s mark-to-market は短期的な不利な価格変動を過大評価している可能性
- AS fill (39%) が全体を引き下げ。non-AS fill は +2.2 bps と健全
- **AS 低減が最大の改善レバー** (7.2 bps impact)

---

## §5 ドキュメント数監査・整理

000# §5 「40 文書以内」に対し 54 文書が存在。

### 実施内容

rev/resp/ver チェーン文書 21 件を `docs/v460/archived/` に移動:

```
002, 003, 004, 006, 007, 008, 011, 012, 016, 017, 020,
024, 025, 026, 027, 029, 030, 035, 044, 045, 049
```

**結果: 54 → 33 文書** (§5 準拠)

---

## §6 リポジトリ整理 (P2)

| 対象 | 件数 | 操作 |
|------|------|------|
| analysis/ (旧分析スクリプト) | 90 files | → archived/analysis/ |
| temp/ (一時ファイル) | 33 files | 削除 (judge, monte_carlo のみ残置) |
| ルート直下 (.eslintrc.js, .prettierrc, test_results.json) | 3 files | → archived/ |

---

## §7 SAC dead code 削除 (P1-A)

v460 からの import がゼロかつ参照先も削除対象の SAC レガシーファイルを削除。

### 削除リスト (~25 files)

```
ztb/sac_v426_improvement/           # v426 レガシー (4 files)
ztb/training/sac_utils.py           # 孤立ユーティリティ
ztb/training/sac_utils_scripts.py   # 同上
ztb/training/sac_v427_*.py          # v427 レガシー (2 files)
ztb/training/train_sac_v432*.py     # v432 レガシー (2 files)
ztb/training/train_v430_*.py        # v430 レガシー (2 files)
ztb/training/optimize_v430_*.py     # v430 レガシー
ztb/training/integrated/            # v434 実験 (3 files)
ztb/training/archive/v435/          # v435 スタブ (7 files)
ztb/training/archive/v433/          # v433 production (4 files)
ztb/utils/v4xx_config_converter.py  # 旧バージョンコンバータ
```

### 保持リスト (ph3 で使用)

```
ztb/training/unified_trainer/algorithms/sac_trainer.py  # ph3 統一トレーナー
ztb/training/algorithms/sac/                           # algorithm factory
ztb/training/callbacks/reinforcement/sac/              # コールバック群
ztb/training/compression/compressor.py                 # pruning
ztb/training/distillation/distiller.py                 # distillation
ztb/training/quantization/quantizer.py                 # quantization
ztb/trading/environment/heavy_env/core.py              # SAC action 環境
```

---

## §8 観測データ収集

| 項目 | 状態 |
|------|------|
| data/v460/raw/orderbook | 2日分 (0213-0214) |
| data/v460/raw/trades | 2日分 (0213-0214) |
| 収集プロセス | 停止中 → **再起動済み** (PID 57812/58860, 168h) |

G1 再検証 (real feature で XGBoost walk-forward) に必要な real microstructure data を蓄積中。

---

## §9 Next Actions

| 優先度 | タスク | 依存 |
|--------|--------|------|
| P0 | 入金して fill test 再開 | 銀行営業日 (月曜) |
| P0 | AS 低減施策の設計・実装 | E5 FAIL 対策: -7.2 bps の削減 |
| P1 | fill_rate 改善 (API error 対策) | E1 FAIL: timeout/api_error cancel の低減 |
| P1 | n≥500 到達して AS>spread/2 判定 | fill test 継続 |
| P1-C | G1 proxy 再検証 (real features) | 観測データ蓄積 |
| P1-D | Real feature 設計拡張 | 観測データ解析 |
| P1-E | ph3 SAC 訓練パイプライン準備 | G1.1 PASS 後 |
| P2 | sac_v430_training_optimizations.py の DynamicLRScheduler 移設 | unified_trainer 内部 |

---

## Appendix A: 000# 改訂

| 日付 | §番号 | 変更内容 | 理由 |
|------|-------|---------|------|
| 2026-02-15 | §3.9, §3.3 | G1.1 暫定判定結果 (FAIL: E1, E4, E5) を記録。§3.9 継続中止 CONTINUE を確認 | 053# |
