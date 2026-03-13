# 128# ログレビュー分析と改善方策

> **セッション**: 128#  
> **日付**: 2026-02-21  
> **対象**: fill_test 全 clean レコード (02/14 ~ 02/21) + retrain_scheduler ログ  
> **Git HEAD**: `870007d8f` (129# OB recorder)  
> **前提**: 127# レビュー修正済、128# dust sweep 実装済、129# OB recorder 実装済

---

## §1 データ概要

| 項目 | 値 |
|---|---|
| 総レコード | 1,565 |
| clean (quarantine 除外) | 1,416 |
| quarantine (blank git_sha) | 213 (13.6%) |
| 約定 (filled) | 941 |
| キャンセル | 475 |
| Skip Gate SKIP | 128 |
| Fill rate (skip_gate 除外) | 73.0% |

### 1.1 旧 run (1771607250) vs 新 run (1771651879)

| run | 期間 | records | filled |
|---|---|---|---|
| 旧 run (…202435dd) | 02/21 02:07–14:31 | 214 | 109 |
| 新 run (…16001dc3) | 02/21 14:31~ | 7+ (進行中) | — |

旧 run は 128#/129# 適用前。新 run で dust sweep + OB recorder が有効。

---

## §2 全体 PnL サマリ

### 2.1 Side 別

| 指標 | BUY | SELL | 全体 |
|---|---|---|---|
| 約定数 | 470 | 471 | 941 |
| 平均 PnL (30s) | **+0.087 bps** | **-0.433 bps** | **-0.173 bps** |
| 累計 PnL | +40.7 bps | **-203.9 bps** | **-163.1 bps** |
| 勝率 | ~50% | ~45% | ~48% |
| AS 率 | 49.1% | 54.6% | — |

**sell が全損失の 125% を生成** (buy +40.7 を差し引いても -163.1 赤字)。  
123# の分析と一致: sell 構造問題が依然継続。

### 2.2 日別推移

| 日付 | n | mean (bps) | sum (bps) | 勝敗 |
|---|---|---|---|---|
| 02/14 | 199 | -0.239 | -47.5 | 98/199 |
| 02/15 | 85 | **-0.976** | -82.9 | 36/85 |
| 02/16 | 6 | -1.155 | -6.9 | 3/6 |
| 02/17 | 82 | -0.473 | -38.8 | 34/82 |
| **02/18** | 148 | **+0.607** | **+89.8** | 79/148 |
| **02/19** | 131 | **+0.457** | **+59.9** | 68/131 |
| 02/20 | 179 | -0.600 | **-107.3** | 76/179 |
| 02/21 | 111 | -0.265 | -29.4 | 56/111 |

02/18-19 のみ黒字 (+149.7 bps)、残りの 6 日間で -313 bps。  
**市場レジームへの適応が不十分**: 有利な日 (ranging + 安定スプレッド) では利益を出せるが、不利な日の損失を制限できていない。

---

## §3 時間帯分析 (UTC)

| UTC 時間 | n | mean (bps) | sum (bps) | 所見 |
|---|---|---|---|---|
| UTC08 | 15 | **-3.805** | **-57.1** | 東京寄り付き — 最悪 |
| UTC21 | 42 | **-1.136** | **-47.7** | 深夜帯 — 高 AS |
| UTC12 | 8 | -1.076 | -8.6 | サンプル少 |
| UTC05 | 67 | -0.694 | -46.5 | 欧州早朝 |
| UTC20 | 91 | **+0.939** | **+85.5** | 最良時間帯 |
| UTC17 | 38 | +0.954 | +36.2 | 欧州午後 |

**UTC08 (JST17:00) は 15 件で -57.1 bps** — 東京株式市場終了直後の高ボラティリティ帯。skip_utc_hours に既に含まれているが、漏れがある可能性。

**UTC20-UTC22 (JST05:00-07:00)** が最も安定した利益帯 (+97.2 bps)。

---

## §4 防御機構の有効性

### 4.1 Volatility Guard (VG)

| 状態 | n | mean (bps) |
|---|---|---|
| VG on | 32 | **+0.480** |
| VG off | 909 | -0.196 |

VG 発動時は +0.480 bps → **有効に機能**。offset boost によりAS 回避に成功。  
ただし発動率 3.4% (32/941) は低く、もう少し感度を上げても良い。

### 4.2 Fast-Fill Defense (FFD)

| 状態 | n | mean (bps) |
|---|---|---|
| FFD on | 18 | +0.242 |
| FFD off | 923 | -0.181 |

FFD も正の効果。サンプル数が少ないため統計的確信は弱い。

### 4.3 Skip Gate

- SKIP 128 件 (全 clean の 9.0%)
- Pass→Filled の平均 PnL: -0.173 bps

skip_gate は「明らかに悪い」ケースを弾いているが、通過した注文の平均がまだ負。  
**ゲートの閾値が甘い**、もしくは **PnL 予測モデル自体の精度が不足**。

### 4.4 Spread と PnL の関係

| Spread 帯 | n | mean PnL (bps) |
|---|---|---|
| 0-2 bps | 209 | **+0.365** |
| 2-4 bps | 442 | -0.228 |
| 4-8 bps | 6 | -1.337 |

**Spread < 2 bps の時のみ黒字** — スプレッドが狭い = 流動性が高い = AS リスク低い。  
相関係数 r=-0.055 (ほぼ無相関) だが、バケット別では明確な傾向。

---

## §5 キャンセル要因分析

| Cancel Reason | 件数 | 割合 |
|---|---|---|
| timeout | 170 | 35.8% |
| skip_gate | 128 | 26.9% |
| **postonly_reject** | **68** | **14.3%** |
| **orderbook_error** | **64** | **13.5%** |
| status_unknown | 23 | 4.8% |
| stale_skip_gate_blocked | 10 | 2.1% |
| stale_reprice_failed | 7 | 1.5% |
| status_unknown_fast | 4 | 0.8% |
| postonly_reject | — | (下記分析) |

### 5.1 postonly_reject (68 件, 5.3%)

post_only 注文がテイカーになるため拒否。原因は **指値が市場を横切った** こと。  
→ offset 計算のレイテンシ中に mid price が移動 → 想定のメイカー価格がテイカー側に。

### 5.2 orderbook_error (64 件)

時間帯分布: UTC17-23 に集中 (44/64 = 69%)。  
**深夜帯の流動性枯渇** or **API レート制限** が原因の可能性。

### 5.3 Regime 別キャンセル

| Regime | BUY avg | SELL avg | buy sum | sell sum |
|---|---|---|---|---|
| ranging | +0.102 | -0.256 | +24.3 | -61.1 |
| trending | **+0.609** | **-0.808** | +58.5 | -79.2 |
| unknown | **-1.384** | -0.388 | -65.0 | -17.8 |

**unknown レジームの buy が -1.384 bps** — レジーム判定延滞時のフォールバック。

---

## §6 retrain_scheduler 問題

### 6.1 OB matched=0 (129# で解決済)

- 全サイクルで `OB matched=0/N` → リトレイン常時スキップ
- 根本原因: `run_observation.py` が Feb 19 で停止 → OB データ 2 日分欠如
- **129#** で fill_test 内に OB recorder を組み込み解決

### 6.2 run_id フィルタの影響

H2 フィルタ (127#) は正常動作しているが、OB データと最新 fill records のタイムスタンプが乖離しているため、enrich 時にマッチング失敗。

**新 run 稼働 + OB recorder により、次の retrain サイクル (1h 後) で OB matched > 0 が見込まれる。**

---

## §7 発見された問題の総括

| # | 問題 | 重要度 | 状態 |
|---|---|---|---|
| P1 | sell 構造損失: 全期間 -203.9 bps | ⭐⭐⭐ | 既知 (123#) |
| P2 | UTC08 (JST17) 極端損失: -3.8 bps/fill | ⭐⭐ | 要対策 |
| P3 | unknown レジーム buy: -1.384 bps | ⭐⭐ | 要対策 |
| P4 | postonly_reject 5.3% (68件) | ⭐⭐ | 要対策 |
| P5 | orderbook_error 深夜集中 (64件) | ⭐ | 要観察 |
| P6 | skip_gate 通過注文が平均負 | ⭐⭐⭐ | 要対策 |
| P7 | retrain 未実行 (OB データ欠如) | ⭐⭐⭐ | **129# 解決済** |
| P8 | 端数 BTC (dust) 蓄積 | ⭐ | **128# 解決済** |
| P9 | spread ≥ 2 bps で平均負 | ⭐⭐ | 要検討 |

---

## §8 改善方策の検討

### 方策 A: 時間帯フィルタ精緻化 (P2 対応)

UTC08 は既に skip_utc_hours に含まれるはずだが、15 件が通過している。

**案 A1**: skip_utc_hours の前後 ±30 分のバッファゾーンを追加  
**案 A2**: UTC21 を sell のみ新規ブロック (42 件で -1.136 bps)  
**案 A3**: "profitable hours only" モード — UTC20/UTC17/UTC06 のみ取引 (大胆)

**推奨**: A2 (sell UTC21 ブロック) + 現行フィルタの漏れ調査

### 方策 B: Skip Gate 高度化 (P6 対応)

skip_gate 通過→約定の平均が -0.173 bps ということは、 **ゲートが十分機能していない**。

**案 B1**: 閾値引き上げ (現在のスコア閾値を厳格化)  
**案 B2**: spread 条件の追加 — spread ≥ 3 bps 時は追加ペナルティ (P9 とセット)  
**案 B3**: retrain 実行後のモデル精度向上に期待 (129# 効果待ち)

**推奨**: B3 (retrain 起動を優先) + B2 (spread ≥ 3 bps 時のガード追加)

### 方策 C: Sell 構造問題への直接介入 (P1 対応)

123# で詳細分析済。sell の AS 率 54.6% (buy 49.1% 比 +5.5pt) が根本要因。

**案 C1**: sell offset を buy の 2 倍に拡大 (現行 0.30 を 0.45 等)  
**案 C2**: sell のみ VG 感度を上げる (VPIN 閾値を下げる)  
**案 C3**: sell の fill_timeout を短縮 (現行 96s → 60s) — 長時間放置による pick-off を防止  
**案 C4**: sell 完全停止 — buy-only モード (最も保守的)

**推奨**: C1 + C3 の組み合わせ。sell offset 拡大 + timeout 短縮で、「浅い位置に短時間だけ置く」戦略。

### 方策 D: Unknown レジーム対策 (P3 対応)

unknown レジームは判定データ不足時のフォールバック。buy が -1.384 bps と突出して悪い。

**案 D1**: unknown 時は取引停止 (最も安全)  
**案 D2**: unknown 時は offset を大幅拡大 (VG 常時 ON 相当)  
**案 D3**: レジーム判定の warm-up 期間短縮 (unknown 状態を減らす)

**推奨**: D2 — unknown 時は VG 強制発動 (offset 2x) で AS 回避

### 方策 E: postonly_reject 低減 (P4 対応)

5.3% のリジェクトは無駄なサイクル消費。

**案 E1**: offset 計算後に再度 mid price を取得し、テイカー側になっていないか確認  
**案 E2**: offset に安全マージン (+0.5 bps) を追加  
**案 E3**: 現行の post_only フラグを維持しつつ、reject 時に即リトライ (reprice)

**推奨**: E1 — 発注直前の二重確認。コスト低く効果が見込める。

### 方策 F: Spread フィルタ (P9 対応)

spread < 2 bps でのみ黒字。spread ≥ 2 bps で参入すると期待値が負。

**案 F1**: spread ≥ 3 bps でスキップ (保守的)  
**案 F2**: spread を skip_gate の特徴量として組み込む (B2 と統合)  
**案 F3**: spread に応じて offset を動的に調整 (spread 広い → offset も大きく)

**推奨**: F2 + F3 の統合。skip_gate が spread を考慮できるようにした上で、通過した場合でも offset を拡大。

---

## §9 優先順位と実施計画

| 優先度 | 方策 | 期待効果 | 工数 |
|---|---|---|---|
| 1 | **B3: retrain 起動待ち** | OB recorder 稼働でモデル自動改善 | 済 (129#) |
| 2 | **C1+C3: sell offset 拡大 + timeout 短縮** | sell 損失 -50% 目標 | 小 (YAML変更) |
| 3 | **A2: UTC21 sell ブロック** | -47.7 bps 回避 | 小 (YAML変更) |
| 4 | **D2: unknown レジーム VG 強制** | -65.0 bps (buy) 回避 | 中 |
| 5 | **F2+F3: spread フィルタ統合** | spread > 2 bps の損失削減 | 中 |
| 6 | **E1: postonly 二重確認** | reject 5.3% → ~1% | 小-中 |

**次セッションでの実施推奨**: 優先度 2-3 (YAML 変更のみで即効性あり)、次に 4-5 (コード変更)。

---

## §10 結論

1. **sell 構造問題が最大のドレイン** (-203.9 bps)。123# 分析と一致し、改善が最優先。
2. **VG / FFD は有効に機能**しているが、カバレッジが低い (3.4% / 1.9%)。感度向上の余地あり。
3. **retrain 未稼働が OB データ欠損に起因** (129# で解決済)。新 run の蓄積後、モデル改善が期待される。
4. **spread < 2 bps 帯でのみ黒字** — 流動性条件による参入フィルタが有効。
5. **immediate wins** は YAML 変更 (sell offset, UTC21 ブロック, sell timeout) で +50~100 bps の改善を見込む。

---

## §11 128# 提案の妥当性チェック（最新状態反映）

評価時点: **2026-02-21 06:20 UTC** (`monitor_fill_test.py`), **2026-02-21 15:21 UTC** (`retrain_scheduler --once`)

### 11.1 まず現況の再確認

- `gate_judgment.py`:
  - G1.1-quick: **PASS**
  - G1.2-full: **WATCH**
  - attempted_fill_rate=76.8%, AS_ratio=26.8%, PnL30=-0.172bps (p_holm=0.3737)
- `retrain_scheduler --once`:
  - latest run: `1771651879_16001dc3`
  - `OB matched=15/15`（129# 効果確認）
  - ただし `Insufficient filled samples: 5` で再学習は未実行

### 11.2 方策A-Fの判定

| 方策 | 妥当性 | コメント |
|---|---|---|
| A1/A2 (時間帯フィルタ強化) | 条件付きで妥当 | UTC21 悪化は再現しており A2 は実施価値あり。A1 の「±30分」は現行実装が時間単位のため追加実装が必要。 |
| A3 (profitable hours only) | 見送り推奨 | サンプル偏りが強く、レジーム変化時に脆い。000# の Gate 検証目的とも衝突。 |
| B1 (閾値厳格化) | 条件付きで妥当 | 現在 mode=pnl なので「AS閾値」ではなく `pnl_threshold` 側で設計し直す必要あり。 |
| B2 (spread 条件を SkipGate へ) | 一部重複 | `spread_jpy` は既に SkipGate 特徴量に含まれる。追加は「ルール併設」か「閾値再学習」。 |
| B3 (retrain 待ち) | 妥当 | 方向は正しい。ただし latest_run_only + min_total_samples=100 で起動直後に学習が進みにくい。 |
| C1 (sell offset 0.45 など大幅拡大) | 現設定では不適 | `max_offset_ratio=0.30` キャップがあるためそのままは効かない。まずキャップ設計を見直すべき。 |
| C3 (sell timeout 短縮) | 条件付きで妥当 | AS削減の可能性はあるが fill率低下とのトレードオフ大。A/B比較で確認必須。 |
| D2 (unknown時VG強制) | 妥当 | unknown帯の悪化対策として筋が良い。まず buy 側で限定導入が安全。 |
| E1 (発注直前二重確認) | 妥当 | postonly_reject 減少の即効施策。実装コストに対し効果が見込める。 |
| F2/F3 (spread統合) | 一部実施済み | `spread_adaptive` が既にある。新規価値は「閾値再推定」と「SkipGate連携の一貫化」。 |

---

## §12 テスト・品質確認（.venv 実行）

`.venv/Scripts/python.exe` で実行:

- `tests/unit/v460/test_retrain_hot_reload.py`
- `tests/unit/v460/test_fill_test_config.py`
- `tests/unit/v460/test_skip_gate_v3.py`
  - **114 passed, 1 warning**
- `tests/unit/v460/test_ob_recorder.py`
  - **12 passed, 1 warning**

結論: 126/127/129 系の基盤変更は、単体テスト上は整合している。

---

## §13 追加提案（過去成果の活用込み）

### 13.1 P0: 再学習の「起動直後スタベーション」解消

現状は `latest_run_only=true` で run 切替直後に学習データ不足になりやすい。  
`v459` 系の「段階ゲート」思想を再利用し、再学習も 2 段に分ける。

- Phase-Bootstrap: `min_total_samples=30`, `min_new_samples=10`（暫定）
- Phase-Stable: `min_total_samples=100`, `min_new_samples=30`（本番）

これにより「新 run でいつまでも未学習」を回避する。

### 13.2 P0: retrain の I/O 重さ削減（毎時 440万 trades 読み込みの抑制）

`retrain_scheduler.log` 上、毎サイクル `trades=4,396,171` を読んでおり重い。  
`v456` での I/O 削減（111#）の教訓を適用し、対象日限定ロードにする。

- fill_records の対象 timestamp から必要日だけ算出
- `data/v460/raw/trades/YYYYMMDD.jsonl.gz` を日単位選択で読む

### 13.3 P0: monitor と gate_judgment の判定体系を一本化

`monitor_fill_test.py` は legacy E1-E8 で FAIL 表示、`gate_judgment.py` は G1.2 WATCH。  
運用判断が割れるので、ph2 では G1.2 系を正とする表示へ統一する。

### 13.4 P1: unknown regime の buy 側ガード追加

現状は `skip_sell_unknown_regime=true` のみ。  
buy unknown が悪化しているため、buy 側にも段階ルールを追加する。

- step1: unknown buy は `offset_boost_factor` を強制適用
- step2: それでも悪化なら unknown buy skip を部分導入

### 13.5 P1: `orderbook_error` の内訳を分離

`orderbook_error` が原因追跡を難しくしている。  
`timeout/rate_limit/empty_book/parse_error` に細分化し、時間帯別に施策を分ける。

### 13.6 P1-P2: 過去資産の実戦投入（111#/112#）

1. `GatesToAlerts` を fill_test 判定に接続（WATCH/FAIL 即通知）
2. `RiskRuleEngine + AdvancedAutoStop` を ph2 ガードに段階導入
3. `run_pnl_monte_carlo.py` を日次自動化し、施策比較を期待値分布で判定
4. `v459/116` の counterfactual 発想を再利用し、「コスト0・完全執行仮定」の上限を定期再計算

---

## §14 更新後の優先順位（実装順）

1. **P0**: retrain bootstrap 2段化（13.1）
2. **P0**: monitor/gate 判定統一（13.3）
3. **P0**: UTC21 sell block の短期 A/B（11.2 A2）
4. **P1**: unknown buy ガード（13.4）
5. **P1**: postonly 二重確認 + orderbook_error 細分化（11.2 E1, 13.5）
6. **P1**: retrain I/O の日付限定ロード（13.2）

---

## §15 実装結果 (130# セッション)

全 6 項目を実装。1016 passed, 0 failed。

| # | 項目 | 変更ファイル | 内容 |
|---|---|---|---|
| 1 | retrain bootstrap 2段化 | `retrain_scheduler.py`, `fill_test.yaml` | `bootstrap_min_total=30, bootstrap_min_new=10, bootstrap_threshold=100` — total < 100 なら Bootstrap Phase で緩い閾値を適用 |
| 2 | monitor/gate 判定統一 | `monitor_fill_test.py` | `g1_1_judgment` → `g1_1_quick_judgment` + `g1_2_full_judgment` に切替。K1-K6 + F1-F8 の二段表示 |
| 3 | UTC21 sell block | `fill_test.yaml` | `skip_utc_hours_sell: [4,8,14,15,16,21]` — UTC21 (JST06) -1.136bps n=42 |
| 4 | unknown buy guard | `maker_price.py`, `fill_config.py`, `fill_test.yaml` | `unknown_buy_offset_boost: 2.0` — unknown regime buy 時に offset 2x (VG 相当) |
| 5a | postonly 二重確認 | `run_fill_test.py` | `place_order` 直前に mid price 再取得、テイカー側なら best_bid/ask に補正 |
| 5b | orderbook_error 細分化 | `run_fill_test.py` | `orderbook_timeout`, `orderbook_rate_limit`, `orderbook_empty`, `sell_guard_reject` に分離 |
| 6 | retrain I/O 日付限定 | `feature_enricher.py` | fill records の timestamp から必要日を算出、`date_filter` で trades/OB を日単位ロード |

テスト修正: `test_fill_quality.py`, `test_regime_detector.py` — skip_utc_hours_sell アサート更新
