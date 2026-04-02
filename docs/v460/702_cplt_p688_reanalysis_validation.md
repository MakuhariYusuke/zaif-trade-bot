# 702# Protocol 688 再分析: NFQ フィルタ修正後の精密検証

## 分析概要

700# で修正した Protocol 688 NFQ フィルタの修正後データを再分析。
従来は全キャンセル 488 件を NFQ として誤カウントしていたバグを修正し、
真の NFQ = 206 件 (14.0%) を正確に分離。
加えてデータの信頼性を 8 つの批判的視点で多角検証。

**分析対象**: 直近 4 日間 (2026-03-30 ~ 2026-04-02)
**データ規模**: 1,883 orders, 410 fills, 1,473 cancels

---

## 1. 基本統計

| 指標 | 値 | 評価 |
|------|----|----|
| Total orders | 1,883 | |
| Fill rate | 21.8% (410/1883) | |
| Avg PnL30 | -0.38 bps | 赤字 |
| Buy avg PnL30 | +0.30 bps | 黒字 |
| Sell avg PnL30 | -1.07 bps | **深刻な赤字** |
| AS rate | 27.3% (112/410) | WARN 水準 (閾値 25%) |
| AS avg PnL30 | -6.58 bps | |
| Non-AS avg PnL30 | +1.94 bps | MM として健全 |

---

## 2. NFQ 修正後の正確なデータ

| 指標 | 修正前 (全キャンセル混入) | 修正後 (真 NFQ) |
|------|--------------------------|----------------|
| NFQ 件数 | 488 | **206** |
| NFQ / Cancel 比率 | 100% (バグ) | **14.0%** |
| NFQ regime分布 | 不明 | ranging=91, down=60, up=55 |
| NFQ side分布 | 不明 | buy=113, sell=93 |

### NFQ の位置づけ
- 第3位のキャンセル理由 (preflight 565 > spread_narrow 254 > **NFQ 206**)
- NFQ buy偏重 (113 vs 93) は在庫スキュー影響
- regime_exit_strategy は trending_down NFQ (60件) のみ対象 → 全 NFQ の 29%

---

## 3. データ整合性チェック (10 項目全パス)

| チェック | 結果 |
|----------|------|
| Total = Filled + Cancels | ✅ 1883 = 410 + 1473 |
| Cancel reasons sum = cancel_total | ✅ 1473 |
| NFQ in reasons = NFQ section | ✅ 206 |
| Side + none = Total | ✅ 1784 + 99 = 1883 |
| Regime sum = Total | ✅ 1883 |
| Cross-tab filled = Basic filled | ✅ 410 |
| Weighted PnL consistency | ✅ diff=0.000000 |
| SHA filled sum | ✅ 410 |
| Hour filled sum | ✅ 410 |
| AS + Non-AS PnL decomposition | ✅ Non-AS = +1.94 bps (健全) |

---

## 4. 批判的検証: データは嘘をついているか?

### P1: 生存バイアス — PnL は氷山の一角
- Fill rate 21.8% → PnL は全体の 22% しか見えていない
- 78% のキャンセル (特に preflight=565=38.4%) の仮想 PnL は不明
- **結論**: PnL 分析は「約定できた注文」の品質のみ反映。キャンセルが利益を避けた可能性あり

### P2: サンプルサイズ — 統計的有意性
| セグメント | n | PnL (bps) | 95%CI (bps) | 有意性 |
|---|---|---|---|---|
| buy/ranging | 95 | +0.61 | ±0.60 | 境界的 |
| buy/trending_down | 61 | +0.35 | ±0.75 | 非有意 |
| buy/trending_up | 49 | -0.35 | ±0.84 | 非有意 |
| sell/ranging | 92 | -0.86 | ±0.61 | **有意 (p<0.05)** |
| sell/trending_down | 59 | -0.53 | ±0.77 | 境界的 |
| sell/trending_up | 54 | -2.01 | ±0.80 | **有意 (p<0.01)** |

→ **sell/trending_up (-2.01 bps) と sell/ranging (-0.86 bps) は統計的に有意な損失**

### P3: 時間期間バイアス
- 4日間で 19 SHA → コード変更が頻繁すぎてベースラインが不安定
- 結論の一般化には注意が必要（特定 SHA に引きずられる可能性）

### P4: SHA 交絡 — b56771a が全損失の 88.6% ★★★
| SHA | Fills | PnL (bps) | 損失貢献 |
|---|---|---|---|
| b56771a | 66 | -2.12 | -139.9 bps (88.6%) |
| その他 (344 fills) | 344 | -0.05 | -17.9 bps (11.4%) |
| **全体** | **410** | **-0.38** | **-157.8 bps** |

**→ b56771a を除外すると全体 PnL は -0.05 bps (ほぼ収支均衡)**
**→ 「sell が赤字」は b56771a が支配している可能性が高い**

これは重大な発見。b56771a が sell/trending_up の損失を集中的に生んでいる可能性がある。

### P5: AS 非対称性
- buy AS = 21.0%, sell AS = 33.7% (1.6 倍)
- 4日間の市場方向性バイアスの可能性（構造的問題か一時的かは判別不能）

### P6: Spread 'unknown' 問題
- 62.3% の注文に spread データなし（全てキャンセルレコード）
- Fill PnL 分析は影響なし（filled only）だが、キャンセルのスプレッド分析は不可

### P7: 時間帯集中 — 12-17h UTC (JST 21-02時) が損失の温床 ★★
| 時間帯 | Fills | Total PnL (bps) | Avg PnL (bps) |
|---|---|---|---|
| 12-17h UTC | 86 | -174.2 | -2.03 |
| その他 (18h) | 324 | +16.5 | +0.05 |

**→ 12-17h を除外すると全体は黒字 (+0.05 bps)**
**→ JST 21-02時 (深夜帯) に損失が集中**

### P8: NFQ の regime_exit_strategy カバー率
- regime_exit_strategy は trending_down NFQ (60件) のみ対象 = 全 NFQ の 29%
- remaining 71% の NFQ (ranging=91, trending_up=55) は別機構が必要

---

## 5. 重大発見サマリ

### 発見 A: b56771a が全損失の 88.6%
- 1 つの SHA が全体の印象を歪めている
- b56771a 除外後の全体 PnL は -0.05 bps（ほぼ均衡）
- **対策**: b56771a の何が悪かったかを特定 → sidecar 状態・AS rate を SHA 別に追跡

### 発見 B: sell/trending_up が有意に最悪 (-2.01 bps)
- 統計的に有意 (95%CI: -1.21 ~ -2.81 bps)
- **対策**: sell_hour_boost の trending_up 時動作、AS 率、offset 設定の再検証

### 発見 C: 12-17h UTC に損失集中 (-174.2 bps)
- 全体の 21% の fill が全体損失の 110% を生成
- **対策**: 時間帯別 EV 調整 or 深夜帯 conservative mode

### 発見 D: NFQ の 71% は regime_exit_strategy の射程外
- ranging (91件) と trending_up (55件) の NFQ は現行施策では対処不能
- **対策**: NFQ 発生条件の根本分析（maker_price の quote 計算ロジック）

---

## 6. 次期改善候補 (Codex タスク候補)

| 優先 | タスク | 根拠 | 期待効果 |
|------|--------|------|----------|
| P0 | sell/trending_up 損失調査 + 対策 | 発見B, 有意な -2.01 bps | sell PnL 改善 +0.5-1.0 bps |
| P0 | 12-17h UTC 時間帯ガード | 発見C, -174.2 bps 集中 | 全体 PnL 転換 -0.38→+0.05 |
| P1 | SHA 別 AS 追跡テレメトリ | 発見A, b56 交絡 | 分析精度向上 |
| P1 | NFQ 発生条件根本分析 | 発見D, 71% 射程外 | NFQ 削減 |
| P2 | spread bucket cancel 分析 | P6, 62% データ欠損 | キャンセル分析改善 |

---

*生成: 2026-04-03 by cplt (702#)*
*入力: protocol_688.json (4日間, NFQフィルタ修正後), p688_validate.py, p688_critical.py*
*再現: `.venv\Scripts\python.exe -m scripts.v460.analysis.run_protocol --protocol 688 --days 4`*
