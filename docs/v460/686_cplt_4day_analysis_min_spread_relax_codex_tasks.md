# 686# 4日間データ分析・min_spread 緩和・Codex タスク設計

| 項目 | 値 |
|------|-----|
| 作成日 | 2026-04-02 |
| 入力 | 0# (提案書), 605# (渙振返り), 672# (深層分析), 684# (統合レビュー), 685# (Phase 1 sell protection) |
| 対象 | `configs/v460/fill_test.yaml` |
| 方式 | YAML hot-reload + Codex タスクプロンプト作成 |

---

## 0. 背景

685# で Phase 1 sell protection（trend boost・SG 閾値・AM hour defense）を適用後、
0#/605#/672# の横断レビューを行い、4日間の fill データ（3/29-4/1, n=244）を統計分析した。
672# の情報理論的結論を実データで検証し、次の施策を設計する。

---

## 1. 4日間データ分析結果 (3/29-4/1, n=244 fills)

### 1.1 Side 別
| Side | n | AS率 | avg PnL30 (bps) |
|------|---|------|:---:|
| BUY  | 122 | 21% | -0.02 |
| SELL | 122 | 30% | **-0.94** |

→ SELL 側が問題。685# の sell protection 効果は今後のデータで検証。

### 1.2 Spread 帯別（核心データ）
| Spread帯 (JPY) | n | AS率 | avg PnL30 (bps) | 評価 |
|:---:|---|---|:---:|------|
| 0-1500   | 12

  | 17% | **+0.91** | **最優良** |
| 1500-2500 | 63  | 25% | -0.18 | ほぼ break-even |
| 2500-3500 | 119 | 24% | -0.68 | 件数×損失で最大 |
| 3500+     | 50  | 32% | **-0.72** | 最高 AS 率 |

→ **672# の「逆選択フィルタの逆説」を実データで完全確認**。min_spread が最良帯をブロックしている。

### 1.3 NFQ 内訳 (4/1, n=344 レコード)
| 理由 | 件数 | 割合 | 対応 |
|------|------|------|------|
| preflight_insufficient | 122 | 35% | 資本制約（ソフトウェアで解決不可） |
| spread_too_narrow | 52 | 15% | **本 686# で緩和** |
| skip_gate | 48 | 14% | **686# Codex bypass タスク** |
| timeout | 13 | 4% | 正常動作 |
| sell_dynamic_kill | 8 | 2% | 685# で調整済み |

### 1.4 SkipGate 予測力検証
| SG Score 四分位 | AS率 | avg PnL30 |
|:---:|:---:|:---:|
| Q1 (-5.57〜0.68) | 29% | -0.16 |
| Q2 (0.69〜1.58) | 26% | -0.34 |
| Q3 (1.59〜2.98) | 25% | -0.13 |
| Q4 (2.99〜6.02) | 25% | -1.24 |

→ **AS 率がフラットで予測力ゼロ**。672# MI≈0 を追加実証。

### 1.5 全 pre-fill 特徴量の AS 判別力
| 特徴量 | |r| (point-biserial) | 備考 |
|--------|:---:|------|
| spread_offset_ratio | 0.155 | pre-fill 最高だが実用不可 |
| actual_measurement_sec | 0.145 | 待機時間と弱い相関 |
| effective_offset_used | 0.135 | offset パイプライン出力 |
| (post-hoc) pnl_bps_30s | 0.709 | 事後指標、予測不可能 |

→ **pre-fill 特徴量は全て |r| < 0.16。現行特徴量セットでの AS 予測は原理的に困難。**

---

## 2. 実施した変更

### 2.1 min_spread_atr_cap_bps: 2.0 → 1.2

**根拠**:
- 0-1500 帯 (n=12): AS=17%, avg PnL=+0.91（最優良バンド）
- Non-AS のみ (n=10): avg PnL=+2.1 bps
- cap_bps=2.0 時の実効 min_spread ≈ 2160 JPY → 1500 以下を完全ブロック
- cap_bps=1.2 時の実効 min_spread ≈ 1536 JPY → 47/97 ブロック注文を解放 (48%)

**リスク**: AS 率 17% は全帯で最低。ナロースプレッド帯での AS は短期 mean reversion が早く、
実害が小さい（672# Glosten-Milgrom モデルとも整合）。

```yaml
# configs/v460/fill_test.yaml
min_spread_atr_cap_bps: 1.2  # 686# 2.0→1.2
```

---

## 3. 追加分析結果

### 3.1 trending_up × buy (Task C)
| 条件 | n | Non-AS avg | AS avg | AS率 |
|------|---|:---:|:---:|---:|
| trending_up × buy | 36 | +0.99 | -6.02 | 22% |
| trending_up × sell | 28 | +0.48 | -5.04 | 32% |
| 他レジーム × buy | 86 | +0.52 | -4.65 | 21% |

→ trending_up/buy の問題は AS 発生時の severity（-6.02）であり AS 率ではない。
トレンド反転時の損失が大きい。レジーム固有の対策ではなく、universal AS severity 軽減が必要。

### 3.2 preflight_insufficient (Task D)
- order_quantity=0.001 BTC × 12,800,000 JPY/BTC ≈ **12,800 JPY**（1注文）
- 総資本 ≈ 21,700 JPY → 1 buy 後に残高 ≈ 8,900 JPY → 次の buy がブロック
- **ソフトウェアでは解決不可。資本増強（40,000+ JPY）が唯一の対策。**

### 3.3 605# Tier 1 残項目 (Task E)
| ID | 内容 | 状態 |
|----|------|------|
| T1-1 | Stage Max Mult | 672# offset 無効につき棚上げ |
| T1-2 | CV Widen | 641# disabled ✅ |
| T1-3 | EV toxic skip | 672# SG MI≈0 につき棚上げ |
| T1-4 | Sidecar TTL | 372# 7800s ✅ |
| T1-5 | sell_dynamic_kill duration | 540# 600s ✅ |

→ 5 件中 3 件完了、2 件は 672# 分析結果により合理的に棚上げ。

### 3.4 685# Codex コードレビュー (Task B)
20 ファイルをレビュー → **PASS**。クリティカルバグなし。
軽微指摘 3 件（docstring 明瞭化、コメント一貫性、テスト命名）。

---

## 4. Codex 次回タスク

### Task SG-1: SkipGate bypass モード
- `prompts/686_codex_task_skipgate_bypass.md`
- SG の予測力ゼロを受け、ブロック停止・スコア記録継続の bypass モード実装
- `skip_gate_bypassed` フラグで事後分析可能
- 期待効果: 48 件/日の不要ブロックを解放

### Task TI-1: テスト基盤修正
- `prompts/686_codex_task_test_infra_fix.md`
- `tests/unit/risk/test_rules.py` の fallback benchmark fixture が pytest-benchmark と競合
- INTERNALERROR 解消で CI exit code 0 を回復

---

## 5. 収益インパクト見積

| 施策 | blocked 解放 | 期待 avg PnL | 日次期待改善 |
|------|:---:|:---:|:---:|
| min_spread cap 1.2 | +12件/日 | +0.91 bps | +10.9 bps |
| SG bypass | +12件/日 | -0.16 bps (Q1 平均) | -1.9 bps |
| **合計** | **+24件/日** | — | **+9.0 bps** |

※ SG bypass の fill は SG score が低い（ブロック対象）ため Q1 平均を使用。
わずかにマイナスだが、母集団拡大でスプレッド収益機会が増え、学習データも豊富になる。

---

## 6. 次のアクション

1. ✅ min_spread_atr_cap_bps 変更（本 686#）
2. → Codex: SG bypass 実装（686_codex_task_skipgate_bypass.md）
3. → Codex: テスト基盤修正（686_codex_task_test_infra_fix.md）
4. → 運用: 資本増強検討（preflight 35% 解消には 40,000+ JPY 必要）
5. → 中期: pre-fill 特徴量の根本改善（マイクロストラクチャ特徴量導入）
