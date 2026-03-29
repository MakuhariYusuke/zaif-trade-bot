# 650# Roundtrip Analysis & Financial Engineering Sections

## 概要
`analyze_fill_logs.py` に金融工学的知見に基づく4つの新分析セクションを追加。
個別 fill の 30s PnL ではなく、実際のラウンドトリップ（買売ペア）の実現損益で
マーケットメイキング戦略の収益構造を評価可能にした。

## 追加セクション

### 1. `section_roundtrip` — Roundtrip PnL (Avellaneda-Stoikov 2008)
- 連続する異サイド約定をペアリングし Gross Roundtrip PnL を算出
- WR、PF、avg win/loss、hold time、regime/entry-side/worst 分解
- 個別 fill PnL と roundtrip PnL の乖離を可視化

### 2. `section_inventory_health` — 在庫非対称性 (Kyle 1985)
- `preflight_insufficient` の集中度から buy/sell 非対称性を定量化
- `balance_jpy/btc_at_order` から 50/50 乖離度を算出 (LOW/MEDIUM/HIGH)
- 連続 preflight_insufficient の max run length (deadlock indicator)

### 3. `section_mcb_impact` — MCB HALT 影響 (Foucault et al. 2007)
- MCB halt の時間帯分布
- MCB 前後の fill PnL 比較 (pre-MCB / during-post-MCB / outside window)
- MCB regime 分布

### 4. `section_spread_fill_quality` — Spread vs PnL (Glosten-Milgrom 1985)
- spread_bps を quartile 分割し各帯の avg PnL / AS率 / avg wait を比較
- 低スプレッド fill の逆選択リスクを定量化
- Side 別の spread-PnL 相関係数

## 2026-03-29 分析結果サマリ (SHA 5832c87fe)
- **13 RT, WR 46.2%, PF 0.98, Total -0.81bps**
- sell-entry 11 RT で損失集中 (-1.06bps)、buy-entry 2 RT は +0.25bps
- **Inventory imbalance 49.5pp [HIGH]** — JPY 0.5%, BTC 99.5%
  - preflight_insufficient が 43.3% of all cycles → buy 注文がほぼ不可能
- **MCB halts 9回** — RT#3 で 50min 閉塞、-14.27bps の最大損失を発生
- **Q1 (低スプレッド < 2.4bps): avg_pnl -0.86bps, AS 14%** vs Q2-Q3: +1.7bps
  → Glosten-Milgrom 逆選択と整合

## 改善示唆 (収益トレードを毀損しない範囲)
1. MCB halt 中の既存ポジション: 緊急ヘッジ or 想定損失の事前評価
2. 在庫リバランス: JPY/BTC 比率が 30pp 超で buy-side offset 調整
3. 低スプレッド環境での出控え: spread < Q25 かつ trending 時の offset 引上げ
4. ranging regime の sell strategy は WR~60%、保護が必要

## テスト
- 既存 21 + 新規 11 = 32 テスト全通過
