# 023# — G1.1 待機中の並行タスク実行計画

| key | value |
|---|---|
| 番号 | 023 |
| フェーズ | ph2 (G1.1-exec) |
| 種別 | plan (計画) |
| 作成日 | 2026-02-14 |
| 状態 | **ACTIVE** |
| 前提文書 | 009# (G1.1 計画), 019# (fill test n=105), 020# (O1-O5), 022# (データロス調査) |

---

## 0. 背景

G1.1 PASS には **n≥200 & 3暦日** が必要。現在 n=105（+ データロスにより未保存分あり）。

fill test 完走までの推定待機時間:
- 残り: 95サイクル × 120秒 ≈ **3.2時間**（サイクル数のみ）
- 3暦日要件: 最速で **2026-02-16** まで（開始日 02/13 → 3日目 02/15 末)
- **結論: 最低2日間の待機が発生** → その間に先行可能なタスクを実行

---

## 1. 優先度分類

| 優先度 | カテゴリ | 概要 | G1.1 依存 |
|---|---|---|---|
| **P0** | データロス対応 | 022# の調査・修正・再起動 | **直接影響** |
| **P1-A** | SAC デッドコード削除 | 021# で特定済み、5ファイル | なし |
| **P1-B** | Monte Carlo PnL 再計算 | n=105→最新での収益見積 | 間接 |
| **P1-C** | G1 プロキシ検証 | XGBoost on OHLCV 104MB | なし |
| **P1-D** | 実特徴量設計拡張 | microstructure features | なし |
| **P1-E** | ph3 SAC 訓練パイプライン | G1.1 PASS 後すぐ開始可能に | なし |
| **P2** | ドキュメント整理 | analysis/ 内の一時ファイル清掃 | なし |

---

## 2. P0: データロス対応（最優先）

### 目的
fill test の JSONL 書込停止を解消し、データ蓄積を再開する。

### 手順

```
Step 1: 022# の外部AIレビュー結果を受領
Step 2: ログ確認 — PID 48100 の stdout/stderr にエラー出力があるか
Step 3: メモリ上データ救出可否の判定
        ├─ 可能 → py-spy / pyrasite でダンプ → JSONL に手動保存
        └─ 不可 → データロスを受容 (n=105 のまま再計数)
Step 4: PID 48100 の安全な停止 (SIGINT → _cleanup_sync → pending order cancel)
Step 5: コード修正 (022# の回答に基づく)
Step 6: fill test 再起動 (修正済みコードで)
```

### 修正方針（暫定）

1. `_save_batch` を try/except で個別ラップし、失敗時に **batch を保持**
2. `except Exception: continue` に **詳細ログ** (traceback) を追加
3. **フォールバック保存**: `_save_batch` N回連続失敗時に全 `records` をフルダンプ
4. **atexit handler** で未保存 batch をファイルに退避

### 完了条件
- [ ] データロスの根本原因特定
- [ ] コード修正 & テスト
- [ ] fill test 再起動 & JSONL 書込確認
- [ ] n=200 到達まで安定稼働確認

---

## 3. P1-A: SAC デッドコード削除

### 目的
021# で特定されたコード重複を解消し、保守性を改善。

### 対象ファイル (021# より)

| ファイル | 行数 | 内容 |
|---|---|---|
| `ztb/training/sac/` 配下 | ~2,500行 | 旧トレーニングコード |
| `ztb/models/` 配下  | ~800行 | 使われていないモデル定義 |
| `scripts/` 配下 | ~1,200行 | 旧スクリプト |

### 手順

```
1. 021# のリストに基づき削除対象を確認
2. grep で import/呼出し元がないことを検証
3. 削除 & テスト実行 (regression 確認)
4. コミット
```

### 所要時間: 1-2h
### G1.1 依存: なし（即時実行可能）

---

## 4. P1-B: Monte Carlo PnL 再計算

### 目的
n=105 の実測データから、v460 戦略の期待収益を Monte Carlo シミュレーションで再見積する。

### 入力
- `fill_records_20260213.jsonl` (n=105)
- 各レコードの `post_fill_30s_pnl` (bps)

### 計算
```
1. 実測 PnL 分布からブートストラップサンプリング (N=10,000 trials)
2. 1日あたりの期待サイクル数 (720 cycles/day @ 120s interval)
3. 手数料控除後の net PnL 分布
4. 95% CI, Sharpe ratio, max drawdown の推定
```

### アウトプット
- 日次・月次 PnL の分布と信頼区間
- G1.1 PASS 時の収益見込み

### 所要時間: 2-3h
### G1.1 依存: 間接（PASS 見込みの判断材料）

---

## 5. P1-C: G1 プロキシ検証 (XGBoost)

### 目的
G1（収益予測モデル）の実現可能性を、OHLCV データで事前検証する。

### 手法
```
1. data/ 内の OHLCV 104MB を読み込み
2. microstructure features (spread, imbalance, VWAP deviation) を算出
3. XGBoost で次期リターン予測モデルを構築
4. Walk-forward CV で Sharpe > 0 を達成可能か検証
```

### 意義
- SAC 訓練の前に、特徴量の有効性を軽量モデルで確認
- G1.1 PASS 後の SAC 設計に必要な特徴量セットを事前決定

### 所要時間: 3-4h
### G1.1 依存: なし

---

## 6. P1-D: 実特徴量設計拡張

### 目的
現在の基本特徴量セットを拡張し、SAC モデルの入力品質を向上。

### 候補特徴量

| カテゴリ | 特徴量 | 根拠 |
|---|---|---|
| Orderbook | bid-ask imbalance (top 5 levels) | 短期方向性 |
| Orderbook | depth-weighted mid price | ノイズ耐性 |
| Trade flow | trade imbalance (buy/sell volume ratio) | モメンタム |
| Trade flow | VWAP deviation | 価格乖離 |
| Volatility | realized vol (1min, 5min) | リスク指標 |
| Microstructure | spread z-score | レジーム検出 |

### 所要時間: 2-3h (設計のみ、実装は P1-C 結果後)
### G1.1 依存: なし

---

## 7. P1-E: ph3 SAC 訓練パイプライン事前構築

### 目的
G1.1 PASS 後に即座に SAC 訓練を開始できるよう、パイプラインを事前構築。

### タスク

```
1. 訓練環境の設計 (gym.Env wrapper)
   - 状態空間: P1-D の特徴量セット
   - 行動空間: {hold, buy_limit, sell_limit} + price offset
   - 報酬関数: realized PnL - spread cost - AS penalty

2. SAC ハイパーパラメータの初期設定
   - alpha (temperature) auto-tuning
   - batch_size, buffer_size
   - network architecture (actor/critic)

3. 評価パイプライン
   - Walk-forward backtesting
   - Sharpe / max drawdown / PnL curve

4. 学習監視ダッシュボード (TensorBoard / W&B)
```

### 所要時間: 4-6h (skeleton 実装)
### G1.1 依存: なし（ただし PASS 条件により微調整必要）

---

## 8. P2: ドキュメント・リポジトリ整理

### 目的
`analysis/` 配下の一時スクリプト群 (50+ ファイル) と `tmp_*.py` の整理。

### 方針
- 必要なものを `tools/` or `scripts/` に移動
- 不要なものはアーカイブ or 削除
- ルート直下の `tmp_*.py` を削除

### 所要時間: 1h
### G1.1 依存: なし

---

## 9. 実行スケジュール

### 2/14 (本日) — P0 集中

| 時間帯 | タスク | 備考 |
|---|---|---|
| AM | **P0**: 022# 外部レビュー送付・結果待ち | データロス原因特定 |
| AM | **P1-A**: SAC デッドコード削除 | P0 待ちの隙に |
| PM | **P0**: コード修正・fill test 再起動 | レビュー結果反映 |
| PM | **P1-B**: Monte Carlo PnL 再計算 | n=105 データ活用 |

### 2/15 (明日) — P1 並行

| 時間帯 | タスク | 備考 |
|---|---|---|
| AM | **P1-C**: G1 プロキシ検証 (XGBoost) | OHLCV 104MB |
| PM | **P1-D**: 実特徴量設計 | P1-C 結果を反映 |
| 終日 | fill test 稼働監視 | n=200 到達確認 |

### 2/16 (明後日) — ph3 準備

| 時間帯 | タスク | 備考 |
|---|---|---|
| AM | **P1-E**: SAC パイプライン skeleton | G1.1 PASS 見込み時 |
| PM | **P2**: リポジトリ整理 | |
| PM | **G1.1 判定実施** | n≥200 & 3暦日達成見込み |

---

## 10. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| P0 未解決でデータロス継続 | G1.1 遅延 | プロセス再起動で暫定対応 (n=105 リセット) |
| n=105 データのみで Monte Carlo 精度不足 | 見積り信頼性低下 | CI を広く取り、感度分析実施 |
| XGBoost で特徴量が有効でない | SAC 設計見直し | 別特徴量セットでの再検証 |
| fill test 再起動後も JSONL 書込失敗 | 長期遅延 | ログ詳細化 + 1サイクルずつ保存に変更 |

---

## 11. 成功基準

- [ ] P0 データロス解消 & fill test 安定稼働
- [ ] G1.1 PASS (n≥200, 3暦日, E1-E5 全 PASS)
- [ ] Monte Carlo で日次正の期待収益確認
- [ ] SAC パイプライン skeleton 完成
- [ ] リポジトリ clean up 完了
