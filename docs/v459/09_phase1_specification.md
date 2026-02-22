# v459 Phase 1: P0バグ修正 仕様書 (09)

**Date**: 2026-01-22  
**Status**: 🔄 In Progress  
**Phase**: Phase 1 (P0 Bug Fixes)

---

## 1. Phase 1概要

### 1.1 目的

Phase 0で基盤整備を完了したv459プロジェクトのPhase 1では、Doc00で定義されたP0（Critical Priority）バグを修正します。

### 1.2 対象バグ（Doc00定義準拠）

| ID | Bug Name | Priority | Status |
|----|----------|----------|--------|
| P0-1 | Entry Gate Crash | Highest | ✅ Phase 0で対応済み |
| P0-2 | Entry Gate Config | High | ✅ Phase 0で対応済み |
| P0-3 | Cost Double-Count | High | 🔄 Phase 1で対応 |
| P0-4 | Val/Test Leakage | High | 🔄 Phase 1で強化 |

### 1.3 完了条件（Doc00準拠）

- [ ] 全P0バグ修正完了
- [ ] 統合テスト全パス
- [ ] 10エピソード手動検証（PnL整合性確認）
- [ ] Phase 1完了報告（Doc10）作成

---

## 2. 既存実装分析

### 2.1 コスト計算アーキテクチャ

#### 現在の実装状況

**Fast Intraday Env V456**:
- `FastIntradayAccounting`でgross_pnl/net_pnl管理
- `net_pnl = gross_pnl - total_fees - total_slippage` (Line 32)
- Environmentは**net_pnl**をベースに報酬計算・balance更新

```python
# ztb/trading/environment/fast_intraday_env_v456.py:771
self.accounting.net_pnl = self.accounting.gross_pnl - self.accounting.total_fees - self.accounting.total_slippage
```

**Reporter (Phase 0.2a強化版)**:
- `record_trade()`でtrade単位記録
- `net_pnl = pnl` (コストはenv側で既に反映済みと想定)
- `stats["net_pnl"]`に累積

```python
# ztb/evaluation/walk_forward/reporter.py:426-435
net_pnl = pnl
if net_pnl > 0:
    winning_trades += 1
elif net_pnl < 0:
    losing_trades += 1
self.stats["gross_pnl"] += pnl
self.stats["net_pnl"] += net_pnl
```

#### P0-3問題：Cost Double-Count

**現象**:
- Envがnet_pnlを計算（コスト込み）
- Reporter側では`record_trade()`がenv infoから`pnl`を受け取る
- `pnl`がgross（コスト未控除）かnet（コスト控除済み）か曖昧
- Reporterが再度feeを加算する可能性（二重計上）

**根本原因**:
1. **env→reporter間のPnL規約が不明確**
   - env infoの`trade_pnl`が`gross`か`net`か不統一
   - reporter側で`fee`と`slippage`パラメータを受け取るが使用していない

2. **Reporter内部の矛盾**
   - `record_trade()`は`fee`と`slippage`を受け取る
   - しかし`net_pnl = pnl`（費用控除なし）
   - `trade_history`に`fee`と`slippage`を記録するが活用されていない

3. **Evaluatorでの不整合**
   - `_evaluate_on_df()`内で`reporter.record_trade()`呼び出し時、envの`info`から`fee_paid`/`slippage_paid`を取得
   - しかしenvの`info["trade_pnl"]`自体がnet（コスト控除済み）の可能性

### 2.2 データリーク対策状況

#### Phase 0での対応

- **CausalOnlineScaler**: `fit(data, end_idx)`でTrain期間限定のfit実装済み（inclusive）
- **CausalGroupedScaler**: 同様にfit範囲制限、警告ベース検証（tolerance=2.0）
- **Reporter**: Trade Type分類強化（8種+reverse/hold）

#### P0-4残存課題

**Val/Test Reporter分離**:
- 現在、`evaluator.py:286-287`で`val_reporter`と`test_reporter`を別インスタンス化
- しかし、同一環境（同一scalerインスタンス）を共有する可能性
- **MTF因果性検証**: Phase 0で仕様策定のみ、実装は未完了

**分離要件**:
1. Val/Test評価時に**完全に独立した環境インスタンス**を使用
2. 各環境が独立したscaler statを持つ（相互汚染防止）
3. Reporter統計もVal/Test完全分離

---

## 3. Phase 1修正方針

### 3.1 P0-3: Cost Double-Count修正

#### 設計原則

**PnL規約の統一**:
- **Env側**: `net_pnl`（コスト控除済み）をinfoで提供
- **Reporter側**: infoから受け取った`pnl`は既に`net`として扱う（再控除しない）
- **検証目的のみ**: `fee`と`slippage`は統計・検証用に記録

#### 実装変更

1. **Env (fast_intraday_env_v456.py)**
   - `info`辞書に`trade_pnl`を追加（net_pnl変化分）
   - 既存の`fee_paid`/`slippage_paid`は継続提供（検証用）
   - `step()`メソッドのinfo構築部分を明確化

2. **Reporter (reporter.py)**
   - `record_trade()`のdocstringを更新
   - `pnl`パラメータは`net_pnl`（コスト控除済み）として明記
   - `fee`/`slippage`は統計・検証用として記録（二重控除しない）
   - `_calculate_profit_factor()`等の計算はnet_pnlベースで正しく動作することを確認

3. **Evaluator (evaluator.py)**
   - `_evaluate_on_df()`の`reporter.record_trade()`呼び出し箇所を確認
   - envの`info`から取得する`pnl`がnetであることをコメント明記
   - 既存のfee/slippage取得は検証用として継続

#### テスト戦略

- **単体テスト**: `test_cost_double_count_prevention.py`
  - Envのinfo["trade_pnl"]がnet_pnlであることを確認
  - Reporter.stats["net_pnl"]がenv.net_pnlと一致することを確認
  - fee/slippageが二重計上されていないことを確認

- **統合テスト**: `test_phase1_integration.py`
  - 10エピソード実行し、各エピソードでenv.net_pnl == reporter.stats["net_pnl"]
  - total_fees/total_slippageの整合性検証

### 3.2 P0-4: Val/Test Leakage対策強化

#### 設計原則

**完全分離の3原則**:
1. **環境分離**: Val/Test評価に独立した環境インスタンス
2. **Scaler分離**: 各環境が独自のscaler stateを保持
3. **Reporter分離**: Val/Test統計は完全に独立（Phase 0で対応済み）

#### 実装変更

1. **Evaluator (evaluator.py)**
   - `_evaluate_on_df()`を修正し、評価ごとに新しい環境インスタンスを生成
   - 環境のscalerがreset()され、prewarmで独立した統計を構築
   - Val評価とTest評価で環境を再作成（scalerリセット含む）

2. **MTF因果性検証（低優先度）**
   - Phase 0で仕様策定済み（Doc04）
   - 実装はPhase 2以降に延期（P0-4の主要部分はscaler分離）

#### テスト戦略

- **単体テスト**: `test_val_test_isolation.py`
  - Val/Test評価で異なる環境インスタンスIDを確認
  - 各環境のscalerが独立したstateを持つことを確認

- **統合テスト**: `test_phase1_integration.py`
  - Walk-Forward評価でVal/Test統計の完全分離を確認
  - Scaler fit範囲がTrain期間に限定されることを検証

---

## 4. 実装計画

### 4.1 実装ファイルリスト

| File | 修正内容 | Lines Est. |
|------|----------|------------|
| `fast_intraday_env_v456.py` | info["trade_pnl"]追加、docstring明確化 | ~30 |
| `reporter.py` | docstring更新、net_pnl規約明記 | ~20 |
| `evaluator.py` | env再生成によるVal/Test分離 | ~40 |

### 4.2 テストファイルリスト

| File | テスト内容 | Tests Est. |
|------|-----------|------------|
| `test_cost_double_count_prevention.py` | P0-3修正検証 | 5 |
| `test_val_test_isolation.py` | P0-4環境分離検証 | 4 |
| `test_phase1_integration.py` | Phase 1統合テスト | 3 |

### 4.3 タイムライン

| Task | 工数 | 依存関係 |
|------|------|----------|
| P0-1/P0-2完了確認 | 0.5h | - |
| P0-3実装 | 2h | - |
| P0-3テスト | 1.5h | P0-3実装 |
| P0-4実装 | 2h | - |
| P0-4テスト | 1h | P0-4実装 |
| 統合テスト | 1.5h | P0-3/P0-4完了 |
| 10エピソード手動検証 | 1h | 統合テスト |
| Doc10作成 | 1h | 全完了 |
| **合計** | **10.5h (1.5日)** | |

---

## 5. リスク評価

### 5.1 高リスク項目

1. **Reporter規約変更の影響範囲**
   - **リスク**: 他のスクリプト（backtest_v456.py等）がreporterの旧規約に依存
   - **対策**: 全呼び出し箇所をgrep検索、影響確認
   - **軽減**: docstringで明確化、後方互換性維持

2. **Env再生成のパフォーマンス影響**
   - **リスク**: Val/Test評価ごとの環境再生成でオーバーヘッド
   - **対策**: 環境生成コストの計測、必要に応じてキャッシュ検討
   - **軽減**: 評価は非リアルタイム、許容可能と想定

### 5.2 中リスク項目

1. **MTF因果性検証の延期**
   - **リスク**: Phase 2以降でMTFリークが発覚する可能性
   - **対策**: Phase 0で仕様策定済み、早期発見が可能
   - **軽減**: P0-4の主要部分（scaler分離）は対応済み

---

## 6. 成功基準

### 6.1 技術基準

- [ ] `test_cost_double_count_prevention.py`: 5/5テストパス
- [ ] `test_val_test_isolation.py`: 4/4テストパス
- [ ] `test_phase1_integration.py`: 3/3テストパス
- [ ] 既存テスト（Phase 0: 77テスト）全パス維持

### 6.2 検証基準

- [ ] 10エピソード手動検証でenv.net_pnl == reporter.stats["net_pnl"]（許容誤差±0.01%）
- [ ] Val/Test reporter統計が完全分離（共通trade無し）
- [ ] total_fees/total_slippageの二重計上なし

### 6.3 ドキュメント基準

- [ ] Doc09（Phase 1仕様書）完成
- [ ] Doc10（Phase 1完了報告）作成
- [ ] 修正内容の変更履歴記録（CHANGELOG.md更新）

---

## 7. Phase 2への引き継ぎ

### 7.1 Phase 1完了後の状態

- ✅ P0バグ全修正完了
- ✅ PnL規約統一（env=net, reporter=検証のみ）
- ✅ Val/Test環境完全分離
- ⏳ MTF因果性実装（Phase 2以降）

### 7.2 Phase 2準備項目

- P1バグ修正（Trade Type分類、Entry Price更新、Reporter統合、AB Testing）
- Reporter統一（3実装→1実装）
- AB Testing有効化
- MTF因果性検証実装（P0-4完全完了）

---

## Appendix A: 既存実装のリファクタリング余地

### A.1 TransactionCostCalculator活用

**既存実装**:
- `ztb/trading/cost/venue_transaction_cost_manager.py`
- `ztb/trading/trade_execution_engine.py`の`TransactionCostCalculator`

**活用可能性**:
- P0-3修正では既存のFastIntradayAccountingで十分（シンプル）
- Phase 5（Paper Trading統合）で本格活用を検討
- 現時点では不要な複雑性を避ける

### A.2 Reporter Trade History構造

**現在**:
```python
self.trade_history.append({
    "trade_type": trade_type,
    "gross_pnl": pnl,
    "net_pnl": net_pnl,
    "fee": fee,
    "slippage": slippage,
    ...
})
```

**改善提案**:
- `gross_pnl`と`net_pnl`の区別を明確化
- Phase 1では`pnl`を`net_pnl`として統一
- `gross_pnl`は将来の詳細分析用に保留（Phase 2以降）

### A.3 Evaluator環境キャッシュ

**現在**: 評価ごとに環境を再生成
**改善案**: Val/Test間で環境を再利用（resetのみ）

**判断**:
- Phase 1では完全分離を優先（再生成）
- パフォーマンス計測後、Phase 2で最適化検討
