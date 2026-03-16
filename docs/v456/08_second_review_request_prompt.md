# v456 Second Review Request: 外部レビュー反映後の再レビュー依頼

> **Date**: 2026-01-13  
> **Purpose**: 初回レビュー指摘事項の反映確認 + 残存課題の発見

---

## 1. 背景

仮想通貨（BTC/JPY）自動取引システム v456 の設計ドキュメントについて、初回の外部レビュー（06_review_response.md）で以下のCritical/Major Issuesが指摘されました：

### 初回レビューで指摘されたCritical Issues
1. **C-1**: MTFリサンプリングで未来データリークの可能性
2. **C-2**: 正規化パイプラインでカテゴリカル/時間特徴量への不適切なスケーリング
3. **C-3**: GRU + Off-Policy SAC のシーケンスリプレイ/burn-in設計が未詳細

### 初回レビューで指摘されたMajor Issues
4. **M-1**: 報酬シェーピング項がPnL項を支配するリスク
5. **M-2**: グローバル特徴量でFX/ベーシスリスクを無視
6. **M-3**: アクションフィルタリングのtrain-liveミスマッチ

これらの指摘を受けて、以下のドキュメントを改訂しました。

---

## 2. 改訂済みドキュメント一覧

以下のドキュメントをレビューしてください：

### 2.1. 00_improvement_proposal.md（改訂版）
- 優先順位の改訂（データ整合性 → MTF → ベースライン → フィルタリング → GRU）
- KPIの統計的根拠追加（95%信頼区間、ブートストラップ検証）
- 段階的マイルストーンの明確化

### 2.2. 01_technical_specification.md（改訂版）
- 報酬シェーピングキャリブレーション検証ロジック追加
- Train-Live Parity: Soft Filter/Calibration Gateの環境step()内部統合
- GRU導入条件の明確化（MLPベースライン Sharpe > 0.3 が前提）
- シーケンスリプレイ（Burn-in方式）設計の追加

### 2.3. 02_feature_engineering_spec.md（改訂版）
- MTFバーアライメント: `get_mtf_closed_bar()` でクローズドバーのみ使用
- 正規化グループ分離: `NORMALIZATION_GROUPS` 定義
- タイムゾーン処理: `validate_and_convert_timestamp()` でnaive timestamp拒否
- FX/ベーシス特徴量追加: `global_fx_adjusted_spread`, `global_usdjpy_return_1m`
- データ鮮度フラグ追加: `global_data_stale_flag`

### 2.4. 03_implementation_checklist.md（改訂版）
- Phase 0（最優先）: データ整合性チェック項目追加
- MTFリーク検出テスト項目追加
- 正規化分離確認項目追加
- Train-Live Parityチェック項目追加

### 2.5. 07_revised_action_plan.md（新規）
- Critical Issues対応表
- 改訂版フェーズ計画（Week 1-6+）
- リスク管理・フォールバック計画

---

## 3. レビュー観点

以下の観点でレビューをお願いします：

### 3.1. 初回指摘事項の反映確認
各Critical/Major Issueに対する対応が**十分かつ適切**か確認してください：

| Issue ID | 指摘内容 | 対応箇所 | 確認ポイント |
|----------|---------|---------|-------------|
| C-1 | MTF未来リーク | 02_feature_engineering_spec.md §3.3.0 | `get_mtf_closed_bar()`のロジックは正しいか？ |
| C-2 | 正規化混在 | 02_feature_engineering_spec.md §1.2 | グループ分離は漏れなく定義されているか？ |
| C-3 | GRU設計不足 | 01_technical_specification.md §6.0-6.3 | Burn-in設計は実装可能か？導入条件は妥当か？ |
| M-1 | 報酬ハッキング | 01_technical_specification.md §2.2.0 | キャリブレーションロジックは有効か？ |
| M-2 | FX無視 | 02_feature_engineering_spec.md §5.2-5.3 | FX調整ロジックは正しいか？ |
| M-3 | Train-Live乖離 | 01_technical_specification.md §4.3 | 環境内統合設計は十分か？ |

### 3.2. 新たな矛盾・問題の発見
改訂により新たに生じた可能性のある問題を探してください：

1. **特徴量数の整合性**: 各ドキュメントで特徴量数が一致しているか？（88? 91?）
2. **優先順位の一貫性**: 00_improvement_proposal.md と 07_revised_action_plan.md で優先順位が一致しているか？
3. **KPI目標の妥当性**: 改訂版KPI（必達: Sharpe > 0.3、挑戦: Sharpe > 1.0）は達成可能か？
4. **実装の複雑性**: 追加された機能（FX取得、データ鮮度フラグ等）により実装が過度に複雑化していないか？
5. **テスト設計の網羅性**: チェックリストのテスト項目は十分か？

### 3.3. 見落としている重要事項
初回レビューで指摘されなかったが、実装上重要な観点はないか：

- バックテストの分割方法（embargo/purged CV）は十分に設計されているか？
- Circuit Breakerの動的閾値調整は考慮されているか？
- ライブ運用時のモニタリング・アラート設計はあるか？

---

## 4. 具体的な質問事項

### Q1: MTFクローズドバー取得ロジックの妥当性
```python
def get_mtf_closed_bar(current_1m_timestamp, mtf_timeframe, mtf_data):
    closed_bar_time = current_1m_timestamp.floor(mtf_timeframe)
    if closed_bar_time == current_1m_timestamp.floor(mtf_timeframe):
        closed_bar_time = closed_bar_time - pd.Timedelta(mtf_timeframe)
    ...
```
このロジックで全てのエッジケース（00:00境界、週末、データ欠損）を正しくカバーできているか？

### Q2: 報酬シェーピング比率の閾値
```python
max_shaping_ratio = 0.5  # シェーピングはPnLの50%以下
target_ratio = 0.3       # 自動キャリブレーション目標
```
この閾値設定は経験的に妥当か？文献等での推奨値はあるか？

### Q3: GRU導入の前提条件
```python
GRU_PREREQUISITE = {
    "mlp_baseline_sharpe": 0.3,  # この値は妥当か？
}
```
Sharpe 0.3 という閾値は、GRU導入を検討するのに十分なベースラインと言えるか？

### Q4: FX調整スプレッドの計算
```python
spread = (local_btcjpy - global_btcusdt * usdjpy * usdt_premium) / (global_btcusdt * usdjpy * usdt_premium)
```
USDTプレミアムを考慮しているが、他に考慮すべき要因（取引所固有のプレミアム等）はないか？

### Q5: データ鮮度フラグの活用
`global_data_stale_flag = 1` の場合、エージェントはどのように振る舞うべきか？
現設計では特徴量として提供するのみだが、フィルタリング/ゲーティングに組み込むべきか？

---

## 5. 参考：改訂版ドキュメント構成

```
docs/v456/
├── 00_improvement_proposal.md      # 改善提案（改訂版）
├── 01_technical_specification.md   # 技術仕様（改訂版）
├── 02_feature_engineering_spec.md  # 特徴量設計（改訂版）
├── 03_implementation_checklist.md  # 実装チェックリスト（改訂版）
├── 04_self_review.md               # セルフレビュー
├── 05_review_request_prompt.md     # 初回レビュー依頼
├── 06_review_response.md           # 初回レビュー回答
├── 07_revised_action_plan.md       # 改訂版アクションプラン（新規）
└── 08_second_review_request_prompt.md  # 本ドキュメント
```

---

## 6. 期待するレビュー出力形式

以下の形式での回答をお願いします：

```markdown
## 初回指摘事項の反映確認

### C-1: MTF未来リーク
- 反映状況: [十分 / 部分的 / 不十分]
- コメント: ...

### C-2: 正規化混在
- 反映状況: [十分 / 部分的 / 不十分]
- コメント: ...

（以下同様）

## 新たに発見された問題

### [Critical / Major / Moderate / Minor]
1. 問題の説明
2. 該当箇所
3. 推奨対応

## 質問への回答

### Q1: MTFクローズドバー
- 回答: ...

（以下同様）

## 総合評価
- 実装着手可否: [可 / 条件付き可 / 要再検討]
- 総合コメント: ...
```

---

## 7. 添付ドキュメント

以下のドキュメントを添付してレビューを依頼します：

1. **00_improvement_proposal.md** - 改善提案（改訂版）
2. **01_technical_specification.md** - 技術仕様（改訂版）
3. **02_feature_engineering_spec.md** - 特徴量設計（改訂版）
4. **03_implementation_checklist.md** - 実装チェックリスト（改訂版）
5. **07_revised_action_plan.md** - 改訂版アクションプラン

---

*本レビュー依頼は、初回レビュー（06_review_response.md）の指摘を反映した改訂版ドキュメントに対するものです。*
*「短期間での高収益性システム構築」という大義のもと、実装前の最終確認として忌憚のないご意見をお願いします。*
