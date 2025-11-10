# アクションシグナルガイド改善：Executive Summary

**作成日**: 2025年11月10日  
**対象**: SAC v445 アクションシグナルガイドシステム  
**ステータス**: 深掘り分析完了 → 実装準備段階

---

## 🎯 コア問題の特定

### 現状：シグナル頻度が目標の1/7-1/17

```
実績: 7日間で 20シグナル → 平均 2.9回/日
目標: スキャルピング対応 → 20-50回/日
ギャップ: 7-17倍の乖離
```

### 根本原因：確率チェーンの組み合わせ

現在の `SignalGuidanceSystem` は複数の確率判定を直列化：

```
1. Base threshold判定            P ≈ 15%
2. Position guidance判定         P ≈ 80%  
3. Trend guidance判定           P ≈ 30%
4. Signal guidance判定          P ≈ 25%
   ↓
複合確率 = 0.15 × 0.8 × 0.3 × 0.25 ≈ 0.9%

結果: ほぼシグナルが発生しない
```

---

## 💡 改善案：3つのPhase戻り

### Phase 1: スコアベース決定論的システム（Week 1-2）

**現在**: 確率的（ランダム要素が強い）
**改善**: 決定論的スコアリング（0-100）

#### 導入するテクニカル指標
| 指標 | 効果 | 実装難度 |
|------|------|--------|
| RSI | 過買い/過売り検出 | ★☆☆ |
| MACD | トレンド強度 | ★★☆ |
| ボリンジャーバンド | サポート/レジスタンス | ★☆☆ |
| ATR | ボラティリティ | ★☆☆ |
| 出来高比率 | トレンド確実性 | ★☆☆ |

#### 期待効果
- シグナル頻度: 3→30回/日（10倍）
- 信頼度: 不明確→0.7-0.9
- スコア: なし→60-100

### Phase 2: マイクロトレンド検出（Week 3）

**導入**: 複数時間軸同時分析
- 1分足トレンド
- 5分足トレンド  
- 15分足トレンド
- 収束度計算

#### 期待効果
- シグナル精度: 向上
- スキャルピング対応: 実装完了
- ボラティリティ適応: 動的ポジションサイジング

### Phase 3: リスク管理と統計検証（Week 4）

**導入**: 
- Kelly criterion によるポジションサイジング
- VaR計算
- パフォーマンス指標（Sharpe, Sortino）
- ブートストラップ検証

#### 期待効果
- 信頼度: 統計的に検証
- リスク: 制御可能
- 年率リターン: 50%+達成

---

## 🚀 実装優先順位

### 即座にやるべき（Week 1）

```python
# 1. テクニカル指標モジュール作成
ztb/trading/signal/technical_indicators.py
  ├─ LightweightIndicators
  │  ├─ RSI計算
  │  ├─ MACD計算
  │  ├─ ボリンジャーバンド計算
  │  ├─ ATR計算
  │  └─ 出来高分析
  └─ IndicatorBuffer（インクリメンタル計算）

# 2. スコアリングシステム作成
ztb/trading/signal/signal_quality_scorer.py
  ├─ SignalQualityScorer
  │  ├─ calculate_score() - 方向別スコア計算
  │  ├─ should_execute() - 実行判定
  │  └─ _generate_reason() - 根拠生成
  └─ SignalScoreResult（データクラス）

# 3. 既存システムに統合
ztb/trading/signal/signal_guidance_system.py
  └─ apply_guidance() メソッド改善
     └─ スコアベース判定 + 従来ロジック（フォールバック）
```

### 期間別タイムライン

```
Week 1 (5-6営業日)
├─ テクニカル指標実装
├─ SignalQualityScorer実装
├─ ユニットテスト作成
└─ バックテスト準備

Week 2
├─ バックテスト実行
├─ パフォーマンス確認
└─ Phase 1効果検証 ✓

Week 3
├─ MicroTrendDetector実装
├─ ScalpingOptimizer実装
└─ マルチ時間軸テスト

Week 4
├─ PerformanceAnalyzer実装
├─ SignificanceValidator実装
└─ 統計検証

Week 5+
├─ 統合テスト
└─ ライブ検証開始
```

---

## 📊 成功基準

実装完了時の確認項目：

| 項目 | 現状 | Phase 1目標 | Phase 2目標 | Phase 3目標 |
|------|------|-----------|-----------|-----------|
| **シグナル頻度** | 2.9/日 | 15-20/日 | 30-40/日 | 30-50/日 |
| **平均スコア** | - | 65+ | 70+ | 75+ |
| **平均信頼度** | - | 0.65 | 0.75 | 0.85 |
| **Sharpe比率** | - | - | 1.5+ | 2.0+ |
| **勝率** | - | - | 52%+ | 55%+ |
| **年率リターン** | - | 20%+ | 40%+ | 50%+ |

---

## 🔍 技術的ハイライト

### 1. テクニカル指標の軽量実装

**依存**: NumPy/Pandasのみ（TA-Lib不要）
**計算量**: O(n) で高速
**精度**: 標準的なテクニカル分析と同等

```python
# 使用例
from ztb.trading.signal.technical_indicators import IndicatorBuffer

buffer = IndicatorBuffer()
buffer.update(open, high, low, close, volume)

rsi = buffer.get_rsi()           # 直近RSI取得
macd, signal, hist = buffer.get_macd()  # MACD取得
upper, mid, lower = buffer.get_bollinger_bands()  # BB取得
```

### 2. スコアベース決定の透明性

**全ての決定に根拠付き**:
- 各テクニカル指標のスコア（0-100）
- 指標の一致度（信頼度）
- 最終判定の理由（`reason` フィールド）

```python
result = scorer.calculate_score('buy')
print(f"Score: {result.score}, Confidence: {result.confidence}")
print(f"Reason: {result.reason}")
print(f"Components: {result.component_scores}")
```

### 3. リスク対応ポジションサイジング

**従来**: 固定的な80%, 10%判定
**改善**: 動的な計算
- Kelly criterion
- VaR（バリュー・アット・リスク）
- ボラティリティ適応

---

## ⚠️ リスク評価

### 実装リスク

| リスク | 確率 | 影響 | 緩和策 |
|--------|------|------|--------|
| テクニカル指標計算誤り | 低 | 中 | ユニットテスト + 既知値との比較 |
| バックテスト過最適化 | 中 | 高 | ウォークフォワード分析 + OOSテスト |
| ライブトレード不採算 | 中 | 高 | 段階的展開 + リスク管理 |

### 技術リスク

- **複雑性増加**: 新しいモジュールの追加
  → 段階的統合でテスト容易性確保
  
- **パフォーマンス**: テクニカル計算の追加
  → 軽量実装 + キャッシング戦略

- **メモリ**: 複数時間軸データ保持
  → `deque` による固定サイズバッファ

---

## 📝 参照ドキュメント

生成されたドキュメント：

1. **`SIGNAL_GUIDANCE_DEEP_ANALYSIS.md`**
   - 現状問題の詳細分析（確率チェーンの計算例等）
   - Phase 1-3の詳細設計
   - 成功指標の定義
   - **ボリューム**: 120+ KB

2. **`SIGNAL_IMPROVEMENT_QUICK_GUIDE.md`**
   - 即座に実装可能な改善（Week 1内容）
   - 完全なコード例（テクニカル指標、スコアラー）
   - ユニットテストコード例
   - バックテストスクリプト例
   - **ボリューム**: 90+ KB

---

## 🎓 学習資源

### 必読論文・記事

1. **テクニカル分析の基礎**
   - RSI: Wilder's Smoothing Method
   - MACD: Moving Average Convergence Divergence
   - Bollinger Bands: Standard Deviation-based bands

2. **ポジションサイジング**
   - Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
   - Fractional Kelly: Risk管理の観点から1/2 Kelly推奨

3. **統計検証**
   - Bootstrap: Resampling method by Efron
   - Walkforward Analysis: Overfitting回避

---

## ✅ アクションアイテム

### 今週中（優先度: 🔴高）

- [ ] `SIGNAL_IMPROVEMENT_QUICK_GUIDE.md` を確認
- [ ] `technical_indicators.py` の実装開始
- [ ] ユニットテスト作成
- [ ] 開発ブランチ作成: `feature/signal-quality-scoring`

### 来週中（優先度: 🟠中）

- [ ] `signal_quality_scorer.py` 実装
- [ ] 既存システムとの統合
- [ ] バックテスト実行
- [ ] Phase 1 の効果検証

### 3週目以降（優先度: 🟡低）

- [ ] Phase 2 実装
- [ ] Phase 3 実装
- [ ] 統合テスト
- [ ] ライブ検証準備

---

## 📞 質問・相談事項

実装過程での主要な判断ポイント：

1. **テクニカル指標の期間設定**
   - 推奨: RSI=14, MACD=(12,26,9), BB=20
   - スキャルピング対応時は短期化も検討

2. **信頼度閾値**
   - Phase 1: 0.70（高い信頼度で実行）
   - Phase 2: 0.65（高頻度対応）
   - ライブ: 運用結果に応じて動的調整

3. **リスク許容度**
   - 初期: 1%/trade
   - Kelly criterion 導入後: 自動計算

4. **バックテスト期間**
   - Phase 1: 3ヶ月
   - Phase 2: 6ヶ月
   - Phase 3: 12ヶ月+ウォークフォワード

---

## 🎬 まとめ

現在のシグナルガイドシステムは**確率チェーンの組み合わせ**により、実効的なシグナル頻度が目標の1/7-1/17に低下しています。

**改善の鍵**は：
1. **確率的→決定論的**: テクニカル指標によるスコアベース判定
2. **単一→複数**: マルチ時間軸分析による精度向上
3. **静的→動的**: リスク適応的なポジション管理

3段階の改善により、**スキャルピングレベルの高頻度取引（30-50回/日）**での**単体採算性（年率50%+）**を目指します。

実装期間: **4-5週間**
期待効果: **10倍のシグナル頻度増加** + **信頼度75%+**
