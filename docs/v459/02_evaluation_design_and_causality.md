# v459 評価設計と因果性検証 (02)

**Date**: 2026-01-22  
**Status**: 📝 Planning  
**Purpose**: Doc01レビューで指摘された評価設計の穴を埋める

---

## 1. 概要

Doc01レビューで指摘された評価設計の問題点に対応し、以下を明確化する：

1. **スケーラfit範囲とリーク検査手順**
2. **Walk-Forwardの因果性保証**
3. **Multi-Seed × Multi-Split検証設計**
4. **ベースライン比較の拡充**
5. **実行モデル定義**

---

## 2. スケーラfit範囲とリーク検査

### 2.1 リーク検査の対象

| 検査対象 | リークの種類 | 検出方法 |
|----------|-------------|----------|
| OnlineScaler | train期間外データでのfit | fit呼び出しタイミング検査 |
| MTF特徴量 | 未来データの参照 | t時点でt+Nのデータ確認 |
| ローリング指標 | look-ahead bias | closed/label設定確認 |
| 特徴量生成 | 全期間統計の使用 | 生成関数の入力範囲確認 |

### 2.2 スケーラfit範囲の規則

```
データ分割イメージ:
|======== Train ========|=== Val ===|=== Test ===|
                        ^           ^
                    fit終了点    fit禁止区間

規則:
1. OnlineScalerのfit()はTrain期間のみで実行
2. Val/Test期間はtransform()のみ使用
3. 増分学習（partial_fit）もTrain期間に限定
```

### 2.3 リーク検査スクリプト仕様

```python
# scripts/v459/check_data_leakage.py

def check_scaler_leakage(scaler, train_end_idx, data):
    """スケーラが適切な範囲でfitされているか検査"""
    # 1. fit履歴の確認（train_end_idx以降のデータがfitに使われていないか）
    # 2. mean/std がtrain期間のみから計算されているか検証
    pass

def check_mtf_causality(df, mtf_columns, timeframes):
    """MTF特徴量が因果性を保っているか検査"""
    for tf in timeframes:
        offset = get_offset_for_timeframe(tf)  # 5m=5, 15m=15, 1h=60
        # t時点のMTF値が t-offset 以前のデータのみから計算されているか
    pass

def check_rolling_causality(df, rolling_columns):
    """ローリング指標がlook-aheadしていないか検査"""
    # closed='left', label='left' の設定確認
    pass

def run_full_leakage_check(config, data_path) -> LeakageReport:
    """全リーク検査を実行"""
    pass
```

### 2.4 検査合格基準

| 検査項目 | 合格条件 | 判定方法 |
|----------|----------|----------|
| Scaler fit範囲 | Train期間100% | fit呼び出しログ |
| MTF因果性 | 全時点でoffset遵守 | サンプル100点検査 |
| Rolling因果性 | look-ahead = 0 | 設定値確認 |
| 特徴生成範囲 | 入力 ⊆ 利用可能期間 | 関数引数追跡 |

### 2.5 ウォームアップ規定（Doc03対応）

> **目的**: ローリング指標やMTFがどこまで過去を参照できるかを明確化

```yaml
warmup_config:
  # 最大参照期間（これ以上過去に過ることはできない）
  max_lookback_1m: 1440     # 1分足: 24時間 = 1440本
  max_lookback_5m: 288      # 5分足: 24時間 = 288本
  max_lookback_15m: 96      # 15分足: 24時間 = 96本
  max_lookback_1h: 168      # 1時間足: 7日 = 168本
  
  # 各期間開始時のウォームアップ
  train_warmup: max_lookback_1h  # Train開始前に168時間のデータが必要
  val_warmup: 0                   # ValはTrain後なので追加不要
  test_warmup: 0                  # TestはVal後なので追加不要
```

#### 境界条件の明確化

| 境界 | 許容範囲 | 禁止事項 |
|------|----------|----------|
| Train開始時点 | Train開始 - max_lookbackまでのデータ | Val/Testのデータ |
| Val開始時点 | Train全期間 + Val開始まで | Testのデータ |
| Test開始時点 | Train + Val全期間 + Test開始まで | 未来データ |

```python
def validate_warmup(period_start_idx: int, data_start_idx: int, max_lookback: int) -> bool:
    """ウォームアップが十分か検証"""
    available_warmup = period_start_idx - data_start_idx
    if available_warmup < max_lookback:
        raise ValueError(
            f"ウォームアップ不足: 必要={max_lookback}, 利用可能={available_warmup}"
        )
    return True
```

---

## 3. Walk-Forward因果性保証

### 3.1 ウィンドウ分割設計

```
Window 1: |==Train 1==|=Val 1=|=Test 1=|
Window 2:        |==Train 2==|=Val 2=|=Test 2=|
Window 3:               |==Train 3==|=Val 3=|=Test 3=|
Window 4:                      |==Train 4==|=Val 4=|=Test 4=|

パラメータ:
- window_size: 全データの25% (各ウィンドウ)
- step_pct: 20% (ウィンドウ間のスライド)
- train/val/test比: 70/15/15
```

### 3.2 各ウィンドウでの処理手順

```
1. データ分割
   - train_df = df[train_start:train_end]
   - val_df = df[val_start:val_end]
   - test_df = df[test_start:test_end]

2. スケーラfit（Trainのみ）
   - scaler.fit(train_df)
   - val_scaled = scaler.transform(val_df)
   - test_scaled = scaler.transform(test_df)

3. 特徴量生成（各期間独立）
   - 各期間の特徴量は「その期間の開始前のデータ」のみから計算

4. モデル学習
   - model.learn(train_scaled, val_scaled)

5. 評価（分離Reporter）
   - val_reporter = BacktestReporter()
   - test_reporter = BacktestReporter()  # 別インスタンス
```

### 3.3 因果性チェックポイント

| チェックポイント | 確認内容 | 実装方法 |
|------------------|----------|----------|
| 分割境界 | オーバーラップなし | index範囲検証 |
| スケーラ状態 | Train終了時点で固定 | state snapshot |
| MTF境界 | 期間跨ぎのリークなし | 境界サンプル検査 |
| Reporter状態 | Val/Test完全分離 | インスタンス分離確認 |

---

## 4. Multi-Seed × Multi-Split検証設計

### 4.1 検証マトリクス

```
         | Split 1 | Split 2 | Split 3 | Split 4 |
---------|---------|---------|---------|---------|
Seed 42  |   ✓     |    ✓    |    ✓    |    ✓    |
Seed 123 |   ✓     |    ✓    |    ✓    |    ✓    |
Seed 456 |   ✓     |    ✓    |    ✓    |    ✓    |
Seed 777 |   ✓     |    ✓    |    ✓    |    ✓    |
---------|---------|---------|---------|---------|
Total: 16組み合わせ
```

### 4.2 Split定義（期間違い）

| Split | データ期間 | 市場特性 |
|-------|----------|----------|
| Split 1 | 2024/01-2024/06 | 上昇トレンド期 |
| Split 2 | 2024/07-2024/12 | ボラティリティ高 |
| Split 3 | 2025/01-2025/06 | レンジ相場期 |
| Split 4 | 2025/07-2025/12 | 直近期間 |

### 4.3 統計的検証基準

| 指標 | 基準 | 計算方法 |
|------|------|----------|
| 中央値ROI | > 5% | 16結果の中央値 |
| 標準偏差 | < 10% | seed間のばらつき |
| 最悪ケース | > -5% | 16結果の最小値 |
| 成功率 | > 75% | ROI > 0%の割合 |

### 4.4 運の良さ検出（Doc03対応: ロバスト版）

> **修正内容**: `std/mean`は平均ゼロ近傍で不安定なため、IQR/MADに置換

```python
import numpy as np
from scipy.stats import iqr, median_abs_deviation

def detect_luck_robust(results_matrix: np.ndarray) -> dict:
    """
    局所的な運の良さを検出（ロバスト版）
    
    Args:
        results_matrix: shape (n_seeds, n_splits) のROI行列
    
    Returns:
        dict: 判定結果と詳細
    """
    warnings = []
    
    # 1. Seed依存性検査（MAD使用）
    seed_medians = np.median(results_matrix, axis=1)  # seed別中央値
    seed_mad = median_abs_deviation(seed_medians)
    seed_median_of_medians = np.median(seed_medians)
    
    # MADが中央値の30%以上なら警告（絶対値で比較）
    if seed_mad > max(abs(seed_median_of_medians) * 0.3, 0.01):
        warnings.append(f"HIGH_SEED_DEPENDENCY: MAD={seed_mad:.4f}")
    
    # 2. Split依存性検査（IQR使用）
    split_medians = np.median(results_matrix, axis=0)  # split別中央値
    split_iqr = iqr(split_medians)
    split_median_of_medians = np.median(split_medians)
    
    # IQRが中央値の50%以上なら警告
    if split_iqr > max(abs(split_median_of_medians) * 0.5, 0.02):
        warnings.append(f"HIGH_PERIOD_DEPENDENCY: IQR={split_iqr:.4f}")
    
    # 3. 全体の安定性検査
    all_values = results_matrix.flatten()
    overall_mad = median_abs_deviation(all_values)
    overall_median = np.median(all_values)
    
    # 全体MADが中央値の50%以上なら警告
    if overall_mad > max(abs(overall_median) * 0.5, 0.03):
        warnings.append(f"HIGH_OVERALL_VARIANCE: MAD={overall_mad:.4f}")
    
    # 4. 最悪ケース検査
    worst_case = np.min(all_values)
    if worst_case < -0.05:  # -5%以下
        warnings.append(f"SEVERE_WORST_CASE: min={worst_case:.4f}")
    
    return {
        "status": "WARN" if warnings else "OK",
        "warnings": warnings,
        "metrics": {
            "seed_mad": seed_mad,
            "split_iqr": split_iqr,
            "overall_mad": overall_mad,
            "overall_median": overall_median,
            "worst_case": worst_case,
            "best_case": np.max(all_values)
        }
    }
```

> **削除**: 元の`detect_luck`にあった「対角線判定」はseed×split行列の構造上意味が薄いため削除

---

## 5. ベースライン比較の拡充

### 5.1 必須ベースライン（Doc00と整合）

> **注**: 最終判定基準はDoc00 Section 5.5を参照

| ベースライン | 実装 | 判定 | 統計検定 |
|-------------|------|------|----------|
| **Buy-and-Hold** | 期間開始で買い、終了で売り | **必須超過** | Mann-Whitney U |
| **SMA Crossover** | 20/50期間SMA | **必須超過** | Mann-Whitney U |
| **Random Action** | 同頻度ランダム売買 | **必須超過** | Mann-Whitney U |
| **Momentum** | 1時間リターン追従 | **参考**（判定外） | 任意 |

### 5.2 追加ベースライン（参考）

| ベースライン | 実装 | 目的 |
|-------------|------|------|
| **Mean Reversion** | ボリンジャーバンド逆張り | 戦略タイプ比較 |
| **Vol Target** | ボラティリティ調整ポジション | リスク調整比較 |
| **Always Flat** | 常にポジションなし | コスト下限確認 |

### 5.3 比較条件の統一（公平性保証）

> **重要**: RLとベースラインで条件を揃える

```yaml
baseline_conditions:
  # コスト統一
  fee_rate: 0.001          # 0.1%片道
  slippage_rate: 0.0005    # 0.05%推定
  
  # 資金統一
  initial_balance: 10000000  # 1000万円
  
  # ポジションサイズ統一（公平性の核）
  position_size_mode: "fixed"  # "fixed" or "risk_adjusted"
  fixed_position_size: 0.01    # 0.01 BTC
  
  # 評価期間統一
  evaluation_period: same_as_model
```

### 5.4 ポジションサイズの公平性規定（Doc03対応）

| 比較モード | 内容 | 適用場面 |
|-------------|------|----------|
| **固定サイズ** | RL/ベースラインともに0.01 BTC固定 | デフォルト（Go/No-Go判定） |
| **リスク調整** | 同一ボラターゲット（例: 2%日次リスク） | 追加検証（参考） |

```python
# ベースライン実行時のポジションサイズ統一
class BaselineRunner:
    def __init__(self, config):
        # RLと同一のポジションサイズを強制
        self.position_size = config.baseline_conditions.fixed_position_size
        # VPM/Allocatorは使用しない（公平性のため）
        self.use_dynamic_sizing = False
```

> **注意**: RL側でVPM/Allocatorを使用する場合は、
> ベースラインも同等のリスク制約を適用するか、
> 「固定サイズ比較」と「リスク調整比較」を別々に実施する。

### 5.4 判定基準

| 比較 | 判定 | 意味 |
|------|------|------|
| Model > BH + SMA + Random + Momentum | **必須合格** | 基本的な優位性 |
| Model > Mean Reversion | 推奨 | 戦略の独自性 |
| Model > Vol Target | 推奨 | リスク調整後の優位性 |

---

## 6. 実行モデル定義

### 6.1 コストパラメータ

| パラメータ | デフォルト値 | 根拠 |
|----------|-------------|------|
| **手数料** | 0.1% | Zaif Maker手数料 |
| **スリッページ** | 0.05% | 推定値（要実測） |
| **約定遅延** | 500ms | 推定値（要実測） |
| **約定率** | 95% | 指値注文想定（要実測） |

### 6.2 Paper Trading実測項目

> **注**: サンプル数は取引頻度に応じて調整

```yaml
paper_trading_metrics:
  # 必須実測
  actual_slippage:
    description: "実際の約定価格と指値の差"
    target: "< 0.1%"
    measurement: "min(1000取引, 1週間全取引)のサンプル"
  
  actual_latency:
    description: "注文発行から約定までの時間"
    target: "< 1000ms"
    measurement: "全取引の95パーセンタイル"
  
  fill_rate:
    description: "指値注文の約定率"
    target: "> 90%"
    measurement: "min(1000注文, 1週間全注文)のサンプル"
  
  # 低頻度戦略向け調整
  min_sample_period: "1週間"  # 取引数が少なくても最低1週間は評価
  min_trades_for_stats: 30    # 統計的に有意な最低取引数
  
  # 追加実測
  price_impact:
    description: "注文による価格変動"
    target: "< 0.05%"
    measurement: "大口注文時の確認"
```

### 6.3 バックテスト vs 実運用の乖離許容

| 指標 | 許容乖離 | 超過時のアクション |
|------|----------|-------------------|
| ROI | -30% | 戦略再検討 |
| Win Rate | -10pp | コスト見直し |
| Trade Count | ±50% | 頻度調整 |
| Sharpe | -0.5 | リスク調整 |

---

## 7. 実行ログ仕様

### 7.1 取引ログフォーマット

```json
{
  "timestamp": "2026-01-22T10:30:00.000Z",
  "trade_id": "uuid",
  "action": "buy|sell|close",
  "intended_price": 15000000,
  "actual_price": 15001500,
  "slippage": 0.01,
  "slippage_pct": 0.0001,
  "latency_ms": 450,
  "fee_paid": 1500,
  "position_before": 0.0,
  "position_after": 0.01,
  "balance_before": 10000000,
  "balance_after": 9848500,
  "model_confidence": 0.85,
  "entry_gate_decision": "allowed|blocked",
  "market_conditions": {
    "spread": 500,
    "volatility_1h": 0.015,
    "trend_signal": "bullish"
  }
}
```

### 7.2 セッションサマリー

```json
{
  "session_id": "v459_paper_001",
  "start_time": "2026-01-22T00:00:00Z",
  "end_time": "2026-01-22T23:59:59Z",
  "metrics": {
    "total_trades": 45,
    "winning_trades": 22,
    "losing_trades": 23,
    "net_pnl": 125000,
    "gross_pnl": 150000,
    "total_fees": 25000,
    "max_drawdown_pct": 2.5,
    "avg_slippage_pct": 0.045,
    "avg_latency_ms": 380,
    "fill_rate": 0.96
  },
  "alerts": [],
  "circuit_breaker_events": []
}
```

---

## 8. 検証完了チェックリスト

### Phase 0完了時

- [ ] リーク検査スクリプト実装完了
- [ ] スケーラfit範囲ルール文書化
- [ ] 実行モデルパラメータ確定
- [ ] ログフォーマット確定

### Phase 4完了時

- [ ] 全16組み合わせ（4seed × 4split）実行完了
- [ ] リーク検査全項目合格
- [ ] ベースライン4種全てで超過
- [ ] 運の良さ検出で「OK」判定

### Phase 5完了時

- [ ] Paper Trading 1週間データ蓄積
- [ ] 実測スリッページ < 0.1%
- [ ] 実測約定率 > 90%
- [ ] バックテスト乖離が許容範囲内

---

## 9. 関連ドキュメント

- [00_project_proposal_v459.md](00_project_proposal_v459.md) - プロジェクト提案書
- [01_review_and_gaps_v459.md](01_review_and_gaps_v459.md) - レビューと課題
- v458/19_phase5_6_final_review.md - v458最終レビュー
- v457/32_seed_stability_test.md - seed安定性テスト

---

**Status**: 📝 Planning  
**Author**: GitHub Copilot  
**Date**: 2026-01-22
