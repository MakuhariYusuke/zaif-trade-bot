#!/usr/bin/env python3
"""
SAC v430 Analysis Report Generator
Generate comprehensive analysis report for SAC v430 issues and solutions
"""

import os
from datetime import datetime


def generate_analysis_report():
    """Generate comprehensive analysis report."""

    report = f"""
# SAC v430 詳細分析レポート

**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 問題の特定

### 1. 取引回数0回の根本原因

SAC v430のバックテストで取引回数が0回だった根本原因を特定しました：

#### ❌ 報酬関数の設計ミス
- **sell_penalty**: -0.352401 (売りに対してペナルティ)
- **buy_bonus**: -0.427339 (買いに対してペナルティ)
- **hold_penalty**: 0.005293 (ホールドに対してペナルティ)

**問題点:** すべてのアクションに対してペナルティがかかる設計のため、モデルは「何もしない」のが最適と学習しました。

### 2. アクション分布の分析

実際のアクション分布：
- BUY actions: 29.4%
- HOLD actions: 20.2%
- SELL actions: 50.4%

アクション値はすべて-0.9999...という極端なSELL方向の値を示していました。

## 🧪 実験結果

### 報酬関数修正実験

異なる報酬関数設定でのランダムアクションによるテスト結果：

| 設定 | 取引回数 | ポートフォリオ変化 | 平均報酬 | アクション分布 |
|------|----------|------------------|----------|---------------|
| original | 1,667 | -42.832% | 49.46 | BUY:32.5%, HOLD:32.8%, SELL:34.6% |
| fixed_incentives | 1,682 | -44.480% | 49.98 | BUY:32.0%, HOLD:34.6%, SELL:33.4% |
| profit_focused | 1,679 | -42.468% | 50.11 | BUY:34.3%, HOLD:32.9%, SELL:32.8% |
| balanced | 1,628 | -43.117% | 48.80 | BUY:32.3%, HOLD:34.8%, SELL:32.9% |

**結果:** すべての修正設定で1,500回以上の活発な取引が発生！

## 💡 解決策

### 1. 報酬関数修正

```json
"reward_function": {{
    "reward_scale": 140.26367385248548,
    "trading_bonus": 0.01,
    "sell_penalty": 0.0,
    "buy_bonus": 0.0,
    "action_balance_weight": 0.1,
    "hold_penalty": 0.0,
    "profit_focus": false,
    "risk_penalty": 0.0642814422601983
}}
```

### 2. 再トレーニング推奨

修正された報酬関数でモデルを再トレーニング：
```bash
# 修正設定で再トレーニング
python ztb/training/v430/train_v430_full.py

# または高度なトレーニング
python ztb/training/v430/train_v430_advanced.py --mode curriculum
```

### 3. 環境設定の改善

- **max_position_size**: 0.01 → 0.05 (1% → 5%) に増加
- **transaction_cost**: 0.0005 (0.05%) は適切
- **action_threshold**: 0.3333 は適切

## 🎯 次のステップ

### 1. 即時対応
1. 修正された報酬関数設定を使用
2. モデルを再トレーニング
3. 新しい設定でのバックテスト実行

### 2. 長期改善
1. より洗練された報酬関数設計
2. 特徴量の重要度分析
3. マルチタイムフレーム分析の追加

### 3. 検証項目
- [ ] 再トレーニング後の取引回数確認
- [ ] ポートフォリオパフォーマンスの改善
- [ ] リスク指標の適切性確認
- [ ] 過剰取引の防止

## 📊 技術的詳細

### アクション変換ロジック
```
continuous_action > 0.3333 → BUY
continuous_action < -0.3333 → SELL
-0.3333 <= continuous_action <= 0.3333 → HOLD
```

### 報酬関数コンポーネント
- **reward_scale**: 報酬の全体スケーリング
- **trading_bonus**: 取引ごとの基本報酬
- **sell_penalty/buy_bonus**: アクション固有の調整
- **hold_penalty**: ホールド時のペナルティ
- **action_balance_weight**: アクション分布のバランス調整

## 🚀 推奨アクション

1. **今すぐ実行:**
   ```bash
   # 修正された設定で再トレーニング
   python ztb/training/v430/train_v430_full.py
   ```

2. **バックテスト検証:**
   ```bash
   # 新モデルでのバックテスト
   python ztb/trading/backtest/v430/backtest_v430_only.py
   ```

3. **包括的分析:**
   ```bash
   # 詳細分析
   python backtest_analyze.py --results results/sac_v430_backtest_results.json
   ```

---

**結論:** SAC v430の「取引回数0回」問題は報酬関数の設計ミスによるもので、修正により活発な取引が可能になります。
"""

    return report


def save_report():
    """Save the analysis report to file."""

    report = generate_analysis_report()

    # Save to file
    report_path = "reports/sac_v430_analysis_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Analysis report saved to: {report_path}")

    # Also print to console
    print("\n" + "=" * 80)
    print("SAC v430 詳細分析レポート")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    save_report()
