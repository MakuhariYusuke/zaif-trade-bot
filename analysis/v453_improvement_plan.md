# v453 Hybrid Strategy Analysis & Improvement Plan

## 現状分析 (v3 Fixed)

### パフォーマンス概要
- **Total Return**: 12.43%
- **Profit Factor**: 1.20
- **Win Rate**: 59.48%
- **Avg Win**: 44.50 / **Avg Loss**: -54.23

### レジーム別損益 (PnL)
| Regime | PnL | Filter Status | 評価 |
| :--- | :--- | :--- | :--- |
| **strong_bull_trend** | **+6,822** | Allowed | ◎ 収益の柱 |
| **consolidation** | **+6,051** | Allowed | ◎ 安定して利益 |
| **moderate_bull_trend** | +3,535 | Allowed | ◯ 良好 |
| **breakout_setup** | +2,640 | Allowed | ◯ 良好 |
| **weak_bull_trend** | +2,540 | Allowed | ◯ 良好 |
| **weak_bear_trend** | +2,015 | Allowed | ◯ 下落トレンドでも利益 |
| **extreme_volatility** | +1,883 | **Excluded** | △ フィルタ済みだがプラス（Force Exitの利確？） |
| **low_volatility_ranging** | +1,799 | Allowed | ◯ 良好 |
| **moderate_bear_trend** | -194 | Allowed | ✕ 微損、除外検討 |
| **strong_bear_trend** | -217 | **Excluded** | ✕ フィルタ済みだが損失（Force Exitの損切り？） |
| **breakdown_setup** | -408 | Allowed | ✕ **損失源**、除外すべき |
| **high_volatility_ranging** | **-1,615** | **Excluded** | ✕ **最大の損失源** |

### 課題と考察
1.  **`high_volatility_ranging` の損失**:
    - すでに除外対象ですが、最大の損失を出しています。
    - 考えられる原因:
        - レジーム判定の遅れにより、突入直後に大きな損失を被ってから Force Exit している。
        - または、このレジームに遷移する前の段階（例: `breakdown_setup`）でポジションを持ち越し、悪化してから損切りしている。
2.  **`breakdown_setup` の損失**:
    - 現在は許可されていますが、損失を出しています。
    - これを早期に除外することで、その後の `strong_bear_trend` や `high_volatility_ranging` への悪化による損失を未然に防げる可能性があります。
3.  **`moderate_bear_trend` の微損**:
    - 大きな損失ではありませんが、マイナスです。リスク回避のため除外候補です。

## 改善提案 (Optimization)

### 新設定: `hybrid_config_v3_optimized.json`
以下のレジームを追加で除外対象とします。

1.  **`breakdown_setup`**: 下落の予兆段階でポジションを解消し、深い下落に巻き込まれるのを防ぐ。
2.  **`moderate_bear_trend`**: 利益が出ていないため、安全側に倒して除外。

```json
"excluded_regimes": [
    "extreme_volatility",
    "high_volatility_ranging",
    "strong_bear_trend",
    "breakdown_setup",      // 追加
    "moderate_bear_trend"   // 追加
]
```

### 期待される効果
- `breakdown_setup` での早期撤退により、その後の大きな下落（`strong_bear_trend` 等）によるドローダウンを軽減。
- 無駄な損失（`moderate_bear_trend`）の削減による Profit Factor の向上。

## 次のアクション
- 作成した `config/v453/hybrid_config_v3_optimized.json` を使用してバックテストを実行し、効果を検証する。
