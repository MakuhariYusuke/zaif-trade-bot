# v453 Hybrid Strategy Fix Verification

## 概要
v453 Hybrid Strategy (v3 Config) のバックテストにおいて、フィルタが正しく機能していない問題（v2bと結果が同一）と、ログスパムの問題を修正し、再検証を行いました。

## 修正内容
1. **Regime Filter Fix**:
   - `HeavyTradingEnv` から `PositionManager` へ `market_regime` を渡す際、Enumオブジェクトそのものではなく `.value` (文字列) を渡すように修正。
   - これにより `hybrid_config_v3.json` の文字列ベースのフィルタ条件が正しくマッチするようになりました。

2. **Log Spam Fix**:
   - `V444RegimeClassifier` 内の `print("DEBUG TREND: ...")` を `logger.debug` に変更し、コンソール出力を抑制しました。

## 検証結果 (v2b vs v3 Fixed)

### 動作の変化
`analysis/compare_v453_runs.py` による比較結果:

| Metric | v2b (Baseline/Broken) | v3 (Fixed) | 変化 |
| :--- | :--- | :--- | :--- |
| **Filter Active Count** | 7444 | **10822** | **+45%** (フィルタが正しく作動) |
| **Effective Actions** | 94 | **284** | **+202%** (Force Exitによる決済増と推測) |
| **Blocked Entries** | 3880 | 3690 | -5% |

### パフォーマンス (v3 Fixed)
- **Total Return**: 12.43%
- **Profit Factor**: 1.2049
- **Max Drawdown**: -2.40%
- **Win Rate**: 18.26%
- **Sharpe Ratio**: 18.02

### 考察
- `filter_active` の大幅な増加は、`extreme_volatility`, `high_volatility_ranging`, `strong_bear_trend` などの除外対象レジームで正しくフィルタが発動していることを示しています。
- `effective_nonhold` (アクション) の増加は、`force_exit: true` 設定により、除外レジーム突入時に即座にポジション解消（決済）が行われたためと考えられます。
- 結果として、危険なレジームでの保有を回避し、ドローダウンを抑制しつつ利益を確保する挙動が確認できました。

## 次のステップ
- この修正版 v3 設定をベースに、さらなるパラメータ調整や、他の期間での検証を推奨します。
