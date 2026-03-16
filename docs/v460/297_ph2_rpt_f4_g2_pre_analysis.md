# 297# F-4 / G-2 事前調査レポート

> **文書番号**: 297#
> **種別**: `analysis` (事前調査)
> **作成日**: 2026-03-06
> **目的**: F-4 (StatisticalValidator A/B統合) および G-2 (168# P3残タスク) の現状把握・工数見積・リスク評価

---

## Part 1: F-4 — StatisticalValidator A/B test integration

### §1.1 現状の統計検定スタック

#### ab_judgment.py (715L) — 現行A/B判定エンジン

| 機能 | 実装場所 | 詳細 |
|------|---------|------|
| **Welch's t検定** | `_compute_statistical_comparison()` L232-253 | scipy.stats.ttest_ind (equal_var=False) |
| **Cohen's d** | `_cohen_d()` L218-226 | pooled std 版、自前実装 |
| **3指標判定** | `evaluate_ab_variant()` L263-458 | fill_rate + avg_pnl30 + downside_p10 |
| **Regime別A/B** | `evaluate_per_regime()` L470-545 | regime単位で3指標判定を個別実行 |
| **Trending Down Sell評価** | `evaluate_trending_down_sell()` L600-715 | カウンターファクチュアル比較付き |
| **ABTestAnalyzer fallback** | L237-253 | scipy失敗時に ztb.adaptation.ab_test.analyzer へ委譲 |

**ABJudgmentResult** には `pnl30_p_value` と `pnl30_effect_size` (Cohen's d) が格納されるが、**多重検定補正なし**、**信頼区間なし**、**Bootstrap CIなし**。

#### ztb/adaptation/ab_test/analyzer.py (約170L) — SAC時代のA/B分析

| 機能 | 詳細 |
|------|------|
| `analyze_parallel()` | ThreadPoolExecutor並列でt検定+Cohen's d+簡易CI |
| `calculate_bootstrap_ci()` | Bootstrap法CI (n=1000) — **v460未使用** |
| `_calculate_confidence_interval()` | 正規近似CI (SE ≈ 1/√(n/2)) |
| `analyze_comparison()` | 仮想データ生成→比較 (実質デッドコード) |

**問題**: `analyze_comparison()` は `np.random.normal` でデータを生成しており、実データ比較にならない。

#### scripts/v459/gate_c3_comparison.py (451L) — scipy-free 検定実装

| 機能 | 詳細 |
|------|------|
| **Mann-Whitney U** | O(n×m) 全ペア比較、正規近似p値 |
| **Cliff's Delta** | 効果量 (negligible/small/medium/large) |
| **Holm-Bonferroni** | 多重比較補正 (純Python実装) |
| `run_comparison()` | SAC vs ベースライン統計比較パイプライン |

**価値**: scipy不要の軽量実装。fill_testの軽量統計検定に理想的。

#### ztb/metrics/statistical_validator.py (492L) — v445 SAC用

| 機能 | 詳細 | 再利用価値 |
|------|------|-----------|
| `validate_performance_metrics()` | Sharpe CI (Bootstrap)、安定性評価 | ★ (SAC専用設計) |
| `validate_multiple_strategies()` | ANOVA + Tukey HSD事後検定 | ★★ (regime間比較に流用可) |
| `validate_signal_quality()` | 予測vs実績の相関分析 | ★★ (SkipGateモデル検証) |
| `_apply_multiple_testing_correction()` | statsmodels.multipletests (Bonferroni等) | ★★★ (A/B判定に直接追加可) |
| `_calculate_sharpe_confidence_interval()` | Bootstrap (n=10000) | ★ |
| `_assess_metric_stability()` | ローリング統計安定性 | ★ |
| `_analyze_signal_thresholds()` | 閾値別t検定 | ★★ |
| `_analyze_prediction_accuracy()` | 方向予測精度 + MSE/MAE | ★ |

**依存**: scipy, statsmodels, ztb.metrics.sharpe_ratio, ztb.metrics.metrics (kurtosis, skewness等)

#### side_regime_dashboard.py — 統計分析なし

`ab_result.pnl30_p_value` を表示するのみ。独自の統計計算はなし。ab_judgment.pyに完全委譲。

### §1.2 StatisticalValidator で追加できる価値

| 現状の欠落 | StatisticalValidator由来の解決策 | 優先度 |
|-----------|-------------------------------|--------|
| **多重検定補正なし** (regime別A/Bで4-5回検定) | `_apply_multiple_testing_correction()` (statsmodels) | **P0** |
| **Cohen's d のCI未算出** | `calculate_bootstrap_ci()` (ab_test/analyzer.py) | P1 |
| **ノンパラメトリック検定なし** | gate_c3_comparison.py の Mann-Whitney + Cliff's Delta | **P0** |
| **安定性分析なし** | `_assess_metric_stability()` / `_analyze_performance_stability()` | P2 |
| **シグナル品質検証なし** | `validate_signal_quality()` — SkipGateモデル予測精度監視 | P2 |

### §1.3 推奨統合方針

**StatisticalValidator をそのまま統合するのは不適切。** 理由:

1. **SAC/リターン系列前提**: `validate_performance_metrics()` は `returns: list[float]` を入力とし、Sharpe ratio中心の設計。fill_testは `FillRecord` ベースの約定品質評価が対象
2. **重量級依存**: `statsmodels` が必要。fill_testは軽量依存を好む
3. **God object**: 492Lで責務が広すぎる（Bootstrap CI、ANOVA、シグナル品質が混在）

**推奨: 必要な関数のみ cherry-pick して `ab_judgment.py` に統合**

| 統合対象 | 方法 | 工数 |
|---------|------|------|
| Holm-Bonferroni | gate_c3_comparison.py の純Python版を `ab_judgment.py` に移植 | 0.2日 |
| Mann-Whitney U + Cliff's Delta | 同上 | 0.2日 |
| Bootstrap CI for Cohen's d | ab_test/analyzer.py の `calculate_bootstrap_ci()` を参考に軽量版作成 | 0.3日 |
| `evaluate_per_regime()` に多重検定補正追加 | Holm-Bonferroni を結果に適用 | 0.1日 |
| **合計** | | **0.8日** |

### §1.4 リスク・安全性評価

- **ライブ影響**: **なし**。ab_judgment.pyは分析専用スクリプト層。fill_loop本体に影響しない
- **回帰リスク**: 低。既存の `evaluate_ab_variant()` の戻り値 `ABJudgmentResult` に新フィールド追加するだけ
- **テスト**: ab_judgment の既存テスト (`tests/v460/test_ab_judgment.py`) が存在するはず
- **実装可能性**: **即時実装可能**

---

## Part 2: G-2 — 168# P3 残タスク

### §2.1 P3-4: UnifiedTrainer god object

**現状**:

| ファイル | 行数 |
|---------|------|
| `ztb/training/unified_trainer/trainer.py` | **2,227L** (168#時点: 2,835L → 22%削減済) |
| `ztb/training/unified_trainer/reporting.py` | 795L |
| `ztb/training/unified_trainer/ensemble_system.py` | 641L |
| `ztb/training/unified_trainer/ensemble_mixin.py` | 267L |
| `ztb/training/unified_trainer/config_manager.py` | 239L |
| `ztb/training/unified_trainer/ui.py` | 236L |
| `ztb/training/unified_trainer/parallel_trainer.py` | 153L |
| `ztb/training/unified_trainer/main.py` | 162L |
| **合計** | **4,860L** (ディレクトリ全体) |

**評価**: 168# 時点の2,835L → 2,227L に削減。reporting.py (795L)、ensemble_system.py (641L)、config_manager.py (239L) が既に分離済。ただし trainer.py 自体はまだ2,227Lで god object のまま。

| 分割候補 | 推定抽出サイズ | 価値 |
|---------|--------------|------|
| 訓練ループ本体 | ~600L | ★★ |
| 評価/検証ロジック | ~400L | ★★ |
| モデル保存/読込 | ~300L | ★ |
| ログ/進捗管理 | ~200L | ★ |

**工数見積**: 2-3日 (リファクタ + テスト修正)
**リスク**: 高。trainer.py は SAC 訓練パイプラインのコア。分割ミスで訓練が壊れる
**ライブ影響**: **なし**。fill_test / fill_loop には一切影響しない (SAC 訓練専用)
**推奨**: **v461 繰越維持**。収益直結しない。168# の評価通り

### §2.2 P3-5: sell pnl120→pnl30 モデル統一

**現状**:

- `ztb/metrics/fill_quality.py` で `pnl120` を計算・記録 (L136, L476, L574, L604)
- `ev_weighted_pnl = 0.4*pnl30 + 0.6*pnl120` の定義あり (L136)
- 168# §1.3 の発見: PnL30は負 (−0.212bps) だが PnL120は正 (+0.237bps)
- **モデル自体は現存しない** — 289# で `ev_weighted_pnl` が同語反復であることが判明し、291# Gemini が保持期間延長を P1 推奨

**評価**: "モデル統一" というよりは「保持期間パラメータの変更」が本質。sell_pnl_measurer の wait を 30s→90s に変更する施策と結合する問題。これは既に別線で検討されている (168# §4.1 施策#1)。

**工数**: 0.3日 (設定変更 + 計測基盤調整)
**リスク**: 中。保持期間変更はPnL計測の基準が変わるため、A/B判定の横比較ができなくなる
**ライブ影響**: あり。pnl_measurer の wait 変更は実取引の保持時間に直結
**推奨**: **A/Bテストとして段階実施**。fill_test の設定で variant として 90s を試す

### §2.3 P3-6: asyncio.to_thread 残5メソッド

**現状の `asyncio.to_thread` 使用箇所**:

| ファイル | 使用数 | 状態 |
|---------|-------|------|
| `ztb/trading/live/exchanges/coincheck/adapter.py` | 9箇所 | ✅ 対応済 |
| `ztb/trading/live/exchanges/bitflyer/adapter.py` | 4箇所 | ✅ 対応済 |
| `ztb/ops/health/check_venue_health.py` | 1箇所 | ✅ 対応済 |

**残存する同期ブロッキング呼び出し**:

| ファイル | 行 | 内容 | 問題度 |
|---------|-----|------|--------|
| `ztb/trading/live_trader/live_trader.py` L295 | `requests.get("https://coincheck.com/api/ticker")` | ★ 低 (初期化時のみ、async context外) |

**評価**: 168# が言及した「残5メソッド」は、その後の 003# (#9) で coincheck/bitflyer adapter が対応済み。live_trader.py L295 の `requests.get` は初期化時のdry-run価格取得で、async loop外での呼び出しのため `to_thread` は不要。

**fill_loop (v460) 側**: `ztb/fill/` 配下に同期ブロッキング呼び出しは発見されなかった。

**工数**: 0日 (実質完了)
**リスク**: なし
**ライブ影響**: なし
**推奨**: **完了扱い (Done)**

### §2.4 CircuitBreaker 統合状態

**現状**:

| 場所 | 状態 |
|------|------|
| `ztb/utils/circuit_breaker.py` (243L) | ✅ 存在、async対応済 |
| `ztb/trading/live_trader/live_trader.py` L505-511 | ✅ **統合済** (`api_circuit_breaker` として使用) |
| `ztb/utils/health_monitor.py` L17 | ✅ import して使用 |
| `ztb/trading/risk/backtest_risk_manager.py` | ✅ 独自circuit_breaker実装あり |
| `ztb/risk/circuit_breakers.py` | ✅ KillSwitch (paper_trader で使用) |
| **ztb/fill/** (fill_loop) | ❌ **未統合** |

**評価**: live_trader (v459 系) では統合済だが、**v460 fill_loop/fill_test には未統合**。168# §3.1 が推奨した「API障害時の自動遮断→無駄サイクル削減」はfill_loop側で実装されていない。

**工数**: 0.5日 (fill_loop_orchestrator に CircuitBreaker wrapping を追加)
**リスク**: 低。CircuitBreaker 自体は成熟。wrapping 追加のみ
**ライブ影響**: あり (安全方向) — API障害時にサイクルを自動停止するため、むしろ安全性向上
**推奨**: **実装可能 (P1)**。ただし fill_test での十分な検証が必要

### §2.5 DrawdownController 統合状態

**現状**:

| 場所 | 状態 |
|------|------|
| `ztb/risk/drawdown_controller.py` (254L) | ✅ 存在 |
| `ztb/risk/risk_manager.py` L11,34 | ✅ RiskManager 内部で使用 |
| `ztb/trading/environment/components/position_manager.py` L707 | コメントのみ |
| **ztb/fill/** (fill_loop) | ❌ **未統合** |

**評価**: DrawdownController は SAC 訓練環境の RiskManager 内部でのみ使用。v460 fill_loop には**未統合**。168# §4.1 施策#3 で「日次worst制限 (例: −50bps/日)」が推奨されたが未実施。

**工数**: 0.5日 (日次PnL集計 + 閾値判定 + fill_cycle停止)
**リスク**: 中。ドローダウン制限の閾値設定次第で正常取引を止める可能性
**ライブ影響**: あり (安全方向) — 大損日の被害軽減
**推奨**: **実装可能 (P1)**。閾値は保守的に設定し、最初は警告のみから開始

---

## Part 3: 総合優先度マトリクス

| タスク | 工数 | 収益直結度 | ライブリスク | 推奨時期 |
|-------|------|-----------|-------------|---------|
| **F-4: Holm-Bonferroni + Mann-Whitney → ab_judgment** | 0.4日 | ★★ (判定精度向上) | なし | **即時** |
| **F-4: Bootstrap CI for Cohen's d** | 0.3日 | ★ (報告品質) | なし | 次バッチ |
| **G-2 P3-6: asyncio.to_thread** | 0日 | - | - | **Done** |
| **G-2 CircuitBreaker → fill_loop** | 0.5日 | ★★ (障害耐性) | 安全方向 | **即時** |
| **G-2 DrawdownController → fill_loop** | 0.5日 | ★★★ (損失抑制) | 安全方向 | **即時** |
| **G-2 P3-5: 保持期間延長** | 0.3日 | ★★★ (PnL改善) | A/Bテスト必要 | 別線で検討中 |
| **G-2 P3-4: UnifiedTrainer分割** | 2-3日 | なし | なし | **v461繰越** |

### 即時実行推奨（本バッチ）
1. **F-4 Holm + Mann-Whitney** + **CircuitBreaker fill_loop統合** + **DrawdownController fill_loop統合**
2. 合計工数: ~1.4日
3. すべて安全方向の改善。ライブ取引への悪影響リスクなし

### 繰越推奨
- P3-4 (UnifiedTrainer分割): 収益直結しない。v461
- P3-5 (保持期間): 別線の A/B テスト施策として管理
