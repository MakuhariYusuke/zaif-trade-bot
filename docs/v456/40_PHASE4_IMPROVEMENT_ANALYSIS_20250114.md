# Phase 4 Improvement Analysis (2025-01-14)

## 1. エグゼクティブサマリー（改善優先順位 Top 3）
1. 評価指標の算出が実データではなく擬似値になっており、Go/No-Go判定を歪める恐れが高い。`ztb/evaluation/walk_forward/evaluator.py:154` `ztb/evaluation/walk_forward/evaluator.py:178` `ztb/evaluation/walk_forward/evaluator.py:179` `ztb/evaluation/walk_forward/evaluator.py:180`
2. 過学習指標の閾値と計算仕様がドキュメント・Go/No-Go基準と不整合（0.8/1.0/1.2 vs <0.3）で、val=0時の扱いも過小評価。`ztb/analysis/evaluation/walk_forward_adapter.py:66` `ztb/analysis/evaluation/walk_forward_adapter.py:169` `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:182`
3. 分割・評価の堅牢性不足（`TimeSeriesWindow`検証が無効、`step_pct`で無限ループ、embargo未実装、sys.path注入依存）により長時間運用の信頼性が低い。`ztb/evaluation/walk_forward/types.py:11` `ztb/evaluation/walk_forward/splitter.py:123` `ztb/evaluation/walk_forward/splitter.py:67` `ztb/evaluation/walk_forward/evaluator.py:54`

## 2. 詳細分析（セクション1-7）
### 2.1 コード品質・パフォーマンス
#### 優先度: HIGH
- **問題**: Sharpe/MaxDrawdown/WinRateが実績リターンではなく簡易値で計算され、評価指標が無効化。
- **影響範囲**: WalkForwardModelEvaluator → WalkForwardUnifiedEvaluator → Pipelineの推奨生成。
- **現在の状態**: 行動列をSharpe計算に使用し、max_drawdownは最終残高1点のみ、win_rateは固定値。`ztb/evaluation/walk_forward/evaluator.py:154` `ztb/evaluation/walk_forward/evaluator.py:178` `ztb/evaluation/walk_forward/evaluator.py:179` `ztb/evaluation/walk_forward/evaluator.py:180`
- **推奨改善**: **既存メトリクス実装を活用すること**。`ztb.metrics.metrics` (sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, win_rate, calculate_feature_metrics等)をimportし、エクイティカーブ（残高推移）やトレードPnLから指標を計算。評価環境からステップごとの残高/リターンを収集し、既存関数に渡す。
- **実装例**:
```python
from ztb.metrics.metrics import (
    sharpe_ratio as calculate_sharpe_ratio,
    max_drawdown as calculate_max_drawdown,
    win_rate as calculate_win_rate,
)

balances = []
trade_pnls = []
while not done:
    ...
    balances.append(eval_env.balance)
    if info.get("trade_pnl") is not None:
        trade_pnls.append(info["trade_pnl"])

# 既存実装を活用
returns = np.diff(balances) / np.maximum(balances[:-1], 1e-12)
sharpe = calculate_sharpe_ratio(returns)  # 既存実装
max_dd = calculate_max_drawdown(np.array(balances))  # 既存実装
win_rate = calculate_win_rate(np.array(trade_pnls)) if trade_pnls else 0.0  # 既存実装
```

#### 優先度: MEDIUM
- **問題**: 大規模データでのメモリ負荷が高い（DataFrameの分割コピー、actions全保持、モデル全保持）。
- **影響範囲**: `train_and_evaluate_window`の長時間実行、100+ウィンドウ時のRAM圧迫。
- **現在の状態**: `df.iloc`で3分割し、`actions`を全保持。`self.models`に全モデルを常時保持。`ztb/evaluation/walk_forward/evaluator.py:32` `ztb/evaluation/walk_forward/evaluator.py:66` `ztb/evaluation/walk_forward/evaluator.py:154`
- **推奨改善**: 環境側に開始/終了インデックスを渡す設計に変更。メトリクス計算については`ztb.metrics.metrics`のオンライン集計対応関数を活用（既存実装で対応可能）。不要なモデルは逐次保存して解放。

**質問への回答**: `df.iloc`による分割は大規模データで新規オブジェクトを大量生成します。環境に「元DataFrame＋インデックス範囲」を渡し、スライスを環境側で参照（必要なら`np.ndarray`ビュー）する方がメモリ効率が高いです。また、Sharpe計算については既存の`ztb.metrics.metrics.sharpe_ratio`を使用すれば、カスタム実装の維持負荷も減少します。

### 2.2 エラーハンドリング・堅牢性
#### 優先度: HIGH
- **問題**: 学習失敗時の回復・部分結果保存がなく、1ウィンドウの失敗で全体停止。
- **影響範囲**: 長時間実行時の信頼性、途中結果の活用不可。
- **現在の状態**: 例外処理なしで学習・評価を実行。`ztb/evaluation/walk_forward/evaluator.py:35` `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:62`
- **推奨改善**: ウィンドウ単位でtry/exceptし、失敗ウィンドウをスキップ・記録。部分結果を保存できる設計へ。
- **実装例**:
```python
performances = []
failed = []
for w in windows:
    try:
        performances.append(evaluator.train_and_evaluate_window(df, w, timesteps))
    except Exception as exc:
        logger.exception("Window %s failed", w.window_id)
        failed.append({"window_id": w.window_id, "error": str(exc)})
```

#### 優先度: MEDIUM
- **問題**: `step_pct`が小さい場合に`step_size=0`となり無限ループ。
- **影響範囲**: 分割処理が停止し、CPU占有。
- **現在の状態**: `int(n * step_pct)`で0になっても検証なし。`ztb/evaluation/walk_forward/splitter.py:123` `ztb/evaluation/walk_forward/splitter.py:167`
- **推奨改善**: `step_size <= 0`を明示的にエラー化し、`pct`合計の検証も追加。

#### 優先度: MEDIUM
- **問題**: `TimeSeriesWindow`のバリデーションが実行されない。
- **影響範囲**: 不正なウィンドウで評価が進行し、誤った結果を生む。
- **現在の状態**: `NamedTuple`に`__post_init__`を定義しても呼ばれない。`ztb/evaluation/walk_forward/types.py:11` `ztb/evaluation/walk_forward/types.py:43`
- **推奨改善**: `@dataclass(frozen=True)`へ変更し検証を有効化、または`__new__`で検証。

#### 優先度: LOW
- **問題**: `embargo_days`が未使用でリーク防止が未実装。
- **影響範囲**: 時系列リーク検出精度が低下。
- **現在の状態**: 変数保持のみ。`ztb/evaluation/walk_forward/splitter.py:67`
- **推奨改善**: 日付インデックスを用いて`val_end`と`test_start`間にembargoを挿入。

**質問への回答**: 現在はウィンドウ失敗時の例外処理がなく、SAC学習が失敗すると全パイプラインが停止します。部分結果の活用は設計上未対応です。

### 2.3 アーキテクチャ・設計原則
#### 優先度: HIGH
- **問題**: `scripts/v456`への`sys.path`注入と直接依存で、本番・再利用性・テスト性が低い。
- **影響範囲**: デプロイ環境差異、依存関係の隠蔽、バージョン衝突。
- **現在の状態**: `sys.path`へ動的追加し、`train_and_evaluate_v456_phase3`を直接 import。`ztb/evaluation/walk_forward/evaluator.py:53` `ztb/evaluation/walk_forward/evaluator.py:141`
- **推奨改善**: 環境生成を`EnvFactory`プロトコルとして注入し、評価戦略を切り替え可能に。
- **実装例**:
```python
class EnvFactory(Protocol):
    def __call__(self, df: pd.DataFrame) -> gym.Env: ...

class WalkForwardModelEvaluator:
    def __init__(self, env_factory: EnvFactory, algo_factory: AlgoFactory) -> None:
        self.env_factory = env_factory
        self.algo_factory = algo_factory
```

#### 優先度: MEDIUM
- **問題**: `WalkForwardResult`の二重定義で型が分裂。
- **影響範囲**: import先により型が分岐し、mypy/実装の混乱。
- **現在の状態**: `ztb/evaluation/walk_forward/types.py:154` と `ztb/evaluation/walk_forward/result.py:11` が重複。
- **推奨改善**: 片方に統一し、`__init__`の公開インターフェースも統一。

#### 優先度: MEDIUM
- **問題**: 設定値が複数箇所でハードコード。
- **影響範囲**: 実験条件の再現性/比較が困難。
- **現在の状態**: `initial_train_pct`等の値を直接指定。`ztb/evaluation/walk_forward/splitter.py:61`
- **推奨改善**: dataclass設定 or YAML/JSONに集約し、実験メタデータに保存。

**質問への回答**: 現状の直接依存は回避可能です。環境生成・アルゴリズム作成を注入し、`Strategy`（SAC/PPO/DQNなど）を差し替えられる設計が望ましいです。

### 2.4 テスト戦略・カバレッジ
#### 優先度: MEDIUM
- **問題**: Splitter/Evaluator/Reporterのユニットテストが存在しない。
- **影響範囲**: 分割バグや評価指標の誤りが検知されない。
- **現在の状態**: テストはAdapter/Pipeline中心で、mock windowのみ。`tests/unit/evaluation/test_walk_forward_adapter.py:24` `tests/unit/evaluation/test_walk_forward_integration_pipeline.py:9`
- **推奨改善**: `splitter`の境界値（step=0, pct合計>1, embargo）と`evaluator`の指標算出をテスト追加。

#### 優先度: LOW
- **問題**: 実データ・大規模データでの統合テストが未計画。
- **推奨改善**: `pytest -m slow`で実データ50K-100Kの小規模サブセット検証を追加。

**質問への回答**: 現状の13テストはすべてサンプルWindowPerformanceを利用するユニットテストで、100K+ timestepsの統合テストは確認できません。

### 2.5 ドキュメント・説明の正確性
#### 優先度: MEDIUM
- **問題**: ドキュメントと実装が一致していない箇所が複数。
- **影響範囲**: 誤った設計理解・誤運用。
- **現在の状態**:
  - ファイル名が異なる（`walk_forward_splitter.py` などの記載に対し実装は `ztb/evaluation/walk_forward/splitter.py` 等）。
  - `save_evaluation(format="yaml"|"pickle")` は未実装。`ztb/analysis/evaluation/walk_forward_integration_pipeline.py:91`
  - `overfitting_indicator`の定義説明が誤り。`ztb/analysis/evaluation/walk_forward_adapter.py:40`

**質問への回答**: `overfitting_ratio` は `val_roi==0` の場合に 0.0 を返す実装で、負の値は `abs` で処理しています。`0` 近傍の扱いが未定義のため、閾値判定が過小評価になります。`ztb/evaluation/walk_forward/types.py:128` `ztb/analysis/evaluation/walk_forward_adapter.py:169`

### 2.6 本番環境対応
#### 優先度: HIGH
- **問題**: 長時間実行の進捗保存・再開機能が未実装。
- **影響範囲**: 数日実行で障害が起きると全損。
- **現在の状態**: Checkpoint/Resume設計なし、途中結果も保存されない。`ztb/evaluation/walk_forward/evaluator.py:91`
- **推奨改善**: SB3の`CheckpointCallback`でウィンドウごとに保存し、再開時にロードできる仕組みを追加。

#### 優先度: MEDIUM
- **問題**: ロギング/モニタリングが最低限（stdout・簡易ログ）。
- **推奨改善**: `logging`の構造化、TensorBoard/MLflow連携、メトリクスとメモリ/時間を記録。

**質問への回答**: 100ウィンドウ×100Kタイムステップを想定した進捗保存・再開機能はコード上確認できません。

### 2.7 パフォーマンス最適化
#### 最適化の機会
- **対象**: SAC学習ループと環境ステップ
- **現在**: 単一環境で逐次学習。メトリクス計算は独自実装。
- **改善案**: Vectorized env/並列実行、特徴量キャッシュ、GPU利用、エピソード評価のバッチ化。メトリクス計算は既存の`ztb.metrics.metrics`の最適化されたルーチンに一本化。
- **期待効果**: 20-50%の実行時間短縮（データサイズと環境実装に依存）。メトリクス計算は5-10%の改善見込み。

**質問への回答**: 現状のコードから見る限り最大ボトルネックはSAC学習と環境ステップです。メトリクス計算やJSON出力は軽量で、I/Oはボトルネックになりにくいです。ただし、メトリクス計算を既存の`ztb.metrics.metrics`に統一することで、実装の保守性と信頼性は大幅に向上します。

### 追加の気づき（本文章外）
- **メトリクス実装の統一化が最重要**：`WalkForwardUnifiedEvaluator`と`WalkForwardEvaluationPipeline`の計算値は、`ztb.metrics.metrics`の公式実装と一致させるべき。`ztb/analysis/evaluation/walk_forward_adapter.py:276`で計算されるsharpe_ratioは、`ztb.metrics.metrics.sharpe_ratio`と完全に一致するか検証が必要。
- `overfitting_ratio`閾値がGo/No-Go基準（<0.3）と乖離。`ztb/analysis/evaluation/walk_forward_adapter.py:66` `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:182`
- `UnifiedEvaluator`がダミー指標を返すため、本番で誤用されると評価結果が虚偽になる。`ztb/analysis/evaluation/unified_evaluation.py:124`
- **推奨**：全メトリクス計算を`ztb.metrics.metrics`に集約し、カスタム実装を排除することで、テスト・検証・保守の負担を大幅削減。

## 3. 実装チェックリスト（難易度・工数）
|項目|難易度|工数目安|理由|既存実装の活用|
|---|---|---|---|---|
|メトリクス計算の既存実装への統一（Sharpe/MaxDD/WinRate等）|S|0.5-1日|既存の`ztb.metrics.metrics`をimportして置き換え。カスタム実装排除。|✅ ztb.metrics.metrics|
|overfitting指標の定義統一と閾値調整|M|0.5-1日|Go/No-Go基準との整合・ドキュメント更新|既存thresholding検討|
|Window分割の検証強化（step_size, pct合計, embargo）|M|0.5-1日|境界条件とリーク防止の追加実装|N/A|
|`TimeSeriesWindow`のバリデーション有効化|S|0.5日|dataclass化/`__new__`実装|N/A|
|環境/アルゴリズムの依存注入（strategy化）|L|2-4日|設計見直しとテスト更新が必要|N/A|
|ウィンドウ単位の例外処理＋部分結果保存|M|1日|パイプラインの制御追加|N/A|
|Checkpoint/Resume + ログ拡充|M|1-2日|SB3 callback + JSON/DB保存|既存logging module活用|
|Splitter/Evaluator/Reporterのユニットテスト追加|M|1-2日|失敗ケースや境界値の作成|pytest, fixtures活用|
|ドキュメント整合性修正（ファイル名/format説明）|S|0.5日|記述整理のみ|N/A|

## 4. Next Steps（推奨実施順序）
1. **メトリクス計算の既存実装への統一（最優先）**
   - `ztb.metrics.metrics` (sharpe_ratio, max_drawdown, win_rate, sortino_ratio, calmar_ratio)
   - `ztb.metrics.metrics` (calculate_feature_metrics等)
   - をWalkForwardModelEvaluatorへimportし、カスタム実装を全削除。
   - 工数：0.5-1日

2. 過学習定義の統一とGo/No-Go基準の整合。

3. Splitter/Windowの検証強化とembargo導入、例外処理・部分結果保存を実装。

4. 環境/アルゴリズムの依存注入と設定の一元管理。

5. Checkpoint/Resume・ロギング拡充で長時間運用に耐える体制へ。

6. テスト追加（境界値＋小規模実データの統合テスト）とドキュメント更新。
