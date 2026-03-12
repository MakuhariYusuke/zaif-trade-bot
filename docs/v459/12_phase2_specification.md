# v459 Phase 2: P1バグ修正 仕様書 (12)

**Date**: 2026-01-22  
**Status**: 📋 Planning  
**Phase**: Phase 2 - P1 Bug Fixes & AB Testing Enablement  
**Predecessor**: Phase 1 (P0 Bug Fixes) - Completed

---

## 1. Executive Summary

### 1.1 Phase 1完了サマリー

v459 Phase 1（P0バグ修正）を完了し、以下の基盤を確立しました：

**成果**:
- ✅ P0バグ全修正（4/4件）
- ✅ テスト103/103パス（100%）（Phase 0: 77件、Phase 1追加: 26件）
- ✅ Phase 2テスト: 16/16パス（Phase 2新規追加）
- ✅ 累積テスト: 119/119パス（Phase 0: 77件、Phase 1: 26件、Phase 2: 16件）
- ✅ PnL規約統一（env=NET, reporter=検証）
- ✅ Val/Test分離保証（環境独立性）

**確立された基盤**:
1. **コスト規約の明確化**: `trade_pnl`はNET PnL（コスト控除済み）、Reporterは二重控除しない
2. **環境分離の保証**: Val/Test評価で独立した環境・scaler・reporterを使用
3. **テスト基盤**: P0修正の検証テスト26件、既存テスト維持

**Phase 2への準備完了**:
- PnL規約が統一されたため、Reporter統合が安全に実施可能
- Val/Test分離が保証されたため、AB Testing実装が信頼性高く実施可能
- Entry Gate安全性確保により、機能拡張が安定して実施可能

### 1.2 Phase 2目的

Phase 1で確立した基盤の上に、P1（High Priority）バグを修正し、AB Testing機能を完全稼働させます。

**Phase 2スコープ**:
- **P1バグ修正**: 4件（Trade Type分類、Entry Price更新、Reporter統合、AB Testing有効化）
- **テスト充実**: 単体・統合テスト追加
- **ドキュメント**: Phase 2完了報告作成

**完了条件**:
- [x] 全P1バグ修正完了（3/3件: P1-1, P1-2, P1-3完了、P1-4はPhase 3延期）
- [x] Reporter統一完了（BacktestReporter統合完了、TrainingReporterはPhase 3対応）
- [ ] AB Testing基盤構築完了（**Phase 3へ正式延期**）
- [x] 全テスト合格維持（Phase 0/1/2統合: Phase 2新規テスト16/16合格）
- [x] Phase 2完了報告（Doc18）作成

**Phase 2完了実績**:
- ✅ P1-1: close_reason実装（TP/SL/reversal/manual検出）
- ✅ P1-2: Entry Price更新バグ修正（反転時の価格更新）
- ✅ P1-3: Reporter統合（BacktestReporter close_reason対応、後方互換性維持）
- ⏸️ P1-4: AB Testing基盤 → **Phase 3へ延期**（理由: 仕様不完全性、緊急度低）

**Phase 3延期項目（明文化）**:
- MTF因果性検証（P2バグ → Phase 3）
- Scaler fit境界の厳密化（警告→エラー化 → Phase 3）
- **AB Testing完全実装（記述統計+統計検定、API定義明確化、既存ツール統合 → Phase 3）**
  - Phase 2実績: P1-1/P1-2/P1-3完了、複数seed記録基盤は構築済み
  - Phase 3実装予定: 統計検定（Mann-Whitney U、Cliff's Delta）、4 seed対応、多重比較補正
- **TrainingReporter統合完了（互換API移植、2実装削除 → Phase 3）**
- entry_reason/hold_reason実装（履歴情報必要 → Phase 3）

**工数見積もり**: 5-6日（Doc14レビュー対応を反映）

---

## 2. P1バグ定義と優先順位

### 2.1 P1バグ一覧（Doc00準拠、Phase 2実績反映）

| ID | Bug Name | Priority | 影響範囲 | Phase 2対応 | 実績 |
|----|----------|----------|----------|-------------|------|
| P1-1 | Trade Type Classification (close_reason) | High | Reporter統計精度 | ✅ 実装 | ✅ 完了 |
| P1-2 | Entry Price Update | High | 反転時のPnL計算 | ✅ 実装 | ✅ 完了 |
| P1-3 | Reporter Unification (Backtest) | High | コード保守性 | ✅ 実装 | ✅ 完了 |
| P1-4 | AB Testing Enablement | High | 複数seed比較 | ⏸️ Phase 3延期 | ⏸️ 延期 |

**Phase 2完了項目**: P1-1, P1-2, P1-3（3/3件、16/16テスト合格）  
**Phase 3延期項目**: P1-4（仕様不完全性のため、完全版を Phase 3で実装）

### 2.2 各バグの詳細

#### P1-1: Trade Type Classification（close明示処理）

**Doc00定義**: "close"の明示処理

**Phase 0.2aの実装**: Doc04準拠の10種分類実装済み
```python
# ztb/evaluation/walk_forward/reporter.py（実装済み）
TradeType: Literal[
    "long_open", "long_close", "long_add", "long_reduce",
    "short_open", "short_close", "short_add", "short_reduce",
    "reverse",  # Long⇄Short反転
    "hold"      # ポジション維持
]
# 合計10種（基本8種 + reverse + hold）
```

**Phase 2での対応**:
1. **主目的**: closeアクションの明示的記録（Doc00準拠）
   - `long_close`/`short_close`に終了理由を記録
   - `close_reason`: `"tp"` (利確), `"sl"` (損切), `"reversal"` (反転決済), `"manual"` (手動)
   - env層で生成、reporter層で記録

2. **副次的拡張**（Phase 3検討）:
   - `entry_reason`: Signal/Reentry判定（履歴情報必要 → Phase 3延期）
   - `hold_reason`: Waiting/Avoiding判定（ゲート情報必要 → Phase 3延期）

**影響**: 統計分析の精度向上、close動作の透明性確保

#### P1-2: Entry Price Update（反転時の価格更新）

**現象**: ポジション反転時に`entry_price`が更新されない
- Long→Short反転時、古いLong entry_priceが残る
- 次回のPnL計算が不正確になる

**影響**: 反転戦略の評価精度低下、PnL計算エラー

**Phase 1での対応**: 未対応（Phase 0では問題未認識）

**Phase 2での対応**: 反転時に`entry_price`を現在価格に更新

#### P1-3: Reporter Unification（3実装の統一）

**現象**: Reporter実装が3箇所に分散
1. `ztb/evaluation/walk_forward/reporter.py`: BacktestReporter（Phase 0強化版）
2. `ztb/training/unified_trainer/components/reporter.py`: TrainingReporter
3. `ztb/training/unified_trainer/reporting.py`: TrainingReporter（旧版）

**影響**: コード重複、保守コスト増加、バグ混入リスク

**Phase 1での対応**: BacktestReporterのみ強化（Doc04仕様）

**Phase 2での対応**: BacktestReporterを標準実装とし、他を統合

#### P1-4: AB Testing Enablement（複数seed比較） - **Phase 3へ延期**

**現象**: Walk-Forward評価は複数seedに対応しているが、結果比較機能が不完全
- 各seedの結果が別ファイルに保存されるが統合されない
- 統計的比較機能なし

**影響**: AB Testing実施不可、戦略改善のPDCAサイクル回らず

**Phase 1での対応**: Val/Test分離保証（AB Testing基盤）

**Phase 2での当初計画**: 複数seed結果の統合・比較機能実装（記述統計のみ）

**Phase 2での最終判断**: **Phase 3へ正式延期**

**延期理由**:
1. **緊急度の再評価**: P1-1/P1-2/P1-3は取引精度・PnL計算に直接影響するが、P1-4は分析機能のため緊急度が相対的に低い
2. **仕様の不完全性**: 
   - API定義不足（`compute_descriptive_stats()`未定義、`initial_capital`未定義）
   - seed数混在（2 seed / 4 seed）
   - 統計検定の記載残存（Phase 2は記述統計のみの方針と矛盾）
3. **既存ツールの存在**: tools/ab_test_runner.pyで基本的な2条件比較は実行可能
4. **Phase 3での完全実装**: 統計検定含む完全版として一括実装する方が効率的

**Phase 3での対応予定**:
- AB Testing基盤完全実装（記述統計+統計検定）
- API定義の明確化（SeedResult, ABTestingComparator, compute_metrics等）
- seed数の統一（4seed推奨）
- 既存ツール（tools/ab_test_runner.py）との統合
- Mann-Whitney U検定、Cliff's Delta、Holm-Bonferroni補正

**Phase 2での代替対応**: 既存のtools/ab_test_runner.pyを使用し、手動でCSV結果を比較

**以下のP1-4仕様は参考資料として保持（Phase 3実装時に活用）**

---

## 3. 既存実装分析

### 3.1 Trade Type分類の現状

#### 既存実装（Phase 0.2a）

**ファイル**: `ztb/evaluation/walk_forward/reporter.py`

**実装箇所**: `classify_trade_type()` (Line 14-64)

**現在の分類ロジック**:
```python
# Phase 0.2aで実装済み（Doc04準拠）
def classify_trade_type(position_before: float, position_after: float) -> str:
    """
    Doc04仕様: 詳細Trade Type分類
    
    Returns:
        Trade Type: "long_open", "long_close", "long_add", "long_reduce",
                    "short_open", "short_close", "short_add", "short_reduce",
                    "reverse", "hold"
    """
    if np.isclose(position_before, position_after, atol=1e-8):
        return "hold"
    
    # Long側の判定
    if position_before >= 0 and position_after >= 0:
        if np.isclose(position_before, 0.0, atol=1e-8):
            return "long_open"
        elif np.isclose(position_after, 0.0, atol=1e-8):
            return "long_close"  # ★ Phase 2でclose_reason追加
        elif position_after > position_before:
            return "long_add"
        else:
            return "long_reduce"
    
    # Short側の判定
    if position_before <= 0 and position_after <= 0:
        if np.isclose(position_before, 0.0, atol=1e-8):
            return "short_open"
        elif np.isclose(position_after, 0.0, atol=1e-8):
            return "short_close"  # ★ Phase 2でclose_reason追加
        elif position_after < position_before:
            return "short_add"
        else:
            return "short_reduce"
    
    # Long⇔Short の反転
    if (position_before > 0 and position_after < 0) or \
       (position_before < 0 and position_after > 0):
        return "reverse"
    
    return "hold"
```

**合計10種の分類**:
- 基本8種: long/short × open/close/add/reduce
- +2種: reverse, hold
- ★ Phase 2で`long_close`/`short_close`に`close_reason`フィールド追加

#### Phase 2での拡張計画

**拡張方針**: 既存10種分類を維持し、close理由を別フィールドで記録

**1. close_reasonフィールド追加**（Phase 2実装）:
```python
@dataclass
class TradeRecord:
    # 既存フィールド（Phase 0.2a）
    trade_type: TradeType  # "long_close", "short_close", etc.
    position_prev: float
    position: float
    net_pnl: float
    # ... 他の既存フィールド
    
    # ★ Phase 2追加フィールド
    close_reason: Optional[Literal["tp", "sl", "reversal", "manual"]] = None
```

**close_reason生成経路**（env層で生成）:
```python
# ztb/trading/environment/fast_intraday_env_v456.py: step()メソッド内
if is_closing_position:  # long_close or short_close
    # 判定優先順位: 反転 > TP/SLトリガー > 手動
    # ※PnL符号ではなく、明示的なexitトリガーの有無で判定
    if is_reversal:
        close_reason = "reversal"  # 反転による決済（最優先）
    elif self._is_take_profit_exit():  # TP条件判定（実装必要）
        close_reason = "tp"  # Take Profit (利確)
    elif self._is_stop_loss_exit():  # SL条件判定（実装必要）
        close_reason = "sl"  # Stop Loss (損切)
    else:
        close_reason = "manual"  # 手動的なclose（時間切れ含む）
    
    info["close_reason"] = close_reason

# TP/SL判定メソッド（新規実装）
def _is_take_profit_exit(self) -> bool:
    """TP条件を満たすかチェック（例: PnL > tp_threshold）"""
    # 実装例: profit_pct > self.tp_threshold
    return False  # 実装により判定

def _is_stop_loss_exit(self) -> bool:
    """SL条件を満たすかチェック（例: PnL < -sl_threshold）"""
    # 実装例: loss_pct < -self.sl_threshold
    return False  # 実装により判定
```

**実装注意**:
- 現状のfast_intraday_env_v456.pyにはTP/SLトリガーフラグが存在しない
- Phase 2では簡易的なTP/SL判定メソッドを実装（PnL閾値ベース）
- Phase 3でより洗練されたTP/SL管理を検討

**reporter層での記録**:
```python
# ztb/evaluation/walk_forward/reporter.py: record_trade()
def record_trade(
    self,
    # 既存引数
    timestamp: Union[str, pd.Timestamp],
    position_prev: float,
    position: float,
    pnl: float,
    # ... 他の既存引数
    
    # ★ Phase 2追加引数（オプショナル）
    close_reason: Optional[str] = None,
) -> None:
    # 既存の記録処理 + close_reasonをTradeRecordに追加
```

**close_reasonデータフロー（Doc17提案反映）**:
```
env層 (fast_intraday_env_v456.py)
  ↓ step()メソッドで判定・生成
  ↓ info["close_reason"] = "tp"|"sl"|"reversal"|"manual"
  ↓
evaluator層（関与せず）
  ↓ infoをそのまま伝搬
  ↓
reporter層 (reporter.py)
  ↓ record_trade(close_reason=info.get("close_reason"))
  ↓
TradeRecord
  ↓ close_reason: Optional[str]フィールド
  ↓
CSV出力
```

**データフロー保証**:
- **逆流防止**: reporterからenvへのフィードバックなし（単方向データフロー）
- **欠損防止**: close_reasonがNoneの場合は後方互換（既存トレードも記録可能）
- **evaluator非関与**: evaluator層はinfoを透過的に伝搬するのみ

**2. entry_reason/hold_reason**（Phase 3延期）:
- `entry_reason`: Signal/Reentry判定にはRL信号の履歴が必要（envは持たない）
- `hold_reason`: Waiting/Avoiding判定にはゲート情報が必要
- → Phase 3でevaluator層での記録を検討

**後方互換性**: 新規引数はすべてOptional、既存コードへの影響なし

### 3.2 Entry Price更新の現状

#### 既存実装

**ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py`

**問題箇所**: ポジション反転時（Line 600-700付近）

**現在の動作**:
```python
# fast_intraday_env_v456.py: step()メソッド内
if position_prev != 0 and position_now * position_prev < 0:
    # Reversal detected
    # ... PnL計算 ...
    # ★ 問題: entry_priceが更新されない
```

**問題の影響**:
- Long 100 @ 1000円でエントリー
- Short 100 @ 1100円に反転（entry_priceは1000円のまま）
- 次回エグジット時、PnL計算がentry_price=1000円を使用（不正確）

#### Phase 2での修正計画

**修正方針**: 反転時に`self.entry_price`を現在価格に更新

**実装箇所**: `fast_intraday_env_v456.py`の`step()`メソッド

**修正コード**:
```python
# Reversalケース（Line 650付近）
if position_prev != 0 and position_now * position_prev < 0:
    # 既存の反転処理
    # ...
    
    # ★ Phase 2追加: entry_price更新
    self.entry_price = execution_price
    # 新ポジションのentry_priceとして記録
```

**テスト戦略**:
- Long→Short反転時、entry_priceがShort約定価格に更新されることを確認
- Short→Long反転時、entry_priceがLong約定価格に更新されることを確認
- 次回エグジット時のPnL計算精度を確認

### 3.3 Reporter実装の現状

#### 3実装の詳細

**1. BacktestReporter（標準実装）**
- ファイル: `ztb/evaluation/walk_forward/reporter.py`
- クラス: `BacktestReporter` (Line 245)
- 状態: ✅ Phase 0.2aで強化完了（Doc04仕様）
- 機能:
  - 詳細Trade Type分類（8種+2種reverse+2種hold）
  - PnL統計（gross/net分離）
  - Val/Test分離対応
  - CSV出力機能

**2. TrainingReporter（新版）**
- ファイル: `ztb/training/unified_trainer/components/reporter.py`
- クラス: `TrainingReporter` (Line 16)
- 状態: 🔄 機能実装済みだが統合未完
- 機能:
  - エピソード統計（return, length, success_rate）
  - TensorBoard統合
  - メトリクス保存

**3. TrainingReporter（旧版）**
- ファイル: `ztb/training/unified_trainer/reporting.py`
- クラス: `TrainingReporter` (Line 25)
- 状態: ⚠️ 重複実装（新版と機能ほぼ同じ）
- 機能: 新版とほぼ同じ、一部古い実装

#### Phase 2での統合計画

**統合方針**: BacktestReporterを標準実装とし、Training用機能を統合

**統合アプローチ**:
1. BacktestReporterに`TrainingMode`フラグを追加
2. Training時はエピソード統計のみ記録（詳細Trade記録OFF）
3. Evaluation時は詳細Trade記録ON
4. TensorBoard統合機能を`BacktestReporter`に追加
5. 旧TrainingReporter 2実装を削除

**実装計画**:
```python
# reporter.py: BacktestReporterの拡張
class BacktestReporter:
    def __init__(
        self,
        output_dir: Path,
        training_mode: bool = False,  # Phase 2追加
        tensorboard_writer: Optional[SummaryWriter] = None,  # Phase 2追加
    ):
        self.training_mode = training_mode
        self.tb_writer = tensorboard_writer
        # ...
    
    def record_trade(self, ...):
        if self.training_mode:
            # 簡易統計のみ記録
            self._update_episode_stats(...)
        else:
            # 詳細Trade記録（既存実装）
            self._record_detailed_trade(...)
        
        # TensorBoard記録
        if self.tb_writer is not None:
            self._log_to_tensorboard(...)
```

**削除対象**:
- `ztb/training/unified_trainer/components/reporter.py` → 削除
- `ztb/training/unified_trainer/reporting.py` → 削除

**マイグレーション**:
- TrainingReporterを使用している箇所を`BacktestReporter(training_mode=True)`に置換
- 既存のTraining ScriptでBacktestReporterを使用するよう修正

### 3.4 AB Testing基盤の現状

#### 既存実装

**ファイル**: `ztb/evaluation/walk_forward/evaluator.py`

**Multi-Seed対応状況**:
```python
# evaluator.py: WalkForwardEvaluator.run()
def run(self, seeds: List[int]) -> Dict[str, Any]:
    """複数seedでWalk-Forward評価実行"""
    for seed in seeds:
        # 各seedで独立した評価実行
        val_reporter = BacktestReporter(...)
        test_reporter = BacktestReporter(...)
        
        # 評価実行
        val_results = self._evaluate_on_df(df_val, val_reporter, seed)
        test_results = self._evaluate_on_df(df_test, test_reporter, seed)
        
        # 結果保存（別ファイル）
        val_reporter.save_results(output_dir / f"val_seed{seed}.csv")
        test_reporter.save_results(output_dir / f"test_seed{seed}.csv")
```

**不足点**:
- 各seedの結果が別ファイルに分散保存される
- **条件A/B（例: Entry Gate ON/OFF）の定義・保存構造が不明**
- 統合比較機能なし
- 統計的検定機能なし（Phase 3で本格実装予定）

#### Phase 2での実装計画

**実装方針**: 結果統合・比較クラスを追加、条件定義・保存構造を設計

**既存実装の活用（Doc16/Doc17提案反映）**:
- **AB実行基盤の再利用**: 新規スクリプト作成ではなく、既存`tools/ab_test_runner.py`を拡張して集計層のみ追加
- **メトリクス集計の共通化**: `BacktestReporter`の集計ロジック活用、`ztb/analysis/baseline_comparison.py`参照
- **TrainingReporter統合の安全性**: 移行期間は互換ラッパー維持、破壊的変更の拡散防止
- **既存ドキュメント参照**: seed安定性基準は`docs/v457/32_seed_stability_test.md`、既知バグ回避は`docs/v458/19_phase5_6_final_review.md`のチェックリスト踏襲

**Open Questions回答（Doc16/Doc17）**:
1. **close_reasonのTP/SLトリガー**: Phase 2では簡易的なPnL閾値判定（`tp_threshold`/`sl_threshold`）、Phase 3で既存リスク判定（`ztb/risk/rules.py`）への統合を検討
2. **close_reason生成箇所**: env層で完結、evaluator層での補足は不要（Phase 2範囲）
3. **AB Testing Phase 2範囲**: 記述統計のみ（mean, std, median）、統計検定はPhase 3へ完全移管 ✅
4. **TrainingReporter削除**: 互換API（record_episode, log_metrics, write_tensorboard）をBacktestReporterに移植後、段階的削除
5. **tp_threshold/sl_threshold参照元**: config経由で取得（既存`stop_loss_pct`等との統合はPhase 3）
6. **initial_capital取得元**: Evaluator/config経由でComparatorに渡す（デフォルト200000.0）

**1. 条件定義・保存構造**:
```yaml
# config/v459/experiments/ab_test_gate.yaml（新規）
conditions:
  - name: "gate_on"
    entry_gate:
      enabled: true
      min_confidence: 0.3  # 既存GateConfig仕様に準拠
  - name: "gate_off"
    entry_gate:
      enabled: false

ab_testing:
  seeds: [0, 1]  # Phase 2: 2 seed（基盤構築）、Phase 3: 4 seed拡張（統計検定要件）
  output_base: "results/ab_test_gate"
```

**結果保存構造**（Phase 2: 2 seed）:
```
results/ab_test_gate/
├── gate_on/
│   ├── seed_0/
│   │   ├── val_seed0.csv
│   │   └── test_seed0.csv
│   └── seed_1/
│       ├── val_seed1.csv
│       └── test_seed1.csv
└── gate_off/
    ├── seed_0/
    │   ├── val_seed0.csv
    │   └── test_seed0.csv
    └── seed_1/
        ├── val_seed1.csv
        └── test_seed1.csv
```

**2. 新規クラス**: `ABTestingComparator`

**設計方針（Doc17 Extensibility/Maintainability提案反映）**:
- **メトリクス計算の共通化**: Reporter/AB comparator/基準比較で共通ユーティリティを使用、指標定義のズレ防止
- **close_reasonデータフロー**: env→info→reporter（逆流・欠損防止）、evaluator層は関与せず
- **AB結果スキーマ固定**: `SeedResult`とsummary CSVの項目を固定、Phase 3拡張時も後方互換性保持

**実装計画**:
```python
# ztb/evaluation/walk_forward/ab_testing.py（新規ファイル）
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class SeedResult:
    """単一seedの評価結果
    
    Phase 3拡張時も後方互換性を保つため、フィールドを固定化:
    - seed: シード番号
    - val_metrics: Val期間の集計メトリクス
    - test_metrics: Test期間の集計メトリクス
    - val_trades: Val期間の詳細トレードデータ
    - test_trades: Test期間の詳細トレードデータ
    """
    seed: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_trades: pd.DataFrame
    test_trades: pd.DataFrame

class ABTestingComparator:
    """
    複数seed結果の統合・比較クラス
    
    Phase 2実装範囲（記述統計のみ、2 seed対応）:
    - 単一条件の複数seed結果統合
    - 基本統計計算（mean, std, median, min, max）
    - 2条件の記述統計比較（平均値・中央値差分）
    - シード数: 2個（基盤構築レベル）
    
    Phase 3実装予定（統計検定、4 seed対応）:
    - Mann-Whitney U検定（ノンパラメトリック）
    - Cliff's Delta効果量計算（|d| > 0.33で中程度）
    - 4 seed対応（統計検定に必要なサンプル数確保）
    - 多重比較補正（Holm-Bonferroni法）
    - ベースライン比較機能（BH/SMA/Random/Momentum）
    - 有意水準: α = 0.05（Doc00 section 5.6準拠）
    """
    
    def __init__(self, output_dir: Path, initial_capital: float = 200000.0):
        self.output_dir = output_dir
        self.results: List[SeedResult] = []
        self.initial_capital = initial_capital  # 資本金を保持
    
    def load_results(self, seeds: List[int]) -> None:
        """各seedの結果をロード
        
        CSVソース:
        - val_seedX.csv: BacktestReporter生成の詳細トレードCSV
        - test_seedX.csv: BacktestReporter生成の詳細トレードCSV
        
        val_metrics生成:
        - CSVから集計: net_roi, win_rate, avg_win, avg_loss, etc.
        - 集計ロジック: _compute_metrics_from_trades()を使用（共通化）
        - 集計ロジック: trades CSVを読み込み、PnL統計を計算
        """
        for seed in seeds:
            val_csv = self.output_dir / f"val_seed{seed}.csv"
            test_csv = self.output_dir / f"test_seed{seed}.csv"
            
            # CSV読み込み
            val_trades = pd.read_csv(val_csv)
            test_trades = pd.read_csv(test_csv)
            
            # メトリクス集計
            val_metrics = self._compute_metrics_from_trades(val_trades)
            test_metrics = self._compute_metrics_from_trades(test_trades)
            
            # SeedResultに格納
            self.results.append(SeedResult(
                seed=seed,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                val_trades=val_trades,
                test_trades=test_trades
            ))
    
    def _compute_metrics_from_trades(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """トレードCSVからメトリクスを計算（共通化推奨）
        
        Args:
            trades_df: BacktestReporterが生成したTrade CSV
        
        Returns:
            集計されたメトリクス辞書
        
        Note:
            Phase 3では共通ユーティリティ（ztb.analysis.metrics_utils）への移行を検討
            - BacktestReporter
            - ABTestingComparator
            - baseline_comparison.py
            で同じ計算ロジックを使用し、指標定義のズレを防止
        """
        if len(trades_df) == 0:
            return {
                "net_roi": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_trades": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
        
        # メトリクス計算（既存BacktestReporter実装と整合）
        return {
            "net_roi": trades_df["net_pnl"].sum() / self.initial_capital,  # self.initial_capitalを使用
            "win_rate": (trades_df["net_pnl"] > 0).sum() / len(trades_df),
            "avg_win": trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean() if (trades_df["net_pnl"] > 0).any() else 0.0,
            "avg_loss": trades_df[trades_df["net_pnl"] < 0]["net_pnl"].mean() if (trades_df["net_pnl"] < 0).any() else 0.0,
            "total_trades": len(trades_df),
            "sharpe_ratio": self._calculate_sharpe_ratio(trades_df["net_pnl"]),
            "max_drawdown": self._calculate_max_drawdown(trades_df["net_pnl"].cumsum()),
        }
    
    def compute_descriptive_stats(self, metric_name: str) -> Dict[str, Any]:
        """記述統計を計算（Phase 2: 統計検定なし、記述統計のみ）
        
        Args:
            metric_name: 集計する指標名（例: "net_roi", "win_rate"）
        
        Returns:
            記述統計の辞書（mean, std, median, min, max, values）
        
        Note:
            Phase 2: 記述統計のみ実装（mean, std, median, min, max）
            Phase 3: 統計検定追加予定（Mann-Whitney U、Cliff's Delta、多重比較補正）
        """
        # Val期間の指標を抽出
        val_values = [r.val_metrics[metric_name] for r in self.results]
        
        # 基本統計
        stats = {
            "mean": np.mean(val_values),
            "std": np.std(val_values),
            "median": np.median(val_values),
            "min": np.min(val_values),
            "max": np.max(val_values),
            "values": val_values,  # 生データも保持
            "n_seeds": len(val_values),
        }
        
        return stats
    
    def generate_report(self) -> pd.DataFrame:
        """統合レポート生成（Phase 2: 記述統計のみ）"""
        # 全seed結果をDataFrameに統合
        records = []
        for result in self.results:
            record = {"seed": result.seed}
            record.update(result.val_metrics)  # Val期間のメトリクスを追加
            records.append(record)
        
        report_df = pd.DataFrame(records)
        
        # 集計統計を追加
        summary = {
            "seed": "summary",
            "net_roi_mean": report_df["net_roi"].mean(),
            "net_roi_std": report_df["net_roi"].std(),
            "win_rate_mean": report_df["win_rate"].mean(),
            # ...他のメトリクスの集計
        }
        report_df = pd.concat([report_df, pd.DataFrame([summary])], ignore_index=True)
        
        return report_df
        
        # すべてのペアについて比較
        greater = sum(1 for x in a for y in b if x > y)
        less = sum(1 for x in a for y in b if x < y)
        
        delta = (greater - less) / (n_a * n_b)
        return delta
    
    def _interpret_effect_size(self, delta: float) -> str:
        """効果量の解釈"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:

**Evaluatorへの統合**:
```python
# evaluator.py: WalkForwardEvaluator.run()の修正
def run(self, seeds: List[int], initial_capital: float = 200000.0) -> Dict[str, Any]:
    """複数seedでWalk-Forward評価実行"""
    # 既存の評価ループ
    for seed in seeds:
        # ... 評価実行 ...
        pass
    
    # ★ Phase 2追加: 結果統合・比較（記述統計のみ）
    comparator = ABTestingComparator(self.output_dir, initial_capital=initial_capital)
    comparator.load_results(seeds)
    
    # 統合レポート生成
    report_df = comparator.generate_report()
    report_df.to_csv(self.output_dir / "ab_testing_summary.csv", index=False)
    
    # 統計情報を返す
    return {
        "seeds": seeds,
        "summary": report_df.to_dict(orient="records"),
    }
```

---

## 4. Phase 2実装計画

### 4.1 実装順序（依存関係考慮）

| 順序 | タスク | 依存 | 工数 | 完了条件 |
|------|--------|------|------|----------|
| 1 | P1-2: Entry Price更新 | なし | 0.5日 | 単体テスト合格 |
| 2 | P1-1: Trade Type拡張 | なし | 0.5日 | 単体テスト合格 |
| 3 | P1-3: Reporter統合 | なし | 1.0日 | マイグレーション完了 |
| 4 | P1-4: AB Testing実装 | P1-3 | 1.0日 | 2 seed比較成功 |
| 5 | 統合テスト作成 | 1-4 | 0.5日 | Phase 2テスト全パス |
| 6 | Doc13完了報告 | 5 | 0.5日 | レビュー合格 |

**合計工数**: 5-6日（Doc14/Doc16レビュー対応、テスト強化含む）

### 4.2 各タスクの詳細

#### タスク1: P1-2 Entry Price更新（0.5日）

**実装ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py`

**修正内容**:
1. `step()`メソッド内の反転処理箇所を特定
2. 反転検出時に`self.entry_price = execution_price`を追加
3. 既存の反転PnL計算ロジックは維持

**テストファイル**: `tests/unit/v459/test_p12_entry_price_update.py`（新規）

**テストケース**:
1. Long→Short反転時、entry_priceが更新されることを確認
2. Short→Long反転時、entry_priceが更新されることを確認
3. 反転後の次回エグジット時、PnL計算が正確であることを確認
4. 通常のエントリー/エグジットでentry_price動作が変わらないことを確認

**完了条件**: テスト4/4パス

#### タスク2: P1-1 Trade Type拡張（0.5日）

**実装ファイル**:
- `ztb/trading/environment/fast_intraday_env_v456.py` (修正)
- `ztb/evaluation/walk_forward/reporter.py` (修正)

**修正内容**:

1. **env層**: close_reason生成ロジック追加
```python
# fast_intraday_env_v456.py: step()メソッド内
if is_closing_position:
    # 判定優先順位: 反転 > TP/SLトリガー > 手動
    # ※PnL符号ではなく、明示的なexitトリガーの有無で判定
    if is_reversal:
        close_reason = "reversal"  # 反転による決済（最優先）
    elif self._is_take_profit_exit():  # TP条件判定
        close_reason = "tp"  # Take Profit
    elif self._is_stop_loss_exit():  # SL条件判定
        close_reason = "sl"  # Stop Loss
    else:
        close_reason = "manual"  # 手動的なclose（時間切れ含む）
    
    info["close_reason"] = close_reason
```

**TP/SL判定メソッド実装**:
```python
# fast_intraday_env_v456.py: __init__()でTP/SL閾値を設定
def __init__(self, config: Dict[str, Any], ...):
    # ... 既存初期化 ...
    
    # TP/SL閾値設定（Phase 2: configから取得）
    self.tp_threshold = config.get('tp_threshold', 0.02)  # デフォルト2%
    self.sl_threshold = config.get('sl_threshold', 0.01)  # デフォルト1%

def _is_take_profit_exit(self) -> bool:
    """TP条件チェック（Phase 2: 簡易実装）"""
    if not hasattr(self, 'tp_threshold') or self.tp_threshold <= 0:
        return False
    
    # 現在価格を取得
    current_price = self._resolve_price()  # 既存メソッド活用
    
    # PnL比率でTP判定
    if self.entry_price == 0:
        return False
    
    pnl_pct = (current_price - self.entry_price) / self.entry_price
    if self.position > 0:  # Long
        return pnl_pct > self.tp_threshold
    elif self.position < 0:  # Short
        return -pnl_pct > self.tp_threshold
    return False

def _is_stop_loss_exit(self) -> bool:
    """SL条件チェック（Phase 2: 簡易実装）"""
    if not hasattr(self, 'sl_threshold') or self.sl_threshold <= 0:
        return False
    
    # 現在価格を取得
    current_price = self._resolve_price()  # 既存メソッド活用
    
    # PnL比率でSL判定
    if self.entry_price == 0:
        return False
    
    pnl_pct = (current_price - self.entry_price) / self.entry_price
    if self.position > 0:  # Long
        return pnl_pct < -self.sl_threshold
    elif self.position < 0:  # Short
        return -pnl_pct < -self.sl_threshold
    return False
```

**設定例**（config/v459/fast_intraday_env.yaml）:
```yaml
environment:
  tp_threshold: 0.02  # 2%利益でTP
  sl_threshold: 0.01  # 1%損失でSL
```

2. **reporter層**: close_reasonフィールド追加
```python
# reporter.py: TradeRecordにフィールド追加
@dataclass
class TradeRecord:
    # 既存フィールド...
    close_reason: Optional[str] = None

# record_trade()に引数追加
def record_trade(self, ..., close_reason: Optional[str] = None):
    # 記録処理
```

3. **CSV出力**: close_reasonカラム追加

**テストファイル**: `tests/unit/v459/test_p11_close_reason.py`（新規）

**テストケース**:
1. close_reasonなしでrecord_trade()呼び出し（後方互換性確認）
2. close_reason="tp"指定時、正しく記録されることを確認
3. close_reason="sl"指定時、正しく記録されることを確認
4. **close_reason="reversal"時、PnLに関わらず正しく記録されることを確認**
5. **close_reason="manual"時（PnL=0ケース含む）、正しく記録されることを確認**
6. CSV出力にclose_reasonカラムが含まれることを確認
7. envからreporterへのclose_reason伝搬が正しいことを確認
8. **反転ケースでtp/slが誤ラベル化されないことを確認**

**完了条件**: テスト8/8パス、既存テスト影響なし

#### タスク3: P1-3 Reporter統合（1.0日）

**実装ファイル**:
- `ztb/evaluation/walk_forward/reporter.py`（拡張）
- `ztb/training/unified_trainer/components/reporter.py`（削除）
- `ztb/training/unified_trainer/reporting.py`（削除）

**修正内容**:

1. **事前準備**: TrainingReporter APIの洗い出し
```powershell
# 使用箇所の特定
git grep "TrainingReporter" --name-only
git grep "from.*reporter import" --name-only
git grep "record_episode\|log_metrics\|write_tensorboard" --name-only
```

**互換性保証**:
- `record_episode(return, length, success_rate)` APIをBacktestReporterに移植
- `log_metrics(metrics_dict)` APIをBacktestReporterに移植
- `write_tensorboard(step, values)` APIをBacktestReporterに移植
- Training側コードが期待するメトリクス形式を保持

**移行計画**:
1. 現行2実装の使用箇所をリスト化
2. BacktestReporterに互換性API追加
3. 使用箇所をBacktestReporterに置換（ラッパー期間あり）
4. テスト全通後、旧2実装削除
git grep "from.*training.*reporter import" --name-only
```

2. **BacktestReporterに`training_mode`フラグ追加**
3. **`training_mode=True`時、簡易統計のみ記録**
4. **TensorBoard統合機能追加**
5. **TrainingReporter使用箇所をBacktestReporterに置換**
6. **旧TrainingReporter 2ファイル削除**

**マイグレーション対象**:
- `ztb/training/unified_trainer/trainer.py`
- その他TrainingReporter使用箇所（grep検索で特定）

**テストファイル**: `tests/unit/v459/test_p13_reporter_unification.py`（新規）

**テストケース**:
1. BacktestReporter(training_mode=True)が簡易統計のみ記録
2. BacktestReporter(training_mode=False)が詳細Trade記録
3. TensorBoard統合機能が動作
4. 既存のTraining Scriptが動作（エンドツーエンド）

**完了条件**: テスト4/4パス、旧ファイル削除完了

#### タスク4: P1-4 AB Testing実装（1.0日）

**実装ファイル**:
- `ztb/evaluation/walk_forward/ab_testing.py`（新規）
- `ztb/evaluation/walk_forward/evaluator.py`（修正）
- `scripts/v459/compare_ab_conditions.py`（新規）

**修正内容**:

1. **ABTestingComparatorクラス実装**
   - `load_results()`でseed結果をロード
   - `compare_metrics()`で統計情報計算（mean, std, median）
   - `compare_two_conditions()`で統計的比較（Mann-Whitney U、Cliff's Delta）
   - `generate_report()`で統合レポート生成

2. **条件定義・保存構造の実装**
```yaml
# config/v459/experiments/ab_test_gate.yaml（新規）
conditions:
  - name: "gate_on"
    entry_gate:
      enabled: true
  - name: "gate_off"
    entry_gate:
      enabled: false

ab_testing:
  seeds: [0, 1]  # Phase 2は2 seed
  output_base: "results/ab_test_gate"
```

3. **比較スクリプト実装**（**Phase 2は記述統計のみ**）
```python
# scripts/v459/compare_ab_conditions.py（新規）
def compare_conditions_descriptive(
    condition_a_dir: Path,
    condition_b_dir: Path,
    metric: str = "net_roi"
) -> Dict[str, Any]:
    """2条件の記述統計比較（Phase 2: 統計検定なし）
    
    Returns:
        {
            "condition_a": {"mean": ..., "std": ..., "median": ..., "seeds": [...]},
            "condition_b": {"mean": ..., "std": ..., "median": ..., "seeds": [...]},
            "difference": {"mean_diff": ..., "median_diff": ...}
        }
    """
    comp_a = ABTestingComparator(condition_a_dir)
    comp_b = ABTestingComparator(condition_b_dir)
    
    comp_a.load_results(seeds=[0, 1])
    comp_b.load_results(seeds=[0, 1])
    
    stats_a = comp_a.compute_descriptive_stats(metric)
    stats_b = comp_b.compute_descriptive_stats(metric)
    
    return {
        "condition_a": stats_a,
        "condition_b": stats_b,
        "difference": {
            "mean_diff": stats_a["mean"] - stats_b["mean"],
            "median_diff": stats_a["median"] - stats_b["median"]
        }
    }
```

4. **evaluator.py統合**: 結果統合機能追加

**Phase 2実装範囲の明確化**:
- ✅ 2条件 × 2 seed比較（基盤構築）
- ✅ 記述統計計算（mean, std, median, min, max）
- ✅ CSVレポート生成
- ⚠️ Mann-Whitney U検定 → Phase 3（サンプル不足）
- ⚠️ Cliff's Delta効果量計算 → Phase 3
- ⚠️ 4seed×4split対応 → Phase 3
- ⚠️ 多重比較補正（Holm-Bonferroni） → Phase 3
- ⚠️ ベースライン比較自動化 → Phase 3

**テストファイル**: `tests/unit/v459/test_p14_ab_testing.py`（新規）

**テストケース**（Phase 2: 記述統計のみ）:
1. 2 seedの結果ロードが成功
2. 記述統計情報（mean, std, median, min, max）が正確に計算
3. `compute_descriptive_stats()`が正しく動作
4. 統合レポートCSV生成が成功
5. 2条件比較が動作（gate_on vs gate_off、記述統計差分のみ）

**完了条件**: テスト5/5パス、2条件 × 2 seed比較成功（記述統計）

#### タスク5: 統合テスト作成（0.5日）

**テストファイル**: `tests/integration/test_v459_phase2_integration.py`（新規）

**テストケース**:
1. Entry Price更新後のPnL計算精度
2. Trade Type拡張後のReporter動作
3. Reporter統合後のTraining Script動作
4. AB Testing後の統計レポート生成
5. Phase 0/1/2全テスト合格（回帰テスト）

**完了条件**: 統合テスト5/5パス、全テスト（Phase 0: 77, Phase 1追加: 26, Phase 2追加: 30）合計133/133パス

#### タスク6: Doc13完了報告（0.5日）

**実装ファイル**: `docs/v459/13_phase2_completion_report.md`（新規）

**記載内容**:
1. Phase 2完了サマリー
2. P1バグ修正詳細
3. テスト結果
4. Phase 3準備状況

**完了条件**: Doc13作成完了、レビュー合格

---

## 5. テスト戦略

### 5.1 単体テスト（新規20件）

| ファイル | テスト数 | 対象 |
|----------|----------|------|
| test_p12_entry_price_update.py | 4 | Entry Price更新 |
| test_p11_close_reason.py | 8 | close_reason記録 |
| test_p13_reporter_unification.py | 4 | Reporter統合 |
| test_p14_ab_testing.py | 5 | AB Testing（記述統計） |
| test_phase2_backward_compat.py | 4 | 後方互換性 |

**合計**: 25単体テスト

**注記**: test_p11_close_reasonは8ケース（反転・PnL=0・手動close含む）

### 5.2 統合テスト（新規5件）

| テスト | 対象 | 完了条件 |
|--------|------|----------|
| Entry Price精度 | Env→Reporter PnL | 誤差<0.01% |
| Trade Type記録 | Reporter CSV出力 | 全フィールド記録 |
| Training動作 | Unified Trainer | エピソード統計正常 |
| AB Testing動作 | 2 seed比較 | 統計レポート生成 |
| 全体回帰テスト | Phase 0/1/2 | 128/128パス |

### 5.3 テスト目標

| Phase | 単体テスト | 統合テスト | 合計 | 目標 |
|-------|------------|------------|------|------|
| Phase 0 | 68 | 9 | 77 | 維持 |
| Phase 1 | 26（追加） | 0 | 26（追加） | 維持 |
| Phase 1累計 | 94 | 9 | 103 | 100%合格 |
| Phase 2 | 25 | 5 | 30 | 100%合格 |
| **合計** | **119** | **14** | **133** | **100%** |

**注記**: 
- Phase 0: 77件（基盤）
- Phase 1追加: 26件（Phase 0基盤に追加）
- Phase 1累計: 103件（Phase 0 77件 + Phase 1追加 26件）
- Phase 2追加: 30件（単体25件+統合5件）
- 最終合計: 133件（Phase 0+1+2統合）

**註**: Phase 1の94テストにはPhase 0単体テスト68件を含む

---

## 6. リスク評価と対策

### 6.1 技術リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| Entry Price更新でPnL計算破壊 | High | Low | 詳細な単体テスト |
| Reporter統合でTraining破壊 | High | Medium | マイグレーション慎重実施 |
| AB Testing統計計算誤差 | Medium | Low | scipy標準実装使用 |
| 後方互換性破壊 | High | Low | Phase 0/1回帰テスト |

### 6.2 対策詳細

**Entry Price更新リスク対策**:
- 修正前にentry_price使用箇所を全grep検索
- 反転PnL計算の既存ロジックは変更しない
- 反転後の次回エグジット時の精度を詳細テスト

**Reporter統合リスク対策**:
- マイグレーション前にTrainingReporter使用箇所を全grep検索
- 段階的マイグレーション（1ファイルずつ）
- 各マイグレーション後にTraining Scriptで動作確認

**AB Testing統計リスク対策**:
- scipy.stats.mannwhitneyuを使用（標準実装）
- Cliff's Delta計算を手動実装（検証済みアルゴリズム）
- 既知データでの統計計算精度確認

**後方互換性リスク対策**:
- Phase 0/1の全テストを実行
- 既存APIに破壊的変更を加えない
- 新規引数はすべてOptional

### 6.3 工数リスク

**見積もり精度**: 中程度

**リスク要因**:
- Reporter統合のマイグレーション範囲が不明確
- AB Testing実装の複雑度が予測困難

**バッファ**: 5-6日見積もりに対し、6日確保を推奨

### 6.4 Phase 3延期項目の明確化

**Phase 2では対応しない項目**:
1. **MTF因果性検証**（P2バグ → Phase 3対応）
   - Phase 0で仕様策定のみ、実装は未完了
   - MTF特徴量の因果性検証をPhase 3で実装

2. **Scaler fit境界の厳密化**（警告→エラー化 → Phase 3対応）
   - 現在はtolerance=2.0の警告ベース
   - Phase 3でエラー化、厳密な因果性保証

3. **AB Testing本格統計検定**（4seed×4split、多重比較補正 → Phase 3対応）
   - Phase 2: 2条件 × 2 seed（基盤構築、**記述統計のみ**）
   - Phase 3: 4seed×4split、Mann-Whitney U検定、Holm-Bonferroni補正、ベースライン比較

4. **entry_reason/hold_reason実装**（履歴情報必要 → Phase 3対応）
   - Phase 2: **close_reasonのみ**実装（env層で生成可能）
   - Phase 3: entry/hold_reasonはevaluator層実装（履歴情報を参照）

**延期理由**:
- Phase 2の焦点: P1バグ修正（Trade Type、Entry Price、Reporter、AB Testing基盤）
- Phase 2工数: 5-6日に抑制
- Phase 3で拡張が容易な設計を採用、段階的実装が現実的

**Phase 3での対応予定**:
- MTF因果性検証実装（2日）
- Scaler境界の警告→エラー化（0.5日）
- 4seed×4split AB Testing実装（1日）
- Holm-Bonferroni補正実装（0.5日）
- entry_reason/hold_reason実装（1日）
- **Phase 3合計工数**: 5日

---

## 7. Phase 3準備（展望）

### 7.1 Phase 2完了後の状態

**確立される機能**:
- ✅ close_reason記録機能（close動作の透明性確保）
- ✅ 正確なPnL計算（反転時も正確）
- ✅ 統一されたReporter実装
- ✅ 動作するAB Testing基盤（記述統計）

**Phase 3での拡張予定**:
- entry_reason/hold_reason実装
- Mann-Whitney U検定・効果量計算
- 4seed×4split対応
- 多重比較補正

**Phase 3への準備**:
- AB Testingを用いたハイパーパラメータ最適化
- 複数報酬設計のAB比較（Stage 1/2/3）
- ベースライン比較（BH, SMA, Random, Momentum）

### 7.2 Phase 3予定作業（概要）

**P2バグ修正**（Doc00準拠）:
- [ ] MTF因果性検証（Phase 0で仕様策定のみ）
- [ ] Scaler fit境界の厳密化（警告→エラー）
- [ ] Baseline比較機能の精度向上

**Report設計実験**:
- [ ] Stage 1: 純PnL報酬（ベースライン）
- [ ] Stage 2: 固定ガイダンス報酬
- [ ] Stage 3: Decay付きガイダンス報酬
- [ ] AB比較と統計的検定

**工数見積もり**: 5-7日

---

## 8. 完了条件（再確認）

### 8.1 機能完了条件

- [ ] P1-1: close明示処理完了（close_reason記録動作）
- [ ] P1-2: Entry Price更新完了（反転時の価格更新）
- [ ] P1-3: Reporter統合完了（3実装→1実装）
- [ ] P1-4: AB Testing基盤構築完了（2条件 × 2 seed比較成功）

### 8.2 テスト完了条件

- [ ] 単体テスト20件全パス
- [ ] 統合テスト5件全パス
- [ ] Phase 0/1/2回帰テスト全パス（133/133テスト合格）

### 8.3 ドキュメント完了条件

- [ ] Phase 2完了報告（Doc13）作成
- [ ] 実装変更の詳細記録
- [ ] Phase 3への引き継ぎ事項明記

---

## 9. まとめ

### 9.1 Phase 2の位置づけ

Phase 2は、Phase 1で確立した基盤の上に、以下を実現します：

1. **close明示処理の完成**: close_reason記録によるclose動作の透明性確保（Doc00 P1-1準拠）
2. **PnL計算の正確性**: 反転時も含めた完全な精度
3. **コードベースの統一**: Reporter実装の一本化、保守性向上
4. **AB Testing基盤**: 2条件比較・統計的検定の基盤構築

**Phase 1からの連続性**:
- Phase 1で確立したPnL規約（env=NET, reporter=検証）を活用
- Phase 1で保証したVal/Test分離をAB Testingに適用
- Phase 1のテスト基盤（103テスト）を継承し、133テストに拡張

### 9.2 Phase 3への橋渡し

Phase 2完了により、以下が可能になります：

- **報酬設計実験**: AB Testingを用いた科学的な報酬設計（Stage 1/2/3）
- **ベースライン比較**: 統一されたReporterによる公平な比較
- **ハイパーパラメータ最適化**: 正確なPnL計算に基づく最適化
- **本格的なAB Testing**: 4seed×4split、多重比較補正の実装

**Phase 3で実装予定の延期項目**:
1. MTF因果性検証（2日）
2. Scaler境界厳密化（0.5日）
3. AB Testing本格版（2日）
4. entry/hold_reason実装（1日）
5. **Phase 3合計工数**: 5日

### 9.3 実装原則

Phase 2でも、Phase 1と同様の原則を維持します：

- ✅ **重複実装の回避**: 既存実装を徹底調査、不要な再実装を避ける
- ✅ **後方互換性**: 既存APIを破壊しない、新規引数はOptional
- ✅ **段階的実装**: close_reasonのみPhase 2、entry/hold_reasonはPhase 3
- ✅ **テストファースト**: 実装前にテスト設計、各タスク完了後に即テスト実行
- ✅ **明確な完了条件**: 133/133テスト合格、2条件×2 seed比較成功（記述統計）

### 9.4 修正版Phase 2計画（Doc14/Doc16/Doc17対応済み）

**Doc14レビュー対応**:
- ✅ P1-1定義をDoc00準拠（close明示処理）に修正
- ✅ TradeType分類を実装準拠（10種）に修正
- ✅ close_reason生成経路を明記（env層）
- ✅ AB条件比較設計を追加
- ✅ Reporter統合詳細設計を追加
- ✅ Phase 3延期項目を明文化

**Doc16再レビュー対応**:
- ✅ **Critical**: close_reason判定順を反転優先に修正、TP/SL判定メソッド実装明記
- ✅ **Major**: AB Testingを記述統計のみに限定（Phase 2）、統計検定はPhase 3へ延期
- ✅ **Major**: min_action_thresholdをmin_confidenceに修正（既存GateConfig準拠）
- ✅ **Major**: ABTestingComparatorのCSV読み込み経路を明記（trades CSV→metrics集計）
- ✅ **Major**: TrainingReporter統合の互換API明記（record_episode, log_metrics, write_tensorboard）
- ✅ **Major**: Phase 3延期でentry/hold_reason明記（close_reasonのみPhase 2）
- ✅ **Minor**: 工数見積もり統一（5-6日）
- ✅ **Minor**: close_reasonテストケースに反転・PnL=0・手動close追加
- ✅ **Minor**: TP/SL判定ロジック実装詳細追加

**Doc17再々レビュー対応**:
- ✅ **Critical**: AB Testing Phase 2スコープを記述統計のみに完全統一、Mann-Whitney/Cliff's Delta削除
- ✅ **Critical**: seed数を2 seedに統一（4 seed設定削除）、ディレクトリ構造明確化
- ✅ **Major**: `compute_descriptive_stats()`定義追加
- ✅ **Major**: `initial_capital`パラメータ追加（Comparator初期化時に指定）
- ✅ **Major**: TP/SL判定の設定定義追加（tp_threshold/sl_threshold、config経由）
- ✅ **Major**: `current_price`参照を`_resolve_price()`に統一
- ✅ **Major**: 成果記述を「close_reasonのみ」に修正
- ✅ **Minor**: Task2完了条件を8/8パスに修正
- ✅ **Minor**: Phase 1テスト数明確化（Phase 0: 77, Phase 1追加: 26, 合計: 103）
- ✅ **Minor**: 最終テスト数を133件に統一（Phase 0: 77, Phase 1追加: 26, Phase 2追加: 30）

**Doc16/Doc17提案・Open Questions対応**:
- ✅ **既存実装活用**: AB実行基盤再利用（tools/ab_test_runner.py拡張）、メトリクス集計共通化（BacktestReporter活用）
- ✅ **TrainingReporter統合**: 互換ラッパー期間維持、段階的削除で破壊的変更防止
- ✅ **TP/SL判定**: Phase 2は簡易PnL閾値判定、Phase 3で既存リスク判定（ztb/risk/rules.py）統合検討
- ✅ **Extensibility**: メトリクス計算共通化（Phase 3で共通ユーティリティ化）、SeedResult/CSV項目固定で後方互換性保持
- ✅ **close_reasonデータフロー**: env→info→reporter（単方向、evaluator非関与）、逆流・欠損防止
- ✅ **Open Questions回答**: tp_threshold/sl_thresholdはconfig経由、initial_capitalはEvaluator/configから渡す、Phase 2は記述統計のみ確定
- ✅ **Major**: `compute_descriptive_stats()`定義追加
- ✅ **Major**: `initial_capital`パラメータ追加（Comparator初期化時に指定）
- ✅ **Major**: TP/SL判定の設定定義追加（tp_threshold/sl_threshold、config経由）
- ✅ **Major**: `current_price`参照を`_resolve_price()`に統一
- ✅ **Major**: 成果記述を「close_reasonのみ」に修正
- ✅ **Minor**: Task2完了条件を8/8パスに修正
- ✅ **Minor**: Phase 1テスト数明確化（Phase 0: 77, Phase 1追加: 26, 合計: 103）
- ✅ **Minor**: 最終テスト数を133件に統一（Phase 0: 77, Phase 1追加: 26, Phase 2追加: 30）

**工数見積もり（最終版）**: 5-6日（Doc14/Doc16/Doc17対応完了、提案・Open Questions対応含む）

**Phase 2は6日以内に完了し、Phase 3へスムーズに移行します。** ✅

---

**End of Specification (Rev. 4 - Doc14/Doc16/Doc17 Reviews + Proposals Addressed)**
