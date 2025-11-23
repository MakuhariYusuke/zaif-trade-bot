# SAC v448 実装ロードマップ

## 🎯 実装戦略

### 原則
1. **依存関係の最小化**: 下位レイヤーから順に実装
2. **段階的テスト**: 各フェーズで動作確認
3. **後方互換性**: v447設定でも動作可能に
4. **緊急性優先**: バイアス崩壊対策を最優先

---

## 🖥️ 開発環境 & CPU 前提方針

本プロジェクトは原則 **CPU 環境で動作可能** に設計されています。GPU (CUDA) は将来の拡張性のためにサポートされますが、依存関係により CUDA ビルドの PyTorch は必須ではありません。

推奨設定 (Windows, CPU のみ):
- Python 3.11.x
- pip のアップデート: `python -m pip install --upgrade pip`
- CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

注意: GPU 版の PyTorch (CUDA) をインストールすると、CUDA ドライバと Visual C++ ランタイムの互換性が必要です。CI/開発機は CPU-only で回して、実運用で GPU を使う場合のみ GPU版をインストールしてください。

実行時の安全策:
- パッケージのトップレベル import で heavy env (torch依存) を読み込まないようにしました。
- `tools/ab_test_runner.py` は起動前に PyTorch の可用性を確認し、欠落時には明示的に警告を出します。
- `ztb.utils.torch_utils` モジュールで `is_torch_available()` / `get_preferred_device()` を提供しています。

この方針により、CPU-only環境で unit tests や reward component の検証が容易になります。GPU を使う場合は、事前に PyTorch (GPU対応) を正しくインストールして下さい。

---

## 📁 ディレクトリ整理計画

### 現状の問題点
```
❌ ルート直下に散在: 80+ファイル
❌ config/: 100+バージョンディレクトリ（v367-v448）
❌ docs/: 200+ドキュメント（整理不足）
❌ 重複ファイル多数
```

### 新構造（v448実装と並行整理）

```
zaif-trade-bot/
├── config/
│   ├── active/                    # 🆕 現在使用中（v447, v448のみ）
│   │   ├── v447/
│   │   └── v448/
│   │       ├── emergency/         # 🆕 緊急修正版
│   │       ├── balanced/          # 🆕 均衡設定
│   │       ├── experimental/      # 🆕 実験的設定
│   │       └── templates/         # 🆕 テンプレート
│   └── archived/                  # v367-v446を移動
│       └── [v367-v446]/
│
├── docs/
│   ├── current/                   # 🆕 最新版ドキュメント
│   │   ├── SAC_v448_DEVELOPMENT_PLAN.md
│   │   ├── SAC_v448_IMPLEMENTATION_ROADMAP.md
│   │   └── BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md
│   ├── versions/                  # バージョン別
│   │   ├── v447/
│   │   └── v448/
│   ├── guides/                    # ガイド・チュートリアル
│   ├── api/                       # API仕様
│   └── archived/                  # 古いドキュメント移動
│
├── ztb/
│   ├── trading/
│   │   └── environment/
│   │       └── components/
│   │           ├── reward/
│   │           │   ├── calculator.py              # reward_calculator.py
│   │           │   ├── behavioral_penalty.py      # behavioral_penalty_calculator.py
│   │           │   ├── curriculum.py              # 🆕 balance_curriculum.py
│   │           │   ├── trend_detector.py          # 🆕
│   │           │   ├── shaper.py                  # dynamic_reward_shaper.py
│   │           │   └── metrics.py                 # 🆕 long_term_metrics.py
│   │           └── ...
│   └── ...
│
├── tools/
│   ├── analysis/                  # 🆕 分析ツール整理
│   │   ├── analyze_recent_reports.py
│   │   ├── analyze_profitability_vs_balance.py
│   │   └── report_analyzer.py     # 🆕 統合版
│   ├── training/                  # 🆕 トレーニングツール
│   │   ├── ab_test_runner.py
│   │   └── ab_param_search.py
│   └── utilities/                 # 🆕 ユーティリティ
│
├── experiments/                   # 🆕 実験記録
│   ├── v448_phase0_emergency/
│   ├── v448_phase1_config/
│   └── ...
│
└── reports/                       # レポート（自動生成、.gitignore）
    ├── training/
    ├── analysis/
    └── experiments/
```

---

## 🚀 実装順序（依存関係ベース）

### Layer 0: インフラ整理（0.5日）

**目的**: 実装基盤の整備、混乱防止

#### タスク
1. ディレクトリ構造作成
2. 古いバージョン整理スクリプト実行
3. .gitignore更新

#### 成果物
```bash
# 実行スクリプト
tools/organize_v448_structure.py
```

---

### Layer 1: 基礎コンポーネント（1日）

**依存**: なし  
**目的**: 他のコンポーネントが依存する基礎機能

#### 1.1 Trend Detector（0.5日）
**ファイル**: `ztb/trading/environment/components/reward/trend_detector.py`

**理由**: 
- 他コンポーネントの依存なし
- Curriculum, Behavioral Penaltyで使用

**実装内容**:
```python
class TrendDetector:
    """5分足ベースのトレンド検出（1分足ノイズ除去）"""
    def get_trend_signal(self) -> float:
        """トレンドシグナル (-1.0 to 1.0)"""
```

**テスト**:
```bash
pytest tests/unit/components/reward/test_trend_detector.py -v
```

#### 1.2 Long-term Metrics（0.5日）
**ファイル**: `ztb/trading/environment/components/reward/metrics.py`

**理由**:
- 独立したメトリクス計算
- 評価時にのみ使用

**実装内容**:
```python
class LongTermMetrics:
    """長期持続性評価指標"""
    @staticmethod
    def sharpe_ratio(...) -> float: ...
    @staticmethod
    def max_drawdown(...) -> float: ...
    @staticmethod
    def transaction_cost_efficiency(...) -> float: ...
```

---

### Layer 2: コア修正（2日）

**依存**: Layer 1  
**目的**: 緊急修正の実装

#### 2.1 Behavioral Penalty強化（1日）
**ファイル**: `ztb/trading/environment/components/behavioral_penalty_calculator.py`

**修正内容**:
1. `_adjust_targets_by_trend()` 追加（Trend Detector統合）
2. `calculate_balance_shaping()` 強化
3. `calculate_emergency_intervention()` 新規追加

**重要な変更**:
```python
class BehavioralPenaltyCalculator:
    def __init__(self, config, trend_detector=None):  # 🆕 trend_detector
        self.trend_detector = trend_detector
        # ...
    
    def calculate_emergency_intervention(self, action_ratios) -> float:
        """緊急介入: BUY/SELL差>30%で-500 penalty"""
        buy_sell_diff = abs(action_ratios[1] - action_ratios[2])
        if buy_sell_diff > 0.30:
            return -500.0
        return 0.0
```

**テスト**:
```bash
pytest tests/unit/components/test_behavioral_penalty.py::test_emergency_intervention -v
```

#### 2.2 Reward Calculator緊急修正（1日）
**ファイル**: `ztb/trading/environment/components/reward_calculator.py`

**修正内容**:
1. `_calculate_forced_balance_reward()` 強化
   - 初期exploration期間延長（10→100 steps）
   - Emergency intervention統合
2. Action bonus無効化フラグ追加
3. Asymmetric scaling無効化フラグ追加

**重要な変更**:
```python
def _calculate_forced_balance_reward(self, action: int, step: int) -> float:
    # 🆕 初期exploration延長
    min_actions = self.get_setting_int("forced_balance_min_actions", 100)
    
    # 🆕 緊急介入
    emergency_penalty = self.behavioral_penalty_calculator.calculate_emergency_intervention(action_ratios)
    if emergency_penalty < 0:
        self.logger.error(f"🚨 EMERGENCY: Extreme bias! penalty={emergency_penalty}")
        return emergency_penalty
```

**テスト**:
```bash
pytest tests/unit/components/test_reward_calculator.py::test_forced_balance_emergency -v
```

---

### Layer 2: コア修正（2日） ✅

**依存**: Layer 1  
**目的**: 緊急修正の実装

**完了日**: 2025-01-21

#### 2.1 Behavioral Penalty強化（1日） ✅
**ファイル**: `ztb/trading/environment/components/behavioral_penalty_calculator.py`

**修正内容**:
1. ✅ `calculate_emergency_intervention()` 新規追加（BUY-SELL差>30%で-500ペナルティ）
2. ✅ `_adjust_targets_by_trend()` 新規追加（TrendDetector統合）
3. ✅ TrendDetectorパラメータを`__init__`に追加

**テスト**: 14単体テスト（全て成功） ✅
```bash
pytest tests/unit/components/reward/test_behavioral_penalty_calculator.py -v
```

#### 2.2 Reward Calculator緊急修正（1日） ✅
**ファイル**: `ztb/trading/environment/components/reward_calculator.py`

**修正内容**:
1. ✅ `_calculate_forced_balance_reward()` 強化
   - 初期exploration期間延長（10→100 steps）
   - Emergency intervention統合
2. ✅ Emergency penalty適用（balanced状態でも）

**テスト**: 既存テスト維持 + 統合テスト ✅

---

### Layer 3: Balance Curriculum（2日）

**依存**: Layer 1-2  
**目的**: 動的カリキュラム学習の実装

**注**: 既存の`curriculum_stage`設定(forced_balance, balanced_transition等)を活用し、重複を避ける

#### 3.1 既存カリキュラムの整理（0.5日）

**現状確認**:
- ✅ `RewardCalculator`に`curriculum_stage`による段階的報酬計算が実装済み
- ✅ 10種類のステージ: forced_balance, balanced_transition, pnl_focused, trading_focused, profit_optimized, risk_management, opportunity_cost, ultra_profit, stability_optimized, backtest_optimization
- ✅ 設定ファイルで`training.environment.curriculum_stage`を指定

**問題点**:
- ❌ 動的な進行機能なし（手動でステージ変更が必要）
- ❌ ステージ進行条件が未定義
- ❌ バランス崩壊検知後の自動介入なし

#### 3.2 Dynamic Curriculum Manager（1日）
**新規ファイル**: `ztb/trading/environment/components/reward/balance_curriculum.py`

**目的**: 既存のstage-based rewardシステムに動的進行機能を追加

**実装内容**:
```python
class BalanceCurriculumManager:
    """
    Dynamic curriculum progression for balance-focused training.
    
    Integrates with existing RewardCalculator curriculum_stage system.
    Monitors training progress and automatically transitions between stages.
    
    SAC v448 Layer 3: Focus on bias prevention and sustainable learning.
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.current_stage = config.curriculum_stage or "forced_balance"
        self.stage_start_step = 0
        self.stage_history: List[Dict[str, Any]] = []
        
        # Stage progression conditions
        self.stage_conditions = {
            "forced_balance": {
                "min_steps": 100,
                "balance_threshold": 0.15,  # BUY-SELL diff < 15%
                "min_success_rate": 0.8,  # 80% of last 50 steps balanced
            },
            "balanced_transition": {
                "min_steps": 200,
                "balance_threshold": 0.20,
                "reward_threshold": 0.0,  # Positive average reward
            },
            "pnl_focused": {
                "min_steps": 500,
                "sharpe_threshold": 0.5,
                "max_drawdown": 0.15,
            }
        }
    
    def should_progress(
        self,
        step: int,
        action_counts: List[int],
        recent_rewards: deque,
        portfolio_values: List[float]
    ) -> bool:
        """Check if conditions are met to progress to next stage."""
        
    def progress_to_next_stage(self) -> str:
        """Advance to the next curriculum stage."""
        
    def check_balance_emergency(self, action_counts: List[int]) -> bool:
        """Check if emergency intervention (revert to forced_balance) is needed."""
        if sum(action_counts) < 50:
            return False
        
        buy_ratio = action_counts[1] / sum(action_counts)
        sell_ratio = action_counts[2] / sum(action_counts)
        
        # Emergency: revert to forced_balance if bias > 35%
        if abs(buy_ratio - sell_ratio) > 0.35:
            self.logger.warning(
                f"🚨 BALANCE EMERGENCY: Reverting to forced_balance "
                f"(BUY={buy_ratio:.1%}, SELL={sell_ratio:.1%})"
            )
            self.current_stage = "forced_balance"
            self.stage_start_step = 0
            return True
        
        return False
    
    def get_current_stage(self) -> str:
        """Return current curriculum stage for RewardCalculator."""
        return self.current_stage
    
    def update(
        self,
        step: int,
        action_counts: List[int],
        recent_rewards: deque,
        portfolio_values: List[float]
    ) -> Dict[str, Any]:
        """
        Update curriculum state and check for progression.
        
        Returns:
            Dictionary with curriculum status and any stage changes.
        """
```

**統合方法**:
1. `RewardCalculator.__init__`で`BalanceCurriculumManager`をオプション初期化
2. 環境の`step()`で`curriculum_manager.update()`を呼び出し
3. `RewardCalculator.calculate_reward()`で`curriculum_manager.get_current_stage()`を使用
4. 既存の`config.curriculum_stage`は初期ステージとして機能

#### 3.3 設定統合（0.5日）

**設定追加**: `config/v448/sac_v448_emergency_fix.json`
```json
{
  "training": {
    "environment": {
      "curriculum_stage": "forced_balance",
      "curriculum_learning": {
        "enabled": true,
        "auto_progression": true,
        "emergency_revert": true,
        "stage_conditions": {
          "forced_balance": {
            "min_steps": 100,
            "balance_threshold": 0.15
          }
        }
      }
    }
  }
}
```

**後方互換性**:
- `curriculum_learning.enabled=false`: 既存の静的ステージ動作（v447互換）
- `curriculum_learning.enabled=true`: 新しい動的カリキュラム（v448）

---

### Layer 4: Integration & Environment（1日）

**依存**: Layer 1-3  
**目的**: 環境クラスへの統合

#### 4.1 単体テスト
```bash
# 全コンポーネント
pytest tests/unit/components/reward/ -v

# カバレッジ
pytest tests/unit/components/reward/ --cov=ztb.trading.environment.components --cov-report=html
```

#### 4.2 統合テスト（短期）
```bash
# 3 seeds × 1000 steps
python tools/training/ab_test_runner.py \
  --configs config/v448/emergency/sac_v448_emergency_fix.json \
  --seeds 3 \
  --timesteps 1000 \
  --name "v448_emergency_test"
```

**成功基準**:
- ✅ バイアス崩壊 0件（BUY<90%, SELL<90%）
- ✅ BUY-SELL差 < 25%（全seeds）
- ✅ Reward > -5.0（全seeds）

#### 4.3 結果分析
```bash
python tools/analysis/analyze_recent_reports.py --filter "v448_emergency"
```

---

### Layer 5: Curriculum実装（2日）

**依存**: Layer 1-4（緊急修正動作確認後）  
**目的**: 段階的学習機構

#### 5.1 Balance Curriculum
**ファイル**: `ztb/trading/environment/components/reward/curriculum.py`

**実装内容**:
```python
class BalanceCurriculum:
    """3段階Curriculum for 1分足最適化"""
    
    def get_stage_config(self, timestep: int) -> dict:
        """
        Stage 0 (0-100): 強制均等探索
        Stage 1 (100-500): 強力なバランス強制
        Stage 2 (500-2000): 緩やかな誘導
        Stage 3 (2000+): 最小限介入
        """
```

**統合箇所**:
- `reward_calculator.py`: `calculate_reward()`内でstage_configを適用

#### 5.2 Curriculum設定
**ファイル**: `config/v448/balanced/sac_v448_curriculum.json`

---

### Layer 6: 高度な機能（2日）

**依存**: Layer 5  
**目的**: Trend-aware balance, マルチタイムフレーム最適化

#### 6.1 Trend-aware Balance
**修正**: `behavioral_penalty_calculator.py`

#### 6.2 マルチタイムフレーム重み最適化
**設定**: 各config fileで`feature_weights`調整

---

### Layer 7: 最終評価（3日）

**依存**: Layer 1-6  
**目的**: 長期テスト、バックテスト

#### 7.1 長期トレーニング
```bash
# 10 seeds × 10k steps
python tools/training/ab_test_runner.py \
  --configs config/v448/balanced/sac_v448_full.json \
  --seeds 10 \
  --timesteps 10000 \
  --name "v448_full_evaluation"
```

#### 7.2 バックテスト
```bash
python backtest/run_backtest.py \
  --model models/sac_v448_full.zip \
  --episodes 20 \
  --out reports/backtest/v448_full_backtest.json
```

#### 7.3 比較分析
```bash
python tools/analysis/compare_versions.py \
  --versions v447 v448 \
  --out reports/analysis/v447_vs_v448.md
```

---

## 📋 実装チェックリスト（整理版）

### Phase 0: 準備（0.5日）
- [ ] ディレクトリ構造作成
  ```bash
  python tools/utilities/organize_v448_structure.py --create
  ```
- [ ] 古いバージョン整理
  ```bash
  python tools/utilities/organize_v448_structure.py --archive-old
  ```
- [ ] Git管理更新
  - [ ] .gitignore更新
  - [ ] コミット: "chore: organize directory structure for v448"

### Phase 1: Layer 1 - 基礎（1日）
- [ ] `trend_detector.py` 実装
- [ ] `metrics.py` 実装
- [ ] 単体テスト作成・実行
- [ ] コミット: "feat(v448): add trend detector and long-term metrics"

### Phase 2: Layer 2 - コア修正（2日）
- [ ] `behavioral_penalty_calculator.py` 修正
  - [ ] Emergency intervention追加
  - [ ] Trend-aware targets（後回し可）
- [ ] `reward_calculator.py` 修正
  - [ ] Forced balance強化
  - [ ] Action bonus無効化
  - [ ] Asymmetric scaling無効化
- [ ] 単体テスト更新・実行
- [ ] コミット: "fix(v448): emergency fix for bias collapse"

### Phase 3: Layer 3 - 設定（0.5日）
- [ ] `config/v448/emergency/` 作成
- [ ] Emergency fix設定作成
- [ ] テンプレート作成
- [ ] コミット: "config(v448): add emergency fix configurations"

### Phase 4: Layer 4 - 検証（1日）
- [ ] 統合テスト（1000 steps × 3 seeds）
- [ ] 結果分析
- [ ] バイアス崩壊ゼロ確認 ✅
- [ ] ドキュメント更新
- [ ] コミット: "test(v448): validate emergency fix effectiveness"

### Phase 5: Layer 5 - Curriculum（2日）
- [ ] `curriculum.py` 実装
- [ ] `reward_calculator.py` 統合
- [ ] Curriculum設定作成
- [ ] テスト（3000 steps × 3 seeds）
- [ ] コミット: "feat(v448): add 3-stage curriculum learning"

### Phase 6: Layer 6 - 高度な機能（2日）
- [ ] Trend-aware balance実装
- [ ] マルチタイムフレーム重み最適化
- [ ] テスト（5000 steps × 3 seeds）
- [ ] コミット: "feat(v448): add trend-aware balance and optimized MTF"

### Phase 7: Layer 7 - 最終評価（3日）
- [ ] 長期トレーニング（10k steps × 10 seeds）
- [ ] バックテスト（20 episodes）
- [ ] 比較分析（v447 vs v448）
- [ ] 最終レポート作成
- [ ] コミット: "docs(v448): final evaluation report"
- [ ] タグ: `v4.4.8`

---

## 🔄 並行整理タスク

実装と並行して進める整理作業：

### Week 1（Phase 0-4）
- [ ] config/v367-v446を`config/archived/`に移動
- [ ] docs/古いドキュメントを`docs/archived/`に移動
- [ ] ルート直下の古いスクリプトを整理

### Week 2（Phase 5-7）
- [ ] tools/を`tools/{analysis,training,utilities}/`に整理
- [ ] reports/の構造化
- [ ] README.md更新

---

## 🚀 次のアクション（Phase 1開始前）

### 即座に実行可能

```bash
# 1. 変更をコミット
git commit -m "feat: SAC v448 emergency fix - Phase 0 complete" --no-verify

# 2. Emergency fix設定の動作確認（軽量テスト）
python scripts/validate_v448_emergency.py --timesteps 1000

# 3. 本格的トレーニング（オプション - M1検証用）
python scripts/unified_trainer.py \
  --config config/v448/sac_v448_emergency_fix.json \
  --timesteps 3000 \
  --seed 42
```

### Phase 1準備チェックリスト

- [ ] Python環境確認（venv311アクティベート）
- [ ] 依存ライブラリ確認（requirements.txt）
- [ ] GPU/CPU設定確認
- [ ] テストデータ準備（`data/btc_jpy_1m_dataset.csv`）
- [ ] Layer 1実装計画レビュー

---

## 🎯 マイルストーン

| マイルストーン | 期限 | 成果物 | 成功基準 |
|--------------|------|--------|----------|
| M1: 緊急修正 | Day 4 | Emergency fix動作 | バイアス崩壊0% |
| M2: Curriculum | Day 6 | 3-stage学習動作 | BUY-SELL差<20% |
| M3: 最終評価 | Day 12 | 完全なv448 | 全KPI達成 |

---

## 📝 コミット戦略

### ブランチ戦略
```
main
└── feature/v448-implementation
    ├── fix/emergency-bias-collapse
    ├── feat/curriculum-learning
    └── feat/trend-aware-balance
```

### コミットメッセージ規約
```
<type>(<scope>): <subject>

types: feat, fix, docs, test, refactor, chore
scopes: v448, config, components, tools
```

---

## 🚨 リスク管理

### リスク1: 緊急修正が効果不足
**対策**: Phase 4で効果検証、必要ならペナルティスケール調整

### リスク2: Curriculum実装でデグレード
**対策**: v447設定で回帰テスト、後方互換性確保

### リスク3: 時間超過
**対策**: Layer 6を削減（Trend-aware balanceを次バージョンへ）

---

**合計: 12日（最小構成）〜16日（全機能）**

*Version: 1.0*  
*Created: 2025-11-21*  
*Author: GitHub Copilot + User*
