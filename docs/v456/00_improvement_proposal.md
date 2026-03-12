# v456 Improvement Proposal: Multi-Timeframe & Integrated Signal System

> **大義**: 短期間での高収益性システム構築  
> **Version**: v456.1 (Revised after External Review)  
> **Last Updated**: 2026-01-13

---

## ⚠️ 外部レビュー反映事項 (2026-01-13追記)

### Critical Issues（最優先で対処）
1. **MTFリサンプリング**: 未来データリークの可能性 → クローズドバーのみ使用を徹底
2. **正規化パイプライン**: カテゴリカル/時間特徴量への`OnlineScaler`適用は不適切 → グループ分離
3. **GRU + Off-Policy SAC**: シーケンスリプレイ設計が未詳細 → MLPベースライン優先

### Major Issues
4. **報酬シェーピング**: シェーピング項がPnLを支配するリスク → キャリブレーション検証
5. **アクションフィルタリング**: train-liveミスマッチ → 環境step()内部統合

### 優先順位の改訂
```
新優先順位:
1. データ整合性（MTFリーク防止、正規化分離）     ← 最優先
2. MTF + Time 特徴量追加
3. 非GRUベースラインの確立（Sharpe > 0.3）
4. フィルタリング/ゲーティング統合
5. GRU導入（Phase 1成功後のみ）                  ← 最後
```

---

## 1. 背景と目的
v455の実験により、HFTエージェントは「生存」と「コスト回避」を学習しましたが、1分足単体の情報だけではスプレッドと手数料（約0.1%）の壁を越えるだけのエッジを見つけるのが困難であることが判明しました（バックテスト結果: -9.3%）。

v456では、**「予測力の向上」**を主眼に置き、過去のバージョンで提案・実装されたものの十分活用されていなかった**マルチタイムフレーム（MTF）分析**と**統合シグナルシステム**を導入します。

### 1.1. 過去バージョンからの教訓 (Lessons Learned)

| Version | 主要成果 | 失敗/課題 | v456への適用 |
|---------|---------|----------|-------------|
| **v448** | BUY/SELLバランス問題の発見 | 1分足での極端なバイアス崩壊 | バランス制御機構の改善 |
| **v449** | GRU導入・マルチエクスチェンジ構想 | Lead-Lag効果の活用未完 | Global Feature Integration |
| **v450** | Curriculum Learning導入 | 段階的学習の調整不足 | Regime-Based Curriculum |
| **v451** | 市場微細構造分析 | 時間帯別パフォーマンス差 | Cyclical Time Features |
| **v453** | ハイブリッドフィルター (+2.23%) | トレード数激減 | Soft Filter実装 |
| **v454** | 高勝率達成 (97.5%) | 収益0%（麻痺状態） | Z-Score Mean Reversion |
| **v455** | 安定基盤確立 | 予測力不足 (-9.3%) | MTF + 統合シグナル |

## 2. コアコンセプト

### 2.1. マルチタイムフレーム（MTF）コンテキスト
1分足のノイズに惑わされず、上位足のトレンド方向に従うことで勝率を向上させます。
既存の `ztb.features.multi_timeframe` モジュールを活用し、以下の情報を観測空間に追加します。

*   **Timeframes**: 5分足 (5m), 15分足 (15m), 1時間足 (1h)
*   **Features**:
    *   トレンド方向 (EMA Cross, Slope)
    *   ボラティリティ状態 (ATR Ratio)
    *   主要なサポート/レジスタンスライン

### 2.2. 階層的トレンド分析 (Hierarchical Trend Analysis)
`docs/ACTION_SIGNAL_GUIDE_INTEGRATED_DOCUMENTATION.md` で定義された階層的アプローチを採用し、トレンドの「質」を評価します。

1.  **Phase 1: 方向判定 (Dow Theory / Ichimoku)**
    *   ダウ理論による高値・安値の切り上げ/切り下げ判定。
    *   一目均衡表（三役好転/逆転）による判定 (`ztb/features/trend/ichimoku.py`活用)。
2.  **Phase 2: 強度判定 (ADX)**
    *   ADX > 25 かつ上昇中であれば「強いトレンド」とみなす。
3.  **Phase 3: 波動分析 (Wave Counting)**
    *   エリオット波動的なカウントを行い、トレンドの終焉（第5波など）を警戒する。

### 2.3. 統合シグナルシステム (Integrated Signal System)
RLエージェントの出力（Action）を単独で使うのではなく、テクニカル指標からのシグナルと融合させます。
`docs/v455/00_high_frequency_trading_proposal.md` で提案されたアーキテクチャを実装します。

*   **RL Action**: ニューラルネットワークによる非線形な予測。
*   **Technical Signal**: ルールベースの堅牢なシグナル（例: ゴールデンクロス、RSIダイバージェンス）。
*   **Fusion Strategy**:
    *   **Filter Mode**: テクニカル指標が「買い」を示唆している時のみ、RLの「買い」アクションを通す。
    *   **Boost Mode**: 両方のシグナルが一致した場合、ポジションサイズを拡大する。

## 3. 実装計画

### Step 1: 特徴量エンジニアリングの強化
*   `ztb/features/multi_timeframe.py` を使用して、5m/15m/1hの特徴量を生成するパイプラインを構築。
*   `ztb/features/trend/ichimoku.py` を統合し、一目均衡表のシグナル（雲抜け、基準線クロス）を特徴量化。

### Step 2: 環境 (Environment) の更新
*   `FastIntradayEnv` の観測空間 (Observation Space) を拡張し、MTF特徴量とテクニカル指標を受け取れるようにする。
*   報酬関数は v455 で安定したものをベースにするが、上位足のトレンドに逆らうポジションにはペナルティを追加することを検討。

### Step 3: モデルの学習
*   拡張された特徴量を用いて SAC (または PPO) を再学習。
*   期待される効果: 上位足のトレンドフォローにより、騙し（False Signal）による損失が減少し、1トレードあたりの期待値が向上する。

## 4. 期待される成果
*   **勝率の向上**: 上位足のフィルターにより、逆張りによる損失を回避。
*   **損小利大**: トレンド方向に長く保有し、逆行時は早めに損切りする挙動の獲得。
*   **プラス収支への転換**: バックテストでの損益分岐点（Break-even）突破。

---

## 5. 過去バージョンから継承すべき重要コンポーネント

### 5.1. v449: Global Feature Integration (Lead-Lag効果)
**概要**: 主要取引所（Binance等）の価格が先行し、小規模取引所が追随するLead-Lag効果を活用。

**v456での活用**:
```python
# 特徴量として追加
- btc_binance_return_1m  # Binance BTC/USDTの直近1分リターン
- btc_binance_return_5m  # 直近5分リターン
- btc_dominance_change   # BTCドミナンス変化
- funding_rate_binance   # 先物Funding Rate（センチメント指標）
```

**実装ファイル**: `ztb/features/global_market.py` (既存)

### 5.2. v450: Regime-Based Curriculum Learning
**概要**: 市場環境（レジーム）に応じた段階的学習。

**v456での活用**:
1. **Stage 1**: トレンド相場のみで学習（方向性を学ぶ）
2. **Stage 2**: レンジ相場を追加（Mean Reversion）
3. **Stage 3**: 高ボラティリティ環境を追加（リスク管理）
4. **Stage 4**: 全レジーム混合（汎化）

**実装ファイル**: `ztb/trading/curriculum/balance_curriculum_manager.py`

### 5.3. v451: Cyclical Time Features
**概要**: 時刻情報を周期的な特徴量として符号化し、特定時間帯の「罠」を回避。

**危険時間帯** (JST):
| 時間帯 | 理由 | 対策 |
|--------|------|------|
| 14:00 | Pre-European Trap（流動性低下後のブレイクアウト） | ポジション縮小 |
| 17:00 | London Open（ストップハント、フェイクアウト） | 待機またはワイドストップ |
| 01:00 | London Fix（大口フローによる乱高下） | エントリー制限 |

**特徴量**:
```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
minute_sin = sin(2π * minute / 60)
minute_cos = cos(2π * minute / 60)
day_of_week_sin = sin(2π * dow / 7)  # 週末効果対応
day_of_week_cos = cos(2π * dow / 7)
```

### 5.4. v453: Soft Filter (ハイブリッドフィルター)
**概要**: 危険な状況でも完全に取引を禁止せず、ポジションサイズやしきい値を調整。

**v456での活用**:
```python
regime_constraints = {
    "high_volatility_ranging": {
        "permission": "restricted",
        "position_multiplier": 0.2,       # 20%のポジション
        "confidence_threshold_mod": +0.3,  # 閾値を上げる
        "entry_strategy": "zscore_mean_reversion"
    },
    "extreme_volatility": {
        "permission": "deny"
    },
    "strong_trend": {
        "permission": "allow",
        "position_multiplier": 1.5,       # 150%のポジション
        "exit_strategy": "trailing_stop"
    }
}
```

### 5.5. v454: Z-Score Mean Reversion戦略
**概要**: レンジ相場でのエントリー判断にZ-Scoreを使用。

**最適パラメータ** (v454実験結果):
- `entry_zscore_threshold`: **1.3** （1.3σ以上の乖離でエントリー）
- `stop_loss_pct`: **0.8%**
- `take_profit_pct`: **1.3%**

### 5.6. v455: 報酬関数の安定化
**概要**: HFT環境で発見されたコスト・ボラティリティ意識の報酬設計。

**継承すべきパラメータ**:
```python
reward_config = {
    "min_edge_mult": 1.5,    # 取引コストの1.5倍以上の期待値が必要
    "vol_floor": 0.002,      # ボラティリティ下限（低すぎると取引禁止）
    "time_decay_per_step": 0.0001  # ポジション保持のコスト
}
```

---

## 6. v456 アーキテクチャ設計

### 6.1. 統合システム全体像
```
┌─────────────────────────────────────────────────────────────┐
│                    v456 Trading System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ MTF Features │  │Global Market│  │  Cyclical Time      │  │
│  │ (5m/15m/1h) │  │ (Lead-Lag)  │  │  (Sin/Cos Hour)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └────────────────┼───────────────────┘              │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │             Integrated Feature Vector                 │  │
│  │   [1m_features + MTF + Global + Time + Regime]        │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SAC Agent (GRU-Enhanced)                │   │
│  │   • Policy Network with GRU (文脈理解)               │   │
│  │   • Value Network with Attention                    │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Integrated Signal System                   │   │
│  │   • RL Action + Technical Pattern Fusion            │   │
│  │   • Calibration Gate (EV-based filtering)           │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Regime-Aware Execution                  │   │
│  │   • Soft Filter (Position sizing)                   │   │
│  │   • Dynamic TP/SL                                   │   │
│  │   • Time-based Entry Restriction                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2. GRU-Enhanced Policy Network (v449継承)
1分足の「文脈」を理解するため、Recurrent構造を導入。

```python
class GRUPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, gru_layers=2):
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean, log_std
        )
        
    def forward(self, obs_sequence, hidden_state=None):
        # obs_sequence: [batch, seq_len, obs_dim]
        gru_out, new_hidden = self.gru(obs_sequence, hidden_state)
        action_params = self.policy_head(gru_out[:, -1, :])  # 最後のステップの出力
        return action_params, new_hidden
```

**推奨シーケンス長**: 60〜120ステップ（1〜2時間分の文脈）

---

## 7. 実装ロードマップ

### Phase 1: 基盤構築 (Week 1-2)
| タスク | 優先度 | 難易度 | 既存資産 |
|--------|--------|--------|----------|
| MTF特徴量パイプライン構築 | 🔴高 | 中 | `ztb/features/generators/multi_timeframe/` |
| Cyclical Time Features追加 | 🔴高 | 低 | v451設計書 |
| Global Market Features追加 | 🟡中 | 低 | `ztb/features/global_market.py` |

### Phase 2: 統合システム (Week 3-4)
| タスク | 優先度 | 難易度 | 既存資産 |
|--------|--------|--------|----------|
| 統合シグナルシステム実装 | 🔴高 | 中 | `docs/v455/00_high_frequency_trading_proposal.md` |
| Calibration Gate実装 | 🔴高 | 中 | `docs/v455/01_calibration_and_execution_model.md` |
| Soft Filter実装 | 🟡中 | 低 | v453/v454実装 |

### Phase 3: 学習と最適化 (Week 5-6)
| タスク | 優先度 | 難易度 | 既存資産 |
|--------|--------|--------|----------|
| GRU Policy Network導入 | 🟡中 | 高 | v449設計 |
| Regime-Based Curriculum | 🟡中 | 中 | v450実装 |
| ハイパーパラメータ最適化 | 🔴高 | 中 | Optuna統合済み |

### Phase 4: 検証と調整 (Week 7-8)
| タスク | 優先度 | 難易度 | 備考 |
|--------|--------|--------|------|
| バックテスト検証 | 🔴高 | 低 | `scripts/v455/backtest_hft.py`流用 |
| ペーパートレード | 🔴高 | 中 | 実環境検証 |
| パラメータファインチューニング | 🟡中 | 中 | - |

---

## 8. リスクと対策

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| MTF特徴量の次元爆発 | 高 | 中 | 特徴量選択（PCA/重要度分析）、正規化 |
| GRU学習の収束遅延 | 中 | 高 | Curriculum Learningで段階的導入 |
| 過学習（特定レジームへの過適応） | 高 | 高 | Domain Randomization、Early Stopping |
| Lead-Lag効果の消失（市場効率化） | 中 | 中 | 定期的な効果検証、フォールバック戦略 |
| 時間帯フィルターによる機会損失 | 中 | 中 | Soft Filter（完全禁止→縮小）で対応 |

---

## 9. 成功指標 (KPI) - 外部レビュー反映版

### ⚠️ 9.0. CRITICAL: KPIの統計的根拠

> **外部レビュー指摘**: Sharpe 1.0、Return +5%等の目標値に統計的な根拠がない

#### 9.0.1. KPI設定の前提条件
```python
# 統計的に有意な目標設定のための前提
KPI_STATISTICAL_BASIS = {
    "sample_size": {
        "min_trades": 200,           # 最低200トレードで評価
        "min_days": 60,              # 最低60日間
        "bootstrap_iterations": 1000, # ブートストラップ反復数
    },
    "confidence_level": 0.95,        # 95%信頼区間
    "benchmark": {
        "buy_and_hold_btc": "v455期間のBTC/JPYリターン",
        "random_baseline": "ランダムエントリーの期待値",
    },
}
```

#### 9.0.2. 統計的検証ルール
```python
def validate_kpi_achievement(results: pd.DataFrame, target: dict) -> dict:
    """
    KPI達成を統計的に検証
    
    Returns:
        dict with {achieved, confidence_interval, p_value}
    """
    # ブートストラップで信頼区間を計算
    bootstrap_samples = []
    for _ in range(1000):
        sample = results.sample(frac=1.0, replace=True)
        bootstrap_samples.append(sample["return"].sum())
    
    ci_lower = np.percentile(bootstrap_samples, 2.5)
    ci_upper = np.percentile(bootstrap_samples, 97.5)
    
    # 帰無仮説: リターン <= 0 のt検定
    t_stat, p_value = stats.ttest_1samp(results["return"], 0)
    
    return {
        "point_estimate": results["return"].sum(),
        "confidence_interval_95": (ci_lower, ci_upper),
        "p_value": p_value,
        "statistically_significant": p_value < 0.05 and ci_lower > 0,
    }
```

### 9.1. 改訂版KPI目標

| 指標 | 現状(v455) | **必達目標** | **挑戦目標** | 統計要件 |
|------|-----------|------------|------------|----------|
| バックテストリターン | -9.3% | **>0%** (95%CI) | **>+5%** | CI下限 > 0 |
| Sharpe Ratio | N/A | **>0.3** | **>1.0** | p < 0.05 |
| 勝率 | N/A | **>50%** | **>55%** | n >= 200 |
| Profit Factor | <1.0 | **>1.1** | **>1.3** | n >= 200 |
| 最大DD | N/A | **<15%** | **<10%** | 95%CI上限 |
| トレード数/日 | ~3.4 | **50-300** | - | 流動性考慮 |

### 9.2. ベースラインとの比較
```python
BASELINE_COMPARISON = {
    "buy_and_hold": {
        "description": "同期間のBTC/JPYホールド",
        "requirement": "v456 return > buy_and_hold return - 5%",
    },
    "random_entry": {
        "description": "ランダムエントリー・固定SL/TP",
        "requirement": "v456 Sharpe > random Sharpe + 0.2",
    },
    "v455": {
        "description": "前バージョン",
        "requirement": "v456 return > v455 return (= -9.3%)",
    },
}
```

### 9.3. 段階的マイルストーン

#### Milestone 1: データ整合性 (Week 1)
- [ ] MTFリサンプリングのリーク検出テスト実装
- [ ] 正規化パイプラインの分離完了
- [ ] タイムゾーン処理の統一

#### Milestone 2: 非GRUベースライン (Week 2-3)
- [ ] MLPベースSACで MTF+Time特徴量を検証
- [ ] **Sharpe > 0.3** を達成（これがGRU導入の前提条件）
- [ ] Return > -5%（v455改善）

#### Milestone 3: フィルタリング統合 (Week 4-5)
- [ ] Soft Filter + Calibration Gateを環境内統合
- [ ] Train-Live Parityテストの実施
- [ ] **Sharpe > 0.5**, **Return > 0%**

#### Milestone 4: GRU導入（条件付き） (Week 6+)
- [ ] **前提条件**: Milestone 3で Sharpe > 0.5 達成
- [ ] シーケンスリプレイ + Burn-in設計完了
- [ ] **挑戦目標**: Sharpe > 1.0, Return > +5%

---

## 10. 追加の強化提案（Critical Enhancements）

### 10.1. 報酬関数の根本的改善（v448/v454の教訓）
v448で発見された「BUY/SELLバランス崩壊」問題と、v454での「高勝率/低収益」のパラドックスを解決するため、報酬関数に以下の改善を適用：

```python
class EnhancedRewardCalculator:
    def calculate_reward(self, trade_result, context):
        # 1. Trade-Based PnL（実現損益ベース）- v454の教訓
        # ステップベースではなくトレード完結時の損益で評価
        pnl_mode = "trade"  # Not "step"
        
        # 2. Balance Enforcement（均衡強制）- v448の教訓
        # BUY/SELL比率が崩壊している場合の強制ペナルティ
        if self.buy_sell_ratio > 0.7 or self.buy_sell_ratio < 0.3:
            balance_penalty = -abs(0.5 - self.buy_sell_ratio) * 10.0
        
        # 3. Regime-Aligned Bonus
        # 上位足トレンドに沿ったアクションにボーナス
        if self._is_trend_aligned(action, mtf_trend):
            regime_bonus = 0.05
        
        return base_pnl + balance_penalty + regime_bonus
```

### 10.2. 動的パラメータ調整システム
過去の実験で最適パラメータは市場状況で変化することが判明。オンラインでの適応機能：

```python
class DynamicParameterAdapter:
    """市場状況に応じたパラメータ動的調整"""
    
    def adapt_parameters(self, current_regime, recent_performance):
        if current_regime == "high_volatility_ranging":
            return {
                "entry_zscore_threshold": 1.3,  # v454最適値
                "stop_loss_pct": 0.008,
                "take_profit_pct": 0.013,
                "position_size_mult": 0.2
            }
        elif current_regime == "strong_trend":
            return {
                "entry_strategy": "momentum_breakout",
                "trailing_stop": True,
                "position_size_mult": 1.5
            }
```

### 10.3. 多段階フィルタリングシステム（Defense in Depth）
v453-v454の「単一フィルター依存」を改善し、多層防御を実装：

```
Layer 1: Time Filter（時間帯フィルター）
  └─ 危険時間帯（14:00, 17:00, 01:00）→ポジション縮小

Layer 2: Regime Filter（レジームフィルター）
  └─ extreme_volatility → エントリー禁止
  └─ high_volatility_ranging → 制限モード

Layer 3: MTF Trend Filter（上位足フィルター）
  └─ 1h足がdowntrend時のBUY → 禁止

Layer 4: Calibration Gate（EV判定）
  └─ 期待値マイナス → エントリー禁止

Layer 5: Risk Budget（リスク予算）
  └─ 日次損失リミット超過 → 全エントリー停止
```

### 10.4. 特徴量の厳選（Feature Engineering Best Practices）
過去のバージョンで効果が確認された特徴量を優先的に採用：

**Tier 1（必須）**:
- MTF EMA Trend Direction (5m/15m/1h)
- Cyclical Time Encoding (hour_sin/cos)
- Vol Rank / Vol Ratio
- Z-Score (for mean reversion regimes)

**Tier 2（推奨）**:
- Lead-Lag Features (Binance return)
- Ichimoku Cloud Status
- ADX Trend Strength
- Funding Rate

**Tier 3（実験的）**:
- Elliott Wave Position
- Harmonic Patterns
- Orderbook Imbalance

---

## 11. v456 実装の優先順位（Action Priority）

> **⚠️ 第1次外部レビュー後に優先順位を大幅改訂**
> 詳細は [07_revised_action_plan.md](07_revised_action_plan.md) を参照

### 改訂後の優先順位（概要）

| Priority | Week | 内容 | 完了基準 |
|----------|------|------|----------|
| **P0** | 1 | データ整合性確保（MTFリーク防止、正規化分離） | テスト100%パス |
| **P1** | 2-3 | 特徴量追加 + MLPベースライン確立 | Sharpe > 0.3 |
| **P2** | 4-5 | フィルタリング統合（Train-Live Parity） | 同一ロジック確認 |
| **P3** | 6+ | GRU導入（**P1成功後のみ**） | 条件付き |

### 旧優先順位（参考・非推奨）

~~以下は初期提案時の優先順位だが、外部レビューで**Critical Issues**が指摘されたため、
上記の改訂優先順位に従うこと。~~

<details>
<summary>旧 Week 1-4 計画（Deprecated）</summary>

- ~~Week 1: Cyclical Time Features, MTF Trend Direction, Vol Rank~~ → **Week 2-3 に移動（P1）**
- ~~Week 2: Soft Filter, Time-based Restriction~~ → **Week 4-5 に移動（P2）**
- ~~Week 3: Calibration Gate, RL + Technical Fusion~~ → **Week 4-5 に移動（P2）**
- ~~Week 4: GRU Policy Network~~ → **Week 6+ に移動（P3: 条件付き）**

</details>

---

## 12. 参考文献
*   `docs/v455/00_high_frequency_trading_proposal.md` (統合シグナルシステムの提案)
*   `docs/v455/15_v455_summary_and_handover.md` (v455ハンドオーバーレポート)
*   `docs/ACTION_SIGNAL_GUIDE_INTEGRATED_DOCUMENTATION.md` (階層的トレンド分析)
*   `docs/v449/comprehensive_improvement_plan.md` (GRU/Lead-Lag提案)
*   `docs/v450/02_curriculum_learning.md` (カリキュラム学習設計)
*   `docs/v451/market_microstructure_analysis.md` (時間帯分析)
*   `docs/v453/strategy_improvement_plan_v453.md` (ハイブリッドフィルター)
*   `docs/v454/02_hybrid_strategy_analysis.md` (Z-Score Mean Reversion)
*   `docs/v448/SAC_v448_DEVELOPMENT_PLAN.md` (BUY/SELLバランス問題分析)
*   `ztb/features/multi_timeframe.py` (MTF実装)
*   `ztb/features/trend/ichimoku.py` (一目均衡表実装)
*   `ztb/features/global_market.py` (グローバル市場特徴量)
*   `ztb/features/generators/multi_timeframe/engine.py` (MTFエンジン)
