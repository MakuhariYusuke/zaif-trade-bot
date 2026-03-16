# v456 Technical Specification: 詳細技術仕様書

> **Version**: v456.1 (Revised after External Review)  
> **Date**: 2026-01-13  
> **Status**: Draft (Revised)

---

## ⚠️ Critical Design Notes (External Review Feedback)

### 外部レビューで指摘された重大な設計課題
1. **GRU + Off-Policy SAC**: シーケンスリプレイとburn-in設計が未詳細化
2. **報酬シェーピング支配**: シェーピング項がPnL項を支配するリスク
3. **アクションフィルタリング**: 環境内部でのフィルタリングがtrain-liveミスマッチを引き起こす

### 対応方針
- GRU導入は**段階的に検討**し、まずは非GRUベースラインを確立
- 報酬シェーピングの係数をPnL項のスケールに合わせてキャリブレーション
- Soft Filter/Gatingは環境の`step()`内部に統合

詳細は各セクションの「⚠️ CRITICAL」マークを参照。

---

## 1. システム概要

### 1.1. 設計目標
1. **予測力の向上**: 1分足単体から脱却し、マルチタイムフレーム情報を活用
2. **フィルタリングの最適化**: 硬直的なフィルターから動的なSoft Filterへ移行
3. **収益性の達成**: バックテストで+5%以上のリターンを目標

### 1.2. 既存資産の活用
| コンポーネント | 現行実装 | v456での活用 |
|---------------|---------|-------------|
| `FastIntradayEnv` | v455で安定化済み | Observation Space拡張 |
| `CalibrationGate` | EV判定ロジック実装済み | 統合シグナルシステムのゲートとして活用 |
| `GlobalMarketFeatureEngineer` | Lead-Lag特徴量 | 外部市場データ統合 |
| `MultiTimeframeFeatureEngineer` | MTF特徴量生成 | 5m/15m/1h特徴量 |
| `compute_hft_reward` | 報酬関数 | パラメータ調整・拡張 |

---

## 2. 報酬関数仕様

### 2.1. 現行報酬関数 (v455)
```
r_t = pnl_norm - fee_norm - slip_norm - churn_penalty - hold_penalty 
      - inventory_risk - edge_penalty - low_vol_penalty - time_decay_penalty
```

**各項の定義**:
| 項 | 定義 | 目的 |
|----|-----|------|
| `pnl_norm` | `pnl / (ATR × max_position)` | 正規化されたPnL |
| `fee_norm` | `fee / (ATR × max_position)` | 手数料コスト |
| `slip_norm` | `slippage / (ATR × max_position)` | スリッページコスト |
| `churn_penalty` | `α × |Δposition| / max_position` | 過剰売買ペナルティ |
| `hold_penalty` | `β × holding_steps` | 長期保持ペナルティ |
| `inventory_risk` | `γ × |position| / max_pos × vol_ratio × 100` | 在庫リスク |
| `edge_penalty` | `rate × max(0, required_edge - expected_move)` | 不十分なエッジへのペナルティ |
| `low_vol_penalty` | `rate × max(0, vol_floor - vol_ratio)` | 低ボラ時のペナルティ |
| `time_decay_penalty` | `ramp × max(0, steps - grace)` | グレース期間後の時間減衰 |

### ⚠️ 2.2. CRITICAL: 報酬シェーピングのキャリブレーション

> **外部レビュー指摘**: シェーピング項の総和がPnL項を圧倒すると報酬ハッキングが発生

#### 2.2.0. 報酬スケールバランスの検証
```python
def validate_reward_scale_balance(
    sample_episodes: List[Dict],
    max_shaping_ratio: float = 0.5,
) -> Dict:
    """
    ⚠️ CRITICAL: シェーピング項がPnL項を支配していないか検証
    
    Args:
        sample_episodes: サンプルエピソード群
        max_shaping_ratio: シェーピング項の許容最大比率
    
    Returns:
        validation_result: 検証結果
    """
    pnl_magnitudes = []
    shaping_magnitudes = []
    
    for episode in sample_episodes:
        for step in episode["steps"]:
            pnl_abs = abs(step["pnl_norm"])
            
            shaping_sum = (
                abs(step.get("churn_penalty", 0)) +
                abs(step.get("hold_penalty", 0)) +
                abs(step.get("inventory_risk", 0)) +
                abs(step.get("edge_penalty", 0)) +
                abs(step.get("low_vol_penalty", 0)) +
                abs(step.get("time_decay_penalty", 0))
            )
            
            pnl_magnitudes.append(pnl_abs)
            shaping_magnitudes.append(shaping_sum)
    
    avg_pnl = np.mean(pnl_magnitudes)
    avg_shaping = np.mean(shaping_magnitudes)
    shaping_ratio = avg_shaping / (avg_pnl + 1e-8)
    
    is_valid = shaping_ratio <= max_shaping_ratio
    
    return {
        "is_valid": is_valid,
        "avg_pnl_magnitude": avg_pnl,
        "avg_shaping_magnitude": avg_shaping,
        "shaping_ratio": shaping_ratio,
        "recommendation": f"Reduce shaping coefficients by {shaping_ratio / max_shaping_ratio:.1f}x" if not is_valid else "OK"
    }


# 実行時チェック用
def auto_calibrate_shaping_coefficients(
    base_coefficients: Dict[str, float],
    sample_episodes: List[Dict],
    target_ratio: float = 0.3,
) -> Dict[str, float]:
    """
    シェーピング係数を自動キャリブレーション
    
    シェーピング項の総和がPnL項の target_ratio 倍になるよう調整
    """
    validation = validate_reward_scale_balance(sample_episodes, max_shaping_ratio=1.0)
    
    if validation["shaping_ratio"] == 0:
        return base_coefficients
    
    scale_factor = target_ratio / validation["shaping_ratio"]
    
    calibrated = {}
    for key, value in base_coefficients.items():
        if key in ["alpha", "beta", "gamma", "edge_penalty_rate", "vol_floor_penalty", "hold_ramp"]:
            calibrated[key] = value * scale_factor
        else:
            calibrated[key] = value
    
    logger.info(f"Shaping coefficients scaled by {scale_factor:.3f}")
    return calibrated
```

### 2.2.1. MTFトレンドアライメントボーナス
```python
def calc_mtf_alignment_bonus(action, mtf_trend_5m, mtf_trend_15m, mtf_trend_1h):
    """
    上位足トレンドとアクションの整合性ボーナス
    
    Args:
        action: -1 (Short), 0 (Flat), +1 (Long)
        mtf_trend_*: -1 (Downtrend), 0 (Neutral), +1 (Uptrend)
    
    Returns:
        bonus: 0.0 ~ 0.15
    """
    alignment_score = 0.0
    
    # 重み: 長期足ほど重要
    weights = {"5m": 0.2, "15m": 0.3, "1h": 0.5}
    
    for tf, trend in [("5m", mtf_trend_5m), ("15m", mtf_trend_15m), ("1h", mtf_trend_1h)]:
        if action * trend > 0:  # 同方向
            alignment_score += weights[tf]
        elif action * trend < 0:  # 逆方向
            alignment_score -= weights[tf] * 1.5  # 逆行ペナルティは大きめ
    
    return max(-0.2, min(0.15, alignment_score * 0.15))
```
```

#### 2.2.2. バランス強制項 (v448教訓)
```python
def calc_balance_enforcement_penalty(buy_ratio, sell_ratio, action):
    """
    BUY/SELLバランス崩壊時の強制ペナルティ
    
    警告ゾーン: 比率が0.3-0.7の範囲外
    危険ゾーン: 比率が0.2-0.8の範囲外
    """
    target_ratio = 0.5
    current_ratio = buy_ratio / (buy_ratio + sell_ratio + 1e-8)
    deviation = abs(current_ratio - target_ratio)
    
    if deviation > 0.3:  # 危険ゾーン
        penalty_mult = 5.0
    elif deviation > 0.2:  # 警告ゾーン
        penalty_mult = 2.0
    else:
        penalty_mult = 0.0
    
    # 偏りを加速させるアクションに追加ペナルティ
    if (current_ratio > 0.5 and action > 0) or (current_ratio < 0.5 and action < 0):
        penalty_mult *= 1.5
    
    return -deviation * penalty_mult
```

### 2.3. 推奨パラメータ設定
```python
REWARD_PARAMS_V456 = {
    # v455継承（安定性確保）
    "alpha": 0.2,                  # Churn penalty coefficient
    "beta": 0.01,                  # Hold penalty coefficient
    "gamma": 0.5,                  # Inventory risk coefficient
    "min_edge_mult": 1.5,          # Required edge multiple
    "edge_penalty_rate": 0.5,      # Edge shortfall penalty
    "vol_floor": 0.002,            # Minimum volatility
    "vol_floor_penalty": 1.0,      # Low-vol penalty
    "hold_grace": 5,               # Grace period (steps)
    "hold_ramp": 0.001,            # Time decay rate
    
    # v456新規
    "mtf_alignment_weight": 0.15,  # MTF alignment bonus weight
    "balance_enforcement": True,   # Enable balance enforcement
    "balance_warning_threshold": 0.2,  # Warning zone threshold
    "balance_danger_threshold": 0.3,   # Danger zone threshold
}
```

---

## 3. 観測空間仕様

> **⚠️ 第2次レビュー対応**: 特徴量数を88に統一
> 詳細な特徴量定義は [02_feature_engineering_spec.md](02_feature_engineering_spec.md) v456.2 を参照

### 3.1. 現行観測空間 (v455)
```
Observation = [Market Features (N), Account State (3)]
```

### 3.2. v456拡張観測空間（88次元）
```python
OBSERVATION_SPACE_V456 = {
    # === Base Features (30次元) ===
    # 1分足特徴量 - 既存維持
    "base_1m_features": 30,     # idx [0:30]
    
    # === MTF Features (27次元) ===
    # 5min/15min/1h 各9特徴量
    "mtf_5min": 9,              # idx [30:39]
    "mtf_15min": 9,             # idx [39:48]
    "mtf_1h": 9,                # idx [48:57]
    
    # === Cyclical Time Features (6次元) ===
    "time_hour_sin": 1,         # idx [57]
    "time_hour_cos": 1,         # idx [58]
    "time_minute_sin": 1,       # idx [59]
    "time_minute_cos": 1,       # idx [60]
    "time_dow_sin": 1,          # idx [61]
    "time_dow_cos": 1,          # idx [62]
    
    # === Global Market Features (9次元) ===
    # 連続値6 + フラグ3
    "global_spread": 1,         # idx [63]
    "global_return_1m": 1,      # idx [64]
    "global_return_5m": 1,      # idx [65]
    "global_vol_1m": 1,         # idx [66]
    "global_vol_ratio": 1,      # idx [67]
    "global_usdt_premium": 1,   # idx [68]
    "global_flag_spread": 1,    # idx [69]
    "global_flag_return": 1,    # idx [70]
    "global_stale_flag": 1,     # idx [71]
    
    # === Regime Features (13次元) ===
    "regime_onehot": 11,        # idx [72:83]
    "vol_rank": 1,              # idx [83]
    "vol_ratio": 1,             # idx [84]
    
    # === Account State (3次元) ===
    "position_norm": 1,         # idx [85]
    "remaining_ttl_norm": 1,    # idx [86]
    "last_cost_norm": 1,        # idx [87]
}

# 合計: 30 + 27 + 6 + 9 + 13 + 3 = 88
TOTAL_OBSERVATION_DIM = 88
```

### 3.3. 特徴量正規化仕様
| 特徴量グループ | 正規化手法 | 備考 |
|--------------|----------|------|
| 価格系 | Log Return | `log(p_t / p_{t-1})` |
| ボラティリティ系 | Z-Score (Rolling 100) | `(x - μ) / σ` |
| 時刻系 | Sin/Cos Encoding | 周期性を保持 |
| Regime | One-Hot | カテゴリカル |
| アカウント状態 | Min-Max (固定範囲) | 範囲は事前定義 |

---

## 4. アクション空間仕様

### 4.1. 現行アクション空間 (v455)
```python
action_space = Box(
    low=[-1.0, 0.0],   # [target_position, ttl_fraction]
    high=[1.0, 1.0],
    dtype=float32
)
```

### 4.2. v456アクション空間（拡張候補）
現行維持を推奨。理由：
1. シンプルさが学習の安定性に寄与
2. ポジションサイズ調整はSoft Filterで対応
3. TTLはレジーム連動で動的調整

### ⚠️ 4.3. CRITICAL: Train-Live Parity（外部レビュー指摘）

> **外部レビュー指摘**: Soft Filter/Gating がtrain時と推論時で異なる場所にあると、エージェントは「フィルタされた行動」を学習できない

#### 4.3.1. 設計原則: 環境内部での統一フィルタリング
```python
class FastIntradayEnvV456(FastIntradayEnv):
    """
    ⚠️ CRITICAL: Soft Filter/Gatingは環境のstep()内部に統合
    
    これにより、学習時も推論時も同一のフィルタリングロジックが適用され、
    train-live parityが保証される。
    """
    
    def __init__(self, *args, soft_filter_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_filter_config = soft_filter_config or {}
        self._calibration_gate = CalibrationGate()  # 内部でゲーティング
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        ⚠️ CRITICAL: フィルタリングをstep()内部で実行
        
        Flow:
        1. Raw action受取
        2. Soft Filter適用（時間/レジーム制約）
        3. Calibration Gate適用（EV判定）
        4. 実効アクションで環境更新
        5. 報酬計算（フィルタリングのコストも含む）
        """
        # 1. 元のアクションを保存（ログ用）
        raw_action = action.copy()
        
        # 2. Soft Filter: ポジションサイズ調整
        filtered_action, multiplier = self._apply_soft_filter(action)
        
        # 3. Calibration Gate: EV判定
        gated_action, gate_info = self._apply_calibration_gate(filtered_action)
        
        # 4. 実効アクションで環境更新
        obs, reward, terminated, truncated, info = super().step(gated_action)
        
        # 5. デバッグ情報追加
        info["raw_action"] = raw_action
        info["filtered_action"] = filtered_action
        info["gated_action"] = gated_action
        info["filter_multiplier"] = multiplier
        info["gate_passed"] = gate_info["passed"]
        
        return obs, reward, terminated, truncated, info
    
    def _apply_soft_filter(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Soft Filter: 時間帯/レジームに基づくポジション調整
        
        ⚠️ 学習時も推論時も同一ロジック
        """
        target_pos = action[0]
        multiplier = 1.0
        
        # 時間帯制約
        hour_jst = self._get_current_hour_jst()
        time_config = TIME_RESTRICTION_CONFIG.get(hour_jst, {})
        multiplier *= time_config.get("position_mult", 1.0)
        
        # レジーム制約
        regime = self._get_current_regime()
        regime_config = REGIME_CONSTRAINTS.get(regime, {})
        multiplier *= regime_config.get("position_multiplier", 1.0)
        
        # フィルタ適用
        filtered_action = action.copy()
        filtered_action[0] = target_pos * multiplier
        
        return filtered_action, multiplier
    
    def _apply_calibration_gate(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Calibration Gate: EV判定によるエントリー制御
        
        ⚠️ EV < 0 の場合、ポジション変更を抑制（0に近づける）
        
        NOTE: CalibrationGate.calculate_ev() の統一API仕様は §5.2 を参照
        """
        target_pos = action[0]
        current_pos = self.position
        
        # ポジション変更量
        delta_pos = target_pos - current_pos
        
        if abs(delta_pos) < 0.01:  # 無視できる変更
            return action, {"passed": True, "ev": 0.0}
        
        # 現在の市場状態を取得
        direction = "long" if delta_pos > 0 else "short"
        regime = self._get_current_regime()
        
        # 統計情報を取得（CalibrationGateが内部で管理）
        stats = self._calibration_gate.get_regime_stats(regime, direction)
        
        # コスト推定
        cost = self._estimate_trade_cost(abs(delta_pos))
        
        # EV計算（§5.2 の統一APIを使用）
        ev = self._calibration_gate.calculate_ev(stats=stats, cost=cost)
        
        if ev > 0:
            # EV正：許可
            return action, {"passed": True, "ev": ev}
        else:
            # EV負：ポジション変更を縮小
            dampening = max(0.0, 1.0 + ev * 2.0)  # ev=-0.5で完全抑制
            gated_action = action.copy()
            gated_action[0] = current_pos + delta_pos * dampening
            return gated_action, {"passed": False, "ev": ev, "dampening": dampening}
```

#### 4.3.2. 重要: ライブ実行時の一貫性
```python
# ❌ 間違った実装: ライブ時のみ外部でフィルタリング
def live_inference_WRONG(model, obs, market_data):
    raw_action = model.predict(obs)
    filtered_action = external_soft_filter(raw_action, market_data)  # ❌
    return filtered_action

# ✅ 正しい実装: 環境を介して統一フィルタリング
def live_inference_CORRECT(env, model, obs):
    raw_action = model.predict(obs)
    # env.step()内部でフィルタリング → train-live parity保証
    obs, reward, done, truncated, info = env.step(raw_action)
    actual_action = info["gated_action"]  # 実際に実行されたアクション
    return actual_action
```

---

## 5. 統合シグナルシステム仕様

### 5.1. アーキテクチャ
```
Input Layer:
├── RL Agent Output (action, confidence)
├── Technical Signals (ichimoku, adx, ...)
└── Regime Context

Signal Fusion:
├── Filter Mode: AND条件でエントリー許可
└── Boost Mode: シグナル一致でサイズ拡大

Calibration Gate:
├── EV計算: EV = p_win * avg_win - (1-p_win) * avg_loss - cost
└── Entry Decision: EV > 0 のみ許可
```

### 5.2. EV計算式（CalibrationGate）- 統一API

> **⚠️ 第2次レビュー対応**: §4.3と§5.2でAPIが混在していた問題を修正
> 統一API: `calculate_ev(stats: Dict, cost: float) -> float`

```python
class CalibrationGate:
    """
    EV（期待値）ベースのエントリー制御ゲート
    
    統一API設計:
    - calculate_ev(stats, cost): 期待値計算（外部から統計+コストを渡す）
    - get_regime_stats(regime, direction): レジーム別統計取得（内部状態管理）
    """
    
    def __init__(self, regime_stats_path: str = None):
        """
        Args:
            regime_stats_path: レジーム別統計ファイル（backtest結果から生成）
        """
        self._regime_stats = self._load_or_init_stats(regime_stats_path)
    
    def get_regime_stats(self, regime: str, direction: str) -> Dict:
        """
        レジーム・方向別の統計情報を取得
        
        Args:
            regime: "trending_up", "ranging", etc.
            direction: "long" or "short"
        
        Returns:
            stats: {p_win_lcb, avg_win, avg_loss, n_eff}
        """
        key = f"{regime}_{direction}"
        if key in self._regime_stats:
            return self._regime_stats[key]
        return self._default_stats()
    
    def calculate_ev(self, stats: Dict, cost: float) -> float:
        """
        期待値計算（統一API）
        
        Args:
            stats: {
                p_win_lcb: float,  # 勝率の下側信頼区間（保守的推定）
                avg_win: float,    # 平均勝ち幅
                avg_loss: float,   # 平均負け幅（正の値）
                n_eff: int,        # 有効サンプル数
            }
            cost: 推定取引コスト（spread + fee + slippage）
        
        Returns:
            ev: 期待値（正なら有利、負なら不利）
        
        Note:
            §4.3 の _apply_calibration_gate() はこのメソッドを呼び出す。
            stats は get_regime_stats() で取得、cost は env._estimate_trade_cost() で計算。
        """
        p_win = stats["p_win_lcb"]  # Lower Confidence Bound使用（保守的）
        avg_win = stats["avg_win"]
        avg_loss = stats["avg_loss"]
        
        ev = p_win * avg_win - (1 - p_win) * avg_loss - cost
        
        return ev
```

### 5.3. コストモデル
```python
def estimate_cost(market_data: Dict, order_size: float) -> float:
    """
    取引コスト推定
    
    Cost = Fee + Spread + Volatility Risk + Impact
    """
    price = market_data["close"]
    spread_ratio = market_data.get("spread_ratio", 0.001)
    atr = market_data["atr"]
    volume = market_data["volume"]
    
    # Components
    fee = price * order_size * FEE_RATE  # 0.1%
    spread_cost = price * order_size * spread_ratio * C_SPREAD
    vol_cost = atr * order_size * C_VOL
    impact_cost = (order_size ** 2 / (volume + 1e-8)) * C_IMP * GAMMA
    
    total_cost = fee + spread_cost + vol_cost + impact_cost
    
    return total_cost
```

---

## 6. GRU Policy Network仕様

### ⚠️ 6.0. CRITICAL: GRU導入の条件と注意事項

> **外部レビュー指摘**: GRU + Off-Policy SAC は非自明な組み合わせ
> - シーケンスリプレイの設計
> - Hidden state burn-in
> - エピソード境界の処理
> これらが未詳細化のまま実装すると性能劣化のリスクあり

#### 6.0.1. 推奨アプローチ: 段階的導入
```
Phase 1 (Required): 非GRUベースラインの確立
   └── MLPベースのSACで MTF + Time + Global 特徴量を検証
   └── 目標: Sharpe > 0.3, Return > 0%

Phase 2 (Optional): GRU導入の検討
   └── Phase 1成功後のみ着手
   └── シーケンスリプレイ設計を完了してから実装
```

#### 6.0.2. GRU導入時の必須設計項目
```python
GRU_DESIGN_CHECKLIST = {
    "sequence_replay": {
        "description": "リプレイバッファからのシーケンス取得方法",
        "options": [
            "stored_hidden",       # Hiddenを保存（メモリ大）
            "burn_in",             # 先頭N stepでhidden再計算
            "no_recurrence",       # Off-policy学習時は非RNN（on-policyのみRNN）
        ],
        "recommended": "burn_in",
        "status": "DESIGN_REQUIRED",
    },
    "episode_boundary": {
        "description": "エピソード境界でのhidden処理",
        "options": [
            "reset_to_zero",       # 0にリセット
            "carry_over",          # 次エピソードに持ち越し
            "learned_init",        # 学習可能な初期値
        ],
        "recommended": "reset_to_zero",
        "status": "DESIGN_REQUIRED",
    },
    "temporal_correlation": {
        "description": "連続ステップの相関対策",
        "options": [
            "sequence_shuffling",  # シーケンス単位でシャッフル
            "subsequence_overlap", # オーバーラップサブシーケンス
        ],
        "recommended": "sequence_shuffling",
        "status": "DESIGN_REQUIRED",
    },
}
```

### 6.1. アーキテクチャ
```python
class GRUPolicyNetwork(nn.Module):
    """
    GRU-Enhanced Policy Network for Contextual RL
    
    ⚠️ NOTE: Phase 2での導入を推奨。Phase 1では標準MLPを使用。
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # GRU for temporal context
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output: mean and log_std for continuous action
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Log std bounds
        self.log_std_min = -20.0
        self.log_std_max = 2.0
    
    def forward(
        self, 
        obs_sequence: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs_sequence: [batch, seq_len, obs_dim]
            hidden: [num_layers, batch, gru_hidden_dim]
        
        Returns:
            mean: [batch, action_dim]
            log_std: [batch, action_dim]
            new_hidden: [num_layers, batch, gru_hidden_dim]
        """
        # Input projection
        x = F.relu(self.input_proj(obs_sequence))
        
        # GRU
        gru_out, new_hidden = self.gru(x, hidden)
        
        # Use last timestep output
        last_out = gru_out[:, -1, :]
        
        # Policy network
        policy_features = self.policy_net(last_out)
        
        # Action distribution parameters
        mean = self.mean_head(policy_features)
        log_std = self.log_std_head(policy_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, new_hidden
```

### 6.2. ⚠️ シーケンスリプレイ設計（Burn-in方式）
```python
class SequenceReplayBuffer:
    """
    ⚠️ CRITICAL: GRU用シーケンスリプレイバッファ
    
    Burn-in方式: 先頭burn_in_lengthステップでhiddenを再計算
    """
    
    def __init__(
        self,
        capacity: int,
        sequence_length: int = 60,
        burn_in_length: int = 20,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        # エピソード単位で保存
        self.episodes = []
        self.episode_starts = []
    
    def sample_sequences(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        シーケンスをサンプリング
        
        Returns:
            sequences: [batch, seq_len, obs_dim]
            actions: [batch, seq_len, action_dim]
            rewards: [batch, seq_len]
        """
        sequences = []
        
        for _ in range(batch_size):
            # ランダムエピソードから開始位置を選択
            ep_idx = np.random.randint(len(self.episodes))
            episode = self.episodes[ep_idx]
            
            max_start = len(episode) - self.sequence_length
            if max_start <= 0:
                start = 0
            else:
                start = np.random.randint(max_start)
            
            seq = episode[start:start + self.sequence_length]
            sequences.append(seq)
        
        return self._collate(sequences)
    
    def compute_burn_in_hidden(
        self, gru_network: nn.Module, sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Burn-in: 先頭ステップでhiddenを計算（勾配なし）
        """
        burn_in_seq = sequence[:, :self.burn_in_length, :]
        
        with torch.no_grad():
            _, _, hidden = gru_network(burn_in_seq, hidden=None)
        
        return hidden.detach()
```

### 6.3. 学習時の設定
```python
GRU_TRAINING_CONFIG = {
    "sequence_length": 60,          # 60 steps = 1 hour context
    "burn_in_length": 20,           # ⚠️ NEW: Hidden state warmup
    "hidden_dim": 256,
    "gru_hidden_dim": 128,
    "gru_layers": 2,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "gradient_clip": 1.0,
    
    # ⚠️ Phase 2での導入条件
    "prerequisite": "Phase 1 (MLP baseline) must achieve Sharpe > 0.3",
}
```

---

## 7. Curriculum Learning仕様

### 7.1. ステージ定義
```python
CURRICULUM_STAGES = {
    "stage_1": {
        "name": "trend_following",
        "description": "強いトレンド相場のみで学習",
        "regime_filter": ["strong_bull_trend", "strong_bear_trend"],
        "duration_steps": 100_000,
        "success_criteria": {"min_return": 0.0, "min_trades": 50},
    },
    "stage_2": {
        "name": "ranging_addition",
        "description": "レンジ相場を追加",
        "regime_filter": ["strong_bull_trend", "strong_bear_trend", "ranging", "sideways"],
        "duration_steps": 100_000,
        "success_criteria": {"min_return": -0.02, "min_trades": 100},
    },
    "stage_3": {
        "name": "volatility_stress",
        "description": "高ボラティリティ環境を追加",
        "regime_filter": "all_except_extreme",
        "duration_steps": 100_000,
        "success_criteria": {"min_return": -0.05, "max_drawdown": 0.15},
    },
    "stage_4": {
        "name": "full_market",
        "description": "全レジーム混合",
        "regime_filter": "all",
        "duration_steps": 200_000,
        "success_criteria": {"min_return": 0.02, "sharpe_ratio": 0.5},
    },
}
```

### 7.2. 進行判定ロジック
```python
def should_advance_stage(current_stage: str, metrics: Dict) -> bool:
    """
    ステージ進行判定
    """
    criteria = CURRICULUM_STAGES[current_stage]["success_criteria"]
    
    # 全ての成功基準を満たしているか
    for key, threshold in criteria.items():
        if key.startswith("min_"):
            metric_name = key[4:]
            if metrics.get(metric_name, -np.inf) < threshold:
                return False
        elif key.startswith("max_"):
            metric_name = key[4:]
            if metrics.get(metric_name, np.inf) > threshold:
                return False
    
    return True
```

---

## 8. バックテスト仕様

### 8.1. テストデータ分割
```
全データ期間: 2024-01-01 ~ 2025-12-31 (2年間)

Training Set:   2024-01-01 ~ 2025-06-30 (18ヶ月)
Validation Set: 2025-07-01 ~ 2025-09-30 (3ヶ月)
Test Set:       2025-10-01 ~ 2025-12-31 (3ヶ月)
```

### 8.2. 評価指標
```python
EVALUATION_METRICS = {
    "primary": {
        "total_return": "累積リターン",
        "sharpe_ratio": "シャープレシオ（年率）",
        "profit_factor": "プロフィットファクター",
    },
    "secondary": {
        "win_rate": "勝率",
        "max_drawdown": "最大ドローダウン",
        "avg_trade_duration": "平均保持時間",
        "trade_count": "トレード数",
    },
    "risk": {
        "var_95": "95% Value at Risk",
        "expected_shortfall": "条件付きVaR",
        "calmar_ratio": "カルマーレシオ",
    },
}
```

### 8.3. Walk-Forward Analysis
```python
WALK_FORWARD_CONFIG = {
    "train_window_days": 90,        # 学習ウィンドウ: 90日
    "test_window_days": 30,         # テストウィンドウ: 30日
    "step_days": 30,                # スライド幅: 30日
    "min_train_samples": 10_000,    # 最小学習サンプル数
}
```

---

## 9. フェイルセーフ機構

### 9.1. リスク管理パラメータ
```python
RISK_LIMITS = {
    "max_position_btc": 0.01,       # 最大ポジション: 0.01 BTC
    "max_daily_loss_pct": 0.05,     # 日次最大損失: 5%
    "max_drawdown_pct": 0.10,       # 最大ドローダウン: 10%
    "max_consecutive_losses": 10,   # 連続損失上限
    "cooldown_after_loss": 5,       # 損失後クールダウン（分）
}
```

### 9.2. サーキットブレーカー
```python
def check_circuit_breaker(state: Dict) -> bool:
    """
    サーキットブレーカー判定
    
    Returns:
        True: 取引停止
        False: 取引継続
    """
    # 日次損失チェック
    if state["daily_loss_pct"] > RISK_LIMITS["max_daily_loss_pct"]:
        logger.warning("Circuit breaker: Daily loss limit exceeded")
        return True
    
    # ドローダウンチェック
    if state["current_drawdown"] > RISK_LIMITS["max_drawdown_pct"]:
        logger.warning("Circuit breaker: Max drawdown exceeded")
        return True
    
    # 連続損失チェック
    if state["consecutive_losses"] >= RISK_LIMITS["max_consecutive_losses"]:
        logger.warning("Circuit breaker: Consecutive loss limit exceeded")
        return True
    
    return False
```

---

## 10. 統合テスト要件

### 10.1. 単体テスト
- [ ] `test_mtf_features.py`: MTF特徴量生成テスト
- [ ] `test_cyclical_time.py`: 時刻エンコーディングテスト
- [ ] `test_reward_v456.py`: 新報酬関数テスト
- [ ] `test_soft_filter.py`: Soft Filterテスト
- [ ] `test_calibration_gate.py`: EV判定テスト（既存拡張）

### 10.2. 統合テスト
- [ ] `test_env_v456.py`: 拡張環境テスト
- [ ] `test_training_pipeline.py`: 学習パイプラインテスト
- [ ] `test_curriculum.py`: カリキュラム進行テスト

### 10.3. システムテスト
- [ ] `test_backtest_full.py`: フルバックテスト
- [ ] `test_paper_trading.py`: ペーパートレード検証
- [ ] `test_failsafe.py`: フェイルセーフ動作確認

---

## Appendix A: 数式まとめ

### A.1. MTFトレンド方向判定
```
trend_5m = sign(EMA_5m[t] - EMA_5m[t-1]) * I(|EMA_slope| > threshold)
```

### A.2. Z-Score計算
```
Z = (price - MA(n)) / σ(n)
```
where n = 20 (default)

### A.3. EV計算
```
EV = P(win) × E[win] - P(loss) × E[loss] - Cost
```

### A.4. Cyclical Time Encoding
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

---

## Appendix B: 設定ファイルテンプレート

```json
{
    "version": "v456",
    "environment": {
        "initial_balance": 100000,
        "max_position": 0.01,
        "commission_rate": 0.001,
        "max_ttl_steps": 60,
        "drawdown_limit": 0.10
    },
    "reward": {
        "alpha": 0.2,
        "beta": 0.01,
        "gamma": 0.5,
        "min_edge_mult": 1.5,
        "vol_floor": 0.002,
        "mtf_alignment_weight": 0.15,
        "balance_enforcement": true
    },
    "features": {
        "mtf_timeframes": ["5m", "15m", "1h"],
        "cyclical_time": true,
        "global_market": true,
        "regime_features": true
    },
    "model": {
        "type": "SAC",
        "policy_type": "GRU",
        "hidden_dim": 256,
        "gru_layers": 2,
        "sequence_length": 60
    },
    "training": {
        "total_timesteps": 500000,
        "curriculum_enabled": true,
        "batch_size": 256,
        "learning_rate": 0.0003
    }
}
```
