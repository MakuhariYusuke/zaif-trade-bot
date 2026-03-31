# 679# SAC 報酬・γ・学習率の根本修正 + 676# セルフレビュー

## 概要
675# で特定した「SAC 実質機能停止」の根本原因を、金融工学・情報理論・制御理論の観点からセルフレビューし、676# の不足点を修正。

## セルフレビュー: 676# の評価

### 676# で正しかった点
- `confidence_roi_full 0.005→0.002`: confidence 回復の方向性は正しい
- `min_trade_count 3→50`: HOLD 収束モデルの deploy 防止
- `min_profit_factor 0.8`: PF gate 追加

### 676# で不足していた点（本 679# で修正）

#### 1. 情報理論的観点 — 報酬信号の entropy 汚染
`use_simple_reward: False` (g2_sac_train.yaml 未指定 → デフォルト使用):
- complex reward = PnL + 12段 curriculum penalties → I(R; PnL) を希釈
- SAC の entropy regularization H(π) と合わさり Q 関数が PnL を学習不能
- Shannon 分解: R = PnL + noise_penalties → I(R; optimal_action) ≪ H(R)
- **修正**: `use_simple_reward: true` を g2_sac_train.yaml に追加

#### 2. 制御理論的観点 — γ=0.80 の破壊的影響
γ=0.80 → effective horizon = 1/(1-γ) = **5 steps = 5分**：
- BTC の mean reversion halflife は 30-60分
- system bandwidth ≈ 0.003 Hz → 市場の支配的周波数 ~0.0003 Hz をカットオフ
- **修正**: γ=0.80 → 0.95 (effective horizon = 20 steps = 20分)
- v459 reward_clean.yaml で γ=0.95 が最良結果を出した実績あり

#### 3. Gradient steps の不足
- gradient_steps=1: 25K transitions に対して 25K gradient updates
- SAC は off-policy → sample reuse が可能
- **修正**: gradient_steps 1→2 (実質 50K gradient updates)

#### 4. dead_zone allowlist 漏れ
676# で `dead_zone: 0.10→0.05` に変更したが、YAML↔Code drift prevention テスト (test_336) の `KNOWN_YAML_OVERRIDES` に追加し忘れ → テスト失敗。本 679# で修正。

#### 5. profit_factor gate の順序分析
trade_count gate → PF gate の順序は意図的だが、SAC が HOLD 収束 (trades=0) している現状では PF gate は dead code。trade_count gate が先にリジェクトするため。本質的問題は SAC が取引しないことであり、P0-1/P0-4/P0-5 で根本解決を図る。

## 変更内容

### configs/v460/experiments/g2_sac_train.yaml
| パラメータ | 旧値 | 新値 | 根拠 |
|-----------|------|------|------|
| environment.use_simple_reward | (未設定=False) | true | 情報理論的: reward-PnL MI 最大化 |
| sac_hyperparameters.gamma | 0.80 | 0.95 | 制御理論的: effective horizon 5→20分 |
| sac_hyperparameters.gradient_steps | 1 | 2 | SGD convergence: 50K effective updates |

### scripts/v460/ml/sac_retrain_scheduler.py
- `SACRetrainConfig` に `use_simple_reward: bool` / `reward_scaling: float` フィールド追加
- `from_yaml_dict()` で environment.use_simple_reward / reward_scaling を読み込み
- `_create_env()` で `RewardSettings(use_simple_reward=..., reward_scaling=...)` を環境に注入

### tests/unit/v460/test_336_yaml_code_drift_prevention.py
- `KNOWN_YAML_OVERRIDES` に `sidecar_dead_zone` を追加 (676# 漏れ修正)

### tests/unit/v460/test_356_g2_sac_blockers.py
- gamma アサーション 0.80→0.95、gradient_steps アサーション追加

## 理論的補足

### Avellaneda-Stoikov フレームワークとの整合
simple reward (PnL 直結) は A-S の utility maximization `E[PnL] - γ Var[PnL]` に近い。
complex reward の ad-hoc penalty 群は A-S の inventory risk-aversion パラメータ γσ²τ と無関係であり、最適 reservation price の学習を妨害する。

### 一目均衡表との接続
17特徴量のうち `ema_velocity_bps` は一目均衡表の「転換線-基準線乖離」に相当する momentum 指標。
γ=0.95 への変更は、一目均衡表の「先行スパン」（26期間先行 = 26分）に近い時間視野を SAC に与えることに相当。
γ=0.80 では「転換線」（9期間 = 9分）の半分以下の視野しかなく、均衡表の情報を活用不能だった。

### Kelly Criterion の観点
SAC の position sizing が Kelly 基準 `f* = μ/σ²` に収束するためには、十分な horizon でリターン分布を推定できる必要がある。
γ=0.80 での 5分 horizon は、μ の推定精度が σ に対して不十分で、Kelly が f*≈0 (HOLD) に退化する。

## 再起動タイミング
本変更は retrain_scheduler が g2_sac_train.yaml を起動時に読み込むため、再起動が必要。
fill_test への直接影響はなし（sidecar signal は retrain 成功後に更新される）。
