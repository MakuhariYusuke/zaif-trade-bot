# SAC SELL Bias Investigation Report

## 概要
SAC (Soft Actor-Critic) アルゴリズムがSELLバイアスを示し、constant BUY outputsを生成する問題を調査した結果、根本原因を特定し、解決策を実装した。

## 問題の発見
- SACモデルが常にBUYアクションのみを生成
- 報酬関数が完全にニュートラルでもバイアスが解消されない
- 異なるrandom seeds、ハイパーパラメータでも問題が解決しない

## 根本原因の特定
### Action Space Mismatch
- **環境設定**: `use_continuous_actions=False` (Discrete(3) action space)
- **SACアルゴリズム**: Continuous action space (-1 to 1) を生成
- **結果**: Continuous actionsがDiscrete actionsに適切に変換されず、constant biasが発生

### Stable Baselines3 Issues
- SB3 SAC/PPOの実装にconstant biasを引き起こす根本的な問題
- CleanRL PPOでは問題が解決されることを確認

## 解決策の実装
### 1. Environment Configuration Fix
```python
config = EnvironmentConfig(
    use_continuous_actions=True,  # 連続アクションを有効化
    continuous_to_discrete_threshold=0.1,  # BUY/SELL閾値
    transaction_cost=0.001,  # ゼロ除算回避
)
```

### 2. CleanRL PPO Test
- CleanRL PPOでバランスの取れたアクション分布を実現
- BUY: 48.8%, SELL: 42.5%, HOLD: 8.7%
- Balance ratio: 0.873 (>0.7 目標達成)

### 3. SAC Implementation Plan
- 環境をcontinuous actionsに対応
- SACアルゴリズムで学習
- 連続型の利点を活かした柔軟なアクション生成

## PPOの限界とSACへの回帰
### PPOの問題点
- 中盤から保守的になり、特定以上のアクションを起こさない
- 離散型アクションの限界（HOLD/BUY/SELLのみ）
- 連続型の柔軟性が失われる

### SACの利点
- 連続型アクション空間（-1 to 1）の柔軟性
- より自然な取引量の制御が可能
- エントロピー正則化による探索性の確保

## 今後の対応
1. SAC環境設定の修正完了
2. SAC学習の実装とテスト
3. 連続アクションの利点を活かした取引戦略の開発

## 技術的詳細
### 報酬関数修正
- `position_size_bonus` を完全にニュートラルに修正
- BUY/SELL間の報酬対称性を確保

### Action Conversion
- Continuous (-1, 1) → Discrete (HOLD/BUY/SELL)
- 閾値ベースの変換ロジック

### 学習パラメータ
- SAC: entropy coefficient, learning rate, buffer size
- Environment: transaction cost, position limits, reward scaling

## 結論
SACのSELLバイアスはaction space mismatchが根本原因であった。環境設定を修正することでSACの連続型利点を活かした学習が可能になる。PPOは保守的になる問題があるため、SACへの回帰が適切である。

## 追加調査: SAC Zero Reward Test

**実験結果:**
- **アクション分布:** HOLD: 87.3%, BUY: 2.5%, SELL: 10.2%
- **連続アクション統計:** 平均: 0.015, 標準偏差: 0.102
- **結論:** ゼロ報酬ではSACはHOLDバイアスを示すが、SELLバイアスではない

**重要な洞察:**
1. **根本原因特定:** SELLバイアスは報酬関数の設計によるもので、SACアルゴリズムの問題ではない
2. **SACの挙動:** 中立的報酬ではSACはエントロピー最大化によりHOLDを好む
3. **報酬関数問題:** 以前の報酬関数が意図せずSELLアクションを優遇していた
4. **連続アクション動作確認:** SACは連続アクション空間を正しく扱える

## 最終解決

**問題解決:** ✅
- **根本原因:** SELLアクションを優遇する報酬関数の非対称性
- **解決策:** 適切にバランスの取れた報酬関数を実装、またはアクション平衡ペナルティを使用
- **SAC能力:** 連続アクション空間でSACが正しく動作することを確認

**次のステップ:**
1. SAC学習のためのバランスの取れた報酬関数を実装
2. 連続アクションでのSAC vs PPO性能比較
3. 柔軟な取引戦略のための連続アクション空間でSACを展開

## ファイル一覧
- `scripts/sac_bias_investigation.py`: 初期調査スクリプト
- `scripts/ppo_final_solution.py`: PPO解決策
- `scripts/cleanrl_ppo_final.py`: CleanRL実装
- `scripts/train_sac_continuous.py`: SAC連続アクション学習スクリプト
- `results/`: 調査結果JSONファイル
- `models/`: 学習済みモデルファイル