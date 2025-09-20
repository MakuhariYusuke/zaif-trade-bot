# Heavy Trading Environment for Reinforcement Learning
# 重特徴量ベースの取引環境

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class HeavyTradingEnv(gym.Env):
    """
    重特徴量ベースの取引環境

    特徴:
    - 状態: 価格系・テクニカル系・リスク系のすべての特徴量を使用
    - 行動: 0=hold, 1=buy, 2=sell
    - リワード: (position * pnl) / (atr_14 + 1e-6) - リスク調整型
    - position: -1 (short) / 0 (flat) / 1 (long)
    - NaN処理: ゼロ埋め
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # デフォルト設定
        self.config = config or {
            'reward_scaling': 1.0,
            'transaction_cost': 0.001,  # 0.1%
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        # データの前処理
        self.df = self._preprocess_data(df)
        self.n_steps = len(self.df)

        # 特徴量の選択（除外する列を指定）
        exclude_cols = ['ts', 'exchange', 'pair', 'episode_id']
        self.features = [c for c in self.df.columns if c not in exclude_cols]

        # 状態空間: 特徴量ベクトル
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32
        )

        # 行動空間: hold, buy, sell
        self.action_space = spaces.Discrete(3)

        # 環境状態
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0

        # 報酬計算用の履歴
        self.reward_history = []
        self.position_history = []

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの前処理"""
        # NaNをゼロで埋める
        df_processed = df.fillna(0).copy()

        # インデックスをリセット
        df_processed = df_processed.reset_index(drop=True)

        # 数値列のみをfloat32に変換（メモリ効率のため）
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].astype(np.float32)

        return df_processed

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """環境のリセット"""
        super().reset(seed=seed)

        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.reward_history = []
        self.position_history = []

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """ステップ実行"""
        # 行動の実行
        old_position = self.position
        self._execute_action(action)

        # PnLの計算
        pnl = self._calculate_pnl()

        # リスク調整リワードの計算
        reward = self._calculate_reward(pnl, old_position)

        # 次のステップへ
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # 次の状態
        if done:
            next_obs = self._get_observation()  # 最後の状態を維持
        else:
            next_obs = self._get_observation()

        # 情報
        info = self._get_info()
        info.update({
            'pnl': pnl,
            'position': self.position,
            'action': action,
            'step': self.current_step
        })

        # 報酬履歴の更新
        self.reward_history.append(reward)
        self.position_history.append(self.position)

        return next_obs, reward, done, False, info

    def _execute_action(self, action: int) -> None:
        """行動の実行"""
        if action == 0:  # HOLD
            pass  # ポジション維持
        elif action == 1:  # BUY
            if self.position <= 0:  # ショートまたはフラットの場合
                # ポジション変更
                if self.position < 0:  # ショートクローズ
                    self._close_position()
                self._open_position(1)
        elif action == 2:  # SELL
            if self.position >= 0:  # ロングまたはフラットの場合
                # ポジション変更
                if self.position > 0:  # ロングクローズ
                    self._close_position()
                self._open_position(-1)

    def _open_position(self, direction: int) -> None:
        """ポジションオープン"""
        current_price = self.df.loc[self.current_step, 'price']
        self.position = direction * self.config['max_position_size']
        self.entry_price = current_price
        self.trades_count += 1

    def _close_position(self) -> None:
        """ポジションクローズ"""
        if self.position != 0:
            self.trades_count += 1
            self.position = 0
            self.entry_price = 0.0

    def _calculate_pnl(self) -> float:
        """PnLの計算"""
        if self.position == 0:
            return 0.0

        current_price = self.df.loc[self.current_step, 'price']
        price_change = current_price - self.entry_price

        # 基本PnL
        pnl = self.position * price_change

        # 取引コストの考慮（エントリー時のみ）
        if abs(self.position) > 0:
            transaction_cost = abs(self.position) * self.entry_price * self.config['transaction_cost']
            pnl -= transaction_cost

        return pnl

    def _calculate_reward(self, pnl: float, old_position: int) -> float:
        """攻め型リスク調整リワードの計算"""
        # ボラティリティ（ATR）を取得
        atr = 1.0

        # 取引コストとスプレッドの計算
        transaction_cost = 0.0
        spread_cost = 0.0

        if abs(self.position) > 0 and old_position == 0:  # 新規ポジションの場合
            try:
                current_price = 1.0  # デフォルト値
                transaction_cost = abs(self.position) * current_price * self.config['transaction_cost']
                # スプレッドを価格の0.05%として仮定
                spread_cost = abs(self.position) * current_price * 0.0005
            except (ValueError, TypeError):
                # 価格が取得できない場合はコストを0とする
                transaction_cost = 0.0
                spread_cost = 0.0

        # 攻め型リワード: (PnL - 手数料 - スプレッド) / ATR
        net_pnl = pnl - transaction_cost - spread_cost
        base_reward = net_pnl / (atr + 1e-6)

        # ドローダウンペナルティ（50%超えの場合）
        drawdown_penalty = self._calculate_drawdown_penalty()

        # 連勝ボーナス
        win_streak_bonus = self._calculate_win_streak_bonus()

        # 最終リワード
        reward = base_reward - drawdown_penalty + win_streak_bonus

        # リワードスケーリング
        reward *= self.config['reward_scaling']

        return reward

    def _calculate_drawdown_penalty(self) -> float:
        """ドローダウンペナルティの計算（50%超えの場合）"""
        if len(self.reward_history) < 20:  # より長い期間でチェック
            return 0.0

        # 最近20ステップの累積リワード
        recent_rewards = self.reward_history[-20:]
        cumulative_reward = sum(recent_rewards)

        # 基準となる初期累積リワード（最初の10ステップ）
        if len(self.reward_history) >= 30:
            initial_rewards = self.reward_history[-30:-20]
            initial_cumulative = sum(initial_rewards)

            # ドローダウンが50%超えた場合のみペナルティ
            if initial_cumulative > 0:
                drawdown_ratio = (initial_cumulative - cumulative_reward) / initial_cumulative
                if drawdown_ratio > 0.5:  # 50%超え
                    return drawdown_ratio * 0.05  # 軽めのペナルティ（5%）

        return 0.0

    def _calculate_win_streak_bonus(self) -> float:
        """連勝ボーナスの計算"""
        if len(self.reward_history) < 5:
            return 0.0

        # 最近5ステップの勝ち数をカウント
        recent_rewards = self.reward_history[-5:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        # 3勝以上でボーナス
        if win_count >= 3:
            bonus = win_count * 0.01  # 1% per win
            return bonus

        return 0.0

    def _get_observation(self) -> np.ndarray:
        """現在の状態を取得"""
        if self.current_step >= self.n_steps:
            # 最後のステップの場合は最後のデータを返す
            step_data = self.df.iloc[-1]
        else:
            step_data = self.df.iloc[self.current_step]

        # 特徴量ベクトルの作成
        obs = step_data[self.features].values.astype(np.float32)

        return obs

    def _get_info(self) -> Dict:
        """追加情報を取得"""
        return {
            'current_step': self.current_step,
            'total_steps': self.n_steps,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'trades_count': self.trades_count,
            'features': self.features,
            'config': self.config
        }

    def render(self, mode: str = 'human') -> None:
        """環境の描画"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.n_steps}")
            print(f"Position: {self.position}")
            print(f"Total PnL: {self.total_pnl:.4f}")
            print(f"Trades: {self.trades_count}")
            if len(self.reward_history) > 0:
                print(f"Last Reward: {self.reward_history[-1]:.6f}")
            print("-" * 40)

    def close(self) -> None:
        """環境のクリーンアップ"""
        pass

    # ユーティリティメソッド
    def get_feature_names(self) -> list:
        """特徴量名を取得"""
        return self.features

    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        if len(self.reward_history) == 0:
            return {}

        rewards = np.array(self.reward_history)

        return {
            'total_reward': np.sum(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_trades': self.trades_count,
            'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-6),
            'win_rate': np.sum(rewards > 0) / len(rewards) if len(rewards) > 0 else 0
        }