"""
マルチモーダル学習 - 統合アーキテクチャ
価格データ、テキスト感情、経済指標を統合した取引AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiModalFeatureEncoder(nn.Module):
    """
    複数モダリティの特徴量を統合エンコードするクラス
    """

    def __init__(self,
                 price_feature_dim: int = 156,
                 text_embedding_dim: int = 768,
                 economic_feature_dim: int = 10,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()

        self.price_feature_dim = price_feature_dim
        self.text_embedding_dim = text_embedding_dim
        self.economic_feature_dim = economic_feature_dim
        self.hidden_dim = hidden_dim

        # 各モダリティのエンコーダー
        self.price_encoder = nn.Sequential(
            nn.Linear(price_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.economic_encoder = nn.Sequential(
            nn.Linear(economic_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # クロスモーダル・アテンション
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # combined_contextの次元に合わせる
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # アテンション出力の投影層
        self.attention_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # モダリティ融合層
        self.modality_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 時間的依存関係モデリング（LSTM）
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,  # 明示的にbatch_first=True
            bidirectional=True
        )

        # 最終出力層
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self,
                price_features: torch.Tensor,
                text_embeddings: torch.Tensor,
                economic_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        複数モダリティの特徴量を統合

        Args:
            price_features: 価格ベース特徴量 [batch, seq_len, price_feature_dim]
            text_embeddings: テキスト埋め込み [batch, seq_len, text_embedding_dim]
            economic_features: 経済指標特徴量 [batch, seq_len, economic_feature_dim]
            attention_mask: アテンションマスク [batch, seq_len]

        Returns:
            統合された特徴量 [batch, seq_len, hidden_dim]
        """

        batch_size, seq_len = price_features.size(0), price_features.size(1)
        logger.info(f"Input shapes: price={price_features.shape}, text={text_embeddings.shape}, econ={economic_features.shape}")

        # 各モダリティのエンコーディング
        price_encoded = self.price_encoder(price_features)  # [batch, seq_len, hidden_dim]
        text_encoded = self.text_encoder(text_embeddings)    # [batch, seq_len, hidden_dim]
        econ_encoded = self.economic_encoder(economic_features)  # [batch, seq_len, hidden_dim]

        logger.info(f"Encoded shapes: price={price_encoded.shape}, text={text_encoded.shape}, econ={econ_encoded.shape}")

        # クロスモーダル・アテンション
        # テキストをクエリとして、価格と経済指標の結合に注目
        combined_context = torch.cat([price_encoded, econ_encoded], dim=-1)  # [batch, seq_len, hidden_dim*2]

        # クエリも同じ次元にするために拡張（テキストを複製）
        query_expanded = torch.cat([text_encoded, text_encoded], dim=-1)  # [batch, seq_len, hidden_dim*2]

        attn_output, attn_weights = self.cross_attention(
            query=query_expanded,
            key=combined_context,
            value=combined_context,
            key_padding_mask=attention_mask
        )  # [batch, seq_len, hidden_dim*2]

        # アテンション出力をhidden_dimに圧縮
        attn_output = self.attention_projection(attn_output)  # [batch, seq_len, hidden_dim]

        # モダリティ融合
        fused_features = torch.cat([
            price_encoded,  # 価格情報
            text_encoded,   # テキスト情報
            attn_output     # クロスモーダル情報
        ], dim=-1)  # [batch, seq_len, hidden_dim*3]

        logger.info(f"fused_features shape before modality_fusion: {fused_features.shape}")
        fused_features = self.modality_fusion(fused_features)  # [batch, seq_len, hidden_dim]
        logger.info(f"fused_features shape after modality_fusion: {fused_features.shape}")

        # 時間的依存関係モデリング
        logger.info(f"fused_features shape before LSTM: {fused_features.shape}")
        temporal_output, _ = self.temporal_encoder(fused_features)  # [batch, seq_len, hidden_dim*2]

        logger.info(f"LSTM output shape: {temporal_output.shape}")
        logger.info(f"LSTM batch_first: {self.temporal_encoder.batch_first}")

        # LSTM出力が正しい形状であることを確認
        if temporal_output.dim() == 3 and temporal_output.shape[0] == batch_size and temporal_output.shape[1] == seq_len:
            # 期待される形状: [batch, seq_len, hidden_dim*2]
            logger.info("LSTM output shape is correct")
        elif temporal_output.dim() == 2:
            # [batch*seq_len, hidden_dim*2] -> [batch, seq_len, hidden_dim*2]
            temporal_output = temporal_output.view(batch_size, seq_len, self.hidden_dim * 2)
            logger.info(f"LSTM output reshaped from 2D to 3D: {temporal_output.shape}")
        elif temporal_output.shape[0] == seq_len and temporal_output.shape[1] == batch_size:
            # batch_first=Falseの場合: [seq_len, batch, hidden_dim*2] -> [batch, seq_len, hidden_dim*2]
            temporal_output = temporal_output.transpose(0, 1)
            logger.info(f"LSTM output transposed: {temporal_output.shape}")
        else:
            logger.warning(f"Unexpected LSTM output shape: {temporal_output.shape}, expected [{batch_size}, {seq_len}, {self.hidden_dim * 2}]")

        # 最終出力
        logger.info(f"Final temporal_output shape: {temporal_output.shape}")
        output = self.output_projection(temporal_output)  # [batch, seq_len, hidden_dim]

        return output

        return output

class MultiModalTradingAgent(nn.Module):
    """
    マルチモーダル取引エージェント
    SACアルゴリズムを拡張したバージョン
    """

    def __init__(self,
                 price_feature_dim: int = 156,
                 text_embedding_dim: int = 768,
                 economic_feature_dim: int = 10,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()

        # マルチモーダル特徴量エンコーダー
        self.feature_encoder = MultiModalFeatureEncoder(
            price_feature_dim=price_feature_dim,
            text_embedding_dim=text_embedding_dim,
            economic_feature_dim=economic_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # SAC Actorネットワーク
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )

        # SAC Criticネットワーク（Twin critics）
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 温度パラメータ（自動エントロピー調整）
        self.log_alpha = torch.tensor(0.0, requires_grad=True)

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def encode_features(self,
                       price_features: torch.Tensor,
                       text_embeddings: torch.Tensor,
                       economic_features: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        マルチモーダル特徴量をエンコード
        """
        return self.feature_encoder(
            price_features, text_embeddings, economic_features, attention_mask
        )

    def get_action(self,
                   state_features: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        行動を選択

        Args:
            state_features: エンコードされた状態特徴量 [batch, hidden_dim]
            deterministic: 確定的行動を選択するか

        Returns:
            行動と対数確率
        """

        # Actorネットワーク
        action_params = self.actor(state_features)  # [batch, action_dim * 2]
        mean, log_std = action_params.chunk(2, dim=-1)

        # 標準偏差を正の値に制限
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        # 行動サンプリング
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()

        # 行動を[-1, 1]に制限
        action = torch.tanh(action)

        # 対数確率計算（tanh変換の補正項を含む）
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_value(self, state_features: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        状態行動価値を評価

        Args:
            state_features: エンコードされた状態特徴量 [batch, hidden_dim]
            action: 行動 [batch, action_dim]

        Returns:
            Q値（2つのCriticの出力）
        """

        sa_features = torch.cat([state_features, action], dim=-1)

        q1 = self.critic1(sa_features)
        q2 = self.critic2(sa_features)

        return q1, q2

class MultiModalDataPreprocessor:
    """
    マルチモーダルデータの前処理クラス
    """

    def __init__(self,
                 sequence_length: int = 60,
                 price_feature_dim: int = 156,
                 economic_feature_dim: int = 10):
        self.sequence_length = sequence_length
        self.price_feature_dim = price_feature_dim
        self.economic_feature_dim = economic_feature_dim

        # 特徴量正規化パラメータ
        self.price_scaler = None
        self.economic_scaler = None

    def preprocess_batch(self,
                        price_data: np.ndarray,
                        news_data: List[Dict],
                        economic_data: np.ndarray,
                        dates: List[datetime]) -> Dict[str, torch.Tensor]:
        """
        バッチデータの前処理

        Args:
            price_data: 価格データ [batch, seq_len, price_features]
            news_data: ニュースデータリスト
            economic_data: 経済指標データ [batch, seq_len, economic_features]
            dates: 日付リスト

        Returns:
            前処理済みデータ辞書
        """

        batch_size = len(price_data)

        # 価格データの正規化
        if self.price_scaler is None:
            # トレーニング時にスケーラーをフィット
            self.price_scaler = self._fit_scaler(price_data.reshape(-1, self.price_feature_dim))

        price_normalized = self._normalize_features(price_data, self.price_scaler)

        # 経済指標データの正規化
        if self.economic_scaler is None:
            self.economic_scaler = self._fit_scaler(economic_data.reshape(-1, self.economic_feature_dim))

        economic_normalized = self._normalize_features(economic_data, self.economic_scaler)

        # ニュースデータの処理
        text_embeddings = self._process_news_data(news_data, batch_size, self.sequence_length)

        # アテンションマスク（ニュースがない部分をマスク）
        attention_mask = self._create_attention_mask(news_data, batch_size, self.sequence_length)

        return {
            'price_features': torch.tensor(price_normalized, dtype=torch.float32),
            'text_embeddings': text_embeddings,
            'economic_features': torch.tensor(economic_normalized, dtype=torch.float32),
            'attention_mask': attention_mask
        }

    def _fit_scaler(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """特徴量スケーラーのフィッティング"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # ゼロ除算防止

        return {'mean': mean, 'std': std}

    def _normalize_features(self, data: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
        """特徴量の正規化"""
        return (data - scaler['mean']) / scaler['std']

    def _process_news_data(self,
                          news_data: List[Dict],
                          batch_size: int,
                          seq_len: int) -> torch.Tensor:
        """
        ニュースデータを埋め込みに変換
        （実際の実装では事前学習済みモデルを使用）
        """
        # 簡易実装：ランダム埋め込み（実際にはBERTなどを使用）
        embeddings = torch.randn(batch_size, seq_len, 768)

        # ニュースがある部分のみ実際の感情スコアを反映
        for b in range(batch_size):
            for t in range(seq_len):
                if t < len(news_data[b]) and news_data[b][t] is not None:
                    sentiment = news_data[b][t].get('sentiment_score', 0.0)
                    # 感情スコアを埋め込みの先頭に反映
                    embeddings[b, t, 0] = sentiment

        return embeddings

    def _create_attention_mask(self,
                             news_data: List[Dict],
                             batch_size: int,
                             seq_len: int) -> torch.Tensor:
        """アテンションマスク作成"""
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        for b in range(batch_size):
            for t in range(seq_len):
                if t >= len(news_data[b]) or news_data[b][t] is None:
                    mask[b, t] = True  # マスク（パディング部分）

        return mask

class MultiModalTrainer:
    """
    マルチモーダル学習のトレーニングクラス
    """

    def __init__(self,
                 agent: MultiModalTradingAgent,
                 preprocessor: MultiModalDataPreprocessor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.agent = agent.to(device)
        self.preprocessor = preprocessor
        self.device = device

        # オプティマイザー
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.agent.critic1.parameters()) + list(self.agent.critic2.parameters()),
            lr=3e-4
        )
        self.alpha_optimizer = torch.optim.Adam([self.agent.log_alpha], lr=3e-4)

        # ターゲットネットワーク
        self.target_agent = MultiModalTradingAgent(
            price_feature_dim=agent.feature_encoder.price_feature_dim,
            text_embedding_dim=agent.feature_encoder.text_embedding_dim,
            economic_feature_dim=agent.feature_encoder.economic_feature_dim,
            action_dim=agent.action_dim,
            hidden_dim=agent.hidden_dim
        ).to(device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        # ハイパーパラメータ
        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -float(agent.action_dim)

    def update(self,
              batch_data: Dict[str, Any],
              rewards: torch.Tensor,
              next_batch_data: Dict[str, Any],
              dones: torch.Tensor) -> Dict[str, float]:
        """
        1ステップの学習更新

        Args:
            batch_data: 現在のバッチデータ
            rewards: 報酬
            next_batch_data: 次のバッチデータ
            dones: エピソード終了フラグ

        Returns:
            損失値辞書
        """

        # データをデバイスに移動
        price_features = batch_data['price_features'].to(self.device)
        text_embeddings = batch_data['text_embeddings'].to(self.device)
        economic_features = batch_data['economic_features'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)

        next_price_features = next_batch_data['price_features'].to(self.device)
        next_text_embeddings = next_batch_data['text_embeddings'].to(self.device)
        next_economic_features = next_batch_data['economic_features'].to(self.device)
        next_attention_mask = next_batch_data['attention_mask'].to(self.device)

        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # 特徴量エンコーディング
        state_features = self.agent.encode_features(
            price_features, text_embeddings, economic_features, attention_mask
        )[:, -1, :]  # 最後のタイムステップを使用

        next_state_features = self.target_agent.encode_features(
            next_price_features, next_text_embeddings, next_economic_features, next_attention_mask
        )[:, -1, :]

        # 行動サンプリング
        actions, log_probs = self.agent.get_action(state_features)

        # Q値計算
        q1, q2 = self.agent.get_value(state_features, actions)
        q_min = torch.min(q1, q2)

        # ターゲットQ値計算
        with torch.no_grad():
            next_actions, next_log_probs = self.target_agent.get_action(next_state_features)
            next_q1, next_q2 = self.target_agent.get_value(next_state_features, next_actions)
            next_q_min = torch.min(next_q1, next_q2)

            alpha = self.agent.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (next_q_min - alpha * next_log_probs)

        # Critic損失
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Actor損失
        actor_loss = (alpha * log_probs - q_min).mean()

        # Alpha損失（自動エントロピー調整）
        alpha_loss = -(self.agent.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # 勾配更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ターゲットネットワーク更新
        self._soft_update_target()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }

    def _soft_update_target(self):
        """ターゲットネットワークのソフト更新"""
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 使用例とテスト
if __name__ == "__main__":
    # モデル初期化
    agent = MultiModalTradingAgent(
        price_feature_dim=156,
        text_embedding_dim=768,
        economic_feature_dim=10,
        action_dim=3,
        hidden_dim=256
    )

    preprocessor = MultiModalDataPreprocessor(
        sequence_length=60,
        price_feature_dim=156,
        economic_feature_dim=10
    )

    trainer = MultiModalTrainer(agent, preprocessor)

    print("マルチモーダル取引エージェント初期化完了")
    print(f"モデルパラメータ数: {sum(p.numel() for p in agent.parameters()):,}")

    # ダミーデータでのテスト
    batch_size = 4
    seq_len = 60

    # ダミー入力データ
    dummy_price = torch.randn(batch_size, seq_len, 156)
    dummy_text = torch.randn(batch_size, seq_len, 768)
    dummy_economic = torch.randn(batch_size, seq_len, 10)
    dummy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # 特徴量エンコーディングテスト
    with torch.no_grad():
        features = agent.encode_features(dummy_price, dummy_text, dummy_economic, dummy_mask)
        print(f"エンコード特徴量形状: {features.shape}")

        # 行動選択テスト
        actions, log_probs = agent.get_action(features[:, -1, :])
        print(f"行動形状: {actions.shape}, 対数確率形状: {log_probs.shape}")

        # Q値評価テスト
        q1, q2 = agent.get_value(features[:, -1, :], actions)
        print(f"Q値形状: q1={q1.shape}, q2={q2.shape}")

    print("""
    🎯 マルチモーダルアーキテクチャの特徴:

    1. クロスモーダル・アテンション
       - テキスト感情をクエリとして価格・経済指標に注目
       - モダリティ間の関連性を学習

    2. 時間的依存関係モデリング
       - 双方向LSTMで長期・短期の依存関係を捕捉
       - シーケンス全体の文脈を理解

    3. 堅牢な特徴量統合
       - 各モダリティのエンコーダーで特徴抽出
       - 融合層で最適な特徴量統合

    4. SACアルゴリズム拡張
       - マルチモーダル状態表現
       - 自動エントロピー調整
       - Twin criticsで安定した学習

    📊 期待効果:
    - ニュース感情による予測精度向上
    - 経済指標のトレンド把握
    - 市場変動時の適応性強化
    - より堅牢な取引戦略

    🔧 実装上の考慮点:
    - データ同期（価格・ニュース・経済指標の時間整合）
    - 特徴量正規化とスケーリング
    - 計算コストの最適化
    - 過学習防止のための正則化
    """)