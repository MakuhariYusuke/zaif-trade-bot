"""
Tests for StrictMaskedPolicy

厳密なアクションマスク適用が学習時と評価時の両方で正しく動作することを検証。
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from ztb.training.policies.strict_masked_policy import StrictMaskedPolicy


@pytest.fixture
def simple_observation_space():
    """シンプルな観測空間（10次元の連続空間）"""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)


@pytest.fixture
def simple_action_space():
    """シンプルな行動空間（3つの離散行動: HOLD, BUY, SELL）"""
    return spaces.Discrete(3)


@pytest.fixture
def policy(simple_observation_space, simple_action_space):
    """StrictMaskedPolicy インスタンス"""
    lr_schedule = lambda _: 3e-4  # noqa: E731
    return StrictMaskedPolicy(
        observation_space=simple_observation_space,
        action_space=simple_action_space,
        lr_schedule=lr_schedule,
    )


class TestStrictMaskedPolicyInit:
    """StrictMaskedPolicy の初期化テスト"""

    def test_initialization(self, policy):
        """ポリシーが正しく初期化されることを確認"""
        assert policy is not None
        assert hasattr(policy, "action_net")
        assert hasattr(policy, "value_net")
        assert hasattr(policy, "mlp_extractor")
        assert hasattr(policy, "features_extractor")

    def test_policy_structure(self, policy):
        """ポリシーのネットワーク構造を確認"""
        # action_net は最終的に3つの出力（HOLD, BUY, SELL）を持つべき
        sample_obs = torch.randn(1, 10)
        features = policy.extract_features(sample_obs, policy.features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
        
        assert logits.shape == (1, 3), "Logits should have shape [batch_size, 3]"


class TestStrictMaskedPolicyForward:
    """forward() メソッドのテスト"""

    def test_forward_without_mask(self, policy):
        """マスクなしでforwardが動作することを確認"""
        obs = torch.randn(4, 10)  # batch_size=4
        
        actions, values, log_probs = policy.forward(obs, deterministic=False)
        
        assert actions.shape == (4,), "Actions should have shape [batch_size]"
        assert values.shape == (4, 1), "Values should have shape [batch_size, 1]"
        assert log_probs.shape == (4,), "Log probs should have shape [batch_size]"
        
        # All actions should be valid (0, 1, or 2)
        assert torch.all((actions >= 0) & (actions < 3))

    def test_forward_with_full_mask(self, policy):
        """全アクション合法のマスクでforwardが動作することを確認"""
        obs = torch.randn(4, 10)
        action_masks = torch.ones(4, 3)  # All actions legal
        
        actions, values, log_probs = policy.forward(
            obs, deterministic=False, action_masks=action_masks
        )
        
        assert actions.shape == (4,)
        assert values.shape == (4, 1)
        assert log_probs.shape == (4,)

    def test_forward_with_partial_mask(self, policy):
        """部分的なマスク（一部アクション非法）でforwardが動作することを確認"""
        obs = torch.randn(4, 10)
        # HOLD のみ合法（BUY, SELL は非法）
        action_masks = torch.tensor([
            [1, 0, 0],  # Only HOLD legal
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ], dtype=torch.float32)
        
        actions, values, log_probs = policy.forward(
            obs, deterministic=True, action_masks=action_masks
        )
        
        # All actions should be HOLD (action=0) because it's the only legal action
        assert torch.all(actions == 0), "All actions should be HOLD when only HOLD is legal"

    def test_forward_illegal_actions_get_zero_probability(self, policy):
        """違法アクションの確率がゼロになることを確認"""
        obs = torch.randn(8, 10)
        # HOLD と BUY のみ合法（SELL は非法）
        action_masks = torch.tensor([
            [1, 1, 0],  # HOLD, BUY legal; SELL illegal
        ] * 8, dtype=torch.float32)
        
        # Get logits directly to check masking
        features = policy.extract_features(obs, policy.features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        logits_raw = policy.action_net(latent_pi)
        
        # Apply mask as in forward()
        logits_masked = torch.where(
            action_masks.bool(),
            logits_raw,
            torch.tensor(-1e9, dtype=logits_raw.dtype),
        )
        
        # SELL (action=2) should have logits=-1e9
        assert torch.allclose(
            logits_masked[:, 2],
            torch.tensor(-1e9, dtype=logits_masked.dtype),
            rtol=1e-4
        ), "Illegal action (SELL) should have logits=-1e9"
        
        # Probabilities after softmax
        probs = torch.softmax(logits_masked, dim=-1)
        
        # SELL probability should be effectively zero
        assert torch.all(probs[:, 2] < 1e-8), "Illegal action (SELL) should have near-zero probability"
        
        # HOLD and BUY probabilities should sum to ~1
        assert torch.allclose(
            probs[:, 0] + probs[:, 1],
            torch.ones(8),
            atol=1e-6
        ), "Legal actions probabilities should sum to 1"

    def test_forward_deterministic_vs_stochastic(self, policy):
        """決定論的vs確率的サンプリングの違いを確認"""
        obs = torch.randn(1, 10)
        action_masks = torch.ones(1, 3)
        
        # Deterministic: should always select the same action
        actions_det = []
        for _ in range(5):
            action, _, _ = policy.forward(obs, deterministic=True, action_masks=action_masks)
            actions_det.append(action.item())
        
        # All deterministic actions should be identical
        assert len(set(actions_det)) == 1, "Deterministic actions should be identical"
        
        # Stochastic: may select different actions (depending on entropy)
        # We just verify it runs without errors
        for _ in range(5):
            action, _, _ = policy.forward(obs, deterministic=False, action_masks=action_masks)
            assert 0 <= action.item() < 3


class TestStrictMaskedPolicyEvaluateActions:
    """evaluate_actions() メソッドのテスト（損失計算時のマスク適用）"""

    def test_evaluate_actions_without_mask(self, policy):
        """マスクなしでevaluate_actionsが動作することを確認"""
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        
        assert values.shape == (4, 1)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

    def test_evaluate_actions_with_mask(self, policy):
        """マスク付きでevaluate_actionsが動作することを確認"""
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        action_masks = torch.tensor([
            [1, 1, 0],  # HOLD, BUY legal
            [1, 1, 0],
            [1, 0, 1],  # HOLD, SELL legal
            [1, 1, 0],
        ], dtype=torch.float32)
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions, action_masks)
        
        assert values.shape == (4, 1)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

    def test_evaluate_actions_illegal_action_low_log_prob(self, policy):
        """
        違法アクションのlog_probが非常に低い（負の大きな値）ことを確認
        
        違法アクションが損失計算に寄与しないことを間接的に検証。
        logits=-1e9 → prob≈0 → log_prob≈-inf となり、損失への影響が最小化される。
        """
        obs = torch.randn(4, 10)
        # 違法アクション（SELL）を強制的に選択
        actions = torch.tensor([2, 2, 2, 2], dtype=torch.long)
        # SELL を非法に設定
        action_masks = torch.tensor([
            [1, 1, 0],  # SELL illegal
        ] * 4, dtype=torch.float32)
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions, action_masks)
        
        # Log probabilities for illegal actions should be very negative
        # (log of near-zero probability)
        assert torch.all(log_probs < -10), (
            f"Log probs for illegal actions should be very negative, got {log_probs}"
        )

    def test_evaluate_actions_entropy_with_mask(self, policy):
        """マスク適用後のエントロピーが正しいことを確認"""
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        
        # Only HOLD legal → entropy should be very low (near zero)
        action_masks_only_hold = torch.tensor([
            [1, 0, 0],
        ] * 4, dtype=torch.float32)
        
        _, _, entropy_low = policy.evaluate_actions(obs, actions, action_masks_only_hold)
        
        # All legal → entropy should be higher
        action_masks_all_legal = torch.ones(4, 3, dtype=torch.float32)
        
        _, _, entropy_high = policy.evaluate_actions(obs, actions, action_masks_all_legal)
        
        # Entropy with only one legal action should be lower than with all legal
        assert torch.all(entropy_low < entropy_high), (
            "Entropy with restricted actions should be lower than with all actions legal"
        )


class TestStrictMaskedPolicyPredictValues:
    """predict_values() メソッドのテスト"""

    def test_predict_values(self, policy):
        """predict_valuesが正しく動作することを確認"""
        obs = torch.randn(4, 10)
        
        values = policy.predict_values(obs)
        
        assert values.shape == (4, 1), "Values should have shape [batch_size, 1]"

    def test_predict_values_consistency(self, policy):
        """
        predict_valuesとevaluate_actionsで計算されるvaluesが一致することを確認
        """
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        
        values_from_predict = policy.predict_values(obs)
        values_from_evaluate, _, _ = policy.evaluate_actions(obs, actions)
        
        # Values should be identical (within numerical precision)
        assert torch.allclose(values_from_predict, values_from_evaluate, atol=1e-6), (
            "Values from predict_values and evaluate_actions should match"
        )


class TestStrictMaskedPolicyIntegration:
    """統合テスト: 実際の学習フローに近いシナリオ"""

    def test_training_step_simulation(self, policy):
        """
        学習ステップのシミュレーション
        
        1. 観測を取得
        2. アクションをサンプリング
        3. アクションを評価（損失計算）
        """
        batch_size = 32
        obs = torch.randn(batch_size, 10)
        
        # Variable action masks (simulating different states)
        action_masks = torch.randint(0, 2, (batch_size, 3), dtype=torch.float32)
        # Ensure at least HOLD is always legal
        action_masks[:, 0] = 1
        
        # Step 1: Sample actions
        actions, values_forward, log_probs_forward = policy.forward(
            obs, deterministic=False, action_masks=action_masks
        )
        
        # Step 2: Evaluate actions (for loss calculation)
        values_eval, log_probs_eval, entropy = policy.evaluate_actions(
            obs, actions, action_masks
        )
        
        # Sanity checks
        assert actions.shape == (batch_size,)
        assert values_forward.shape == (batch_size, 1)
        assert log_probs_forward.shape == (batch_size,)
        assert values_eval.shape == (batch_size, 1)
        assert log_probs_eval.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        
        # Values should be similar (same observation, same network)
        assert torch.allclose(values_forward, values_eval, atol=1e-5)

    def test_no_illegal_actions_sampled(self, policy):
        """
        マスク適用後、違法アクションがサンプリングされないことを確認
        
        大量のサンプルを生成して統計的に検証。
        """
        n_samples = 1000
        obs = torch.randn(n_samples, 10)
        
        # SELL (action=2) を全サンプルで非法に設定
        action_masks = torch.tensor([
            [1, 1, 0],  # HOLD, BUY legal; SELL illegal
        ] * n_samples, dtype=torch.float32)
        
        actions, _, _ = policy.forward(obs, deterministic=False, action_masks=action_masks)
        
        # SELL (action=2) が一度もサンプリングされないことを確認
        assert torch.all(actions != 2), (
            f"Illegal action (SELL=2) was sampled: {actions[actions == 2].shape[0]} times"
        )
        
        # HOLD と BUY のみがサンプリングされるべき
        unique_actions = torch.unique(actions)
        assert torch.all(unique_actions < 2), f"Only HOLD(0) and BUY(1) should be sampled, got {unique_actions}"


class TestStrictMaskedPolicyEdgeCases:
    """エッジケースのテスト"""

    def test_single_legal_action_deterministic(self, policy):
        """合法アクションが1つだけの場合（決定論的）"""
        obs = torch.randn(4, 10)
        # Only HOLD (action=0) is legal
        action_masks = torch.tensor([
            [1, 0, 0],
        ] * 4, dtype=torch.float32)
        
        actions, _, _ = policy.forward(obs, deterministic=True, action_masks=action_masks)
        
        # All actions must be HOLD
        assert torch.all(actions == 0)

    def test_single_legal_action_stochastic(self, policy):
        """合法アクションが1つだけの場合（確率的）"""
        obs = torch.randn(100, 10)
        # Only BUY (action=1) is legal
        action_masks = torch.tensor([
            [0, 1, 0],
        ] * 100, dtype=torch.float32)
        
        actions, _, _ = policy.forward(obs, deterministic=False, action_masks=action_masks)
        
        # All actions must be BUY (even in stochastic mode)
        assert torch.all(actions == 1)

    def test_batch_size_one(self, policy):
        """バッチサイズ=1の場合"""
        obs = torch.randn(1, 10)
        action_masks = torch.tensor([[1, 1, 0]], dtype=torch.float32)
        
        actions, values, log_probs = policy.forward(obs, deterministic=False, action_masks=action_masks)
        
        assert actions.shape == (1,)
        assert values.shape == (1, 1)
        assert log_probs.shape == (1,)

    def test_large_batch_size(self, policy):
        """大きなバッチサイズ（256）での動作確認"""
        batch_size = 256
        obs = torch.randn(batch_size, 10)
        action_masks = torch.ones(batch_size, 3, dtype=torch.float32)
        
        actions, values, log_probs = policy.forward(obs, deterministic=False, action_masks=action_masks)
        
        assert actions.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size,)
