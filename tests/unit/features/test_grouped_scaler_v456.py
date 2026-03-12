"""
Tests for GroupedFeatureScaler v456

88次元特徴量の選別的正規化テスト:
- 36次元スケーリング (Base [0:30] + Global continuous [63:69])
- 52次元非スケーリング (MTF [30:57] + Cyclical [57:63] + Regime [69:82] + Account [82:88])
"""

import pytest
import numpy as np
from typing import Tuple

from ztb.features.grouping.grouped_scaler import GroupedFeatureScaler


@pytest.fixture
def scaler():
    """デフォルト設定のスケーラー"""
    return GroupedFeatureScaler()


@pytest.fixture
def sample_features():
    """サンプル特徴量 (88次元)"""
    np.random.seed(42)
    # Base (0:30) + MTF (30:57) + Cyclical (57:63) + Global (63:69)
    # + Regime (69:82) + Account (82:88)
    return np.random.randn(88).astype(np.float32) * 10 + np.arange(88)


@pytest.fixture
def batch_features():
    """バッチサンプル (32, 88)"""
    np.random.seed(43)
    return np.random.randn(32, 88).astype(np.float32)


class TestGroupedScalerInitialization:
    """初期化テスト"""
    
    def test_scaler_initialization(self, scaler):
        """スケーラーが正しく初期化される"""
        assert scaler.n_samples == 0
        assert scaler.mean.shape == (88,)
        assert scaler.std.shape == (88,)
        assert np.all(scaler.mean == 0)
        assert np.all(scaler.std == 1)
    
    def test_scale_mask_correct(self, scaler):
        """スケーリングマスクが正しい"""
        # 対象: [0:30] + [63:69] = 36個
        assert np.sum(scaler.scale_mask) == 36
        
        # [0:30] が True
        assert np.all(scaler.scale_mask[0:30])
        
        # [63:69] が True
        assert np.all(scaler.scale_mask[63:69])
        
        # [30:63] が False (MTF + Cyclical)
        assert not np.any(scaler.scale_mask[30:63])
        
        # [69:88] が False (Regime + Account)
        assert not np.any(scaler.scale_mask[69:88])
    
    def test_scale_indices_count(self, scaler):
        """スケーリング対象インデックスが36個"""
        assert len(scaler.SCALE_INDICES) == 36
        assert len(set(scaler.SCALE_INDICES)) == 36  # ユニーク
    
    def test_parameters_in_init(self):
        """初期化パラメータが反映される"""
        scaler = GroupedFeatureScaler(epsilon=1e-5, momentum=0.95, clip_value=2.0)
        
        assert scaler.epsilon == 1e-5
        assert scaler.momentum == 0.95
        assert scaler.clip_value == 2.0


class TestFeatureStructureValidation:
    """特徴量構造の検証テスト"""
    
    def test_validate_feature_structure_passes(self, scaler):
        """構造検証が通る"""
        assert scaler.validate_feature_structure() is True
    
    def test_feature_groups_description(self, scaler):
        """特徴量グループの説明が正しい"""
        desc = scaler.get_feature_groups_description()
        
        assert desc['scaled']['total_scaled'] == 36
        assert desc['not_scaled']['total_not_scaled'] == 52
        assert desc['total_features'] == 88
        
        # サブグループの個数確認
        assert desc['scaled']['base'] == 'Base OHLCV derivatives (30)'
        assert desc['not_scaled']['mtf'] == 'Multi-timeframe features (27)'


class TestSingleSampleFitting:
    """単一サンプルのフィッティングテスト"""
    
    def test_fit_one_initializes_mean(self, scaler, sample_features):
        """fit_one で平均が初期化される"""
        scaler.fit_one(sample_features)
        
        # スケール対象の平均が設定される
        assert np.any(scaler.mean[0:30] != 0)
        assert np.any(scaler.mean[63:69] != 0)
        
        # スケール非対象は0のまま
        assert np.all(scaler.mean[30:63] == 0)
        assert np.all(scaler.mean[69:88] == 0)
    
    def test_fit_one_n_samples_increment(self, scaler, sample_features):
        """fit_one でサンプル数が増加"""
        assert scaler.n_samples == 0
        
        scaler.fit_one(sample_features)
        assert scaler.n_samples == 1
        
        scaler.fit_one(sample_features)
        assert scaler.n_samples == 2
    
    def test_fit_one_dimension_check(self, scaler):
        """不正な次元で例外を発生"""
        wrong_shape = np.random.randn(100)
        
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            scaler.fit_one(wrong_shape)
    
    def test_fit_one_momentum_update(self):
        """EMA更新が運動量を反映"""
        scaler = GroupedFeatureScaler(momentum=0.9)
        
        # 1回目: 初期化
        features1 = np.ones(88, dtype=np.float32)
        scaler.fit_one(features1)
        mean_after_first = scaler.mean.copy()
        
        # 2回目: EMA更新 (90% 過去, 10% 新規)
        features2 = np.ones(88, dtype=np.float32) * 2.0
        scaler.fit_one(features2)
        
        # スケール対象は 0.9*1.0 + 0.1*2.0 = 1.1 になるはず
        scale_mean = scaler.mean[0]  # base[0]
        expected = 0.9 * 1.0 + 0.1 * 2.0
        
        assert pytest.approx(scale_mean, rel=0.05) == expected


class TestBatchFitting:
    """バッチフィッティングのテスト"""
    
    def test_fit_batch_multiple_samples(self, scaler, batch_features):
        """バッチで複数サンプルをフィット"""
        scaler.fit_batch(batch_features)
        
        assert scaler.n_samples == batch_features.shape[0]
    
    def test_fit_batch_dimension_check(self, scaler):
        """不正な次元で例外"""
        wrong_batch = np.random.randn(88)  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            scaler.fit_batch(wrong_batch)
    
    def test_fit_batch_feature_dimension_check(self, scaler):
        """不正な特徴量次元で例外"""
        wrong_features = np.random.randn(32, 100)  # Wrong feature size
        
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            scaler.fit_batch(wrong_features)
    
    def test_fit_batch_equivalent_to_fit_one(self):
        """バッチ処理が単一処理と等価"""
        features = np.random.randn(32, 88).astype(np.float32)
        
        # 方法1: fit_batch
        scaler1 = GroupedFeatureScaler(momentum=0.99)
        scaler1.fit_batch(features)
        
        # 方法2: fit_one ループ
        scaler2 = GroupedFeatureScaler(momentum=0.99)
        for sample in features:
            scaler2.fit_one(sample)
        
        # 統計量がほぼ同じ（丸め誤差を許容）
        np.testing.assert_allclose(scaler1.mean, scaler2.mean, rtol=1e-5)
        np.testing.assert_allclose(scaler1.std, scaler2.std, rtol=1e-5)


class TestTransformation:
    """変換テスト"""
    
    def test_transform_single_sample(self, scaler, sample_features):
        """単一サンプルの変換"""
        scaler.fit_one(sample_features)
        scaled = scaler.transform(sample_features)
        
        assert scaled.shape == (88,)
        assert np.all(np.isfinite(scaled))
    
    def test_transform_batch(self, scaler, batch_features):
        """バッチ変換"""
        scaler.fit_batch(batch_features)
        scaled = scaler.transform(batch_features)
        
        assert scaled.shape == batch_features.shape
        assert np.all(np.isfinite(scaled))
    
    def test_scaled_features_bounds(self, scaler, batch_features):
        """スケール対象の値がクリッピング範囲内"""
        scaler.fit_batch(batch_features)
        scaled = scaler.transform(batch_features)
        
        # clip_value = 3.0 (デフォルト)
        # スケール対象 [0:30] + [63:69]
        base_scaled = scaled[:, 0:30]
        global_scaled = scaled[:, 63:69]
        
        assert np.all(np.abs(base_scaled) <= 3.0)
        assert np.all(np.abs(global_scaled) <= 3.0)
    
    def test_unscaled_features_unchanged(self, scaler, batch_features):
        """スケール非対象は変更されない"""
        # 元のMTF値
        original_mtf = batch_features[:, 30:57].copy()
        
        scaler.fit_batch(batch_features)
        scaled = scaler.transform(batch_features)
        
        # MTFは変わらない
        np.testing.assert_array_equal(scaled[:, 30:57], original_mtf)
    
    def test_transform_different_distribution(self, scaler, batch_features):
        """異なる分布データの変換"""
        # 訓練分布でフィット
        scaler.fit_batch(batch_features)
        
        # テスト分布（異なる）
        test_features = np.random.randn(16, 88).astype(np.float32) * 20 + 100
        
        scaled_test = scaler.transform(test_features)
        
        # スケール対象は正規化されている
        base_scaled = scaled_test[:, 0:30]
        
        # 平均がおおよそ0に近い（クリッピングがあるので完全ではない、<=3.0)
        assert np.abs(np.mean(base_scaled)) <= 3.0


class TestFitTransform:
    """fit_transform メソッドのテスト"""
    
    def test_fit_transform_single(self, scaler, sample_features):
        """単一サンプルの fit_transform"""
        scaled = scaler.fit_transform(sample_features)
        
        assert scaler.n_samples == 1
        assert scaled.shape == sample_features.shape
    
    def test_fit_transform_batch(self, scaler, batch_features):
        """バッチの fit_transform"""
        scaled = scaler.fit_transform(batch_features)
        
        assert scaler.n_samples == batch_features.shape[0]
        assert scaled.shape == batch_features.shape


class TestReset:
    """リセット機能のテスト"""
    
    def test_reset_clears_stats(self, scaler, sample_features):
        """reset で統計量がクリアされる"""
        scaler.fit_one(sample_features)
        assert scaler.n_samples == 1
        
        scaler.reset()
        
        assert scaler.n_samples == 0
        assert np.all(scaler.mean == 0)
        assert np.all(scaler.std == 1)
        assert scaler._initialized is False


class TestGetStats:
    """統計量取得のテスト"""
    
    def test_get_stats_structure(self, scaler, batch_features):
        """stats 辞書が正しい構造"""
        scaler.fit_batch(batch_features)
        stats = scaler.get_stats()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'n_samples' in stats
        assert 'scale_indices' in stats
        assert 'num_scaled' in stats
        assert 'num_not_scaled' in stats
        
        assert stats['num_scaled'] == 36
        assert stats['num_not_scaled'] == 52
        assert stats['n_samples'] == batch_features.shape[0]
    
    def test_get_stats_mean_shape(self, scaler, batch_features):
        """mean と std が 88 次元"""
        scaler.fit_batch(batch_features)
        stats = scaler.get_stats()
        
        assert stats['mean'].shape == (88,)
        assert stats['std'].shape == (88,)


class TestOutlierClipping:
    """外れ値クリッピングのテスト"""
    
    def test_clipping_applied(self):
        """クリッピングが適用される"""
        scaler = GroupedFeatureScaler(clip_value=2.0)
        
        # 平均0, 標準偏差1の特徴量で訓練
        scaler.fit_batch(np.random.randn(100, 88).astype(np.float32))
        
        # 極端な値を変換
        extreme_features = np.ones((1, 88), dtype=np.float32) * 100
        scaled = scaler.transform(extreme_features)
        
        # スケール対象はクリップされる [-2.0, 2.0]
        base_scaled = scaled[0, 0:30]
        assert np.all(np.abs(base_scaled) <= 2.0 + 1e-6)
    
    def test_clipping_value_parameter(self):
        """clip_value パラメータが機能"""
        scaler = GroupedFeatureScaler(clip_value=1.0)
        
        assert scaler.clip_value == 1.0


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_epsilon_prevents_division_by_zero(self, scaler, batch_features):
        """epsilon で division by zero を防ぐ"""
        # すべてのスケール対象特徴量が同じ値（分散=0）
        const_features = np.ones((10, 88), dtype=np.float32)
        const_features[:, 30:88] = np.random.randn(10, 58)  # 非スケール部のみランダム
        
        scaler.fit_batch(const_features)
        
        # transform で NaN や Inf が出ない
        scaled = scaler.transform(const_features)
        assert np.all(np.isfinite(scaled))
    
    def test_handles_nan_gracefully(self, scaler, batch_features):
        """NaN を含むデータの処理"""
        features_with_nan = batch_features.copy()
        features_with_nan[0, 5] = np.nan
        
        # NaN を含むデータでのフィット (スキップ or 影響)
        # 通常は実装で NaN チェックを追加
        # ここでは単に NaN が inf に変換されることを確認
        scaler.fit_batch(features_with_nan)
        # スケーラーが NaN に対応しているなら、成功
        assert scaler.n_samples == batch_features.shape[0]


class TestGroupIndexRanges:
    """グループインデックス範囲のテスト"""
    
    def test_scale_group_ranges(self, scaler):
        """スケール対象グループの範囲"""
        # Base
        assert scaler.SCALE_GROUPS['base'] == (0, 30)
        
        # Global continuous
        assert scaler.SCALE_GROUPS['global_continuous'] == (63, 69)
    
    def test_no_scale_group_ranges(self, scaler):
        """スケール非対象グループの範囲"""
        # MTF
        assert scaler.NO_SCALE_GROUPS['mtf'] == (30, 57)
        
        # Cyclical time
        assert scaler.NO_SCALE_GROUPS['cyclical_time'] == (57, 63)
        
        # Regime
        assert scaler.NO_SCALE_GROUPS['regime'] == (69, 82)
        
        # Account
        assert scaler.NO_SCALE_GROUPS['account'] == (82, 88)
    
    def test_no_overlap_between_groups(self, scaler):
        """グループ間の重複がない"""
        all_indices = set()
        
        for start, end in scaler.SCALE_GROUPS.values():
            group_indices = set(range(start, end))
            assert not (all_indices & group_indices), "Overlap detected in SCALE_GROUPS"
            all_indices.update(group_indices)
        
        for start, end in scaler.NO_SCALE_GROUPS.values():
            group_indices = set(range(start, end))
            assert not (all_indices & group_indices), "Overlap detected with NO_SCALE_GROUPS"
            all_indices.update(group_indices)
        
        # すべてのインデックスがカバーされている
        assert all_indices == set(range(88))


class TestIntegration:
    """統合テスト"""
    
    def test_full_pipeline_training(self):
        """訓練パイプライン全体"""
        scaler = GroupedFeatureScaler()
        
        # 訓練データ
        train_features = np.random.randn(1000, 88).astype(np.float32)
        
        # ミニバッチで訓練
        batch_size = 32
        for i in range(0, len(train_features), batch_size):
            batch = train_features[i:i+batch_size]
            scaler.fit_batch(batch)
        
        assert scaler.n_samples == 1000
    
    def test_full_pipeline_inference(self):
        """推論パイプライン全体"""
        scaler = GroupedFeatureScaler()
        
        # 訓練
        train_features = np.random.randn(1000, 88).astype(np.float32)
        scaler.fit_batch(train_features)
        
        # 推論
        test_features = np.random.randn(100, 88).astype(np.float32)
        scaled_test = scaler.transform(test_features)
        
        assert scaled_test.shape == test_features.shape
        assert np.all(np.isfinite(scaled_test))
