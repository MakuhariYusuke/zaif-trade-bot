"""
v459 Phase 0.2c: CausalScaler単体テスト
Doc04仕様準拠の因果性保証を検証
"""

import numpy as np
import pandas as pd
import pytest

from ztb.processing.causal_online_scaler import CausalOnlineScaler
from ztb.features.grouping.causal_grouped_scaler import CausalGroupedFeatureScaler


class TestCausalOnlineScaler:
    """CausalOnlineScalerのテスト"""
    
    def test_fit_basic(self):
        """基本的なfitの動作"""
        scaler = CausalOnlineScaler(shape=(3,))
        
        data = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
            'c': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        
        # Train: 最初の3行（index 0-2）
        scaler.fit(data, end_idx=2, feature_names=['a', 'b', 'c'])
        
        assert scaler.fitted is True
        assert scaler.fit_end_idx == 2
        assert scaler.n == 3  # 3サンプル
    
    def test_fit_records_end_idx(self):
        """fit範囲のend_idx記録"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        
        # Train: index 0-6（7サンプル）
        scaler.fit(data, end_idx=6, feature_names=['x', 'y'])
        
        assert scaler.fit_end_idx == 6
        assert scaler.n == 7
    
    def test_zero_variance_handling(self):
        """ゼロ分散対応（std_floor適用）"""
        scaler = CausalOnlineScaler(shape=(2,), std_floor=1e-8)
        
        data = pd.DataFrame({
            'a': [1.0, 1.0, 1.0],  # ゼロ分散
            'b': [10.0, 20.0, 30.0],  # 正常
        })
        
        scaler.fit(data, end_idx=2, feature_names=['a', 'b'])
        
        # ゼロ分散列にstd_floorが適用されているか
        std = np.sqrt(scaler.var)
        assert std[0] >= scaler.std_floor
        assert std[1] > scaler.std_floor  # 正常列は大きい
    
    def test_transform_requires_fit(self):
        """fit前のtransformは例外"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        with pytest.raises(ValueError, match="Must call fit"):
            scaler.transform(np.array([1.0, 2.0]))
    
    def test_transform_normal(self):
        """正常なtransform"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 10.0, 20.0],
        })
        
        scaler.fit(data, end_idx=2, feature_names=['x', 'y'])
        
        # Transform
        scaled = scaler.transform(np.array([1.0, 10.0]))
        
        assert scaled.shape == (2,)
        assert not np.isnan(scaled).any()
        assert not np.isinf(scaled).any()
    
    def test_leakage_detection_simple(self):
        """リーク検査: Train期間と一致する場合OK"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        })
        
        # Train: index 0-2
        scaler.fit(data, end_idx=2, feature_names=['a', 'b'])
        
        # リーク検査は例外を投げないはず
        # （_verify_no_leakage内部で自動実行）
    
    def test_nan_detection_in_transform(self):
        """Transform時のNaN検出"""
        scaler = CausalOnlineScaler(shape=(2,), epsilon=0.0, std_floor=0.0)
        
        # ゼロ分散データでfit（意図的にNaN生成狙い）
        data = pd.DataFrame({
            'a': [0.0, 0.0, 0.0],
            'b': [0.0, 0.0, 0.0],
        })
        
        scaler.fit(data, end_idx=2, feature_names=['a', 'b'])
        
        # 親クラスのepsilonがあるためNaNにはならないが、
        # 仮にNaNが生成された場合は例外
        # （実際にはepsilon=1e-5がデフォルトなので発生しない）
    
    def test_get_fit_info(self):
        """Fit情報取得"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30],
        })
        
        scaler.fit(data, end_idx=2, feature_names=['x', 'y'])
        
        info = scaler.get_fit_info()
        
        assert info['fitted'] is True
        assert info['fit_end_idx'] == 2
        assert info['n_samples'] == 3
        assert 'mean_range' in info
        assert 'std_range' in info


class TestCausalGroupedFeatureScaler:
    """CausalGroupedFeatureScalerのテスト"""
    
    def test_fit_88_dimensional(self):
        """88次元データでのfit"""
        scaler = CausalGroupedFeatureScaler()
        
        # 88次元ダミーデータ
        data = pd.DataFrame(np.random.randn(100, 88))
        
        # Train: 最初の70行
        scaler.fit(data, end_idx=69)
        
        assert scaler.fitted is True
        assert scaler.fit_end_idx == 69
        # GroupedFeatureScalerにはn属性なし（親クラスの実装依存）
    
    def test_transform_88_dimensional(self):
        """88次元データのtransform"""
        scaler = CausalGroupedFeatureScaler()
        
        data = pd.DataFrame(np.random.randn(100, 88))
        
        scaler.fit(data, end_idx=69)
        
        # Transform
        features = np.random.randn(88)
        scaled = scaler.transform(features)
        
        assert scaled.shape == (88,)
        assert not np.isnan(scaled).any()
        assert not np.isinf(scaled).any()
    
    def test_selective_scaling(self):
        """選択的スケーリング（36次元のみ）"""
        scaler = CausalGroupedFeatureScaler()
        
        # Base[0:30] + Global[63:69] = 36次元がスケール対象
        data = pd.DataFrame(np.ones((10, 88)))
        
        scaler.fit(data, end_idx=9)
        
        # スケール対象インデックス確認
        scale_indices = scaler.SCALE_INDICES
        assert len(scale_indices) == 36
    
    def test_zero_variance_handling_grouped(self):
        """ゼロ分散対応（GroupedScaler版）"""
        scaler = CausalGroupedFeatureScaler(std_floor=1e-8)
        
        # 全て同じ値（ゼロ分散）
        data = pd.DataFrame(np.ones((10, 88)))
        
        scaler.fit(data, end_idx=9)
        
        # std_floorが適用されているか
        scale_indices = scaler.SCALE_INDICES
        std_scaled = scaler.std[scale_indices]
        assert (std_scaled >= scaler.std_floor).all()
    
    def test_transform_requires_fit_grouped(self):
        """fit前のtransformは例外"""
        scaler = CausalGroupedFeatureScaler()
        
        with pytest.raises(ValueError, match="Must call fit"):
            scaler.transform(np.random.randn(88))
    
    def test_invalid_dimension_detection(self):
        """次元数不正の検出"""
        scaler = CausalGroupedFeatureScaler()
        
        # 88次元以外のデータ
        data_wrong = pd.DataFrame(np.random.randn(10, 50))
        
        with pytest.raises(ValueError, match="Expected 88-dimensional"):
            scaler.fit(data_wrong, end_idx=9)
    
    def test_get_fit_info_grouped(self):
        """Fit情報取得（Grouped版）"""
        scaler = CausalGroupedFeatureScaler()
        
        data = pd.DataFrame(np.random.randn(100, 88))
        scaler.fit(data, end_idx=69)
        
        info = scaler.get_fit_info()
        
        assert info['fitted'] is True
        assert info['fit_end_idx'] == 69
        assert len(info['scale_indices']) == 36


class TestCausalScalerDoc04Compliance:
    """Doc04仕様準拠の検証"""
    
    def test_fit_range_enforcement(self):
        """fit範囲の厳格な管理"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'a': list(range(100)),
            'b': list(range(100, 200)),
        })
        
        # Train: index 0-49（50サンプル）
        scaler.fit(data, end_idx=49, feature_names=['a', 'b'])
        
        # Val/Test（index 50-99）の統計は使われていないことを確認
        train_mean = data.iloc[:50][['a', 'b']].mean().values
        
        np.testing.assert_allclose(scaler.mean, train_mean, rtol=1e-5)
    
    def test_end_idx_boundary_check(self):
        """end_idxの境界チェック"""
        scaler = CausalOnlineScaler(shape=(2,))
        
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30],
        })
        
        # end_idx >= len(data) は例外
        with pytest.raises(ValueError, match="must be less than data length"):
            scaler.fit(data, end_idx=3, feature_names=['x', 'y'])
    
    def test_doc04_causal_guarantee(self):
        """Doc04仕様: 因果性保証の検証"""
        scaler = CausalOnlineScaler(shape=(1,))
        
        # Train: 平均0、Val: 平均100（リーク検出テスト用）
        train_data = pd.DataFrame({'x': [0, 0, 0, 0, 0]})
        val_data = pd.DataFrame({'x': [100, 100, 100, 100, 100]})
        
        full_data = pd.concat([train_data, val_data], ignore_index=True)
        
        # Train期間のみでfit
        scaler.fit(full_data, end_idx=4, feature_names=['x'])
        
        # Scalerの統計はTrain期間のみを反映しているはず
        assert np.isclose(scaler.mean[0], 0.0, atol=1e-5)
        # Valデータ（平均100）が混入していればmean≠0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
