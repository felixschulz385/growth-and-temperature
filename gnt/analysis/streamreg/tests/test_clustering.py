import numpy as np
import pytest
from gnt.analysis.streamreg.estimators.ols import OnlineRLS, ClusterStatsAggregator


class TestClusterStatsAggregator:
    """Tests for cluster statistics aggregation."""
    
    def test_update_stats(self):
        """Test updating cluster statistics."""
        aggregator = ClusterStatsAggregator(n_features=2)
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        errors = np.array([0.1, 0.2, 0.3])
        cluster_ids = np.array(['A', 'A', 'B'])
        
        stats_dict = {}
        aggregator.update_stats(stats_dict, cluster_ids, X, y, errors)
        
        # Check that stats were created
        assert 'A' in stats_dict
        assert 'B' in stats_dict
        
        # Check cluster A (2 observations)
        assert stats_dict['A']['count'] == 2
        assert np.allclose(stats_dict['A']['X_sum'], [1+3, 2+4])
        
        # Check cluster B (1 observation)
        assert stats_dict['B']['count'] == 1
        assert np.allclose(stats_dict['B']['X_sum'], [5, 6])
    
    def test_merge_stats(self):
        """Test merging cluster statistics."""
        aggregator = ClusterStatsAggregator(n_features=2)
        
        source = {
            'A': {
                'X_sum': np.array([1, 2]),
                'residual_sum': 0.5,
                'count': 10,
                'XtX': np.eye(2),
                'X_residual_sum': np.array([0.1, 0.2]),
                'Xy': np.array([1, 1])
            }
        }
        
        target = {
            'A': {
                'X_sum': np.array([2, 3]),
                'residual_sum': 0.3,
                'count': 5,
                'XtX': 2 * np.eye(2),
                'X_residual_sum': np.array([0.2, 0.3]),
                'Xy': np.array([2, 2])
            }
        }
        
        aggregator.merge_stats(source, target)
        
        # Check merged values
        assert target['A']['count'] == 15
        assert np.allclose(target['A']['X_sum'], [3, 5])
        assert np.isclose(target['A']['residual_sum'], 0.8)


class TestClusterRobustSE:
    """Tests for cluster-robust standard errors."""
    
    def test_one_way_clustering(self):
        """Test one-way cluster-robust SEs."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        n_clusters = 10
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n_samples)
        clusters = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
        
        rls = OnlineRLS(n_features=n_features)
        rls.partial_fit(X, y, cluster1=clusters)
        
        # Get both classical and cluster-robust SEs
        se_classical = rls.get_standard_errors('classical')
        se_cluster = rls.get_standard_errors('one_way')
        
        # Cluster-robust SEs should generally be different (usually larger)
        assert not np.allclose(se_classical, se_cluster)
        
        # Both should be positive
        assert np.all(se_classical > 0)
        assert np.all(se_cluster > 0)
    
    def test_two_way_clustering(self):
        """Test two-way cluster-robust SEs."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n_samples)
        
        # Create two clustering dimensions
        cluster1 = np.repeat(np.arange(10), 10)
        cluster2 = np.tile(np.arange(10), 10)
        
        rls = OnlineRLS(n_features=n_features)
        rls.partial_fit(X, y, cluster1=cluster1, cluster2=cluster2)
        
        se_two_way = rls.get_standard_errors('two_way')
        
        # Should have valid SEs
        assert np.all(se_two_way > 0)
        assert se_two_way.shape == (n_features,)


def test_monte_carlo_cluster_coverage():
    """Monte Carlo test: verify cluster-robust CI coverage."""
    np.random.seed(42)
    n_simulations = 100
    n_samples = 200
    n_clusters = 20
    true_theta = np.array([1.0, 2.0])
    
    coverage_count = 0
    
    for sim in range(n_simulations):
        # Generate clustered data
        clusters = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
        
        X = np.random.randn(n_samples, 2)
        
        # Add cluster effects
        cluster_effects = np.random.randn(n_clusters) * 0.5
        cluster_noise = cluster_effects[clusters]
        
        y = X @ true_theta + cluster_noise + 0.1 * np.random.randn(n_samples)
        
        # Fit model
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y, cluster1=clusters)
        
        # Get cluster-robust SEs
        se = rls.get_standard_errors('one_way')
        
        # Check if true parameter is within 95% CI
        for feat_idx in range(2):
            ci_lower = rls.theta[feat_idx] - 1.96 * se[feat_idx]
            ci_upper = rls.theta[feat_idx] + 1.96 * se[feat_idx]
            
            if ci_lower <= true_theta[feat_idx] <= ci_upper:
                coverage_count += 1
    
    # Coverage should be approximately 95% (190 out of 200)
    coverage_rate = coverage_count / (n_simulations * 2)
    assert 0.90 <= coverage_rate <= 0.98  # Allow some variation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
