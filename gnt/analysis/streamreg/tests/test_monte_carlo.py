"""
Monte Carlo tests for validating statistical properties of estimators.
"""
import numpy as np
import pytest
from scipy import stats
from gnt.analysis.streamreg.estimators.ols import OnlineRLS


def test_unbiasedness():
    """Monte Carlo test: Verify estimator is unbiased."""
    np.random.seed(123)
    n_simulations = 200
    n_samples = 300
    n_features = 3
    true_theta = np.array([0.5, -1.0, 2.0])
    
    estimates = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, n_features)
        y = X @ true_theta + np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    mean_bias = np.abs(estimates.mean(axis=0) - true_theta)
    
    # Bias should be very small
    assert np.all(mean_bias < 0.05)


def test_consistency():
    """Monte Carlo test: Verify estimator is consistent (converges as n→∞)."""
    np.random.seed(456)
    true_theta = np.array([1.0, 2.0])
    sample_sizes = [100, 500, 2000]
    
    variances = []
    
    for n_samples in sample_sizes:
        estimates = []
        
        for _ in range(50):
            X = np.random.randn(n_samples, 2)
            y = X @ true_theta + np.random.randn(n_samples)
            
            rls = OnlineRLS(n_features=2, alpha=1e-4)
            rls.partial_fit(X, y)
            estimates.append(rls.theta)
        
        estimates = np.array(estimates)
        variances.append(np.var(estimates, axis=0).mean())
    
    # Variance should decrease as sample size increases
    assert variances[1] < variances[0]
    assert variances[2] < variances[1]


def test_t_statistic_distribution():
    """Monte Carlo test: Verify t-statistics follow standard normal under H0."""
    np.random.seed(789)
    n_simulations = 500
    n_samples = 200
    
    # True parameter is zero (testing H0: beta = 0)
    true_theta = np.array([0.0, 0.0])
    
    t_statistics = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        y = X @ true_theta + np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        
        se = rls.get_standard_errors('classical')
        t_stats = rls.theta / se
        t_statistics.extend(t_stats)
    
    t_statistics = np.array(t_statistics)
    
    # Test if t-statistics follow standard normal
    # Shapiro-Wilk test for normality
    _, p_value = stats.shapiro(t_statistics[:200])  # Use subset for test
    assert p_value > 0.01  # Should not reject normality
    
    # Check mean and std
    assert np.abs(t_statistics.mean()) < 0.1
    assert 0.9 < t_statistics.std() < 1.1


def test_r_squared_properties():
    """Monte Carlo test: Verify R² has expected properties."""
    np.random.seed(321)
    n_simulations = 100
    n_samples = 300
    
    r_squareds = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        # True R² should be high (signal-to-noise ratio is high)
        y = 5 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        
        r_squareds.append(rls.get_r_squared())
    
    r_squareds = np.array(r_squareds)
    
    # All R² should be in [0, 1]
    assert np.all(r_squareds >= 0)
    assert np.all(r_squareds <= 1)
    
    # With high signal-to-noise, mean R² should be high
    assert r_squareds.mean() > 0.95


def test_f_statistic_distribution():
    """Monte Carlo test: Verify F-statistic distribution under H0."""
    np.random.seed(654)
    n_simulations = 200
    n_samples = 100
    n_features = 2
    
    f_statistics = []
    
    # Under null: all coefficients are zero
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)  # Pure noise, no relationship
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        f_stat, df_model, df_resid = rls.get_f_statistic()
        f_statistics.append(f_stat)
    
    f_statistics = np.array(f_statistics)
    
    # F-statistics should be mostly small under null
    # Mean F should be close to 1 under null
    assert 0.5 < f_statistics.mean() < 2.0
    
    # Very few should be large
    assert np.sum(f_statistics > 10) / len(f_statistics) < 0.05


def test_heteroskedasticity_robustness():
    """Monte Carlo test: Verify estimator handles heteroskedasticity."""
    np.random.seed(987)
    n_simulations = 100
    n_samples = 300
    true_theta = np.array([1.0, 2.0])
    
    estimates = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        
        # Heteroskedastic errors: variance depends on X
        error_var = 0.1 + 0.5 * X[:, 0]**2
        errors = np.random.randn(n_samples) * np.sqrt(error_var)
        
        y = X @ true_theta + errors
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    
    # Estimator should still be unbiased even with heteroskedasticity
    bias = np.abs(estimates.mean(axis=0) - true_theta)
    assert np.all(bias < 0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
