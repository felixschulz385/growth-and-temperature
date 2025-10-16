import numpy as np
import pytest
from gnt.analysis.streamreg.transforms import FeatureTransformer


class TestFeatureTransformer:
    """Tests for FeatureTransformer."""
    
    def test_intercept_only(self):
        """Test transformer with only intercept."""
        transformer = FeatureTransformer([], ['x1', 'x2'], add_intercept=True)
        
        X = np.array([[1, 2], [3, 4]])
        X_transformed = transformer.transform(X)
        
        assert X_transformed.shape == (2, 3)
        assert np.all(X_transformed[:, 0] == 1)  # Intercept
        assert np.allclose(X_transformed[:, 1:], X)
    
    def test_quadratic(self):
        """Test quadratic transformation."""
        transformations = [{'type': 'quadratic', 'features': ['x1']}]
        transformer = FeatureTransformer(transformations, ['x1', 'x2'], add_intercept=False)
        
        X = np.array([[2, 3], [4, 5]])
        X_transformed = transformer.transform(X)
        
        assert X_transformed.shape == (2, 3)  # x1, x2, x1^2
        assert np.allclose(X_transformed[:, 0], [2, 4])
        assert np.allclose(X_transformed[:, 1], [3, 5])
        assert np.allclose(X_transformed[:, 2], [4, 16])
    
    def test_interaction(self):
        """Test interaction transformation."""
        transformations = [{'type': 'interaction', 'feature_pairs': [['x1', 'x2']]}]
        transformer = FeatureTransformer(transformations, ['x1', 'x2'], add_intercept=False)
        
        X = np.array([[2, 3], [4, 5]])
        X_transformed = transformer.transform(X)
        
        assert X_transformed.shape == (2, 3)  # x1, x2, x1*x2
        assert np.allclose(X_transformed[:, 2], [6, 20])
    
    def test_polynomial(self):
        """Test polynomial transformation."""
        transformations = [{'type': 'polynomial', 'features': ['x1'], 'degree': 3}]
        transformer = FeatureTransformer(transformations, ['x1'], add_intercept=False)
        
        X = np.array([[2], [3]])
        X_transformed = transformer.transform(X)
        
        assert X_transformed.shape == (2, 3)  # x1, x1^2, x1^3
        assert np.allclose(X_transformed[:, 0], [2, 3])
        assert np.allclose(X_transformed[:, 1], [4, 9])
        assert np.allclose(X_transformed[:, 2], [8, 27])
    
    def test_combined_transformations(self):
        """Test combining multiple transformations."""
        transformations = [
            {'type': 'quadratic', 'features': ['x1']},
            {'type': 'interaction', 'feature_pairs': [['x1', 'x2']]}
        ]
        transformer = FeatureTransformer(transformations, ['x1', 'x2'], add_intercept=True)
        
        X = np.array([[2, 3]])
        X_transformed = transformer.transform(X)
        
        # intercept, x1, x2, x1^2, x1*x2
        assert X_transformed.shape == (1, 5)
        assert X_transformed[0, 0] == 1  # intercept
        assert X_transformed[0, 3] == 4  # x1^2
        assert X_transformed[0, 4] == 6  # x1*x2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
