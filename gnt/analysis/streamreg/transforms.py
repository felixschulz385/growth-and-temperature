import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    Flexible feature engineering system for online learning.
    Supports intercept, quadratic terms, interactions, polynomials, and 2SLS predicted substitution.
    Can be initialized from R-style formulas or configuration dictionaries.
    """
    
    def __init__(self, transformations: List[Dict[str, Any]], base_features: List[str], 
                 add_intercept: bool = True, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature transformer.
        
        Parameters:
        -----------
        transformations : list of dict
            List of transformation specifications from YAML config or formula
        base_features : list of str
            Original feature column names
        add_intercept : bool
            Whether to add intercept term as first feature
        context : dict, optional
            Additional context for transformations (e.g., for 2SLS)
        """
        self.transformations = transformations or []
        self.base_features = base_features.copy()
        self.add_intercept = add_intercept
        self.context = context or {}
        
        # Initialize feature names
        self.feature_names = []
        if self.add_intercept:
            self.feature_names.append("intercept")
        self.feature_names.extend(base_features)
        
        self.n_base_features = len(base_features)
        self.n_total_features = len(self.feature_names)
        
        self._predicted_substitutions = {}
        
        self._parse_transformations()
    
    def _parse_transformations(self):
        """Parse transformation specifications and update feature names."""
        for i, transform in enumerate(self.transformations):
            try:
                transform_type = transform.get('type', '').lower()
                
                if transform_type == 'quadratic':
                    self._add_quadratic_features(transform)
                elif transform_type == 'interaction':
                    self._add_interaction_features(transform)
                elif transform_type == 'polynomial':
                    self._add_polynomial_features(transform)
                elif transform_type == 'predicted_substitution':
                    self._add_predicted_substitution(transform)
                elif transform_type == 'custom':
                    self._add_custom_transformation(transform)
                else:
                    logger.warning(f"Unknown transformation type: {transform_type}")
                    
            except Exception as e:
                logger.error(f"Error parsing transformation {i}: {e}")
                raise ValueError(f"Invalid transformation specification: {transform}")
    
    def _add_quadratic_features(self, transform: Dict[str, Any]):
        """Add quadratic terms for specified features."""
        features = transform.get('features', [])
        if not features:
            logger.warning("Quadratic transformation specified but no features provided")
            return
        
        missing_features = [f for f in features if f not in self.base_features]
        if missing_features:
            raise ValueError(f"Quadratic features not found in base features: {missing_features}")
        
        for feature in features:
            quad_name = f"{feature}_squared"
            self.feature_names.append(quad_name)
            self.n_total_features += 1
    
    def _add_interaction_features(self, transform: Dict[str, Any]):
        """Add interaction terms between specified feature pairs."""
        feature_pairs = transform.get('feature_pairs', [])
        
        if not feature_pairs:
            features = transform.get('features', [])
            if features:
                feature_pairs = list(combinations(features, 2))
        
        if not feature_pairs:
            logger.warning("Interaction transformation specified but no feature pairs provided")
            return
        
        all_features = set()
        for pair in feature_pairs:
            all_features.update(pair)
        
        missing_features = [f for f in all_features if f not in self.base_features]
        if missing_features:
            raise ValueError(f"Interaction features not found in base features: {missing_features}")
        
        for feat1, feat2 in feature_pairs:
            interaction_name = f"{feat1}_x_{feat2}"
            self.feature_names.append(interaction_name)
            self.n_total_features += 1
    
    def _add_polynomial_features(self, transform: Dict[str, Any]):
        """Add polynomial terms for specified features."""
        features = transform.get('features', [])
        degree = transform.get('degree', 2)
        
        if not features:
            logger.warning("Polynomial transformation specified but no features provided")
            return
        
        if degree < 2:
            logger.warning(f"Polynomial degree {degree} < 2, skipping transformation")
            return
        
        missing_features = [f for f in features if f not in self.base_features]
        if missing_features:
            raise ValueError(f"Polynomial features not found in base features: {missing_features}")
        
        for feature in features:
            for d in range(2, degree + 1):
                poly_name = f"{feature}_pow{d}"
                self.feature_names.append(poly_name)
                self.n_total_features += 1

    def _add_predicted_substitution(self, transform: Dict[str, Any]):
        """Handle predicted value substitution for 2SLS."""
        original = transform.get('original')
        predicted = transform.get('predicted')
        
        if not original or not predicted:
            raise ValueError("Predicted substitution requires 'original' and 'predicted' fields")
        
        if original not in self.base_features:
            raise ValueError(f"Original feature '{original}' not found in base features")
        
        self._predicted_substitutions[original] = {
            'predicted_name': predicted,
            'first_stage_coefficients': transform.get('first_stage_coefficients'),
            'first_stage_feature_config': transform.get('first_stage_feature_config')
        }
    
    def _add_custom_transformation(self, transform: Dict[str, Any]):
        """Add custom transformation."""
        name = transform.get('name', 'custom')
        features = transform.get('features', [])
        
        missing_features = [f for f in features if f not in self.base_features]
        if missing_features:
            raise ValueError(f"Custom transformation features not found: {missing_features}")
        
        custom_name = f"{name}_transform"
        self.feature_names.append(custom_name)
        self.n_total_features += 1
        
        logger.debug(f"Added custom transformation: {custom_name}")
    
    def transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply all transformations to input data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data, shape (n_samples, n_features)
            May include extra columns (instruments) beyond base_features
        feature_names : list of str, optional
            Names of columns in X (for validation)
            
        Returns:
        --------
        X_transformed : np.ndarray
            Transformed data, shape (n_samples, n_total_features)
        """
        # Handle case where X includes extra columns (e.g., instruments for 2SLS)
        if X.shape[1] > self.n_base_features:
            X_base = X[:, :self.n_base_features]
            Z_instruments = X[:, self.n_base_features:]
            return self.transform_with_instruments(X_base, Z_instruments, feature_names)
        
        if X.shape[1] != self.n_base_features:
            raise ValueError(f"Input data has {X.shape[1]} features, expected {self.n_base_features}")
        
        if feature_names and len(feature_names) > self.n_base_features:
            feature_names = feature_names[:self.n_base_features]
        elif feature_names and len(feature_names) != self.n_base_features:
            raise ValueError(f"Feature names length {len(feature_names)} doesn't match base features {self.n_base_features}")
        
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_transformed = np.column_stack([intercept, X])
        else:
            X_transformed = X.copy()
        
        for transform in self.transformations:
            try:
                transform_type = transform.get('type', '').lower()
                
                if transform_type == 'quadratic':
                    X_transformed = self._apply_quadratic(X_transformed, X, transform, feature_names)
                elif transform_type == 'interaction':
                    X_transformed = self._apply_interaction(X_transformed, X, transform, feature_names)
                elif transform_type == 'polynomial':
                    X_transformed = self._apply_polynomial(X_transformed, X, transform, feature_names)
                elif transform_type == 'predicted_substitution':
                    logger.warning(f"Predicted substitution requires explicit instruments, skipping transformation")
                    continue
                elif transform_type == 'custom':
                    X_transformed = self._apply_custom(X_transformed, X, transform, feature_names)
                    
            except Exception as e:
                logger.error(f"Error applying transformation {transform}: {e}")
                raise
        
        return X_transformed
    
    def transform_with_instruments(self, X: np.ndarray, Z: Optional[np.ndarray] = None, 
                                 feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply all transformations to input data with explicit instrument handling for 2SLS.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (endogenous + exogenous variables), shape (n_samples, n_base_features)
        Z : np.ndarray, optional
            Instrument variables, shape (n_samples, n_instruments)
        feature_names : list of str, optional
            Names of columns in X (for validation)
            
        Returns:
        --------
        X_transformed : np.ndarray
            Transformed data, shape (n_samples, n_total_features)
        """
        if X.shape[1] != self.n_base_features:
            raise ValueError(f"Input data has {X.shape[1]} features, expected {self.n_base_features}")
        
        if feature_names and len(feature_names) != self.n_base_features:
            raise ValueError(f"Feature names length {len(feature_names)} doesn't match base features {self.n_base_features}")
        
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_transformed = np.column_stack([intercept, X])
        else:
            X_transformed = X.copy()
        
        for transform in self.transformations:
            try:
                transform_type = transform.get('type', '').lower()
                
                if transform_type == 'quadratic':
                    X_transformed = self._apply_quadratic(X_transformed, X, transform, feature_names)
                elif transform_type == 'interaction':
                    X_transformed = self._apply_interaction(X_transformed, X, transform, feature_names)
                elif transform_type == 'polynomial':
                    X_transformed = self._apply_polynomial(X_transformed, X, transform, feature_names)
                elif transform_type == 'predicted_substitution':
                    X_transformed = self._apply_predicted_substitution_with_instruments(
                        X_transformed, X, Z, transform, feature_names)
                elif transform_type == 'custom':
                    X_transformed = self._apply_custom(X_transformed, X, transform, feature_names)
                    
            except Exception as e:
                logger.error(f"Error applying transformation {transform}: {e}")
                raise
        
        return X_transformed
    
    def _apply_quadratic(self, X_current: np.ndarray, X_original: np.ndarray, 
                        transform: Dict[str, Any], feature_names: Optional[List[str]]) -> np.ndarray:
        """Apply quadratic transformation."""
        features = transform.get('features', [])
        
        for feature in features:
            idx = self.base_features.index(feature)
            quad_values = X_original[:, idx] ** 2
            X_current = np.column_stack([X_current, quad_values])
        
        return X_current
    
    def _apply_interaction(self, X_current: np.ndarray, X_original: np.ndarray,
                          transform: Dict[str, Any], feature_names: Optional[List[str]]) -> np.ndarray:
        """Apply interaction transformation."""
        feature_pairs = transform.get('feature_pairs', [])
        
        if not feature_pairs:
            features = transform.get('features', [])
            if features:
                feature_pairs = list(combinations(features, 2))
        
        for feat1, feat2 in feature_pairs:
            idx1 = self.base_features.index(feat1)
            idx2 = self.base_features.index(feat2)
            interaction_values = X_original[:, idx1] * X_original[:, idx2]
            X_current = np.column_stack([X_current, interaction_values])
        
        return X_current
    
    def _apply_polynomial(self, X_current: np.ndarray, X_original: np.ndarray,
                         transform: Dict[str, Any], feature_names: Optional[List[str]]) -> np.ndarray:
        """Apply polynomial transformation."""
        features = transform.get('features', [])
        degree = transform.get('degree', 2)
        
        for feature in features:
            idx = self.base_features.index(feature)
            for d in range(2, degree + 1):
                poly_values = X_original[:, idx] ** d
                X_current = np.column_stack([X_current, poly_values])
        
        return X_current
    
    def _apply_predicted_substitution_with_instruments(self, X_current: np.ndarray, X_original: np.ndarray,
                                                     Z: Optional[np.ndarray], transform: Dict[str, Any], 
                                                     feature_names: Optional[List[str]]) -> np.ndarray:
        """Apply predicted substitution transformation for 2SLS with explicit instruments."""
        original = transform.get('original')
        first_stage_coefficients = transform.get('first_stage_coefficients')
        first_stage_feature_config = transform.get('first_stage_feature_config')
        first_stage_feature_names = transform.get('first_stage_feature_names')
        add_intercept_first_stage = transform.get('add_intercept_first_stage', True)
        
        if first_stage_coefficients is None:
            raise ValueError(f"No first stage coefficients found for {original}")
        
        if Z is None:
            raise ValueError(f"Instruments (Z) required for predicted substitution of {original}")
        
        # Find the position of the original variable in X_current
        if self.add_intercept:
            orig_idx_in_current = 1 + self.base_features.index(original)
        else:
            orig_idx_in_current = self.base_features.index(original)
        
        # Extract exogenous variables from X_original (exclude endogenous variables)
        endogenous_indices = [i for i, feat in enumerate(self.base_features) if feat == original]
        exogenous_indices = [i for i in range(len(self.base_features)) if i not in endogenous_indices]
        
        if exogenous_indices:
            X_exogenous = X_original[:, exogenous_indices]
            first_stage_input_features = np.column_stack([X_exogenous, Z])
        else:
            first_stage_input_features = Z
        
        # Apply first stage feature engineering if specified
        if first_stage_feature_config and first_stage_feature_config.get('transformations'):
            try:
                if first_stage_feature_names:
                    temp_feature_names = first_stage_feature_names.copy()
                else:
                    exog_names = [self.base_features[i] for i in exogenous_indices]
                    instr_names = [f"instrument_{i}" for i in range(Z.shape[1])]
                    temp_feature_names = exog_names + instr_names
                
                first_stage_transformer = FeatureTransformer(
                    transformations=first_stage_feature_config['transformations'],
                    base_features=temp_feature_names,
                    add_intercept=False,
                    context=self.context
                )
                first_stage_input_features = first_stage_transformer.transform(first_stage_input_features)
                
            except Exception as e:
                logger.debug(f"First stage feature engineering failed for {original}: {e}")
        
        # Handle intercept for first stage prediction
        expected_dims = len(first_stage_coefficients)
        current_dims = first_stage_input_features.shape[1]
        
        if add_intercept_first_stage and expected_dims == current_dims + 1:
            intercept = np.ones((first_stage_input_features.shape[0], 1), dtype=first_stage_input_features.dtype)
            first_stage_features_final = np.column_stack([intercept, first_stage_input_features])
        elif expected_dims == current_dims:
            first_stage_features_final = first_stage_input_features
        else:
            raise ValueError(
                f"Feature dimension mismatch for {original}: "
                f"have {current_dims} features, need {expected_dims} coefficients"
            )
                
        # Compute predicted values
        coefficients_array = np.array(first_stage_coefficients)
        if coefficients_array.ndim == 1:
            coefficients_array = coefficients_array.reshape(-1, 1)
        
        try:
            predicted_values = first_stage_features_final @ coefficients_array.flatten()
        except ValueError as e:
            raise ValueError(f"Matrix dimension mismatch in first stage prediction for {original}: {e}")
        
        # Replace the original variable with predicted values
        X_current[:, orig_idx_in_current] = predicted_values
        
        return X_current
    
    def _apply_custom(self, X_current: np.ndarray, X_original: np.ndarray,
                     transform: Dict[str, Any], feature_names: Optional[List[str]]) -> np.ndarray:
        """Apply custom transformation."""
        custom_values = np.ones((X_original.shape[0], 1))
        return np.column_stack([X_current, custom_values])
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names after transformations."""
        return self.feature_names.copy()
    
    def get_base_feature_names(self) -> List[str]:
        """Get original base feature names."""
        return self.base_features.copy()
    
    def get_n_features(self) -> int:
        """Get total number of features after transformations."""
        return self.n_total_features
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about applied transformations."""
        return {
            'add_intercept': self.add_intercept,
            'n_base_features': self.n_base_features,
            'n_total_features': self.n_total_features,
            'base_feature_names': self.base_features.copy(),
            'all_feature_names': self.feature_names.copy(),
            'transformations': self.transformations.copy(),
            'predicted_substitutions': self._predicted_substitutions.copy()
        }
    
    @staticmethod
    def from_config(config: Dict[str, Any], base_features: List[str], 
                   add_intercept: bool = True, context: Optional[Dict[str, Any]] = None) -> 'FeatureTransformer':
        """Create FeatureTransformer from configuration dictionary."""
        transformations = config.get('transformations', [])
        return FeatureTransformer(transformations, base_features, add_intercept, context)
    
    @staticmethod
    def from_formula(formula: str, add_intercept: Optional[bool] = None, 
                    context: Optional[Dict[str, Any]] = None) -> Tuple['FeatureTransformer', 'FormulaParser']:
        """
        Create FeatureTransformer from R-style formula.
        
        Parameters:
        -----------
        formula : str
            R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
        add_intercept : bool, optional
            Override intercept from formula
        context : dict, optional
            Additional context for transformations
            
        Returns:
        --------
        transformer : FeatureTransformer
            Configured transformer
        parser : FormulaParser
            Parsed formula object (contains target, features, instruments)
        """
        parser = FormulaParser.parse(formula)
        
        # Use formula's intercept setting unless explicitly overridden
        use_intercept = parser.has_intercept if add_intercept is None else add_intercept
        
        config = parser.get_feature_config()
        transformer = FeatureTransformer.from_config(
            config, 
            parser.features, 
            add_intercept=use_intercept,
            context=context
        )
        
        return transformer, parser