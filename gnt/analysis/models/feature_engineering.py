import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from itertools import combinations
import warnings
import re

logger = logging.getLogger(__name__)

class FormulaParser:
    """
    Parser for R-style formulas with support for:
    - Basic terms: y ~ x1 + x2
    - Interactions: x1:x2 or x1*x2 (includes main effects)
    - Polynomials: I(x^2) or poly(x, 2)
    - Instruments: y ~ x1 + x2 | z1 + z2 (for 2SLS)
    """
    
    def __init__(self, formula: str):
        """
        Initialize formula parser.
        
        Parameters:
        -----------
        formula : str
            R-style formula string (e.g., "y ~ x1 + x2 + I(x1^2) | z1")
        """
        self.formula = formula.strip()
        self.target = None
        self.features = []
        self.instruments = []
        self.transformations = []
        self.has_intercept = True
        
        self._parse_formula()
    
    def _parse_formula(self):
        """Parse the formula string."""
        # Split by ~ to separate target from features
        if '~' not in self.formula:
            raise ValueError(f"Formula must contain '~': {self.formula}")
        
        parts = self.formula.split('~')
        if len(parts) != 2:
            raise ValueError(f"Formula must have exactly one '~': {self.formula}")
        
        self.target = parts[0].strip()
        right_side = parts[1].strip()
        
        # Split by | to separate features from instruments
        if '|' in right_side:
            feature_part, instrument_part = right_side.split('|', 1)
            self.instruments = self._parse_term_list(instrument_part.strip())
        else:
            feature_part = right_side
        
        # Parse features and transformations
        self._parse_features(feature_part)
    
    def _parse_features(self, feature_part: str):
        """Parse feature part of formula and extract transformations."""
        # Check for intercept removal
        if '-1' in feature_part or '- 1' in feature_part:
            self.has_intercept = False
            feature_part = re.sub(r'-\s*1', '', feature_part)
        
        # Split by + but respect parentheses
        terms = self._split_respecting_parens(feature_part, '+')
        
        base_features = set()
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # Handle I() notation for arbitrary transformations
            if term.startswith('I(') and term.endswith(')'):
                self._parse_I_notation(term, base_features)
            # Handle poly() notation
            elif term.startswith('poly(') and term.endswith(')'):
                self._parse_poly_notation(term, base_features)
            # Handle interactions with *
            elif '*' in term:
                self._parse_interaction_star(term, base_features)
            # Handle interactions with :
            elif ':' in term:
                self._parse_interaction_colon(term, base_features)
            # Simple feature
            else:
                base_features.add(term)
        
        self.features = sorted(base_features)
    
    def _parse_I_notation(self, term: str, base_features: set):
        """Parse I() notation for transformations like I(x^2) or I(x*y)."""
        inner = term[2:-1].strip()
        
        # Handle powers: x^2, x^3, etc.
        if '^' in inner:
            match = re.match(r'(\w+)\s*\^\s*(\d+)', inner)
            if match:
                var_name = match.group(1)
                power = int(match.group(2))
                base_features.add(var_name)
                
                if power == 2:
                    self.transformations.append({
                        'type': 'quadratic',
                        'features': [var_name]
                    })
                elif power > 2:
                    self.transformations.append({
                        'type': 'polynomial',
                        'features': [var_name],
                        'degree': power
                    })
        # Handle products: x*y
        elif '*' in inner:
            vars_in_product = [v.strip() for v in inner.split('*')]
            for var in vars_in_product:
                base_features.add(var)
            
            if len(vars_in_product) == 2:
                self.transformations.append({
                    'type': 'interaction',
                    'feature_pairs': [vars_in_product]
                })
    
    def _parse_poly_notation(self, term: str, base_features: set):
        """Parse poly() notation like poly(x, 2) or poly(x, 3)."""
        inner = term[5:-1].strip()
        parts = [p.strip() for p in inner.split(',')]
        
        if len(parts) != 2:
            logger.warning(f"Invalid poly() notation: {term}")
            return
        
        var_name = parts[0]
        try:
            degree = int(parts[1])
        except ValueError:
            logger.warning(f"Invalid degree in poly(): {term}")
            return
        
        base_features.add(var_name)
        
        if degree > 1:
            self.transformations.append({
                'type': 'polynomial',
                'features': [var_name],
                'degree': degree
            })
    
    def _parse_interaction_star(self, term: str, base_features: set):
        """Parse interaction with * (includes main effects): x1*x2 = x1 + x2 + x1:x2."""
        vars_in_interaction = [v.strip() for v in term.split('*')]
        
        # Add all main effects
        for var in vars_in_interaction:
            base_features.add(var)
        
        # Add all pairwise interactions
        if len(vars_in_interaction) == 2:
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': [vars_in_interaction]
            })
        elif len(vars_in_interaction) > 2:
            pairs = list(combinations(vars_in_interaction, 2))
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': pairs
            })
    
    def _parse_interaction_colon(self, term: str, base_features: set):
        """Parse interaction with : (no main effects): x1:x2."""
        vars_in_interaction = [v.strip() for v in term.split(':')]
        
        # Add variables to base features (needed for computation)
        for var in vars_in_interaction:
            base_features.add(var)
        
        # Add interaction term only
        if len(vars_in_interaction) == 2:
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': [vars_in_interaction]
            })
    
    def _parse_term_list(self, term_list: str) -> List[str]:
        """Parse a simple list of terms separated by +."""
        return [t.strip() for t in term_list.split('+') if t.strip()]
    
    def _split_respecting_parens(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter while respecting parentheses."""
        parts = []
        current = []
        paren_depth = 0
        
        for char in text:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == delimiter and paren_depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration from parsed formula.
        
        Returns:
        --------
        config : dict
            Configuration dictionary for FeatureTransformer
        """
        return {
            'transformations': self.transformations
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed formula to dictionary representation."""
        return {
            'formula': self.formula,
            'target': self.target,
            'features': self.features,
            'instruments': self.instruments,
            'has_intercept': self.has_intercept,
            'transformations': self.transformations
        }
    
    @classmethod
    def parse(cls, formula: str) -> 'FormulaParser':
        """Convenience method to parse a formula."""
        return cls(formula)


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
        
        # logger.info(f"FeatureTransformer initialized:")
        # logger.info(f"  Add intercept: {self.add_intercept}")
        # logger.info(f"  Base features: {self.n_base_features}")
        # logger.info(f"  Total features after transformations: {self.n_total_features}")
    
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
            
        logger.info(f"Added {len(features)} quadratic features")
    
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
            
        logger.info(f"Added {len(feature_pairs)} interaction features")
    
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
                
        logger.info(f"Added polynomial features up to degree {degree} for {len(features)} features")
    
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
        
        logger.info(f"Registered predicted substitution: {original} -> {predicted}")
    
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
        
        logger.info(f"Added custom transformation: {custom_name}")
    
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
                logger.warning(f"First stage feature engineering failed for {original}: {e}")
        
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