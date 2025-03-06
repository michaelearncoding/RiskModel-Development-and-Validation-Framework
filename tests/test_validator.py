"""
Unit tests for the model validation module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.model_validation.validator import ModelValidator, validate_model


class TestModelValidator(unittest.TestCase):
    """Test cases for the ModelValidator class."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        
        # Convert to DataFrames with named features
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Split test into test and validation
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            self.X_test, self.y_test, test_size=0.5, random_state=42
        )
        
        # Train a simple model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Create a simple configuration for testing
        self.config = {
            'validation': {
                'cv_folds': 3,
                'performance_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'stability_test_periods': [3, 6],
                'sensitivity_variables': ['feature_0', 'feature_1'],
                'psi_threshold': 0.1,
                'monitoring_frequency': 'monthly'
            }
        }
        
        # Create validator instance
        self.validator = ModelValidator(self.model, self.config)
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.model, self.model)
        self.assertEqual(self.validator.config, self.config)
    
    def test_performance_testing(self):
        """Test performance testing functionality."""
        # Run performance testing
        results = self.validator.performance_testing(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        # Check results format
        self.assertIn('train', results)
        self.assertIn('test', results)
        
        # Check metrics in results
        for dataset in ['train', 'test']:
            self.assertIn('accuracy', results[dataset])
            self.assertIn('precision', results[dataset])
            self.assertIn('recall', results[dataset])
            self.assertIn('f1_score', results[dataset])
            self.assertIn('roc_auc', results[dataset])
            
            # Check metric values are reasonable
            self.assertTrue(0 <= results[dataset]['accuracy'] <= 1)
            self.assertTrue(0 <= results[dataset]['precision'] <= 1)
            self.assertTrue(0 <= results[dataset]['recall'] <= 1)
            self.assertTrue(0 <= results[dataset]['f1_score'] <= 1)
            self.assertTrue(0 <= results[dataset]['roc_auc'] <= 1)
    
    def test_discrimination_assessment(self):
        """Test discrimination assessment."""
        # Run discrimination assessment
        results = self.validator.assess_discrimination(self.X_test, self.y_test)
        
        # Check results
        self.assertIn('auc', results)
        self.assertIn('fpr', results)
        self.assertIn('tpr', results)
        self.assertIn('thresholds', results)
        self.assertIn('ks_statistic', results)
        self.assertIn('gini', results)
        
        # Check values
        self.assertTrue(0 <= results['auc'] <= 1)
        self.assertTrue(0 <= results['ks_statistic'] <= 1)
        self.assertTrue(-1 <= results['gini'] <= 1)
    
    def test_calibration_assessment(self):
        """Test calibration assessment."""
        # Run calibration assessment
        results = self.validator.assess_calibration(self.X_test, self.y_test)
        
        # Check results
        self.assertIn('mean_predicted_probs', results)
        self.assertIn('observed_probs', results)
        self.assertIn('brier_score', results)
        self.assertIn('calibration_error', results)
        
        # Check values
        self.assertTrue(0 <= results['brier_score'] <= 1)
        self.assertTrue(0 <= results['calibration_error'] <= 1)
    
    def test_stability_testing(self):
        """Test stability testing."""
        # Run stability testing
        results = self.validator.stability_testing(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        # Check results
        self.assertIn('feature_psi', results)
        self.assertIn('feature_stability', results)
        self.assertIn('performance_stability', results)
        
        # Check values
        for feature, psi in results['feature_psi'].items():
            self.assertTrue(psi >= 0)
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        # Run sensitivity analysis
        results = self.validator.sensitivity_analysis(self.X_test, self.y_test)
        
        # Check results
        self.assertIn('feature_sensitivity', results)
        self.assertIn('detailed_sensitivity', results)
        
        # Check feature_sensitivity format
        self.assertTrue(len(results['feature_sensitivity']) > 0)
        for item in results['feature_sensitivity']:
            self.assertIn('feature', item)
            self.assertIn('sensitivity_score', item)
            self.assertTrue(item['sensitivity_score'] >= 0)
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        # Run benchmark comparison
        results = self.validator.benchmark_comparison(self.X_test, self.y_test)
        
        # Check results
        self.assertIn('comparison', results)
        self.assertIn('primary_model_metrics', results)
        
        # Check comparison format
        self.assertTrue(len(results['comparison']) > 0)
        for item in results['comparison']:
            self.assertIn('model', item)
            self.assertIn('accuracy', item)
            self.assertIn('f1_score', item)
            self.assertIn('roc_auc', item)
    
    def test_validate_model_function(self):
        """Test the validate_model function."""
        # Run full validation
        results = validate_model(
            self.model, 
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            X_val=self.X_val, y_val=self.y_val, 
            config=self.config
        )
        
        # Check results
        self.assertIn('performance_results', results)
        self.assertIn('discrimination_results', results)
        self.assertIn('calibration_results', results)
        self.assertIn('stability_results', results)
        self.assertIn('sensitivity_results', results)
        self.assertIn('benchmark_results', results)
        self.assertIn('overall_assessment', results)


if __name__ == '__main__':
    unittest.main() 