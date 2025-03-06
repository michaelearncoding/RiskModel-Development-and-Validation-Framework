"""
Unit tests for the model development module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.model_development.models import CreditRiskModel, train_model


class TestCreditRiskModel(unittest.TestCase):
    """Test cases for the CreditRiskModel class."""
    
    def setUp(self):
        """Set up test data."""
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
        
        # Create model instances
        self.models = {}
        for model_type in ['logistic_regression', 'random_forest', 'gradient_boosting']:
            self.models[model_type] = train_model(
                self.X_train, self.y_train, model_type=model_type
            )
    
    def test_model_initialization(self):
        """Test that models can be initialized correctly."""
        for model_type, model in self.models.items():
            # Create CreditRiskModel instance
            credit_model = CreditRiskModel(model_type, model=model)
            
            # Check model attributes
            self.assertEqual(credit_model.model_type, model_type)
            self.assertIsNotNone(credit_model.model)
            
            # Check feature names are stored
            self.assertEqual(len(credit_model.feature_names), self.X_train.shape[1])
    
    def test_model_predict(self):
        """Test model prediction functionality."""
        for model_type, model in self.models.items():
            credit_model = CreditRiskModel(model_type, model=model)
            
            # Test predict method
            predictions = credit_model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))
            self.assertTrue(all(pred in [0, 1] for pred in predictions))
            
            # Test predict_proba method
            probabilities = credit_model.predict_proba(self.X_test)
            self.assertEqual(len(probabilities), len(self.y_test))
            self.assertTrue(all(0 <= prob <= 1 for prob in probabilities))
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        for model_type, model in self.models.items():
            credit_model = CreditRiskModel(model_type, model=model)
            
            # Evaluate model
            metrics = credit_model.evaluate(self.X_test, self.y_test)
            
            # Check metrics are calculated
            expected_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                'confusion_matrix', 'classification_report'
            ]
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
            
            # Check metric values are reasonable
            self.assertTrue(0 <= metrics['accuracy'] <= 1)
            self.assertTrue(0 <= metrics['precision'] <= 1)
            self.assertTrue(0 <= metrics['recall'] <= 1)
            self.assertTrue(0 <= metrics['f1_score'] <= 1)
            self.assertTrue(0 <= metrics['roc_auc'] <= 1)
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        for model_type, model in self.models.items():
            credit_model = CreditRiskModel(model_type, model=model)
            
            # Create a temporary directory for model
            os.makedirs('temp_models', exist_ok=True)
            model_path = f'temp_models/test_{model_type}.pkl'
            
            # Save the model
            credit_model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model
            loaded_model = CreditRiskModel.load_model(model_path, model_type)
            
            # Check loaded model works
            predictions_original = credit_model.predict(self.X_test)
            predictions_loaded = loaded_model.predict(self.X_test)
            
            # Predictions should be identical
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
            
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
            
            if os.path.exists('temp_models'):
                os.rmdir('temp_models')


if __name__ == '__main__':
    unittest.main() 