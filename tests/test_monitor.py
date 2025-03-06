"""
Unit tests for the model monitoring module.
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
from src.monitoring.monitor import ModelMonitor, run_monitoring


class TestModelMonitor(unittest.TestCase):
    """Test cases for the ModelMonitor class."""
    
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
        
        # Split into train and current period sets
        self.X_train, self.X_current, self.y_train, self.y_current = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Create a drift in current data
        self.X_current_drift = self.X_current.copy()
        # Apply drift to some features
        for i in range(3):
            self.X_current_drift[f'feature_{i}'] = self.X_current_drift[f'feature_{i}'] * 1.5
        
        # Train a simple model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Create a simple configuration for testing
        self.config = {
            'validation': {
                'psi_threshold': 0.2,
                'monitoring_frequency': 'monthly'
            }
        }
        
        # Create monitor instance
        self.monitor = ModelMonitor(self.model, self.config, model_id="test_model")
        self.monitor.set_reference_data(self.X_train, self.y_train)
    
    def test_monitor_initialization(self):
        """Test that monitor initializes correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.model, self.model)
        self.assertEqual(self.monitor.model_id, "test_model")
        self.assertIsNotNone(self.monitor.reference_data)
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Run monitoring on current data (no drift)
        result = self.monitor.monitor_performance(self.X_current, self.y_current, period="current")
        
        # Check result structure
        self.assertEqual(result.model_id, "test_model")
        self.assertEqual(result.period, "current")
        self.assertIn('accuracy', result.performance_metrics)
        self.assertIn('precision', result.performance_metrics)
        self.assertIn('recall', result.performance_metrics)
        self.assertIn('f1_score', result.performance_metrics)
        self.assertIn('roc_auc', result.performance_metrics)
        
        # Check alert status (should be OK for non-drifted data)
        self.assertEqual(result.alert_status, "OK")
    
    def test_data_drift_detection(self):
        """Test data drift detection."""
        # Run monitoring on drifted data
        result = self.monitor.monitor_performance(self.X_current_drift, self.y_current, period="drift")
        
        # Check drift detection
        self.assertTrue(result.data_drift_metrics['drift_detected'])
        self.assertTrue(len(result.data_drift_metrics['drifted_features']) > 0)
        
        # Check alert status (should be WARNING or CRITICAL for drifted data)
        self.assertIn(result.alert_status, ["WARNING", "CRITICAL"])
        
        # Check alerts
        drift_alerts = [alert for alert in result.alert_details if alert['type'] == 'DATA_DRIFT']
        self.assertTrue(len(drift_alerts) > 0)
    
    def test_run_monitoring_function(self):
        """Test the run_monitoring function."""
        # Run monitoring
        result, monitor = run_monitoring(
            self.model, 
            self.X_train, self.y_train, 
            self.X_current, self.y_current, 
            model_id="test_model", 
            config=self.config
        )
        
        # Check results
        self.assertIsNotNone(result)
        self.assertEqual(result.model_id, "test_model")
        self.assertIn('accuracy', result.performance_metrics)
    
    def test_psi_calculation(self):
        """Test PSI calculation."""
        # Create series with known distributions
        expected = pd.Series(np.random.normal(0, 1, 1000))
        
        # Create actual with same distribution (should have low PSI)
        actual_same = pd.Series(np.random.normal(0, 1, 1000))
        psi_same = self.monitor._calculate_psi(expected, actual_same)
        
        # Create actual with different distribution (should have higher PSI)
        actual_diff = pd.Series(np.random.normal(1, 1.5, 1000))
        psi_diff = self.monitor._calculate_psi(expected, actual_diff)
        
        # PSI should be low for similar distributions
        self.assertLess(psi_same, 0.25)
        
        # PSI should be higher for different distributions
        self.assertGreater(psi_diff, psi_same)
    
    def test_multiple_monitoring_periods(self):
        """Test monitoring over multiple periods."""
        # First period - no drift
        self.monitor.monitor_performance(self.X_current, self.y_current, period="period_1")
        
        # Second period - slight drift
        X_slight_drift = self.X_current.copy()
        X_slight_drift['feature_0'] = X_slight_drift['feature_0'] * 1.2
        self.monitor.monitor_performance(X_slight_drift, self.y_current, period="period_2")
        
        # Third period - more drift
        self.monitor.monitor_performance(self.X_current_drift, self.y_current, period="period_3")
        
        # Check we have 3 monitoring results
        self.assertEqual(len(self.monitor.monitoring_results), 3)
        
        # The last period should have worse alert status
        self.assertIn(self.monitor.monitoring_results[2].alert_status, ["WARNING", "CRITICAL"])


if __name__ == '__main__':
    unittest.main() 