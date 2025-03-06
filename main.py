#!/usr/bin/env python
"""
Main script for demonstrating the Credit Risk Model Development and Validation Framework.

This script showcases the end-to-end workflow:
1. Data generation and preprocessing
2. Model development and training
3. Model validation
4. Stress testing
5. Model monitoring

Run this script to see the entire framework in action.
"""

import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import local modules
from src.data_processing.generate_synthetic_data import generate_credit_data, split_and_save_data
from src.data_processing.preprocess import preprocess_data, create_feature_pipeline
from src.model_development.models import train_model, compare_models, CreditRiskModel
from src.model_validation.validator import ModelValidator, validate_model
from src.stress_testing.stress_tester import StressTester, run_stress_test
from src.monitoring.monitor import ModelMonitor, simulate_monitoring_over_time


def load_config():
    """Load configuration from config file."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "data",
        "models",
        "reports",
        "reports/validation",
        "reports/monitoring",
        "reports/stress_testing"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Created necessary directories")


def step1_generate_data(config, n_samples=10000):
    """Generate and preprocess synthetic credit data."""
    logger.info("Step 1: Generating synthetic credit data")
    
    # Generate data
    data = generate_credit_data(n_samples=n_samples, random_seed=42)
    logger.info(f"Generated {len(data)} samples of synthetic credit data")
    
    # Split data
    train_data, test_data, val_data = split_and_save_data(
        data, output_dir='data', test_size=0.2, validation_size=0.1
    )
    logger.info(f"Split data into train ({len(train_data)}), test ({len(test_data)}), and validation ({len(val_data)}) sets")
    
    # Process data
    target_variable = config['data']['target_variable']
    pipeline = create_feature_pipeline(config, target_col=target_variable)
    
    X_train, y_train = preprocess_data(
        train_data, config, target_col=target_variable, is_training=True
    )
    
    X_test, y_test = preprocess_data(
        test_data, config, target_col=target_variable, is_training=False, preprocessing_pipeline=pipeline
    )
    
    X_val, y_val = preprocess_data(
        val_data, config, target_col=target_variable, is_training=False, preprocessing_pipeline=pipeline
    )
    
    logger.info("Completed data preprocessing")
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def step2_develop_model(X_train, y_train, X_test, y_test, model_types=['logistic_regression', 'random_forest', 'gradient_boosting']):
    """Develop and train credit risk models."""
    logger.info("Step 2: Developing credit risk models")
    
    # Compare different model types
    model_results = compare_models(X_train, y_train, X_test, y_test, model_types=model_types)
    logger.info(f"Model comparison results:\n{model_results}")
    
    # Select best model
    best_model_type = model_results.iloc[model_results['roc_auc'].idxmax()]['model_type']
    logger.info(f"Best model type: {best_model_type}")
    
    # Train best model with hyperparameter tuning
    logger.info(f"Training {best_model_type} model with hyperparameter tuning")
    best_model = train_model(X_train, y_train, model_type=best_model_type, tune=True)
    
    # Create model instance
    credit_model = CreditRiskModel(best_model_type, model=best_model)
    
    # Save model
    model_path = f'models/credit_risk_{best_model_type}.pkl'
    credit_model.save_model(model_path)
    logger.info(f"Saved best model to {model_path}")
    
    return credit_model


def step3_validate_model(credit_model, X_train, y_train, X_test, y_test, X_val, y_val, config):
    """Validate the credit risk model."""
    logger.info("Step 3: Validating credit risk model")
    
    # Perform validation
    validation_results = validate_model(
        credit_model.model, 
        X_train, y_train, 
        X_test, y_test, 
        X_val=X_val, y_val=y_val, 
        config=config
    )
    
    # Create validator for reports and visualization
    validator = ModelValidator(credit_model.model, config)
    validator.validate(X_train, y_train, X_test, y_test, X_val, y_val)
    
    # Generate validation report
    timestamp = datetime.now().strftime("%Y%m%d")
    report_path = f'reports/validation/validation_report_{timestamp}.md'
    validator.generate_report(report_path)
    logger.info(f"Generated validation report: {report_path}")
    
    # Generate visualizations
    validator.plot_validation_results('reports/validation')
    logger.info("Generated validation visualizations")
    
    overall_assessment = validation_results['overall_assessment']
    logger.info(f"Validation result: {overall_assessment['result']}")
    logger.info(f"Risk rating: {overall_assessment['risk_rating']}")
    
    return validation_results


def step4_stress_test_model(credit_model, X_test, y_test, config):
    """Perform stress testing on the model."""
    logger.info("Step 4: Performing stress testing")
    
    # Initialize stress tester
    stress_tester = StressTester(credit_model.model, config)
    
    # Run stress tests
    stress_results = stress_tester.run_stress_test(X_test, y_test)
    logger.info("Completed stress testing")
    
    # Calculate capital requirements
    capital_results = stress_tester.calculate_capital_requirements(X_test, portfolio_size=10000000, lgd=0.6)
    logger.info("Calculated capital requirements under stress scenarios")
    
    # Identify vulnerable segments
    vulnerable_segments = stress_tester.identify_vulnerable_segments(X_test, y_test)
    logger.info(f"Identified {len(vulnerable_segments['segments'])} potentially vulnerable segments")
    
    # Generate stress testing report
    timestamp = datetime.now().strftime("%Y%m%d")
    report_path = f'reports/stress_testing/stress_test_report_{timestamp}.md'
    stress_tester.generate_report(report_path)
    logger.info(f"Generated stress testing report: {report_path}")
    
    # Generate visualizations
    stress_tester.plot_stress_results('reports/stress_testing')
    logger.info("Generated stress testing visualizations")
    
    return stress_results


def step5_monitor_model(credit_model, X_train, y_train, config):
    """Set up model monitoring."""
    logger.info("Step 5: Setting up model monitoring")
    
    # Create monitor
    monitor = ModelMonitor(credit_model.model, config, model_id="credit_risk_model_v1")
    
    # Set reference data
    monitor.set_reference_data(X_train, y_train)
    logger.info("Set reference data for monitoring")
    
    # Create drift scenarios for simulation
    def create_slight_drift(X, y):
        """Create slight data drift."""
        X_drift = X.copy()
        for col in X_drift.select_dtypes(include=['number']).columns:
            X_drift[col] = X_drift[col] * np.random.normal(1, 0.05, size=len(X_drift))
        return X_drift, y

    def create_moderate_drift(X, y):
        """Create moderate data drift."""
        X_drift = X.copy()
        for col in X_drift.select_dtypes(include=['number']).columns:
            X_drift[col] = X_drift[col] * np.random.normal(1.05, 0.1, size=len(X_drift))
        
        # Introduce some systematic drift in specific columns
        if 'income' in X_drift.columns:
            X_drift['income'] = X_drift['income'] * 1.15  # Simulate income inflation
        
        return X_drift, y

    def create_severe_drift(X, y):
        """Create severe data drift and target shift."""
        X_drift = X.copy()
        
        # Apply severe drift to all numeric columns
        for col in X_drift.select_dtypes(include=['number']).columns:
            X_drift[col] = X_drift[col] * np.random.normal(1.1, 0.15, size=len(X_drift))
        
        # Introduce dramatic shifts in key variables
        if 'income' in X_drift.columns:
            X_drift['income'] = X_drift['income'] * 1.3  # Dramatic increase in income
        
        if 'debt_to_income' in X_drift.columns:
            X_drift['debt_to_income'] = X_drift['debt_to_income'] * 1.25  # Higher debt ratios
        
        # Simulate economic shock affecting performance and data distributions
        y_drift = y.copy()
        # Increase default rate by 20% if 'debt_to_income' is above median
        if 'debt_to_income' in X_drift.columns:
            high_risk_idx = X_drift['debt_to_income'] > X_drift['debt_to_income'].median()
            y_drift[high_risk_idx] = 1  # 1 is the default label
        
        return X_drift, y_drift
    
    # Define drift scenarios
    drift_scenarios = {
        1: lambda X, y: (X.copy(), y.copy()),  # No drift
        2: create_slight_drift,
        3: create_slight_drift,
        4: create_moderate_drift,
        5: create_moderate_drift, 
        6: create_severe_drift
    }
    
    # Simulate monitoring over time
    logger.info("Simulating model monitoring over 6 time periods")
    simulation_monitor = simulate_monitoring_over_time(
        credit_model.model, 
        X_train, y_train, 
        drift_scenarios, 
        periods=6,
        model_id="credit_risk_model_v1",
        config=config
    )
    
    # Save monitoring results
    simulation_monitor.save_results('reports/monitoring')
    logger.info("Saved monitoring results")
    
    # Check final monitoring results
    latest_result = simulation_monitor.monitoring_results[-1]
    critical_alerts = sum(1 for alert in latest_result.alert_details if alert['severity'] == 'HIGH')
    warning_alerts = sum(1 for alert in latest_result.alert_details if alert['severity'] == 'MEDIUM')
    
    if latest_result.alert_status == "CRITICAL" or critical_alerts > 0:
        recommendation = "Model retraining is REQUIRED. Critical issues detected."
    elif latest_result.alert_status == "WARNING" or warning_alerts > 0:
        recommendation = "Model retraining is RECOMMENDED. Multiple warning alerts detected."
    else:
        recommendation = "Model retraining is NOT necessary at this time."
        
    logger.info(f"Monitoring recommendation: {recommendation}")
    
    return simulation_monitor


def main():
    """Main function demonstrating the end-to-end workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Credit Risk Model Framework")
    parser.add_argument(
        "--samples", "-n", type=int, default=10000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--models", "-m", nargs='+', 
        default=['logistic_regression', 'random_forest', 'gradient_boosting'],
        help="Model types to compare"
    )
    parser.add_argument(
        "--skip-stress", action="store_true",
        help="Skip stress testing step"
    )
    parser.add_argument(
        "--skip-monitoring", action="store_true",
        help="Skip monitoring step"
    )
    args = parser.parse_args()
    
    # Set up environment
    setup_directories()
    config = load_config()
    
    # Step 1: Generate and preprocess data
    X_train, y_train, X_test, y_test, X_val, y_val = step1_generate_data(config, n_samples=args.samples)
    
    # Step 2: Develop model
    credit_model = step2_develop_model(X_train, y_train, X_test, y_test, model_types=args.models)
    
    # Step 3: Validate model
    validation_results = step3_validate_model(credit_model, X_train, y_train, X_test, y_test, X_val, y_val, config)
    
    # Step 4: Stress test model (optional)
    if not args.skip_stress:
        stress_results = step4_stress_test_model(credit_model, X_test, y_test, config)
    
    # Step 5: Monitor model (optional)
    if not args.skip_monitoring:
        monitoring_results = step5_monitor_model(credit_model, X_train, y_train, config)
    
    logger.info("Completed end-to-end workflow demonstration")
    logger.info("Explore the reports directory to see detailed results")


if __name__ == "__main__":
    main() 