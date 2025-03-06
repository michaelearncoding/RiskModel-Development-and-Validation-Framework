#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit risk model stress testing module.

This module provides functions for stress testing credit risk models
under different economic scenarios, including normal, recession, 
and severe recession conditions. It evaluates how model performance
and risk predictions change under these varying scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report

def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class StressTester:
    """Class for stress testing credit risk models."""
    
    def __init__(self, model, config=None):
        """
        Initialize the stress tester.
        
        Args:
            model: Trained model object with predict_proba method
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or load_config()
        self.stress_results = {}
        self.stress_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get stress scenarios from config
        self.scenarios = self.config['stress_testing']['scenarios']
    
    def apply_macroeconomic_stress(self, X, scenario_name):
        """
        Apply macroeconomic stress conditions to the data.
        
        Args:
            X: Feature matrix to modify with stress conditions
            scenario_name: Name of the scenario to apply
        
        Returns:
            X_stressed: Stressed feature matrix
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available scenarios: {list(self.scenarios.keys())}")
        
        # Get scenario parameters
        scenario = self.scenarios[scenario_name]
        
        # Create a copy of the data to modify
        X_stressed = X.copy()
        
        # Apply stress to macroeconomic variables
        if 'unemployment_rate' in X_stressed.columns and 'unemployment_rate' in scenario:
            # Apply unemployment stress (could be direct replacement or multiplicative)
            X_stressed['unemployment_rate'] = scenario['unemployment_rate']
            
        if 'interest_rate' in X_stressed.columns and 'interest_rate' in scenario:
            # Apply interest rate stress
            X_stressed['interest_rate'] = scenario['interest_rate']
            
        if 'gdp_growth' in X_stressed.columns and 'gdp_growth' in scenario:
            # Apply GDP growth stress
            X_stressed['gdp_growth'] = scenario['gdp_growth']
        
        # Apply additional stress effects to credit-related variables
        # In a recession, credit scores tend to be lower, utilization higher, etc.
        stress_intensity = self._get_scenario_intensity(scenario_name)
        
        # Impact on credit scores (-5% to -15% based on severity)
        if 'credit_score' in X_stressed.columns:
            credit_score_impact = 1.0 - (stress_intensity * 0.15)
            X_stressed['credit_score'] = X_stressed['credit_score'] * credit_score_impact
            
        # Impact on utilization rates (+10% to +30% based on severity)
        if 'utilization_rate' in X_stressed.columns:
            utilization_impact = 1.0 + (stress_intensity * 0.3)
            X_stressed['utilization_rate'] = np.minimum(X_stressed['utilization_rate'] * utilization_impact, 1.0)
            
        # Impact on debt-to-income (+5% to +25% based on severity)
        if 'debt_to_income' in X_stressed.columns:
            dti_impact = 1.0 + (stress_intensity * 0.25)
            X_stressed['debt_to_income'] = np.minimum(X_stressed['debt_to_income'] * dti_impact, 1.0)
            
        # Impact on delinquency history (+20% to +100% based on severity)
        if 'delinquency_history' in X_stressed.columns:
            delinquency_impact = 1.0 + (stress_intensity * 1.0)
            X_stressed['delinquency_history'] = X_stressed['delinquency_history'] * delinquency_impact
        
        return X_stressed
    
    def _get_scenario_intensity(self, scenario_name):
        """
        Get stress intensity factor based on scenario name.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            intensity: Stress intensity factor (0.0-1.0)
        """
        # Define intensity based on scenario severity
        if scenario_name == 'baseline':
            return 0.0
        elif scenario_name == 'moderate_recession':
            return 0.5
        elif scenario_name == 'severe_recession':
            return 1.0
        else:
            # Default to moderate stress
            return 0.5
    
    def run_stress_test(self, X, y=None):
        """
        Run stress tests for all defined scenarios.
        
        Args:
            X: Feature matrix
            y: Target vector (optional, for performance evaluation)
        
        Returns:
            stress_results: Dictionary with stress test results
        """
        print("Running stress tests...")
        
        # Get baseline predictions
        baseline_proba = self.model.predict_proba(X)
        baseline_pred = (baseline_proba >= 0.5).astype(int)
        
        # Initialize results dictionary
        results = {
            'stress_date': self.stress_date,
            'scenarios': {},
            'model_type': self.model.model_type if hasattr(self.model, 'model_type') else 'unknown'
        }
        
        # Store baseline metrics
        baseline_metrics = {
            'mean_probability': np.mean(baseline_proba),
            'predicted_default_rate': np.mean(baseline_pred),
            'probability_distribution': self._get_probability_distribution(baseline_proba)
        }
        
        if y is not None:
            baseline_metrics.update({
                'accuracy': np.mean(baseline_pred == y),
                'roc_auc': roc_auc_score(y, baseline_proba)
            })
        
        results['scenarios']['baseline'] = baseline_metrics
        
        # Run each stress scenario
        for scenario_name in self.scenarios:
            if scenario_name == 'baseline':
                continue  # Skip baseline as we've already calculated it
                
            print(f"Running scenario: {scenario_name}")
            
            # Apply stress conditions
            X_stressed = self.apply_macroeconomic_stress(X, scenario_name)
            
            # Get predictions under stress
            stress_proba = self.model.predict_proba(X_stressed)
            stress_pred = (stress_proba >= 0.5).astype(int)
            
            # Calculate stress metrics
            stress_metrics = {
                'mean_probability': np.mean(stress_proba),
                'predicted_default_rate': np.mean(stress_pred),
                'probability_distribution': self._get_probability_distribution(stress_proba),
                'change_from_baseline': {
                    'mean_probability_change': np.mean(stress_proba - baseline_proba),
                    'mean_probability_pct_change': (np.mean(stress_proba) / np.mean(baseline_proba) - 1) * 100,
                    'default_rate_change': np.mean(stress_pred) - np.mean(baseline_pred),
                    'default_rate_pct_change': (np.mean(stress_pred) / np.mean(baseline_pred) - 1) * 100 if np.mean(baseline_pred) > 0 else np.inf,
                    'prediction_changes': np.sum(stress_pred != baseline_pred),
                    'prediction_changes_pct': np.mean(stress_pred != baseline_pred) * 100
                }
            }
            
            if y is not None:
                stress_metrics.update({
                    'accuracy': np.mean(stress_pred == y),
                    'roc_auc': roc_auc_score(y, stress_proba),
                    'performance_change': {
                        'accuracy_change': np.mean(stress_pred == y) - np.mean(baseline_pred == y),
                        'auc_change': roc_auc_score(y, stress_proba) - roc_auc_score(y, baseline_proba)
                    }
                })
            
            results['scenarios'][scenario_name] = stress_metrics
        
        # Calculate scenario sensitivity (how much predictions change across scenarios)
        all_probas = [results['scenarios'][s]['mean_probability'] for s in results['scenarios']]
        results['scenario_sensitivity'] = {
            'probability_range': max(all_probas) - min(all_probas),
            'probability_std': np.std(all_probas),
            'probability_coefficient_variation': np.std(all_probas) / np.mean(all_probas) if np.mean(all_probas) > 0 else np.inf,
            'sensitivity_rating': self._get_sensitivity_rating(all_probas)
        }
        
        self.stress_results = results
        return results
    
    def _get_probability_distribution(self, probas, n_bins=10):
        """
        Calculate probability distribution across bins.
        
        Args:
            probas: Probability predictions
            n_bins: Number of bins
            
        Returns:
            distribution: List of bin counts
        """
        hist, bin_edges = np.histogram(probas, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Convert to percentage of population
        distribution = {
            'bin_centers': bin_centers.tolist(),
            'bin_percentages': (hist / len(probas)).tolist()
        }
        
        return distribution
    
    def _get_sensitivity_rating(self, probas):
        """
        Get a qualitative rating of model sensitivity to stress.
        
        Args:
            probas: List of probability means across scenarios
            
        Returns:
            rating: Sensitivity rating (Low, Medium, High)
        """
        cv = np.std(probas) / np.mean(probas) if np.mean(probas) > 0 else np.inf
        
        if cv < 0.1:
            return "Low"
        elif cv < 0.3:
            return "Medium"
        else:
            return "High"
    
    def calculate_capital_requirements(self, X, portfolio_size=1000000, lgd=0.6):
        """
        Calculate expected loss and capital requirements under different scenarios.
        
        Args:
            X: Feature matrix
            portfolio_size: Total portfolio size in currency units
            lgd: Loss Given Default (fraction of exposure lost when default occurs)
            
        Returns:
            capital_results: Dictionary with capital requirement calculations
        """
        if not self.stress_results:
            raise ValueError("No stress results available. Run run_stress_test() first.")
        
        print("Calculating capital requirements...")
        
        # Initialize results
        capital_results = {}
        
        # Calculate for each scenario
        for scenario_name, metrics in self.stress_results['scenarios'].items():
            # Get predicted default rate for this scenario
            pd = metrics['predicted_default_rate']
            
            # Calculate expected loss (EL = PD * LGD * EAD)
            expected_loss = pd * lgd * portfolio_size
            
            # Calculate unexpected loss (simplified approach)
            # Using Basel formula (simplified): UL = LGD * sqrt(PD * (1-PD))
            unexpected_loss_pct = lgd * np.sqrt(pd * (1 - pd))
            unexpected_loss = unexpected_loss_pct * portfolio_size
            
            # Capital requirement (covering unexpected loss)
            capital_requirement = unexpected_loss
            
            # Store results
            capital_results[scenario_name] = {
                'probability_of_default': pd,
                'loss_given_default': lgd,
                'exposure_at_default': portfolio_size,
                'expected_loss': expected_loss,
                'unexpected_loss': unexpected_loss,
                'capital_requirement': capital_requirement,
                'capital_requirement_pct': capital_requirement / portfolio_size * 100
            }
        
        # Calculate increase in capital needed under stress
        baseline_capital = capital_results['baseline']['capital_requirement']
        for scenario_name in capital_results:
            if scenario_name != 'baseline':
                scenario_capital = capital_results[scenario_name]['capital_requirement']
                capital_results[scenario_name]['capital_increase'] = scenario_capital - baseline_capital
                capital_results[scenario_name]['capital_increase_pct'] = (scenario_capital / baseline_capital - 1) * 100
        
        return capital_results
    
    def identify_vulnerable_segments(self, X, y=None, segments=None):
        """
        Identify customer segments most vulnerable to stress conditions.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            segments: Dictionary mapping segment names to boolean masks
            
        Returns:
            vulnerability_results: Dictionary with vulnerability analysis
        """
        if not self.stress_results:
            raise ValueError("No stress results available. Run run_stress_test() first.")
        
        print("Identifying vulnerable segments...")
        
        # If no segments provided, create some based on available features
        if segments is None:
            segments = self._create_default_segments(X)
        
        # Initialize results
        vulnerability_results = {}
        
        # Get baseline predictions
        baseline_proba = self.model.predict_proba(X)
        
        # For each segment, analyze vulnerability across scenarios
        for segment_name, segment_mask in segments.items():
            segment_results = {}
            
            # Skip empty segments
            if np.sum(segment_mask) == 0:
                continue
            
            # Get baseline metrics for this segment
            segment_baseline_proba = baseline_proba[segment_mask]
            segment_baseline_default_rate = np.mean(segment_baseline_proba >= 0.5)
            
            segment_results['baseline'] = {
                'segment_size': np.sum(segment_mask),
                'segment_size_pct': np.mean(segment_mask) * 100,
                'mean_probability': np.mean(segment_baseline_proba),
                'predicted_default_rate': segment_baseline_default_rate
            }
            
            if y is not None:
                segment_y = y[segment_mask]
                segment_results['baseline']['actual_default_rate'] = np.mean(segment_y)
            
            # For each stress scenario, calculate segment metrics
            max_default_increase = 0
            max_scenario = 'baseline'
            
            for scenario_name in self.scenarios:
                if scenario_name == 'baseline':
                    continue
                
                # Apply stress to full dataset (needed for feature interactions)
                X_stressed = self.apply_macroeconomic_stress(X, scenario_name)
                
                # Get predictions for this segment under stress
                segment_stress_proba = self.model.predict_proba(X_stressed)[segment_mask]
                segment_stress_default_rate = np.mean(segment_stress_proba >= 0.5)
                
                # Calculate changes
                default_rate_change = segment_stress_default_rate - segment_baseline_default_rate
                default_rate_pct_change = (segment_stress_default_rate / segment_baseline_default_rate - 1) * 100 if segment_baseline_default_rate > 0 else np.inf
                
                segment_results[scenario_name] = {
                    'mean_probability': np.mean(segment_stress_proba),
                    'predicted_default_rate': segment_stress_default_rate,
                    'default_rate_change': default_rate_change,
                    'default_rate_pct_change': default_rate_pct_change
                }
                
                # Track scenario with largest impact
                if default_rate_change > max_default_increase:
                    max_default_increase = default_rate_change
                    max_scenario = scenario_name
            
            # Add vulnerability assessment
            segment_results['vulnerability'] = {
                'max_default_increase': max_default_increase,
                'max_default_increase_pct': (segment_results[max_scenario]['predicted_default_rate'] / segment_results['baseline']['predicted_default_rate'] - 1) * 100 if segment_results['baseline']['predicted_default_rate'] > 0 else np.inf,
                'most_impactful_scenario': max_scenario,
                'vulnerability_rating': self._get_vulnerability_rating(max_default_increase)
            }
            
            vulnerability_results[segment_name] = segment_results
        
        # Rank segments by vulnerability
        segment_rankings = []
        for segment_name, results in vulnerability_results.items():
            segment_rankings.append({
                'segment': segment_name,
                'max_default_increase': results['vulnerability']['max_default_increase'],
                'vulnerability_rating': results['vulnerability']['vulnerability_rating']
            })
        
        segment_rankings.sort(key=lambda x: x['max_default_increase'], reverse=True)
        
        return {
            'segment_analysis': vulnerability_results,
            'segment_rankings': segment_rankings
        }
    
    def _create_default_segments(self, X):
        """
        Create default customer segments based on available features.
        
        Args:
            X: Feature matrix
            
        Returns:
            segments: Dictionary mapping segment names to boolean masks
        """
        segments = {}
        
        # Check for key segmentation variables and create segments
        if 'credit_score' in X.columns:
            cs = X['credit_score']
            segments['low_credit_score'] = cs < 600
            segments['medium_credit_score'] = (cs >= 600) & (cs < 720)
            segments['high_credit_score'] = cs >= 720
        
        if 'income' in X.columns:
            inc = X['income']
            segments['low_income'] = inc < inc.quantile(0.33)
            segments['medium_income'] = (inc >= inc.quantile(0.33)) & (inc < inc.quantile(0.67))
            segments['high_income'] = inc >= inc.quantile(0.67)
        
        if 'debt_to_income' in X.columns:
            dti = X['debt_to_income']
            segments['low_dti'] = dti < 0.2
            segments['medium_dti'] = (dti >= 0.2) & (dti < 0.4)
            segments['high_dti'] = dti >= 0.4
        
        if 'employment_status' in X.columns:
            segments['employed'] = X['employment_status'] == 'Employed'
            segments['self_employed'] = X['employment_status'] == 'Self-Employed'
            segments['unemployed'] = X['employment_status'] == 'Unemployed'
        
        if 'age' in X.columns:
            segments['young'] = X['age'] < 30
            segments['middle_aged'] = (X['age'] >= 30) & (X['age'] < 50)
            segments['senior'] = X['age'] >= 50
        
        return segments
    
    def _get_vulnerability_rating(self, default_increase):
        """
        Get a qualitative rating of segment vulnerability.
        
        Args:
            default_increase: Increase in default rate under stress
            
        Returns:
            rating: Vulnerability rating (Low, Medium, High)
        """
        if default_increase < 0.05:  # Less than 5 percentage points increase
            return "Low"
        elif default_increase < 0.15:  # Less than 15 percentage points increase
            return "Medium"
        else:
            return "High"
    
    def generate_report(self, output_path=None):
        """
        Generate a stress testing report.
        
        Args:
            output_path: Path to save the report (default: stress_test_report.json)
        
        Returns:
            report_path: Path to the saved report
        """
        if not self.stress_results:
            raise ValueError("No stress results available. Run run_stress_test() first.")
        
        if output_path is None:
            # Create default path in reports directory
            output_dir = Path(__file__).parents[2] / "reports"
            os.makedirs(output_dir, exist_ok=True)
            
            model_type = self.stress_results.get('model_type', 'model')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{model_type}_stress_test_{timestamp}.json"
        
        # Save report as JSON
        with open(output_path, 'w') as f:
            json.dump(self.stress_results, f, indent=2, default=str)
        
        print(f"Stress test report saved to {output_path}")
        return output_path
    
    def plot_stress_results(self, output_dir=None):
        """
        Generate stress testing plots.
        
        Args:
            output_dir: Directory to save plots
        
        Returns:
            output_paths: List of paths to the saved plots
        """
        if not self.stress_results:
            raise ValueError("No stress results available. Run run_stress_test() first.")
        
        if output_dir is None:
            # Create default directory in reports
            output_dir = Path(__file__).parents[2] / "reports" / "plots"
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        # 1. Default Rate by Scenario
        scenarios = list(self.stress_results['scenarios'].keys())
        default_rates = [self.stress_results['scenarios'][s]['predicted_default_rate'] for s in scenarios]
        
        plt.figure(figsize=(10, 6))
        bar_colors = ['blue' if s == 'baseline' else 'orange' if s == 'moderate_recession' else 'red' for s in scenarios]
        plt.bar(scenarios, default_rates, color=bar_colors)
        plt.title('Predicted Default Rate by Scenario')
        plt.ylabel('Default Rate')
        plt.xticks(rotation=45)
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        default_rate_path = output_dir / "stress_default_rates.png"
        plt.savefig(default_rate_path)
        plt.close()
        output_paths.append(default_rate_path)
        
        # 2. Probability Distributions
        plt.figure(figsize=(12, 8))
        
        for scenario in scenarios:
            dist = self.stress_results['scenarios'][scenario]['probability_distribution']
            plt.plot(dist['bin_centers'], dist['bin_percentages'], 
                     label=scenario, 
                     linewidth=3 if scenario == 'baseline' else 2,
                     linestyle='-' if scenario == 'baseline' else '--')
        
        plt.title('Score Distribution by Scenario')
        plt.xlabel('Probability Score')
        plt.ylabel('Percentage of Population')
        plt.legend()
        plt.tight_layout()
        
        dist_path = output_dir / "stress_score_distributions.png"
        plt.savefig(dist_path)
        plt.close()
        output_paths.append(dist_path)
        
        # 3. Performance Impact (if available)
        if 'roc_auc' in self.stress_results['scenarios']['baseline']:
            auc_values = [self.stress_results['scenarios'][s].get('roc_auc', 0) for s in scenarios]
            
            plt.figure(figsize=(10, 6))
            plt.bar(scenarios, auc_values, color=bar_colors)
            plt.title('Model Performance (AUC) by Scenario')
            plt.ylabel('AUC')
            plt.xticks(rotation=45)
            plt.ylim(bottom=0.5)  # AUC is between 0.5 and 1
            plt.tight_layout()
            
            auc_path = output_dir / "stress_auc_performance.png"
            plt.savefig(auc_path)
            plt.close()
            output_paths.append(auc_path)
        
        return output_paths


def run_stress_test(model, X, y=None, config=None):
    """
    Run comprehensive stress tests on a model.
    
    Args:
        model: Trained model object
        X: Feature matrix
        y: Target vector (optional)
        config: Configuration dictionary
    
    Returns:
        stress_tester: StressTester instance with results
    """
    stress_tester = StressTester(model, config)
    
    # Run basic stress tests
    stress_tester.run_stress_test(X, y)
    
    # Calculate capital requirements
    capital_results = stress_tester.calculate_capital_requirements(X)
    stress_tester.stress_results['capital_requirements'] = capital_results
    
    # Identify vulnerable segments
    if y is not None:
        vulnerability_results = stress_tester.identify_vulnerable_segments(X, y)
        stress_tester.stress_results['vulnerable_segments'] = vulnerability_results
    
    # Generate report and plots
    stress_tester.generate_report()
    stress_tester.plot_stress_results()
    
    return stress_tester


def main():
    """Main function to demonstrate stress testing."""
    from src.data_processing.preprocess import load_data, preprocess_data
    from src.model_development.models import load_config, CreditRiskModel
    import glob
    
    # Load configuration
    config = load_config()
    
    # Get paths from config
    test_path = config['data']['test_path']
    target_variable = config['data']['target_variable']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    test_df = load_data(test_path)
    
    # Ensure we have macroeconomic variables for stress testing
    if not any(col in test_df.columns for col in ['unemployment_rate', 'interest_rate', 'gdp_growth']):
        print("Warning: No macroeconomic variables found in data. Adding dummy variables.")
        test_df['unemployment_rate'] = 5.0  # Baseline unemployment rate
        test_df['interest_rate'] = 3.0  # Baseline interest rate
        test_df['gdp_growth'] = 2.0  # Baseline GDP growth
    
    # Preprocess data
    preprocessed, _ = preprocess_data(test_df, config, target_col=target_variable)
    
    # Prepare features and target
    X_test = preprocessed.drop(columns=[target_variable])
    y_test = preprocessed[target_variable]
    
    # Load the best model
    model_dir = Path(__file__).parents[2] / "models"
    model_files = glob.glob(str(model_dir / "*.pkl"))
    
    if not model_files:
        print("No model files found. Training a new model...")
        from src.model_development.models import train_model
        
        # Get train data for training a new model
        train_path = config['data']['train_path']
        train_df = load_data(train_path)
        train_preprocessed, _ = preprocess_data(train_df, config, target_col=target_variable)
        X_train = train_preprocessed.drop(columns=[target_variable])
        y_train = train_preprocessed[target_variable]
        
        model = train_model(X_train, y_train, 'random_forest')
    else:
        # Load the most recent model
        model_path = sorted(model_files)[-1]
        model_type = os.path.basename(model_path).split('_')[0]
        print(f"Loading model from {model_path}...")
        model = CreditRiskModel.load_model(model_path, model_type)
    
    # Run stress tests
    stress_tester = run_stress_test(model, X_test, y_test, config)
    
    # Print key stress test results
    results = stress_tester.stress_results
    print("\nKey stress test results:")
    for scenario in results['scenarios']:
        print(f"{scenario} scenario:")
        print(f"  Predicted default rate: {results['scenarios'][scenario]['predicted_default_rate']:.4f}")
        if scenario != 'baseline':
            change = results['scenarios'][scenario]['change_from_baseline']['default_rate_pct_change']
            print(f"  Change from baseline: {change:.2f}%")
    
    print(f"\nScenario sensitivity: {results['scenario_sensitivity']['sensitivity_rating']}")
    
    if 'capital_requirements' in results:
        print("\nCapital requirements:")
        for scenario in results['capital_requirements']:
            cap_req = results['capital_requirements'][scenario]['capital_requirement']
            cap_pct = results['capital_requirements'][scenario]['capital_requirement_pct']
            print(f"  {scenario}: {cap_req:.2f} ({cap_pct:.2f}% of portfolio)")
    
    if 'vulnerable_segments' in results and 'segment_rankings' in results['vulnerable_segments']:
        print("\nMost vulnerable segments:")
        for i, segment in enumerate(results['vulnerable_segments']['segment_rankings'][:3]):
            print(f"  {i+1}. {segment['segment']} - {segment['vulnerability_rating']} vulnerability")

if __name__ == "__main__":
    main() 