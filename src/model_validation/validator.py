#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit risk model validation module.

This module provides functions and classes for validating credit risk models,
including performance testing, stability analysis, and sensitivity testing.
The validation follows regulatory guidelines for model risk management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, StratifiedKFold
import shap
import scipy.stats as stats

def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ModelValidator:
    """Class for validating credit risk models."""
    
    def __init__(self, model, config=None):
        """
        Initialize the model validator.
        
        Args:
            model: Trained model object with predict_proba method
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or load_config()
        self.validation_results = {}
        self.validation_date = datetime.now().strftime("%Y-%m-%d")
    
    def validate(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        """
        Run comprehensive model validation.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            y_test: Test target vector
            X_val: Validation feature matrix (optional)
            y_val: Validation target vector (optional)
        
        Returns:
            validation_results: Dictionary with validation results
        """
        print("Beginning comprehensive model validation...")
        
        # Use validation set if provided, otherwise use test set
        X_val = X_val if X_val is not None else X_test
        y_val = y_val if y_val is not None else y_test
        
        # Performance testing
        performance_results = self.performance_testing(X_train, y_train, X_test, y_test)
        
        # Discrimination power
        discrimination_results = self.assess_discrimination(X_test, y_test)
        
        # Calibration assessment
        calibration_results = self.assess_calibration(X_test, y_test)
        
        # Stability testing
        stability_results = self.stability_testing(X_train, y_train, X_test, y_test)
        
        # Sensitivity analysis
        sensitivity_results = self.sensitivity_analysis(X_test, y_test)
        
        # Benchmark comparison
        benchmark_results = self.benchmark_comparison(X_test, y_test)
        
        # Collect all results
        self.validation_results = {
            'validation_date': self.validation_date,
            'model_type': self.model.model_type if hasattr(self.model, 'model_type') else 'unknown',
            'performance': performance_results,
            'discrimination': discrimination_results,
            'calibration': calibration_results,
            'stability': stability_results,
            'sensitivity': sensitivity_results,
            'benchmark': benchmark_results
        }
        
        print("Model validation completed.")
        return self.validation_results
    
    def performance_testing(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model performance on training and test sets.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            y_test: Test target vector
        
        Returns:
            results: Dictionary with performance metrics
        """
        print("Performing performance testing...")
        
        # Function to calculate metrics
        def calculate_metrics(X, y, set_name):
            y_proba = self.model.predict_proba(X)
            y_pred = (y_proba >= 0.5).astype(int)
            
            # Basic classification metrics
            metrics = {
                f'{set_name}_accuracy': accuracy_score(y, y_pred),
                f'{set_name}_precision': precision_score(y, y_pred),
                f'{set_name}_recall': recall_score(y, y_pred),
                f'{set_name}_f1': f1_score(y, y_pred),
                f'{set_name}_roc_auc': roc_auc_score(y, y_proba),
                f'{set_name}_average_precision': average_precision_score(y, y_proba),
                f'{set_name}_brier_score': brier_score_loss(y, y_proba),
            }
            
            # Calculate Kolmogorov-Smirnov statistic
            # Sort by probability
            sorted_indices = np.argsort(y_proba)
            sorted_y = y.iloc[sorted_indices] if hasattr(y, 'iloc') else y[sorted_indices]
            sorted_proba = y_proba[sorted_indices]
            
            # Calculate cumulative distributions
            tpr = np.cumsum(sorted_y) / sum(sorted_y)
            fpr = np.cumsum(1 - sorted_y) / sum(1 - sorted_y)
            
            # Calculate KS statistic
            ks_statistic = np.max(np.abs(tpr - fpr))
            metrics[f'{set_name}_ks_statistic'] = ks_statistic
            
            # Calculate Gini coefficient
            gini = 2 * metrics[f'{set_name}_roc_auc'] - 1
            metrics[f'{set_name}_gini_coefficient'] = gini
            
            return metrics
        
        # Calculate metrics for training and test sets
        train_metrics = calculate_metrics(X_train, y_train, 'train')
        test_metrics = calculate_metrics(X_test, y_test, 'test')
        
        # Combine results
        results = {**train_metrics, **test_metrics}
        
        # Calculate overfitting indicators
        results['roc_auc_difference'] = train_metrics['train_roc_auc'] - test_metrics['test_roc_auc']
        results['accuracy_difference'] = train_metrics['train_accuracy'] - test_metrics['test_accuracy']
        results['overfitting_indicator'] = results['roc_auc_difference'] > 0.05  # Flag if difference > 5%
        
        return results
    
    def assess_discrimination(self, X, y):
        """
        Assess the model's discrimination power.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            results: Dictionary with discrimination metrics
        """
        print("Assessing discrimination power...")
        
        # Calculate probabilities
        y_proba = self.model.predict_proba(X)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        
        # Calculate Gini coefficient
        gini = 2 * roc_auc - 1
        
        # Calculate KS statistic and find the threshold that maximizes it
        ks_values = np.abs(tpr - fpr)
        ks_statistic = np.max(ks_values)
        ks_threshold = thresholds[np.argmax(ks_values)]
        
        # Create bins for score distribution analysis
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate default rates by score bin
        default_rates = []
        population_pcts = []
        for i in range(n_bins):
            bin_mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
            if np.sum(bin_mask) > 0:
                bin_default_rate = np.mean(y.iloc[bin_mask] if hasattr(y, 'iloc') else y[bin_mask])
                bin_population_pct = np.sum(bin_mask) / len(y_proba)
                default_rates.append(bin_default_rate)
                population_pcts.append(bin_population_pct)
            else:
                default_rates.append(np.nan)
                population_pcts.append(0)
        
        # Calculate concentration metrics
        # Percent of population in top 3 score bins
        top_concentration = np.sum(population_pcts[-3:])
        # Percent of defaults captured in bottom 3 score bins
        default_concentration = np.sum([dr * pp for dr, pp in zip(default_rates[:3], population_pcts[:3])]) / np.mean(y)
        
        # Results
        results = {
            'roc_auc': roc_auc,
            'gini_coefficient': gini,
            'ks_statistic': ks_statistic,
            'ks_threshold': ks_threshold,
            'top_score_concentration': top_concentration,
            'bottom_default_concentration': default_concentration,
            'score_bins': bin_centers.tolist(),
            'default_rates': default_rates,
            'population_pcts': population_pcts,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        return results
    
    def assess_calibration(self, X, y):
        """
        Assess model calibration (reliability of probability estimates).
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            results: Dictionary with calibration metrics
        """
        print("Assessing model calibration...")
        
        # Calculate probabilities
        y_proba = self.model.predict_proba(X)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10, strategy='quantile')
        
        # Calculate mean absolute calibration error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        # Calculate Hosmer-Lemeshow statistic
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        observed = []
        expected = []
        sizes = []
        
        for i in range(n_bins):
            bin_mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
            if np.sum(bin_mask) > 0:
                bin_y = y.iloc[bin_mask] if hasattr(y, 'iloc') else y[bin_mask]
                bin_proba = y_proba[bin_mask]
                
                observed.append(np.sum(bin_y))
                expected.append(np.sum(bin_proba))
                sizes.append(len(bin_y))
        
        # Calculate Hosmer-Lemeshow statistic
        hl_statistic = np.sum([(o - e)**2 / (e * (1 - e/s)) for o, e, s in zip(observed, expected, sizes) if s > 0 and 0 < e < s])
        
        # Calculate p-value (chi-squared distribution with n_bins-2 degrees of freedom)
        hl_pvalue = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
        
        # Results
        results = {
            'calibration_curve_observed': prob_true.tolist(),
            'calibration_curve_predicted': prob_pred.tolist(),
            'calibration_error': calibration_error,
            'hosmer_lemeshow_statistic': hl_statistic,
            'hosmer_lemeshow_pvalue': hl_pvalue,
            'well_calibrated': hl_pvalue > 0.05  # p-value > 0.05 indicates good calibration
        }
        
        return results
    
    def stability_testing(self, X_train, y_train, X_test, y_test):
        """
        Test model stability using cross-validation and out-of-time validation.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            y_test: Test target vector
        
        Returns:
            results: Dictionary with stability metrics
        """
        print("Performing stability testing...")
        
        # Cross-validation stability
        n_folds = 5
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            # Train a model on this fold (assuming the model has a fit method)
            if hasattr(self.model, 'model_type'):
                from src.model_development.models import CreditRiskModel
                fold_model = CreditRiskModel(self.model.model_type)
                fold_model.fit(X_fold_train, y_fold_train)
                y_proba = fold_model.predict_proba(X_fold_val)
            else:
                # If using a pre-trained model without fit method, use it directly
                # This is less accurate for stability testing but can work as a fallback
                y_proba = self.model.predict_proba(X_fold_val)
            
            cv_scores.append(roc_auc_score(y_fold_val, y_proba))
        
        # Calculate coefficient of variation for stability
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_variation = cv_std / cv_mean if cv_mean > 0 else np.nan
        
        # Calculate Population Stability Index (PSI)
        # Here we'll use a simplified approach by binning model scores
        def calculate_psi(expected, actual, bins=10):
            """Calculate Population Stability Index between two score distributions."""
            # Bin both distributions
            min_score = min(expected.min(), actual.min())
            max_score = max(expected.max(), actual.max())
            
            bins = np.linspace(min_score, max_score, bins + 1)
            
            expected_counts, _ = np.histogram(expected, bins=bins)
            actual_counts, _ = np.histogram(actual, bins=bins)
            
            # Add a small value to avoid division by zero
            expected_pct = expected_counts / float(sum(expected_counts)) + 1e-6
            actual_pct = actual_counts / float(sum(actual_counts)) + 1e-6
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            return psi
        
        # Calculate PSI between training and test scores
        train_scores = self.model.predict_proba(X_train)
        test_scores = self.model.predict_proba(X_test)
        
        psi_value = calculate_psi(train_scores, test_scores)
        
        # Results
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_variation': cv_variation,
            'model_stability_good': cv_variation < 0.1,  # CV < 10% indicates good stability
            'psi': psi_value,
            'population_stability_good': psi_value < 0.1  # PSI < 0.1 indicates good stability
        }
        
        return results
    
    def sensitivity_analysis(self, X, y):
        """
        Perform sensitivity analysis on key features.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            results: Dictionary with sensitivity analysis results
        """
        print("Performing sensitivity analysis...")
        
        # Get variables to test sensitivity on
        sensitive_vars = self.config['validation'].get('sensitivity_variables', [])
        
        # If no variables specified or none are in X, choose top features
        if not sensitive_vars or not any(var in X.columns for var in sensitive_vars):
            # Try to get feature importances from model if available
            if hasattr(self.model, 'feature_importance') and self.model.feature_importance is not None:
                sensitive_vars = list(self.model.feature_importance.head(3).index)
            else:
                # If no feature importance available, use SHAP values
                try:
                    explainer = shap.Explainer(self.model.predict, X.iloc[:100] if hasattr(X, 'iloc') else X[:100])
                    shap_values = explainer(X.iloc[:100] if hasattr(X, 'iloc') else X[:100])
                    mean_shap = np.abs(shap_values.values).mean(0)
                    sensitive_vars = list(X.columns[np.argsort(mean_shap)[-3:]])
                except:
                    # Fallback: just use first 3 columns
                    sensitive_vars = list(X.columns[:3])
        
        # Filter to only variables in X
        sensitive_vars = [var for var in sensitive_vars if var in X.columns]
        
        if not sensitive_vars:
            print("No valid sensitivity variables found.")
            return {'no_sensitivity_vars': True}
        
        # Baseline performance
        baseline_proba = self.model.predict_proba(X)
        baseline_auc = roc_auc_score(y, baseline_proba)
        
        # Test sensitivity by perturbing each variable
        sensitivity_results = {}
        
        for var in sensitive_vars:
            # Get the variable's statistics
            if hasattr(X, 'iloc'):
                var_mean = X[var].mean()
                var_std = X[var].std()
            else:
                var_mean = np.mean(X[:, list(X.columns).index(var)])
                var_std = np.std(X[:, list(X.columns).index(var)])
            
            variations = {}
            perturbations = [-2, -1, 1, 2]  # Multiples of standard deviation
            
            for p in perturbations:
                # Create a perturbed copy of X
                X_perturbed = X.copy()
                
                # Apply perturbation
                if hasattr(X, 'iloc'):
                    X_perturbed[var] = X_perturbed[var] + p * var_std
                else:
                    X_perturbed[:, list(X.columns).index(var)] = X_perturbed[:, list(X.columns).index(var)] + p * var_std
                
                # Make predictions with perturbed data
                perturbed_proba = self.model.predict_proba(X_perturbed)
                perturbed_auc = roc_auc_score(y, perturbed_proba)
                
                # Calculate change
                auc_change = perturbed_auc - baseline_auc
                auc_change_pct = auc_change / baseline_auc * 100 if baseline_auc > 0 else np.inf
                
                # Average absolute score change
                score_change = np.mean(np.abs(perturbed_proba - baseline_proba))
                
                variations[f'perturbation_{p}'] = {
                    'auc': perturbed_auc,
                    'auc_change': auc_change,
                    'auc_change_pct': auc_change_pct,
                    'avg_score_change': score_change
                }
            
            # Calculate overall sensitivity metrics for this variable
            max_auc_change_pct = max([abs(v['auc_change_pct']) for k, v in variations.items()])
            avg_score_change = np.mean([v['avg_score_change'] for k, v in variations.items()])
            
            sensitivity_results[var] = {
                'variations': variations,
                'max_auc_change_pct': max_auc_change_pct,
                'avg_score_change': avg_score_change,
                'sensitivity_rating': 'High' if max_auc_change_pct > 10 else ('Medium' if max_auc_change_pct > 5 else 'Low')
            }
        
        # Overall sensitivity assessment
        max_sensitivities = [result['max_auc_change_pct'] for var, result in sensitivity_results.items()]
        overall_sensitivity = 'High' if any(s > 10 for s in max_sensitivities) else ('Medium' if any(s > 5 for s in max_sensitivities) else 'Low')
        
        results = {
            'analyzed_variables': sensitive_vars,
            'variable_results': sensitivity_results,
            'overall_sensitivity': overall_sensitivity,
            'baseline_auc': baseline_auc
        }
        
        return results
    
    def benchmark_comparison(self, X, y):
        """
        Compare model to simple benchmarks.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            results: Dictionary with benchmark comparison results
        """
        print("Comparing model to benchmarks...")
        
        # Get model predictions and performance
        model_proba = self.model.predict_proba(X)
        model_auc = roc_auc_score(y, model_proba)
        
        # 1. Benchmark: Random model (AUC = 0.5)
        random_auc = 0.5
        random_lift = (model_auc - random_auc) / random_auc * 100 if random_auc > 0 else np.inf
        
        # 2. Benchmark: Simple logistic regression
        from sklearn.linear_model import LogisticRegression
        
        # Train a simple model with default parameters
        simple_model = LogisticRegression(random_state=42)
        simple_model.fit(X, y)
        
        simple_proba = simple_model.predict_proba(X)[:, 1]
        simple_auc = roc_auc_score(y, simple_proba)
        
        simple_lift = (model_auc - simple_auc) / simple_auc * 100 if simple_auc > 0 else np.inf
        
        # 3. If model has a single most important feature, use that as a benchmark
        single_feature_auc = None
        single_feature_lift = None
        
        if hasattr(self.model, 'feature_importance') and self.model.feature_importance is not None:
            top_feature = self.model.feature_importance.index[0]
            
            if top_feature in X.columns:
                # Use the top feature as a single-feature model
                single_feature_model = LogisticRegression(random_state=42)
                if hasattr(X, 'iloc'):
                    single_feature_model.fit(X[[top_feature]], y)
                    single_feature_proba = single_feature_model.predict_proba(X[[top_feature]])[:, 1]
                else:
                    # Reshape for sklearn
                    single_feature_idx = list(X.columns).index(top_feature)
                    single_feature_model.fit(X[:, single_feature_idx].reshape(-1, 1), y)
                    single_feature_proba = single_feature_model.predict_proba(X[:, single_feature_idx].reshape(-1, 1))[:, 1]
                
                single_feature_auc = roc_auc_score(y, single_feature_proba)
                single_feature_lift = (model_auc - single_feature_auc) / single_feature_auc * 100 if single_feature_auc > 0 else np.inf
        
        # Compare to regulatory minimums (typically AUC > 0.7 for credit risk models)
        regulatory_min_auc = 0.7
        meets_regulatory_min = model_auc >= regulatory_min_auc
        
        # Results
        results = {
            'model_auc': model_auc,
            'random_auc': random_auc,
            'random_comparison': {
                'lift_percentage': random_lift,
                'significant_improvement': model_auc > 0.55  # >5% better than random
            },
            'simple_model_auc': simple_auc,
            'simple_model_comparison': {
                'lift_percentage': simple_lift,
                'significant_improvement': simple_lift > 5  # >5% better than simple model
            },
            'meets_regulatory_minimum': meets_regulatory_min
        }
        
        if single_feature_auc is not None:
            results['single_feature_auc'] = single_feature_auc
            results['single_feature_comparison'] = {
                'feature_name': top_feature,
                'lift_percentage': single_feature_lift,
                'significant_improvement': single_feature_lift > 5  # >5% better than single feature
            }
        
        return results
    
    def generate_report(self, output_path=None):
        """
        Generate a model validation report.
        
        Args:
            output_path: Path to save the report (default: validation_report.json)
        
        Returns:
            report_path: Path to the saved report
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate() first.")
        
        if output_path is None:
            # Create default path in reports directory
            output_dir = Path(__file__).parents[2] / "reports"
            os.makedirs(output_dir, exist_ok=True)
            
            model_type = self.validation_results.get('model_type', 'model')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{model_type}_validation_{timestamp}.json"
        
        # Save report as JSON
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"Validation report saved to {output_path}")
        return output_path
    
    def plot_validation_results(self, output_dir=None):
        """
        Generate validation plots.
        
        Args:
            output_dir: Directory to save plots
        
        Returns:
            output_paths: List of paths to the saved plots
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate() first.")
        
        if output_dir is None:
            # Create default directory in reports
            output_dir = Path(__file__).parents[2] / "reports" / "plots"
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        # 1. ROC Curve
        if 'discrimination' in self.validation_results:
            fpr = self.validation_results['discrimination']['fpr']
            tpr = self.validation_results['discrimination']['tpr']
            roc_auc = self.validation_results['discrimination']['roc_auc']
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            roc_path = output_dir / "roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            output_paths.append(roc_path)
        
        # 2. Calibration Curve
        if 'calibration' in self.validation_results:
            observed = self.validation_results['calibration']['calibration_curve_observed']
            predicted = self.validation_results['calibration']['calibration_curve_predicted']
            
            plt.figure(figsize=(10, 8))
            plt.plot(predicted, observed, 's-', label='Calibration curve')
            plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Observed fraction of positives')
            plt.title('Calibration Curve')
            plt.legend(loc="lower right")
            
            calibration_path = output_dir / "calibration_curve.png"
            plt.savefig(calibration_path)
            plt.close()
            output_paths.append(calibration_path)
        
        # 3. Score Distribution
        if 'discrimination' in self.validation_results:
            bins = self.validation_results['discrimination']['score_bins']
            default_rates = self.validation_results['discrimination']['default_rates']
            population_pcts = self.validation_results['discrimination']['population_pcts']
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Bar chart for population distribution
            ax1.bar(range(len(bins)), population_pcts, alpha=0.6, color='blue', label='Population %')
            ax1.set_xlabel('Score Bin')
            ax1.set_ylabel('Population %', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Line chart for default rates
            ax2 = ax1.twinx()
            ax2.plot(range(len(bins)), default_rates, 'ro-', label='Default Rate')
            ax2.set_ylabel('Default Rate', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title('Score Distribution and Default Rates')
            plt.xticks(range(len(bins)), [f"{b:.2f}" for b in bins], rotation=45)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
            
            plt.tight_layout()
            
            dist_path = output_dir / "score_distribution.png"
            plt.savefig(dist_path)
            plt.close()
            output_paths.append(dist_path)
        
        return output_paths


def validate_model(model, X_train, y_train, X_test, y_test, X_val=None, y_val=None, config=None):
    """
    Validate a model using the ModelValidator.
    
    Args:
        model: Trained model object
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector
        X_val: Validation feature matrix (optional)
        y_val: Validation target vector (optional)
        config: Configuration dictionary
    
    Returns:
        validator: ModelValidator instance with results
    """
    validator = ModelValidator(model, config)
    validator.validate(X_train, y_train, X_test, y_test, X_val, y_val)
    
    # Generate report and plots
    validator.generate_report()
    validator.plot_validation_results()
    
    return validator


def main():
    """Main function to demonstrate model validation."""
    from src.data_processing.preprocess import load_data, preprocess_data
    from src.model_development.models import load_config, CreditRiskModel
    import glob
    
    # Load configuration
    config = load_config()
    
    # Get paths from config
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    validation_path = config['data']['validation_path']
    target_variable = config['data']['target_variable']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    val_df = load_data(validation_path)
    
    train_preprocessed, pipeline = preprocess_data(
        train_df, 
        config, 
        target_col=target_variable, 
        is_training=True
    )
    
    test_preprocessed, _ = preprocess_data(
        test_df, 
        config, 
        target_col=target_variable, 
        is_training=False, 
        preprocessing_pipeline=pipeline
    )
    
    val_preprocessed, _ = preprocess_data(
        val_df, 
        config, 
        target_col=target_variable, 
        is_training=False, 
        preprocessing_pipeline=pipeline
    )
    
    # Prepare features and target
    X_train = train_preprocessed.drop(columns=[target_variable])
    y_train = train_preprocessed[target_variable]
    
    X_test = test_preprocessed.drop(columns=[target_variable])
    y_test = test_preprocessed[target_variable]
    
    X_val = val_preprocessed.drop(columns=[target_variable])
    y_val = val_preprocessed[target_variable]
    
    # Load the best model
    model_dir = Path(__file__).parents[2] / "models"
    model_files = glob.glob(str(model_dir / "*.pkl"))
    
    if not model_files:
        print("No model files found. Training a new model...")
        from src.model_development.models import train_model
        model = train_model(X_train, y_train, 'random_forest')
    else:
        # Load the most recent model
        model_path = sorted(model_files)[-1]
        model_type = os.path.basename(model_path).split('_')[0]
        print(f"Loading model from {model_path}...")
        model = CreditRiskModel.load_model(model_path, model_type)
    
    # Validate the model
    validator = validate_model(model, X_train, y_train, X_test, y_test, X_val, y_val, config)
    
    # Print key validation results
    results = validator.validation_results
    print("\nKey validation results:")
    print(f"Model performance (Test AUC): {results['discrimination']['roc_auc']:.4f}")
    print(f"Model discrimination (Gini): {results['discrimination']['gini_coefficient']:.4f}")
    print(f"Model calibration error: {results['calibration']['calibration_error']:.4f}")
    print(f"Population stability (PSI): {results['stability']['psi']:.4f}")
    print(f"Overall sensitivity: {results['sensitivity']['overall_sensitivity']}")
    print(f"Benchmark comparison (vs. simple model): {results['benchmark']['simple_model_comparison']['lift_percentage']:.2f}% lift")

if __name__ == "__main__":
    main() 