"""
Model Performance Monitoring Module.

This module provides functionality to track model performance over time,
detect data drift, and generate alerts when model performance degrades.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yaml
import json
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import ks_2samp
from dataclasses import dataclass, asdict
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config file."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class MonitoringResult:
    """Data class for storing monitoring results."""
    timestamp: str
    model_id: str
    model_version: str
    period: str
    performance_metrics: dict
    data_drift_metrics: dict
    stability_metrics: dict
    alert_status: str
    alert_details: list


class ModelMonitor:
    """Class for monitoring model performance and data drift over time."""

    def __init__(self, model, config=None, model_id=None, model_version="1.0"):
        """
        Initialize the ModelMonitor.

        Args:
            model: Trained model object with predict and predict_proba methods
            config: Configuration dictionary
            model_id: Unique identifier for the model
            model_version: Version of the model
        """
        self.model = model
        self.config = config if config is not None else load_config()
        self.model_id = model_id or "default_model"
        self.model_version = model_version
        self.monitoring_results = []
        self.reference_data = None
        self.reference_features = None
        self.reference_target = None
        self.thresholds = {
            "auc_threshold": 0.05,
            "ks_threshold": 0.1,
            "psi_threshold": 0.2,
            "csi_threshold": 0.2,
        }
        if config and "validation" in config:
            if "psi_threshold" in config["validation"]:
                self.thresholds["psi_threshold"] = config["validation"]["psi_threshold"]

    def set_reference_data(self, X, y):
        """
        Set reference data for drift detection.

        Args:
            X: Feature data
            y: Target data
        """
        self.reference_features = X.copy()
        self.reference_target = y.copy()
        self.reference_data = {
            "X": X,
            "y": y,
            "predictions": self.model.predict(X),
            "probabilities": self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None,
            "feature_distributions": {col: X[col].describe().to_dict() for col in X.columns},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.info(f"Reference data set for model {self.model_id} (version {self.model_version})")

    def monitor_performance(self, X, y, period="current"):
        """
        Monitor model performance on new data.

        Args:
            X: Feature data for the current period
            y: Target data for the current period
            period: String identifier for the monitoring period

        Returns:
            MonitoringResult object with performance metrics and alerts
        """
        if self.reference_data is None:
            logger.warning("Reference data not set. Using current data as reference.")
            self.set_reference_data(X, y)
            return None

        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(y, predictions, probabilities)

        # Calculate data drift metrics
        data_drift_metrics = self._detect_data_drift(X)

        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(X, y, probabilities)

        # Generate alerts
        alert_status, alert_details = self._generate_alerts(
            performance_metrics, data_drift_metrics, stability_metrics
        )

        # Create monitoring result
        result = MonitoringResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_id=self.model_id,
            model_version=self.model_version,
            period=period,
            performance_metrics=performance_metrics,
            data_drift_metrics=data_drift_metrics,
            stability_metrics=stability_metrics,
            alert_status=alert_status,
            alert_details=alert_details
        )

        self.monitoring_results.append(result)
        return result

    def _calculate_performance_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate model performance metrics."""
        metrics = {}

        # Binary classification metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate basic metrics
        metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

        # Calculate AUC if probabilities are available
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            
            # Calculate precision-recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)
            
            # Calculate KS statistic
            pos_probs = y_prob[y_true == 1]
            neg_probs = y_prob[y_true == 0]
            if len(pos_probs) > 0 and len(neg_probs) > 0:
                metrics["ks_statistic"] = np.max(np.abs(
                    np.cumsum(pos_probs) / np.sum(pos_probs) - 
                    np.cumsum(neg_probs) / np.sum(neg_probs)
                ))
            else:
                metrics["ks_statistic"] = 0
                
            # Calculate Gini coefficient
            metrics["gini"] = 2 * metrics["roc_auc"] - 1 if "roc_auc" in metrics else 0

        return metrics

    def _detect_data_drift(self, X_current):
        """Detect drift in feature distributions."""
        drift_metrics = {
            "feature_drift": {},
            "overall_drift_score": 0.0,
            "drifted_features": []
        }
        
        ref_X = self.reference_features
        
        # Calculate Population Stability Index (PSI) for each feature
        overall_drift_score = 0
        num_features = len(X_current.columns)
        
        for col in X_current.columns:
            if col in ref_X.columns:
                # Different approach for numerical vs categorical features
                if pd.api.types.is_numeric_dtype(X_current[col]):
                    psi = self._calculate_psi(ref_X[col], X_current[col])
                    ks_stat, ks_pvalue = ks_2samp(ref_X[col].dropna(), X_current[col].dropna())
                    
                    drift_metrics["feature_drift"][col] = {
                        "psi": psi,
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "drift_detected": psi > self.thresholds["psi_threshold"] or ks_stat > self.thresholds["ks_threshold"]
                    }
                else:
                    # For categorical features
                    csi = self._calculate_csi(ref_X[col], X_current[col])
                    drift_metrics["feature_drift"][col] = {
                        "csi": csi,
                        "drift_detected": csi > self.thresholds["csi_threshold"]
                    }
                
                # Track drifted features
                if drift_metrics["feature_drift"][col]["drift_detected"]:
                    drift_metrics["drifted_features"].append(col)
                    overall_drift_score += 1
        
        drift_metrics["overall_drift_score"] = overall_drift_score / num_features if num_features > 0 else 0
        drift_metrics["drift_detected"] = len(drift_metrics["drifted_features"]) > 0
        
        return drift_metrics

    def _calculate_psi(self, expected, actual, bins=10):
        """
        Calculate Population Stability Index (PSI) for a numeric feature.
        
        PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        """
        if len(expected.dropna()) == 0 or len(actual.dropna()) == 0:
            return 0
            
        # Create bins based on expected distribution
        try:
            expected_quant = pd.qcut(expected.dropna(), q=bins, duplicates='drop')
            bin_edges = pd.qcut(expected.dropna(), q=bins, duplicates='drop', retbins=True)[1]
            
            # Ensure we have the full range by adding min/max boundaries if needed
            if bin_edges[0] > min(expected.min(), actual.min()):
                bin_edges = np.concatenate([[min(expected.min(), actual.min()) - 0.001], bin_edges])
            if bin_edges[-1] < max(expected.max(), actual.max()):
                bin_edges = np.concatenate([bin_edges, [max(expected.max(), actual.max()) + 0.001]])
            
            # Count observations in each bin
            expected_counts = pd.cut(expected.dropna(), bins=bin_edges).value_counts().sort_index()
            actual_counts = pd.cut(actual.dropna(), bins=bin_edges).value_counts().sort_index()
            
            # Align indices and fill missing bins with 0
            all_bins = expected_counts.index.union(actual_counts.index)
            expected_counts = expected_counts.reindex(all_bins, fill_value=0)
            actual_counts = actual_counts.reindex(all_bins, fill_value=0)
            
            # Convert to percentages
            expected_pct = expected_counts / expected_counts.sum()
            actual_pct = actual_counts / actual_counts.sum()
            
            # Add a small number to avoid division by zero or log(0)
            expected_pct = expected_pct.apply(lambda x: max(x, 0.0001))
            actual_pct = actual_pct.apply(lambda x: max(x, 0.0001))
            
            # Calculate PSI
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi = psi_values.sum()
            
            return psi
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0

    def _calculate_csi(self, expected, actual):
        """
        Calculate Characteristic Stability Index (CSI) for categorical features.
        Similar to PSI but for categorical variables.
        """
        if len(expected.dropna()) == 0 or len(actual.dropna()) == 0:
            return 0
            
        try:
            # Get value counts as percentages
            expected_counts = expected.value_counts(normalize=True)
            actual_counts = actual.value_counts(normalize=True)
            
            # Combine all categories
            all_categories = set(expected_counts.index).union(set(actual_counts.index))
            
            # Reindex and fill missing with small value
            expected_pct = expected_counts.reindex(all_categories, fill_value=0.0001)
            actual_pct = actual_counts.reindex(all_categories, fill_value=0.0001)
            
            # Ensure no zeros
            expected_pct = expected_pct.apply(lambda x: max(x, 0.0001))
            actual_pct = actual_pct.apply(lambda x: max(x, 0.0001))
            
            # Calculate CSI
            csi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            csi = csi_values.sum()
            
            return csi
        except Exception as e:
            logger.warning(f"Error calculating CSI: {e}")
            return 0

    def _calculate_stability_metrics(self, X, y, probabilities):
        """Calculate model stability metrics over time."""
        stability_metrics = {}
        
        # Compare performance with reference
        if self.reference_data["probabilities"] is not None and probabilities is not None:
            # Calculate reference performance
            ref_auc = roc_auc_score(self.reference_target, self.reference_data["probabilities"])
            
            # Calculate current performance
            current_auc = roc_auc_score(y, probabilities)
            
            # Calculate stability metrics
            stability_metrics["auc_delta"] = current_auc - ref_auc
            stability_metrics["auc_ratio"] = current_auc / ref_auc if ref_auc > 0 else float('inf')
            
            # Calculate score distribution stability
            stability_metrics["score_psi"] = self._calculate_psi(
                pd.Series(self.reference_data["probabilities"]),
                pd.Series(probabilities)
            )
            
            # Check for significant changes in score distribution
            stability_metrics["stable_score_distribution"] = stability_metrics["score_psi"] < self.thresholds["psi_threshold"]
            
            # Calculate threshold stability
            def optimal_threshold(y_true, y_score):
                # Find threshold that maximizes F1 score
                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                return thresholds[np.argmax(f1_scores)]
            
            ref_threshold = optimal_threshold(self.reference_target, self.reference_data["probabilities"])
            current_threshold = optimal_threshold(y, probabilities)
            
            stability_metrics["threshold_delta"] = current_threshold - ref_threshold
            stability_metrics["stable_threshold"] = abs(stability_metrics["threshold_delta"]) < 0.1
        
        return stability_metrics

    def _generate_alerts(self, performance_metrics, data_drift_metrics, stability_metrics):
        """Generate alerts based on monitoring results."""
        alerts = []
        alert_status = "OK"
        
        # Performance degradation alerts
        if "roc_auc" in performance_metrics and self.reference_data["probabilities"] is not None:
            ref_auc = roc_auc_score(self.reference_target, self.reference_data["probabilities"])
            current_auc = performance_metrics["roc_auc"]
            
            if current_auc < ref_auc - self.thresholds["auc_threshold"]:
                alerts.append({
                    "type": "PERFORMANCE_DEGRADATION",
                    "severity": "HIGH",
                    "message": f"Model AUC decreased from {ref_auc:.4f} to {current_auc:.4f}",
                    "metric": "roc_auc",
                    "threshold": self.thresholds["auc_threshold"],
                    "value": current_auc,
                    "reference": ref_auc
                })
                alert_status = "WARNING"
        
        # Data drift alerts
        if data_drift_metrics["drift_detected"]:
            drifted_features = data_drift_metrics["drifted_features"]
            alerts.append({
                "type": "DATA_DRIFT",
                "severity": "MEDIUM" if len(drifted_features) < 3 else "HIGH",
                "message": f"Data drift detected in {len(drifted_features)} features: {', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}",
                "features": drifted_features,
                "drift_score": data_drift_metrics["overall_drift_score"]
            })
            alert_status = "WARNING"
        
        # Score stability alerts
        if "score_psi" in stability_metrics and stability_metrics["score_psi"] > self.thresholds["psi_threshold"]:
            alerts.append({
                "type": "SCORE_DISTRIBUTION_SHIFT",
                "severity": "MEDIUM",
                "message": f"Score distribution has shifted: PSI = {stability_metrics['score_psi']:.4f}",
                "metric": "score_psi",
                "threshold": self.thresholds["psi_threshold"],
                "value": stability_metrics["score_psi"]
            })
            alert_status = "WARNING"
        
        # Critical alerts change status to CRITICAL
        critical_count = sum(1 for alert in alerts if alert["severity"] == "HIGH")
        if critical_count > 0:
            alert_status = "CRITICAL"
        
        return alert_status, alerts

    def plot_monitoring_results(self, output_dir=None):
        """
        Generate monitoring visualizations.
        
        Args:
            output_dir: Directory to save plots. If None, plots are displayed but not saved.
        """
        if not self.monitoring_results:
            logger.warning("No monitoring results available to plot")
            return
        
        # Create output directory if it doesn't exist
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        timestamps = [result.timestamp for result in self.monitoring_results]
        
        # Performance metrics over time
        self._plot_performance_over_time(timestamps, output_dir)
        
        # Data drift visualization
        self._plot_data_drift(timestamps, output_dir)
        
        # Alert frequency
        self._plot_alert_frequency(timestamps, output_dir)
    
    def _plot_performance_over_time(self, timestamps, output_dir=None):
        """Plot performance metrics over time."""
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "ks_statistic"]
        metric_data = {metric: [] for metric in metrics_to_plot}
        
        for result in self.monitoring_results:
            for metric in metrics_to_plot:
                if metric in result.performance_metrics:
                    metric_data[metric].append(result.performance_metrics[metric])
                else:
                    metric_data[metric].append(None)
        
        # Filter out metrics with no data
        metrics_to_plot = [m for m in metrics_to_plot if any(v is not None for v in metric_data[m])]
        
        if not metrics_to_plot:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric in metrics_to_plot:
            values = metric_data[metric]
            if any(v is not None for v in values):
                ax.plot(timestamps, values, marker='o', label=metric)
        
        ax.set_title("Model Performance Metrics Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "performance_over_time.png"))
            plt.close()
        else:
            plt.show()

    def _plot_data_drift(self, timestamps, output_dir=None):
        """Plot data drift metrics over time."""
        # Extract drift scores
        overall_drift = [result.data_drift_metrics.get("overall_drift_score", 0) for result in self.monitoring_results]
        
        # Count drifted features
        drifted_feature_counts = [len(result.data_drift_metrics.get("drifted_features", [])) for result in self.monitoring_results]
        
        # Plot overall drift and drifted feature counts
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Overall Drift Score', color=color)
        ax1.plot(timestamps, overall_drift, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Drifted Feature Count', color=color)
        ax2.plot(timestamps, drifted_feature_counts, marker='s', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Data Drift Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "data_drift_over_time.png"))
            plt.close()
        else:
            plt.show()
            
        # Get the top drifted features
        feature_drift_counts = {}
        for result in self.monitoring_results:
            for feature in result.data_drift_metrics.get("drifted_features", []):
                if feature in feature_drift_counts:
                    feature_drift_counts[feature] += 1
                else:
                    feature_drift_counts[feature] = 1
        
        # Plot top drifted features
        if feature_drift_counts:
            top_features = sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            features = [f[0] for f in top_features]
            counts = [f[1] for f in top_features]
            
            ax.barh(features, counts)
            ax.set_title("Top Drifted Features")
            ax.set_xlabel("Number of Drift Occurrences")
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, "top_drifted_features.png"))
                plt.close()
            else:
                plt.show()

    def _plot_alert_frequency(self, timestamps, output_dir=None):
        """Plot alert frequency over time."""
        alert_counts = [len(result.alert_details) for result in self.monitoring_results]
        alert_statuses = [result.alert_status for result in self.monitoring_results]
        
        # Create a mapping for status colors
        status_colors = {
            "OK": "green",
            "WARNING": "orange",
            "CRITICAL": "red"
        }
        
        # Get colors based on statuses
        colors = [status_colors.get(status, "gray") for status in alert_statuses]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(timestamps, alert_counts, color=colors)
        ax.set_title("Alert Frequency Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Number of Alerts")
        
        # Add a legend for the colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=status) 
                           for status, color in status_colors.items()
                           if status in alert_statuses]
        ax.legend(handles=legend_elements)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "alert_frequency.png"))
            plt.close()
        else:
            plt.show()

    def save_results(self, output_dir):
        """
        Save monitoring results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert monitoring results to JSON-serializable format
        results_json = []
        for result in self.monitoring_results:
            result_dict = asdict(result)
            results_json.append(result_dict)
        
        # Save monitoring results
        with open(os.path.join(output_dir, f"monitoring_results_{self.model_id}.json"), "w") as f:
            json.dump(results_json, f, indent=2)
        
        # Save the latest result as a separate file
        if self.monitoring_results:
            latest = asdict(self.monitoring_results[-1])
            with open(os.path.join(output_dir, f"latest_monitoring_{self.model_id}.json"), "w") as f:
                json.dump(latest, f, indent=2)
        
        # Generate and save visualizations
        self.plot_monitoring_results(output_dir)
        
        logger.info(f"Monitoring results saved to {output_dir}")


def run_monitoring(model, X_train, y_train, X_current, y_current, model_id=None, config=None):
    """
    Run model monitoring for a single period.
    
    Args:
        model: Trained model object
        X_train: Training features (reference data)
        y_train: Training targets (reference data)
        X_current: Current period features
        y_current: Current period targets
        model_id: Model identifier
        config: Configuration dictionary
        
    Returns:
        MonitoringResult object
    """
    monitor = ModelMonitor(model, config=config, model_id=model_id)
    monitor.set_reference_data(X_train, y_train)
    result = monitor.monitor_performance(X_current, y_current)
    return result, monitor


def simulate_monitoring_over_time(model, X_train, y_train, drift_scenarios, periods=6, model_id=None, config=None):
    """
    Simulate model monitoring over multiple time periods with drift scenarios.
    
    Args:
        model: Trained model object
        X_train: Training features
        y_train: Training targets
        drift_scenarios: Dict mapping period to drift function that creates drifted data
        periods: Number of periods to simulate
        model_id: Model identifier
        config: Configuration dictionary
        
    Returns:
        ModelMonitor instance with monitoring results
    """
    monitor = ModelMonitor(model, config=config, model_id=model_id)
    monitor.set_reference_data(X_train, y_train)
    
    # For each period, apply appropriate drift and monitor
    for period in range(1, periods + 1):
        period_name = f"period_{period}"
        
        # Apply drift scenario if available for this period
        if period in drift_scenarios:
            X_current, y_current = drift_scenarios[period](X_train.copy(), y_train.copy())
        else:
            # Use training data with small random noise if no drift scenario
            X_current = X_train.copy()
            for col in X_current.select_dtypes(include=['number']).columns:
                X_current[col] = X_current[col] * np.random.normal(1, 0.02, size=len(X_current))
            y_current = y_train.copy()
        
        # Monitor performance
        monitor.monitor_performance(X_current, y_current, period=period_name)
    
    return monitor


def main():
    """Main function to demonstrate model monitoring capability."""
    import os
    import sys
    
    # Add the parent directory to the path to import from other modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_processing.generate_synthetic_data import generate_credit_data
    from src.data_processing.preprocess import preprocess_data
    from src.model_development.models import CreditRiskModel, train_model
    
    # Load configuration
    config = load_config()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_credit_data(n_samples=5000, random_seed=42)
    
    # Preprocess data
    print("Preprocessing data...")
    target_variable = config['data']['target_variable']
    
    X_train = df.drop(columns=[target_variable])
    y_train = df[target_variable]
    
    # Train a model
    print("Training model...")
    model = train_model(X_train, y_train, model_type="logistic_regression")
    
    # Create some drift scenarios to simulate monitoring
    def create_slight_drift(X, y):
        X_drift = X.copy()
        for col in X_drift.select_dtypes(include=['number']).columns:
            X_drift[col] = X_drift[col] * np.random.normal(1, 0.05, size=len(X_drift))
        return X_drift, y
    
    def create_moderate_drift(X, y):
        X_drift = X.copy()
        for col in X_drift.select_dtypes(include=['number']).columns:
            X_drift[col] = X_drift[col] * np.random.normal(1.05, 0.1, size=len(X_drift))
        
        # Introduce some systematic drift in specific columns
        if 'income' in X_drift.columns:
            X_drift['income'] = X_drift['income'] * 1.15  # Simulate income inflation
        
        return X_drift, y
    
    def create_severe_drift(X, y):
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
        # Increase default rate by 20% 
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
    
    # Run monitoring simulation
    print("Simulating monitoring over time...")
    monitor = simulate_monitoring_over_time(
        model, X_train, y_train, 
        drift_scenarios, 
        periods=6,
        model_id="credit_risk_model_v1"
    )
    
    # Generate and save monitoring results
    output_dir = "reports/monitoring"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating monitoring visualizations...")
    monitor.plot_monitoring_results(output_dir)
    monitor.save_results(output_dir)
    
    print(f"Model monitoring complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
