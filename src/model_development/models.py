#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit risk model development module.

This module provides functions for training, evaluating, and optimizing
different types of credit risk models, including logistic regression,
random forest, gradient boosting, and neural networks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import pickle
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna

def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class CreditRiskModel:
    """Base class for credit risk models."""
    
    def __init__(self, model_type, config=None, params=None):
        """
        Initialize the credit risk model.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'gradient_boosting', 'neural_network')
            config: Configuration dictionary
            params: Model parameters (overrides config)
        """
        self.model_type = model_type
        self.config = config or load_config()
        self.model = None
        self.feature_importance = None
        
        # Get default parameters from config if not provided
        if params is None and config is not None:
            self.params = self.config['models'].get(model_type, {})
        else:
            self.params = params or {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                penalty=self.params.get('penalty', 'l2'),
                C=self.params.get('C', 1.0),
                solver=self.params.get('solver', 'liblinear'),
                max_iter=self.params.get('max_iter', 100),
                random_state=42
            )
        
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 10),
                min_samples_split=self.params.get('min_samples_split', 10),
                min_samples_leaf=self.params.get('min_samples_leaf', 4),
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 5),
                random_state=42
            )
        
        elif self.model_type == 'neural_network':
            # Neural network will be initialized in fit method
            self.model = None
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            self: The fitted model instance
        """
        if self.model_type == 'neural_network':
            self._fit_neural_network(X, y)
        else:
            self.model.fit(X, y)
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
                ).sort_values(ascending=False)
            elif hasattr(self.model, 'coef_'):
                # For logistic regression
                self.feature_importance = pd.Series(
                    np.abs(self.model.coef_[0]),
                    index=X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
                ).sort_values(ascending=False)
        
        return self
    
    def _fit_neural_network(self, X, y):
        """
        Fit a neural network model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Get neural network parameters
        hidden_layers = self.params.get('hidden_layers', [64, 32])
        activation = self.params.get('activation', 'relu')
        dropout_rate = self.params.get('dropout_rate', 0.3)
        batch_size = self.params.get('batch_size', 64)
        epochs = self.params.get('epochs', 50)
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=X_array.shape[1], activation=activation))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Fit model with validation split
        model.fit(
            X_array, y_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.model = model
    
    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            y_pred: Binary predictions
        """
        if self.model_type == 'neural_network':
            # Convert probabilities to binary predictions
            probs = self.predict_proba(X)
            return (probs > 0.5).astype(int)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            y_proba: Predicted probabilities
        """
        if self.model_type == 'neural_network':
            # For neural network, return probabilities directly
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            return self.model.predict(X_array).flatten()
        else:
            # For sklearn models, get the positive class probability
            proba = self.model.predict_proba(X)
            return proba[:, 1]
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Probability threshold for binary predictions
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        # Predict probabilities and binary outcomes
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'average_precision': average_precision_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        }
        
        # Calculate Kolmogorov-Smirnov statistic
        # Sort by probability
        sorted_indices = np.argsort(y_proba)
        sorted_y = y[sorted_indices]
        sorted_proba = y_proba[sorted_indices]
        
        # Calculate cumulative distributions
        tpr = np.cumsum(sorted_y) / sum(sorted_y)
        fpr = np.cumsum(1 - sorted_y) / sum(1 - sorted_y)
        
        # Calculate KS statistic
        ks_statistic = np.max(np.abs(tpr - fpr))
        metrics['ks_statistic'] = ks_statistic
        
        # Calculate Gini coefficient
        gini = 2 * metrics['roc_auc'] - 1
        metrics['gini_coefficient'] = gini
        
        return metrics
    
    def plot_roc_curve(self, X, y, ax=None):
        """
        Plot the ROC curve.
        
        Args:
            X: Feature matrix
            y: True labels
            ax: Matplotlib axis
        
        Returns:
            ax: Matplotlib axis with ROC curve
        """
        y_proba = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'{self.model_type} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        
        return ax
    
    def plot_precision_recall_curve(self, X, y, ax=None):
        """
        Plot the Precision-Recall curve.
        
        Args:
            X: Feature matrix
            y: True labels
            ax: Matplotlib axis
        
        Returns:
            ax: Matplotlib axis with Precision-Recall curve
        """
        y_proba = self.predict_proba(X)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, label=f'{self.model_type} (AP = {ap:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        return ax
    
    def plot_feature_importance(self, top_n=20, ax=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            ax: Matplotlib axis
        
        Returns:
            ax: Matplotlib axis with feature importance plot
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model.")
            return None
        
        # Select top N features
        top_features = self.feature_importance.head(top_n)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        top_features.sort_values().plot(kind='barh', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.set_xlabel('Importance')
        
        return ax
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'neural_network':
            # Save Keras model
            self.model.save(filepath)
        else:
            # Save scikit-learn model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, filepath, model_type):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            model_type: Type of model
        
        Returns:
            model: Loaded model instance
        """
        instance = cls(model_type=model_type)
        
        if model_type == 'neural_network':
            # Load Keras model
            instance.model = tf.keras.models.load_model(filepath)
        else:
            # Load scikit-learn model
            with open(filepath, 'rb') as f:
                instance.model = pickle.load(f)
        
        return instance


def hyperparameter_tuning(X, y, model_type, param_grid=None, n_trials=50, cv=5):
    """
    Perform hyperparameter tuning using either grid search or Optuna.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model
        param_grid: Parameter grid for grid search
        n_trials: Number of Optuna trials
        cv: Number of cross-validation folds
    
    Returns:
        best_params: Best parameters
        best_score: Best cross-validation score
    """
    if model_type == 'neural_network':
        # Use Optuna for neural network
        return _tune_neural_network(X, y, n_trials=n_trials, cv=cv)
    else:
        # Use grid search for scikit-learn models
        return _grid_search(X, y, model_type, param_grid, cv=cv)


def _grid_search(X, y, model_type, param_grid=None, cv=5):
    """
    Perform grid search for scikit-learn models.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model
        param_grid: Parameter grid
        cv: Number of cross-validation folds
    
    Returns:
        best_params: Best parameters
        best_score: Best cross-validation score
    """
    # Default parameter grids if not provided
    if param_grid is None:
        if model_type == 'logistic_regression':
            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            raise ValueError(f"Unsupported model type for grid search: {model_type}")
    
    # Initialize model
    model = CreditRiskModel(model_type)
    
    # Initialize grid search
    grid_search = GridSearchCV(
        model.model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    print(f"Starting grid search for {model_type}...")
    start_time = time.time()
    grid_search.fit(X, y)
    elapsed_time = time.time() - start_time
    print(f"Grid search completed in {elapsed_time:.2f} seconds.")
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    return best_params, best_score


def _tune_neural_network(X, y, n_trials=50, cv=5):
    """
    Tune neural network hyperparameters using Optuna.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of Optuna trials
        cv: Number of cross-validation folds
    
    Returns:
        best_params: Best parameters
        best_score: Best cross-validation score
    """
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
        
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    def objective(trial):
        # Define hyperparameters to optimize
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'hidden_units_{i}', 16, 128))
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=X_array.shape[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        
        # Define cross-validation splits
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        scores = []
        for train_idx, val_idx in kf.split(X_array, y_array):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate on validation set
            y_proba = model.predict(X_val).flatten()
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)
        
        # Return mean score
        return np.mean(scores)
    
    # Create Optuna study
    print("Starting neural network hyperparameter tuning with Optuna...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    elapsed_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds.")
    
    # Get best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Convert hidden layers to format expected by CreditRiskModel
    n_layers = best_params.pop('n_layers')
    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(best_params.pop(f'hidden_units_{i}'))
    
    best_params['hidden_layers'] = hidden_layers
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    return best_params, best_score


def train_model(X_train, y_train, model_type, params=None, tune=False):
    """
    Train a credit risk model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        model_type: Type of model
        params: Model parameters
        tune: Whether to perform hyperparameter tuning
    
    Returns:
        model: Trained model
    """
    if tune:
        # Perform hyperparameter tuning
        best_params, _ = hyperparameter_tuning(X_train, y_train, model_type)
        
        # Update parameters with best parameters
        params = best_params
    
    # Initialize and fit model
    model = CreditRiskModel(model_type=model_type, params=params)
    model.fit(X_train, y_train)
    
    return model


def compare_models(X_train, y_train, X_test, y_test, model_types=None, params_dict=None):
    """
    Train and compare multiple models.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector
        model_types: List of model types to compare
        params_dict: Dictionary of model parameters
    
    Returns:
        models: Dictionary of trained models
        comparison: DataFrame with model comparison metrics
    """
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    if params_dict is None:
        params_dict = {}
    
    models = {}
    metrics = []
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        
        # Get parameters for this model type
        params = params_dict.get(model_type, None)
        
        # Train model
        model = train_model(X_train, y_train, model_type, params)
        models[model_type] = model
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        # Collect metrics for comparison
        metrics.append({
            'model_type': model_type,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'roc_auc': test_metrics['roc_auc'],
            'ks_statistic': test_metrics['ks_statistic'],
            'gini_coefficient': test_metrics['gini_coefficient']
        })
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(metrics).set_index('model_type')
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_type, model in models.items():
        model.plot_roc_curve(X_test, y_test, ax=ax)
    
    return models, comparison


def main():
    """Main function to demonstrate model development."""
    from src.data_processing.preprocess import load_data, preprocess_data
    
    # Load configuration
    config = load_config()
    
    # Get paths from config
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    target_variable = config['data']['target_variable']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
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
    
    # Prepare features and target
    X_train = train_preprocessed.drop(columns=[target_variable])
    y_train = train_preprocessed[target_variable]
    
    X_test = test_preprocessed.drop(columns=[target_variable])
    y_test = test_preprocessed[target_variable]
    
    # Compare models
    models, comparison = compare_models(X_train, y_train, X_test, y_test)
    
    # Display comparison
    print("\nModel Comparison:")
    print(comparison)
    
    # Save best model
    best_model_type = comparison['roc_auc'].idxmax()
    best_model = models[best_model_type]
    
    model_dir = Path(__file__).parents[2] / "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / f"{best_model_type}_model.pkl"
    
    best_model.save_model(model_path)
    print(f"\nBest model ({best_model_type}) saved to {model_path}")

if __name__ == "__main__":
    main() 