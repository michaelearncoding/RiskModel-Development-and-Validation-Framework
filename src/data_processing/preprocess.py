#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit risk data preprocessing module.

This module contains functions for preprocessing credit risk data,
including handling missing values, feature encoding, transformation,
and normalization to prepare data for model development.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce
import yaml
from pathlib import Path

def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path):
    """
    Load credit risk data from CSV file.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Pandas DataFrame with loaded data
    """
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime
    if 'application_date' in df.columns:
        df['application_date'] = pd.to_datetime(df['application_date'])
    
    return df

def get_feature_lists(config):
    """
    Extract feature lists from configuration.
    
    Args:
        config: Project configuration dictionary
        
    Returns:
        Dictionary with categorical, numerical and date column lists
    """
    cat_cols = config['features']['categorical_columns']
    num_cols = config['features']['numerical_columns']
    date_cols = config['features'].get('date_columns', [])
    
    return {
        'categorical': cat_cols,
        'numerical': num_cols,
        'date': date_cols
    }

def extract_date_features(df, date_columns):
    """
    Extract features from date columns.
    
    Args:
        df: DataFrame with date columns
        date_columns: List of date column names
        
    Returns:
        DataFrame with extracted date features
    """
    df_copy = df.copy()
    
    for col in date_columns:
        if col in df_copy.columns:
            # Extract useful date components
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            
            # Calculate days since a reference date (for time series features)
            reference_date = pd.Timestamp('2018-01-01')
            df_copy[f'days_since_ref_{col}'] = (df_copy[col] - reference_date).dt.days
    
    return df_copy

def handle_missing_values(df, num_cols, cat_cols, strategy='knn'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with missing values
        num_cols: List of numerical columns
        cat_cols: List of categorical columns
        strategy: Imputation strategy ('knn', 'mean', 'median', 'most_frequent')
        
    Returns:
        DataFrame with imputed values
    """
    df_copy = df.copy()
    
    # Check for missing values
    missing_stats = df_copy[num_cols + cat_cols].isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].index.tolist()
    
    if not missing_cols:
        print("No missing values found.")
        return df_copy
    
    print(f"Handling missing values with strategy: {strategy}")
    print(f"Columns with missing values: {missing_cols}")
    
    # Handle missing numerical values
    num_missing = [col for col in missing_cols if col in num_cols]
    if num_missing:
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_copy[num_missing] = imputer.fit_transform(df_copy[num_missing])
        else:
            imputer = SimpleImputer(strategy=strategy)
            df_copy[num_missing] = imputer.fit_transform(df_copy[num_missing])
    
    # Handle missing categorical values
    cat_missing = [col for col in missing_cols if col in cat_cols]
    if cat_missing:
        imputer = SimpleImputer(strategy='most_frequent')
        df_copy[cat_missing] = imputer.fit_transform(df_copy[cat_missing])
    
    return df_copy

def encode_categorical_features(df, cat_cols, method='onehot', target_col=None):
    """
    Encode categorical features using specified method.
    
    Args:
        df: DataFrame with categorical features
        cat_cols: List of categorical column names
        method: Encoding method ('onehot', 'ordinal', 'target', 'woe')
        target_col: Target column name for target encoding
        
    Returns:
        DataFrame with encoded features and encoding objects
    """
    df_copy = df.copy()
    encoders = {}
    
    if not cat_cols:
        return df_copy, encoders
    
    print(f"Encoding categorical features with method: {method}")
    
    if method == 'onehot':
        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df_copy[cat_cols])
        feature_names = encoder.get_feature_names_out(cat_cols)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_copy.index)
        
        # Drop original categorical columns and append encoded ones
        df_copy = df_copy.drop(columns=cat_cols)
        df_copy = pd.concat([df_copy, encoded_df], axis=1)
        
        encoders['onehot'] = encoder
        
    elif method == 'ordinal':
        # Ordinal encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols])
        encoders['ordinal'] = encoder
        
    elif method == 'target' and target_col is not None:
        # Target encoding
        encoder = ce.TargetEncoder(cols=cat_cols)
        df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols], df_copy[target_col])
        encoders['target'] = encoder
        
    elif method == 'woe' and target_col is not None:
        # Weight of Evidence encoding
        encoder = ce.WOEEncoder(cols=cat_cols)
        df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols], df_copy[target_col])
        encoders['woe'] = encoder
    
    return df_copy, encoders

def scale_numerical_features(df, num_cols, method='standard'):
    """
    Scale numerical features using specified method.
    
    Args:
        df: DataFrame with numerical features
        num_cols: List of numerical column names
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        DataFrame with scaled features and scaler object
    """
    df_copy = df.copy()
    
    if not num_cols:
        return df_copy, None
    
    # Select columns that exist in the dataframe
    valid_cols = [col for col in num_cols if col in df_copy.columns]
    
    print(f"Scaling numerical features with method: {method}")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        # Default to standard scaling
        scaler = StandardScaler()
    
    # Fit and transform
    df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
    
    return df_copy, scaler

def create_feature_pipeline(config, target_col=None):
    """
    Create a scikit-learn pipeline for feature preprocessing.
    
    Args:
        config: Project configuration
        target_col: Target column name for supervised encoding methods
        
    Returns:
        Scikit-learn ColumnTransformer pipeline
    """
    feature_lists = get_feature_lists(config)
    cat_cols = feature_lists['categorical']
    num_cols = feature_lists['numerical']
    
    # Get preprocessing settings from config
    imputation_strategy = config['features']['transformation']['imputation_strategy']
    scaling_method = config['features']['transformation']['scaling_method']
    encoding_method = config['features']['transformation']['encoding_method']
    
    # Create transformers list
    transformers = []
    
    # Numerical feature transformer
    if num_cols:
        if imputation_strategy == 'knn':
            num_pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler())
            ])
        else:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=imputation_strategy)),
                ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler())
            ])
        
        transformers.append(('num', num_pipeline, num_cols))
    
    # Categorical feature transformer
    if cat_cols:
        if encoding_method == 'onehot':
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
        elif encoding_method == 'ordinal':
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        elif encoding_method == 'target' and target_col is not None:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', ce.TargetEncoder(cols=cat_cols))
            ])
        else:
            # Default to one-hot encoding
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
        
        transformers.append(('cat', cat_pipeline, cat_cols))
    
    # Create and return column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    return preprocessor

def preprocess_data(df, config, target_col=None, is_training=True, preprocessing_pipeline=None):
    """
    Preprocess data using configuration settings.
    
    Args:
        df: DataFrame to preprocess
        config: Project configuration
        target_col: Target column name
        is_training: Whether this is training data (to fit transformers) or test data
        preprocessing_pipeline: Optional existing pipeline to use for transformation
        
    Returns:
        Preprocessed DataFrame and preprocessing objects
    """
    print(f"Preprocessing {'training' if is_training else 'test/validation'} data...")
    
    # Get feature lists
    feature_lists = get_feature_lists(config)
    cat_cols = feature_lists['categorical']
    num_cols = feature_lists['numerical']
    date_cols = feature_lists['date']
    
    # Extract date features
    if date_cols:
        df = extract_date_features(df, date_cols)
    
    # Create or use preprocessing pipeline
    if is_training or preprocessing_pipeline is None:
        preprocessing_pipeline = create_feature_pipeline(config, target_col)
        # Fit and transform
        features = preprocessing_pipeline.fit_transform(df)
    else:
        # Transform only using existing pipeline
        features = preprocessing_pipeline.transform(df)
    
    # Get transformed column names
    feature_names = []
    
    # Extract feature names from the pipeline
    for name, transformer, columns in preprocessing_pipeline.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # For one-hot encoding, get the feature names
            if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                cat_feature_names = transformer.named_steps['encoder'].get_feature_names_out(columns)
                feature_names.extend(cat_feature_names)
            else:
                # For other encoders, keep original column names
                feature_names.extend(columns)
    
    # Add passthrough columns
    if preprocessing_pipeline.remainder == 'passthrough':
        passthrough_cols = [col for col in df.columns if col not in cat_cols + num_cols]
        if passthrough_cols:
            feature_names.extend(passthrough_cols)
    
    # Create DataFrame with transformed features
    preprocessed_df = pd.DataFrame(features, columns=feature_names, index=df.index)
    
    # Add target column if available
    if target_col is not None and target_col in df.columns:
        preprocessed_df[target_col] = df[target_col].values
    
    print(f"Preprocessed data shape: {preprocessed_df.shape}")
    return preprocessed_df, preprocessing_pipeline

def main():
    """Main function to demonstrate preprocessing pipeline."""
    # Load configuration
    config = load_config()
    
    # Get paths from config
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    target_variable = config['data']['target_variable']
    
    # Load data
    print(f"Loading training data from {train_path}")
    train_df = load_data(train_path)
    
    print(f"Loading test data from {test_path}")
    test_df = load_data(test_path)
    
    # Preprocess training data
    train_preprocessed, pipeline = preprocess_data(
        train_df, 
        config, 
        target_col=target_variable, 
        is_training=True
    )
    
    # Preprocess test data using the same pipeline
    test_preprocessed, _ = preprocess_data(
        test_df, 
        config, 
        target_col=target_variable, 
        is_training=False, 
        preprocessing_pipeline=pipeline
    )
    
    # Display preprocessing results
    print("\nTraining data sample after preprocessing:")
    print(train_preprocessed.head())
    
    print("\nTest data sample after preprocessing:")
    print(test_preprocessed.head())
    
    # Check for any remaining missing values
    train_missing = train_preprocessed.isnull().sum().sum()
    test_missing = test_preprocessed.isnull().sum().sum()
    
    print(f"\nMissing values after preprocessing - Train: {train_missing}, Test: {test_missing}")

if __name__ == "__main__":
    main() 