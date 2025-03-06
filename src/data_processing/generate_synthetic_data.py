#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate synthetic credit risk data for model development and validation.

This script creates realistic synthetic data that mimics retail credit risk profiles,
including customer demographics, loan information, credit history,
and macroeconomic factors. The generated data is suitable for developing
and validating credit risk models.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os
import yaml
import datetime as dt
from pathlib import Path

# Read configuration
def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_customer_ids(n_samples):
    """Generate unique customer IDs."""
    return [f"CUST{i:08d}" for i in range(1, n_samples + 1)]

def generate_dates(n_samples, start_date='2018-01-01', end_date='2022-12-31'):
    """Generate random dates within a range."""
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
    
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    
    random_days = np.random.randint(0, days_between_dates, size=n_samples)
    random_dates = [start_date + dt.timedelta(days=day) for day in random_days]
    
    return pd.Series(random_dates)

def generate_categorical_features(n_samples):
    """Generate categorical features like employment status, housing, education, etc."""
    
    # Employment status
    employment_status_options = ['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student']
    employment_status_probs = [0.65, 0.15, 0.08, 0.07, 0.05]
    
    # Housing status
    housing_status_options = ['Owner', 'Mortgage', 'Rent', 'Living with Parents', 'Other']
    housing_status_probs = [0.3, 0.35, 0.2, 0.1, 0.05]
    
    # Loan purpose
    loan_purpose_options = ['Debt Consolidation', 'Home Improvement', 'Major Purchase', 'Education', 'Medical', 'Vacation', 'Other']
    loan_purpose_probs = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
    
    # Education
    education_options = ['High School', 'College', 'Bachelor', 'Master', 'PhD']
    education_probs = [0.25, 0.3, 0.3, 0.1, 0.05]
    
    # Generate categorical features
    employment_status = np.random.choice(employment_status_options, size=n_samples, p=employment_status_probs)
    housing_status = np.random.choice(housing_status_options, size=n_samples, p=housing_status_probs)
    loan_purpose = np.random.choice(loan_purpose_options, size=n_samples, p=loan_purpose_probs)
    education = np.random.choice(education_options, size=n_samples, p=education_probs)
    
    return {
        'employment_status': employment_status,
        'housing_status': housing_status,
        'loan_purpose': loan_purpose,
        'education': education
    }

def generate_numerical_features(n_samples, correlate_with_target=None):
    """
    Generate numerical features for credit risk data.
    
    Args:
        n_samples: Number of samples to generate
        correlate_with_target: Optional binary array to correlate features with
        
    Returns:
        Dictionary of numerical features
    """
    # Age - normally distributed around 40 with std of 12
    age = np.random.normal(40, 12, n_samples).clip(18, 80).astype(int)
    
    # Income - log-normal distribution to simulate income inequality
    income_mean = np.log(60000)
    income_std = 0.7
    income = np.random.lognormal(income_mean, income_std, n_samples)
    
    # Loan amount - log-normal distribution
    loan_mean = np.log(15000)
    loan_std = 0.8
    loan_amount = np.random.lognormal(loan_mean, loan_std, n_samples)
    
    # Loan term (in months) - discrete values
    loan_term_options = [12, 24, 36, 48, 60, 72]
    loan_term_probs = [0.1, 0.15, 0.35, 0.2, 0.15, 0.05]
    loan_term = np.random.choice(loan_term_options, size=n_samples, p=loan_term_probs)
    
    # Credit score - normally distributed around 700 with std of 100
    credit_score = np.random.normal(700, 100, n_samples).clip(300, 850).astype(int)
    
    # Number of credit lines - Poisson distribution
    num_credit_lines = np.random.poisson(4, n_samples).clip(0, 20)
    
    # Credit utilization rate - Beta distribution to keep between 0 and 1
    utilization_alpha = 2
    utilization_beta = 5
    utilization_rate = np.random.beta(utilization_alpha, utilization_beta, n_samples)
    
    # Debt to income - Beta distribution * scaling factor
    dti_alpha = 2
    dti_beta = 8
    debt_to_income = np.random.beta(dti_alpha, dti_beta, n_samples) * 0.6
    
    # Delinquency history (count of past delinquencies) - Poisson
    delinquency_history = np.random.poisson(0.5, n_samples).clip(0, 10)
    
    # If a target is provided, add correlation between the target and certain features
    if correlate_with_target is not None:
        # Lower credit scores for defaults
        credit_score = credit_score - correlate_with_target * np.random.normal(100, 30, n_samples)
        credit_score = credit_score.clip(300, 850).astype(int)
        
        # Higher DTI for defaults
        debt_to_income = debt_to_income + correlate_with_target * np.random.beta(5, 2, n_samples) * 0.3
        debt_to_income = debt_to_income.clip(0, 0.8)
        
        # More delinquencies for defaults
        delinquency_history = delinquency_history + correlate_with_target * np.random.poisson(2, n_samples)
        delinquency_history = delinquency_history.clip(0, 10)
        
        # Higher utilization for defaults
        utilization_rate = utilization_rate + correlate_with_target * np.random.beta(5, 2, n_samples) * 0.4
        utilization_rate = utilization_rate.clip(0, 1)
    
    return {
        'age': age,
        'income': income,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_score': credit_score,
        'num_credit_lines': num_credit_lines,
        'utilization_rate': utilization_rate,
        'delinquency_history': delinquency_history
    }

def generate_macroeconomic_features(n_samples, dates):
    """
    Generate macroeconomic features that vary by date.
    
    Args:
        n_samples: Number of samples to generate
        dates: Series of dates for each sample
        
    Returns:
        Dictionary of macroeconomic features
    """
    # Create date buckets by quarter
    quarters = pd.PeriodIndex(dates, freq='Q')
    unique_quarters = quarters.unique()
    
    # Generate base economic indicators for each quarter
    quarter_unemployment = {}
    quarter_interest_rate = {}
    quarter_gdp_growth = {}
    
    # Start with reasonable economic values
    current_unemployment = 5.0
    current_interest_rate = 3.0
    current_gdp_growth = 2.5
    
    # Create time series with some persistence
    for quarter in unique_quarters:
        # Add some random changes with persistence
        current_unemployment += np.random.normal(0, 0.3)
        current_unemployment = np.clip(current_unemployment, 3.0, 10.0)
        
        current_interest_rate += np.random.normal(0, 0.2)
        current_interest_rate = np.clip(current_interest_rate, 0.5, 8.0)
        
        current_gdp_growth += np.random.normal(0, 0.5)
        current_gdp_growth = np.clip(current_gdp_growth, -3.0, 5.0)
        
        quarter_unemployment[quarter] = current_unemployment
        quarter_interest_rate[quarter] = current_interest_rate
        quarter_gdp_growth[quarter] = current_gdp_growth
    
    # Map back to individual records
    unemployment_rate = quarters.map(lambda q: quarter_unemployment[q])
    interest_rate = quarters.map(lambda q: quarter_interest_rate[q])
    gdp_growth = quarters.map(lambda q: quarter_gdp_growth[q])
    
    return {
        'unemployment_rate': unemployment_rate.values,
        'interest_rate': interest_rate.values,
        'gdp_growth': gdp_growth.values
    }

def generate_target(n_samples, numerical_features):
    """
    Generate default flag (target variable) with realistic correlations.
    
    The probability of default is influenced by:
    - Credit score (lower = higher default)
    - Debt to income (higher = higher default)
    - Utilization rate (higher = higher default)
    - Delinquency history (higher = higher default)
    """
    # Extract relevant features for calculating default probability
    credit_score = numerical_features['credit_score']
    dti = numerical_features['debt_to_income']
    utilization = numerical_features['utilization_rate']
    delinquencies = numerical_features['delinquency_history']
    
    # Normalize features for probability calculation
    norm_credit = 1 - ((credit_score - 300) / 550)  # Invert so higher credit score = lower risk
    norm_dti = dti  # Already between 0-1
    norm_util = utilization  # Already between 0-1
    norm_delinq = delinquencies / 10  # Scale to 0-1
    
    # Base default probability - around 3-5% overall
    # Combine features with different weights
    default_prob = (
        0.02 +  # Base probability
        norm_credit * 0.15 +  # Credit score effect
        norm_dti * 0.12 +  # DTI effect
        norm_util * 0.08 +  # Utilization effect
        norm_delinq * 0.13  # Delinquency effect
    )
    
    # Ensure probabilities are between 0 and 1
    default_prob = np.clip(default_prob, 0.001, 0.999)
    
    # Generate binary outcome
    default_flag = np.random.binomial(1, default_prob, n_samples)
    
    return default_flag

def generate_credit_data(n_samples, random_seed=42):
    """Generate a complete synthetic credit dataset."""
    np.random.seed(random_seed)
    
    # Generate IDs and application dates
    customer_ids = generate_customer_ids(n_samples)
    application_dates = generate_dates(n_samples)
    
    # Generate categorical features
    categorical_features = generate_categorical_features(n_samples)
    
    # Generate numerical features
    numerical_features = generate_numerical_features(n_samples)
    
    # Generate target variable based on feature correlations
    default_flag = generate_target(n_samples, numerical_features)
    
    # Now go back and make some features correlate with the target
    numerical_features = generate_numerical_features(n_samples, correlate_with_target=default_flag)
    
    # Generate macroeconomic features
    macro_features = generate_macroeconomic_features(n_samples, application_dates)
    
    # Combine all features into a dataframe
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'application_date': application_dates,
        'default_flag': default_flag,
        **categorical_features,
        **numerical_features,
        **macro_features
    })
    
    return df

def split_and_save_data(df, output_dir, test_size=0.2, validation_size=0.1):
    """Split the data into train, test, and validation sets and save them."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_samples = len(df_shuffled)
    n_test = int(n_samples * test_size)
    n_validation = int(n_samples * validation_size)
    n_train = n_samples - n_test - n_validation
    
    # Split the data
    train_df = df_shuffled.iloc[:n_train]
    validation_df = df_shuffled.iloc[n_train:n_train + n_validation]
    test_df = df_shuffled.iloc[n_train + n_validation:]
    
    # Save the dataframes
    train_df.to_csv(os.path.join(output_dir, 'credit_data_train.csv'), index=False)
    validation_df.to_csv(os.path.join(output_dir, 'credit_data_validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'credit_data_test.csv'), index=False)
    
    print(f"Saved data to {output_dir}:")
    print(f"  Training set: {len(train_df)} records")
    print(f"  Validation set: {len(validation_df)} records")
    print(f"  Test set: {len(test_df)} records")

def main():
    """Main function to generate and save synthetic credit data."""
    config = load_config()
    
    # Get parameters from config
    n_samples = config['data']['synthetic_data_size']
    random_seed = config['data']['random_seed']
    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/data'
    
    print(f"Generating {n_samples} synthetic credit records...")
    credit_data = generate_credit_data(n_samples, random_seed)
    
    print(f"Data generation complete. Summary:")
    print(f"  Total records: {len(credit_data)}")
    print(f"  Default rate: {credit_data['default_flag'].mean():.2%}")
    
    # Print some basic statistics
    print("\nNumerical feature statistics:")
    for col in ['credit_score', 'income', 'debt_to_income', 'loan_amount']:
        print(f"  {col}: mean = {credit_data[col].mean():.2f}, std = {credit_data[col].std():.2f}")
    
    print("\nCategorical feature distributions:")
    for col in ['employment_status', 'housing_status', 'loan_purpose']:
        print(f"  {col}:")
        for category, count in credit_data[col].value_counts().items():
            print(f"    {category}: {count/len(credit_data):.2%}")
    
    # Split and save the data
    split_and_save_data(credit_data, output_dir)

if __name__ == "__main__":
    main() 