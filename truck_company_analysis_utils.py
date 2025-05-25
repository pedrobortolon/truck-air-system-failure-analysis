import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy import stats

def map_high_correlations(correlations, threshold=0.75):
    correlated_variables_map = dict()

    # iterate through the items below quadrant 
    for i in range(1, len(correlations.iloc[1:,1:].index)):
        correlated_variables = set()
        for j in range(i):
            if correlations.iloc[i,j] >= threshold:
                correlated_variables.add(correlations.columns[j])
        if len(correlated_variables) > 0:
            correlated_variables_map[correlations.index[i]] = correlated_variables

    return correlated_variables_map


def cohens_d(group0, group1):
    n0, n1 = len(group0), len(group1)
    pooled_std = np.sqrt(((n0-1)*group0.std()**2 + (n1-1)*group1.std()**2)/(n0+n1-2))
    return (group1.mean() - group0.mean())/pooled_std

def calculate_point_biserial(df, target_column, numeric_columns=None):
    df = df.copy()
    
    # Convert target to numeric if it's boolean or categorical
    if not np.issubdtype(df[target_column].dtype, np.number):
        df[target_column] = pd.factorize(df[target_column])[0]
    
    # If numeric_columns not specified, use all numeric columns except target
    if numeric_columns is None:
        numeric_columns = [col for col in df.select_dtypes(include=['number']).columns 
                          if col != target_column]
    
    # Verify target is binary
    unique_values = df[target_column].nunique(dropna=True)
    if unique_values != 2:
        raise ValueError(f"Target column '{target_column}' must have exactly 2 unique values. Found {unique_values}.")
    
    results = []
    
    for col in numeric_columns:
        # Create temporary series with missing values dropped pairwise
        temp_df = df[[target_column, col]].dropna()
        
        if len(temp_df) < 2:
            results.append({
                'column': col,
                'correlation': np.nan,
                'p_value': np.nan,
                'message': 'Insufficient non-null data'
            })
            continue
            
        # Split data into two groups based on target values
        group0 = temp_df[temp_df[target_column] == temp_df[target_column].unique()[0]][col]
        group1 = temp_df[temp_df[target_column] == temp_df[target_column].unique()[1]][col]
        
        try:
            # Calculate point biserial correlation
            r_pb, p_value = stats.pointbiserialr(temp_df[target_column], temp_df[col])
            d = cohens_d(group0, group1)
            
            results.append({
                'column': col,
                'correlation': r_pb,
                'p_value': p_value,
                'cohens_d': d,
                'group0_mean': group0.mean(),
                'group1_mean': group1.mean(),
                'group0_size': len(group0),
                'group1_size': len(group1),
                'message': 'Success'
            })
        except Exception as e:
            results.append({
                'column': col,
                'correlation': np.nan,
                'p_value': np.nan,
                'message': f'Error: {str(e)}'
            })
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by absolute correlation if we have valid values
    if 'correlation' in result_df.columns:
        result_df['abs_correlation'] = result_df['correlation'].abs()
        result_df = result_df.sort_values('abs_correlation', ascending=False).drop('abs_correlation', axis=1)
    
    return result_df