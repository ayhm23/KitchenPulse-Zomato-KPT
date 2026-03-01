"""
Feature Store Builder

Assembles final feature bundle for analysis and reporting.
"""

import pandas as pd
import numpy as np
from signal_denoiser import correct_button_bias
from kitchen_load_index import compute_kitchen_load_index


class FeatureStoreBuilder:
    """
    Builds a comprehensive feature store from raw order data.
    """
    
    def __init__(self):
        """Initialize feature store builder."""
        pass
    
    def build_features(self, df):
        """
        Build complete feature set.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw order data
            
        Returns:
        --------
        pd.DataFrame
            Feature-enriched dataframe
        """
        # Apply signal denoising
        df = correct_button_bias(df)
        
        # Compute kitchen load index
        df = compute_kitchen_load_index(df)
        
        # Additional features can be added here if dataset supplies columns
        # such as button presses or item counts. currently not applicable.
        
        # Time-based features (use order_time since timestamp may not exist)
        df['order_time'] = pd.to_datetime(df['order_time'])
        df['hour'] = df['order_time'].dt.hour
        df['day_of_week'] = df['order_time'].dt.dayofweek
        
        return df


def build_feature_store(df):
    """
    Build feature store from raw data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw order dataframe
        
    Returns:
    --------
    pd.DataFrame
        Feature-enriched dataframe ready for analysis
    """
    builder = FeatureStoreBuilder()
    features_df = builder.build_features(df)
    
    return features_df
