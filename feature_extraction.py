import numpy as np
import pandas as pd


def extract_statistical_features(X):
    """
    Extract statistical features from the input data.
    This can be used for additional feature engineering.
    
    :param X: Input feature matrix (n_samples, n_features)
    :return: Enhanced feature matrix
    """
    # For now, we'll use the features as-is from the dataset
    # The TCP-SYNC dataset already has comprehensive flow-based features
    return X


def extract_ddos_indicators(df):
    """
    Extract DDoS-specific indicator features that capture attack patterns.
    These features help the CNN learn to identify DDoS similar to analysis.py logic.
    
    :param df: DataFrame with network traffic features
    :return: DataFrame with added DDoS indicator features
    """
    print("\nExtracting DDoS indicator features...")
    df_enhanced = df.copy()
    
    # High destination frequency indicator (key DDoS pattern)
    if 'dest_frequency' in df.columns:
        # Flag destinations receiving >50 packets (DDoS threshold from analysis.py)
        df_enhanced['is_high_traffic_dest'] = (df['dest_frequency'] > 50).astype(int)
        
        # Logarithmic scaling of destination frequency (helps with extreme values)
        df_enhanced['dest_frequency_log'] = np.log1p(df['dest_frequency'])
        
        print(f"  High traffic destinations: {df_enhanced['is_high_traffic_dest'].sum()}")
    
    # Source concentration indicator
    if 'src_frequency' in df.columns:
        df_enhanced['src_frequency_log'] = np.log1p(df['src_frequency'])
    
    # Destination rank indicators (popular targets)
    if 'dest_rank' in df.columns:
        # Top 10 most targeted destinations
        df_enhanced['is_top_dest'] = (df['dest_rank'] <= 10).astype(int)
        
        # Top 100 most targeted destinations
        df_enhanced['is_top100_dest'] = (df['dest_rank'] <= 100).astype(int)
    
    # Temporal anomaly indicators
    if 'time_delta' in df.columns:
        # Very fast consecutive packets (common in DDoS)
        df_enhanced['is_rapid_fire'] = (df['time_delta'] < 0.001).astype(int)
        
        # Time delta statistics
        df_enhanced['time_delta_log'] = np.log1p(df['time_delta'].abs())
    
    # Connection pattern indicators
    if 'src_dest_frequency' in df.columns:
        # Repeated connections between same source-dest pair
        df_enhanced['is_repeated_connection'] = (df['src_dest_frequency'] > 10).astype(int)
        df_enhanced['src_dest_frequency_log'] = np.log1p(df['src_dest_frequency'])
    
    # Packet rate indicators (if we have rolling window features)
    if 'dest_freq_rolling_mean' in df.columns:
        # Increasing destination frequency over time (DDoS buildup)
        df_enhanced['dest_freq_increasing'] = (
            df['dest_frequency'] > df['dest_freq_rolling_mean']
        ).astype(int)
    
    print(f"  Total DDoS indicator features added: {len(df_enhanced.columns) - len(df.columns)}")
    
    return df_enhanced


def create_temporal_features(df):
    """
    Create temporal features from time-based data.
    
    :param df: DataFrame with time information
    :return: DataFrame with added temporal features
    """
    if 'Time' in df.columns:
        df['Time_Delta'] = df['Time'].diff().fillna(0)
        df['Time_Cumsum'] = df['Time'].cumsum()
    return df


def create_ratio_features(df):
    """
    Create ratio-based features that might be indicative of DDOS attacks.
    
    :param df: DataFrame with network flow features
    :return: DataFrame with added ratio features
    """
    df_enhanced = df.copy()
    
    # Forward/Backward packet ratio
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df_enhanced['Fwd_Bwd_Pkt_Ratio'] = df['Tot Fwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-8)
    
    # Bytes per packet ratio
    if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
        df_enhanced['Avg_Fwd_Pkt_Size'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1e-8)
    
    if 'TotLen Bwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df_enhanced['Avg_Bwd_Pkt_Size'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-8)
    
    # Flow duration per packet
    if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        total_pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
        df_enhanced['Duration_Per_Pkt'] = df['Flow Duration'] / (total_pkts + 1e-8)
    
    return df_enhanced


def select_important_features(X, feature_names, top_k=None):
    """
    Select the most important features for the model.
    This can be based on domain knowledge or feature importance from a trained model.
    
    :param X: Feature matrix
    :param feature_names: List of feature names
    :param top_k: Number of top features to select (None = use all)
    :return: Selected features and their names
    """
    if top_k is None or top_k >= len(feature_names):
        return X, feature_names
    
    # For now, return all features
    # In practice, you could use feature selection methods here
    return X, feature_names


def aggregate_flow_features(df, window_size=10):
    """
    Aggregate features over a sliding window to capture temporal patterns.
    Useful for detecting DDOS patterns that emerge over time.
    
    :param df: DataFrame with network flow data
    :param window_size: Size of the sliding window
    :return: DataFrame with aggregated features
    """
    df_agg = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Rolling mean
        df_agg[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        # Rolling std
        df_agg[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    return df_agg


def detect_anomaly_features(X):
    """
    Create features that might indicate anomalous behavior typical of DDOS attacks.
    
    :param X: Feature matrix
    :return: Anomaly indicator features
    """
    # This function can be expanded with domain-specific anomaly indicators
    # For example: unusually high packet rates, abnormal flag combinations, etc.
    return X


def reshape_for_cnn(X, sequence_length=10):
    """
    Reshape feature matrix for CNN input.
    CNN expects 3D input: (n_samples, sequence_length, n_features)
    
    For network flow data, we can:
    1. Treat each flow as a 1D sequence with spatial structure
    2. Reshape features into a 2D grid if applicable
    
    :param X: Feature matrix (n_samples, n_features)
    :param sequence_length: Length of sequence for temporal modeling
    :return: Reshaped data for CNN
    """
    n_samples, n_features = X.shape
    
    # Option 1: Treat as 1D CNN with single timestep
    # Shape: (n_samples, n_features, 1)
    X_reshaped = X.reshape(n_samples, n_features, 1)
    
    return X_reshaped


def reshape_for_cnn_2d(X, height=None, width=None):
    """
    Reshape features into 2D grid for 2D CNN.
    
    :param X: Feature matrix (n_samples, n_features)
    :param height: Height of the 2D grid (auto-calculated if None)
    :param width: Width of the 2D grid (auto-calculated if None)
    :return: Reshaped data (n_samples, height, width, 1)
    """
    n_samples, n_features = X.shape
    
    if height is None and width is None:
        # Try to create a square-ish grid
        height = int(np.ceil(np.sqrt(n_features)))
        width = int(np.ceil(n_features / height))
    
    # Pad features if necessary
    target_features = height * width
    if n_features < target_features:
        padding = np.zeros((n_samples, target_features - n_features))
        X_padded = np.hstack([X, padding])
    else:
        X_padded = X[:, :target_features]
    
    # Reshape to 2D grid
    X_reshaped = X_padded.reshape(n_samples, height, width, 1)
    
    return X_reshaped


if __name__ == '__main__':
    print("Feature extraction module loaded.")
    print("\nAvailable functions:")
    print("  - extract_statistical_features(X)")
    print("  - extract_ddos_indicators(df)")
    print("  - create_temporal_features(df)")
    print("  - create_ratio_features(df)")
    print("  - select_important_features(X, feature_names, top_k)")
    print("  - aggregate_flow_features(df, window_size)")
    print("  - detect_anomaly_features(X)")
    print("  - reshape_for_cnn(X, sequence_length)")
    print("  - reshape_for_cnn_2d(X, height, width)")
    
    print("\n" + "="*60)
    print("DDoS Detection Feature Hierarchy:")
    print("="*60)
    print("1. Destination Frequency Features (PRIMARY)")
    print("   - dest_frequency: How many times a destination is targeted")
    print("   - is_high_traffic_dest: Flag for destinations with >50 packets")
    print("   - dest_rank: Popularity rank of the destination")
    print("")
    print("2. Temporal Pattern Features")
    print("   - time_delta: Time between packets")
    print("   - is_rapid_fire: Very fast consecutive packets")
    print("   - rolling statistics: Moving averages and max values")
    print("")
    print("3. Connection Pattern Features")
    print("   - src_dest_frequency: Repeated connections between pairs")
    print("   - src_frequency: Source activity level")
    print("")
    print("These features enable the CNN to learn DDoS patterns similar")
    print("to the threshold-based detection in analysis.py")
    print("="*60)
