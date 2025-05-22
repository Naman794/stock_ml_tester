# utils/feature_engineering.py
import pandas as pd
import numpy as np

def create_features(df_historical: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates features for predicting stock price movement.

    Args:
        df_historical (pd.DataFrame): DataFrame with historical stock data,
                                      must include 'Close', 'High', 'Low' columns,
                                      and be indexed by Date.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X (features) and y (target).
                                         Returns empty DataFrames/Series if not possible.
    """
    if not isinstance(df_historical, pd.DataFrame):
        print("[ERROR] Input to create_features is not a Pandas DataFrame.")
        return pd.DataFrame(), pd.Series(dtype='float64')
        
    if df_historical.empty:
        print("[ERROR] Input DataFrame to create_features is empty.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    required_columns = ['Close', 'High', 'Low', 'Open', 'Volume'] # Added Open and Volume for potential future use
    if not all(col in df_historical.columns for col in required_columns):
        print(f"[ERROR] Historical data missing one of required columns: {required_columns}")
        return pd.DataFrame(), pd.Series(dtype='float64')

    df = df_historical.copy()

    # Ensure 'Close' is numeric and handle potential issues
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True) # Drop rows where 'Close' became NaN

    if df.empty:
        print("[ERROR] DataFrame became empty after coercing 'Close' to numeric.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    # Basic Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Volatility (Standard Deviation)
    df['Volatility10'] = df['Close'].rolling(window=10).std()
    df['Volatility50'] = df['Close'].rolling(window=50).std() # Added 50-day volatility

    # Lagged Return (1-day)
    df['Return_1D'] = df['Close'].pct_change(1)

    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] # MACD Histogram

    # RSI (Relative Strength Index) - 14 period
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # Fill initial NaNs in RSI with 50 (neutral) or use ffill after some values are computed
    # df['RSI14'].fillna(50, inplace=True) 
    # A common practice is to ffill after enough data points have allowed calculation
    # However, dropping NaNs at the end is safer for model training

    # Target variable: 1 if next day's close is higher, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaN values created by rolling windows or shifts
    df.dropna(inplace=True)

    # Define feature set
    feature_columns = [
        'MA10', 'MA50',
        'Volatility10', 'Volatility50',
        'Return_1D',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI14'
        # Add other relevant columns from df if needed, e.g., 'Volume' directly or volume-based indicators
    ]
    
    # Ensure all selected feature columns exist (they should if defined above)
    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    
    if not existing_feature_columns:
        print("[ERROR] No feature columns available after processing.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    X = df[existing_feature_columns]
    y = df['Target']
    
    if X.empty or y.empty:
        print("[WARNING] Feature matrix X or target y is empty after final processing.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    print(f"Successfully created features. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

if __name__ == '__main__':
    # Example Usage (assuming you have a CSV file for testing)
    # Create a dummy CSV for testing
    dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
    data = {
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 100,
        'High': np.random.rand(200) * 10 + 200, # Open + some value
        'Low': np.random.rand(200) * 10 + 90,   # Open - some value
        'Close': np.random.rand(200) * 100 + 100, # Similar to Open
        'Volume': np.random.randint(100000, 1000000, size=200)
    }
    for i in range(1, 200): # Ensure High is highest, Low is lowest
        data['High'][i] = data['Open'][i] + np.random.rand() * 10
        data['Low'][i] = data['Open'][i] - np.random.rand() * 10
        if data['Low'][i] < 0: data['Low'][i] = 0
        data['Close'][i] = np.random.uniform(data['Low'][i], data['High'][i])


    sample_df = pd.DataFrame(data)
    sample_df.set_index('Date', inplace=True)
    
    print("Sample DataFrame head:")
    print(sample_df.head())

    X_features, y_target = create_features(sample_df)

    if not X_features.empty:
        print("\nFeatures (X) head:")
        print(X_features.head())
        print("\nTarget (y) head:")
        print(y_target.head())
        print(f"\nShapes: X: {X_features.shape}, y: {y_target.shape}")
    else:
        print("\nCould not generate features from the sample data.")