# backtester.py (New conceptual script)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier # Or your chosen model
from sklearn.metrics import classification_report, accuracy_score

# Assuming these utils are in the correct path or your PYTHONPATH is set
from utils.data_loader import load_historical_data, ensure_data_available
from utils.feature_engineering import create_features
# We might need a leaner train_predict function for backtesting, 
# or adapt the existing model_trainer's one.
# For now, let's assume a simple train and predict function.

def train_model_for_backtest(X_train, y_train, params=None):
    """
    Trains a model for a single fold of the backtest.
    Returns a trained model.
    """
    if params:
        model = RandomForestClassifier(**params, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def run_walk_forward_validation(stock_name: str, n_splits: int = 5, initial_train_size_ratio: float = 0.5):
    """
    Performs walk-forward validation for a given stock.

    Args:
        stock_name (str): The stock to backtest.
        n_splits (int): Number of walk-forward splits. More splits = more robust but slower.
        initial_train_size_ratio (float): Proportion of data for the initial training set.
                                         The rest will be divided among the test folds.

    Returns:
        pd.DataFrame: DataFrame containing actuals and predictions for all test folds.
    """
    print(f"\n--- Starting Walk-Forward Validation for: {stock_name} ---")
    
    try:
        ensure_data_available(stock_name.upper())
        df_historical = load_historical_data(stock_name.upper())
        if df_historical.empty:
            print(f"No historical data for {stock_name}.")
            return pd.DataFrame()

        X_full, y_full = create_features(df_historical)
        if X_full.empty or y_full.empty:
            print(f"Feature creation failed or resulted in empty data for {stock_name}.")
            return pd.DataFrame()

        # TimeSeriesSplit can also be used to define the walk-forward structure
        # For a more manual approach, or if you want expanding window:
        
        all_predictions = []
        all_actuals = []
        all_indices = [] # To store the original indices of test samples

        # Calculate the size of the initial training set
        # Ensure it's an integer and there's enough data for n_splits
        if len(X_full) < n_splits * 2: # Need at least 2 samples per split (1 train, 1 test)
            print(f"Not enough data for {stock_name} to perform {n_splits} splits. Min required: {n_splits*2}")
            return pd.DataFrame()
            
        # More refined split logic for walk-forward
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_count = 0
        for train_index, test_index in tscv.split(X_full):
            fold_count += 1
            print(f"\nProcessing Fold {fold_count}/{n_splits} for {stock_name}...")
            
            X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
            y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

            if X_train.empty or y_train.empty:
                print(f"Skipping fold {fold_count} due to empty training set.")
                continue
            if X_test.empty or y_test.empty:
                print(f"Skipping fold {fold_count} due to empty test set.")
                continue

            print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

            # Here you could implement hyperparameter tuning for each fold's training set
            # For simplicity, we use fixed hyperparameters or the best ones found earlier.
            # Let's assume we have some pre-tuned best_params, or use defaults
            # best_rf_params = {'n_estimators': 100, 'max_depth': 20, ...} # Example
            
            model = train_model_for_backtest(X_train, y_train) # params=best_rf_params)
            
            predictions_fold = model.predict(X_test)
            
            all_predictions.extend(predictions_fold)
            all_actuals.extend(y_test.values)
            all_indices.extend(y_test.index) # Keep original index for context

            print(f"Fold {fold_count} Accuracy: {accuracy_score(y_test, predictions_fold):.4f}")

        if not all_actuals:
            print(f"No results generated from walk-forward validation for {stock_name}.")
            return pd.DataFrame()

        results_df = pd.DataFrame({
            'Timestamp': all_indices, # Use original datetime index
            'Actual': all_actuals,
            'Predicted': all_predictions
        }).set_index('Timestamp')

        print(f"\n--- Overall Walk-Forward Results for: {stock_name} ---")
        print(classification_report(results_df['Actual'], results_df['Predicted'], zero_division=0))
        overall_accuracy = accuracy_score(results_df['Actual'], results_df['Predicted'])
        print(f"Overall Walk-Forward Accuracy: {overall_accuracy:.4f}")
        
        # Plotting overall results
        plt.figure(figsize=(12, 6))
        plt.plot(results_df.index, results_df['Actual'], label='Actual Movements', marker='.', linestyle='-', alpha=0.7)
        plt.plot(results_df.index, results_df['Predicted'], label='Predicted Movements (Walk-Forward)', marker='x', linestyle='--', alpha=0.7)
        plt.title(f'Walk-Forward Validation: Actual vs. Predicted Movements for {stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Movement (0=Down, 1=Up)')
        plt.legend()
        plt.tight_layout()
        # You might want to save this plot or pass the fig object
        # For now, just show it if running interactively
        # plt.show() 
        
        # Save the plot to static for webapp if needed
        plot_filename = f"webapp/static/{stock_name}_walk_forward_plot.png"
        plt.savefig(plot_filename)
        print(f"Saved walk-forward plot to {plot_filename}")
        plt.close()


        return results_df

    except FileNotFoundError as e:
        print(f"[ERROR] Backtest: Data file not found for {stock_name}. {e}")
        return pd.DataFrame()
    except Exception as e:
        import traceback
        print(f"[ERROR] Backtest: An error occurred for {stock_name}. {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

if __name__ == '__main__':
    # Example of how to run the backtester
    # stocks_to_backtest = ["INFY", "RELIANCE", "TCS"] # Add more stocks from your list
    stocks_to_backtest = ["INFY"] # Start with one for testing
    
    all_backtest_results = {}
    for stock in stocks_to_backtest:
        # Make sure you have historical data CSVs like 'data/historical_INFY.csv'
        # And ensure your ensure_data_available and load_historical_data can find them
        # Also, ensure create_features works with the loaded data.
        results = run_walk_forward_validation(stock, n_splits=5) # Using 5 splits
        if not results.empty:
            all_backtest_results[stock] = results
            print(f"\nBacktest results for {stock}:")
            print(results.head())
            
            # Here, you could calculate financial metrics based on 'results_df'
            # e.g., simulate a simple strategy:
            # Buy if Predicted == 1, Sell/Hold if Predicted == 0
            # Calculate hypothetical returns based on actual price changes on those days.