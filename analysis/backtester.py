# backtester.py
# This script provides functionalities for performing walk-forward validation
# on a stock price movement classification model using different algorithms,
# and calculating relevant financial metrics, including transaction costs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import models for comparison
import xgboost as xgb
import lightgbm as lgb

# Ensure your utility modules are correctly referenced.
from utils.data_loader import load_historical_data, ensure_data_available
from utils.feature_engineering import create_features
# from utils.logger import log_to_discord # Uncomment if you want to log from here

def train_model_for_backtest(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = 'random_forest', 
    params: dict = None
) -> object | None:
    """
    Trains a specified classification model for a single fold of the backtest.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        model_type (str, optional): Type of model to train. 
                                    Options: 'random_forest', 'xgboost', 'lightgbm'.
                                    Defaults to 'random_forest'.
        params (dict, optional): Hyperparameters for the chosen model.
                                 If None, default parameters are used. Defaults to None.

    Returns:
        object | None: A trained scikit-learn compatible model,
                       or None if training data is empty or model type is invalid.
    """
    if X_train.empty or y_train.empty:
        print("[ERROR] Training data (X_train or y_train) is empty in train_model_for_backtest.")
        return None
    
    model = None
    if model_type == 'random_forest':
        model_params = params if params else {'n_estimators': 100, 'random_state': 42}
        model = RandomForestClassifier(**model_params)
    elif model_type == 'xgboost':
        model_params = params if params else {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False, 'random_state': 42}
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'lightgbm':
        model_params = params if params else {'objective': 'binary', 'metric': 'binary_logloss', 'random_state': 42, 'verbose': -1}
        model = lgb.LGBMClassifier(**model_params)
    else:
        print(f"[ERROR] Invalid model_type: {model_type}. Supported types: 'random_forest', 'xgboost', 'lightgbm'.")
        return None
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[ERROR] Failed to fit {model_type} model: {e}")
        return None
    return model

def calculate_and_display_financial_metrics(
    results_df: pd.DataFrame,
    historical_prices_for_period: pd.Series,
    stock_name: str,
    model_name_for_plot: str, # Added to distinguish plots
    transaction_cost_pct: float = 0.001,
    risk_free_rate_annual: float = 0.05,
    trading_days_per_year: int = 252
) -> dict | None:
    """
    Calculates and displays financial metrics from backtest results, including transaction costs.
    (Docstring details from previous version are applicable here too)
    Args:
        results_df (pd.DataFrame): DataFrame with 'Actual' and 'Predicted' movements.
        historical_prices_for_period (pd.Series): 'Close' prices for the period.
        stock_name (str): Name of the stock.
        model_name_for_plot (str): Name of the model for plot titles/filenames.
        transaction_cost_pct (float, optional): Cost per trade. Defaults to 0.001.
        risk_free_rate_annual (float, optional): Annual risk-free rate. Defaults to 0.05.
        trading_days_per_year (int, optional): Trading days in a year. Defaults to 252.

    Returns:
        dict | None: A dictionary containing calculated financial metrics, or None.
    """
    if results_df.empty:
        print(f"[ERROR] Cannot calculate financial metrics for {stock_name} with {model_name_for_plot}: results_df is empty.")
        return None

    print(f"\n--- Financial Metrics for {stock_name} using {model_name_for_plot} (Tx Cost: {transaction_cost_pct*100:.3f}%) ---")

    hit_rate = accuracy_score(results_df['Actual'], results_df['Predicted'])
    print(f"Hit Rate (Directional Accuracy): {hit_rate:.4f}")

    aligned_close_prices = historical_prices_for_period.reindex(results_df.index)
    actual_daily_stock_returns = aligned_close_prices.pct_change().shift(-1)
    results_df['Actual_Stock_Return'] = actual_daily_stock_returns
    results_df.dropna(subset=['Actual_Stock_Return'], inplace=True)

    if results_df.empty:
        print(f"[ERROR] results_df for {stock_name} ({model_name_for_plot}) empty after aligning stock returns for metrics.")
        return None

    results_df['Signal'] = results_df['Predicted']
    results_df['Position'] = 0
    results_df['Trades'] = 0
    current_position = 0
    for i in range(len(results_df)):
        signal = results_df['Signal'].iloc[i]
        if current_position == 0 and signal == 1:
            current_position = 1
            results_df.loc[results_df.index[i], 'Position'] = 1
            results_df.loc[results_df.index[i], 'Trades'] = 1
        elif current_position == 1 and signal == 0:
            current_position = 0
            results_df.loc[results_df.index[i], 'Position'] = 0 
            results_df.loc[results_df.index[i], 'Trades'] = 1
        elif current_position == 1 and signal == 1:
             results_df.loc[results_df.index[i], 'Position'] = 1
    
    results_df['Effective_Position_For_Return'] = results_df['Position'].shift(1).fillna(0)
    results_df['Strategy_Return_Pre_Cost'] = results_df['Actual_Stock_Return'] * results_df['Effective_Position_For_Return']
    results_df['Transaction_Cost_Impact'] = results_df['Trades'] * transaction_cost_pct
    results_df['Strategy_Return_Post_Cost'] = results_df['Strategy_Return_Pre_Cost'] - results_df['Transaction_Cost_Impact']
    
    results_df['Cumulative_Strategy_Return_Post_Cost'] = (1 + results_df['Strategy_Return_Post_Cost']).cumprod()
    results_df['Cumulative_Buy_And_Hold_Return'] = (1 + results_df['Actual_Stock_Return']).cumprod()
    
    plt.figure(figsize=(12, 7))
    results_df['Cumulative_Strategy_Return_Post_Cost'].plot(label=f'Strategy ({model_name_for_plot}, TxCost: {transaction_cost_pct*100:.2f}%)')
    results_df['Cumulative_Buy_And_Hold_Return'].plot(label='Buy & Hold', linestyle='--')
    plt.title(f'Cumulative Returns (Post-Cost): {model_name_for_plot} vs. B&H for {stock_name}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns (1 = Start)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_filename_cum_returns_post_cost = f"webapp/static/{stock_name}_{model_name_for_plot}_cum_returns_post_cost.png"
    try:
        plt.savefig(plot_filename_cum_returns_post_cost)
        print(f"Saved post-cost cumulative returns plot to {plot_filename_cum_returns_post_cost}")
    except Exception as e:
        print(f"[ERROR] Failed to save post-cost cumulative returns plot: {e}")
    plt.close()

    total_strategy_return_post_cost_pct = (results_df['Cumulative_Strategy_Return_Post_Cost'].iloc[-1] - 1) if not results_df.empty else np.nan
    total_buy_hold_return_pct = (results_df['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) if not results_df.empty else np.nan
    num_trades = results_df['Trades'].sum()
    print(f"Total Strategy Return (Post-Cost): {total_strategy_return_post_cost_pct:.4%}")
    print(f"Total Buy & Hold Return (over backtest period): {total_buy_hold_return_pct:.4%}")
    print(f"Number of Trades (entries/exits): {num_trades}")

    sharpe_ratio_annualized_post_cost = np.nan
    if len(results_df['Strategy_Return_Post_Cost']) > 1 and results_df['Strategy_Return_Post_Cost'].std() != 0:
        daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
        excess_daily_returns_post_cost = results_df['Strategy_Return_Post_Cost'] - daily_risk_free_rate
        sharpe_ratio_annualized_post_cost = (excess_daily_returns_post_cost.mean() / excess_daily_returns_post_cost.std()) * np.sqrt(trading_days_per_year)
        print(f"Annualized Sharpe Ratio (Strategy, Post-Cost): {sharpe_ratio_annualized_post_cost:.4f}")
    else:
        print("Sharpe Ratio (Post-Cost): Not calculable.")

    max_drawdown_post_cost_pct = np.nan
    if not results_df['Cumulative_Strategy_Return_Post_Cost'].empty and not results_df['Cumulative_Strategy_Return_Post_Cost'].isnull().all() :
        cumulative_returns_post_cost = results_df['Cumulative_Strategy_Return_Post_Cost']
        peak_post_cost = cumulative_returns_post_cost.expanding(min_periods=1).max()
        drawdown_post_cost = (cumulative_returns_post_cost - peak_post_cost) / peak_post_cost
        max_drawdown_post_cost_pct = drawdown_post_cost.min()
        print(f"Maximum Drawdown (Strategy, Post-Cost): {max_drawdown_post_cost_pct:.4%}")
    else:
        print("Maximum Drawdown (Post-Cost): Not calculable.")
        
    return {
        "hit_rate": hit_rate,
        "total_strategy_return_post_cost_pct": total_strategy_return_post_cost_pct,
        "total_buy_and_hold_return_pct": total_buy_hold_return_pct,
        "annualized_sharpe_ratio_post_cost": sharpe_ratio_annualized_post_cost,
        "max_drawdown_post_cost_pct": max_drawdown_post_cost_pct,
        "number_of_trades": num_trades
    }

def run_walk_forward_validation(
    stock_name: str, 
    model_type_to_run: str, # Added model_type parameter
    n_splits: int = 5, 
    transaction_cost: float = 0.001
) -> pd.DataFrame | None:
    """
    Performs walk-forward validation using a specified model type.
    (Docstring details from previous version are applicable here too)

    Args:
        stock_name (str): The stock ticker or name to backtest.
        model_type_to_run (str): Type of model ('random_forest', 'xgboost', 'lightgbm').
        n_splits (int, optional): Number of walk-forward splits. Defaults to 5.
        transaction_cost (float, optional): Percentage cost per trade. Defaults to 0.001.

    Returns:
        pd.DataFrame | None: DataFrame with backtest results, or None if failed.
    """
    print(f"\n--- Starting Walk-Forward Validation for {stock_name} using {model_type_to_run} ---")
    
    try:
        ensure_data_available(stock_name.upper())
        df_historical_full = load_historical_data(stock_name.upper())
        
        if df_historical_full.empty or 'Close' not in df_historical_full.columns:
            print(f"[ERROR] No historical data or 'Close' column for {stock_name}.")
            return None
        
        df_historical_full['Close'] = pd.to_numeric(df_historical_full['Close'], errors='coerce')
        df_historical_full.dropna(subset=['Close'], inplace=True)
        if df_historical_full.empty:
            print(f"[ERROR] Historical data for {stock_name} empty after processing 'Close'.")
            return None

        X_full, y_full = create_features(df_historical_full)
        
        if X_full.empty or y_full.empty:
            print(f"[ERROR] Feature creation resulted in empty X or y for {stock_name}.")
            return None

        all_predictions_list = []
        all_actuals_list = []
        all_indices_list = [] 

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_count = 0
        for train_index, test_index in tscv.split(X_full):
            fold_count += 1
            print(f"\nProcessing Fold {fold_count}/{n_splits} for {stock_name} with {model_type_to_run}...")
            
            X_train_fold, X_test_fold = X_full.iloc[train_index], X_full.iloc[test_index]
            y_train_fold, y_test_fold = y_full.iloc[train_index], y_full.iloc[test_index]

            if X_train_fold.empty or y_train_fold.empty or X_test_fold.empty or y_test_fold.empty:
                print(f"[WARNING] Skipping fold {fold_count} due to empty train/test data.")
                continue
            
            model_fold = train_model_for_backtest(X_train_fold, y_train_fold, model_type=model_type_to_run) # Pass model_type
            if model_fold is None:
                print(f"[WARNING] Failed to train {model_type_to_run} for fold {fold_count}. Skipping.")
                continue
            
            predictions_fold = model_fold.predict(X_test_fold)
            all_predictions_list.extend(predictions_fold)
            all_actuals_list.extend(y_test_fold.values)
            all_indices_list.extend(y_test_fold.index) 
            print(f"Fold {fold_count} Accuracy ({model_type_to_run}): {accuracy_score(y_test_fold, predictions_fold):.4f}")

        if not all_actuals_list:
            print(f"[WARNING] No results from any fold for {stock_name} with {model_type_to_run}.")
            return None

        results_df = pd.DataFrame({
            'Timestamp': all_indices_list,
            'Actual': all_actuals_list,
            'Predicted': all_predictions_list
        }).set_index('Timestamp')

        print(f"\n--- Overall Walk-Forward Classification Report for {stock_name} ({model_type_to_run}) ---")
        print(classification_report(results_df['Actual'], results_df['Predicted'], zero_division=0))
        
        plt.figure(figsize=(12, 7))
        plt.plot(results_df.index, results_df['Actual'], label='Actual Movements', marker='o', linestyle='', markersize=5, alpha=0.6)
        plt.plot(results_df.index, results_df['Predicted'], label=f'Predicted ({model_type_to_run})', marker='x', linestyle='', markersize=5, alpha=0.6)
        plt.title(f'Walk-Forward: Actual vs. Predicted Movements for {stock_name} ({model_type_to_run})', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Movement (0=Down, 1=Up)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_filename_movements = f"webapp/static/{stock_name}_{model_type_to_run}_walk