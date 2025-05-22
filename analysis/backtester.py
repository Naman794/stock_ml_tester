# backtester.py
# This script provides functionalities for performing walk-forward validation
# on a stock price movement classification model using different algorithms,
# with optional hyperparameter tuning, and calculating financial metrics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb
import lightgbm as lgb

from utils.data_loader import load_historical_data, ensure_data_available
from utils.feature_engineering import create_features
# from utils.logger import log_to_discord # Uncomment for logging

def train_model_for_backtest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    tune_hyperparameters: bool = False, # Flag to enable/disable tuning
    params: dict = None # Pre-defined params if not tuning
) -> object | None:
    """
    Trains a specified classification model for a single fold of the backtest,
    with optional hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        model_type (str, optional): Type of model. 
                                    Options: 'random_forest', 'xgboost', 'lightgbm'.
                                    Defaults to 'random_forest'.
        tune_hyperparameters (bool, optional): Whether to perform hyperparameter tuning. 
                                             Defaults to False.
        params (dict, optional): Pre-defined hyperparameters if not tuning or as base for tuning.
                                 If None and not tuning, default model parameters are used.
                                 If tuning, these can be starting points or override defaults
                                 if not in the grid. Defaults to None.

    Returns:
        object | None: A trained scikit-learn compatible model, or None if error.
    """
    if X_train.empty or y_train.empty:
        print("[ERROR] Training data (X_train or y_train) is empty in train_model_for_backtest.")
        return None

    model_instance = None
    param_grid = None
    default_params = {}

    # Define models, default parameters, and parameter grids for tuning
    if model_type == 'random_forest':
        default_params = {'n_estimators': 100, 'random_state': 42}
        model_instance = RandomForestClassifier(**(params if params and not tune_hyperparameters else default_params))
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100], # Reduced grid for example speed
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3]
            }
    elif model_type == 'xgboost':
        default_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 
                          'use_label_encoder': False, 'random_state': 42, 'n_estimators': 100}
        model_instance = xgb.XGBClassifier(**(params if params and not tune_hyperparameters else default_params))
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100], # Reduced grid
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.7, 1.0],
                'colsample_bytree': [0.7, 1.0]
            }
    elif model_type == 'lightgbm':
        default_params = {'objective': 'binary', 'metric': 'binary_logloss', 
                          'random_state': 42, 'n_estimators': 100, 'verbose': -1}
        model_instance = lgb.LGBMClassifier(**(params if params and not tune_hyperparameters else default_params))
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100], # Reduced grid
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [20, 31, 40],
                'subsample': [0.7, 1.0], # Requires 'boosting_type':'gbdt' often
                'colsample_bytree': [0.7, 1.0]
            }
    else:
        print(f"[ERROR] Invalid model_type: {model_type}.")
        return None

    if tune_hyperparameters and param_grid:
        print(f"\n⚙️ Tuning hyperparameters for {model_type}...")
        tscv = TimeSeriesSplit(n_splits=3) # Fewer splits for faster tuning in backtest
        
        # Ensure scoring is appropriate for your problem, e.g. 'accuracy', 'f1', 'roc_auc'
        grid_search = GridSearchCV(estimator=model_instance, param_grid=param_grid,
                                   cv=tscv, n_jobs=-1, verbose=0, scoring='accuracy') 
                                   # verbose=0 to reduce output during multiple backtest folds
        try:
            grid_search.fit(X_train, y_train)
            print(f"Best parameters found for {model_type}: {grid_search.best_params_}")
            # log_to_discord(f"Best params for {model_type} (stock context if available): {grid_search.best_params_}")
            model_to_train = grid_search.best_estimator_
        except Exception as e:
            print(f"[ERROR] GridSearchCV for {model_type} failed: {e}. Using default/provided params.")
            # log_to_discord(f"[ERROR] GridSearchCV for {model_type} failed: {e}")
            model_to_train = model_instance # Fallback to the initialized model instance
    else:
        model_to_train = model_instance # Use model with default or pre-set params

    try:
        model_to_train.fit(X_train, y_train)
    except Exception as e:
        print(f"[ERROR] Failed to fit {model_type} model (final attempt): {e}")
        return None
    return model_to_train


def calculate_and_display_financial_metrics(
    results_df: pd.DataFrame,
    historical_prices_for_period: pd.Series,
    stock_name: str,
    model_name_for_plot: str,
    transaction_cost_pct: float = 0.001,
    risk_free_rate_annual: float = 0.05,
    trading_days_per_year: int = 252
) -> dict | None:
    """
    Calculates and displays financial metrics from backtest results.
    (Refer to previous version for full docstring details)
    """
    if results_df.empty:
        print(f"[ERROR] Metrics: results_df empty for {stock_name} ({model_name_for_plot}).")
        return None

    print(f"\n--- Financial Metrics for {stock_name} using {model_name_for_plot} (Tx Cost: {transaction_cost_pct*100:.3f}%) ---")
    hit_rate = accuracy_score(results_df['Actual'], results_df['Predicted'])
    print(f"Hit Rate (Directional Accuracy): {hit_rate:.4f}")

    aligned_close_prices = historical_prices_for_period.reindex(results_df.index)
    actual_daily_stock_returns = aligned_close_prices.pct_change().shift(-1)
    results_df['Actual_Stock_Return'] = actual_daily_stock_returns
    results_df.dropna(subset=['Actual_Stock_Return'], inplace=True)

    if results_df.empty:
        print(f"[ERROR] Metrics: results_df for {stock_name} ({model_name_for_plot}) empty after return alignment.")
        return None

    results_df['Signal'] = results_df['Predicted']
    results_df['Position'] = 0; results_df['Trades'] = 0; current_position = 0
    for i in range(len(results_df)):
        signal = results_df['Signal'].iloc[i]
        if current_position == 0 and signal == 1:
            current_position = 1; results_df.loc[results_df.index[i], 'Position'] = 1; results_df.loc[results_df.index[i], 'Trades'] = 1
        elif current_position == 1 and signal == 0:
            current_position = 0; results_df.loc[results_df.index[i], 'Position'] = 0; results_df.loc[results_df.index[i], 'Trades'] = 1
        elif current_position == 1 and signal == 1:
             results_df.loc[results_df.index[i], 'Position'] = 1
    
    results_df['Effective_Position_For_Return'] = results_df['Position'].shift(1).fillna(0)
    results_df['Strategy_Return_Pre_Cost'] = results_df['Actual_Stock_Return'] * results_df['Effective_Position_For_Return']
    results_df['Transaction_Cost_Impact'] = results_df['Trades'] * transaction_cost_pct
    results_df['Strategy_Return_Post_Cost'] = results_df['Strategy_Return_Pre_Cost'] - results_df['Transaction_Cost_Impact']
    
    results_df['Cumulative_Strategy_Return_Post_Cost'] = (1 + results_df['Strategy_Return_Post_Cost']).cumprod()
    results_df['Cumulative_Buy_And_Hold_Return'] = (1 + results_df['Actual_Stock_Return']).cumprod()
    
    plt.figure(figsize=(12, 7))
    results_df['Cumulative_Strategy_Return_Post_Cost'].plot(label=f'Strategy ({model_name_for_plot}, TxC: {transaction_cost_pct*100:.2f}%)')
    results_df['Cumulative_Buy_And_Hold_Return'].plot(label='Buy & Hold', linestyle='--')
    plt.title(f'Cumulative Returns (Post-Cost): {model_name_for_plot} vs. B&H for {stock_name}', fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Cumulative Returns (1 = Start)'); plt.legend(); plt.grid(True); plt.tight_layout()
    plot_filename = f"webapp/static/{stock_name}_{model_name_for_plot}_cum_returns_post_cost.png"
    try: plt.savefig(plot_filename); print(f"Saved plot: {plot_filename}")
    except Exception as e: print(f"[ERROR] Failed to save plot {plot_filename}: {e}")
    plt.close()

    total_strategy_ret_pc = (results_df['Cumulative_Strategy_Return_Post_Cost'].iloc[-1] - 1) if not results_df.empty else np.nan
    total_buy_hold_ret_pc = (results_df['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) if not results_df.empty else np.nan
    num_trades = results_df['Trades'].sum()
    print(f"Total Strategy Return (Post-Cost): {total_strategy_ret_pc:.4%}")
    print(f"Total Buy & Hold Return: {total_buy_hold_ret_pc:.4%}")
    print(f"Number of Trades: {num_trades}")

    sharpe_post_cost = np.nan
    if len(results_df['Strategy_Return_Post_Cost']) > 1 and results_df['Strategy_Return_Post_Cost'].std() != 0:
        daily_rf = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
        excess_ret = results_df['Strategy_Return_Post_Cost'] - daily_rf
        sharpe_post_cost = (excess_ret.mean() / excess_ret.std()) * np.sqrt(trading_days_per_year)
        print(f"Annualized Sharpe Ratio (Strategy, Post-Cost): {sharpe_post_cost:.4f}")
    else: print("Sharpe Ratio (Post-Cost): Not calculable.")

    max_dd_post_cost = np.nan
    if not results_df['Cumulative_Strategy_Return_Post_Cost'].empty and not results_df['Cumulative_Strategy_Return_Post_Cost'].isnull().all():
        cum_ret = results_df['Cumulative_Strategy_Return_Post_Cost']
        peak = cum_ret.expanding(min_periods=1).max()
        drawdown = (cum_ret - peak) / peak
        max_dd_post_cost = drawdown.min()
        print(f"Maximum Drawdown (Strategy, Post-Cost): {max_dd_post_cost:.4%}")
    else: print("Maximum Drawdown (Post-Cost): Not calculable.")
        
    return { "hit_rate": hit_rate, "total_strategy_return_post_cost_pct": total_strategy_ret_pc, 
             "total_buy_and_hold_return_pct": total_buy_hold_ret_pc, "annualized_sharpe_ratio_post_cost": sharpe_post_cost, 
             "max_drawdown_post_cost_pct": max_dd_post_cost, "number_of_trades": num_trades }

def run_walk_forward_validation(
    stock_name: str,
    model_type_to_run: str,
    n_splits: int = 5,
    transaction_cost: float = 0.001,
    perform_tuning: bool = False # Added flag to control tuning in the main loop
) -> pd.DataFrame | None:
    """
    Performs walk-forward validation using a specified model type, with optional tuning.
    (Refer to previous version for full docstring details)
    """
    print(f"\n--- Starting Walk-Forward for {stock_name} using {model_type_to_run} (Tuning: {perform_tuning}) ---")
    
    try:
        ensure_data_available(stock_name.upper())
        df_historical_full = load_historical_data(stock_name.upper())
        if df_historical_full.empty or 'Close' not in df_historical_full.columns: return None
        df_historical_full['Close'] = pd.to_numeric(df_historical_full['Close'], errors='coerce')
        df_historical_full.dropna(subset=['Close'], inplace=True)
        if df_historical_full.empty: return None
        X_full, y_full = create_features(df_historical_full)
        if X_full.empty or y_full.empty: return None

        all_preds, all_actuals, all_indices = [], [], []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_count = 0
        for train_idx, test_idx in tscv.split(X_full):
            fold_count += 1
            print(f"\nFold {fold_count}/{n_splits} for {stock_name} with {model_type_to_run}...")
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
            if X_train.empty or y_train.empty or X_test.empty or y_test.empty: continue
            
            # Pass perform_tuning flag to the model training function
            model = train_model_for_backtest(X_train, y_train, model_type=model_type_to_run, tune_hyperparameters=perform_tuning)
            if model is None: continue
            
            preds_fold = model.predict(X_test)
            all_preds.extend(preds_fold); all_actuals.extend(y_test.values); all_indices.extend(y_test.index)
            print(f"Fold {fold_count} Accuracy ({model_type_to_run}): {accuracy_score(y_test, preds_fold):.4f}")

        if not all_actuals: return None
        results_df = pd.DataFrame({'Timestamp': all_indices, 'Actual': all_actuals, 'Predicted': all_preds}).set_index('Timestamp')

        print(f"\n--- Overall Report for {stock_name} ({model_type_to_run}) ---")
        print(classification_report(results_df['Actual'], results_df['Predicted'], zero_division=0))
        
        plt.figure(figsize=(12,7)); plt.plot(results_df.index, results_df['Actual'], label='Actual', marker='o', ls='', ms=5, alpha=0.6)
        plt.plot(results_df.index, results_df['Predicted'], label=f'Predicted ({model_type_to_run})', marker='x', ls='', ms=5, alpha=0.6)
        plt.title(f'Walk-Forward: Actual vs. Predicted - {stock_name} ({model_type_to_run})', fontsize=16)
        plt.xlabel('Date'); plt.ylabel('Movement (0=Down, 1=Up)'); plt.legend(); plt.grid(True); plt.tight_layout()
        plot_mv_fname = f"webapp/static/{stock_name}_{model_type_to_run}_walk_forward_movements.png"
        try: plt.savefig(plot_mv_fname); print(f"Saved plot: {plot_mv_fname}")
        except Exception as e: print(f"[ERROR] Failed to save plot {plot_mv_fname}: {e}")
        plt.close()
        
        hist_close_prices = df_historical_full.loc[y_full.index, 'Close']
        prices_for_res_period = hist_close_prices.reindex(results_df.index)
        if prices_for_res_period.isnull().any():
            prices_for_res_period.ffill(inplace=True); prices_for_res_period.bfill(inplace=True)

        calculate_and_display_financial_metrics(results_df.copy(), prices_for_res_period, stock_name, 
                                                model_name_for_plot=model_type_to_run, 
                                                transaction_cost_pct=transaction_cost)
        return results_df
    except Exception as e:
        import traceback
        print(f"[ERROR] Backtest Execution for {stock_name} ({model_type_to_run}) failed: {e}")
        print(traceback.format_exc()); return None

if __name__ == '__main__':
    stocks_to_backtest = ["INFY", "TCS"] # Keep it short for example
    model_types_to_test = ['random_forest', 'xgboost', 'lightgbm']
    simulated_transaction_cost = 0.001 
    
    # Flag to control if hyperparameter tuning is run for each model in each fold.
    # WARNING: Setting this to True will make the backtest VERY SLOW.
    # It's often better to find good hyperparameters once, then use them in the backtest.
    TUNE_MODELS_DURING_BACKTEST = False 

    for stock_symbol in stocks_to_backtest:
        for model_name in model_types_to_test:
            print(f"\n>>>> Processing: {stock_symbol} with Model: {model_name} (TxCost={simulated_transaction_cost*100:.3f}%, Tuning: {TUNE_MODELS_DURING_BACKTEST}) <<<<")
            backtest_df = run_walk_forward_validation(
                stock_name=stock_symbol,
                model_type_to_run=model_name,
                n_splits=3, # Reduced n_splits for faster example runs
                transaction_cost=simulated_transaction_cost,
                perform_tuning=TUNE_MODELS_DURING_BACKTEST 
            )
            if backtest_df is not None: print(f"Completed {stock_symbol} with {model_name}.")
            else: print(f"Failed or no results for {stock_symbol} with {model_name}.")
    print("\n================== Backtesting Run Complete ==================")