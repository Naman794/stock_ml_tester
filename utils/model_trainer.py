# utils/model_trainer.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score # Added accuracy_score
import matplotlib.pyplot as plt
from utils.logger import log_to_discord # Assuming you might want to log results

def train_and_predict(X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = False) -> tuple[int | None, plt.Figure | None]:
    """
    Trains a RandomForestClassifier, optionally tunes hyperparameters,
    makes predictions, and evaluates the model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning.

    Returns:
        tuple[int | None, plt.Figure | None]: 
            - The latest prediction (1 for Up, 0 for Down), or None if error.
            - A matplotlib Figure object for the prediction plot, or None if error.
    """
    if X.empty or y.empty:
        print("[ERROR] X or y is empty in train_and_predict.")
        return None, None

    # Splitting data chronologically
    # shuffle=False is critical for time series if not using TimeSeriesSplit directly for the main split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if X_train.empty or y_train.empty:
        print("[ERROR] Training set is empty after split.")
        return None, None
    if X_test.empty or y_test.empty:
        print("[ERROR] Test set is empty after split.")
        # Or, if you want to train and predict only if there's a test set for evaluation:
        # return None, None 
        # For now, we'll allow it to proceed and predict on X_test even if y_test is empty for prediction output,
        # but evaluation will fail or be meaningless. The check on X_test.empty is more crucial for model.predict()

    model = RandomForestClassifier(random_state=42) # Initialize with a random_state

    if tune_hyperparameters:
        print("\n⚙️ Tuning hyperparameters for RandomForestClassifier...")
        # Define the parameter grid to search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            # 'class_weight': ['balanced', 'balanced_subsample', None] # Consider if classes are imbalanced
        }

        # Use TimeSeriesSplit for cross-validation
        # n_splits can be adjusted; a common value is 3 to 5.
        # Using a smaller number of splits because stock data might not be extremely long after feature engineering.
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=tscv, n_jobs=-1, verbose=1, scoring='accuracy') # Use 'f1' or 'roc_auc' if more appropriate
        
        try:
            grid_search.fit(X_train, y_train) # Tune on the training set
            print(f"Best parameters found: {grid_search.best_params_}")
            log_to_discord(f"Hyperparameter Tuning Best Params: {grid_search.best_params_}")
            model = grid_search.best_estimator_ # Use the best model found
        except ValueError as ve:
            print(f"[ERROR] ValueError during GridSearchCV: {ve}. This can happen if a split is too small.")
            log_to_discord(f"[ERROR] GridSearchCV failed for a stock: {ve}")
            # Fallback to default model if tuning fails
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("Using default RandomForestClassifier due to tuning error.")
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during GridSearchCV: {e}")
            log_to_discord(f"[ERROR] GridSearchCV unexpected error: {e}")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("Using default RandomForestClassifier due to tuning error.")
            model.fit(X_train, y_train)

    else: # No hyperparameter tuning
        model = RandomForestClassifier(n_estimators=100, random_state=42) # Default model
        model.fit(X_train, y_train)

    latest_prediction_on_X = None
    if not X.empty: # Predict on the last sample of the original X to get the "next day" forecast
        # The model is trained. To predict the next movement based on the *absolute latest* data in X:
        # We need features for the last available day.
        # X already contains all features up to the last point where 'Target' could be calculated.
        # So, X.iloc[-1] would be the features for the last day for which we know the outcome.
        # To predict the *next* outcome (for which we don't have a Target yet),
        # we'd ideally need features from the day corresponding to the last y value.
        # Let's assume X_test[-1:] gives us the features for the latest test data point.
        if not X_test.empty:
            try:
                # This predicts on the last instance of the TEST set.
                # To get a true "next day beyond all known data", you'd need to engineer features
                # for the very last day of df_historical BEFORE it was trimmed by dropna due to Target.
                # However, for now, let's use the last test point prediction as "latest_prediction".
                last_test_point_features = X_test.iloc[-1:]
                prediction_for_last_test_point = model.predict(last_test_point_features)[0]
                latest_prediction_on_X = int(prediction_for_last_test_point)
            except Exception as e:
                print(f"[ERROR] Could not make a prediction on the last X_test sample: {e}")
                latest_prediction_on_X = None # Fallback

    predictions_on_test = None
    if not X_test.empty and not y_test.empty:
        predictions_on_test = model.predict(X_test)
        print("\n📊 Classification Report (on Test Set):\n")
        print(classification_report(y_test, predictions_on_test, zero_division=0))
        test_accuracy = accuracy_score(y_test, predictions_on_test)
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        log_to_discord(f"Classification Report (Test Set): Accuracy {test_accuracy:.4f}") # Log accuracy
    else:
        print("\n📊 Classification Report: Test set is empty, cannot generate report.")


    # Plotting actual vs predicted for the TEST SET
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted figure size
    if predictions_on_test is not None and not y_test.empty:
        ax.plot(y_test.values, label="Actual Test Movements", marker='o', linestyle='-', alpha=0.7)
        ax.plot(predictions_on_test, label="Predicted Test Movements", marker='x', linestyle='--', alpha=0.7)
        ax.set_title("Actual vs. Predicted Stock Movement (Test Set)")
    else:
        ax.set_title("Stock Movement (Test Set - Data Unavailable for Plot)")
        
    ax.set_xlabel("Sample Index in Test Set")
    ax.set_ylabel("Movement (0=Down, 1=Up)")
    ax.legend()
    
    # The 'latest_prediction' should ideally be for the day *after* the last day in y_test.
    # For now, we return the prediction for the last instance in X_test.
    # A more robust way would be to re-engineer features for the very last day of the original dataset
    # before 'Target' caused NaNs, and predict on that.
    # For this iteration, latest_prediction_on_X serves as this.
    
    return latest_prediction_on_X, fig