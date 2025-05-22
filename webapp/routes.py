# webapp/routes.py
from flask import Blueprint, render_template, request
from analysis.ml_models import predict_next_close, train_model, build_feature_matrix # Ensure build_feature_matrix is imported
import pandas as pd
import os
import joblib # For loading the model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split # For recreating the test set

main = Blueprint("main", __name__)
MODEL_DIR = "models" # Assuming models are saved here by analysis/ml_models.py

@main.route("/")
def home():
    trending = []
    try:
        # It's better to use a full path or make sure the CWD is correct
        trending_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trending_stocks.csv')
        if os.path.exists(trending_csv_path):
            df = pd.read_csv(trending_csv_path)
            trending = df.head(10).to_dict(orient="records")
        else:
            print(f"Error: trending_stocks.csv not found at {trending_csv_path}")
            # Consider logging to discord here too
    except Exception as e:
        print(f"Error loading trending stocks: {e}")
    return render_template("home.html", trending=trending)


@main.route("/prediction")
def show_predictions():
    stocks = ["Infosys", "TCS", "Reliance"] # You might want to make this dynamic
    predictions_display_data = [] # Renamed to avoid conflict if 'predictions' is used elsewhere

    for stock_name in stocks:
        current_prediction_val = predict_next_close(stock_name) # Get the single next day prediction
        
        model_path = os.path.join(MODEL_DIR, f"{stock_name}_model.pkl")
        historical_data_path = os.path.join("data", f"historical_{stock_name}.csv") # Path to historical data

        r2_on_test = "-"
        plot_img_filename = f"{stock_name}_trend.png"
        plot_img_path = os.path.join("webapp", "static", plot_img_filename)


        try:
            if not os.path.exists(model_path):
                print(f"Model for {stock_name} not found at {model_path}. Training now...")
                train_model(stock_name) # Train if model doesn't exist

            if os.path.exists(model_path) and os.path.exists(historical_data_path):
                model = joblib.load(model_path)
                
                # Recreate the test set in the same way as in analysis/ml_models.py
                # This requires access to build_feature_matrix and the same X, y definition
                df_features = build_feature_matrix(stock_name) # From analysis.ml_models
                if df_features is not None and not df_features.empty:
                    # Ensure columns match those used in analysis/ml_models.py
                    feature_cols = ["prev_close", "open", "high", "low", "volume", "shock_percent", "assets", "liabilities"]
                    X = df_features[feature_cols]
                    y = df_features["close"]

                    # Important: fillna to match the training process
                    X = X.fillna(method='ffill').fillna(method='bfill')
                    
                    if not X.empty and not y.empty:
                        # Use the same train_test_split parameters as in analysis/ml_models.py
                        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Assuming random_state=42 was used
                        
                        if not X_test.empty:
                            y_pred_test = model.predict(X_test)
                            r2_on_test = round(r2_score(y_test, y_pred_test), 2)
                        else:
                            print(f"X_test is empty for {stock_name}, cannot calculate R2 score.")
                    else:
                        print(f"Feature matrix X or target y is empty for {stock_name} after processing.")
                else:
                    print(f"Feature matrix could not be built for {stock_name}.")

            else:
                print(f"Model or historical data not found for {stock_name} to calculate R2 score.")

            # Plotting current actuals and the single next day prediction
            if os.path.exists(historical_data_path):
                df_hist = pd.read_csv(historical_data_path)
                df_hist["close"] = pd.to_numeric(df_hist["close"], errors="coerce")
                df_hist = df_hist.dropna(subset=["close"]).tail(30)

                plt.figure(figsize=(6, 3))
                df_hist["close"].plot(label="Actual Recent Closes", color="skyblue")
                if current_prediction_val is not None:
                    plt.axhline(y=current_prediction_val, color='orange', linestyle='--', label=f"Predicted Next Close: ₹{current_prediction_val}")
                plt.title(f"{stock_name} - Recent Prices & Next Prediction")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_img_path) # Save to webapp/static
                plt.close()
            else:
                print(f"Historical data for plotting not found: {historical_data_path}")


        except FileNotFoundError as fnf_error:
            print(f"File not found during evaluation for {stock_name}: {fnf_error}")
        except Exception as e:
            print(f"Error evaluating or plotting {stock_name}: {e}")

        predictions_display_data.append({
            "stock": stock_name,
            "predicted_next_close": current_prediction_val if current_prediction_val is not None else "N/A",
            "r2_score_on_test": r2_on_test, # This R2 is from the model's test set
            "img": plot_img_filename
        })
    
    # The template name in your app.py for '/prediction' POST is 'prediction.html', 
    # but for GET here it seems you intended to show multiple predictions, so let's make a new template or adjust.
    # For now, assuming prediction.html can handle a list of 'predictions_display_data'.
    # Your original template in logs.html seems more suited for this loop:
    # Changing template to "logs.html" based on its structure, and variable name to `predictions`
    return render_template("logs.html", predictions=predictions_display_data)


@main.route("/logs") # Your original /logs route
def view_logs():
    records = []
    log_csv_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'predictions_log.csv')
    try:
        if os.path.exists(log_csv_path):
            df = pd.read_csv(log_csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by="timestamp", ascending=False)
            records = df.to_dict(orient="records")
        else:
            print(f"Log file not found: {log_csv_path}")
    except Exception as e:
        print(f"Error loading logs: {e}")
        
    # Ensure your logs.html template is designed to iterate through 'logs'
    return render_template("logs.html", logs=records) # Passing 'logs=records'