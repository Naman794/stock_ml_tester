# analysis/ml_models.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_feature_matrix(stock_name):
    try:
        # Load core historical
        hist_path = f"data/historical_{stock_name}.csv"
        df = pd.read_csv(hist_path)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna()

        # Load balance sheet
        try:
            bs_df = pd.read_csv(f"data/statement_balancesheet_{stock_name}.csv")
            df["assets"] = bs_df.get("totalAssets", [np.nan]*len(df))
            df["liabilities"] = bs_df.get("totalLiabilities", [np.nan]*len(df))
        except:
            df["assets"] = np.nan
            df["liabilities"] = np.nan

        # Load shockers
        try:
            shock_df = pd.read_csv("data/shockers.csv")
            match = shock_df[shock_df["ticker_id"].str.contains(stock_name, case=False, na=False)]
            df["shock_percent"] = match["percent_change"].values[0] if not match.empty else 0
        except:
            df["shock_percent"] = 0

        # Lag features
        df["prev_close"] = df["close"].shift(1)
        df["price_change"] = df["close"].pct_change()

        # Drop NA rows
        df = df.dropna()

        return df
    except Exception as e:
        print(f"[❌] Feature matrix error for {stock_name}: {e}")
        return None

def train_model(stock_name):
    df = build_feature_matrix(stock_name)
    if df is None or df.empty:
        return None

    X = df[["prev_close", "open", "high", "low", "volume", "shock_percent", "assets", "liabilities"]]
    y = df["close"]

    X = X.fillna(method='ffill').fillna(method='bfill')  # handle any leftover missing values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = f"{MODEL_DIR}/{stock_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Trained model for {stock_name}")
    return model_path

def predict_next_close(stock_name):
    model_path = f"{MODEL_DIR}/{stock_name}_model.pkl"
    csv_path = f"data/historical_{stock_name}.csv"
    if not os.path.exists(model_path):
        train_model(stock_name)

    model = joblib.load(model_path)
    df = build_feature_matrix(stock_name)

    if df is None or df.empty:
        return None

    X = df[["prev_close", "open", "high", "low", "volume", "shock_percent", "assets", "liabilities"]].tail(1)
    X = X.fillna(method='ffill').fillna(method='bfill')

    prediction = model.predict(X)[0]
    return round(prediction, 2)
