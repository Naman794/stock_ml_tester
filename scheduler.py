# scheduler.py

import schedule
import time
import threading
from filters.fetch_all_data import fetch_all_data
from analysis.ml_models import train_model, predict_next_close
from utils.logger import log_to_discord
from fetch.csv_writer import save_to_csv
import pandas as pd
from datetime import datetime
import os

STOCKS = ["Infosys", "TCS", "Reliance"]

def log_prediction_and_alert(stock, predicted):
    log_path = "logs/predictions_log.csv"
    hist_path = f"data/historical_{stock}.csv"
    
    try:
        df = pd.read_csv(hist_path)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        actual = df["close"].dropna().iloc[-1]
        error = abs(predicted - actual) / actual

        # Log to file
        entry = pd.DataFrame([{
            "timestamp": datetime.now(),
            "stock": stock,
            "predicted_close": predicted,
            "actual_close": actual,
            "error_pct": round(error * 100, 2)
        }])
        if os.path.exists(log_path):
            entry.to_csv(log_path, mode="a", header=False, index=False)
        else:
            entry.to_csv(log_path, index=False)

        # Discord alert if error > 10%
        if error > 0.10:
            log_to_discord(f"⚠️ {stock} prediction error > 10%\nActual: ₹{actual}\nPredicted: ₹{predicted}")
    
    except Exception as e:
        print(f"[❌] Prediction log error for {stock}: {e}")

def daily_job():
    log_to_discord("📦 Starting scheduled data fetch + retraining.")
    
    fetch_all_data()
    for stock in STOCKS:
        train_model(stock)
        predicted = predict_next_close(stock)
        log_prediction_and_alert(stock, predicted)
        log_to_discord(f"✅ {stock} retrained. Predicted close: ₹{predicted}")

    log_to_discord("✅ All stocks updated successfully.")

def start_scheduler():
    schedule.every(12).hours.do(daily_job)  # or change to .day.at("06:00")

    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(30)

    thread = threading.Thread(target=run_schedule)
    thread.daemon = True
    thread.start()
