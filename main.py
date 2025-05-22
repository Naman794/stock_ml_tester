# main.py

import os
import threading
from analysis.ml_models import train_model, predict_next_close
from scheduler import start_scheduler
from fetch.api_manager import run_all_scheduled
from webapp.app import create_app
from config import API_BASE, API_KEY

def run_pipeline_from_all_csvs():
    print("📁 Using all available CSV data from /data/")
    historical_files = [f for f in os.listdir("data") if f.startswith("historical_") and f.endswith(".csv")]
    top_stocks = sorted(historical_files)[:10]  # Top 10 alphabetically

    for file in top_stocks:
        stock_name = file.replace("historical_", "").replace(".csv", "")
        print(f"🔍 Preparing features for {stock_name}")

        try:
            train_model(stock_name)  # ML uses all related files internally
            predicted = predict_next_close(stock_name)
            print(f"✅ {stock_name} → Predicted Close: ₹{predicted}")
        except Exception as e:
            print(f"[❌] Error processing {stock_name}: {e}")

def run_webapp():
    app = create_app()
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    print("🚀 Starting Stock Intelligence Suite")

    start_scheduler()             # Schedule auto-fetch/retrain
    run_all_scheduled()           # Initial fetch check (throttled)
    run_pipeline_from_all_csvs()  # Use all data/*.csv files

    # Start Flask dashboard
    threading.Thread(target=run_webapp).start()
