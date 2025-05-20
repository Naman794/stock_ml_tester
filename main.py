# main.py

import os
import threading
from time import sleep
from filters.fetch_all_data import fetch_all_data
from analysis.ml_models import train_model, predict_next_close
from scheduler import start_scheduler
from webapp.app import create_app
from fetch.api_manager import fetch_historical
from fetch.csv_writer import save_to_csv
from analysis.ml_models import train_model, predict_next_close
from filters.fetch_all_data import TOP_10_STOCKS


STOCKS = TOP_10_STOCKS

def run_all_models():
    for stock in ["Infosys", "TCS", "Reliance"]:
        print(f"🔁 Running ML for {stock}")
        train_model(stock)
        prediction = predict_next_close(stock)
        print(f"📈 {stock} → Tomorrow prediction: ₹{prediction}")

def ensure_data_folder():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def run_pipeline_for_stock(stock_name):
    print(f"\n📊 Running pipeline for: {stock_name}")
    try:
        response = fetch_historical(stock_name, period="6m")
        data = response.get("data", [])
        if not data:
            print(f"[⚠️] No historical data for {stock_name}")
            return
    except Exception as e:
        print(f"[❌] Failed to fetch historical data: {e}")
        return

    filename = f"historical_{stock_name.replace(' ', '_')}.csv"
    save_to_csv(data, filename)
    train_model(f"data/{filename}")
    predicted = predict_next_close(f"data/{filename}")
    print(f"✅ {stock_name} → Predicted tomorrow close: ₹{predicted}")

def run_all_pipelines():
    ensure_data_folder()
    for stock in STOCKS:
        run_pipeline_for_stock(stock)

def run_webapp():
    app = create_app()
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    print("🚀 Launching Stock Prediction Engine + Dashboard...")
    
    start_scheduler()
    # Thread 1: Run pipeline
    pipeline_thread = threading.Thread(target=run_all_pipelines)
    pipeline_thread.start()


    # Thread 2: Run Flask app
    app_thread = threading.Thread(target=run_webapp)
    app_thread.start()

    # Wait for both to finish (or keep app running)
    pipeline_thread.join()
    app_thread.join()
