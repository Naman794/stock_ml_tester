# fetch/api_manager.py

import http.client
import json
import time
import os
from urllib.parse import urlencode
from datetime import datetime, timedelta

from config import API_BASE, API_KEY
from fetch.csv_writer import save_to_csv
from utils.logger import log_to_discord

HEADERS = {"X-Api-Key": API_KEY}
API_HOST = API_BASE
LAST_RUN_FILE = "data/last_run.json"
RUN_INTERVALS = {
    "trending": 1,           # in hours
    "mutual_funds": 12,
    "news": 3,
    "nse_active": 2,
    "bse_active": 2,
    "shockers": 1,
    "historical": 24
}

# Load last run timestamps
def load_last_run():
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, "r") as f:
            return json.load(f)
    return {}

def update_last_run(key):
    timestamps = load_last_run()
    timestamps[key] = datetime.now().isoformat()
    os.makedirs("data", exist_ok=True)
    with open(LAST_RUN_FILE, "w") as f:
        json.dump(timestamps, f)

def should_run(key):
    timestamps = load_last_run()
    last_time = timestamps.get(key)
    if not last_time:
        return True
    elapsed = datetime.now() - datetime.fromisoformat(last_time)
    return elapsed > timedelta(hours=RUN_INTERVALS.get(key, 24))

# Generic GET request with retry + logging
def fetch_endpoint(path, params=None):
    conn = http.client.HTTPSConnection(API_HOST)
    try:
        if params:
            query = urlencode(params)
            path = f"{path}?{query}"
        conn.request("GET", path, headers=HEADERS)
        res = conn.getresponse()
        if res.status != 200:
            log_to_discord(f"[❌] API failed for {path} → {res.status}")
            return {}
        data = res.read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        log_to_discord(f"[❌] API Error on {path}: {e}")
        return {}
    finally:
        conn.close()

# Endpoint functions
def fetch_trending():
    if should_run("trending"):
        data = fetch_endpoint("/trending")
        parsed = data.get("trending_stocks", {}).get("top_gainers", []) + data.get("trending_stocks", {}).get("top_losers", [])
        save_to_csv(parsed, "trending.csv")
        update_last_run("trending")
        log_to_discord("📊 Trending stocks fetched.")

def fetch_mutual_funds():
    if should_run("mutual_funds"):
        data = fetch_endpoint("/mutual_funds")
        save_to_csv(data.get("data", []), "mutual_funds.csv")
        update_last_run("mutual_funds")
        log_to_discord("💰 Mutual funds data updated.")

def fetch_news():
    if should_run("news"):
        data = fetch_endpoint("/news")
        save_to_csv(data.get("data", []), "news.csv")
        update_last_run("news")
        log_to_discord("🗞 News data refreshed.")

def fetch_nse_active():
    if should_run("nse_active"):
        data = fetch_endpoint("/NSE_most_active")
        save_to_csv(data.get("data", []), "nse_active.csv")
        update_last_run("nse_active")
        log_to_discord("📈 NSE Most Active stocks updated.")

def fetch_bse_active():
    if should_run("bse_active"):
        data = fetch_endpoint("/BSE_most_active")
        save_to_csv(data.get("data", []), "bse_active.csv")
        update_last_run("bse_active")
        log_to_discord("📉 BSE Most Active stocks updated.")

def fetch_shockers():
    if should_run("shockers"):
        data = fetch_endpoint("/price_shockers")
        save_to_csv(data.get("data", []), "shockers.csv")
        update_last_run("shockers")
        log_to_discord("⚡ Price shockers refreshed.")

def fetch_historical(stock_name, period="6m", filter="price"):
    key = f"historical_{stock_name}"
    if should_run(key):
        data = fetch_endpoint("/historical", {
            "stock_name": stock_name,
            "period": period,
            "filters": filter
        })
        save_to_csv(data.get("data", []), f"historical_{stock_name}.csv")
        update_last_run(key)
        log_to_discord(f"📚 Historical data updated for {stock_name}.")

# Run all at once
def run_all_scheduled(stock_list=["Infosys", "TCS", "Reliance"]):
    fetch_trending()
    fetch_news()
    fetch_mutual_funds()
    fetch_nse_active()
    fetch_bse_active()
    fetch_shockers()
    for stock in stock_list:
        fetch_historical(stock)
