# filters/fetch_all_data.py

import http.client
import json
import os
import pandas as pd
from urllib.parse import urlencode, urlparse
from config import API_BASE, API_KEY

# ✅ Parse base URL correctly
parsed_url = urlparse(API_BASE)
API_HOST = parsed_url.netloc
HEADERS = {"X-Api-Key": API_KEY}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Define your Top 10 stocks (adjust names if API expects different format)
TOP_10_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "BHARTIARTL", "SBI", "KOTAKBANK"
]

def fetch_and_save(endpoint, filename, params=None):
    conn = http.client.HTTPSConnection(API_HOST)
    try:
        path = endpoint
        if params:
            query = urlencode(params)
            path = f"{endpoint}?{query}"
        conn.request("GET", path, headers=HEADERS)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        parsed = json.loads(data)

        # Normalize
        if isinstance(parsed, dict):
            if "data" in parsed:
                parsed = parsed["data"]
            elif "trending_stocks" in parsed:
                parsed = parsed["trending_stocks"].get("top_gainers", []) + parsed["trending_stocks"].get("top_losers", [])
            elif "results" in parsed:
                parsed = parsed["results"]

        # Save if valid
        if isinstance(parsed, list) and len(parsed) > 0:
            df = pd.DataFrame(parsed)
        elif isinstance(parsed, dict) and len(parsed) > 0:
            df = pd.DataFrame([parsed])

        if df is not None:
            df.to_csv(f"{DATA_DIR}/{filename}", index=False)
            print(f"[✅] Saved {filename} ({len(df)} rows)")
        else:
            print(f"[⚠️] No data returned for {endpoint}")

    except Exception as e:
        print(f"[❌] Error fetching {endpoint}: {e}")
    finally:
        conn.close()

def fetch_all_data():
    print("📦 Fetching all Indian Stock API GET endpoints...\n")

    # General GETs
    fetch_and_save("/news", "news.csv")
    fetch_and_save("/trending", "trending_stocks.csv")
    fetch_and_save("/commodities", "commodities.csv")
    fetch_and_save("/mutual_funds", "mutual_funds.csv")
    fetch_and_save("/price_shockers", "shockers.csv")
    fetch_and_save("/BSE_most_active", "bse_active.csv")
    fetch_and_save("/NSE_most_active", "nse_active.csv")
    fetch_and_save("/mutual_fund_search", "mf_search.csv")
    fetch_and_save("/fetch_52_week_high_low_data", "fifty_two_week.csv")

    # Stock-specific GETs
    for stock in TOP_10_STOCKS:
        print(f"\n🔍 Fetching data for: {stock}")
        fetch_and_save("/statement", f"statement_balancesheet_{stock}.csv", {"stock_name": stock, "stats": "balancesheet"})
        fetch_and_save("/statement", f"statement_cashflow_{stock}.csv", {"stock_name": stock, "stats": "cashflow"})
        fetch_and_save("/statement", f"statement_quarter_{stock}.csv", {"stock_name": stock, "stats": "quarter_results"})
        fetch_and_save("/statement", f"statement_yoy_{stock}.csv", {"stock_name": stock, "stats": "yoy_results"})
        fetch_and_save("/corporate_actions", f"corporate_actions_{stock}.csv", {"stock_name": stock})
        fetch_and_save("/historical_data", f"historical_{stock}.csv", {"stock_name": stock, "period": "max", "filters": "default"})
        fetch_and_save("/stock_target_price", f"target_price_{stock}.csv", {"stock_id": stock})
        fetch_and_save("/recent_announcements", f"recent_announcements_{stock}.csv", {"stock_name": stock})
        fetch_and_save("/stock", f"stock_details_{stock}.csv", {"stock_name" : stock })
        fetch_and_save("/industry_search", f"industries_{stock}.csv", {"stock_name" : stock})

    print("\n✅ All data fetched and stored in /data/ folder.\n")


