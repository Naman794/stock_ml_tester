# filters/fetch_all_data.py

import http.client
import json
import os
import pandas as pd
from urllib.parse import urlencode

API_HOST = "stock.indianapi.in"
API_KEY = "Naman@#2025API!Project"
HEADERS = {"X-Api-Key": API_KEY}

DATA_DIR = "data"

# Ensure data/ folder exists
os.makedirs(DATA_DIR, exist_ok=True)

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

        # Normalize data if nested
        if isinstance(parsed, dict):
            if "data" in parsed:
                parsed = parsed["data"]
            elif "trending_stocks" in parsed:
                parsed = parsed["trending_stocks"].get("top_gainers", []) + parsed["trending_stocks"].get("top_losers", [])
            elif "results" in parsed:
                parsed = parsed["results"]

        # Save
        if isinstance(parsed, list) and len(parsed) > 0:
            df = pd.DataFrame(parsed)
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

    fetch_and_save("/news", "news.csv")
    fetch_and_save("/stock", "stock_details.csv")
    fetch_and_save("/trending", "trending_stocks.csv")
    fetch_and_save("/commodities", "commodities.csv")
    fetch_and_save("/mutual_funds", "mutual_funds.csv")
    fetch_and_save("/price_shockers", "shockers.csv")
    fetch_and_save("/BSE_most_active", "bse_active.csv")
    fetch_and_save("/NSE_most_active", "nse_active.csv")
    fetch_and_save("/industry", "industries.csv")
    fetch_and_save("/mf_search", "mf_search.csv")
    fetch_and_save("/fetch_52week", "fifty_two_week.csv")

    # Parameterized GETs
    stock_samples = ["Infosys", "TCS", "Reliance"]
    for stock in stock_samples:
        fetch_and_save("/statement", f"statement_balancesheet_{stock}.csv", {"stock_name": stock, "stats": "balancesheet"})
        fetch_and_save("/statement", f"statement_cashflow_{stock}.csv", {"stock_name": stock, "stats": "cashflow"})
        fetch_and_save("/statement", f"statement_quarter_{stock}.csv", {"stock_name": stock, "stats": "quarter_results"})
        fetch_and_save("/statement", f"statement_yoy_{stock}.csv", {"stock_name": stock, "stats": "yoy_results"})
        fetch_and_save("/corporate_actions", f"corporate_actions_{stock}.csv", {"stock_name": stock})
        fetch_and_save("/historical", f"historical_{stock}.csv", {"stock_name": stock, "period": "6m", "filters": "price"})
        fetch_and_save("/target_price", f"target_price_{stock}.csv", {"stock_id": stock})
        fetch_and_save("/recent_announcement", f"recent_announcements_{stock}.csv", {"stock_name": stock})

    print("\n✅ All data fetched and stored in /data/ folder.\n")

# Run test manually
if __name__ == "__main__":
    fetch_all_data()
