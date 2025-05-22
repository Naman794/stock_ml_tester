# filters/fetch_all_data.py

import http.client
import json
import os
import pandas as pd
from urllib.parse import urlencode, urlparse # No, this was API_BASE, changed to API_HOST
from config import API_BASE, API_KEY # API_BASE is "stock.indianapi.in"
from utils.logger import log_to_discord # Using your logger

# Parse base URL correctly
# API_HOST should be just the hostname, e.g., "stock.indianapi.in"
parsed_url = urlparse(f"https://{API_BASE}") # Ensure scheme for proper parsing if API_BASE doesn't include it
API_HOST = parsed_url.netloc
if not API_HOST: # Fallback if API_BASE was already just the host
    API_HOST = API_BASE

HEADERS = {"X-Api-Key": API_KEY}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define your Top 10 stocks (adjust names if API expects different format)
TOP_10_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "BHARTIARTL", "SBI", "KOTAKBANK"
]

def fetch_and_save(endpoint, filename, params=None):
    conn = http.client.HTTPSConnection(API_HOST)
    df = None # Initialize df to None
    try:
        path = endpoint
        if params:
            query = urlencode(params)
            path = f"{endpoint}?{query}"
        
        print(f"ℹ️ Requesting: https://{API_HOST}{path}") # Log the request URL
        conn.request("GET", path, headers=HEADERS)
        res = conn.getresponse()
        
        data = res.read().decode("utf-8")

        if res.status == 200:
            try:
                parsed_json = json.loads(data)

                # More robust normalization:
                # Check if the primary content is in a known key like 'data', 'results', etc.
                # This part is highly dependent on your API's response structure for *successful* calls.
                # You'll need to adapt this based on how your API returns valid data vs. errors.
                
                actual_data = None
                if isinstance(parsed_json, dict):
                    if "data" in parsed_json and isinstance(parsed_json["data"], (list, dict)):
                        actual_data = parsed_json["data"]
                    elif "results" in parsed_json and isinstance(parsed_json["results"], (list, dict)):
                        actual_data = parsed_json["results"]
                    elif "trending_stocks" in parsed_json and isinstance(parsed_json["trending_stocks"], dict): # Specific to /trending
                         actual_data = parsed_json["trending_stocks"].get("top_gainers", []) + \
                                       parsed_json["trending_stocks"].get("top_losers", [])
                    # Add more specific checks for other endpoints if their structure varies
                    # For example, if some endpoints return data at the top level of the JSON:
                    elif endpoint == "/BSE_most_active" or endpoint == "/NSE_most_active": # Assuming these return a list directly if successful
                        if isinstance(parsed_json, list): # Or check for a key that always exists in success
                           actual_data = parsed_json
                    # If the JSON itself is the data (e.g. for stock details which might be a single dict)
                    elif "ticker" in parsed_json or "companyName" in parsed_json: # Heuristic for detail-like objects
                        actual_data = parsed_json
                    else: # If it's a dict but doesn't match known data containers, it might be an error or unexpected structure
                        log_to_discord(f"[⚠️] Unexpected JSON structure for {endpoint} with params {params}. Content: {data[:200]}")
                        print(f"[⚠️] Unexpected JSON structure for {endpoint}. Data: {data[:200]}")
                        actual_data = None # Explicitly set to None

                elif isinstance(parsed_json, list): # If the top level is a list, assume it's the data
                    actual_data = parsed_json
                else:
                    log_to_discord(f"[⚠️] Unexpected data type after JSON parse for {endpoint}: {type(parsed_json)}")
                    print(f"[⚠️] Unexpected data type after JSON parse for {endpoint}: {type(parsed_json)}")


                if actual_data is not None:
                    if isinstance(actual_data, list):
                        if len(actual_data) > 0:
                            df = pd.DataFrame(actual_data)
                        else:
                            print(f"[ℹ️] No data entries returned for {endpoint} (empty list).")
                            # Optionally save an empty CSV or a CSV with headers only
                            # For now, we'll just not save, df remains None
                    elif isinstance(actual_data, dict): # For single item results
                        if len(actual_data) > 0:
                            df = pd.DataFrame([actual_data])
                        else:
                             print(f"[ℹ️] No data entries returned for {endpoint} (empty dict).")
                    else:
                        print(f"[⚠️] 'actual_data' is not a list or dict for {endpoint}")

                else: # actual_data is None
                    # This implies the structure didn't match any success case,
                    # it might be an API error message that is valid JSON but not data.
                    # e.g. {'error': 'message'} or {'detail': [{'type': 'missing', ...}]}
                    log_to_discord(f"[API_ERROR_JSON] API returned a JSON error for {endpoint} with params {params}. Response: {parsed_json}")
                    print(f"[API_ERROR_JSON] API returned a JSON error for {endpoint}. Response: {parsed_json}")


            except json.JSONDecodeError:
                log_to_discord(f"[❌] JSON Decode Error for {endpoint} with params {params}. Status: {res.status}. Response: {data[:200]}")
                print(f"[❌] JSON Decode Error for {endpoint}. Response: {data[:200]}")
        else:
            log_to_discord(f"[❌] HTTP Error for {endpoint} with params {params}. Status: {res.status}. Response: {data[:200]}")
            print(f"[❌] HTTP Error for {endpoint}. Status: {res.status}. Response: {data[:200]}")

        if df is not None and not df.empty:
            # Ensure data dir exists (already done globally, but good practice if function is isolated)
            # os.makedirs(DATA_DIR, exist_ok=True) 
            
            # Before saving, maybe a quick check for essential columns if possible/known
            # For example, historical data should have 'date' and 'close'
            # if "historical" in filename and not ({'date', 'close'} <= set(df.columns)):
            #     print(f"[⚠️] Critical columns missing in data for {filename}. Not saving.")
            #     log_to_discord(f"[CRITICAL_DATA_ISSUE] Critical columns missing in {filename}. Not saving.")
            # else:
            df.to_csv(f"{DATA_DIR}/{filename}", index=False)
            print(f"[✅] Saved {filename} ({len(df)} rows)")

        elif df is not None and df.empty: # df was created but resulted in an empty DataFrame
            print(f"[ℹ️] Processed data for {filename} but it resulted in an empty DataFrame. Not saving.")
        else: # df is None (error occurred or no data)
            print(f"[⚠️] No valid data processed or error occurred for {filename}. Not saving CSV.")

    except Exception as e:
        # Log the exception to discord and console
        log_to_discord(f"[❌] Outer Exception in fetch_and_save for {endpoint} with params {params}: {e}")
        print(f"[❌] Error fetching {endpoint}: {e}")
    finally:
        conn.close()

# ... (rest of your fetch_all_data function) ...

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


