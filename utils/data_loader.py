import os
from filters.fetch_all_data import fetch_and_save

DATA_DIR = "data"

REQUIRED_FILES = [
    "historical_{stock}.csv",
    "statement_balancesheet_{stock}.csv",
    "statement_cashflow_{stock}.csv",
    "statement_yoy_{stock}.csv",
    "statement_quarter_{stock}.csv",
    "corporate_actions_{stock}.csv",
    "target_price_{stock}.csv",
    "recent_announcements_{stock}.csv",
    "industries_{stock}.csv",
    "stock_details_{stock}.csv"
]

def ensure_data_available(stock):
    for template in REQUIRED_FILES:
        filename = template.format(stock=stock)
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"📁 Missing: {filename} — fetching...")
            if "historical" in filename:
                fetch_and_save("/historical_data", filename, {"stock_name": stock, "period": "6m", "filters": "price"})
            elif "balancesheet" in filename:
                fetch_and_save("/statement", filename, {"stock_name": stock, "stats": "balancesheet"})
            elif "cashflow" in filename:
                fetch_and_save("/statement", filename, {"stock_name": stock, "stats": "cashflow"})
            elif "yoy" in filename:
                fetch_and_save("/statement", filename, {"stock_name": stock, "stats": "yoy_results"})
            elif "quarter" in filename:
                fetch_and_save("/statement", filename, {"stock_name": stock, "stats": "quarter_results"})
            elif "corporate_actions" in filename:
                fetch_and_save("/corporate_actions", filename, {"stock_name": stock})
            elif "target_price" in filename:
                fetch_and_save("/stock_target_price", filename, {"stock_id": stock})
            elif "recent_announcements" in filename:
                fetch_and_save("/recent_announcements", filename, {"stock_name": stock})
            elif "industries" in filename:
                fetch_and_save("/industry_search", filename, {"stock_name": stock})
            elif "stock_details" in filename:
                fetch_and_save("/stock", filename, {"stock_name": stock})

def load_historical_data(stock):
    filepath = os.path.join(DATA_DIR, f"historical_{stock}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing historical data for {stock}")
    import pandas as pd
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
