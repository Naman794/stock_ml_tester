import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    print(f"🔍 Ensuring all data is available for: {stock}")
    for template in REQUIRED_FILES:
        filename = template.format(stock=stock)
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"📁 Missing: {filename} — fetching from API...")
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
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def create_features(df):
    df = df.copy()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    X = df[['MA10', 'MA50', 'Volatility']]
    y = df['Target']
    return X, y

def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, predictions))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(y_test.reset_index(drop=True).values, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.set_title("Stock Movement Prediction")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Movement (0=Down, 1=Up)")
    ax.legend()
    return predictions[-1], fig

def analyze_stock(stock):
    stock = stock.upper()
    ensure_data_available(stock)
    df = load_historical_data(stock)
    X, y = create_features(df)
    prediction, fig = train_and_predict(X, y)
    fig.show()
    return prediction

if __name__ == "__main__":
    stock = input("🔎 Enter stock name (e.g., INFY, RELIANCE): ").upper()
    try:
        result = analyze_stock(stock)
        print(f"\n📈 Prediction for {stock}: {'🔼 Up' if result == 1 else '🔽 Down'}")
    except Exception as e:
        print(f"❌ Error: {e}")
