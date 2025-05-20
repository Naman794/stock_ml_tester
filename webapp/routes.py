from flask import Blueprint, render_template, request
from analysis.ml_models import predict_next_close, train_model
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

main = Blueprint("main", __name__)

@main.route("/")
def home():
    trending = []
    try:
        df = pd.read_csv("data/trending_stocks.csv")
        trending = df.head(10).to_dict(orient="records")  # Top 10 trending stocks
    except Exception as e:
        print("Error loading trending stocks:", e)

    return render_template("home.html", trending=trending)


@main.route("/prediction")
def show_predictions():
    stocks = ["Infosys", "TCS", "Reliance"]
    predictions = []

    for stock in stocks:
        pred = predict_next_close(stock)
        model_path = f"models/{stock}_model.pkl"
        hist_path = f"data/historical_{stock}.csv"

        # Accuracy
        r2 = "-"
        try:
            df = pd.read_csv(hist_path)
            df = df.dropna(subset=["close", "open", "high", "low", "volume"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            y_true = df["close"].tail(30)

            # Model retrain & eval
            train_model(stock)
            y_pred = [predict_next_close(stock)] * len(y_true)
            r2 = round(r2_score(y_true, y_pred), 2)
        except Exception as e:
            print(f"Error evaluating {stock}: {e}")

        # Plot
        try:
            df = pd.read_csv(hist_path)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna().tail(30)

            plt.figure(figsize=(6, 3))
            df["close"].plot(label="Actual", color="skyblue")
            plt.axhline(y=pred, color='orange', linestyle='--', label=f"Predicted ₹{pred}")
            plt.title(f"{stock} - Price vs Predicted")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"webapp/static/{stock}_trend.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting {stock}: {e}")

        predictions.append({
            "stock": stock,
            "predicted": pred,
            "r2_score": r2,
            "img": f"{stock}_trend.png"
        })

    return render_template("prediction.html", predictions=predictions)


@main.route("/logs")
def view_logs():
    try:
        df = pd.read_csv("logs/predictions_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp", ascending=False)
        records = df.to_dict(orient="records")
    except Exception as e:
        print("Error loading logs:", e)
        records = []

    return render_template("logs.html", logs=records)
