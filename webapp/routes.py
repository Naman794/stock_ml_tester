from flask import Blueprint, render_template
from analysis.ml_models import predict_next_close, train_model
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

main = Blueprint("main", __name__)

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
        except:
            pass

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
        except:
            pass

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
    except:
        records = []

    return render_template("logs.html", logs=records)