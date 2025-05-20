from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, predictions))

    fig, ax = plt.subplots()
    ax.plot(y_test.reset_index(drop=True).values, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.set_title("Stock Movement Prediction")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Movement (0=Down, 1=Up)")
    ax.legend()
    return predictions[-1], fig
