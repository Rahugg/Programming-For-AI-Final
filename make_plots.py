import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.data_utils import load_data, preprocess_data, split_data

# === 1. Load dataset ===
df = load_data("data/cardetailsv4.csv")
X, y, columns = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# === 2. Load best model ===
bundle = joblib.load("models/best_model.joblib")
model = bundle["model"]

# === 3. Predict ===
y_pred = model.predict(X_test)

# === 4. Actual vs Predicted plot ===
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", linewidth=2
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Best Model)")
plt.grid(alpha=0.3)

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/actual_vs_predicted.png", dpi=300)
plt.close()

print("Saved: reports/actual_vs_predicted.png")

# === 5. Feature Importance ===
# NOTE: Works only for RF / XGBoost
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]   # top 15 features

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(columns)[indices], importances[indices])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance (Best Model)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300)
    plt.close()

    print("Saved: reports/feature_importance.png")
else:
    print("This model does not support feature_importances_.")
