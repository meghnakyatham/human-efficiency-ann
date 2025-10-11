import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

model = load_model("mlp_efficiency_model.keras")

df = pd.read_csv("../data/synthetic_efficiency.csv")

X = df.drop(columns=['efficiency_score'])
y = df['efficiency_score']

scaler = joblib.load("../scaler.save")
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled).ravel()

print("MAE (Mean Absolute Error):", mean_absolute_error(y, y_pred))
print("MSE (Mean Squared Error):", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.4, color='blue')
plt.plot([0, 100], [0, 100], 'r--')  # diagonal line
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.title("ANN Predicted vs Actual Efficiency")
plt.show()
