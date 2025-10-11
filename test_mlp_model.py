import pandas as pd
from tensorflow import keras

model = keras.models.load_model("mlp_efficiency_model.keras")

df = pd.read_csv("../data/synthetic_efficiency.csv")

X = df.drop(columns=['efficiency_score'])
y = df['efficiency_score']

predictions = model.predict(X[:5])  # Predict on first 5 rows
print("\n Predicted Efficiency Values:")
print(predictions)

print("\n Actual Efficiency Values:")
print(y[:5].values)
