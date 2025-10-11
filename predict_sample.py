import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1️⃣ Load the trained model
model = load_model("mlp_efficiency_model.keras")

# 2️⃣ Load the scaler
scaler = joblib.load("../scaler.save")

# 3️⃣ Example input for one person
# Format: [sleep_hours, sleep_quality, steps, sitting_hours, exercise_mins,
# calories, water_l, screen_time, stress_level, age, bmi, resting_hr]

sample = np.array([[7.0, 0.8, 8000, 7.0, 30, 2200, 2.0, 4.0, 3.0, 25, 23.5, 68]])

# 4️⃣ Scale the input (same way model was trained)
sample_scaled = scaler.transform(sample)

# 5️⃣ Predict efficiency
pred = model.predict(sample_scaled)[0][0]

# 6️⃣ Print the result
print(f"Predicted Efficiency Score: {pred:.1f}/100")
