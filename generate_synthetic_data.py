"""Generate a synthetic dataset for daily efficiency prediction.
Creates data/synthetic_efficiency.csv
"""
import numpy as np
import pandas as pd
import os

OUT = os.path.join(os.path.dirname(__file__), "synthetic_efficiency.csv")
np.random.seed(42)

N = 2000  # number of samples (days or person-days)
# Features (reasonable ranges)
sleep_hours = np.clip(np.random.normal(7, 1.5, N), 3, 10)
sleep_quality = np.clip(np.random.beta(2,2, N), 0.0, 1.0)
steps = np.clip(np.random.normal(6000, 3000, N), 0, 30000)
sitting_hours = np.clip(np.random.normal(8, 2.5, N), 0, 20)
exercise_mins = np.clip(np.random.normal(30, 20, N), 0, 180)
calories = np.clip(np.random.normal(2200, 400, N), 800, 4000)
water_l = np.clip(np.random.normal(2.0, 0.7, N), 0.2, 6.0)
screen_time = np.clip(np.random.normal(6, 2.5, N), 0, 20)
stress_level = np.clip(np.random.normal(4, 2, N), 0, 10)
age = np.random.randint(18, 65, N)
bmi = np.clip(np.random.normal(24, 4, N), 15, 40)
resting_hr = np.clip(np.random.normal(70, 10, N), 45, 120)

# Create a synthetic target (efficiency score 0-100) with some realistic correlations
eff = (
    0.25 * (sleep_hours - 4) + 
    15 * sleep_quality + 
    0.0005 * steps - 
    0.9 * sitting_hours + 
    0.02 * exercise_mins - 
    0.0008 * calories + 
    2.5 * water_l - 
    1.2 * screen_time - 
    2.0 * (stress_level - 3)
)
# normalize to 0-100
eff = (eff - eff.min()) / (eff.max() - eff.min()) * 100
# add noise
eff = np.clip(eff + np.random.normal(0, 5, N), 0, 100)

df = pd.DataFrame({
    "sleep_hours": sleep_hours,
    "sleep_quality": sleep_quality,
    "steps": steps,
    "sitting_hours": sitting_hours,
    "exercise_mins": exercise_mins,
    "calories": calories,
    "water_l": water_l,
    "screen_time": screen_time,
    "stress_level": stress_level,
    "age": age,
    "bmi": bmi,
    "resting_hr": resting_hr,
    "efficiency_score": eff
})
df.to_csv(OUT, index=False)
print("Saved synthetic dataset to", OUT)
