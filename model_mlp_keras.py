# src/model_mlp_keras.py
"""
Train a simple MLP to predict efficiency_score from the CSV.
Usage:
  python src/model_mlp_keras.py --data data/synthetic_efficiency.csv --epochs 30 --batch_size 32
"""

import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # regression output
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main(data_path, epochs, batch_size):
    print("Loading data:", data_path)
    df = pd.read_csv(data_path)
    if 'efficiency_score' not in df.columns:
        raise ValueError("CSV must contain column 'efficiency_score'")
    X = df.drop(columns=['efficiency_score']).values
    y = df['efficiency_score'].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

    # Scale inputs
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Save scaler so we can use it later
    joblib.dump(scaler, "scaler.save")
    print("Saved scaler to scaler.save")

    # Build & train model
    model = build_model(X_train_s.shape[1])
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=1
    )

    # Save the trained model
    model.save("mlp_efficiency_model.keras", save_format="keras")
    print("Saved model to mlp_efficiency_model.h5")

    # Evaluate on test set
    y_pred = model.predict(X_test_s).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.3f}")
    print(f"Test MSE: {mse:.3f}")
    print(f"Test R2: {r2:.3f}")

    # Plot training/validation loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_training_loss.png")
    print("Saved training loss plot to mlp_training_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to CSV data file')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch_size)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
