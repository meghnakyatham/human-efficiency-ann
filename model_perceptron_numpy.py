import numpy as np

# Simple dataset: AND logic gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Initialize weights
weights = np.random.rand(2)
bias = np.random.rand(1)
lr = 0.1
epochs = 10

# Training loop
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = 1 if linear_output >= 0.5 else 0
        error = y[i] - y_pred
        # Update weights and bias
        weights += lr * error * X[i]
        bias += lr * error
        errors += abs(error)
    print(f"Epoch {epoch+1}, errors: {errors}")

print("Training complete!")
print("Weights:", weights)
print("Bias:", bias)
