# model_script.py

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Dummy training data (features: [spread, order_size, volatility])
X_train = np.array([
    [0.01, 1, 0.5],  # maker
    [0.22, 18, 0.6], # taker
    [0.03, 3, 0.4],  # maker
    [0.08, 5, 0.7],  # 50/50
    [0.15, 12, 0.5], # taker
    [0.04, 2, 0.3],  # maker
    [0.30, 25, 0.6], # taker
    [0.10, 7, 0.8],  # 50/50
    [0.05, 6, 0.2],  # maker
    [0.28, 24, 0.9], # taker
    [0.18, 15, 0.4], # taker
    [0.09, 4, 0.7],  # 50/50
    [0.12, 9, 0.3],  # 50/50
    [0.35, 30, 0.5], # taker
    [0.07, 5, 0.4],  # maker
    [0.40, 35, 0.8], # taker
    [0.02, 1, 0.6],  # maker
    [0.25, 22, 0.7], # taker
    [0.06, 3, 0.2],  # maker
    [0.13, 10, 0.5], # 50/50
])

y_train = np.array([
    0,  # maker
    1,  # taker
    0,  # maker
    1,  # 50/50
    1,  # taker
    0,  # maker
    1,  # taker
    1,  # 50/50
    0,  # maker
    1,  # taker
    1,  # taker
    1,  # 50/50
    1,  # 50/50
    1,  # taker
    0,  # maker
    1,  # taker
    0,  # maker
    1,  # taker
    0,  # maker
    1   # 50/50
])

# Train the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Save the model
model_filename = "maker_taker_classifier.pkl"
output_path = os.path.join("..", "models", model_filename)
joblib.dump(logreg, output_path)
print(f"Model saved as {output_path}")

# Load the model
loaded_model = joblib.load(output_path)

# Test the loaded model with a sample input
sample_input = np.array([[0.1, 10, 0.5]])  # Example input
prediction = loaded_model.predict(sample_input)
print(f"Prediction for {sample_input}: {prediction}")
