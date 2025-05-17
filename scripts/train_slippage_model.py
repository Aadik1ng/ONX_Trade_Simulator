import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Assuming a simple linear relationship for demonstration
# In a real scenario, you would load actual trade data

def generate_mock_slippage_data(num_samples=1000):
    # Features: spread, order_size, volatility
    spreads = np.random.uniform(0.01, 0.5, num_samples)
    order_sizes = np.random.uniform(1, 1000, num_samples)
    volatilities = np.random.uniform(0.1, 1.0, num_samples) # as decimal

    # Target: slippage
    # Mock slippage calculation (simplified: proportional to spread, size, and volatility)
    slippage = (spreads * 0.5) + (order_sizes * 0.01) + (volatilities * 10) + np.random.normal(0, 5, num_samples)
    slippage = np.maximum(0, slippage) # Slippage should not be negative

    X = np.vstack((spreads, order_sizes, volatilities)).T
    y = slippage

    return X, y

def train_slippage_model(model_path="slippage_regressor.pkl"):
    print("Generating mock slippage data...")
    X, y = generate_mock_slippage_data(num_samples=5000)

    # Split data (optional for simple linear regression, good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Linear Regression model for slippage...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate (optional)
    score = model.score(X_test, y_test)
    print(f"Model R^2 score on test data: {score:.4f}")

    print(f"Saving trained model to {model_path}")
    # Ensure directory exists if needed
    output_path = os.path.join("..", "models", model_path) # Save to models directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    joblib.dump(model, output_path)
    print("Model training complete.")

if __name__ == "__main__":
    train_slippage_model() 