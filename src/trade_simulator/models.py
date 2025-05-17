import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

# --- Almgren-Chriss Market Impact Model ---
def temporary_impact(volume, alpha, eta):
    return eta * volume ** alpha

def permanent_impact(volume, beta, gamma):
    return gamma * volume ** beta

def hamiltonian(inventory, sell_amount, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step=0.5):
    temp_impact = risk_aversion * sell_amount * permanent_impact(sell_amount / time_step, beta, gamma)
    perm_impact = risk_aversion * (inventory - sell_amount) * time_step * temporary_impact(sell_amount / time_step, alpha, eta)
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step * ((inventory - sell_amount) ** 2)
    return temp_impact + perm_impact + exec_risk

def optimal_execution(time_steps, total_shares, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step_size=0.5):
    value_function = np.zeros((time_steps, total_shares + 1), dtype="float64")
    best_moves = np.zeros((time_steps, total_shares + 1), dtype="int")
    inventory_path = np.zeros((time_steps, 1), dtype="int")
    inventory_path[0] = total_shares
    optimal_trajectory = []

    # Terminal condition
    for shares in range(total_shares + 1):
        exp_arg = shares * temporary_impact(shares / time_step_size, alpha, eta)
        exp_arg = np.clip(exp_arg, -700, 700)
        value_function[time_steps - 1, shares] = np.exp(exp_arg)
        best_moves[time_steps - 1, shares] = shares

    # Backward induction
    for t in range(time_steps - 2, -1, -1):
        for shares in range(total_shares + 1):
            exp_arg = hamiltonian(shares, shares, risk_aversion, alpha, beta, gamma, eta, volatility, time_step_size)
            exp_arg = np.clip(exp_arg, -700, 700)
            best_value = value_function[t + 1, 0] * np.exp(exp_arg)
            best_share_amount = shares
            for n in range(shares):
                exp_arg2 = hamiltonian(shares, n, risk_aversion, alpha, beta, gamma, eta, volatility, time_step_size)
                exp_arg2 = np.clip(exp_arg2, -700, 700)
                current_value = value_function[t + 1, shares - n] * np.exp(exp_arg2)
                if current_value < best_value:
                    best_value = current_value
                    best_share_amount = n
            value_function[t, shares] = best_value
            best_moves[t, shares] = best_share_amount

    # Optimal trajectory
    for t in range(1, time_steps):
        inventory_path[t] = inventory_path[t - 1] - best_moves[t, inventory_path[t - 1]]
        optimal_trajectory.append(best_moves[t, inventory_path[t - 1]])

    optimal_trajectory = np.asarray(optimal_trajectory)
    return value_function, best_moves, inventory_path, optimal_trajectory

def almgren_chriss_market_impact(order_size, volatility, temp_impact_eta, perm_impact_gamma, risk_aversion, time_steps=51, alpha=1, beta=1):
    """
    Calculate expected market impact using the Almgren-Chriss model.
    Returns the total market impact in USD (approximate, based on optimal execution path).
    """
    # Use optimal_execution to get the execution path
    _, _, inventory_path, optimal_traj = optimal_execution(
        time_steps, int(order_size), risk_aversion, alpha, beta, perm_impact_gamma, temp_impact_eta, volatility=volatility, time_step_size=0.5
    )
    # Market impact: sum of temporary and permanent impacts along the path
    temp_impact = 0.0
    perm_impact = 0.0
    for t, shares in enumerate(optimal_traj):
        temp_impact += temporary_impact(shares, alpha, temp_impact_eta)
        perm_impact += permanent_impact(shares, beta, perm_impact_gamma)
    return float(temp_impact + perm_impact)

# --- Slippage Regression Model ---
class SlippageRegressor:
    """Linear/Quantile regression for slippage estimation."""
    def __init__(self):
        self.model = LinearRegression()
        # self.model = QuantileRegressor(quantile=0.5)  # For quantile regression

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# --- Maker/Taker Proportion Model ---
class MakerTakerClassifier:
    """Logistic regression for predicting maker/taker proportion."""
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --- Fee Calculation ---
def calculate_fees(quantity_usd, fee_tier, fee_tiers=None):
    """
    Calculate expected fees based on quantity and fee tier.
    Parameters:
        quantity_usd (float): Order size in USD
        fee_tier (str): Fee tier string
        fee_tiers (dict): Fee tier table (maker/taker rates)
    Returns:
        float: Expected fee (USD)
    """
    if fee_tiers is None:
        # Default fee tiers
        fee_tiers = {
            "Tier 1": {"maker": 0.0008, "taker": 0.0010},
            "Tier 2": {"maker": 0.0006, "taker": 0.0008},
            "Tier 3": {"maker": 0.0004, "taker": 0.0006},
        }
    # Assume taker for market order
    rate = fee_tiers.get(fee_tier, {"taker": 0.001})["taker"]
    return float(quantity_usd * rate) 