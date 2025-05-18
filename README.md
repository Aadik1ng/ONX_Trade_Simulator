# Trade Simulator (OKX)

A real-time trade simulation and analytics tool for OKX, featuring:
- Live L2 orderbook data via WebSocket
- Streamlit UI with input (left) and output (right) panels
- Models for slippage, fees, market impact (Almgren-Chriss), and maker/taker prediction
- Performance and latency metrics

## Features
- **Inputs:** Exchange, Spot Asset, Order Type, Quantity, Volatility, Fee Tier
- **Outputs:** Expected Slippage, Fees, Market Impact, Net Cost, Maker/Taker %, Latency
- **WebSocket:** Real-time L2 orderbook from OKX
- **Models:**
  - Almgren-Chriss market impact (quantitative execution model)
  - Linear/quantile regression for slippage
  - Logistic regression for maker/taker split
  - Rule-based fee calculation
- **Performance:** Latency metrics for data processing and UI

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your API keys in the `api_key` file (see example in repo).
3. **Train the models (run these from the project root directory):**
   ```bash
   python scripts/train_model.py
   python scripts/train_slippage_model.py
   ```
4. Run the app (from the project root directory):
   ```bash
   streamlit run src/trade_simulator/main.py
   ```

## File Structure

```
Trade_Simulator/
├── .cursorignore
├── .gitignore
├── api_key
├── docs/
│   └── Understanding the Almgren-Chriss Model .md
├── file_descriptions.txt
├── git_commit_script.py
├── git_individual_commit_script.py
├── Local_ignore/ # Ignored by git and cursor
├── models/
│   ├── maker_taker_classifier.pkl
│   └── slippage_regressor.pkl
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_model.py
│   └── train_slippage_model.py
├── src/
│   └── trade_simulator/
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       ├── utils.py
│       └── ws_client.py
├── trade_sim_results/ # Contains simulation output files
```

## Documentation
- See inline docstrings in each module for model and algorithm details.
- Almgren-Chriss model: see `Understanding the Almgren-Chriss Model .md` for theory and implementation notes.

### Model Details

This simulator employs several models to estimate various trade execution costs and characteristics:

-   **Market Impact (Almgren-Chriss):** The core market impact is estimated using the [Almgren-Chriss optimal execution model](Understanding the Almgren-Chriss%20Model%20.md). This model aims to find a trade schedule that balances market impact costs (temporary and permanent) against volatility risk. The implementation in `models.py` calculates the expected impact based on input parameters like quantity, volatility, and model coefficients (eta, gamma, risk aversion). Refer to the dedicated markdown file for a detailed theoretical background and the implementation specifics.

-   **Slippage (Linear Regression):** Expected slippage is predicted using a trained Linear Regression model. This model is trained offline (`train_slippage_model.py`) on mock data based on factors like bid-ask spread, order size, and volatility. The trained model (`slippage_regressor.pkl`) is loaded by `main.py` and used for real-time predictions. A fallback heuristic (half-spread * order size) is used if the model cannot be loaded.

-   **Maker/Taker Proportion (Logistic Regression):** The likelihood of an order being executed as a maker or taker is estimated using a pre-trained Logistic Regression model (`maker_taker_classifier.pkl`). This model considers features like spread, order size, and volatility to predict the percentage split between maker and taker fills.

-   **Fees (Rule-Based):** Trading fees are calculated based on a simple rule-based model using predefined fee tiers (`utils.py`) and the trade quantity.

### Performance Analysis and Optimization

The simulator measures several latency metrics in real-time to provide insight into performance:

-   **Processing Latency:** Time taken to process a single orderbook tick.
-   **UI Update Latency:** Time from data reception to the Streamlit UI updating.
-   **End-to-End Latency:** Total time from the server timestamp of the data to the UI update.

Future work includes:

-   **Performance Analysis Report:** A detailed report on observed performance characteristics.
-   **Benchmarking Results:** Documenting latency and processing speed under various load conditions.
-   **Optimization Documentation:** Explanations of implemented optimization techniques (e.g., threading for non-blocking WebSocket, data structure choices, etc.) and their impact.

## License
MIT
