import streamlit as st
import nest_asyncio
import asyncio
from trade_simulator.ws_client import OrderbookWebSocketClient
from trade_simulator.models import almgren_chriss_market_impact, calculate_fees
from trade_simulator.utils import FEE_TIERS
import numpy as np
import logging
import json
from datetime import datetime, timezone
import os
import joblib
import time

nest_asyncio.apply()

st.set_page_config(layout="wide", page_title="Trade Simulator - OKX")

# Setup logging for debug
logger = logging.getLogger("trade_simulator_debug")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Load trained Maker/Taker classifier ---
mt_classifier = None
try:
    mt_classifier = joblib.load("models/maker_taker_classifier.pkl")
    logger.info("Loaded trained Maker/Taker classifier from models/maker_taker_classifier.pkl")
except Exception as e:
    logger.warning(f"Could not load trained Maker/Taker classifier: {e}")

# --- Load trained Slippage Regressor ---
slippage_regressor = None
try:
    slippage_regressor = joblib.load("models/slippage_regressor.pkl")
    logger.info("Loaded trained Slippage Regressor from models/slippage_regressor.pkl")
except Exception as e:
    logger.warning(f"Could not load trained Slippage Regressor: {e}. Slippage will use heuristic.")

# --- Streamlit session state for orderbook and outputs ---
if "orderbook" not in st.session_state:
    st.session_state.orderbook = None
if "last_tick_latency" not in st.session_state:
    st.session_state.last_tick_latency = None
if "ws_running" not in st.session_state:
    st.session_state.ws_running = False
if "ws_stop" not in st.session_state:
    st.session_state.ws_stop = False

# --- Sidebar (left panel) for input parameters ---
with st.sidebar:
    st.title("Trade Simulator Inputs")
    exchange = st.selectbox("Exchange", ["OKX"], index=0)
    spot_asset = st.text_input("Spot Asset (e.g. BTC-USDT)", value="BTC-USDT")
    order_type = st.selectbox("Order Type", ["market"], index=0)
    quantity = st.number_input("Quantity (USD equivalent)", min_value=1.0, value=10000.0, step=1.0)  # Default to $10,000
    volatility = st.number_input("Volatility (annualized, %)", min_value=0.0, value=60.0, step=0.1)
    fee_tier = st.selectbox("Fee Tier", ["Tier 1", "Tier 2", "Tier 3"], index=0)
    st.markdown("---")
    ws_button = st.button("Start WebSocket" if not st.session_state.ws_running else "Stop WebSocket")
    st.caption("Parameters auto-update with real-time data.")

# --- WebSocket client logic ---
async def ws_tick_handler(data, latency):
    st.session_state.orderbook = data
    st.session_state.last_tick_latency = latency
    st.session_state.data_received_time = time.time()
    # Parse ISO8601 'timestamp' field to Unix timestamp for end-to-end latency
    if "timestamp" in data:
        try:
            dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            st.session_state.server_timestamp = dt.timestamp()
        except Exception as e:
            st.session_state.server_timestamp = None
    else:
        st.session_state.server_timestamp = None
    st.rerun()

async def run_ws():
    # Store the client in session_state
    client = OrderbookWebSocketClient()
    st.session_state.ws_client = client
    await client.start(ws_tick_handler)

# Start/stop WebSocket on button click
if ws_button:
    if not st.session_state.ws_running:
        st.session_state.ws_running = True
        st.session_state.ws_stop = False
        st.warning("WebSocket will block the UI until stopped. To stop, use the Stop WebSocket button.")
        asyncio.run(run_ws())
    else:
        st.session_state.ws_running = False
        st.session_state.ws_stop = True
        st.warning("WebSocket connection closing requested.")

# --- Main panel (right) for outputs ---
st.title("Trade Simulation Outputs")
col1, col2 = st.columns(2)

# --- Compute outputs if orderbook is available ---
orderbook = st.session_state.orderbook
slippage = "-"
fees = "-"
market_impact = "-"
net_cost = "-"
maker_pct = "-"
taker_pct = "-"
processing_latency = st.session_state.last_tick_latency or "-"
# UI Update Latency: time from data received to now (i.e., rerun)
if "data_received_time" in st.session_state:
    ui_update_latency = (time.time() - st.session_state.data_received_time) * 1000  # ms
else:
    ui_update_latency = "-"
# End-to-End Latency: time from server timestamp to now
if "server_timestamp" in st.session_state and st.session_state.server_timestamp:
    end_to_end_latency = (time.time() - st.session_state.server_timestamp) * 1000  # ms
else:
    end_to_end_latency = "-"

# Default Almgren-Chiss parameters for real-world resonance
TEMP_IMPACT_ETA = 0.1
PERM_IMPACT_GAMMA = 0.1
RISK_AVERSION = 0.01
TIME_STEPS = 51
ALPHA = 1
BETA = 1

if orderbook:
    # Example: use best bid/ask for price, and order size for impact
    try:
        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])
        best_ask = float(asks[0][0]) if asks else None
        best_bid = float(bids[0][0]) if bids else None
        mid_price = (best_ask + best_bid) / 2 if best_ask and best_bid else None
        # Ensure order_size is at least 1 base unit
        order_size = max(1, int(quantity / mid_price)) if mid_price else max(1, int(quantity))
        # Use user volatility as decimal
        vol = float(volatility) / 100.0
        # Debug logging
        logger.info(f"order_size: {order_size}, mid_price: {mid_price}, TEMP_IMPACT_ETA: {TEMP_IMPACT_ETA}, PERM_IMPACT_GAMMA: {PERM_IMPACT_GAMMA}, RISK_AVERSION: {RISK_AVERSION}, vol: {vol}")
        # Almgren-Chriss market impact
        market_impact = almgren_chriss_market_impact(order_size, vol, TEMP_IMPACT_ETA, PERM_IMPACT_GAMMA, RISK_AVERSION, TIME_STEPS, ALPHA, BETA)
        # Fees
        fees = calculate_fees(quantity, fee_tier, FEE_TIERS)
        # Calculate Slippage using the trained model or fallback heuristic
        spread = (best_ask - best_bid) if (best_ask and best_bid) else 0.0
        # Features for slippage model: spread, order_size, volatility (annualized as decimal)
        slippage_features = np.array([[spread, order_size, vol]])
        if slippage_regressor is not None:
            try:
                slippage = slippage_regressor.predict(slippage_features)[0]
            except Exception as e:
                logger.warning(f"Error using trained slippage regressor, falling back to heuristic: {e}")
                # Heuristic fallback (can be the previous one or a simpler one)
                if best_ask and best_bid:
                     slippage = (best_ask - best_bid) / 2 * order_size
                else:
                     slippage = 0.0 # Default to 0 if no valid spread
        else:
            # Heuristic fallback if model not loaded
            if best_ask and best_bid:
                slippage = (best_ask - best_bid) / 2 * order_size
            else:
                slippage = 0.0 # Default to 0 if no valid spread
        # Ensure slippage is not negative
        slippage = max(0.0, float(slippage))
        # Net cost
        net_cost = float(slippage) + float(fees) + float(market_impact)

        # --- Maker/Taker proportion using trained logistic regression or fallback heuristic ---
        features = np.array([[spread, order_size, vol]])
        if mt_classifier is not None:
            try:
                proba = mt_classifier.predict_proba(features)
                maker_pct = int(round(proba[0][0] * 100))
                taker_pct = int(round(proba[0][1] * 100))
            except Exception as e:
                logger.warning(f"Error using trained classifier, falling back to heuristic: {e}")
                if spread < 0.05 and order_size <= 2:
                    maker_pct = 80
                    taker_pct = 20
                elif spread < 0.1 and order_size <= 5:
                    maker_pct = 50
                    taker_pct = 50
                else:
                    maker_pct = 0
                    taker_pct = 100
        else:
            # Heuristic fallback
            if spread < 0.05 and order_size <= 2:
                maker_pct = 80
                taker_pct = 20
            elif spread < 0.1 and order_size <= 5:
                maker_pct = 50
                taker_pct = 50
            else:
                maker_pct = 0
                taker_pct = 100
        # Remove negative zero
        maker_pct = max(0, int(maker_pct))
        taker_pct = max(0, int(taker_pct))

        # --- Save results to JSON file ---
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {
                "exchange": exchange,
                "spot_asset": spot_asset,
                "order_type": order_type,
                "quantity": quantity,
                "volatility": volatility,
                "fee_tier": fee_tier,
                "order_size": order_size,
                "mid_price": mid_price,
                "TEMP_IMPACT_ETA": TEMP_IMPACT_ETA,
                "PERM_IMPACT_GAMMA": PERM_IMPACT_GAMMA,
                "RISK_AVERSION": RISK_AVERSION,
                "TIME_STEPS": TIME_STEPS,
                "ALPHA": ALPHA,
                "BETA": BETA
            },
            "outputs": {
                "slippage": float(slippage) if isinstance(slippage, (float, int)) else None,
                "fees": float(fees) if isinstance(fees, (float, int)) else None,
                "market_impact": float(market_impact) if isinstance(market_impact, (float, int)) else None,
                "net_cost": float(net_cost) if isinstance(net_cost, (float, int)) else None,
                "maker_pct": maker_pct,
                "taker_pct": taker_pct,
                "processing_latency_ms": float(processing_latency) if isinstance(processing_latency, (float, int)) else None,
                "slippage_model_used": slippage_regressor is not None
            }
        }
        outdir = "trade_sim_results"
        os.makedirs(outdir, exist_ok=True)
        fname = f"trade_sim_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(outdir, fname), "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        st.warning(f"Error in output calculation: {e}")

with col1:
    st.subheader("Expected Slippage")
    st.metric("Slippage (USD)", f"{slippage:.4f}" if isinstance(slippage, (float, int)) else slippage)
    st.subheader("Expected Fees")
    st.metric("Fees (USD)", f"{fees:.4f}" if isinstance(fees, (float, int)) else fees)
    st.subheader("Expected Market Impact")
    st.metric("Market Impact (USD)", f"{market_impact:.4f}" if isinstance(market_impact, (float, int)) else market_impact)
    st.subheader("Net Cost")
    st.metric("Net Cost (USD)", f"{net_cost:.4f}" if isinstance(net_cost, (float, int)) else net_cost)

with col2:
    st.subheader("Maker/Taker Proportion")
    st.metric("Maker %", maker_pct)
    st.metric("Taker %", taker_pct)
    st.subheader("Internal Latency")
    st.metric("Processing Latency (ms)", f"{processing_latency:.2f}" if isinstance(processing_latency, (float, int)) else processing_latency)
    st.metric("UI Update Latency (ms)", ui_update_latency)
    st.metric("End-to-End Latency (ms)", end_to_end_latency)

st.markdown("---")
st.caption("Live orderbook and model outputs will appear here as the system runs.") 